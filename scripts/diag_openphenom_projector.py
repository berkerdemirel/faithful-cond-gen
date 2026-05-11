"""
Diagnostic: Direct projector output quality for openphenom vs dinov2 REPA models.

1. Load both models
2. Take a small batch of real images
3. For each model: encode → add noise at t=0.01 → velocity_prediction(return_projected=True)
4. Also run teacher encoder on same images
5. Compare projector output vs teacher output (cosine sim)
6. Check if projector features for different images are discriminative

This tells us if the projector itself is working, independent of the caching pipeline.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate

from faithful_cond_gen.utils.checkpoints import get_checkpoint_path
from faithful_cond_gen.pl_modules.generator_pl import GeneratorPL
from faithful_cond_gen.model.generator import GeneratorWrapper
from faithful_cond_gen.data.rxrx1 import RxRx1DataModule, RxRx1DataConfig
from faithful_cond_gen.model.repa_encoder import load_repa_encoder

DEVICE = "cuda:0"
N = 16


def load_model(gen_dir):
    cfg = OmegaConf.load(gen_dir / "gen_config.yaml")
    gen_cfg = instantiate(cfg.model)
    pl = GeneratorPL.load_from_checkpoint(
        get_checkpoint_path(cfg.checkpoint_key),
        generator=GeneratorWrapper(gen_cfg),
        map_location=DEVICE, strict=False
    )
    if hasattr(pl, "ema"):
        pl.ema.apply()
    pl.to(DEVICE).eval()
    return pl, cfg


@torch.no_grad()
def main():
    base = Path("outputs/gen")

    # Load data
    dm = RxRx1DataModule(RxRx1DataConfig(
        data_dir="/mnt/pvc/AutoSync/data/rxrx1",
        img_size=[512, 512], resize=[256, 256],
        reduce_channels=False, augment_train=False, normalize=False,
        use_numpy=True, use_parquet=False,
        batch_size=N, num_workers=4, val_size=0.1,
        seed=1337, rare_threshold=20, held_out_pairs=None,
    ))
    batch = next(iter(dm.train_dataloader()))
    images = batch[0][:N].to(DEVICE)
    meta = batch[1] if len(batch) > 1 else {}
    print(f"Images: {images.shape}, range [{images.min():.2f}, {images.max():.2f}]")
    if "cell_type_id" in meta:
        print(f"Cell types: {meta['cell_type_id'][:N].tolist()}")
        print(f"Sirna IDs: {meta['sirna_id'][:N].tolist()}")

    models = {
        "repa_full (dinov2)": base / "rxrx1_repa_full",
        "repa_openphenom_full": base / "rxrx1_repa_openphenom_full",
    }

    for model_name, gen_dir in models.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        pl, cfg = load_model(gen_dir)
        encoder_name = pl.generator.cfg.repa_encoder
        print(f"  encoder: {encoder_name}")

        # Load teacher
        teacher = load_repa_encoder(
            encoder_name=encoder_name,
            resolution=256, in_channels=6, device=DEVICE,
        ).eval()
        print(f"  teacher embed_dim: {teacher.embed_dim}")

        # Teacher features on clean images
        t_out = teacher(images)
        # t_out: (N, num_patches, D)
        print(f"  teacher raw output: {t_out.shape}")
        t_pooled = t_out.mean(dim=1) if t_out.ndim == 3 else t_out
        t_norm = F.normalize(t_pooled, dim=-1)
        print(f"  teacher pooled: {t_pooled.shape}, L2 norm: {t_pooled.norm(dim=-1).mean():.4f}")

        # Encode to latent
        latents = pl.generator.encode(images)
        print(f"  latents: {latents.shape}")

        # Projector features at t=0.01 (matching real extraction)
        t_val = 0.01
        noisy = (1 - t_val) ** 0.5 * latents + t_val ** 0.5 * torch.randn_like(latents)
        t_tensor = torch.full((N,), t_val, device=DEVICE)
        cond_ids = torch.zeros(N, 2, device=DEVICE, dtype=torch.long)

        v, zs = pl.generator.velocity_prediction(noisy, t_tensor, cond_ids, return_projected=True)
        proj_out = zs[0]  # (N, P, D) or (N, D)
        print(f"  projector raw output: {proj_out.shape}")
        p_pooled = proj_out.mean(dim=1) if proj_out.ndim == 3 else proj_out
        p_norm = F.normalize(p_pooled, dim=-1)
        print(f"  projector pooled: {p_pooled.shape}, L2 norm: {p_pooled.norm(dim=-1).mean():.4f}")

        # Also get projector features at t=0.04 (the cutoff used in generation)
        t_val2 = 0.04
        noisy2 = (1 - t_val2) ** 0.5 * latents + t_val2 ** 0.5 * torch.randn_like(latents)
        t_tensor2 = torch.full((N,), t_val2, device=DEVICE)
        v2, zs2 = pl.generator.velocity_prediction(noisy2, t_tensor2, cond_ids, return_projected=True)
        p2_pooled = zs2[0].mean(dim=1) if zs2[0].ndim == 3 else zs2[0]
        p2_norm = F.normalize(p2_pooled, dim=-1)

        # 1. Teacher vs Projector alignment (per-sample cosine sim)
        sim_tp = (t_norm * p_norm).sum(dim=-1)
        print(f"\n  Teacher vs Projector(t=0.01) cosine sim: mean={sim_tp.mean():.4f} std={sim_tp.std():.4f}")
        sim_tp2 = (t_norm * p2_norm).sum(dim=-1)
        print(f"  Teacher vs Projector(t=0.04) cosine sim: mean={sim_tp2.mean():.4f} std={sim_tp2.std():.4f}")

        # 2. Projector(0.01) vs Projector(0.04) consistency
        sim_pp = (p_norm * p2_norm).sum(dim=-1)
        print(f"  Projector(t=0.01) vs Projector(t=0.04): mean={sim_pp.mean():.4f}")

        # 3. Pairwise discrimination: are different images distinguishable?
        t_sim_matrix = t_norm @ t_norm.T
        p_sim_matrix = p_norm @ p_norm.T
        mask = ~torch.eye(N, dtype=torch.bool, device=DEVICE)
        print(f"\n  Teacher pairwise (off-diag): mean={t_sim_matrix[mask].mean():.4f} std={t_sim_matrix[mask].std():.4f}")
        print(f"  Projector pairwise (off-diag): mean={p_sim_matrix[mask].mean():.4f} std={p_sim_matrix[mask].std():.4f}")

        # 4. Check individual patch-level alignment (before pooling)
        if proj_out.ndim == 3 and t_out.ndim == 3:
            # Per-patch cosine sim
            t_patches_norm = F.normalize(t_out, dim=-1)
            p_patches_norm = F.normalize(proj_out, dim=-1)
            patch_sim = (t_patches_norm * p_patches_norm).sum(dim=-1)  # (N, P)
            print(f"\n  Patch-level teacher vs projector:")
            print(f"    mean={patch_sim.mean():.4f} std={patch_sim.std():.4f}")
            print(f"    per-sample mean: {patch_sim.mean(dim=1).tolist()}")

        # 5. Feature effective rank
        feats = p_pooled.float()
        feats_centered = feats - feats.mean(dim=0)
        s = torch.linalg.svdvals(feats_centered)
        s_frac = s / s.sum()
        ent = -(s_frac * torch.log(s_frac + 1e-10)).sum()
        print(f"\n  Projector effective rank (this batch): {torch.exp(ent).item():.1f}")

        t_feats = t_pooled.float()
        t_centered = t_feats - t_feats.mean(dim=0)
        s_t = torch.linalg.svdvals(t_centered)
        s_t_frac = s_t / s_t.sum()
        ent_t = -(s_t_frac * torch.log(s_t_frac + 1e-10)).sum()
        print(f"  Teacher effective rank (this batch): {torch.exp(ent_t).item():.1f}")

        # 6. Check norm distribution of raw features (before normalization)
        print(f"\n  Teacher raw norms: mean={t_pooled.norm(dim=-1).mean():.2f} std={t_pooled.norm(dim=-1).std():.2f}")
        print(f"  Projector raw norms: mean={p_pooled.norm(dim=-1).mean():.2f} std={p_pooled.norm(dim=-1).std():.2f}")

        del pl, teacher
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
