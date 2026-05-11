"""
Diagnostic: Verify real feature extraction for openphenom.

1. Load cached real features
2. Load the model and run the same extraction on a small batch
3. Compare: do the features match?
4. Also check: does the teacher on these same images give correlated features?

This verifies the real feature caching pipeline is correct.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate

from faithful_cond_gen.utils.checkpoints import get_checkpoint_path
from faithful_cond_gen.pl_modules.generator_pl import GeneratorPL
from faithful_cond_gen.model.generator import GeneratorWrapper
from faithful_cond_gen.data.rxrx1 import RxRx1DataModule, RxRx1DataConfig
from faithful_cond_gen.model.repa_encoder import load_repa_encoder

DEVICE = "cuda:0"
N = 32


@torch.no_grad()
def main():
    base = Path("/mnt/pvc/faithful-cond-gen/outputs")

    for model_label, gen_dir_name, ckpt_key, real_feat_path in [
        ("repa_full (dinov2)", "rxrx1_repa_full", "rxrx1_repa_full_v1",
         "real_rxrx1_aligned/rxrx1_repa_full_v1/train_features.pt"),
        ("repa_openphenom_full", "rxrx1_repa_openphenom_full", "rxrx1_repa_openphenom_full_v1",
         "real_rxrx1_aligned/rxrx1_repa_openphenom_full_v1/train_features.pt"),
    ]:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_label}")
        print(f"{'='*70}")

        # Load model
        gen_dir = base / "gen" / gen_dir_name
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

        # Load teacher
        encoder_name = pl.generator.cfg.repa_encoder
        teacher = load_repa_encoder(
            encoder_name=encoder_name,
            resolution=256, in_channels=6, device=DEVICE,
        ).eval()

        # Load cached real features
        cached = torch.load(base / real_feat_path, map_location="cpu", weights_only=False)
        cached_feats = cached["features"]
        cached_meta = cached.get("metadata", {})
        print(f"Cached real features: {cached_feats.shape}")
        print(f"Cached metadata keys: {list(cached_meta.keys())}")
        print(f"Cached timestep: {cached.get('timestep', 'N/A')}")

        # Load data (same config as extraction script)
        dm = RxRx1DataModule(RxRx1DataConfig(
            data_dir="/mnt/pvc/AutoSync/data/rxrx1",
            img_size=[512, 512], resize=[256, 256],
            reduce_channels=False, augment_train=False, normalize=False,
            use_numpy=True, use_parquet=False,
            batch_size=N, num_workers=4, val_size=0.1,
            seed=1337, rare_threshold=20, held_out_pairs=None,
        ))
        # Get first batch from train
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        images = batch[0][:N].to(DEVICE)
        meta = batch[1] if len(batch) > 1 else {}
        print(f"Images: {images.shape}, range [{images.min():.2f}, {images.max():.2f}]")

        # Run the same extraction as extract_aligned_features_real.py
        # 1. Ensure [0,1] range (extraction script does this)
        if images.min() < 0:
            images = (images + 1) / 2
        images = images.contiguous()

        # 2. Encode to latent
        latents = pl.generator.encode(images)

        # 3. Add noise at t=0.01
        t_val = 0.01
        alpha_bar = 1 - t_val
        noise = torch.randn_like(latents)
        noisy = np.sqrt(alpha_bar) * latents + np.sqrt(1 - alpha_bar) * noise
        t_tensor = torch.full((N,), t_val, device=DEVICE, dtype=torch.float32)

        # 4. Get condition IDs
        cond_dict = meta.get("cond", meta) if isinstance(meta, dict) else meta
        cond_ids = []
        for k in ["cell_type_id", "sirna_id"]:
            if k in cond_dict:
                cond_ids.append(cond_dict[k].to(DEVICE))
            elif isinstance(meta, dict) and k in meta:
                cond_ids.append(meta[k].to(DEVICE))
        if cond_ids:
            cond_ids = torch.stack(cond_ids, dim=1)[:N]
        else:
            print(f"  WARNING: no condition keys found. meta type={type(meta)}, keys={meta.keys() if isinstance(meta, dict) else 'N/A'}")
            cond_ids = torch.zeros(N, 2, device=DEVICE, dtype=torch.long)

        # 5. Forward pass
        v, zs = pl.generator.velocity_prediction(noisy, t_tensor, cond_ids, return_projected=True)
        proj_out = zs[0]
        if proj_out.dim() == 3:
            fresh_feats = proj_out.mean(dim=1)
        else:
            fresh_feats = proj_out

        print(f"Freshly extracted features: {fresh_feats.shape}")

        # Compare with cached (first N samples should match if dataloader is deterministic)
        cached_first_n = cached_feats[:N].to(DEVICE)

        # L2 normalize both
        fresh_norm = F.normalize(fresh_feats.float(), dim=-1)
        cached_norm = F.normalize(cached_first_n.float(), dim=-1)

        # Per-sample cosine sim
        sim = (fresh_norm * cached_norm).sum(dim=-1)
        print(f"\nFresh vs Cached (first {N}) cosine sim:")
        print(f"  mean={sim.mean():.4f} std={sim.std():.4f}")
        print(f"  min={sim.min():.4f} max={sim.max():.4f}")

        # Raw value comparison
        diff = (fresh_feats.cpu() - cached_first_n.cpu()).abs()
        print(f"  Raw diff: mean={diff.mean():.4f} max={diff.max():.4f}")

        # Check if it's just noise differences (different random seeds for noise)
        # Try with no noise (t=0)
        t_zero = torch.full((N,), 0.001, device=DEVICE, dtype=torch.float32)
        v0, zs0 = pl.generator.velocity_prediction(latents, t_zero, cond_ids, return_projected=True)
        proj0 = zs0[0]
        if proj0.dim() == 3:
            feats_noiseless = proj0.mean(dim=1)
        else:
            feats_noiseless = proj0
        noiseless_norm = F.normalize(feats_noiseless.float(), dim=-1)

        sim_noiseless = (noiseless_norm * cached_norm).sum(dim=-1)
        print(f"\nNoiseless(t=0.001) vs Cached: {sim_noiseless.mean():.4f}")

        # Teacher on same images
        teacher_out = teacher(images)
        if teacher_out.ndim == 3:
            teacher_pooled = teacher_out.mean(dim=1)
        else:
            teacher_pooled = teacher_out
        teacher_norm = F.normalize(teacher_pooled.float(), dim=-1)

        sim_teacher_fresh = (teacher_norm * fresh_norm).sum(dim=-1)
        sim_teacher_cached = (teacher_norm * cached_norm).sum(dim=-1)
        print(f"\nTeacher vs Fresh projector: {sim_teacher_fresh.mean():.4f}")
        print(f"Teacher vs Cached features: {sim_teacher_cached.mean():.4f}")

        # Are cached features = teacher features (not projector)?
        # Check L2 norms
        print(f"\nNorm comparison:")
        print(f"  Teacher raw norm: {teacher_pooled.norm(dim=-1).mean():.2f}")
        print(f"  Fresh projector norm: {fresh_feats.norm(dim=-1).mean():.2f}")
        print(f"  Cached features norm: {cached_first_n.norm(dim=-1).mean():.2f}")

        # Effective dimensionality check
        for name, feats in [("Teacher", teacher_pooled), ("Fresh proj", fresh_feats), ("Cached", cached_first_n)]:
            f = feats.float()
            f = f - f.mean(dim=0)
            s = torch.linalg.svdvals(f)
            sf = s / s.sum()
            ent = -(sf * torch.log(sf + 1e-10)).sum()
            print(f"  {name} eff rank: {torch.exp(ent).item():.1f}")

        del pl, teacher
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
