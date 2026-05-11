"""
Diagnostic: REPA alignment quality for OpenPhenom teacher.

Check A: real images
  teacher(clean_image) vs projector(real_latent + t=0.01 noise)

Check B: generated images
  teacher(gen_image) vs saved aligned_features from generation

Both checks on a small subset (a few conditions, ~32 samples each).
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

CHECKPOINT_KEY = "rxrx1_repa_openphenom_full_v1"
GEN_DIR = Path("outputs/gen/rxrx1_repa_openphenom_full")
DEVICE = "cuda:0"
N_SAMPLES = 32  # per check


@torch.no_grad()
def main():
    # --- Load model ---
    print(f"Loading {CHECKPOINT_KEY}...")
    cfg = OmegaConf.load(GEN_DIR / "gen_config.yaml")
    gen_cfg = instantiate(cfg.model)
    pl = GeneratorPL.load_from_checkpoint(
        get_checkpoint_path(CHECKPOINT_KEY),
        generator=GeneratorWrapper(gen_cfg),
        map_location=DEVICE, strict=False
    )
    if hasattr(pl, "ema"):
        pl.ema.apply()
    pl.to(DEVICE).eval()
    print(f"  use_repa={pl.generator.cfg.use_repa}, encoder={pl.generator.cfg.repa_encoder}")
    # Load teacher encoder manually (not auto-loaded outside training)
    teacher = load_repa_encoder(
        encoder_name=pl.generator.cfg.repa_encoder,
        resolution=256, in_channels=6, device=DEVICE,
    ).eval()
    print(f"  teacher loaded: {pl.generator.cfg.repa_encoder}, embed_dim={teacher.embed_dim}")

    # ----------------------------------------------------------------
    # CHECK A: real images
    # ----------------------------------------------------------------
    print("\n=== CHECK A: real images — teacher(clean) vs projector(noisy, t=0.01) ===")
    dm = RxRx1DataModule(RxRx1DataConfig(
        data_dir="/mnt/pvc/AutoSync/data/rxrx1",
        img_size=[512, 512], resize=[256, 256],
        reduce_channels=False, augment_train=False, normalize=False,
        use_numpy=True, use_parquet=False,
        batch_size=N_SAMPLES, num_workers=4, val_size=0.1,
        seed=1337, rare_threshold=20, held_out_pairs=None,
    ))
    batch = next(iter(dm.train_dataloader()))
    images = batch[0][:N_SAMPLES].to(DEVICE)  # (N, 6, 256, 256)
    print(f"  images shape: {images.shape}, range [{images.min():.2f}, {images.max():.2f}]")

    # Teacher on clean images
    teacher_out = teacher(images)  # (N, P, D) or (N, D)
    t_feats = teacher_out.mean(dim=1) if teacher_out.ndim == 3 else teacher_out
    t_feats = F.normalize(t_feats, dim=-1)
    print(f"  teacher features: {t_feats.shape}")

    # Encode to latent (handles RxRx1 6-ch folding internally)
    latents = pl.generator.encode(images)
    t_val = 0.01
    noisy = (1 - t_val) ** 0.5 * latents + t_val ** 0.5 * torch.randn_like(latents)
    t_tensor = torch.full((N_SAMPLES,), t_val, device=DEVICE)
    cond_ids = torch.zeros(N_SAMPLES, 2, device=DEVICE, dtype=torch.long)

    v, zs = pl.generator.velocity_prediction(noisy, t_tensor, cond_ids, return_projected=True)
    proj_out = zs[0]  # (N, P, D) or (N, D)
    p_feats = proj_out.mean(dim=1) if proj_out.ndim == 3 else proj_out
    p_feats = F.normalize(p_feats, dim=-1)
    print(f"  projector features: {p_feats.shape}")

    sim_A = (t_feats * p_feats).sum(dim=-1)
    rand_sim = (t_feats * F.normalize(torch.randn_like(t_feats), dim=-1)).sum(dim=-1)
    print(f"  cosine sim teacher vs projector: mean={sim_A.mean():.4f} std={sim_A.std():.4f}")
    print(f"  baseline (random):               mean={rand_sim.mean():.4f}")

    # ----------------------------------------------------------------
    # CHECK B: generated images
    # ----------------------------------------------------------------
    print("\n=== CHECK B: generated images — teacher(gen) vs saved aligned features ===")
    data = torch.load(GEN_DIR / "aligned_mean_features.pt", map_location="cpu", weights_only=False)
    saved_feats = data["features"]
    filenames = data["filenames"]

    # Pick N_SAMPLES generated images
    idx = list(range(N_SAMPLES))
    saved = F.normalize(saved_feats[idx].to(DEVICE), dim=-1)

    gen_images = torch.stack([
        torch.load(GEN_DIR / "images" / filenames[i], map_location="cpu", weights_only=False)
        for i in idx
    ]).to(DEVICE)
    if gen_images.max() > 1.5:
        gen_images = gen_images / 255.0
    print(f"  gen images: {gen_images.shape}, range [{gen_images.min():.2f}, {gen_images.max():.2f}]")

    teacher_out = teacher(gen_images)
    t_gen = teacher_out.mean(dim=1) if teacher_out.ndim == 3 else teacher_out
    t_gen = F.normalize(t_gen, dim=-1)

    sim_B = (t_gen * saved).sum(dim=-1)
    print(f"  cosine sim teacher vs saved:     mean={sim_B.mean():.4f} std={sim_B.std():.4f}")

    # ----------------------------------------------------------------
    print("\n=== SUMMARY ===")
    print(f"  A) real   teacher vs projector(t=0.01): {sim_A.mean():.4f} ± {sim_A.std():.4f}")
    print(f"  B) gen    teacher vs saved feats:       {sim_B.mean():.4f} ± {sim_B.std():.4f}")
    print(f"  Baseline (random):                      {rand_sim.mean():.4f}")


if __name__ == "__main__":
    main()
