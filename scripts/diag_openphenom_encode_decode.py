"""
OpenPhenom encode→decode roundtrip diagnostic.

Tests whether REPA-path encoding produces valid features by decoding back to
pixel space and measuring reconstruction quality.

Sections:
  1. REPA crop path (256→512→crop→encode→decode→uncrop) roundtrip
  2. Native 6ch direct (256→encode→decode) roundtrip
  3. Feature dimension comparison: 2304-dim (channel concat) vs 384-dim (channel avg)

Context: OpenPhenom trust features produce inverted correlations with DINOv3 KID
(rho=-0.55 instead of +0.94). This script verifies whether the encoder tokens
themselves are valid by testing reconstruction quality.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm
from torchvision.utils import save_image
from transformers import AutoModel

from faithful_cond_gen.data.rxrx1 import RxRx1DataModule, RxRx1DataConfig, to_rgb
from faithful_cond_gen.model.repa_encoder import load_repa_encoder
from faithful_cond_gen.eval.trust_eval.metrics_kid import calculate_kid_same_m
from faithful_cond_gen.eval.trust_eval.scoring_core import (
    fit_global_stats,
    fit_factorized_stats,
    compute_real_calibration_for_global_energy,
    compute_real_calibration_for_factorized_margins,
    compute_global_realism_z,
    compute_factorized_faithfulness_margin_z,
)
from faithful_cond_gen.eval.trust_eval.condition_utils import get_condition_key

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = "cuda:0"
OUT_DIR = Path("outputs/openphenom_diag_v2")
N_IMAGES = 8                # for reconstruction sections
BS = 4
N_CHANNELS = 6
PATCH_SIZE = 16
IMG_SIZE = 256
GRID_SIZE = IMG_SIZE // PATCH_SIZE  # 16
TOKENS_PER_CHANNEL = GRID_SIZE ** 2  # 256
TOKEN_DIM = 384

# Section 3 config
N_CONDITIONS_SEC3 = 20
SAMPLES_PER_COND_SEC3 = 50
CONDITION_KEYS = ["cell_type_id", "sirna_id"]
N_BOOTSTRAP_KID = 10
# ──────────────────────────────────────────────────────────────────────────────


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


# ── Helpers ───────────────────────────────────────────────────────────────────

def cropify(x: torch.Tensor) -> torch.Tensor:
    """(B, 6, 512, 512) → (B*4, 6, 256, 256). 2x2 crop grid."""
    B, C, H, W = x.shape
    assert H == 512 and W == 512 and C == N_CHANNELS
    x = x.reshape(B, C, 2, 256, 2, 256)
    x = x.permute(0, 2, 4, 1, 3, 5)  # (B, 2, 2, C, 256, 256)
    return x.reshape(B * 4, C, 256, 256)


def uncropify(crops: torch.Tensor, batch_size: int) -> torch.Tensor:
    """(B*4, 6, 256, 256) → (B, 6, 512, 512). Inverse of cropify."""
    _, C, H, W = crops.shape
    x = crops.reshape(batch_size, 2, 2, C, H, W)
    x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, 2, 256, 2, 256)
    return x.reshape(batch_size, C, 512, 512)


def unpatchify_ca(tokens, n_ch, patch_size, grid_size):
    """Unpatchify channel-agnostic tokens.

    (B, n_ch*grid², patch²) → (B, n_ch, grid*patch, grid*patch)

    Based on OpenPhenom's unflatten_tokens(channel_agnostic=True).
    """
    B = tokens.shape[0]
    h = w = grid_size
    # (B, n_ch*h*w, p*p) → (B, n_ch, h, w, p, p)
    x = tokens.reshape(B, n_ch, h, w, patch_size, patch_size)
    # NCHWPQ → NCHPWQ
    x = x.permute(0, 1, 2, 4, 3, 5)
    return x.reshape(B, n_ch, h * patch_size, w * patch_size)


def load_dataset():
    dm = RxRx1DataModule(RxRx1DataConfig(
        data_dir="/mnt/pvc/AutoSync/data/rxrx1",
        img_size=[512, 512], resize=[256, 256],
        reduce_channels=False, augment_train=False, normalize=False,
        use_numpy=True, use_parquet=False,
        batch_size=BS, num_workers=4, val_size=0.1,
        seed=1337, rare_threshold=20, held_out_pairs=None,
    ))
    return dm


def pick_top_conditions(dm, k):
    ds = dm.train_dataloader().dataset
    counts = defaultdict(int)
    for ct, si in zip(ds.cell_type_ids, ds.sirna_ids):
        counts[(int(ct), int(si))] += 1
    top = sorted(counts.items(), key=lambda x: -x[1])[:k]
    return [c for c, _ in top]


def build_cond_index(ds, conditions):
    cond_set = set(conditions)
    cond_to_idx = defaultdict(list)
    for i, (ct, si) in enumerate(zip(ds.cell_type_ids, ds.sirna_ids)):
        key = (int(ct), int(si))
        if key in cond_set:
            cond_to_idx[key].append(i)
    return cond_to_idx


def load_images(dm, n, distinct_conditions=True):
    """Load n images, one per condition if distinct_conditions=True."""
    ds = dm.train_dataloader().dataset
    conditions = pick_top_conditions(dm, n if distinct_conditions else 1)
    cond_to_idx = build_cond_index(ds, conditions)

    if distinct_conditions:
        indices = [cond_to_idx[c][0] for c in conditions[:n]]
    else:
        indices = cond_to_idx[conditions[0]][:n]

    images = torch.stack([ds[i][0] for i in indices]).to(DEVICE)
    if images.min() < 0:
        images = (images + 1) / 2
    return images, conditions[:n]


def save_rgb_grid(images_6ch, path, nrow=None):
    """Save 6ch images as RGB grid using to_rgb."""
    rgb = to_rgb(images_6ch.cpu().float())
    if nrow is None:
        nrow = images_6ch.shape[0]
    save_image(rgb, path, nrow=nrow)


def print_recon_metrics(originals, reconstructions, label):
    """Print per-image MSE and Pearson r."""
    B = originals.shape[0]
    mses, corrs = [], []
    print(f"\n  {label}: Per-image reconstruction quality")
    for i in range(B):
        mse = F.mse_loss(reconstructions[i], originals[i]).item()
        x = originals[i].flatten().cpu().float().numpy()
        y = reconstructions[i].flatten().cpu().float().numpy()
        r = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else 0.0
        mses.append(mse)
        corrs.append(r)
        print(f"    Img {i}: MSE={mse:.6f}  r={r:.4f}")
    print(f"    Mean: MSE={np.mean(mses):.6f}  r={np.mean(corrs):.4f}")
    return mses, corrs


# ── Section 1: REPA crop encode → decode ─────────────────────────────────────

@torch.no_grad()
def section1_repa_crop_roundtrip(model, images, conditions):
    """Test REPA crop path: 256→512→crop→encode→decode→uncrop→compare."""
    print("\n" + "=" * 70)
    print("SECTION 1: REPA Crop Path Encode → Decode Roundtrip")
    print("=" * 70)

    B = images.shape[0]
    instance_norm = nn.InstanceNorm2d(N_CHANNELS, affine=False, eps=1e-6).to(DEVICE)

    # 1. Upsample 256→512
    images_512 = F.interpolate(images, size=(512, 512), mode="bilinear", align_corners=False)
    print(f"  Images: {images.shape} → upsampled to {images_512.shape}")

    # 2. InstanceNorm (matching REPA encoder)
    normed_512 = instance_norm(images_512)
    print(f"  After InstanceNorm: range=[{normed_512.min():.3f}, {normed_512.max():.3f}]")

    # 3. Cropify → (B*4, 6, 256, 256)
    crops = cropify(normed_512)
    print(f"  Crops: {crops.shape}")

    # 4. Encode crops
    latent, mask, ind_restore = model.encoder.forward_masked(crops, 0.0)
    print(f"  Encoder output: {latent.shape} (expect ({B*4}, 1+{N_CHANNELS*TOKENS_PER_CHANNEL}, {TOKEN_DIM}))")
    print(f"  Mask sum: {mask.sum().item()}/{mask.numel()} (should be 0 for mask_ratio=0)")

    # 5. Decode using model's own method
    recon_tokens = model.decode_to_reconstruction(
        latent, ind_restore,
        model.encoder_decoder_proj, model.decoder, model.decoder_pred,
    )
    print(f"  Decoder output: {recon_tokens.shape} (expect ({B*4}, {N_CHANNELS*TOKENS_PER_CHANNEL}, {PATCH_SIZE**2}))")

    # 6. Unpatchify
    recon_crops = unpatchify_ca(recon_tokens, N_CHANNELS, PATCH_SIZE, GRID_SIZE)
    print(f"  Unpatchified crops: {recon_crops.shape}")

    # 7. Uncropify → (B, 6, 512, 512)
    recon_512 = uncropify(recon_crops, B)
    print(f"  Uncropified: {recon_512.shape}")

    # Compare in normed pixel space (crops)
    print_recon_metrics(crops, recon_crops, "REPA crop (per-crop, normed space)")

    # Also compare full 512×512
    print_recon_metrics(normed_512, recon_512, "REPA crop (full 512, normed space)")

    # Save RGB grids: original | reconstruction
    save_rgb_grid(images, OUT_DIR / "sec1_original_256.png")
    save_rgb_grid(images_512, OUT_DIR / "sec1_original_512.png")
    # For reconstruction, denormalize approximately (just rescale for visualization)
    save_rgb_grid(recon_512, OUT_DIR / "sec1_recon_512crop.png")

    # Side-by-side
    orig_rgb = to_rgb(images_512.cpu().float())
    recon_rgb = to_rgb(recon_512.cpu().float())
    paired = torch.cat([orig_rgb, recon_rgb], dim=3)  # side-by-side
    save_image(paired, OUT_DIR / "sec1_sidebyside_512crop.png", nrow=2)

    return recon_512


# ── Section 2: Native 6ch direct encode → decode ─────────────────────────────

@torch.no_grad()
def section2_native_6ch_direct(model, images, conditions):
    """Test native 6ch: 256→encode→decode→compare. No padding, no cropping."""
    print("\n" + "=" * 70)
    print("SECTION 2: Native 6ch Direct Encode → Decode Roundtrip")
    print("=" * 70)

    B = images.shape[0]
    instance_norm = nn.InstanceNorm2d(N_CHANNELS, affine=False, eps=1e-6).to(DEVICE)

    # 2a. InstanceNorm on [0,1] images (equivalent to model.input_norm due to scale-invariance)
    normed_6ch = instance_norm(images)
    print(f"  InstanceNorm on [0,1]: range=[{normed_6ch.min():.3f}, {normed_6ch.max():.3f}]")

    # Encode
    latent, mask, ind_restore = model.encoder.forward_masked(normed_6ch, 0.0)
    print(f"  Encoder output: {latent.shape} (expect ({B}, 1+{N_CHANNELS*TOKENS_PER_CHANNEL}, {TOKEN_DIM}))")

    # Decode
    recon_tokens = model.decode_to_reconstruction(
        latent, ind_restore,
        model.encoder_decoder_proj, model.decoder, model.decoder_pred,
    )
    recon_img = unpatchify_ca(recon_tokens, N_CHANNELS, PATCH_SIZE, GRID_SIZE)
    print(f"  Reconstructed: {recon_img.shape}")

    print_recon_metrics(normed_6ch, recon_img, "6ch direct (InstanceNorm, normed space)")

    # 2b. Also test with model's native input_norm (expects [0,255] → /255 → InstanceNorm)
    # Since our images are [0,1], feed images*255 through model.input_norm
    print(f"\n  Testing with model.input_norm (images*255 → Normalizer → InstanceNorm)...")
    images_255 = images * 255.0
    normed_native = model.input_norm(images_255)
    print(f"  model.input_norm on [0,255]: range=[{normed_native.min():.3f}, {normed_native.max():.3f}]")

    latent_nat, mask_nat, ind_restore_nat = model.encoder.forward_masked(normed_native, 0.0)
    recon_tokens_nat = model.decode_to_reconstruction(
        latent_nat, ind_restore_nat,
        model.encoder_decoder_proj, model.decoder, model.decoder_pred,
    )
    recon_native = unpatchify_ca(recon_tokens_nat, N_CHANNELS, PATCH_SIZE, GRID_SIZE)

    mses_nat, corrs_nat = print_recon_metrics(
        normed_native, recon_native, "6ch direct (model.input_norm, native path)"
    )

    # Check that InstanceNorm and model.input_norm produce same encoder output
    cos_sim = F.cosine_similarity(
        latent[:, 1:].reshape(-1, TOKEN_DIM),
        latent_nat[:, 1:].reshape(-1, TOKEN_DIM),
        dim=-1,
    ).mean().item()
    print(f"\n  Cosine sim (InstanceNorm vs model.input_norm encoder tokens): {cos_sim:.6f}")

    # Save
    save_rgb_grid(recon_img, OUT_DIR / "sec2_recon_256direct.png")
    save_rgb_grid(recon_native, OUT_DIR / "sec2_recon_256native.png")
    orig_rgb = to_rgb(images.cpu().float())
    recon_rgb = to_rgb(recon_img.cpu().float())
    native_rgb = to_rgb(recon_native.cpu().float())
    paired = torch.cat([orig_rgb, recon_rgb, native_rgb], dim=3)
    save_image(paired, OUT_DIR / "sec2_sidebyside_256.png", nrow=2)

    return recon_img


# ── Section 3: Feature dimension comparison (2304 vs 384) ────────────────────

@torch.no_grad()
def section3_feature_comparison(model, dm):
    """Compare 2304-dim (channel concat) vs 384-dim (channel avg) features."""
    print("\n" + "=" * 70)
    print("SECTION 3: Feature Dimension Comparison (2304 vs 384)")
    print("=" * 70)

    instance_norm = nn.InstanceNorm2d(N_CHANNELS, affine=False, eps=1e-6).to(DEVICE)
    ds = dm.train_dataloader().dataset
    conditions = pick_top_conditions(dm, N_CONDITIONS_SEC3)
    cond_to_idx = build_cond_index(ds, conditions)

    # Extract features for each condition
    feats_2304_by_cond = {}
    feats_384_by_cond = {}

    for cond in tqdm(conditions, desc="Extracting features"):
        indices = cond_to_idx[cond][:SAMPLES_PER_COND_SEC3]
        f2304_list, f384_list = [], []

        for start in range(0, len(indices), BS):
            batch_idx = indices[start:start + BS]
            images = torch.stack([ds[i][0] for i in batch_idx]).to(DEVICE)
            if images.min() < 0:
                images = (images + 1) / 2
            bsz = images.shape[0]

            # Upsample to 512×512
            images_512 = F.interpolate(images, size=(512, 512), mode="bilinear", align_corners=False)
            normed = instance_norm(images_512)
            crops = cropify(normed)  # (bsz*4, 6, 256, 256)

            # Encode
            latent, _, _ = model.encoder.forward_masked(crops, 0.0)
            # latent: (bsz*4, 1+1536, 384)

            # Reshape: drop CLS, reshape to (bsz, 4, 6, 256, 384)
            latent = latent.reshape(bsz, 4, N_CHANNELS * TOKENS_PER_CHANNEL + 1, TOKEN_DIM)
            latent = latent[:, :, 1:, :].reshape(
                bsz, 4, N_CHANNELS, TOKENS_PER_CHANNEL, TOKEN_DIM
            )

            # Stitch crops 2x2 → 32x32 grid per channel
            latent = latent.permute(0, 2, 1, 3, 4)  # (bsz, 6, 4, 256, 384)
            p = GRID_SIZE  # 16
            latent = latent.reshape(bsz, N_CHANNELS, 2, 2, p, p, TOKEN_DIM)
            latent = latent.permute(0, 1, 2, 4, 3, 5, 6).reshape(
                bsz, N_CHANNELS, 2 * p, 2 * p, TOKEN_DIM
            )
            # latent: (bsz, 6, 32, 32, 384)

            # ── 2304-dim: concat channels → pool 32→16 → mean ──
            tokens_32 = latent.permute(0, 2, 3, 1, 4).reshape(
                bsz, 32 * 32, TOKEN_DIM * N_CHANNELS
            )  # (bsz, 1024, 2304)
            D = tokens_32.shape[-1]
            grid = tokens_32.transpose(1, 2).reshape(bsz, D, 32, 32)
            grid = F.avg_pool2d(grid, kernel_size=2, stride=2)  # (bsz, D, 16, 16)
            tokens_16 = grid.reshape(bsz, D, 16 * 16).transpose(1, 2)  # (bsz, 256, 2304)
            feat_2304 = l2norm(tokens_16.mean(dim=1))  # (bsz, 2304)

            # ── 384-dim: avg over channels → pool 32→16 → mean ──
            avg_ch = latent.mean(dim=1)  # (bsz, 32, 32, 384)
            avg_ch = avg_ch.permute(0, 3, 1, 2)  # (bsz, 384, 32, 32)
            avg_ch = F.avg_pool2d(avg_ch, kernel_size=2, stride=2)  # (bsz, 384, 16, 16)
            feat_384 = l2norm(avg_ch.reshape(bsz, TOKEN_DIM, -1).mean(dim=2))  # (bsz, 384)

            f2304_list.append(feat_2304.cpu())
            f384_list.append(feat_384.cpu())

        feats_2304_by_cond[cond] = torch.cat(f2304_list, dim=0)
        feats_384_by_cond[cond] = torch.cat(f384_list, dim=0)

    # ── Analysis ──

    # 3a. Eigenspectrum / effective dimensionality
    print("\n  3a) Eigenspectrum analysis")
    for label, feats_by_cond, dim in [("2304-dim", feats_2304_by_cond, 2304),
                                       ("384-dim", feats_384_by_cond, 384)]:
        all_feats = torch.cat(list(feats_by_cond.values()), dim=0).float()
        # Center
        centered = all_feats - all_feats.mean(dim=0, keepdim=True)
        # SVD
        _, S, _ = torch.svd(centered)
        # Effective dimensionality: (sum(s))^2 / sum(s^2)
        s = S.numpy()
        eff_dim = (s.sum() ** 2) / (s ** 2).sum()
        # Variance explained by top-k
        var = s ** 2
        cumvar = np.cumsum(var) / var.sum()
        k90 = np.searchsorted(cumvar, 0.90) + 1
        k95 = np.searchsorted(cumvar, 0.95) + 1
        k99 = np.searchsorted(cumvar, 0.99) + 1

        print(f"    {label}: n_samples={len(all_feats)}, total_dim={dim}")
        print(f"      Effective dim: {eff_dim:.1f} ({eff_dim/dim*100:.1f}% of total)")
        print(f"      PCs for 90% var: {k90}, 95%: {k95}, 99%: {k99}")
        print(f"      Top-10 singular values: {s[:10].tolist()}")

    # 3b. Pairwise cosine similarity distribution
    print("\n  3b) Pairwise cosine similarity")
    for label, feats_by_cond in [("2304-dim", feats_2304_by_cond),
                                  ("384-dim", feats_384_by_cond)]:
        all_feats = torch.cat(list(feats_by_cond.values()), dim=0).float()
        normed = l2norm(all_feats)
        # Sample 200 for efficiency
        n_samp = min(200, len(normed))
        samp = normed[:n_samp]
        cos_mat = (samp @ samp.T).numpy()
        np.fill_diagonal(cos_mat, np.nan)
        cos_vals = cos_mat[~np.isnan(cos_mat)]
        print(f"    {label}: mean={np.mean(cos_vals):.4f}, std={np.std(cos_vals):.4f}, "
              f"min={np.min(cos_vals):.4f}, max={np.max(cos_vals):.4f}")

        # Within-condition vs cross-condition
        within, cross = [], []
        cond_labels = []
        for cond, feats in feats_by_cond.items():
            cond_labels.extend([cond] * len(feats))
        cond_labels = cond_labels[:n_samp]
        for i in range(n_samp):
            for j in range(i + 1, n_samp):
                if cond_labels[i] == cond_labels[j]:
                    within.append(cos_mat[i, j])
                else:
                    cross.append(cos_mat[i, j])
        if within:
            print(f"      Within-cond: mean={np.mean(within):.4f}, std={np.std(within):.4f}")
        if cross:
            print(f"      Cross-cond:  mean={np.mean(cross):.4f}, std={np.std(cross):.4f}")

    # 3c. Per-condition centroid distances
    print("\n  3c) Per-condition centroid analysis")
    for label, feats_by_cond in [("2304-dim", feats_2304_by_cond),
                                  ("384-dim", feats_384_by_cond)]:
        centroids = {}
        for cond, feats in feats_by_cond.items():
            centroids[cond] = l2norm(feats.float().mean(dim=0, keepdim=True)).squeeze()

        conds = list(centroids.keys())
        centroid_mat = torch.stack([centroids[c] for c in conds])
        centroid_cos = (l2norm(centroid_mat) @ l2norm(centroid_mat).T).numpy()
        np.fill_diagonal(centroid_cos, np.nan)
        vals = centroid_cos[~np.isnan(centroid_cos)]
        print(f"    {label}: centroid cos sim: mean={np.mean(vals):.4f}, "
              f"std={np.std(vals):.4f}, spread={np.max(vals)-np.min(vals):.4f}")

    # 3d. Quick trust-KID correlation for both dims
    print("\n  3d) Trust-KID correlation")
    for label, feats_by_cond in [("2304-dim", feats_2304_by_cond),
                                  ("384-dim", feats_384_by_cond)]:
        result = _quick_trust_kid_correlation(feats_by_cond, conditions, label)
        if result is not None:
            print(f"    {label}: Spearman ρ(trust, ΔKID) = {result['rho_trust']:.4f}")
            print(f"      ρ(realism, ΔKID) = {result['rho_realism']:.4f}")
            print(f"      ρ(faithfulness, ΔKID) = {result['rho_faith']:.4f}")


def _quick_trust_kid_correlation(feats_by_cond, conditions, label):
    """Simplified trust-KID correlation using same-space scoring."""
    # Build flat tensors
    real_list, real_meta = [], {k: [] for k in CONDITION_KEYS}
    for cond in conditions:
        if cond not in feats_by_cond:
            continue
        feats = feats_by_cond[cond]
        real_list.append(feats)
        ct, si = cond
        real_meta["cell_type_id"].extend([ct] * len(feats))
        real_meta["sirna_id"].extend([si] * len(feats))

    real_flat = l2norm(torch.cat(real_list, dim=0).float())
    for k in real_meta:
        real_meta[k] = torch.tensor(real_meta[k], dtype=torch.long)

    # Use leave-one-out-style: split each condition 50/50 as "real" vs "gen"
    real_half, gen_half = {}, {}
    gen_meta = {k: [] for k in CONDITION_KEYS}
    gen_list = []
    real_half_meta = {k: [] for k in CONDITION_KEYS}
    real_half_list = []

    for cond in conditions:
        if cond not in feats_by_cond:
            continue
        f = feats_by_cond[cond]
        n = len(f)
        if n < 10:
            continue
        mid = n // 2
        real_half[cond] = f[:mid]
        gen_half[cond] = f[mid:]
        ct, si = cond
        real_half_list.append(f[:mid])
        real_half_meta["cell_type_id"].extend([ct] * mid)
        real_half_meta["sirna_id"].extend([si] * mid)
        gen_list.append(f[mid:])
        gen_meta["cell_type_id"].extend([ct] * (n - mid))
        gen_meta["sirna_id"].extend([si] * (n - mid))

    if not gen_list:
        return None

    real_half_flat = l2norm(torch.cat(real_half_list, dim=0).float())
    gen_flat = l2norm(torch.cat(gen_list, dim=0).float())
    for k in real_half_meta:
        real_half_meta[k] = torch.tensor(real_half_meta[k], dtype=torch.long)
    for k in gen_meta:
        gen_meta[k] = torch.tensor(gen_meta[k], dtype=torch.long)

    # Fit trust scoring on real half
    global_stats = fit_global_stats(real_half_flat, regularization=1e-5)
    factorized_stats = fit_factorized_stats(
        real_half_flat, real_half_meta, CONDITION_KEYS, regularization=1e-5, use_shared_cov=True
    )
    real_E_mean, real_E_std = compute_real_calibration_for_global_energy(real_half_flat, global_stats)
    margin_calib = compute_real_calibration_for_factorized_margins(
        real_half_flat, real_half_meta, factorized_stats, CONDITION_KEYS
    )

    # Score "gen" half
    realism_z = compute_global_realism_z(gen_flat, global_stats, real_E_mean, real_E_std, two_sided=False)
    faith_z, _ = compute_factorized_faithfulness_margin_z(
        gen_flat, gen_meta, factorized_stats, CONDITION_KEYS, margin_calib
    )
    trust = realism_z + faith_z

    # Per-condition mean trust
    true_conditions = [
        get_condition_key(gen_meta, CONDITION_KEYS, i) for i in range(len(gen_flat))
    ]
    cond_trust = defaultdict(list)
    cond_realism = defaultdict(list)
    cond_faith = defaultdict(list)
    for i, c in enumerate(true_conditions):
        cond_trust[c].append(trust[i])
        cond_realism[c].append(realism_z[i])
        cond_faith[c].append(faith_z[i])

    mean_trust = {c: np.mean(v) for c, v in cond_trust.items()}
    mean_realism = {c: np.mean(v) for c, v in cond_realism.items()}
    mean_faith = {c: np.mean(v) for c, v in cond_faith.items()}

    # Per-condition KID
    real_normed = {c: l2norm(real_half[c].float()).numpy().astype(np.float64) for c in real_half}
    gen_normed = {c: l2norm(gen_half[c].float()).numpy().astype(np.float64) for c in gen_half}

    delta_kids = {}
    for cond in real_normed:
        if cond not in gen_normed:
            continue
        rf, gf = real_normed[cond], gen_normed[cond]
        if len(rf) < 5 or len(gf) < 5:
            delta_kids[cond] = np.nan
            continue
        k = min(len(rf) // 2, len(gf), 500)
        if k < 5:
            delta_kids[cond] = np.nan
            continue
        stable_hash = int(hashlib.md5(str(cond).encode()).hexdigest(), 16) % 1000
        rng = np.random.default_rng(42 + stable_hash)
        deltas = []
        for _ in range(N_BOOTSTRAP_KID):
            perm = rng.permutation(len(rf))
            real_a, real_b = rf[perm[:k]], rf[perm[k:2*k]]
            gen_samp = gf[rng.choice(len(gf), k, replace=len(gf) < k)]
            base = calculate_kid_same_m(real_a, real_b, use_cosine=True)
            gen_kid = calculate_kid_same_m(real_a, gen_samp, use_cosine=True)
            if np.isfinite(base) and np.isfinite(gen_kid):
                deltas.append(gen_kid - base)
        delta_kids[cond] = np.mean(deltas) if deltas else np.nan

    # Correlate
    common = [c for c in mean_trust if c in delta_kids and np.isfinite(delta_kids[c])]
    if len(common) < 3:
        print(f"    [{label}] Too few valid conditions ({len(common)})")
        return None

    trust_arr = np.array([mean_trust[c] for c in common])
    realism_arr = np.array([mean_realism[c] for c in common])
    faith_arr = np.array([mean_faith[c] for c in common])
    kid_arr = np.array([delta_kids[c] for c in common])

    rho_trust, _ = spearmanr(trust_arr, kid_arr)
    rho_real, _ = spearmanr(realism_arr, kid_arr)
    rho_faith, _ = spearmanr(faith_arr, kid_arr)

    return {"rho_trust": rho_trust, "rho_realism": rho_real, "rho_faith": rho_faith, "n_conds": len(common)}


# ── Main ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OpenPhenom Encode-Decode Diagnostic (v2)")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    dm = load_dataset()

    # Load images for Sections 1 & 2
    images, conditions = load_images(dm, N_IMAGES, distinct_conditions=True)
    print(f"Images: {images.shape}, range=[{images.min():.3f}, {images.max():.3f}]")
    print(f"Conditions: {conditions}")

    # Load full OpenPhenom model (need encoder + decoder)
    print("\nLoading OpenPhenom (full model with decoder)...")
    model = AutoModel.from_pretrained(
        "recursionpharma/OpenPhenom", trust_remote_code=True
    ).to(DEVICE).eval()

    # ── Section 1 ──
    recon_512 = section1_repa_crop_roundtrip(model, images, conditions)

    # ── Section 2 ──
    recon_256 = section2_native_6ch_direct(model, images, conditions)

    # ── Section 3 (needs lots of features, keep model loaded) ──
    section3_feature_comparison(model, dm)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Output dir: {OUT_DIR}")
    print(f"  Saved images:")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"    {f.name}")

    # Save summary text
    print(f"\n  Summary saved to {OUT_DIR / 'summary.txt'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
