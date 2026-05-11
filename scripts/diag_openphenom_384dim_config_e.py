"""
Config E retest with 384-dim OpenPhenom features (channel avg instead of concat).

Loads pre-generated images from rxrx1_repa_openphenom_full, extracts:
  - 384-dim OpenPhenom features (avg over 6 channels, then pool 32→16, mean-patch)
  - 2304-dim OpenPhenom features (concat 6 channels, pool 32→16, mean-patch) for comparison
  - DINOv3 features (for KID ground truth)

Then runs trust-KID correlation in 3 configs:
  D-384: OP-384 trust + OP-384 KID (same-space)
  D-2304: OP-2304 trust + OP-2304 KID (same-space)
  E-384: OP-384 trust + DINOv3 KID (cross-space)
  E-2304: OP-2304 trust + DINOv3 KID (cross-space)
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm

from faithful_cond_gen.data.rxrx1 import RxRx1DataModule, RxRx1DataConfig
from faithful_cond_gen.model.repa_encoder import REPAEncoder, load_repa_encoder
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
GEN_DIR = Path("outputs/gen/rxrx1_repa_openphenom_full/images")
TOP_K = 50
SAMPLES_PER_COND = 50  # use first 50 of 100 generated
GEN_BATCH_SIZE = 16
CONDITION_KEYS = ["cell_type_id", "sirna_id"]
N_BOOTSTRAP_KID = 10
# ──────────────────────────────────────────────────────────────────────────────


def l2norm(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


# ── Load generated images ────────────────────────────────────────────────────

def load_gen_images(conditions):
    """Load generated .pt images by condition."""
    gen_by_cond = {}
    for ct, si in tqdm(conditions, desc="Loading gen images"):
        prefix = f"cell{ct}_sirna{si}"
        imgs = []
        for idx in range(SAMPLES_PER_COND):
            p = GEN_DIR / f"{prefix}_{idx}.pt"
            if p.exists():
                imgs.append(torch.load(p, map_location="cpu"))
        if imgs:
            gen_by_cond[(ct, si)] = torch.stack(imgs)
    return gen_by_cond


def load_real_images(dm, conditions):
    """Load real images by condition from dataset."""
    ds = dm.train_dataloader().dataset
    cond_set = set(conditions)
    cond_to_idx = defaultdict(list)
    for i, (ct, si) in enumerate(zip(ds.cell_type_ids, ds.sirna_ids)):
        key = (int(ct), int(si))
        if key in cond_set:
            cond_to_idx[key].append(i)

    real_by_cond = {}
    for cond in tqdm(conditions, desc="Loading real images"):
        indices = cond_to_idx[cond]
        real_by_cond[cond] = torch.stack([ds[i][0] for i in indices])
    return real_by_cond


# ── OpenPhenom 384-dim and 2304-dim extraction ───────────────────────────────

@torch.no_grad()
def extract_openphenom_both_dims(images_by_cond):
    """Extract both 384-dim (channel avg) and 2304-dim (channel concat) features."""
    encoder = load_repa_encoder("openphenom", 256, 6, device=DEVICE).eval()
    # We'll call _encode_latent_openphenom directly for the 384 path

    feats_384 = {}
    feats_2304 = {}

    for cond, imgs in tqdm(images_by_cond.items(), desc="OP features"):
        f384_list, f2304_list = [], []

        for start in range(0, len(imgs), GEN_BATCH_SIZE):
            batch = imgs[start:start + GEN_BATCH_SIZE].to(DEVICE)
            if batch.min() < 0:
                batch = (batch + 1) / 2
            B = batch.shape[0]

            # Upsample to 512 (encoder does this internally, but we need the intermediate)
            batch_512 = F.interpolate(batch, size=(512, 512), mode="bilinear", align_corners=False)

            # Get (B, 6, 32, 32, 384) latent
            latent = encoder._encode_latent_openphenom(batch_512)

            # ── 2304-dim: concat channels → pool → mean (matches _forward_openphenom) ──
            tokens_32 = latent.permute(0, 2, 3, 1, 4).reshape(
                B, 32 * 32, 384 * 6
            )
            D = tokens_32.shape[-1]
            grid = tokens_32.transpose(1, 2).reshape(B, D, 32, 32)
            grid = F.avg_pool2d(grid, kernel_size=2, stride=2)
            tokens_16 = grid.reshape(B, D, 256).transpose(1, 2)  # (B, 256, 2304)
            feat_2304 = l2norm(tokens_16.mean(dim=1))  # (B, 2304)

            # ── 384-dim: avg over channels → pool → mean ──
            avg_ch = latent.mean(dim=1)  # (B, 32, 32, 384)
            avg_ch = avg_ch.permute(0, 3, 1, 2)  # (B, 384, 32, 32)
            avg_ch = F.avg_pool2d(avg_ch, kernel_size=2, stride=2)  # (B, 384, 16, 16)
            feat_384 = l2norm(avg_ch.reshape(B, 384, 256).mean(dim=2))  # (B, 384)

            f384_list.append(feat_384.cpu())
            f2304_list.append(feat_2304.cpu())

        feats_384[cond] = torch.cat(f384_list, dim=0)
        feats_2304[cond] = torch.cat(f2304_list, dim=0)

    del encoder
    torch.cuda.empty_cache()
    return feats_384, feats_2304


# ── DINOv3 features ──────────────────────────────────────────────────────────

@torch.no_grad()
def extract_dinov3_features(images_by_cond):
    encoder = REPAEncoder(
        encoder_name="dinov3-vit-l",
        resolution=256,
        in_channels=6,
    ).to(DEVICE).eval()
    print(f"DINOv3 loaded, embed_dim={encoder.embed_dim}")

    feats = {}
    for cond, imgs in tqdm(images_by_cond.items(), desc="DINOv3 features"):
        fl = []
        for start in range(0, len(imgs), GEN_BATCH_SIZE):
            batch = imgs[start:start + GEN_BATCH_SIZE].to(DEVICE)
            if batch.min() < 0:
                batch = (batch + 1) / 2
            out = encoder(batch)
            pooled = out.mean(dim=1) if out.ndim == 3 else out
            fl.append(pooled.cpu())
        feats[cond] = torch.cat(fl, dim=0)

    del encoder
    torch.cuda.empty_cache()
    return feats


# ── Trust-KID correlation ────────────────────────────────────────────────────

def compute_per_condition_kid(real_feats, gen_feats, use_cosine=True):
    kids = {}
    for cond in real_feats:
        if cond not in gen_feats:
            continue
        rf = real_feats[cond].numpy().astype(np.float64)
        gf = gen_feats[cond].numpy().astype(np.float64)
        if len(rf) < 20 or len(gf) < 5:
            kids[cond] = np.nan
            continue
        k = min(len(rf) // 2, len(gf), 500)
        if k < 5:
            kids[cond] = np.nan
            continue
        stable_hash = int(hashlib.md5(str(cond).encode()).hexdigest(), 16) % 1000
        rng = np.random.default_rng(42 + stable_hash)
        deltas = []
        for _ in range(N_BOOTSTRAP_KID):
            perm = rng.permutation(len(rf))
            real_a, real_b = rf[perm[:k]], rf[perm[k:2*k]]
            gen_samp = gf[rng.choice(len(gf), k, replace=False)]
            base = calculate_kid_same_m(real_a, real_b, use_cosine=use_cosine)
            gen_kid = calculate_kid_same_m(real_a, gen_samp, use_cosine=use_cosine)
            if np.isfinite(base) and np.isfinite(gen_kid):
                deltas.append(gen_kid - base)
        kids[cond] = np.mean(deltas) if deltas else np.nan
    return kids


def trust_kid_correlate(real_feats, gen_feats, conditions, label,
                        kid_real=None, kid_gen=None):
    """Fit trust on real, score gen, correlate with KID."""
    real_list, real_meta = [], {k: [] for k in CONDITION_KEYS}
    for cond in conditions:
        if cond not in real_feats:
            continue
        feats = real_feats[cond]
        real_list.append(feats)
        ct, si = cond
        real_meta["cell_type_id"].extend([ct] * len(feats))
        real_meta["sirna_id"].extend([si] * len(feats))
    real_flat = l2norm(torch.cat(real_list, dim=0).float())
    for k in real_meta:
        real_meta[k] = torch.tensor(real_meta[k], dtype=torch.long)

    gen_list, gen_meta = [], {k: [] for k in CONDITION_KEYS}
    for cond in conditions:
        if cond not in gen_feats:
            continue
        feats = gen_feats[cond]
        gen_list.append(feats)
        ct, si = cond
        gen_meta["cell_type_id"].extend([ct] * len(feats))
        gen_meta["sirna_id"].extend([si] * len(feats))
    gen_flat = l2norm(torch.cat(gen_list, dim=0).float())
    for k in gen_meta:
        gen_meta[k] = torch.tensor(gen_meta[k], dtype=torch.long)

    print(f"\n  [{label}] real={real_flat.shape}, gen={gen_flat.shape}")

    # Fit
    global_stats = fit_global_stats(real_flat, regularization=1e-5)
    factorized_stats = fit_factorized_stats(
        real_flat, real_meta, CONDITION_KEYS, regularization=1e-5, use_shared_cov=True
    )
    real_E_mean, real_E_std = compute_real_calibration_for_global_energy(real_flat, global_stats)
    margin_calib = compute_real_calibration_for_factorized_margins(
        real_flat, real_meta, factorized_stats, CONDITION_KEYS
    )

    # Score
    realism_z = compute_global_realism_z(gen_flat, global_stats, real_E_mean, real_E_std, two_sided=False)
    faith_z, _ = compute_factorized_faithfulness_margin_z(
        gen_flat, gen_meta, factorized_stats, CONDITION_KEYS, margin_calib
    )
    trust = realism_z + faith_z

    # Per-condition means
    true_conds = [get_condition_key(gen_meta, CONDITION_KEYS, i) for i in range(len(gen_flat))]
    cond_trust = defaultdict(list)
    cond_realism = defaultdict(list)
    cond_faith = defaultdict(list)
    for i, c in enumerate(true_conds):
        cond_trust[c].append(trust[i])
        cond_realism[c].append(realism_z[i])
        cond_faith[c].append(faith_z[i])
    mean_trust = {c: np.mean(v) for c, v in cond_trust.items()}
    mean_realism = {c: np.mean(v) for c, v in cond_realism.items()}
    mean_faith = {c: np.mean(v) for c, v in cond_faith.items()}

    # KID (use provided or same-space)
    if kid_real is not None and kid_gen is not None:
        kr = {c: l2norm(kid_real[c].float()).numpy().astype(np.float64) for c in conditions if c in kid_real}
        kg = {c: l2norm(kid_gen[c].float()).numpy().astype(np.float64) for c in conditions if c in kid_gen}
    else:
        kr = {c: l2norm(real_feats[c].float()).numpy().astype(np.float64) for c in conditions if c in real_feats}
        kg = {c: l2norm(gen_feats[c].float()).numpy().astype(np.float64) for c in conditions if c in gen_feats}

    delta_kids = compute_per_condition_kid(
        {c: torch.from_numpy(kr[c]) for c in kr},
        {c: torch.from_numpy(kg[c]) for c in kg},
    )

    common = [c for c in mean_trust if c in delta_kids and np.isfinite(delta_kids[c])]
    if len(common) < 3:
        print(f"  [{label}] Too few conditions ({len(common)})")
        return None

    trust_arr = np.array([mean_trust[c] for c in common])
    realism_arr = np.array([mean_realism[c] for c in common])
    faith_arr = np.array([mean_faith[c] for c in common])
    kid_arr = np.array([delta_kids[c] for c in common])

    rho_trust, p_trust = spearmanr(trust_arr, kid_arr)
    rho_real, _ = spearmanr(realism_arr, kid_arr)
    rho_faith, _ = spearmanr(faith_arr, kid_arr)

    print(f"  [{label}] N={len(common)} conditions")
    print(f"  [{label}] ρ(trust, ΔKID)         = {rho_trust:.4f}  (p={p_trust:.4f})")
    print(f"  [{label}] ρ(realism, ΔKID)       = {rho_real:.4f}")
    print(f"  [{label}] ρ(faithfulness, ΔKID)  = {rho_faith:.4f}")

    return {"rho_trust": rho_trust, "rho_realism": rho_real, "rho_faith": rho_faith}


# ── Main ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main():
    print("=" * 70)
    print("Config E retest: 384-dim vs 2304-dim OpenPhenom features")
    print("=" * 70)

    # Dataset + conditions
    print("\n[1/5] Loading dataset...")
    dm = RxRx1DataModule(RxRx1DataConfig(
        data_dir="/mnt/pvc/AutoSync/data/rxrx1",
        img_size=[512, 512], resize=[256, 256],
        reduce_channels=False, augment_train=False, normalize=False,
        use_numpy=True, use_parquet=False,
        batch_size=GEN_BATCH_SIZE, num_workers=4, val_size=0.1,
        seed=1337, rare_threshold=20, held_out_pairs=None,
    ))
    ds = dm.train_dataloader().dataset
    counts = defaultdict(int)
    for ct, si in zip(ds.cell_type_ids, ds.sirna_ids):
        counts[(int(ct), int(si))] += 1
    conditions = [c for c, _ in sorted(counts.items(), key=lambda x: -x[1])[:TOP_K]]
    print(f"  {len(conditions)} conditions, min={sorted(counts.values())[-TOP_K]} samples")

    # Load images
    print("\n[2/5] Loading images...")
    gen_images = load_gen_images(conditions)
    real_images = load_real_images(dm, conditions)
    print(f"  Gen: {len(gen_images)} conds, Real: {len(real_images)} conds")

    # Extract OP features (both dims)
    print("\n[3/5] Extracting OpenPhenom features (384 + 2304)...")
    real_op384, real_op2304 = extract_openphenom_both_dims(real_images)
    gen_op384, gen_op2304 = extract_openphenom_both_dims(gen_images)

    # Quick feature stats
    for label, feats in [("OP-384 real", real_op384), ("OP-2304 real", real_op2304)]:
        all_f = torch.cat(list(feats.values()), dim=0)
        nf = l2norm(all_f.float())
        samp = nf[:200]
        cos = (samp @ samp.T).numpy()
        np.fill_diagonal(cos, np.nan)
        vals = cos[~np.isnan(cos)]
        print(f"  {label}: dim={all_f.shape[1]}, pairwise cos mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

    # Extract DINOv3 features
    print("\n[4/5] Extracting DINOv3 features...")
    real_dinov3 = extract_dinov3_features(real_images)
    gen_dinov3 = extract_dinov3_features(gen_images)

    # Free images
    del real_images, gen_images
    torch.cuda.empty_cache()

    # Run configs
    print("\n[5/5] Trust-KID correlations")

    print("\n" + "=" * 70)
    print("D-2304: OP-2304 trust + OP-2304 KID (same-space, original)")
    print("=" * 70)
    res_D2304 = trust_kid_correlate(real_op2304, gen_op2304, conditions, "D-2304")

    print("\n" + "=" * 70)
    print("D-384: OP-384 trust + OP-384 KID (same-space)")
    print("=" * 70)
    res_D384 = trust_kid_correlate(real_op384, gen_op384, conditions, "D-384")

    print("\n" + "=" * 70)
    print("E-2304: OP-2304 trust + DINOv3 KID (cross-space, original)")
    print("=" * 70)
    res_E2304 = trust_kid_correlate(
        real_op2304, gen_op2304, conditions, "E-2304",
        kid_real=real_dinov3, kid_gen=gen_dinov3,
    )

    print("\n" + "=" * 70)
    print("E-384: OP-384 trust + DINOv3 KID (cross-space)")
    print("=" * 70)
    res_E384 = trust_kid_correlate(
        real_op384, gen_op384, conditions, "E-384",
        kid_real=real_dinov3, kid_gen=gen_dinov3,
    )

    print("\n" + "=" * 70)
    print("C: DINOv3 trust + DINOv3 KID (reference)")
    print("=" * 70)
    res_C = trust_kid_correlate(real_dinov3, gen_dinov3, conditions, "C-ref")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    def _r(res, key="rho_trust"):
        return f"{res[key]:.4f}" if res else "N/A"

    print(f"  {'Config':<12} {'ρ(trust,KID)':>14} {'ρ(realism,KID)':>16} {'ρ(faith,KID)':>14}")
    print(f"  {'-'*58}")
    for name, res in [("D-2304", res_D2304), ("D-384", res_D384),
                      ("E-2304", res_E2304), ("E-384", res_E384),
                      ("C-ref", res_C)]:
        print(f"  {name:<12} {_r(res,'rho_trust'):>14} {_r(res,'rho_realism'):>16} {_r(res,'rho_faith'):>14}")

    print("\nDone.")


if __name__ == "__main__":
    main()
