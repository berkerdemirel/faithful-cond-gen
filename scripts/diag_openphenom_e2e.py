"""
End-to-end diagnostic for openphenom REPA aligned features.

Self-contained script that reproduces the full pipeline on 50 (cell_type, sirna)
pairs to isolate where the correlation breaks:

  1. Load RxRx1 train set, pick top-50 (ct, sirna) pairs by sample count.
  2. Load repa_openphenom_full model checkpoint.
  3. REAL features: encode real images → add noise at t=0.01 → projector → mean-pool → L2-norm.
  4. GEN features: run full generation (250 steps) → capture projector at final step → mean-pool → L2-norm.
  5. Also extract DINOv3 meanpatch features for both real and generated (ground truth reference).
  6. Compute per-condition KID (cosine kernel) for aligned_mean and dinov3.
  7. Compute trust scores and Spearman correlation (trust vs KID).

All done in a single script with no external cached files.
"""

import sys
import hashlib
from collections import defaultdict

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from hydra.utils import instantiate
from scipy.stats import spearmanr
from pathlib import Path
from tqdm import tqdm

from faithful_cond_gen.utils.checkpoints import get_checkpoint_path
from faithful_cond_gen.pl_modules.generator_pl import GeneratorPL
from faithful_cond_gen.model.generator import GeneratorWrapper
from faithful_cond_gen.data.rxrx1 import RxRx1DataModule, RxRx1DataConfig
from faithful_cond_gen.model.repa_encoder import REPAEncoder, load_repa_encoder
from faithful_cond_gen.eval.trust_eval.metrics_kid import calculate_kid_same_m
from faithful_cond_gen.eval.trust_eval.scoring_core import (
    normalize_features,
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
GEN_DIR = Path("outputs/gen/rxrx1_repa_openphenom_full")
TOP_K_CONDITIONS = 50
SAMPLES_PER_CONDITION = 50      # generate this many per condition
GEN_BATCH_SIZE = 16
NUM_INFERENCE_STEPS = 250
T_CUTOFF = 0.04
NOISE_TIMESTEP = 0.01           # for real feature extraction
CONDITION_KEYS = ["cell_type_id", "sirna_id"]
N_BOOTSTRAP_KID = 10
# ──────────────────────────────────────────────────────────────────────────────


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


def kid_cosine(X, Y):
    """Unbiased MMD^2 with cosine kernel, matching pipeline exactly."""
    return calculate_kid_same_m(X, Y, use_cosine=True)


# ── Step 1: Pick top-50 conditions ───────────────────────────────────────────
def pick_top_conditions(dm, k=TOP_K_CONDITIONS):
    """Pick the k (cell_type_id, sirna_id) pairs with most training samples."""
    ds = dm.train_dataloader().dataset
    # Fast: use pre-computed numpy arrays instead of iterating __getitem__
    ct_arr = ds.cell_type_ids
    si_arr = ds.sirna_ids
    counts = defaultdict(int)
    for ct, si in zip(ct_arr, si_arr):
        counts[(int(ct), int(si))] += 1
    top = sorted(counts.items(), key=lambda x: -x[1])[:k]
    print(f"Top-{k} conditions by sample count:")
    for (ct, si), n in top[:5]:
        print(f"  cell={ct} sirna={si}: {n} samples")
    print(f"  ... (min={top[-1][1]} samples)")
    return [c for c, _ in top]


# ── Step 2: Load model ───────────────────────────────────────────────────────
def load_model():
    cfg = OmegaConf.load(GEN_DIR / "gen_config.yaml")
    gen_cfg = instantiate(cfg.model)
    pl = GeneratorPL.load_from_checkpoint(
        get_checkpoint_path(cfg.checkpoint_key),
        generator=GeneratorWrapper(gen_cfg),
        map_location=DEVICE, strict=False,
    )
    if hasattr(pl, "ema"):
        pl.ema.apply()
    pl.to(DEVICE).eval()
    print(f"Loaded model: use_repa={pl.generator.cfg.use_repa}, encoder={pl.generator.cfg.repa_encoder}")
    return pl


# ── Step 3: Extract real aligned features ─────────────────────────────────────
def build_cond_index(ds, conditions):
    """Fast condition → dataset index mapping using pre-computed arrays."""
    cond_set = set(conditions)
    cond_to_idx = defaultdict(list)
    for i, (ct, si) in enumerate(zip(ds.cell_type_ids, ds.sirna_ids)):
        key = (int(ct), int(si))
        if key in cond_set:
            cond_to_idx[key].append(i)
    return cond_to_idx


@torch.no_grad()
def extract_real_features(pl, dm, conditions):
    """For each condition, get real images → encode → noise → projector → mean-pool."""
    ds = dm.train_dataloader().dataset
    cond_to_idx = build_cond_index(ds, conditions)

    all_feats = {}
    for cond in tqdm(conditions, desc="Real features"):
        indices = cond_to_idx[cond]
        feats_list = []
        ct, si = cond
        cond_ids = torch.tensor([[ct, si]], device=DEVICE, dtype=torch.long)

        for start in range(0, len(indices), GEN_BATCH_SIZE):
            batch_idx = indices[start : start + GEN_BATCH_SIZE]
            images = torch.stack([ds[i][0] for i in batch_idx]).to(DEVICE)
            B = images.shape[0]

            # Match pipeline: ensure [0,1]
            if images.min() < 0:
                images = (images + 1) / 2
            images = images.contiguous()

            # Encode to latent
            latents = pl.generator.encode(images)

            # Add noise at t=NOISE_TIMESTEP (linear schedule)
            alpha_bar = 1 - NOISE_TIMESTEP
            noise = torch.randn_like(latents)
            noisy = np.sqrt(alpha_bar) * latents + np.sqrt(1 - alpha_bar) * noise
            t_tensor = torch.full((B,), NOISE_TIMESTEP, device=DEVICE, dtype=torch.float32)
            batch_cond = cond_ids.expand(B, -1)

            # Forward pass → projector features
            _, zs = pl.generator.velocity_prediction(
                noisy, t_tensor, batch_cond, return_projected=True
            )
            proj = zs[0]  # (B, num_patches, D)
            pooled = proj.mean(dim=1) if proj.ndim == 3 else proj  # (B, D)
            feats_list.append(pooled.cpu())

        all_feats[cond] = torch.cat(feats_list, dim=0)
    return all_feats


# ── Step 4: Generate samples + capture aligned features ──────────────────────
@torch.no_grad()
def generate_and_extract(pl, conditions):
    """Generate SAMPLES_PER_CONDITION images per condition, capture projector at final step."""
    all_gen_feats = {}
    all_gen_images = {}

    for cond in tqdm(conditions, desc="Generating"):
        ct, si = cond
        cond_ids = torch.tensor([[ct, si]], device=DEVICE, dtype=torch.long)
        feats_list = []
        imgs_list = []

        remaining = SAMPLES_PER_CONDITION
        while remaining > 0:
            bs = min(remaining, GEN_BATCH_SIZE)
            batch_cond = cond_ids.expand(bs, -1)

            # This matches generate_samples_repa.py: feature_capture_idx=None → final Stage 2
            images, aligned_features = pl.generator.sample(
                cond_ids=batch_cond,
                num_inference_steps=NUM_INFERENCE_STEPS,
                t_cutoff=T_CUTOFF,
                cfg_scale=1.0,
                return_aligned_features=True,
                feature_capture_idx=None,  # default = final Stage 2
            )
            images = torch.clamp(images, 0, 1)

            # aligned_features is List[Tensor(B, num_patches, D)] for single mode
            proj = aligned_features[0]  # (B, P, D)
            pooled = proj.mean(dim=1) if proj.ndim == 3 else proj
            feats_list.append(pooled.cpu())
            imgs_list.append(images.cpu())
            remaining -= bs

        all_gen_feats[cond] = torch.cat(feats_list, dim=0)
        all_gen_images[cond] = torch.cat(imgs_list, dim=0)

    return all_gen_feats, all_gen_images


# ── Step 5: DINOv3 features (reference ground truth) ─────────────────────────
@torch.no_grad()
def extract_dinov3_features(real_images_by_cond, gen_images_by_cond):
    """Extract DINOv3 meanpatch features for real and gen images."""
    encoder = REPAEncoder(
        encoder_name="dinov3-vit-l",
        resolution=256,
        in_channels=6,
    ).to(DEVICE).eval()
    print(f"DINOv3 encoder loaded, embed_dim={encoder.embed_dim}")

    def extract(images_by_cond, desc):
        feats = {}
        for cond, imgs in tqdm(images_by_cond.items(), desc=desc):
            fl = []
            for start in range(0, len(imgs), GEN_BATCH_SIZE):
                batch = imgs[start : start + GEN_BATCH_SIZE].to(DEVICE)
                if batch.min() < 0:
                    batch = (batch + 1) / 2
                out = encoder(batch)  # (B, P, D)
                pooled = out.mean(dim=1) if out.ndim == 3 else out
                fl.append(pooled.cpu())
            feats[cond] = torch.cat(fl, dim=0)
        return feats

    # For real images, we need to get them from the dataset
    # real_images_by_cond is already loaded
    real_dinov3 = extract(real_images_by_cond, "DINOv3 real")
    gen_dinov3 = extract(gen_images_by_cond, "DINOv3 gen")

    del encoder
    torch.cuda.empty_cache()
    return real_dinov3, gen_dinov3


# ── Step 5b: OpenPhenom features (post-hoc teacher) ─────────────────────────
@torch.no_grad()
def extract_openphenom_features(real_images_by_cond, gen_images_by_cond):
    """Extract OpenPhenom meanpatch features for real and gen images (post-hoc)."""
    encoder = load_repa_encoder(
        encoder_name="openphenom",
        resolution=256,
        in_channels=6,
        device=DEVICE,
    ).eval()
    print(f"OpenPhenom encoder loaded, embed_dim={encoder.embed_dim}")

    def extract(images_by_cond, desc):
        feats = {}
        for cond, imgs in tqdm(images_by_cond.items(), desc=desc):
            fl = []
            for start in range(0, len(imgs), GEN_BATCH_SIZE):
                batch = imgs[start : start + GEN_BATCH_SIZE].to(DEVICE)
                if batch.min() < 0:
                    batch = (batch + 1) / 2
                out = encoder(batch)  # (B, P, D)
                pooled = out.mean(dim=1) if out.ndim == 3 else out
                fl.append(pooled.cpu())
            feats[cond] = torch.cat(fl, dim=0)
        return feats

    real_op = extract(real_images_by_cond, "OpenPhenom real")
    gen_op = extract(gen_images_by_cond, "OpenPhenom gen")

    del encoder
    torch.cuda.empty_cache()
    return real_op, gen_op


# ── Step 6: Per-condition KID ─────────────────────────────────────────────────
def compute_per_condition_kid(real_feats_by_cond, gen_feats_by_cond, use_cosine=True):
    """Compute bootstrapped delta-KID per condition, exactly as the pipeline does."""
    kids = {}
    for cond in real_feats_by_cond:
        if cond not in gen_feats_by_cond:
            continue
        real_f = real_feats_by_cond[cond].numpy().astype(np.float64)
        gen_f = gen_feats_by_cond[cond].numpy().astype(np.float64)

        if len(real_f) < 20 or len(gen_f) < 5:
            kids[cond] = np.nan
            continue

        k = min(len(real_f) // 2, len(gen_f), 500)
        if k < 5:
            kids[cond] = np.nan
            continue

        stable_hash = int(hashlib.md5(str(cond).encode()).hexdigest(), 16) % 1000
        rng = np.random.default_rng(42 + stable_hash)
        deltas = []
        for _ in range(N_BOOTSTRAP_KID):
            perm = rng.permutation(len(real_f))
            real_a, real_b = real_f[perm[:k]], real_f[perm[k : 2 * k]]
            gen_samp = gen_f[rng.choice(len(gen_f), k, replace=False)]
            base = calculate_kid_same_m(real_a, real_b, use_cosine=use_cosine)
            gen_kid = calculate_kid_same_m(real_a, gen_samp, use_cosine=use_cosine)
            if np.isfinite(base) and np.isfinite(gen_kid):
                deltas.append(gen_kid - base)
        kids[cond] = np.mean(deltas) if deltas else np.nan
    return kids


# ── Step 7: Trust scores + correlation ────────────────────────────────────────
def compute_trust_and_correlate(real_feats_by_cond, gen_feats_by_cond, conditions, label,
                                kid_real_by_cond=None, kid_gen_by_cond=None):
    """Fit trust scoring on real, score gen, correlate with KID."""
    # Stack into flat tensors with metadata
    real_list, real_meta = [], {k: [] for k in CONDITION_KEYS}
    for cond in conditions:
        feats = real_feats_by_cond[cond]
        real_list.append(feats)
        ct, si = cond
        real_meta["cell_type_id"].extend([ct] * len(feats))
        real_meta["sirna_id"].extend([si] * len(feats))
    real_flat = l2norm(torch.cat(real_list, dim=0).float())
    for k in real_meta:
        real_meta[k] = torch.tensor(real_meta[k], dtype=torch.long)

    gen_list, gen_meta = [], {k: [] for k in CONDITION_KEYS}
    for cond in conditions:
        if cond not in gen_feats_by_cond:
            continue
        feats = gen_feats_by_cond[cond]
        gen_list.append(feats)
        ct, si = cond
        gen_meta["cell_type_id"].extend([ct] * len(feats))
        gen_meta["sirna_id"].extend([si] * len(feats))
    gen_flat = l2norm(torch.cat(gen_list, dim=0).float())
    for k in gen_meta:
        gen_meta[k] = torch.tensor(gen_meta[k], dtype=torch.long)

    print(f"\n  [{label}] real={real_flat.shape}, gen={gen_flat.shape}")

    # Fit global + factorized stats on real (L2-normalized)
    global_stats = fit_global_stats(real_flat, regularization=1e-5)
    factorized_stats = fit_factorized_stats(
        real_flat, real_meta, CONDITION_KEYS, regularization=1e-5, use_shared_cov=True
    )
    real_E_mean, real_E_std = compute_real_calibration_for_global_energy(real_flat, global_stats)
    margin_calib = compute_real_calibration_for_factorized_margins(
        real_flat, real_meta, factorized_stats, CONDITION_KEYS
    )

    # Score generated
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

    # L2-normalize per-condition features for KID
    # If kid_real/kid_gen provided, use those for KID (cross-space evaluation)
    if kid_real_by_cond is not None and kid_gen_by_cond is not None:
        real_kid_normed = {c: l2norm(kid_real_by_cond[c].float()).numpy().astype(np.float64) for c in conditions if c in kid_real_by_cond}
        gen_kid_normed = {c: l2norm(kid_gen_by_cond[c].float()).numpy().astype(np.float64) for c in conditions if c in kid_gen_by_cond}
    else:
        real_kid_normed = {c: l2norm(real_feats_by_cond[c].float()).numpy().astype(np.float64) for c in conditions}
        gen_kid_normed = {c: l2norm(gen_feats_by_cond[c].float()).numpy().astype(np.float64)
                      for c in conditions if c in gen_feats_by_cond}

    # Per-condition delta-KID
    delta_kids = compute_per_condition_kid(
        {c: torch.from_numpy(real_kid_normed[c]) for c in real_kid_normed},
        {c: torch.from_numpy(gen_kid_normed[c]) for c in gen_kid_normed},
        use_cosine=True,
    )

    # Correlate
    common = [c for c in mean_trust if c in delta_kids and np.isfinite(delta_kids[c])]
    if len(common) < 3:
        print(f"  [{label}] Too few valid conditions ({len(common)}) for correlation")
        return None

    trust_arr = np.array([mean_trust[c] for c in common])
    realism_arr = np.array([mean_realism[c] for c in common])
    faith_arr = np.array([mean_faith[c] for c in common])
    kid_arr = np.array([delta_kids[c] for c in common])

    rho_trust, p_trust = spearmanr(trust_arr, kid_arr)
    rho_real, _ = spearmanr(realism_arr, kid_arr)
    rho_faith, _ = spearmanr(faith_arr, kid_arr)

    print(f"  [{label}] N conditions: {len(common)}")
    print(f"  [{label}] Spearman ρ (trust vs ΔKID):         {rho_trust:.4f} (p={p_trust:.4f})")
    print(f"  [{label}] Spearman ρ (realism vs ΔKID):       {rho_real:.4f}")
    print(f"  [{label}] Spearman ρ (faithfulness vs ΔKID):  {rho_faith:.4f}")

    # Feature space diagnostics (use scoring-space features)
    print(f"\n  [{label}] Feature diagnostics:")
    scoring_real = {c: l2norm(real_feats_by_cond[c].float()).numpy() for c in conditions}
    scoring_gen = {c: l2norm(gen_feats_by_cond[c].float()).numpy() for c in conditions if c in gen_feats_by_cond}
    all_real = np.concatenate(list(scoring_real.values()), axis=0)
    all_gen = np.concatenate(list(scoring_gen.values()), axis=0)
    cross = all_gen[:200] @ all_real[:200].T
    print(f"    Cross real-gen cosine sim: mean={cross.mean():.4f}")
    gg = all_gen[:200] @ all_gen[:200].T
    np.fill_diagonal(gg, 0)
    print(f"    Gen pairwise sim: mean={gg.sum()/(200*199):.4f}")

    # Print a few per-condition KIDs
    sorted_kids = sorted([(c, delta_kids[c]) for c in common if np.isfinite(delta_kids[c])],
                         key=lambda x: x[1])
    print(f"\n  [{label}] ΔKID range: [{sorted_kids[0][1]:.6f}, {sorted_kids[-1][1]:.6f}]")
    print(f"    Best 3: {[(c, f'{v:.6f}') for c, v in sorted_kids[:3]]}")
    print(f"    Worst 3: {[(c, f'{v:.6f}') for c, v in sorted_kids[-3:]]}")

    return {
        "rho_trust": rho_trust,
        "mean_trust": mean_trust,
        "mean_realism": mean_realism,
        "mean_faith": mean_faith,
        "delta_kids": delta_kids,
        "common": common,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def main():
    print("=" * 70)
    print("OpenPhenom REPA Aligned Features: End-to-End Diagnostic")
    print("=" * 70)

    # Step 1: Data + conditions
    print("\n[1/7] Loading dataset and picking top conditions...")
    dm = RxRx1DataModule(RxRx1DataConfig(
        data_dir="/mnt/pvc/AutoSync/data/rxrx1",
        img_size=[512, 512], resize=[256, 256],
        reduce_channels=False, augment_train=False, normalize=False,
        use_numpy=True, use_parquet=False,
        batch_size=GEN_BATCH_SIZE, num_workers=4, val_size=0.1,
        seed=1337, rare_threshold=20, held_out_pairs=None,
    ))
    conditions = pick_top_conditions(dm, TOP_K_CONDITIONS)
    print(f"  Selected {len(conditions)} conditions")

    # Step 2: Model
    print("\n[2/7] Loading model...")
    pl = load_model()

    # Step 3: Real features
    print("\n[3/7] Extracting real aligned features (projector at t=0.01)...")
    real_aligned = extract_real_features(pl, dm, conditions)
    for c in list(real_aligned.keys())[:3]:
        f = real_aligned[c]
        print(f"  cond={c}: {f.shape}, norm={f.norm(dim=-1).mean():.2f}")

    # Step 4: Generate + capture features
    print(f"\n[4/7] Generating {SAMPLES_PER_CONDITION} samples per condition ({len(conditions)} conditions)...")
    gen_aligned, gen_images = generate_and_extract(pl, conditions)
    for c in list(gen_aligned.keys())[:3]:
        f = gen_aligned[c]
        print(f"  cond={c}: {f.shape}, norm={f.norm(dim=-1).mean():.2f}")

    # Collect real images for DINOv3
    print("\n[5/7] Collecting real images for DINOv3...")
    ds = dm.train_dataloader().dataset
    cond_to_idx = build_cond_index(ds, conditions)
    real_images_by_cond = {}
    for cond in tqdm(conditions, desc="Loading real images"):
        indices = cond_to_idx[cond]
        real_images_by_cond[cond] = torch.stack([ds[i][0] for i in indices])

    # Free model memory before DINOv3
    del pl
    torch.cuda.empty_cache()

    # Step 5: DINOv3 features
    print("\n[6/8] Extracting DINOv3 features...")
    real_dinov3, gen_dinov3 = extract_dinov3_features(real_images_by_cond, gen_images)

    # Step 5b: OpenPhenom post-hoc features
    print("\n[7/8] Extracting OpenPhenom post-hoc features...")
    real_openphenom, gen_openphenom = extract_openphenom_features(real_images_by_cond, gen_images)

    # Free image memory
    del real_images_by_cond, gen_images
    torch.cuda.empty_cache()

    # Step 6+7: Compute trust + correlate
    print("\n[8/8] Computing trust scores and correlations...")

    print("\n" + "=" * 70)
    print("A) ALIGNED_MEAN trust, ALIGNED_MEAN KID (same-space)")
    print("=" * 70)
    res_A = compute_trust_and_correlate(real_aligned, gen_aligned, conditions, "aligned_same")

    print("\n" + "=" * 70)
    print("B) ALIGNED_MEAN trust, DINOv3 KID (cross-space, matches pipeline)")
    print("=" * 70)
    res_B = compute_trust_and_correlate(
        real_aligned, gen_aligned, conditions, "aligned_cross",
        kid_real_by_cond=real_dinov3, kid_gen_by_cond=gen_dinov3,
    )

    print("\n" + "=" * 70)
    print("C) DINOv3 trust, DINOv3 KID (same-space, reference)")
    print("=" * 70)
    res_C = compute_trust_and_correlate(real_dinov3, gen_dinov3, conditions, "dinov3")

    print("\n" + "=" * 70)
    print("D) OpenPhenom post-hoc trust, OpenPhenom KID (same-space)")
    print("=" * 70)
    res_D = compute_trust_and_correlate(real_openphenom, gen_openphenom, conditions, "openphenom_same")

    print("\n" + "=" * 70)
    print("E) OpenPhenom post-hoc trust, DINOv3 KID (cross-space)")
    print("=" * 70)
    res_E = compute_trust_and_correlate(
        real_openphenom, gen_openphenom, conditions, "openphenom_cross",
        kid_real_by_cond=real_dinov3, kid_gen_by_cond=gen_dinov3,
    )

    # Final summary
    def _rho(res):
        return res["rho_trust"] if res else None

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  A) aligned trust + aligned KID (same-space):       {_rho(res_A)}")
    print(f"  B) aligned trust + DINOv3 KID (cross-space):       {_rho(res_B)}")
    print(f"  C) dinov3 trust + dinov3 KID (same-space):         {_rho(res_C)}")
    print(f"  D) openphenom trust + openphenom KID (same-space): {_rho(res_D)}")
    print(f"  E) openphenom trust + DINOv3 KID (cross-space):    {_rho(res_E)}")
    print(f"  Pipeline (full dataset, cross-space):")
    print(f"    aligned_mean: 0.07")
    print(f"    dinov3:       0.74")

    # ── Cross-space diagnostic ───────────────────────────────────────────
    if res_C and res_D and res_E:
        print("\n" + "=" * 70)
        print("CROSS-SPACE DIAGNOSTIC")
        print("=" * 70)

        # Find conditions common to all three results
        common_all = sorted(
            set(res_C["common"]) & set(res_D["common"]) & set(res_E["common"])
        )
        print(f"  Conditions common to C, D, E: {len(common_all)}")

        if len(common_all) >= 3:
            # 1. Trust ranking agreement: Spearman(OP trust, DINOv3 trust)
            op_trust = np.array([res_E["mean_trust"][c] for c in common_all])
            dv3_trust = np.array([res_C["mean_trust"][c] for c in common_all])
            rho_trust_agree, _ = spearmanr(op_trust, dv3_trust)
            print(f"\n  1) Trust ranking agreement:")
            print(f"     Spearman(OP trust, DINOv3 trust):           {rho_trust_agree:.4f}")

            # 2. KID ranking agreement: Spearman(OP KID, DINOv3 KID)
            op_kid = np.array([res_D["delta_kids"][c] for c in common_all])
            dv3_kid = np.array([res_C["delta_kids"][c] for c in common_all])
            rho_kid_agree, _ = spearmanr(op_kid, dv3_kid)
            print(f"\n  2) KID ranking agreement:")
            print(f"     Spearman(OP KID, DINOv3 KID):               {rho_kid_agree:.4f}")

            # 3. Component decomposition vs DINOv3 KID
            op_realism = np.array([res_E["mean_realism"][c] for c in common_all])
            op_faith = np.array([res_E["mean_faith"][c] for c in common_all])
            dv3_realism = np.array([res_C["mean_realism"][c] for c in common_all])
            dv3_faith = np.array([res_C["mean_faith"][c] for c in common_all])
            # DINOv3 KID is the ground truth for cross-space configs
            dv3_kid_arr = np.array([res_C["delta_kids"][c] for c in common_all])

            rho_op_real_kid, _ = spearmanr(op_realism, dv3_kid_arr)
            rho_op_faith_kid, _ = spearmanr(op_faith, dv3_kid_arr)
            rho_dv3_real_kid, _ = spearmanr(dv3_realism, dv3_kid_arr)
            rho_dv3_faith_kid, _ = spearmanr(dv3_faith, dv3_kid_arr)

            print(f"\n  3) Component decomposition vs DINOv3 KID:")
            print(f"     Spearman(OP realism, DINOv3 KID):           {rho_op_real_kid:.4f}")
            print(f"     Spearman(OP faithfulness, DINOv3 KID):      {rho_op_faith_kid:.4f}")
            print(f"     Spearman(DINOv3 realism, DINOv3 KID):       {rho_dv3_real_kid:.4f}  [reference]")
            print(f"     Spearman(DINOv3 faithfulness, DINOv3 KID):  {rho_dv3_faith_kid:.4f}  [reference]")
        else:
            print("  Too few common conditions for cross-space diagnostic")


if __name__ == "__main__":
    main()
