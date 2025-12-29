# import logging
# import os

# import hydra
# import numpy as np
# import pandas as pd
# import torch
# from faithful_cond_gen.eval.scoring.cosine import CosineScore
# from faithful_cond_gen.eval.scoring.knn import KNNScore

# # Import Scorers (ensure registry visibility)
# from faithful_cond_gen.eval.scoring.mahalanobis import MahalanobisScore
# from faithful_cond_gen.eval.scoring.marginal_linear_probe import (
#     MarginalLinearProbeScore,
# )
# from faithful_cond_gen.eval.scoring.relative_mahalanobis import RelativeMahalanobisScore
# from hydra.utils import instantiate
# from omegaconf import DictConfig
# from scipy import linalg
# from scipy.stats import spearmanr
# from sklearn.metrics import roc_auc_score, roc_curve
# from tqdm import tqdm

# log = logging.getLogger(__name__)


# # --- UTILS ---
# def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     """Numpy implementation of Frechet Distance (FID) on precomputed stats."""
#     mu1 = np.atleast_1d(mu1)
#     mu2 = np.atleast_1d(mu2)

#     sigma1 = np.atleast_2d(sigma1)
#     sigma2 = np.atleast_2d(sigma2)

#     assert (
#         mu1.shape == mu2.shape
#     ), "Training and test mean vectors have different lengths"
#     assert (
#         sigma1.shape == sigma2.shape
#     ), "Training and test covariances have different dimensions"

#     diff = mu1 - mu2

#     # Product might be almost singular
#     covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
#     if not np.isfinite(covmean).all():
#         log.warning("FID calculation produced infinite values, adding epsilon.")
#         offset = np.eye(sigma1.shape[0]) * eps
#         covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

#     # Numerical error might give slight complex component
#     if np.iscomplexobj(covmean):
#         if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
#             m = np.max(np.abs(covmean.imag))
#             log.warning(f"Imaginary component {m}")
#         covmean = covmean.real

#     tr_covmean = np.trace(covmean)

#     return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# def compute_statistics(features):
#     """Computes mu, sigma for FID."""
#     if isinstance(features, torch.Tensor):
#         features = features.cpu().numpy()
#     mu = np.mean(features, axis=0)
#     sigma = np.cov(features, rowvar=False)
#     return mu, sigma


# def calculate_kernel_distance(
#     features1, features2, kid_subsets=100, kid_subset_size=None
# ):
#     """Calculate KID (Kernel Inception Distance) between two feature sets.

#     Uses polynomial kernel k(x, y) = (x·y / dim + 1)^3
#     Returns the mean KID value across subsets.
#     """
#     if isinstance(features1, torch.Tensor):
#         features1 = features1.cpu().numpy()
#     if isinstance(features2, torch.Tensor):
#         features2 = features2.cpu().numpy()

#     features1 = np.asarray(features1, dtype=np.float64)
#     features2 = np.asarray(features2, dtype=np.float64)

#     n1, n2 = len(features1), len(features2)
#     dim = features1.shape[1]

#     # Set subset size to minimum of available samples or a reasonable default
#     if kid_subset_size is None:
#         kid_subset_size = min(100, n1, n2)
#     else:
#         kid_subset_size = min(kid_subset_size, n1, n2)

#     if kid_subset_size < 2:
#         return np.nan

#     m = kid_subset_size
#     kid_values = np.empty(kid_subsets, dtype=np.float64)

#     for i in range(kid_subsets):
#         idx_1 = np.random.choice(n1, m, replace=False)
#         idx_2 = np.random.choice(n2, m, replace=False)

#         X = features1[idx_1]  # (m, dim)
#         Y = features2[idx_2]  # (m, dim)

#         gram_xx = (X @ X.T / dim + 1.0) ** 3
#         gram_yy = (Y @ Y.T / dim + 1.0) ** 3
#         gram_xy = (X @ Y.T / dim + 1.0) ** 3

#         # Unbiased MMD² estimate
#         sum_xx = (gram_xx.sum() - np.trace(gram_xx)) / (m * (m - 1))
#         sum_yy = (gram_yy.sum() - np.trace(gram_yy)) / (m * (m - 1))
#         sum_xy = gram_xy.sum() / (m * m)

#         kid_values[i] = sum_xx + sum_yy - 2.0 * sum_xy

#     return kid_values.mean()


# def infer_conditioning_keys(metadata):
#     """Infer actual conditioning keys by excluding known auxiliary keys."""
#     # These are auxiliary metadata fields, not actual conditioning
#     EXCLUDED_KEYS = {"comp_category", "labels", "comp_cat", "category"}

#     all_keys = set(metadata.keys())
#     conditioning_keys = sorted(all_keys - EXCLUDED_KEYS)

#     log.info(f"Inferred conditioning keys: {conditioning_keys}")
#     if EXCLUDED_KEYS & all_keys:
#         log.info(f"Excluded auxiliary keys: {sorted(EXCLUDED_KEYS & all_keys)}")

#     return conditioning_keys


# def filter_metadata(metadata, conditioning_keys):
#     """Filter metadata to only include actual conditioning keys."""
#     return {k: metadata[k] for k in conditioning_keys if k in metadata}


# def hash_condition(cond_dict):
#     """Consistent hashing for grouping."""
#     # Convert tensors to items
#     clean = {}
#     for k, v in cond_dict.items():
#         if isinstance(v, torch.Tensor):
#             clean[k] = v.item()
#         else:
#             clean[k] = v
#     # Sort keys for determinism
#     s = sorted(clean.items())
#     return str(s)


# def resolve_path(base_dir, split_name):
#     """Finds {split_name}_features.pt in directory."""
#     path = os.path.join(base_dir, f"{split_name}_features.pt")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Missing feature cache: {path}")
#     return path


# # --- METRICS ---
# def compute_ood_metrics(id_scores, ood_scores):
#     """Computes AUROC and FPR@95%TPR assuming Higher Score = Anomaly."""
#     y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
#     y_scores = np.concatenate([id_scores, ood_scores])

#     # AUROC
#     auroc = roc_auc_score(y_true, y_scores)

#     # FPR95
#     fpr, tpr, thresholds = roc_curve(y_true, y_scores)
#     # Find FPR when TPR >= 0.95
#     # Since fpr/tpr are sorted by threshold, we look for index
#     idx = np.where(tpr >= 0.95)[0][0]
#     fpr95 = fpr[idx]

#     return auroc, fpr95


# # --- MAIN ---
# @hydra.main(
#     config_path="../../../configs", config_name="eval_scoring", version_base=None
# )
# def main(cfg: DictConfig):
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # 1. Load Data Paths
#     real_dir = cfg.real_feats_path
#     train_path = resolve_path(real_dir, "train")
#     val_path = resolve_path(real_dir, "val")
#     gen_path = cfg.gen_feats_path

#     # 2. Load Payloads
#     log.info("Loading Feature Caches...")
#     train_pl = torch.load(train_path, map_location="cpu")
#     val_pl = torch.load(val_path, map_location="cpu")
#     gen_pl = torch.load(gen_path, map_location="cpu")

#     # 3. Infer and filter conditioning keys
#     log.info("Inferring conditioning keys from metadata...")
#     conditioning_keys = infer_conditioning_keys(train_pl["metadata"])

#     train_metadata_filtered = filter_metadata(train_pl["metadata"], conditioning_keys)
#     val_metadata_filtered = filter_metadata(val_pl["metadata"], conditioning_keys)
#     gen_metadata_filtered = filter_metadata(gen_pl["metadata"], conditioning_keys)

#     # 4. Fit Scorer
#     log.info(f"Initializing & Fitting Scorer: {cfg.scorer._target_}")
#     scorer = instantiate(cfg.scorer, device=device)
#     scorer.fit(train_pl["features"], train_metadata_filtered)

#     # 5. Score Samples (Global)
#     log.info("Scoring Validation Set (ID)...")
#     val_scores = scorer.score(val_pl["features"], val_metadata_filtered).cpu().numpy()

#     log.info("Scoring Generated Set (Test)...")
#     gen_scores = scorer.score(gen_pl["features"], gen_metadata_filtered).cpu().numpy()

#     # 6. Compute Global OOD Metrics
#     # Assumption: Val is "Real/Good", Gen is "Suspect".
#     # Metric checks: Can we distinguish Gen from Real Val based on faithfulness?
#     # Note: If Gen is perfect, AUROC should be 0.5 (indistinguishable).
#     # If Gen is bad, AUROC -> 1.0.
#     auroc, fpr95 = compute_ood_metrics(val_scores, gen_scores)
#     log.info(
#         f"Global Detection Metrics (Val vs Gen): AUROC={auroc:.4f}, FPR95={fpr95:.4f}"
#     )

#     # 7. Per-Condition Analysis (Relative FID & Mean Scores)
#     log.info("Starting Per-Condition Analysis (FID & Ranking)...")

#     # We need to group indices by condition for Train, Val, and Gen
#     # Use FILTERED metadata (only conditioning keys, not comp_category etc.)
#     def group_indices(metadata):
#         groups = {}
#         N = len(next(iter(metadata.values())))
#         keys = sorted(metadata.keys())
#         for i in range(N):
#             c = {k: metadata[k][i] for k in keys}
#             h = hash_condition(c)
#             if h not in groups:
#                 groups[h] = []
#             groups[h].append(i)
#         return groups

#     train_groups = group_indices(train_metadata_filtered)
#     val_groups = group_indices(val_metadata_filtered)
#     gen_groups = group_indices(gen_metadata_filtered)

#     # Intersection of conditions present in GEN (we only care about what we generated)
#     conditions_to_eval = list(gen_groups.keys())

#     results = []

#     for cond_hash in tqdm(conditions_to_eval, desc="Analyzing Conditions"):
#         # Indices
#         idx_gen = gen_groups[cond_hash]
#         idx_val = val_groups.get(cond_hash, [])
#         idx_train = train_groups.get(cond_hash, [])

#         # 1. Mean Score (Metric)
#         score_mean = np.mean(gen_scores[idx_gen])

#         # 2. FID and KID Metrics
#         # Both require enough samples

#         fid_train_gen = np.nan
#         fid_train_val = np.nan
#         rel_fid = np.nan

#         kid_train_gen = np.nan
#         kid_train_val = np.nan
#         rel_kid = np.nan

#         # Retrieve features
#         feat_gen = gen_pl["features"][idx_gen]

#         if len(idx_train) > 5 and len(idx_val) > 5:
#             feat_train = train_pl["features"][idx_train]
#             feat_val = val_pl["features"][idx_val]

#             # --- FID Calculations ---
#             feat_train_np = feat_train.numpy()
#             feat_val_np = feat_val.numpy()
#             feat_gen_np = feat_gen.numpy()

#             # Compute Stats
#             mu_train, cov_train = compute_statistics(feat_train_np)
#             mu_val, cov_val = compute_statistics(feat_val_np)
#             mu_gen, cov_gen = compute_statistics(feat_gen_np)

#             # FID (Train vs Gen)
#             fid_train_gen = calculate_frechet_distance(
#                 mu_train, cov_train, mu_gen, cov_gen
#             )

#             # FID (Train vs Val) -> "Difficulty/Baseline"
#             fid_train_val = calculate_frechet_distance(
#                 mu_train, cov_train, mu_val, cov_val
#             )

#             # Relative FID
#             if fid_train_val > 1e-6:
#                 rel_fid = fid_train_gen / fid_train_val

#             # --- KID Calculations ---
#             # KID (Train vs Gen)
#             kid_train_gen = calculate_kernel_distance(
#                 feat_train, feat_gen, kid_subsets=100
#             )

#             # KID (Train vs Val) -> "Difficulty/Baseline"
#             kid_train_val = calculate_kernel_distance(
#                 feat_train, feat_val, kid_subsets=100
#             )

#             # Relative KID
#             if kid_train_val > 1e-9:
#                 rel_kid = kid_train_gen / kid_train_val

#         results.append(
#             {
#                 "condition_hash": cond_hash,
#                 "n_gen": len(idx_gen),
#                 "n_val": len(idx_val),
#                 "n_train": len(idx_train),
#                 "mean_score": score_mean,
#                 "fid_gen": fid_train_gen,
#                 "fid_val": fid_train_val,
#                 "relative_fid": rel_fid,
#                 "kid_gen": kid_train_gen,
#                 "kid_val": kid_train_val,
#                 "relative_kid": rel_kid,
#             }
#         )

#     df = pd.DataFrame(results)

#     # 8. Ranking Analysis
#     # We sort by Mean Score (Ascending, assuming low score = good)
#     df_sorted = df.sort_values("mean_score", ascending=True)

#     # Compute Pairwise Correlations
#     df_complete = df.dropna(subset=["relative_fid", "relative_kid"])

#     correlations = {}

#     if len(df_complete) > 10:
#         # Score vs RelFID
#         corr_score_fid, _ = spearmanr(
#             df_complete["mean_score"], df_complete["relative_fid"]
#         )
#         correlations["score_vs_rel_fid"] = corr_score_fid
#         log.info(f"Spearman Correlation (Score vs RelFID): {corr_score_fid:.4f}")

#         # Score vs RelKID
#         corr_score_kid, _ = spearmanr(
#             df_complete["mean_score"], df_complete["relative_kid"]
#         )
#         correlations["score_vs_rel_kid"] = corr_score_kid
#         log.info(f"Spearman Correlation (Score vs RelKID): {corr_score_kid:.4f}")

#         # RelFID vs RelKID
#         corr_fid_kid, _ = spearmanr(
#             df_complete["relative_fid"], df_complete["relative_kid"]
#         )
#         correlations["rel_fid_vs_rel_kid"] = corr_fid_kid
#         log.info(f"Spearman Correlation (RelFID vs RelKID): {corr_fid_kid:.4f}")
#     else:
#         correlations["score_vs_rel_fid"] = np.nan
#         correlations["score_vs_rel_kid"] = np.nan
#         correlations["rel_fid_vs_rel_kid"] = np.nan
#         log.warning(
#             f"Not enough samples for correlation analysis (need >10, got {len(df_complete)})"
#         )

#     # 9. Save
#     os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)

#     # Save CSV
#     csv_path = cfg.output_path.replace(".pt", "_analysis.csv")
#     df_sorted.to_csv(csv_path, index=False)
#     log.info(f"Analysis saved to {csv_path}")

#     # Save raw for safety
#     payload = {
#         "df": df_sorted,
#         "global_metrics": {
#             "auroc": auroc,
#             "fpr95": fpr95,
#             "correlations": correlations,
#         },
#         "scores_val": val_scores,
#         "scores_gen": gen_scores,
#     }
#     torch.save(payload, cfg.output_path)
#     log.info(f"Raw payload saved to {cfg.output_path}")

#     # 9. Print Top/Bottom Conditions
#     # Configure pandas display options to show full condition hash
#     pd.set_option("display.max_colwidth", None)
#     pd.set_option("display.width", None)

# display_cols = [
#     "condition_hash",
#     "n_gen",
#     "n_val",
#     "mean_score",
#     "relative_fid",
#     "relative_kid",
# ]

#     log.info("\n--- Top 5 Best Conditions (Lowest Score) ---")
#     print(df_sorted[display_cols].head(5))

#     log.info("\n--- Top 5 Worst Conditions (Highest Score) ---")
#     print(df_sorted[display_cols].tail(5))


# if __name__ == "__main__":
#     main()


import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from scipy import linalg
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

log = logging.getLogger(__name__)

# -------------------------
# Utils: reproducibility
# -------------------------


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism knobs (safe defaults; you can relax later for speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_int_hash(s: str) -> int:
    # Stable across runs (unlike Python's built-in hash with hash randomization)
    h = 2166136261
    for ch in s.encode("utf-8"):
        h ^= ch
        h *= 16777619
        h &= 0xFFFFFFFF
    return int(h)


# -------------------------
# Utils: metadata
# -------------------------


def infer_conditioning_keys(metadata: Dict) -> List[str]:
    EXCLUDED_KEYS = {"comp_category", "labels", "comp_cat", "category"}
    return sorted(set(metadata.keys()) - EXCLUDED_KEYS)


def filter_metadata(
    metadata: Dict, conditioning_keys: List[str]
) -> Dict[str, torch.Tensor]:
    return {k: metadata[k] for k in conditioning_keys if k in metadata}


def hash_condition(cond_dict: Dict) -> str:
    clean = {}
    for k, v in cond_dict.items():
        clean[k] = v.item() if isinstance(v, torch.Tensor) else v
    return str(sorted(clean.items()))


def resolve_path(base_dir: str, split_name: str) -> str:
    path = os.path.join(base_dir, f"{split_name}_features.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing feature cache: {path}")
    return path


# -------------------------
# OOD metrics
# -------------------------


def compute_ood_metrics(
    id_scores: np.ndarray, ood_scores: np.ndarray
) -> Tuple[float, float]:
    """
    AUROC + FPR@95%TPR.
    Convention: higher score = more anomalous.
    """
    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y_true, y_scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    mask = tpr >= 0.95
    if not np.any(mask):
        fpr95 = 1.0
    else:
        # Standard: minimal FPR achievable while keeping TPR >= 95%
        fpr95 = float(np.min(fpr[mask]))

    return float(auroc), float(fpr95)


# -------------------------
# FID utilities (kept as "legacy" reporting)
# -------------------------


def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


# -------------------------
# KID utilities (primary for per-condition quality)
# -------------------------


def calculate_kid_same_m(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Unbiased MMD^2 estimate with polynomial kernel: k(x,y)=(x·y/d + 1)^3
    Assumes X and Y have same number of samples m>=2.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"KID expects equal sample sizes, got {X.shape[0]} vs {Y.shape[0]}"
        )
    m = X.shape[0]
    if m < 2:
        return np.nan

    dim = X.shape[1]

    Kxx = (X @ X.T / dim + 1.0) ** 3
    Kyy = (Y @ Y.T / dim + 1.0) ** 3
    Kxy = (X @ Y.T / dim + 1.0) ** 3

    kxx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    kyy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    kxy = Kxy.sum() / (m * m)

    return float(kxx + kyy - 2.0 * kxy)


def bootstrap_kid_metrics(
    feats_real_all: np.ndarray,
    feats_gen_all: np.ndarray,
    seed: int,
    n_bootstrap: int,
    k_cap: int = 1000,
    min_real_pool: int = 20,
    min_k: int = 5,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Fixed-budget bootstrap over pooled reals.

    For each bootstrap:
      - split reals into Real_A (k) and Real_B (k)
      - sample Gen (k) without replacement
      - base_kid = KID(Real_A, Real_B)
      - gen_kid  = KID(Real_A, Gen)

    Returns:
      base_mean/std, gen_mean/std,
      delta_mean/std where delta = gen - base,
      rel = gen_mean / (base_mean + eps),
      z_mean/z_std where z_i = (delta_i) / (std(base) + eps),
      plus k_used and n_boot_used.
    """
    n_real = feats_real_all.shape[0]
    n_gen = feats_gen_all.shape[0]

    out = {
        "kid_base_mean": np.nan,
        "kid_base_std": np.nan,
        "kid_gen_mean": np.nan,
        "kid_gen_std": np.nan,
        "kid_delta_mean": np.nan,
        "kid_delta_std": np.nan,
        "kid_rel": np.nan,
        "kid_z_mean": np.nan,
        "kid_z_std": np.nan,
        "k_used": np.nan,
        "n_boot_used": 0.0,
    }

    if n_real < min_real_pool:
        return out

    k = min(n_real // 2, n_gen, k_cap)
    if k < min_k:
        return out

    rng = np.random.default_rng(seed)

    base_kids: List[float] = []
    gen_kids: List[float] = []
    deltas: List[float] = []

    for _ in range(n_bootstrap):
        perm = rng.permutation(n_real)
        idx_a = perm[:k]
        idx_b = perm[k : 2 * k]

        # In rare cases n_real might be exactly 2k; that's fine.
        real_a = feats_real_all[idx_a]
        real_b = feats_real_all[idx_b]

        idx_g = rng.choice(n_gen, size=k, replace=False)
        gen_samp = feats_gen_all[idx_g]

        base = calculate_kid_same_m(real_a, real_b)
        gen = calculate_kid_same_m(real_a, gen_samp)

        # KID can be NaN if something degenerate happens; skip those draws.
        if np.isfinite(base) and np.isfinite(gen):
            base_kids.append(base)
            gen_kids.append(gen)
            deltas.append(gen - base)

    if len(deltas) == 0:
        return out

    base_arr = np.asarray(base_kids, dtype=np.float64)
    gen_arr = np.asarray(gen_kids, dtype=np.float64)
    delta_arr = np.asarray(deltas, dtype=np.float64)

    base_mean = float(base_arr.mean())
    base_std = float(base_arr.std(ddof=1)) if len(base_arr) > 1 else 0.0
    gen_mean = float(gen_arr.mean())
    gen_std = float(gen_arr.std(ddof=1)) if len(gen_arr) > 1 else 0.0
    delta_mean = float(delta_arr.mean())
    delta_std = float(delta_arr.std(ddof=1)) if len(delta_arr) > 1 else 0.0

    # z-score style summary (uses std of baseline across bootstraps)
    denom = base_std + eps
    z_arr = delta_arr / denom
    z_mean = float(z_arr.mean())
    z_std = float(z_arr.std(ddof=1)) if len(z_arr) > 1 else 0.0

    out.update(
        {
            "kid_base_mean": base_mean,
            "kid_base_std": base_std,
            "kid_gen_mean": gen_mean,
            "kid_gen_std": gen_std,
            "kid_delta_mean": delta_mean,
            "kid_delta_std": delta_std,
            "kid_rel": float(gen_mean / (base_mean + eps)),
            "kid_z_mean": z_mean,
            "kid_z_std": z_std,
            "k_used": float(k),
            "n_boot_used": float(len(deltas)),
        }
    )
    return out


def bootstrap_unconditional_stratified_metrics(
    train_pl,
    val_pl,
    gen_pl,
    train_groups: Dict[str, List[int]],
    val_groups: Dict[str, List[int]],
    gen_groups: Dict[str, List[int]],
    *,
    seed: int,
    n_bootstrap: int,
    per_cond_cap: int = 250,
    min_real_pool: int = 20,
    min_k_per_cond: int = 5,
    fid_eps: float = 1e-6,
    rel_eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Unconditional (overall) metrics via stratified fixed-budget bootstrap.
    For each bootstrap:
      - For each condition c (present in gen and real pool):
          real_pool_c = train_c ∪ val_c
          k_c = min(floor(len(real_pool_c)/2), len(gen_c), per_cond_cap)
          sample 2*k_c reals -> split into realA_c, realB_c
          sample k_c gens -> gen_c
      - Concatenate across conditions:
          Real_A = concat(realA_c), Real_B = concat(realB_c), Gen = concat(gen_c)
      - Compute:
          fid_base = FID(Real_A, Real_B)
          fid_gen  = FID(Real_A, Gen)
          rel_fid  = fid_gen / (fid_base + rel_eps)
          kid_base = KID(Real_A, Real_B)
          kid_gen  = KID(Real_A, Gen)
          delta_kid = kid_gen - kid_base

    Returns mean/std across bootstraps + total sample size used.
    """
    # Precompute per-condition pools (as numpy) to avoid repeated tensor slicing overhead.
    real_pools: Dict[str, np.ndarray] = {}
    gen_pools: Dict[str, np.ndarray] = {}
    for cond, idx_gen in gen_groups.items():
        idx_train = train_groups.get(cond, [])
        idx_val = val_groups.get(cond, [])
        n_real = len(idx_train) + len(idx_val)
        if n_real < min_real_pool:
            continue
        if len(idx_gen) < min_k_per_cond:
            continue

        feats_real_list = []
        if len(idx_train) > 0:
            feats_real_list.append(train_pl["features"][idx_train])
        if len(idx_val) > 0:
            feats_real_list.append(val_pl["features"][idx_val])
        if len(feats_real_list) == 0:
            continue

        real_pools[cond] = torch.cat(feats_real_list, dim=0).numpy()
        gen_pools[cond] = gen_pl["features"][idx_gen].numpy()

    eligible = sorted(real_pools.keys())
    if len(eligible) == 0:
        return {
            "uncond_fid_base_mean": np.nan,
            "uncond_fid_base_std": np.nan,
            "uncond_fid_gen_mean": np.nan,
            "uncond_fid_gen_std": np.nan,
            "uncond_rel_fid_mean": np.nan,
            "uncond_rel_fid_std": np.nan,
            "uncond_kid_base_mean": np.nan,
            "uncond_kid_base_std": np.nan,
            "uncond_kid_gen_mean": np.nan,
            "uncond_kid_gen_std": np.nan,
            "uncond_delta_kid_mean": np.nan,
            "uncond_delta_kid_std": np.nan,
            "uncond_total_k_mean": 0.0,
            "uncond_n_conds": 0.0,
            "uncond_n_boot_used": 0.0,
        }
    rng = np.random.default_rng(seed)

    fid_base_list, fid_gen_list, rel_fid_list = [], [], []
    kid_base_list, kid_gen_list, delta_kid_list = [], [], []
    total_k_list = []

    for _ in range(n_bootstrap):
        real_a_chunks = []
        real_b_chunks = []
        gen_chunks = []
        total_k = 0

        for cond in eligible:
            real_pool = real_pools[cond]
            gen_pool = gen_pools[cond]
            n_real = real_pool.shape[0]
            n_gen = gen_pool.shape[0]

            k_c = min(n_real // 2, n_gen, per_cond_cap)
            if k_c < min_k_per_cond:
                continue

            perm = rng.permutation(n_real)
            idx_a = perm[:k_c]
            idx_b = perm[k_c : 2 * k_c]
            idx_g = rng.choice(n_gen, size=k_c, replace=False)

            real_a_chunks.append(real_pool[idx_a])
            real_b_chunks.append(real_pool[idx_b])
            gen_chunks.append(gen_pool[idx_g])
            total_k += k_c

        if total_k < 2:  # nothing usable
            continue

        Real_A = np.concatenate(real_a_chunks, axis=0)
        Real_B = np.concatenate(real_b_chunks, axis=0)
        Gen = np.concatenate(gen_chunks, axis=0)

        # KID (requires equal sizes; ensured by construction)
        kid_base = calculate_kid_same_m(Real_A, Real_B)
        kid_gen = calculate_kid_same_m(Real_A, Gen)
        if not (np.isfinite(kid_base) and np.isfinite(kid_gen)):
            continue

        # FID
        mu_a, cov_a = compute_statistics(Real_A)
        mu_b, cov_b = compute_statistics(Real_B)
        mu_g, cov_g = compute_statistics(Gen)

        fid_base = calculate_frechet_distance(mu_a, cov_a, mu_b, cov_b, eps=fid_eps)
        fid_gen = calculate_frechet_distance(mu_a, cov_a, mu_g, cov_g, eps=fid_eps)

        fid_base_list.append(fid_base)
        fid_gen_list.append(fid_gen)
        rel_fid_list.append(fid_gen / (fid_base + rel_eps))
        kid_base_list.append(kid_base)
        kid_gen_list.append(kid_gen)
        delta_kid_list.append(kid_gen - kid_base)
        total_k_list.append(total_k)

    def mean_std(xs: List[float]) -> Tuple[float, float]:
        if len(xs) == 0:
            return (np.nan, np.nan)
        arr = np.asarray(xs, dtype=np.float64)
        m = float(arr.mean())
        s = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        return (m, s)

    fid_base_mean, fid_base_std = mean_std(fid_base_list)
    fid_gen_mean, fid_gen_std = mean_std(fid_gen_list)
    rel_fid_mean, rel_fid_std = mean_std(rel_fid_list)
    kid_base_mean, kid_base_std = mean_std(kid_base_list)
    kid_gen_mean, kid_gen_std = mean_std(kid_gen_list)
    delta_kid_mean, delta_kid_std = mean_std(delta_kid_list)
    total_k_mean, _ = mean_std(total_k_list)

    return {
        "uncond_fid_base_mean": fid_base_mean,
        "uncond_fid_base_std": fid_base_std,
        "uncond_fid_gen_mean": fid_gen_mean,
        "uncond_fid_gen_std": fid_gen_std,
        "uncond_rel_fid_mean": rel_fid_mean,
        "uncond_rel_fid_std": rel_fid_std,
        "uncond_kid_base_mean": kid_base_mean,
        "uncond_kid_base_std": kid_base_std,
        "uncond_kid_gen_mean": kid_gen_mean,
        "uncond_kid_gen_std": kid_gen_std,
        "uncond_delta_kid_mean": delta_kid_mean,
        "uncond_delta_kid_std": delta_kid_std,
        "uncond_total_k_mean": (
            float(total_k_mean) if np.isfinite(total_k_mean) else 0.0
        ),
        "uncond_n_conds": float(len(eligible)),
        "uncond_n_boot_used": float(len(fid_gen_list)),
    }


# -------------------------
# Main
# -------------------------


@hydra.main(
    config_path="../../../configs", config_name="eval_scoring", version_base=None
)
def main(cfg: DictConfig):
    # Reproducibility
    seed = int(getattr(cfg, "seed", 123))
    seed_everything(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load data
    real_dir = cfg.real_feats_path
    train_path = resolve_path(real_dir, "train")
    val_path = resolve_path(real_dir, "val")
    gen_path = cfg.gen_feats_path

    log.info("Loading Caches...")
    train_pl = torch.load(train_path, map_location="cpu")
    val_pl = torch.load(val_path, map_location="cpu")
    gen_pl = torch.load(gen_path, map_location="cpu")

    # 2) Metadata filtering
    keys = infer_conditioning_keys(train_pl["metadata"])
    train_meta = filter_metadata(train_pl["metadata"], keys)
    val_meta = filter_metadata(val_pl["metadata"], keys)
    gen_meta = filter_metadata(gen_pl["metadata"], keys)

    # 3) Fit scorer (train only; no pooling for OOD thresholding)
    log.info(f"Fitting Scorer: {cfg.scorer._target_}")
    scorer = instantiate(cfg.scorer, device=device)
    scorer.fit(train_pl["features"], train_meta)

    # 4) Global OOD metrics (Val vs Gen; no pooling)
    log.info("Computing Global Scores...")
    val_scores = scorer.score(val_pl["features"], val_meta).cpu().numpy()
    gen_scores = scorer.score(gen_pl["features"], gen_meta).cpu().numpy()

    auroc, fpr95 = compute_ood_metrics(val_scores, gen_scores)
    log.info(f"Global AUROC: {auroc:.4f} | FPR95: {fpr95:.4f}")

    # 5) Per-condition analysis
    log.info("Starting Per-Condition Analysis (bootstrap KID + legacy RelFID)...")

    def group_indices(metadata: Dict[str, torch.Tensor]) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {}
        N = len(next(iter(metadata.values())))
        ksorted = sorted(metadata.keys())
        for i in range(N):
            c = {k: metadata[k][i] for k in ksorted}
            h = hash_condition(c)
            groups.setdefault(h, []).append(i)
        return groups

    train_groups = group_indices(train_meta)
    val_groups = group_indices(val_meta)
    gen_groups = group_indices(gen_meta)

    # 4b) Unconditional (overall) quality metrics via stratified bootstrap
    # Pooling train+val is intentional here (quality eval, not OOD thresholding).
    UNCOND_BOOT = int(getattr(cfg, "uncond_bootstrap", 10))
    UNCOND_PER_COND_CAP = int(
        getattr(cfg, "uncond_per_cond_cap", 250)
    )  # keep KID feasible
    UNCOND_MIN_REAL_POOL = int(getattr(cfg, "uncond_min_real_pool", 20))
    UNCOND_MIN_K = int(getattr(cfg, "uncond_min_k_per_cond", 5))

    uncond_metrics = bootstrap_unconditional_stratified_metrics(
        train_pl=train_pl,
        val_pl=val_pl,
        gen_pl=gen_pl,
        train_groups=train_groups,
        val_groups=val_groups,
        gen_groups=gen_groups,
        seed=(seed + 99991) & 0xFFFFFFFF,
        n_bootstrap=UNCOND_BOOT,
        per_cond_cap=UNCOND_PER_COND_CAP,
        min_real_pool=UNCOND_MIN_REAL_POOL,
        min_k_per_cond=UNCOND_MIN_K,
    )
    log.info(
        "Unconditional (stratified) | "
        f"FID(gen)={uncond_metrics['uncond_fid_gen_mean']:.4f}±{uncond_metrics['uncond_fid_gen_std']:.4f}, "
        f"FID(base)={uncond_metrics['uncond_fid_base_mean']:.4f}±{uncond_metrics['uncond_fid_base_std']:.4f}, "
        f"RelFID={uncond_metrics['uncond_rel_fid_mean']:.4f}±{uncond_metrics['uncond_rel_fid_std']:.4f} | "
        f"KID(gen)={uncond_metrics['uncond_kid_gen_mean']:.4f}±{uncond_metrics['uncond_kid_gen_std']:.4f}, "
        f"KID(base)={uncond_metrics['uncond_kid_base_mean']:.4f}±{uncond_metrics['uncond_kid_base_std']:.4f}, "
        f"ΔKID={uncond_metrics['uncond_delta_kid_mean']:.4f}±{uncond_metrics['uncond_delta_kid_std']:.4f} | "
        f"conds={int(uncond_metrics['uncond_n_conds'])}, total_k≈{int(uncond_metrics['uncond_total_k_mean'])}, "
        f"boot_used={int(uncond_metrics['uncond_n_boot_used'])}/{UNCOND_BOOT}"
    )

    conditions = list(gen_groups.keys())
    results = []

    # Parameters (can be moved into cfg later)
    N_BOOTSTRAP = int(getattr(cfg, "kid_bootstrap", 10))
    MIN_REAL_POOL = int(getattr(cfg, "kid_min_real_pool", 20))
    K_CAP = int(getattr(cfg, "kid_k_cap", 1000))
    MIN_K = int(getattr(cfg, "kid_min_k", 5))
    EPS = float(getattr(cfg, "kid_eps", 1e-8))

    MIN_SAMPLES_FOR_FID = int(getattr(cfg, "fid_min_samples", 8))

    for cond in tqdm(conditions, desc="Analyzing"):
        idx_gen = gen_groups[cond]
        idx_train = train_groups.get(cond, [])
        idx_val = val_groups.get(cond, [])

        n_gen = len(idx_gen)
        n_train = len(idx_train)
        n_val = len(idx_val)

        # Mean scorer value on generated samples
        score_mean = float(np.mean(gen_scores[idx_gen])) if n_gen > 0 else np.nan

        # --- A) Bootstrap KID metrics on pooled reals (train+val) ---
        # Pooling here is only for estimating "real baseline variability" under the condition.
        kid_metrics = {
            "kid_base_mean": np.nan,
            "kid_base_std": np.nan,
            "kid_gen_mean": np.nan,
            "kid_gen_std": np.nan,
            "kid_delta_mean": np.nan,
            "kid_delta_std": np.nan,
            "kid_rel": np.nan,
            "kid_z_mean": np.nan,
            "kid_z_std": np.nan,
            "k_used": np.nan,
            "n_boot_used": 0.0,
        }

        if n_gen > 0:
            feats_real_list = []
            if n_train > 0:
                feats_real_list.append(train_pl["features"][idx_train])
            if n_val > 0:
                feats_real_list.append(val_pl["features"][idx_val])

            if len(feats_real_list) > 0:
                feats_real_all = torch.cat(feats_real_list, dim=0).numpy()
                feats_gen_all = gen_pl["features"][idx_gen].numpy()

                # Per-condition deterministic seed so results don’t depend on iteration order
                cond_seed = (seed + stable_int_hash(cond)) & 0xFFFFFFFF

                kid_metrics = bootstrap_kid_metrics(
                    feats_real_all=feats_real_all,
                    feats_gen_all=feats_gen_all,
                    seed=cond_seed,
                    n_bootstrap=N_BOOTSTRAP,
                    k_cap=K_CAP,
                    min_real_pool=MIN_REAL_POOL,
                    min_k=MIN_K,
                    eps=EPS,
                )

        # --- B) Legacy RelFID (still useful to log, not recommended for ranking in high-d/small-n) ---
        rel_fid = np.nan
        fid_gen = np.nan
        fid_val = np.nan

        if (
            n_train >= MIN_SAMPLES_FOR_FID
            and n_val >= MIN_SAMPLES_FOR_FID
            and n_gen >= MIN_SAMPLES_FOR_FID
        ):
            ft_train = train_pl["features"][idx_train].numpy()
            ft_val = val_pl["features"][idx_val].numpy()
            ft_gen = gen_pl["features"][idx_gen].numpy()

            mu_t, cov_t = compute_statistics(ft_train)
            mu_v, cov_v = compute_statistics(ft_val)
            mu_g, cov_g = compute_statistics(ft_gen)

            fid_gen = calculate_frechet_distance(mu_t, cov_t, mu_g, cov_g)
            fid_val = calculate_frechet_distance(mu_t, cov_t, mu_v, cov_v)
            if fid_val > 1e-6:
                rel_fid = float(fid_gen / fid_val)

        results.append(
            {
                "condition_hash": cond,
                "n_gen": n_gen,
                "n_train": n_train,
                "n_val": n_val,
                "n_real_pool": n_train + n_val,
                "mean_score": score_mean,
                # KID bootstrap summaries
                **kid_metrics,
                # Legacy FID reporting
                "fid_gen": fid_gen,
                "fid_val": fid_val,
                "relative_fid": rel_fid,
            }
        )

    df = pd.DataFrame(results)

    # 6) Pairwise correlations between all metrics you care about
    # (Spearman; pandas does pairwise complete observations)
    corr_cols = [
        "mean_score",
        "relative_fid",
        "kid_base_mean",
        "kid_base_std",
        "kid_gen_mean",
        "kid_gen_std",
        "kid_delta_mean",
        "kid_delta_std",
        "kid_rel",
        "kid_z_mean",
        "kid_z_std",
        "k_used",
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr_df = df[corr_cols].corr(method="spearman")

    # Log a few key ones explicitly (plus full matrix saved)
    def log_corr(a: str, b: str) -> None:
        sub = df[[a, b]].dropna()
        if len(sub) < 6:
            log.info(f"Spearman({a} vs {b}): NaN (n={len(sub)})")
            return
        r, _ = spearmanr(sub[a].values, sub[b].values)
        log.info(f"Spearman({a} vs {b}): {float(r):.4f} (n={len(sub)})")

    # log_corr("mean_score", "kid_delta_mean")
    # log_corr("mean_score", "kid_z_mean")
    # log_corr("mean_score", "kid_rel")
    # log_corr("mean_score", "relative_fid")

    log_corr("mean_score", "kid_gen_mean")
    log_corr("mean_score", "kid_delta_mean")
    log_corr("mean_score", "kid_base_std")
    log_corr("mean_score", "k_used")
    log_corr("kid_gen_mean", "kid_delta_mean")
    log_corr("kid_base_std", "kid_delta_mean")
    log_corr("kid_base_std", "k_used")
    log_corr("relative_fid", "kid_delta_mean")
    log_corr("mean_score", "n_real_pool")
    log_corr("kid_base_std", "n_real_pool")
    log_corr("k_used", "n_real_pool")

    # 7) Save
    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)

    csv_path = cfg.output_path.replace(".pt", "_analysis.csv")
    df_sorted = df.sort_values("mean_score", ascending=True)
    df_sorted.to_csv(csv_path, index=False)
    log.info(f"Saved analysis to {csv_path}")

    corr_csv_path = cfg.output_path.replace(".pt", "_corr_spearman.csv")
    corr_df.to_csv(corr_csv_path, index=True)
    log.info(f"Saved Spearman correlation matrix to {corr_csv_path}")

    payload = {
        "df": df_sorted,
        "global_metrics": {"auroc": auroc, "fpr95": fpr95},
        "spearman_corr": corr_df,
        "seed": seed,
        "params": {
            "kid_bootstrap": N_BOOTSTRAP,
            "kid_min_real_pool": MIN_REAL_POOL,
            "kid_k_cap": K_CAP,
            "kid_min_k": MIN_K,
            "kid_eps": EPS,
            "fid_min_samples": MIN_SAMPLES_FOR_FID,
        },
    }
    torch.save(payload, cfg.output_path)
    log.info(f"Saved raw payload to {cfg.output_path}")

    # 8) Display
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 1400)
    pd.set_option("display.max_columns", None)

    display_cols = [
        "condition_hash",
        "n_gen",
        "n_train",
        "n_val",
        "n_real_pool",
        "k_used",
        "n_boot_used",
        "mean_score",
        "kid_base_mean",
        "kid_base_std",
        "kid_gen_mean",
        "kid_gen_std",
        "kid_delta_mean",
        "kid_delta_std",
        "kid_rel",
        "kid_z_mean",
        "kid_z_std",
        "relative_fid",
    ]
    display_cols = [c for c in display_cols if c in df_sorted.columns]

    log.info("\n--- Top 5 Best (Lowest Score) ---")
    print(df_sorted[display_cols].head(5))

    log.info("\n--- Top 5 Worst (Highest Score) ---")
    print(df_sorted[display_cols].tail(5))


if __name__ == "__main__":
    main()
