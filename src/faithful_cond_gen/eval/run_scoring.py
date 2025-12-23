import logging
import os

import hydra
import numpy as np
import pandas as pd
import torch
from faithful_cond_gen.eval.scoring.cosine import CosineScore
from faithful_cond_gen.eval.scoring.knn import KNNScore

# Import Scorers (ensure registry visibility)
from faithful_cond_gen.eval.scoring.mahalanobis import MahalanobisScore
from faithful_cond_gen.eval.scoring.marginal_linear_probe import (
    MarginalLinearProbeScore,
)
from faithful_cond_gen.eval.scoring.relative_mahalanobis import RelativeMahalanobisScore
from hydra.utils import instantiate
from omegaconf import DictConfig
from scipy import linalg
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

log = logging.getLogger(__name__)

# --- UTILS ---


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of Frechet Distance (FID) on precomputed stats."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        log.warning("FID calculation produced infinite values, adding epsilon.")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight complex component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            log.warning(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def compute_statistics(features):
    """Computes mu, sigma for FID."""
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def infer_conditioning_keys(metadata):
    """Infer actual conditioning keys by excluding known auxiliary keys."""
    # These are auxiliary metadata fields, not actual conditioning
    EXCLUDED_KEYS = {"comp_category", "labels", "comp_cat", "category"}
    
    all_keys = set(metadata.keys())
    conditioning_keys = sorted(all_keys - EXCLUDED_KEYS)
    
    log.info(f"Inferred conditioning keys: {conditioning_keys}")
    if EXCLUDED_KEYS & all_keys:
        log.info(f"Excluded auxiliary keys: {sorted(EXCLUDED_KEYS & all_keys)}")
    
    return conditioning_keys


def filter_metadata(metadata, conditioning_keys):
    """Filter metadata to only include actual conditioning keys."""
    return {k: metadata[k] for k in conditioning_keys if k in metadata}


def hash_condition(cond_dict):
    """Consistent hashing for grouping."""
    # Convert tensors to items
    clean = {}
    for k, v in cond_dict.items():
        if isinstance(v, torch.Tensor):
            clean[k] = v.item()
        else:
            clean[k] = v
    # Sort keys for determinism
    s = sorted(clean.items())
    return str(s)


def resolve_path(base_dir, split_name):
    """Finds {split_name}_features.pt in directory."""
    path = os.path.join(base_dir, f"{split_name}_features.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing feature cache: {path}")
    return path


# --- METRICS ---


def compute_ood_metrics(id_scores, ood_scores):
    """Computes AUROC and FPR@95%TPR assuming Higher Score = Anomaly."""
    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_scores = np.concatenate([id_scores, ood_scores])

    # AUROC
    auroc = roc_auc_score(y_true, y_scores)

    # FPR95
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Find FPR when TPR >= 0.95
    # Since fpr/tpr are sorted by threshold, we look for index
    idx = np.where(tpr >= 0.95)[0][0]
    fpr95 = fpr[idx]

    return auroc, fpr95


# --- MAIN ---


@hydra.main(config_path="../../configs", config_name="eval_scoring", version_base=None)
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Data Paths
    real_dir = cfg.real_feats_path
    train_path = resolve_path(real_dir, "train")
    val_path = resolve_path(real_dir, "val")
    gen_path = cfg.gen_feats_path

    # 2. Load Payloads
    log.info("Loading Feature Caches...")
    train_pl = torch.load(train_path, map_location="cpu")
    val_pl = torch.load(val_path, map_location="cpu")
    gen_pl = torch.load(gen_path, map_location="cpu")

    # 3. Infer and filter conditioning keys
    log.info("Inferring conditioning keys from metadata...")
    conditioning_keys = infer_conditioning_keys(train_pl["metadata"])
    
    train_metadata_filtered = filter_metadata(train_pl["metadata"], conditioning_keys)
    val_metadata_filtered = filter_metadata(val_pl["metadata"], conditioning_keys)
    gen_metadata_filtered = filter_metadata(gen_pl["metadata"], conditioning_keys)
    
    # 4. Fit Scorer
    log.info(f"Initializing & Fitting Scorer: {cfg.scorer._target_}")
    scorer = instantiate(cfg.scorer, device=device)
    scorer.fit(train_pl["features"], train_metadata_filtered)

    # 5. Score Samples (Global)
    log.info("Scoring Validation Set (ID)...")
    val_scores = scorer.score(val_pl["features"], val_metadata_filtered).cpu().numpy()

    log.info("Scoring Generated Set (Test)...")
    gen_scores = scorer.score(gen_pl["features"], gen_metadata_filtered).cpu().numpy()

    # 6. Compute Global OOD Metrics
    # Assumption: Val is "Real/Good", Gen is "Suspect".
    # Metric checks: Can we distinguish Gen from Real Val based on faithfulness?
    # Note: If Gen is perfect, AUROC should be 0.5 (indistinguishable).
    # If Gen is bad, AUROC -> 1.0.
    auroc, fpr95 = compute_ood_metrics(val_scores, gen_scores)
    log.info(
        f"Global Detection Metrics (Val vs Gen): AUROC={auroc:.4f}, FPR95={fpr95:.4f}"
    )

    # 7. Per-Condition Analysis (Relative FID & Mean Scores)
    log.info("Starting Per-Condition Analysis (FID & Ranking)...")

    # We need to group indices by condition for Train, Val, and Gen
    # Use FILTERED metadata (only conditioning keys, not comp_category etc.)
    def group_indices(metadata):
        groups = {}
        N = len(next(iter(metadata.values())))
        keys = sorted(metadata.keys())
        for i in range(N):
            c = {k: metadata[k][i] for k in keys}
            h = hash_condition(c)
            if h not in groups:
                groups[h] = []
            groups[h].append(i)
        return groups

    train_groups = group_indices(train_metadata_filtered)
    val_groups = group_indices(val_metadata_filtered)
    gen_groups = group_indices(gen_metadata_filtered)

    # Intersection of conditions present in GEN (we only care about what we generated)
    conditions_to_eval = list(gen_groups.keys())

    results = []

    for cond_hash in tqdm(conditions_to_eval, desc="Analyzing Conditions"):
        # Indices
        idx_gen = gen_groups[cond_hash]
        idx_val = val_groups.get(cond_hash, [])
        idx_train = train_groups.get(cond_hash, [])

        # 1. Mean Score (Metric)
        score_mean = np.mean(gen_scores[idx_gen])

        # 2. FID Metrics
        # Only possible if we have enough samples in Train/Val for covariance
        # Minimum samples for covariance: D > N usually, but typically need at least 2 samples
        # to not crash. Ideally N > 50.

        fid_train_gen = np.nan
        fid_train_val = np.nan
        rel_fid = np.nan

        # Retrieve features
        feat_gen = gen_pl["features"][idx_gen].numpy()

        if len(idx_train) > 5 and len(idx_val) > 5:
            feat_train = train_pl["features"][idx_train].numpy()
            feat_val = val_pl["features"][idx_val].numpy()

            # Compute Stats
            mu_train, cov_train = compute_statistics(feat_train)
            mu_val, cov_val = compute_statistics(feat_val)
            mu_gen, cov_gen = compute_statistics(feat_gen)

            # FID (Train vs Gen)
            fid_train_gen = calculate_frechet_distance(
                mu_train, cov_train, mu_gen, cov_gen
            )

            # FID (Train vs Val) -> "Difficulty/Baseline"
            fid_train_val = calculate_frechet_distance(
                mu_train, cov_train, mu_val, cov_val
            )

            # Relative FID
            # Ratio < 1.0 means Generator is better than Validation set (Unlikely but possible)
            # Ratio near 1.0 means Generator is as good as Validation (Perfect)
            # Ratio >> 1.0 means Generator is bad
            if fid_train_val > 1e-6:
                rel_fid = fid_train_gen / fid_train_val

        results.append(
            {
                "condition_hash": cond_hash,
                "n_gen": len(idx_gen),
                "n_val": len(idx_val),
                "n_train": len(idx_train),
                "mean_score": score_mean,
                "fid_gen": fid_train_gen,
                "fid_val": fid_train_val,
                "relative_fid": rel_fid,
            }
        )

    df = pd.DataFrame(results)

    # 8. Ranking Analysis
    # We sort by Mean Score (Ascending, assuming low score = good)
    df_sorted = df.sort_values("mean_score", ascending=True)

    # Compute Correlation if FID is available
    df_fid = df.dropna(subset=["relative_fid"])
    spearman = np.nan
    if len(df_fid) > 10:
        # Does the cheap score correlate with the expensive FID?
        corr, _ = spearmanr(df_fid["mean_score"], df_fid["relative_fid"])
        spearman = corr
        log.info(f"Spearman Correlation (Score vs RelFID): {spearman:.4f}")

    # 9. Save
    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)

    # Save CSV
    csv_path = cfg.output_path.replace(".pt", "_analysis.csv")
    df_sorted.to_csv(csv_path, index=False)
    log.info(f"Analysis saved to {csv_path}")

    # Save raw for safety
    payload = {
        "df": df_sorted,
        "global_metrics": {"auroc": auroc, "fpr95": fpr95, "spearman": spearman},
        "scores_val": val_scores,
        "scores_gen": gen_scores,
    }
    torch.save(payload, cfg.output_path)
    log.info(f"Raw payload saved to {cfg.output_path}")

    # 9. Print Top/Bottom Conditions
    log.info("\n--- Top 5 Best Conditions (Lowest Score) ---")
    print(df_sorted[["condition_hash", "mean_score", "relative_fid"]].head(5))

    log.info("\n--- Top 5 Worst Conditions (Highest Score) ---")
    print(df_sorted[["condition_hash", "mean_score", "relative_fid"]].tail(5))


if __name__ == "__main__":
    main()
