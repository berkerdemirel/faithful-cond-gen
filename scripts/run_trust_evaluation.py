"""
Trust Score Evaluation: Extended Analysis with Decile Binning.

Implements the evaluation layers from the research plan:
1. Condition-level ranking validity (T̄ vs KID correlation)
2. Failure detection (PR/AUROC for predicting bad conditions)
   - 2A: Condition-level OOD (seen vs unseen conditions)
   - 2B: Sample-level OOD (seen vs unseen samples, marginal models)
3. Real vs Generated OOD detection (sample-level)
4. Decile binning analysis
5. Correlation with Alaa et al. metrics
6. Multi-backbone aggregation

Usage:
    uv run python scripts/run_trust_evaluation.py --dataset celeba
    uv run python scripts/run_trust_evaluation.py --dataset celeba --normalize-features l2

Flags:
    --normalize-features {none,l2}  Apply L2 normalization to all features after loading.
                                    Default: none (backward-compatible).
"""

import argparse
import hashlib
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from faithful_cond_gen.eval.trust_eval_extensions import (
    bootstrap_kid_for_bin,
    compute_real_sample_scores,
    compute_trust_results_from_features,
    condition_to_signature,
    create_image_grid,
    fit_trust_scoring_components,
    get_image_path,
    score_trust_from_components,
)

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path("outputs/trust_evaluation")

# Feature configurations - use meanpatch features for consistent KID comparison
# dinov3 uses REPAEncoder meanpatch to match aligned_mean feature space
FEATURE_CONFIGS = {
    # DINOv3 meanpatch features (consistent with REPA training)
    ("celeba", "vanilla_full", "dinov3"): (
        "celeba_vanilla_full",
        "dinov3_meanpatch_features.pt",
    ),
    ("celeba", "vanilla_marginal", "dinov3"): (
        "celeba_vanilla_marginal",
        "dinov3_meanpatch_features.pt",
    ),
    ("celeba", "repa_full", "dinov3"): (
        "celeba_repa_full",
        "dinov3_meanpatch_features.pt",
    ),
    ("celeba", "repa_marginal", "dinov3"): (
        "celeba_repa_marginal",
        "dinov3_meanpatch_features.pt",
    ),
    # Aligned mean features (from REPA training, now with correct ordering)
    ("celeba", "repa_full", "aligned_mean"): (
        "celeba_repa_full",
        "aligned_mean_features.pt",
    ),
    ("celeba", "repa_marginal", "aligned_mean"): (
        "celeba_repa_marginal",
        "aligned_mean_features.pt",
    ),
}

# Real feature paths - use meanpatch for dinov3 comparisons
REAL_FEATURE_PATHS = {
    ("celeba", "dinov3"): "outputs/real_celeba_dinov3_meanpatch/train_features.pt",
    (
        "celeba",
        "aligned_mean",
    ): "outputs/real_celeba_dinov3_meanpatch/train_features.pt",  # Same real features
}

# NOTE: For consistent KID computation, all features should use the same extraction method:
# - dinov3: REPAEncoder(dinov3-vit-l) mean-pooled patch tokens (not eval CLS/pooler)
# - aligned_mean: REPA aligned features mean-pooled (same encoder, extracted during generation)
# This ensures both use cosine-similar representations for fair comparison.

CONDITION_ATTRS = {
    "celeba": ["Male", "Smiling", "Blond_Hair", "Eyeglasses"],
    "rxrx1": ["cell_type_id", "sirna_id"],
}

# Marginal model seen combos (for CelebA)
MARGINAL_SEEN_COMBOS = {
    (0, 0, 0, 0),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
}


# ============================================================================
# Feature Normalization and Loading Utilities
# ============================================================================


def l2_normalize_features(features: torch.Tensor) -> torch.Tensor:
    """
    Apply L2 normalization to feature vectors.

    Args:
        features: Feature tensor of shape (N, D)

    Returns:
        L2-normalized features with unit norm per row
    """
    norms = features.norm(dim=1, keepdim=True)
    return features / (norms + 1e-12)


def apply_normalization(
    features: torch.Tensor, normalize_mode: str, feature_name: str = "features"
) -> torch.Tensor:
    """
    Apply normalization to features based on mode.

    Args:
        features: Feature tensor of shape (N, D)
        normalize_mode: One of "none" or "l2"
        feature_name: Name for logging purposes

    Returns:
        Normalized (or unchanged) features
    """
    if normalize_mode == "l2":
        logger.info(f"  Applying L2 normalization to {feature_name} ({features.shape})")
        return l2_normalize_features(features)
    return features


def get_filenames_from_meta(meta: Dict) -> Optional[List[str]]:
    """
    Extract filename list from metadata dictionary.

    Checks common keys: filenames, file_names, paths, image_paths, img_paths.

    Returns:
        List of filename strings, or None if not found
    """
    for key in ["filenames", "file_names", "paths", "image_paths", "img_paths"]:
        if key in meta:
            v = meta[key]
            if isinstance(v, torch.Tensor):
                v = v.tolist()
            return [str(x) for x in v]
    return None


def verify_feature_ordering(meta1: Dict, meta2: Dict, name1: str, name2: str) -> bool:
    """
    Verify that two feature caches have matching sample ordering.

    Checks filename metadata if available. Raises ValueError if mismatch detected.
    Emits warning if metadata not available.

    Args:
        meta1, meta2: Metadata dictionaries from feature caches
        name1, name2: Names for error messages

    Returns:
        True if verified matching, False if could not verify (no metadata)

    Raises:
        ValueError: If filenames exist but don't match
    """
    names1 = get_filenames_from_meta(meta1) if isinstance(meta1, dict) else None
    names2 = get_filenames_from_meta(meta2) if isinstance(meta2, dict) else None

    if names1 is not None and names2 is not None:
        if len(names1) != len(names2):
            raise ValueError(
                f"Feature ordering mismatch: {name1} has {len(names1)} samples, "
                f"{name2} has {len(names2)} samples. Cannot safely compare."
            )
        # Check first 100 samples for efficiency
        mismatches = [
            (i, n1, n2)
            for i, (n1, n2) in enumerate(zip(names1[:100], names2[:100]))
            if n1 != n2
        ]
        if mismatches:
            first_mismatch = mismatches[0]
            raise ValueError(
                f"Feature ordering mismatch between {name1} and {name2}. "
                f"First mismatch at index {first_mismatch[0]}: "
                f"'{first_mismatch[1]}' vs '{first_mismatch[2]}'. "
                f"Regenerate caches with consistent ordering."
            )
        logger.info(f"  Verified: {name1} and {name2} have matching sample order")
        return True
    else:
        missing = []
        if names1 is None:
            missing.append(name1)
        if names2 is None:
            missing.append(name2)
        warnings.warn(
            f"Cannot verify feature ordering: {', '.join(missing)} missing filename metadata. "
            f"Assuming same ordering. Consider regenerating caches with filenames.",
            UserWarning,
        )
        return False


# ============================================================================
# Helper functions
# ============================================================================


def get_effective_kid_mode(normalize_mode: str, feature_type: str) -> str:
    """
    Determine the effective KID computation mode.

    Logic:
    - If features are L2-normalized (normalize_mode="l2"), use cosine KID everywhere
      because cosine similarity on L2-normalized vectors = dot product
    - If no normalization (normalize_mode="none"), use feature-type-based logic:
      - aligned_mean: cosine (norms weren't optimized during training)
      - dinov3: standard (use raw feature magnitudes)

    Args:
        normalize_mode: Feature normalization mode ("none" or "l2")
        feature_type: Type of features ("dinov3" or "aligned_mean")

    Returns:
        KID mode string: "cosine" or "standard"
    """
    if normalize_mode == "l2":
        # L2-normalized features should use cosine KID
        return "cosine"
    else:
        # Use feature-type-based logic
        if feature_type == "aligned_mean":
            return "cosine"  # Norms weren't optimized, normalize internally
        else:
            return "standard"  # Use raw features with dimension scaling


# def load_trust_scores(dataset: str) -> List[Dict]:
#     """
#     Load precomputed trust scores from disk.

#     Args:
#         dataset: Dataset name (e.g., "celeba", "rxrx1")

#     Returns:
#         List of trust score result dictionaries

#     Raises:
#         FileNotFoundError: If trust scores file doesn't exist
#     """
#     path = TRUST_SCORES_DIR / f"trust_scores_{dataset}.pt"
#     if not path.exists():
#         raise FileNotFoundError(
#             f"Trust scores not found at {path}. Run compute_trust_scores.py first."
#         )
#     return torch.load(path, map_location="cpu", weights_only=False)


def get_per_condition_stats(results: Dict) -> pd.DataFrame:
    """
    Aggregate per-sample scores into per-condition statistics.

    Args:
        results: Trust results dict with keys: true_conditions, trust_updated,
                 realism_global_z, faithfulness_margin_z, authenticity, gen_center_dist_real

    Returns:
        DataFrame with one row per condition containing mean/std of scores
    """
    conditions = results["true_conditions"]
    trust = results["trust_updated"]
    realism = results["realism_global_z"]
    faithfulness = results["faithfulness_margin_z"]
    auth = results["authenticity"]
    gen_center_dist = results["gen_center_dist_real"]

    # Group by condition
    data = {}
    for i, cond in enumerate(conditions):
        if cond not in data:
            data[cond] = {
                "trust": [],
                "realism": [],
                "faithfulness": [],
                "auth": [],
                "gen_center_dist": [],
            }
        data[cond]["trust"].append(trust[i])
        data[cond]["realism"].append(realism[i])
        data[cond]["faithfulness"].append(faithfulness[i])
        data[cond]["auth"].append(auth[i])
        data[cond]["gen_center_dist"].append(gen_center_dist[i])

    rows = []
    for cond, vals in data.items():
        rows.append(
            {
                "condition": cond,
                "n_samples": len(vals["trust"]),
                "trust_mean": np.mean(vals["trust"]),
                "trust_std": np.std(vals["trust"]),
                "realism_mean": np.mean(vals["realism"]),
                "faithfulness_mean": np.mean(vals["faithfulness"]),
                "faithfulness_std": np.std(vals["faithfulness"]),
                "auth_mean": np.mean(vals["auth"]),
                "gen_center_dist_mean": np.mean(vals["gen_center_dist"]),
            }
        )

    return pd.DataFrame(rows)


def calculate_kid_same_m(
    X: np.ndarray, Y: np.ndarray, use_cosine: bool = False
) -> float:
    """
    Unbiased MMD^2 estimate with polynomial kernel.

    Args:
        X, Y: Feature arrays (same size)
        use_cosine: If True, L2-normalize features and use k(x,y)=(x·y + 1)^3
                   If False, use standard k(x,y)=(x·y/d + 1)^3
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

    if use_cosine:
        # L2-normalize features for cosine-based kernel
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
        # Cosine kernel: k(x,y) = (x·y + 1)^3
        Kxx = (X @ X.T + 1.0) ** 3
        Kyy = (Y @ Y.T + 1.0) ** 3
        Kxy = (X @ Y.T + 1.0) ** 3
    else:
        # Standard kernel: k(x,y) = (x·y/d + 1)^3
        dim = X.shape[1]
        Kxx = (X @ X.T / dim + 1.0) ** 3
        Kyy = (Y @ Y.T / dim + 1.0) ** 3
        Kxy = (X @ Y.T / dim + 1.0) ** 3

    kxx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    kyy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    kxy = Kxy.sum() / (m * m)

    return float(kxx + kyy - 2.0 * kxy)


# ============================================================================
# Ranking Metrics: Top-K Overlap and Stratified Correlations
# ============================================================================


def topk_overlap(ranking1: np.ndarray, ranking2: np.ndarray, k: int = 10) -> float:
    """
    Compute Jaccard similarity of top-k items in two rankings.

    Args:
        ranking1, ranking2: Arrays of condition indices sorted by score (best first)
        k: Number of top items to compare

    Returns:
        Jaccard similarity in [0, 1]
    """
    k = min(k, len(ranking1), len(ranking2))
    if k == 0:
        return np.nan

    # Convert to hashable types (tuples) for set operations
    top1 = set(
        tuple(x) if hasattr(x, "__iter__") and not isinstance(x, str) else x
        for x in ranking1[:k]
    )
    top2 = set(
        tuple(x) if hasattr(x, "__iter__") and not isinstance(x, str) else x
        for x in ranking2[:k]
    )

    intersection = len(top1 & top2)
    union = len(top1 | top2)

    return intersection / union if union > 0 else 0.0


def stratified_correlation(
    scores: np.ndarray,
    kid_gt: np.ndarray,
    support_sizes: np.ndarray,
    bins: List[float] = None,
) -> Dict[str, Dict]:
    """
    Compute ranking correlation stratified by support size.

    This helps control for the confound where conditions with larger support
    may have more reliable KID estimates.

    Args:
        scores: Per-condition trust scores
        kid_gt: Per-condition ground-truth delta KID
        support_sizes: Number of real samples per condition
        bins: Bin edges for support sizes (default: [0, 20, 50, 100, inf])

    Returns:
        Dict with correlation results per bin
    """
    if bins is None:
        bins = [0, 20, 50, 100, np.inf]

    results = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (support_sizes >= lo) & (support_sizes < hi)
        n_in_bin = mask.sum()

        bin_name = f"{int(lo)}-{int(hi) if hi != np.inf else 'inf'}"

        if n_in_bin >= 3:
            valid = ~np.isnan(scores[mask]) & ~np.isnan(kid_gt[mask])
            if valid.sum() >= 3:
                rho, p = spearmanr(scores[mask][valid], kid_gt[mask][valid])
                tau, tau_p = kendalltau(scores[mask][valid], kid_gt[mask][valid])
                results[bin_name] = {
                    "n": int(valid.sum()),
                    "spearman_rho": rho,
                    "spearman_p": p,
                    "kendall_tau": tau,
                    "kendall_p": tau_p,
                }
            else:
                results[bin_name] = {
                    "n": int(n_in_bin),
                    "spearman_rho": np.nan,
                    "kendall_tau": np.nan,
                }
        else:
            results[bin_name] = {
                "n": int(n_in_bin),
                "spearman_rho": np.nan,
                "kendall_tau": np.nan,
            }

    return results


# ============================================================================
# Layer 1: Condition-Level Ranking Validity
# ============================================================================


def evaluate_ranking_validity(
    trust_results: Dict,
    real_feats: torch.Tensor,
    real_meta: Dict,
    gen_feats: torch.Tensor,
    condition_keys: List[str],
    feature_type: str = "dinov3",
    n_bootstrap: int = 10,
    kid_mode: str = "auto",  # "auto" | "standard" | "cosine"
) -> Dict:
    """
    Layer 1: Does trust-based ranking agree with ground-truth quality ranking?

    Computes Spearman/Kendall correlation between per-condition mean trust
    and per-condition delta KID.

    Args:
        trust_results: Dict with trust scores and conditions
        real_feats: Real features tensor
        real_meta: Real metadata dict
        gen_feats: Generated features tensor
        condition_keys: List of condition attribute names
        feature_type: Type of features ("dinov3" or "aligned_mean")
        n_bootstrap: Number of bootstrap iterations for KID
        kid_mode: KID computation mode:
            - "auto": use_cosine = (feature_type == "aligned_mean") [current behavior]
            - "standard": use_cosine = False always
            - "cosine": use_cosine = True always
    """
    # Determine effective KID mode
    if kid_mode == "auto":
        effective_cosine = feature_type == "aligned_mean"
    elif kid_mode == "cosine":
        effective_cosine = True
    else:  # "standard"
        effective_cosine = False
    print(
        f"    KID mode: {kid_mode} -> effective_cosine={effective_cosine} (feature_type={feature_type})"
    )
    # Get per-condition trust stats
    cond_stats = get_per_condition_stats(trust_results)

    # Compute per-condition delta KID
    conditions = trust_results["true_conditions"]
    trust_scores = trust_results["trust_updated"]

    # Group features by condition
    gen_by_cond = {}
    for i, cond in enumerate(conditions):
        if cond not in gen_by_cond:
            gen_by_cond[cond] = []
        gen_by_cond[cond].append(i)

    # Get real features grouped by condition
    real_by_cond = {}
    n_real = len(real_feats)
    for i in range(n_real):
        cond = tuple(
            int(
                real_meta[k][i].item()
                if isinstance(real_meta[k][i], torch.Tensor)
                else real_meta[k][i]
            )
            for k in condition_keys
        )
        if cond not in real_by_cond:
            real_by_cond[cond] = []
        real_by_cond[cond].append(i)

    # Compute delta KID per condition
    delta_kids = {}
    base_kids = {}
    gen_kids = {}
    for cond in gen_by_cond:
        gen_idx = gen_by_cond[cond]
        real_idx = real_by_cond.get(cond, [])

        if len(real_idx) < 20 or len(gen_idx) < 5:
            delta_kids[cond] = np.nan
            continue

        gen_f = gen_feats[gen_idx].numpy()
        real_f = real_feats[real_idx].numpy()

        # Bootstrap delta KID
        k = min(len(real_idx) // 2, len(gen_idx), 500)
        if k < 5:
            delta_kids[cond] = np.nan
            continue

        # Use stable hash (md5) instead of Python's hash() which varies per process
        stable_hash = int(hashlib.md5(str(cond).encode()).hexdigest(), 16) % 1000
        rng = np.random.default_rng(42 + stable_hash)
        deltas = []
        bases = []
        gens = []
        for _ in range(n_bootstrap):
            perm = rng.permutation(len(real_idx))
            idx_a, idx_b = perm[:k], perm[k : 2 * k]
            real_a, real_b = real_f[idx_a], real_f[idx_b]
            gen_samp = gen_f[rng.choice(len(gen_f), k, replace=False)]

            base = calculate_kid_same_m(real_a, real_b, use_cosine=effective_cosine)
            gen_kid = calculate_kid_same_m(
                real_a, gen_samp, use_cosine=effective_cosine
            )
            if np.isfinite(base) and np.isfinite(gen_kid):
                deltas.append(gen_kid - base)
                bases.append(base)
                gens.append(gen_kid)

        delta_kids[cond] = np.mean(deltas) if deltas else np.nan
        base_kids[cond] = np.mean(bases) if bases else np.nan
        gen_kids[cond] = np.mean(gens) if gens else np.nan

    # Merge with trust stats
    cond_stats["delta_kid"] = cond_stats["condition"].apply(
        lambda c: delta_kids.get(c, np.nan)
    )
    # Add real support count (for stratified correlation)
    cond_stats["n_real"] = cond_stats["condition"].apply(
        lambda c: len(real_by_cond.get(c, []))
    )
    # cond_stats["base_kid"] = cond_stats["condition"].apply(
    #     lambda c: base_kids.get(c, np.nan)
    # )
    # cond_stats["gen_kid"] = cond_stats["condition"].apply(
    #     lambda c: gen_kids.get(c, np.nan)
    # )
    cond_stats["kid_real_real"] = cond_stats["condition"].apply(
        lambda c: base_kids.get(c, np.nan)
    )
    cond_stats["kid_real_gen"] = cond_stats["condition"].apply(
        lambda c: gen_kids.get(c, np.nan)
    )
    cond_stats["base_kid"] = cond_stats["kid_real_real"]
    cond_stats["gen_kid"] = cond_stats["kid_real_gen"]

    # Compute correlations (only for conditions with valid KID)
    valid = cond_stats[cond_stats["delta_kid"].notna()]
    if len(valid) < 3:
        return {
            "spearman_rho": np.nan,
            "kendall_tau": np.nan,
            "n_conditions": len(valid),
        }

    # Trust score correlation (lower trust = better, lower delta_kid = better)
    spearman_rho, spearman_p = spearmanr(valid["trust_mean"], valid["delta_kid"])
    kendall_tau, kendall_p = kendalltau(valid["trust_mean"], valid["delta_kid"])

    result = {
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "kendall_tau": kendall_tau,
        "kendall_p": kendall_p,
        "n_conditions": len(valid),
        "cond_stats": cond_stats,
    }

    # Also compute correlation for realism and faithfulness components separately
    rho_realism, _ = spearmanr(valid["realism_mean"], valid["delta_kid"])
    rho_faithfulness, _ = spearmanr(valid["faithfulness_mean"], valid["delta_kid"])
    result["spearman_rho_realism"] = rho_realism
    result["spearman_rho_faithfulness"] = rho_faithfulness

    # Top-k overlap: do trust rankings match KID rankings?
    # Sort by trust (lower = better trust), sort by delta_kid (lower = better quality)
    trust_ranking = valid.sort_values("trust_mean")["condition"].tolist()
    kid_ranking = valid.sort_values("delta_kid")["condition"].tolist()

    for k in [5, 10, 20]:
        if len(trust_ranking) >= k:
            result[f"topk_overlap_{k}"] = topk_overlap(
                np.array(trust_ranking), np.array(kid_ranking), k
            )

    # Stratified correlation by real support size (not gen samples)
    scores_arr = valid["trust_mean"].values
    kid_arr = valid["delta_kid"].values
    real_support_arr = valid["n_real"].values
    result["stratified_correlation"] = stratified_correlation(
        scores_arr, kid_arr, real_support_arr
    )

    return result


# ============================================================================
# Layer 2: Failure Detection (PR/AUROC)
# ============================================================================


def evaluate_seen_vs_unseen_detection(
    trust_results: Dict,
    dataset: str,
    model: str,
    output_dir: Path,
    config_key: str,
) -> Dict:
    """
    Sample-level ROC/AUROC between seen vs unseen conditions (marginal models only).

    Labels: seen=0, unseen=1 based on MARGINAL_SEEN_COMBOS.
    """
    if dataset != "celeba" or "marginal" not in model:
        return {"status": "not_applicable"}

    conditions = trust_results["true_conditions"]
    trust_scores = trust_results["trust_updated"]
    realism_scores = trust_results["realism_global_z"]
    faithfulness_scores = trust_results["faithfulness_margin_z"]

    # Build labels: seen=0, unseen=1
    labels = np.array([0 if c in MARGINAL_SEEN_COMBOS else 1 for c in conditions])

    n_seen = (labels == 0).sum()
    n_unseen = (labels == 1).sum()

    if n_seen == 0 or n_unseen == 0:
        return {
            "status": "insufficient_samples",
            "n_seen": n_seen,
            "n_unseen": n_unseen,
        }

    results = {
        "n_seen": int(n_seen),
        "n_unseen": int(n_unseen),
    }

    # Score variants
    score_variants = {
        "trust": (trust_scores, "Trust Score"),
        "realism": (realism_scores, "Realism Only"),
        "faithfulness": (faithfulness_scores, "Faithfulness Only"),
    }

    # Create ROC plot
    fig, ax = plt.subplots(figsize=(8, 6))

    for score_name, (scores, label) in score_variants.items():
        valid = np.isfinite(scores)
        if valid.sum() < 10:
            continue

        scores_v = scores[valid]
        labels_v = labels[valid]

        # AUROC
        auroc = roc_auc_score(labels_v, scores_v)

        # AUPRC
        precision, recall, _ = precision_recall_curve(labels_v, scores_v)
        auprc = auc(recall, precision)

        results[f"{score_name}_auroc"] = float(auroc)
        results[f"{score_name}_auprc"] = float(auprc)

        # ROC curve
        fpr, tpr, _ = roc_curve(labels_v, scores_v)
        ax.plot(fpr, tpr, label=f"{label} (AUROC={auroc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC: Seen vs Unseen Conditions - {config_key}")
    ax.legend()
    ax.grid(alpha=0.3)

    roc_path = output_dir / f"seen_unseen_roc_{config_key.replace('/', '_')}.png"
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    results["roc_path"] = str(roc_path)
    results["status"] = "success"

    return results


def evaluate_failure_detection(
    marginal_results: Dict,
    dataset: str,
) -> Dict:
    """
    Layer 2: Can trust predict which conditions fail?

    For marginal model, label conditions as "bad" if they are unseen (OOD).
    Evaluate if trust score can detect these at the CONDITION level (not sample level).
    """
    if dataset != "celeba":
        # For RxRx1, need different logic for holdout conditions
        return {"auroc": np.nan, "auprc": np.nan}

    # Aggregate trust scores per condition
    conditions = marginal_results["true_conditions"]
    trust_scores = marginal_results["trust_updated"]

    # Group by condition and compute mean trust per condition
    trust_by_cond = {}
    for i, cond in enumerate(conditions):
        if cond not in trust_by_cond:
            trust_by_cond[cond] = []
        trust_by_cond[cond].append(trust_scores[i])

    # Build condition-level labels and scores
    cond_labels = []
    cond_scores = []
    for cond, scores_list in trust_by_cond.items():
        if cond in MARGINAL_SEEN_COMBOS:
            cond_labels.append(0)  # seen
        else:
            cond_labels.append(1)  # unseen (OOD)
        cond_scores.append(np.mean(scores_list))

    cond_labels = np.array(cond_labels)
    cond_scores = np.array(cond_scores)

    n_seen = (cond_labels == 0).sum()
    n_unseen = (cond_labels == 1).sum()

    if n_seen == 0 or n_unseen == 0:
        return {
            "auroc": np.nan,
            "auprc": np.nan,
            "n_seen_conds": n_seen,
            "n_unseen_conds": n_unseen,
        }

    # AUROC: higher score = more likely OOD
    auroc = roc_auc_score(cond_labels, cond_scores)

    # AUPRC
    precision, recall, _ = precision_recall_curve(cond_labels, cond_scores)
    auprc = auc(recall, precision)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "n_seen_conds": int(n_seen),
        "n_unseen_conds": int(n_unseen),
        "n_total_conds": len(trust_by_cond),
    }


# ============================================================================
# Layer 4: Correlation with Alaa et al. Metrics
# ============================================================================


def evaluate_alaa_correlation(trust_results: Dict) -> Dict:
    """
    Layer 4: Does our score align with established sample-level metrics?

    Compute correlations between trust, realism, faithfulness and Alaa et al. metrics.
    """
    trust = trust_results["trust_updated"]
    realism = trust_results["realism_global_z"]
    faithfulness = trust_results["faithfulness_margin_z"]
    gen_center_dist = trust_results["gen_center_dist_real"]
    auth = trust_results["authenticity"]

    # Filter valid samples
    valid = np.isfinite(trust) & np.isfinite(realism) & np.isfinite(faithfulness)
    trust_v = trust[valid]
    realism_v = realism[valid]
    faithfulness_v = faithfulness[valid]
    gen_center_dist_v = gen_center_dist[valid]
    auth_v = auth[valid]

    results = {}

    # Realism vs gen_center_dist (should correlate - both capture support)
    if len(trust_v) > 10:
        rho, p = spearmanr(realism_v, gen_center_dist_v)
        results["realism_vs_gen_center_dist"] = {"spearman_rho": rho, "p": p}

    # Faithfulness vs gen_center_dist (should be weaker)
    if len(trust_v) > 10:
        rho, p = spearmanr(faithfulness_v, gen_center_dist_v)
        results["faithfulness_vs_gen_center_dist"] = {"spearman_rho": rho, "p": p}

    # Trust vs gen_center_dist
    if len(trust_v) > 10:
        rho, p = spearmanr(trust_v, gen_center_dist_v)
        results["trust_vs_gen_center_dist"] = {"spearman_rho": rho, "p": p}

    # Trust vs authenticity
    if len(trust_v) > 10:
        rho, p = spearmanr(trust_v, auth_v)
        results["trust_vs_authenticity"] = {"spearman_rho": rho, "p": p}

    return results


# ============================================================================
# Layer 5: Multi-Backbone Aggregation
# ============================================================================


def evaluate_multi_backbone(all_results: List[Dict], model: str) -> Dict:
    """
    Layer 5: Multi-feature-type analysis.

    Reports per-feature-type aggregate statistics only.
    NOTE: Sample-level correlations across feature types removed because there's
    no guarantee samples are in the same order across different feature caches.
    """
    # Filter to single model
    model_results = [r for r in all_results if r["model"] == model]
    if len(model_results) < 1:
        return {"n_feature_types": 0}

    feature_types = [
        r.get("feature_type", r.get("encoder", "unknown")) for r in model_results
    ]

    # Per-feature-type aggregate statistics
    feature_stats = []
    for r in model_results:
        trust = r["trust_updated"]
        realism = r["realism_global_z"]
        faithfulness = r["faithfulness_margin_z"]

        feature_stats.append(
            {
                "feature_type": r.get("feature_type", r.get("encoder", "unknown")),
                "n_samples": r["n_samples"],
                "trust_mean": float(np.mean(trust)),
                "trust_std": float(np.std(trust)),
                "realism_mean": float(np.mean(realism)),
                "realism_std": float(np.std(realism)),
                "faithfulness_mean": float(np.mean(faithfulness)),
                "faithfulness_std": float(np.std(faithfulness)),
                "authenticity_mean": float(np.mean(r["authenticity"])),
            }
        )

    result = {
        "n_feature_types": len(feature_types),
        "feature_types": feature_types,
        "feature_stats": pd.DataFrame(feature_stats),
    }

    return result


# ============================================================================
# Extension Tasks
# ============================================================================


def evaluate_full_condition_ranking(
    trust_results: Dict,
    ranking_results: Dict,
    dataset: str,
    condition_keys: List[str],
    output_dir: Path,
    config_key: str,
) -> Dict:
    """
    Task 1: Full condition ranking + gap analysis.

    - Export full ranking table with all conditions
    - Add hamming weight, seen/unseen labels
    - Compute Pearson correlation (in addition to Spearman/Kendall)
    - Gap analysis: consecutive delta KID differences
    - Generate DataFrame for inline table in report (NO scatter plots)
    """
    if "cond_stats" not in ranking_results:
        return {"status": "no_cond_stats"}

    cond_stats = ranking_results["cond_stats"].copy()

    # Add condition_str (human-readable format)
    def format_condition(cond):
        return ", ".join(f"{k}={v}" for k, v in zip(condition_keys, cond))

    cond_stats["condition_str"] = cond_stats["condition"].apply(format_condition)

    # Add seen/unseen label
    if dataset == "celeba":
        cond_stats["seen_unseen"] = cond_stats["condition"].apply(
            lambda c: "seen" if c in MARGINAL_SEEN_COMBOS else "unseen"
        )
    else:
        cond_stats["seen_unseen"] = "unknown"

    # Add hamming weight (number of 1s in condition)
    cond_stats["hamming_weight"] = cond_stats["condition"].apply(lambda c: sum(c))

    # Filter to valid conditions
    valid = cond_stats[cond_stats["delta_kid"].notna()].copy()
    if len(valid) >= 2:
        # Gap analysis: sort by delta_kid and compute consecutive differences
        sorted_by_kid = valid.sort_values("delta_kid")
        gaps = sorted_by_kid["delta_kid"].diff().dropna()
        mean_gap = gaps.mean()
        max_gap = gaps.max()
        gap_analysis = {"mean_gap": float(mean_gap), "max_gap": float(max_gap)}

        # Pearson correlation
        pearson_rho, pearson_p = pearsonr(valid["trust_mean"], valid["delta_kid"])

        cols = ["condition_str"]
        # include these if available (they will be for your ranking_results after the patch above)
        for c in ["kid_real_real", "kid_real_gen", "delta_kid"]:
            if c in valid.columns:
                cols.append(c)
        cols += ["trust_mean", "realism_mean", "faithfulness_mean"]
        table_df = valid.sort_values("delta_kid")[cols].copy()

        # Save CSV with full details
        csv_path = output_dir / f"full_ranking_{config_key.replace('/', '_')}.csv"
        cols_to_save = [
            "condition_str",
            "seen_unseen",
            "hamming_weight",
            "n_samples",
            "n_real",
            "trust_mean",
            "trust_std",
            "realism_mean",
            "faithfulness_mean",
            "kid_real_real",
            "kid_real_gen",
            "delta_kid",
        ]
        cols_to_save = [c for c in cols_to_save if c in cond_stats.columns]
        cond_stats[cols_to_save].to_csv(csv_path, index=False)

        return {
            "status": "success",
            "pearson_rho": float(pearson_rho),
            "pearson_p": float(pearson_p),
            "gap_analysis": gap_analysis,
            "csv_path": str(csv_path),
            "n_conditions": len(valid),
            "table_df": table_df,  # For inline markdown table
        }
    else:
        return {"status": "insufficient_conditions", "n_conditions": len(valid)}


def create_realism_faithfulness_grids(
    trust_results: Dict,
    model_dir: str,
    dataset: str,
    condition_keys: List[str],
    output_dir: Path,
    config_key: str,
    seed: int = 42,
) -> Dict:
    """
    Task 2: Create 2×2 grids per condition (realism × faithfulness).

    Uses robust corner picking to ensure grids are generated for all conditions.
    Fixes indexing bug by mapping global index to within-condition index.
    """
    # Sanity check: verify image directory exists
    image_dir = Path(f"outputs/gen/{model_dir}/images")
    if image_dir.exists():
        sample_files = list(image_dir.glob("*.png"))[:3]
        if sample_files:
            print(
                f"    Image dir check: {image_dir} exists, sample files: {[f.name for f in sample_files]}"
            )
    else:
        print(f"    Warning: Image directory not found: {image_dir}")

    conditions = trust_results["true_conditions"]
    realism_z = trust_results["realism_global_z"]
    faithfulness_z = trust_results["faithfulness_margin_z"]

    # Build global_idx -> within_condition_idx mapping
    global_to_local_idx = {}
    cond_counters = {}
    for global_idx, cond in enumerate(conditions):
        if cond not in cond_counters:
            cond_counters[cond] = 0
        local_idx = cond_counters[cond]
        global_to_local_idx[global_idx] = (cond, local_idx)
        cond_counters[cond] += 1

    # Group by condition with local indices
    samples_by_cond = {}
    for global_idx, cond in enumerate(conditions):
        if cond not in samples_by_cond:
            samples_by_cond[cond] = []
        _, local_idx = global_to_local_idx[global_idx]
        samples_by_cond[cond].append(
            {
                "global_idx": global_idx,
                "local_idx": local_idx,
                "realism_z": realism_z[global_idx],
                "faithfulness_z": faithfulness_z[global_idx],
            }
        )

    # Create grids directory
    grid_dir = output_dir / "grids" / config_key.replace("/", "_")
    grid_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    manifest = []
    grids_created = 0

    # Select conditions (all 16 in CelebA, top 16 by count otherwise)
    if dataset == "celeba" and len(samples_by_cond) == 16:
        selected_conditions = list(samples_by_cond.keys())
    else:
        cond_counts = [
            (cond, len(samples)) for cond, samples in samples_by_cond.items()
        ]
        cond_counts.sort(key=lambda x: x[1], reverse=True)
        selected_conditions = [c for c, _ in cond_counts[:16]]

    for cond in selected_conditions:
        samples = samples_by_cond[cond]
        if len(samples) < 4:
            continue  # Need at least 4 samples

        # Robust corner picking: greedy selection without duplicates
        r_values = np.array([s["realism_z"] for s in samples])
        f_values = np.array([s["faithfulness_z"] for s in samples])

        # Track which samples are already used
        used_indices = set()

        def pick_best(score_fn):
            """Pick best sample by score_fn, excluding already used."""
            scores = score_fn(r_values, f_values)
            sorted_idx = np.argsort(scores)
            for idx in sorted_idx:
                if idx not in used_indices:
                    used_indices.add(idx)
                    return samples[idx]
            # Fallback: return best even if used (shouldn't happen with 4+ samples)
            return samples[sorted_idx[0]]

        corners = {
            # lowest r + lowest f (good realism, good faithfulness)
            "good_r_good_f": pick_best(lambda r, f: r + f),
            # lowest r + highest f (good realism, bad faithfulness)
            "good_r_bad_f": pick_best(lambda r, f: r - f),
            # highest r + lowest f (bad realism, good faithfulness)
            "bad_r_good_f": pick_best(lambda r, f: -r + f),
            # highest r + highest f (bad realism, bad faithfulness)
            "bad_r_bad_f": pick_best(lambda r, f: -r - f),
        }

        # Build grid
        image_paths = []
        titles = []
        scores = []
        selected_info = {}

        quad_order = [
            ("good_r_good_f", "Good R + Good F"),
            ("good_r_bad_f", "Good R + Bad F"),
            ("bad_r_good_f", "Bad R + Good F"),
            ("bad_r_bad_f", "Bad R + Bad F"),
        ]

        for quad_name, title in quad_order:
            sample = corners[quad_name]
            local_idx = sample["local_idx"]
            r = sample["realism_z"]
            f = sample["faithfulness_z"]

            # Get image path using local index
            img_path = get_image_path(cond, local_idx, model_dir, condition_keys)
            image_paths.append(img_path)
            titles.append(title)
            scores.append((r, f))

            selected_info[quad_name] = {
                "local_idx": local_idx,
                "realism_z": float(r),
                "faithfulness_z": float(f),
                "filename": img_path.name,
            }

        # Create grid
        try:
            grid_img = create_image_grid(image_paths, titles, scores)
            cond_str = "_".join(f"{k}{v}" for k, v in zip(condition_keys, cond))
            grid_path = grid_dir / f"condition_{cond_str}.png"
            grid_img.save(grid_path)

            manifest.append(
                {
                    "condition": str(cond),
                    "condition_str": cond_str,
                    "grid_path": str(grid_path.relative_to(output_dir)),
                    "good_r_good_f": str(selected_info["good_r_good_f"]),
                    "good_r_bad_f": str(selected_info["good_r_bad_f"]),
                    "bad_r_good_f": str(selected_info["bad_r_good_f"]),
                    "bad_r_bad_f": str(selected_info["bad_r_bad_f"]),
                }
            )
            grids_created += 1
        except Exception as e:
            print(f"  Warning: Failed to create grid for {cond}: {e}")

    # Save manifest
    if manifest:
        manifest_df = pd.DataFrame(manifest)
        manifest_path = grid_dir / "manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
    else:
        manifest_path = None

    return {
        "status": "success" if grids_created > 0 else "no_grids_created",
        "n_grids": grids_created,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "grid_dir": str(grid_dir),
    }


def stratified_subsample_real(
    real_feats: torch.Tensor,
    real_meta: Dict,
    condition_keys: List[str],
    max_samples: int,
    seed: int,
) -> np.ndarray:
    """
    Stratified subsample of real samples by condition.

    Returns indices into real_feats that maintain proportional representation
    of each condition.

    Args:
        real_feats: Real features tensor [N, D]
        real_meta: Metadata dict with condition keys
        condition_keys: List of condition attribute names
        max_samples: Maximum number of samples to return
        seed: Random seed for reproducibility

    Returns:
        np.ndarray of indices into real_feats
    """
    n_total = len(real_feats)
    if n_total <= max_samples:
        return np.arange(n_total)

    # Build condition tuples for each sample
    conditions = []
    for i in range(n_total):
        cond = tuple(
            int(
                real_meta[k][i].item()
                if isinstance(real_meta[k][i], torch.Tensor)
                else real_meta[k][i]
            )
            for k in condition_keys
        )
        conditions.append(cond)

    # Group indices by condition
    cond_to_indices = {}
    for i, cond in enumerate(conditions):
        if cond not in cond_to_indices:
            cond_to_indices[cond] = []
        cond_to_indices[cond].append(i)

    # Proportional allocation
    rng = np.random.default_rng(seed)
    selected_indices = []

    for cond, indices in cond_to_indices.items():
        # Allocate proportionally, at least 1 if non-empty
        n_cond = len(indices)
        n_alloc = max(1, int(np.round(n_cond / n_total * max_samples)))
        n_alloc = min(n_alloc, n_cond)  # Don't oversample

        # Sample from this stratum
        sampled = rng.choice(indices, size=n_alloc, replace=False)
        selected_indices.extend(sampled)

    # If we overshot due to rounding, trim
    selected_indices = np.array(selected_indices)
    if len(selected_indices) > max_samples:
        selected_indices = rng.choice(selected_indices, size=max_samples, replace=False)

    return selected_indices


def evaluate_sample_ood_detection(
    trust_results: Dict,
    real_feats: torch.Tensor,
    real_meta: Dict,
    gen_feats: torch.Tensor,
    gen_meta: Dict,
    condition_keys: List[str],
    dataset: str,
    model: str,
    output_dir: Path,
    config_key: str,
    max_real: int = 10000,
    n_resamples: int = 3,
    fit_fraction: float = 0.5,
) -> Dict:
    """
    Task 3: Sample-based OOD detection (real vs generated).

    Uses cross-fitting to avoid calibration bias:
    - Split real samples into fit_set and score_set
    - Fit scoring components on fit_set only
    - Score both score_set (held-out real) and gen samples using fit_set components

    This ensures fair comparison because neither real nor gen samples
    were used to fit the scoring Gaussians.

    Args:
        trust_results: Dict with trust scores and conditions (unused, kept for API compat)
        real_feats: Real features tensor
        real_meta: Real metadata dict
        gen_feats: Generated features tensor
        gen_meta: Generated metadata dict (required for re-scoring gen samples)
        condition_keys: List of condition attribute names
        dataset: Dataset name
        model: Model name
        output_dir: Output directory for plots
        config_key: Configuration key for naming files
        max_real: Maximum real samples to use (stratified subsample if exceeded)
        n_resamples: Number of resamples for CI computation
        fit_fraction: Fraction of real samples used for fitting (rest for scoring)
    """
    # Check if resampling is needed
    n_total_real = len(real_feats)
    do_resample = n_total_real > max_real

    results = {}

    filter_by_seen = "marginal" in model and dataset == "celeba"
    seen_combos = MARGINAL_SEEN_COMBOS if filter_by_seen else None

    # Define splits for gen samples (seen vs unseen for marginal models)
    splits = []
    if filter_by_seen:
        # Need to compute gen conditions for split
        n_gen = len(gen_feats)
        gen_conditions = []
        for i in range(n_gen):
            cond = tuple(
                int(gen_meta[k][i].item() if isinstance(gen_meta[k][i], torch.Tensor) else gen_meta[k][i])
                for k in condition_keys
            )
            gen_conditions.append(cond)
        gen_seen_mask = np.array([c in MARGINAL_SEEN_COMBOS for c in gen_conditions])
        splits.append(("seen", gen_seen_mask))
        splits.append(("unseen", ~gen_seen_mask))
    else:
        splits.append(("all", np.ones(len(gen_feats), dtype=bool)))

    for split_name, gen_mask in splits:
        gen_feats_split = gen_feats[gen_mask]
        # Build gen_meta_split
        gen_meta_split = {}
        for k in condition_keys:
            if isinstance(gen_meta[k], torch.Tensor):
                gen_meta_split[k] = gen_meta[k][gen_mask]
            elif isinstance(gen_meta[k], np.ndarray):
                gen_meta_split[k] = gen_meta[k][gen_mask]
            else:
                gen_meta_split[k] = [gen_meta[k][i] for i, m in enumerate(gen_mask) if m]

        if len(gen_feats_split) == 0:
            continue

        n_gen = len(gen_feats_split)

        # Storage for metrics across resamples
        resample_metrics = {
            "trust": {"auroc": [], "auprc": [], "fpr": [], "tpr": []},
            "realism": {"auroc": [], "auprc": [], "fpr": [], "tpr": []},
            "faithfulness": {"auroc": [], "auprc": [], "fpr": [], "tpr": []},
        }

        actual_resamples = n_resamples if do_resample else 1

        for resample_idx in range(actual_resamples):
            seed = 42 + resample_idx
            rng = np.random.default_rng(seed)

            # Get real sample indices (stratified subsample if needed)
            if do_resample:
                real_indices = stratified_subsample_real(
                    real_feats, real_meta, condition_keys, max_real, seed
                )
            else:
                real_indices = np.arange(n_total_real)

            # Split into fit_set and score_set (cross-fitting)
            n_subset = len(real_indices)
            n_fit = int(n_subset * fit_fraction)
            perm = rng.permutation(n_subset)
            fit_idx = real_indices[perm[:n_fit]]
            score_idx = real_indices[perm[n_fit:]]

            # Build fit metadata
            fit_meta = {}
            for k in condition_keys:
                if isinstance(real_meta[k], torch.Tensor):
                    fit_meta[k] = real_meta[k][fit_idx]
                elif isinstance(real_meta[k], np.ndarray):
                    fit_meta[k] = real_meta[k][fit_idx]
                else:
                    fit_meta[k] = [real_meta[k][i] for i in fit_idx]
            fit_feats = real_feats[fit_idx]

            # Build score metadata (held-out real samples)
            score_meta = {}
            for k in condition_keys:
                if isinstance(real_meta[k], torch.Tensor):
                    score_meta[k] = real_meta[k][score_idx]
                elif isinstance(real_meta[k], np.ndarray):
                    score_meta[k] = real_meta[k][score_idx]
                else:
                    score_meta[k] = [real_meta[k][i] for i in score_idx]
            score_feats = real_feats[score_idx]

            # Filter fit set by seen combos if needed (marginal models)
            from faithful_cond_gen.eval.trust_eval_extensions import (
                _filter_feats_and_meta_by_seen_combos,
            )
            if filter_by_seen and seen_combos:
                fit_feats_filtered, fit_meta_filtered = _filter_feats_and_meta_by_seen_combos(
                    fit_feats, fit_meta, condition_keys, seen_combos
                )
            else:
                fit_feats_filtered, fit_meta_filtered = fit_feats, fit_meta

            # Fit scoring components on fit_set only (LDA-style shared cov for fair margin comparison)
            components = fit_trust_scoring_components(
                fit_feats_filtered, fit_meta_filtered, condition_keys,
                regularization=1e-5, use_shared_cov=True,
            )

            # Score held-out real samples using fitted components
            real_realism_z, real_faithfulness_z, real_trust_z = score_trust_from_components(
                score_feats, score_meta, components
            )

            # Score generated samples using the SAME fitted components
            gen_realism_z, gen_faithfulness_z, gen_trust_z = score_trust_from_components(
                gen_feats_split, gen_meta_split, components
            )

            n_real = len(real_trust_z)

            # Create binary labels: real=0 (ID), gen=1 (OOD)
            labels = np.concatenate([np.zeros(n_real), np.ones(n_gen)])

            # Score variants
            score_variants = {
                "trust": (
                    np.concatenate([real_trust_z, gen_trust_z]),
                    "Trust Score",
                ),
                "realism": (
                    np.concatenate([real_realism_z, gen_realism_z]),
                    "Realism Only",
                ),
                "faithfulness": (
                    np.concatenate([real_faithfulness_z, gen_faithfulness_z]),
                    "Faithfulness Only",
                ),
            }

            for score_name, (scores, label) in score_variants.items():
                # Filter valid scores
                valid = np.isfinite(scores)
                if valid.sum() < 10:
                    continue

                scores_v = scores[valid]
                labels_v = labels[valid]

                # AUROC
                auroc = roc_auc_score(labels_v, scores_v)

                # AUPRC
                precision, recall, _ = precision_recall_curve(labels_v, scores_v)
                auprc = auc(recall, precision)

                # ROC curve
                fpr, tpr, _ = roc_curve(labels_v, scores_v)

                resample_metrics[score_name]["auroc"].append(auroc)
                resample_metrics[score_name]["auprc"].append(auprc)
                resample_metrics[score_name]["fpr"].append(fpr)
                resample_metrics[score_name]["tpr"].append(tpr)

        # Aggregate results
        split_results = {
            "n_real_total": int(n_total_real),
            "n_real_used": int(min(n_total_real, max_real)),
            "n_gen": int(n_gen),
            "n_resamples": actual_resamples,
        }

        # Create ROC plot with all resamples
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

        for score_name in ["trust", "realism", "faithfulness"]:
            metrics = resample_metrics[score_name]
            if not metrics["auroc"]:
                continue

            aurocs = np.array(metrics["auroc"])
            auprcs = np.array(metrics["auprc"])

            # Mean and CI
            mean_auroc = np.mean(aurocs)
            mean_auprc = np.mean(auprcs)

            if len(aurocs) > 1:
                ci_low_auroc, ci_high_auroc = np.percentile(aurocs, [2.5, 97.5])
                ci_low_auprc, ci_high_auprc = np.percentile(auprcs, [2.5, 97.5])
            else:
                ci_low_auroc = ci_high_auroc = mean_auroc
                ci_low_auprc = ci_high_auprc = mean_auprc

            # Store aggregated metrics
            split_results[f"{score_name}_mean_auroc"] = float(mean_auroc)
            split_results[f"{score_name}_ci_low_auroc"] = float(ci_low_auroc)
            split_results[f"{score_name}_ci_high_auroc"] = float(ci_high_auroc)
            split_results[f"{score_name}_mean_auprc"] = float(mean_auprc)
            split_results[f"{score_name}_ci_low_auprc"] = float(ci_low_auprc)
            split_results[f"{score_name}_ci_high_auprc"] = float(ci_high_auprc)

            # Also store as simple auroc/auprc for backward compatibility
            split_results[f"{score_name}_auroc"] = float(mean_auroc)
            split_results[f"{score_name}_auprc"] = float(mean_auprc)

            # Plot each resample curve with alpha=0.3
            label_map = {
                "trust": "Trust Score",
                "realism": "Realism Only",
                "faithfulness": "Faithfulness Only",
            }
            for i, (fpr, tpr) in enumerate(zip(metrics["fpr"], metrics["tpr"])):
                if i == 0:
                    # First curve gets the label with mean±CI annotation
                    if len(aurocs) > 1:
                        label_str = f"{label_map[score_name]} (AUROC={mean_auroc:.4f} [{ci_low_auroc:.4f}, {ci_high_auroc:.4f}])"
                    else:
                        label_str = f"{label_map[score_name]} (AUROC={mean_auroc:.4f})"
                    ax_roc.plot(
                        fpr, tpr, label=label_str, alpha=0.3 if do_resample else 1.0
                    )
                else:
                    ax_roc.plot(
                        fpr, tpr, alpha=0.3, color=ax_roc.get_lines()[-1].get_color()
                    )

        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        title_suffix = f" ({actual_resamples} resamples)" if do_resample else ""
        ax_roc.set_title(
            f"ROC Curve: Real vs Gen({split_name}) - {config_key}{title_suffix}"
        )
        ax_roc.legend(loc="lower right", fontsize=8)
        ax_roc.grid(alpha=0.3)

        roc_path = (
            output_dir / f"ood_roc_{config_key.replace('/', '_')}_{split_name}.png"
        )
        fig_roc.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close(fig_roc)

        split_results["roc_path"] = str(roc_path)

        # Backward compatibility: n_real
        split_results["n_real"] = split_results["n_real_used"]

        results[split_name] = split_results

    return results


def evaluate_decile_binning(
    trust_results: Dict,
    real_feats: torch.Tensor,
    real_meta: Dict,
    gen_feats: torch.Tensor,
    condition_keys: List[str],
    feature_type: str,
    output_dir: Path,
    config_key: str,
    n_bins: int = 10,
    n_bootstrap: int = 10,
    kid_mode: str = "auto",  # "auto" | "standard" | "cosine"
) -> pd.DataFrame:
    """
    Task 4: Decile binning with ablations.

    For each ranking mode (trust, realism, faithfulness):
    - Sort samples by score (best→worst)
    - Split into n_bins equal-sized bins
    - For each bin:
        - Compute unconditional KID vs real
        - Bootstrap for confidence intervals
    - Return DataFrame with all results
    - Plot line chart with error bars

    Args:
        trust_results: Dict with trust scores and conditions
        real_feats: Real features tensor
        real_meta: Real metadata dict
        gen_feats: Generated features tensor
        condition_keys: List of condition attribute names
        feature_type: Type of features ("dinov3" or "aligned_mean")
        output_dir: Output directory for plots/csvs
        config_key: Configuration key for naming files
        n_bins: Number of decile bins
        n_bootstrap: Number of bootstrap iterations
        kid_mode: KID computation mode:
            - "auto": use_cosine = (feature_type == "aligned_mean") [current behavior]
            - "standard": use_cosine = False always
            - "cosine": use_cosine = True always
    """
    # Determine effective KID mode
    if kid_mode == "auto":
        effective_cosine = feature_type == "aligned_mean"
    elif kid_mode == "cosine":
        effective_cosine = True
    else:  # "standard"
        effective_cosine = False
    print(
        f"    KID mode: {kid_mode} -> effective_cosine={effective_cosine} (feature_type={feature_type})"
    )

    # Get scores
    trust_scores = trust_results["trust_updated"]
    realism_scores = trust_results["realism_global_z"]
    faithfulness_scores = trust_results["faithfulness_margin_z"]

    ranking_modes = {
        "trust": trust_scores,
        "realism": realism_scores,
        "faithfulness": faithfulness_scores,
    }

    real_feats_np = real_feats.numpy()
    gen_feats_np = gen_feats.numpy()

    all_results = []

    for mode_name, scores in ranking_modes.items():
        # Sort by score (lower = better)
        sorted_idx = np.argsort(scores)

        # Split into bins
        bin_size = len(sorted_idx) // n_bins
        bins = []
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_idx)
            bins.append(sorted_idx[start:end])

        for bin_idx, bin_indices in enumerate(bins):
            if len(bin_indices) == 0:
                continue

            gen_feats_bin = gen_feats_np[bin_indices]
            score_range = (scores[bin_indices].min(), scores[bin_indices].max())

            # Bootstrap KID
            kid_stats = bootstrap_kid_for_bin(
                gen_feats_bin,
                real_feats_np,
                n_bootstrap=n_bootstrap,
                use_cosine=effective_cosine,
            )

            all_results.append(
                {
                    "ranking_mode": mode_name,
                    "bin_idx": bin_idx,
                    "score_min": float(score_range[0]),
                    "score_max": float(score_range[1]),
                    "mean_kid": kid_stats["mean_kid"],
                    "ci_low": kid_stats["ci_low"],
                    "ci_high": kid_stats["ci_high"],
                    "n_samples": len(bin_indices),
                }
            )

    df = pd.DataFrame(all_results)

    # Save CSV
    csv_path = output_dir / f"decile_binning_{config_key.replace('/', '_')}.csv"
    df.to_csv(csv_path, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for mode_name in ["trust", "realism", "faithfulness"]:
        mode_df = df[df["ranking_mode"] == mode_name]
        if len(mode_df) == 0:
            continue

        x = mode_df["bin_idx"]
        y = mode_df["mean_kid"]
        yerr_low = mode_df["mean_kid"] - mode_df["ci_low"]
        yerr_high = mode_df["ci_high"] - mode_df["mean_kid"]
        yerr = np.array([yerr_low, yerr_high])

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=mode_name.capitalize(),
            marker="o",
            capsize=4,
            alpha=0.8,
        )

    ax.set_xlabel("Bin Index (0=best, 9=worst)")
    ax.set_ylabel("KID (lower = better)")
    ax.set_title(f"Decile Binning: KID vs Ranking Mode - {config_key}")
    ax.legend()
    ax.grid(alpha=0.3)

    plot_path = output_dir / f"decile_binning_{config_key.replace('/', '_')}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return df


# ============================================================================
# Main
# ============================================================================


def create_report(
    dataset: str,
    all_results: List[Dict],
    ranking_results: Dict,
    failure_results: Dict,
    seen_unseen_results: Dict,
    alaa_results: Dict,
    multi_backbone: Dict,
    task1_results: Dict,
    task2_results: Dict,
    task3_results: Dict,
    task4_results: Dict,
    output_dir: Path,
    normalize_mode: str = "none",
):
    """
    Create comprehensive markdown report.

    Args:
        dataset: Dataset name
        all_results: All trust score results
        ranking_results: Layer 1 ranking validity results
        failure_results: Layer 2A condition-level OOD results
        seen_unseen_results: Layer 2B sample-level OOD results
        alaa_results: Layer 4 Alaa correlation results
        multi_backbone: Layer 5 multi-backbone results
        task1_results: Task 1 full ranking results
        task2_results: Task 2 grid results
        task3_results: Task 3 (Layer 3) real vs gen OOD results
        task4_results: Task 4 decile binning results
        output_dir: Output directory for report
        normalize_mode: Feature normalization mode used
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    report = []
    report.append(f"# Trust Score Evaluation Report: {dataset.upper()}\n")

    # Summary
    report.append("## Summary\n")
    report.append(f"- **Dataset**: {dataset}")
    report.append(f"- **Models**: {set(r['model'] for r in all_results)}")
    feature_types = set(
        r.get("feature_type", r.get("encoder", "unknown")) for r in all_results
    )
    report.append(f"- **Feature Types**: {feature_types}")
    report.append(f"- **Feature Normalization**: {normalize_mode}\n")

    # Layer 1: Ranking Validity
    report.append("---\n## Layer 1: Condition-Level Ranking Validity\n")
    report.append("*Does trust-based ranking correlate with ground-truth KID?*\n")

    for config_key in sorted(ranking_results.keys()):
        r = ranking_results[config_key]
        report.append(f"\n### {config_key}\n")
        if isinstance(r, dict) and "spearman_rho" in r:
            report.append("**Trust scores (lower = better):**")
            report.append(
                f"- Spearman ρ (trust vs ΔKID): **{r.get('spearman_rho', np.nan):.4f}**"
            )
            report.append(
                f"- Kendall τ (trust vs ΔKID): {r.get('kendall_tau', np.nan):.4f}"
            )

            # Component correlations
            if "spearman_rho_realism" in r:
                report.append("\n**Components:**")
                report.append(
                    f"- Spearman ρ (realism vs ΔKID): {r.get('spearman_rho_realism', np.nan):.4f}"
                )
                report.append(
                    f"- Spearman ρ (faithfulness vs ΔKID): {r.get('spearman_rho_faithfulness', np.nan):.4f}"
                )

            report.append(f"\n- N conditions: {r.get('n_conditions', 0)}")

            # Top-k overlap
            for k in [5, 10, 20]:
                key = f"topk_overlap_{k}"
                if key in r:
                    report.append(
                        f"- Top-{k} overlap (trust vs KID ranking): {r[key]:.4f}"
                    )

            # Stratified correlation
            if "stratified_correlation" in r:
                report.append("\n**Stratified by support size:**")
                for bin_name, stats in r["stratified_correlation"].items():
                    if stats["n"] > 0:
                        report.append(
                            f"  - {bin_name} samples (n={stats['n']}): ρ={stats.get('spearman_rho', np.nan):.4f}"
                        )
            report.append("")

    # Layer 2: Failure Detection (unified section with clear subsections)
    if failure_results or seen_unseen_results:
        report.append("---\n## Layer 2: Failure Detection (OOD, Seen vs Unseen)\n")
        report.append(
            "*Can trust score detect out-of-distribution conditions/samples?*\n"
        )

        # Layer 2A: Condition-level OOD detection
        if failure_results:
            report.append("\n### Layer 2A: Condition-Level OOD Detection\n")
            report.append(
                "**Task**: Classify entire conditions as seen vs unseen.\n"
                "- **Positive class (1)**: Unseen attribute combinations (OOD conditions)\n"
                "- **Negative class (0)**: Seen attribute combinations (training conditions)\n"
                "- **Input**: Mean trust score per condition\n"
                "- **Granularity**: One prediction per unique condition (n_conditions = n_seen + n_unseen)\n"
            )
            for config_key, fr in sorted(failure_results.items()):
                report.append(f"\n#### {config_key}\n")
                report.append(
                    f"- **Condition-level AUROC**: **{fr.get('auroc', np.nan):.4f}**"
                )
                report.append(
                    f"- **Condition-level AUPRC**: {fr.get('auprc', np.nan):.4f}"
                )
                report.append(
                    f"- N seen conditions: {fr.get('n_seen_conds', 0)}, "
                    f"N unseen conditions: {fr.get('n_unseen_conds', 0)}"
                )
                report.append(
                    f"- Total conditions evaluated: {fr.get('n_total_conds', fr.get('n_seen_conds', 0) + fr.get('n_unseen_conds', 0))}\n"
                )

        # Layer 2B: Sample-level OOD detection (seen vs unseen, marginal models)
        if seen_unseen_results:
            report.append(
                "\n### Layer 2B: Sample-Level OOD Detection (Marginal Models)\n"
            )
            report.append(
                "**Task**: Classify individual generated samples as from seen vs unseen conditions.\n"
                "- **Positive class (1)**: Samples generated for unseen conditions\n"
                "- **Negative class (0)**: Samples generated for seen conditions\n"
                "- **Input**: Per-sample trust/realism/faithfulness scores\n"
                "- **Granularity**: One prediction per generated sample (n_samples = n_seen + n_unseen)\n"
            )
            for config_key, r in sorted(seen_unseen_results.items()):
                report.append(f"\n#### {config_key}\n")
                if r.get("status") == "success":
                    report.append(
                        f"- **Sample-level Trust AUROC**: **{r.get('trust_auroc', np.nan):.4f}**"
                    )
                    report.append(
                        f"- **Sample-level Realism AUROC**: {r.get('realism_auroc', np.nan):.4f}"
                    )
                    report.append(
                        f"- **Sample-level Faithfulness AUROC**: {r.get('faithfulness_auroc', np.nan):.4f}"
                    )
                    report.append(
                        f"- N seen samples: {r.get('n_seen', 0)}, N unseen samples: {r.get('n_unseen', 0)}"
                    )
                    report.append(f"- ROC plot: {Path(r['roc_path']).name}")
                else:
                    report.append(f"- Status: {r.get('status', 'unknown')}")
                report.append("")

    # =========================================================================
    # Layer 3: Real vs Generated Detection
    # =========================================================================
    # (This is Task 3 below - sample-level detection of generated samples)

    # Layer 4: Alaa Correlation
    if alaa_results:
        report.append("---\n## Layer 4: Correlation with Alaa et al. Metrics\n")
        report.append("*Does our score align with α-precision/authenticity?*\n")

        for config_key, results in sorted(alaa_results.items()):
            report.append(f"\n### {config_key}\n")
            for metric, vals in results.items():
                if isinstance(vals, dict) and "spearman_rho" in vals:
                    report.append(f"- {metric}: ρ = {vals['spearman_rho']:.4f}")
            report.append("")

    # Layer 5: Multi-feature-type
    if multi_backbone:
        report.append("---\n## Layer 5: Multi-Feature-Type Aggregate Statistics\n")
        report.append("*Per-feature-type aggregate scores*\n")

        for config_key, results in sorted(multi_backbone.items()):
            report.append(f"\n### {config_key}\n")
            report.append(f"- N feature types: {results.get('n_feature_types', 0)}")

            if "feature_stats" in results:
                report.append("\n**Per-feature-type statistics:**\n")
                report.append(
                    results["feature_stats"].round(4).to_markdown(index=False)
                )
            report.append("")

    # Extension Tasks
    if task1_results:
        report.append("---\n## Task 1: Full Condition Ranking + Gap Analysis\n")
        report.append("*Complete ranking sorted by ΔKID (best→worst)*\n")

        for config_key, r in sorted(task1_results.items()):
            report.append(f"\n### {config_key}\n")
            if r.get("status") == "success":
                report.append(f"- **Pearson ρ**: {r.get('pearson_rho', np.nan):.4f}")
                report.append(
                    f"- **Pearson p-value**: {r.get('pearson_p', np.nan):.4f}"
                )
                report.append(
                    f"- **Mean ΔKID gap**: {r['gap_analysis']['mean_gap']:.6f}"
                )
                report.append(f"- **Max ΔKID gap**: {r['gap_analysis']['max_gap']:.6f}")
                report.append(f"- **N conditions**: {r.get('n_conditions', 0)}")
                report.append(f"- **CSV**: {Path(r['csv_path']).name}")

                # Add inline table (sorted by delta_kid)
                if "table_df" in r and r["table_df"] is not None:
                    report.append("\n**Ranking Table:**\n")
                    table_df = r["table_df"].copy()
                    # Format numeric columns
                    for col in [
                        "kid_real_real",
                        "kid_real_gen",
                        "delta_kid",
                        "trust_mean",
                        "realism_mean",
                        "faithfulness_mean",
                    ]:

                        if col in table_df.columns:
                            table_df[col] = table_df[col].apply(lambda x: f"{x:.6f}")
                    report.append(table_df.to_markdown(index=False))
            else:
                report.append(f"- Status: {r.get('status', 'unknown')}")
            report.append("")

    if task2_results:
        report.append("---\n## Task 2: Realism/Faithfulness 2×2 Grids\n")
        report.append("*Visual examples of quadrants*\n")

        for config_key, r in sorted(task2_results.items()):
            report.append(f"\n### {config_key}\n")
            if r.get("status") == "success":
                report.append(f"- **Grids created**: {r.get('n_grids', 0)}")
                report.append(f"- **Manifest**: {Path(r['manifest_path']).name}")
                report.append(f"- **Directory**: {Path(r['grid_dir']).name}")
            else:
                report.append(f"- Status: {r.get('status', 'unknown')}")
            report.append("")

    # =========================================================================
    # Layer 3: Real vs Generated OOD Detection
    # =========================================================================
    if task3_results:
        report.append("\n---\n")
        report.append("## Layer 3: Real vs Generated OOD Detection\n")
        report.append(
            "**Task**: Classify samples as real (in-distribution) vs generated (out-of-distribution).\n"
            "- **Positive class (1)**: Generated samples\n"
            "- **Negative class (0)**: Real samples\n"
            "- **Input**: Per-sample trust/realism/faithfulness scores\n"
            "- **Note**: Higher AUROC indicates trust scores can distinguish real from generated samples.\n"
        )

        for config_key, splits_dict in sorted(task3_results.items()):
            report.append(f"\n### {config_key}\n")
            for split_name, r in sorted(splits_dict.items()):
                report.append(f"\n**Real vs Gen({split_name}):**")
                report.append(
                    f"- **Real-vs-Gen Trust AUROC**: **{r.get('trust_auroc', np.nan):.4f}**"
                )
                report.append(
                    f"- **Real-vs-Gen Realism AUROC**: {r.get('realism_auroc', np.nan):.4f}"
                )
                report.append(
                    f"- **Real-vs-Gen Faithfulness AUROC**: {r.get('faithfulness_auroc', np.nan):.4f}"
                )
                report.append(
                    f"- Real-vs-Gen Trust AUPRC: {r.get('trust_auprc', np.nan):.4f}"
                )
                report.append(
                    f"- N real samples: {r.get('n_real', 0)}, N gen samples: {r.get('n_gen', 0)}"
                )
            report.append("")

    if task4_results:
        report.append("---\n## Task 4: Decile Binning Analysis\n")
        report.append("*KID degradation across score bins*\n")

        for config_key, df in sorted(task4_results.items()):
            report.append(f"\n### {config_key}\n")
            if df is not None and not df.empty:
                # Show full table for trust mode (all 10 bins)
                trust_df = df[df["ranking_mode"] == "trust"]
                if not trust_df.empty:
                    cols = [
                        "bin_idx",
                        "score_min",
                        "score_max",
                        "mean_kid",
                        "ci_low",
                        "ci_high",
                        "n_samples",
                    ]
                    cols = [c for c in cols if c in trust_df.columns]
                    report.append("\n**Trust ranking (all bins):**\n")
                    report.append(trust_df[cols].round(6).to_markdown(index=False))

                csv_name = f"decile_binning_{config_key.replace('/', '_')}.csv"
                plot_name = f"decile_binning_{config_key.replace('/', '_')}.png"
                report.append(f"\n- **Full CSV**: {csv_name}")
                report.append(f"- **Plot**: {plot_name}")
            report.append("")

    # Write report
    with open(output_dir / f"TRUST_EVALUATION_{dataset}.md", "w") as f:
        f.write("\n".join(report))

    print(f"\nReport saved to {output_dir / f'TRUST_EVALUATION_{dataset}.md'}")


def verify_consolidated_features(
    gen_feats: torch.Tensor,
    gen_meta: Dict,
    condition_keys: List[str],
    filenames: Optional[List[str]] = None,
    n_spot_checks: int = 10,
) -> bool:
    """
    Verify that consolidated features have proper metadata alignment.

    Checks:
    1. All condition keys exist in metadata
    2. Metadata arrays have same length as features
    3. For samples with filenames, condition values match filename encoding

    Args:
        gen_feats: Generated features tensor (N, D)
        gen_meta: Metadata dict with condition keys
        condition_keys: Expected condition attribute keys
        filenames: Optional list of filenames for cross-check
        n_spot_checks: Number of random samples to spot-check

    Returns:
        True if all checks pass

    Raises:
        ValueError if any check fails
    """
    N = gen_feats.shape[0]

    # Check 1: All condition keys exist
    missing_keys = [k for k in condition_keys if k not in gen_meta]
    if missing_keys:
        raise ValueError(f"Missing condition keys in metadata: {missing_keys}")

    # Check 2: Metadata arrays have correct length
    for key in condition_keys:
        meta_arr = gen_meta[key]
        meta_len = len(meta_arr) if hasattr(meta_arr, "__len__") else meta_arr.shape[0]
        if meta_len != N:
            raise ValueError(
                f"Metadata '{key}' length mismatch: {meta_len} vs {N} features"
            )

    # Check 3: Spot-check filename -> condition consistency
    if filenames and len(filenames) == N:
        rng = np.random.default_rng(42)
        check_indices = rng.choice(N, size=min(n_spot_checks, N), replace=False)

        for idx in check_indices:
            fname = filenames[idx]
            # Parse signature from filename (e.g., "Blond_Hair0_..._0.png")
            stem = Path(fname).stem
            sig, _ = stem.rsplit("_", 1)

            # Parse condition from signature
            expected_cond = {}
            parts = sig.split("_")
            buffer = []
            for p in parts:
                if p and p[-1] in ["0", "1"] and len(p) > 1:
                    attr_name = "_".join(buffer + [p[:-1]])
                    expected_cond[attr_name] = int(p[-1])
                    buffer = []
                else:
                    buffer.append(p)

            # Compare with metadata
            for key in condition_keys:
                if key in expected_cond:
                    meta_val = gen_meta[key][idx]
                    if isinstance(meta_val, torch.Tensor):
                        meta_val = meta_val.item()
                    if expected_cond[key] != meta_val:
                        raise ValueError(
                            f"Condition mismatch at index {idx}: "
                            f"filename '{fname}' implies {key}={expected_cond[key]}, "
                            f"but metadata has {key}={meta_val}"
                        )

        logger.info(f"  ✓ Spot-checked {len(check_indices)} samples: conditions match filenames")

    logger.info(f"  ✓ Metadata integrity verified: {N} samples, {len(condition_keys)} condition keys")
    return True


def load_features_for_dataset(
    dataset: str, model: str, feature_type: str, normalize_mode: str = "none"
) -> Tuple[
    Optional[torch.Tensor], Optional[Dict], Optional[torch.Tensor], Optional[Dict]
]:
    """
    Load real and generated features for a specific model/feature_type.

    Args:
        dataset: Dataset name (e.g., "celeba", "rxrx1")
        model: Model name (e.g., "vanilla_full", "repa_marginal")
        feature_type: Feature type (e.g., "dinov3", "aligned_mean")
        normalize_mode: Normalization to apply ("none" or "l2")

    Returns:
        Tuple of (real_feats, real_meta, gen_feats, gen_meta)
        Returns (None, None, None, None) if features not found
    """
    # Get feature config
    config_key = (dataset, model, feature_type)
    if config_key not in FEATURE_CONFIGS:
        logger.warning(f"No config for {config_key}")
        return None, None, None, None

    gen_dir, feature_file = FEATURE_CONFIGS[config_key]

    # Real features - use meanpatch path for consistency
    real_key = (dataset, feature_type)
    if real_key in REAL_FEATURE_PATHS:
        real_path = Path(REAL_FEATURE_PATHS[real_key])
    else:
        # Fallback to old path
        real_path = Path(f"outputs/real_{dataset}_dinov3_meanpatch/train_features.pt")

    gen_path = Path(f"outputs/gen/{gen_dir}/{feature_file}")

    if not real_path.exists():
        logger.warning(f"Real features not found at {real_path}")
        return None, None, None, None

    if not gen_path.exists():
        logger.warning(f"Generated features not found at {gen_path}")
        return None, None, None, None

    logger.info(f"  Loading real features from: {real_path}")
    logger.info(f"  Loading gen features from: {gen_path}")

    data = torch.load(real_path, map_location="cpu", weights_only=False)
    real_feats, real_meta = data["features"], data.get("metadata", {})

    data = torch.load(gen_path, map_location="cpu", weights_only=False)
    gen_feats, gen_meta = data["features"], data.get("metadata", {})
    gen_filenames = data.get("filenames", None)

    # Verify consolidated feature integrity
    condition_keys = CONDITION_ATTRS.get(dataset, [])
    if condition_keys and gen_meta:
        try:
            verify_consolidated_features(
                gen_feats, gen_meta, condition_keys, filenames=gen_filenames
            )
        except ValueError as e:
            logger.error(f"Consolidation integrity check FAILED: {e}")
            raise

    # Apply normalization if requested (once, at load time)
    real_feats = apply_normalization(real_feats, normalize_mode, f"real_{feature_type}")
    gen_feats = apply_normalization(gen_feats, normalize_mode, f"gen_{feature_type}")

    return real_feats, real_meta, gen_feats, gen_meta


def print_cosine_kid_transitivity_checks(
    dataset: str,
    model: str,
    condition_keys: List[str],
    ref_trust_results: Dict,
    n_conditions: int = 3,
    k: int = 256,
    seed: int = 0,
):
    """
    Console-only sanity check for cosine-KID scales and "triangulation" across:
      - KID_rr: cosineKID(real_dino_A, real_dino_B)
      - KID_rg_dino: cosineKID(real_dino_A, gen_dino)
      - KID_rg_aligned: cosineKID(real_dino_A, gen_aligned)
      - KID_gg_cross: cosineKID(gen_dino, gen_aligned)

    Also prints:
      - feature L2 norms (pre-normalization)
      - paired cosine similarities for (real-real), (real-gen_dino), (real-gen_aligned), (gen_dino-gen_aligned)
      - a basic filename-based alignment check (if metadata has filenames/paths)
    """
    print("\n" + "=" * 80)
    print(
        f"[TRANSITIVITY CHECK] dataset={dataset} model={model}  (cosine-KID + paired cosines)"
    )
    print("=" * 80)

    # Must have aligned_mean config to run
    if (dataset, model, "aligned_mean") not in FEATURE_CONFIGS or (
        dataset,
        model,
        "dinov3",
    ) not in FEATURE_CONFIGS:
        print(
            "  Skipping: this model/dataset does not have both dinov3 and aligned_mean feature caches."
        )
        return

    # Load features
    real_feats, real_meta, gen_dino_feats, gen_dino_meta = load_features_for_dataset(
        dataset, model, "dinov3"
    )
    _, _, gen_aligned_feats, gen_aligned_meta = load_features_for_dataset(
        dataset, model, "aligned_mean"
    )

    if real_feats is None or gen_dino_feats is None or gen_aligned_feats is None:
        print("  Skipping: failed to load one of real/dino_gen/aligned_gen features.")
        return

    # Verify feature ordering between dinov3 and aligned_mean caches
    # This check happens once at load-time as required
    try:
        verify_feature_ordering(
            gen_dino_meta, gen_aligned_meta, "gen_dinov3", "gen_aligned_mean"
        )
    except ValueError as e:
        print(f"  ERROR: {e}")
        print("  Transitivity check aborted due to ordering mismatch.")
        return

    # Convert to numpy
    real_np = real_feats.numpy()
    gen_dino_np = gen_dino_feats.numpy()
    gen_aligned_np = gen_aligned_feats.numpy()

    # Length check (should pass if verification passed)
    if len(gen_dino_np) != len(gen_aligned_np):
        print(
            f"  ERROR: dinov3 ({len(gen_dino_np)}) and aligned_mean ({len(gen_aligned_np)}) "
            f"gen feature lengths differ. Transitivity check aborted."
        )
        return

    # Build gen indices by condition from trust_results ordering (assumed same ordering as gen feature cache)
    if "true_conditions" not in ref_trust_results:
        print("  Skipping: ref_trust_results missing true_conditions.")
        return

    gen_conditions = ref_trust_results["true_conditions"]
    if len(gen_conditions) != len(gen_dino_np):
        print(
            f"  WARNING: true_conditions length ({len(gen_conditions)}) != gen feats length ({len(gen_dino_np)})."
        )
        print(
            "           Results may be misindexed. Consider aligning conditions by filenames in metadata."
        )
        # still proceed, but clip to min
        n = min(len(gen_conditions), len(gen_dino_np))
        gen_conditions = gen_conditions[:n]
        gen_dino_np = gen_dino_np[:n]
        gen_aligned_np = gen_aligned_np[:n]

    gen_by_cond = {}
    for i, cond in enumerate(gen_conditions):
        gen_by_cond.setdefault(cond, []).append(i)

    # Build real indices by condition from real_meta
    real_by_cond = {}
    for i in range(len(real_np)):
        cond = tuple(
            int(
                real_meta[k][i].item()
                if isinstance(real_meta[k][i], torch.Tensor)
                else real_meta[k][i]
            )
            for k in condition_keys
        )
        real_by_cond.setdefault(cond, []).append(i)

    # Select candidate conditions with enough support
    candidates = []
    for cond, gidx in gen_by_cond.items():
        ridx = real_by_cond.get(cond, [])
        if len(ridx) >= 2 * max(10, min(k, 500)) and len(gidx) >= max(10, min(k, 500)):
            candidates.append(cond)

    if not candidates:
        print("  No conditions have sufficient real/gen support for this check.")
        return

    # Pick a few conditions deterministically (sorted) for reproducibility
    candidates = sorted(candidates)
    selected = candidates[:n_conditions]

    # Helpers
    def _l2norm_stats(X):
        n = np.linalg.norm(X, axis=1)
        return float(np.median(n)), float(np.mean(n)), float(np.std(n))

    def _paired_cos_mean(A, B):
        # paired dot products after L2 normalization
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        return float(np.mean(np.sum(A * B, axis=1)))

    rng = np.random.default_rng(seed)

    for cond in selected:
        ridx = np.array(real_by_cond[cond], dtype=int)
        gidx = np.array(gen_by_cond[cond], dtype=int)

        # Choose effective sample size
        k_eff = min(k, len(gidx), len(ridx) // 2, 500)
        if k_eff < 10:
            continue

        # Sample real split A/B and gen indices
        perm_r = rng.permutation(ridx)[: 2 * k_eff]
        real_a = real_np[perm_r[:k_eff]]
        real_b = real_np[perm_r[k_eff:]]
        chosen_g = rng.choice(gidx, size=k_eff, replace=False)
        gen_d = gen_dino_np[chosen_g]
        gen_a = gen_aligned_np[chosen_g]

        # Cosine-KIDs
        kid_rr = calculate_kid_same_m(real_a, real_b, use_cosine=True)
        kid_rg_dino = calculate_kid_same_m(real_a, gen_d, use_cosine=True)
        kid_rg_aligned = calculate_kid_same_m(real_a, gen_a, use_cosine=True)
        kid_gg_cross = calculate_kid_same_m(gen_d, gen_a, use_cosine=True)

        # Paired cosine similarity sanity checks
        cos_rr = _paired_cos_mean(real_a, real_b)
        cos_rg_d = _paired_cos_mean(real_a, gen_d)
        cos_rg_a = _paired_cos_mean(real_a, gen_a)
        cos_gg = _paired_cos_mean(gen_d, gen_a)

        # Norm stats (pre-normalization; should not matter for cosine-kid but helps spot degeneracy)
        real_med, real_mean, real_std = _l2norm_stats(real_a)
        gd_med, gd_mean, gd_std = _l2norm_stats(gen_d)
        ga_med, ga_mean, ga_std = _l2norm_stats(gen_a)

        # Print
        cond_str = ", ".join(f"{k}={v}" for k, v in zip(condition_keys, cond))
        print("\n" + "-" * 80)
        print(
            f"Condition: {cond_str}   (k={k_eff}, n_real={len(ridx)}, n_gen={len(gidx)})"
        )
        print("Cosine-KID (MMD^2, lower better; small negatives are OK):")
        print(f"  KID_rr          = {kid_rr:.6f}   [real_dino vs real_dino]")
        print(f"  KID_rg_dino     = {kid_rg_dino:.6f}   [real_dino vs gen_dino]")
        print(f"  KID_rg_aligned  = {kid_rg_aligned:.6f}   [real_dino vs gen_aligned]")
        print(f"  KID_gg_cross    = {kid_gg_cross:.6f}   [gen_dino  vs gen_aligned]")
        print("Paired mean cosine similarities (higher is more aligned directionally):")
        print(f"  mean cos(real_a, real_b)      = {cos_rr:.4f}")
        print(f"  mean cos(real_a, gen_dino)    = {cos_rg_d:.4f}")
        print(f"  mean cos(real_a, gen_aligned) = {cos_rg_a:.4f}")
        print(f"  mean cos(gen_dino, gen_aligned)= {cos_gg:.4f}")
        print("Median/mean/std of L2 norms (pre-normalization):")
        print(
            f"  real_a  norm: med={real_med:.3f} mean={real_mean:.3f} std={real_std:.3f}"
        )
        print(f"  gen_dino norm: med={gd_med:.3f} mean={gd_mean:.3f} std={gd_std:.3f}")
        print(f"  gen_algn norm: med={ga_med:.3f} mean={ga_mean:.3f} std={ga_std:.3f}")

        # Minimal “triangulation” commentary (console only, not a statistical claim)
        if (
            np.isfinite(kid_rg_dino)
            and np.isfinite(kid_rg_aligned)
            and abs(kid_rg_dino) > 1e-12
        ):
            ratio = kid_rg_aligned / kid_rg_dino
            print(f"Scale check: KID_rg_aligned / KID_rg_dino = {ratio:.2f}x")
            if cos_gg > 0.85 and ratio > 5.0:
                print(
                    "  NOTE: high gen_dino↔gen_aligned cosine but much larger cross-space KID suggests distribution-level mismatch"
                )
                print(
                    "        (not just feature norm), or ordering misalignment if metadata alignment is imperfect."
                )

    print("\n[TRANSITIVITY CHECK DONE]\n")


def main():
    """
    Main entry point for trust score evaluation.

    Runs all evaluation layers and generates a comprehensive report.
    """
    parser = argparse.ArgumentParser(
        description="Trust Score Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default run (no normalization)
  python scripts/run_trust_evaluation.py --dataset celeba

  # With L2 normalization on all features
  python scripts/run_trust_evaluation.py --dataset celeba --normalize-features l2
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="celeba",
        choices=["celeba", "rxrx1"],
        help="Dataset to evaluate (default: celeba)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/trust_evaluation",
        help="Output directory for reports and artifacts (default: outputs/trust_evaluation)",
    )
    parser.add_argument(
        "--enable-grids",
        action="store_true",
        help="Enable 2x2 grid generation (Task 2, slower)",
    )
    parser.add_argument(
        "--normalize-features",
        type=str,
        choices=["none", "l2"],
        default="none",
        help="Feature normalization mode. 'none': no normalization (default, backward-compatible). "
        "'l2': L2-normalize all feature vectors immediately after loading.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    condition_keys = CONDITION_ATTRS.get(args.dataset, [])
    normalize_mode = args.normalize_features

    print("=" * 60)
    print("TRUST SCORE EVALUATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Normalize features: {normalize_mode}")
    if normalize_mode == "l2":
        logger.info(
            "L2 normalization enabled: ALL feature vectors will be normalized after loading"
        )

    # # Load trust scores
    # print("\nLoading trust scores...")
    # all_results = load_trust_scores(args.dataset)
    # print(f"Loaded results for {len(all_results)} configurations")

    # Compute trust scores on the fly (from cached features)
    print("\\nComputing trust scores on the fly (from cached features)...")
    all_results: List[Dict] = []
    feature_cache: Dict[
        Tuple[str, str], Tuple[torch.Tensor, Dict, torch.Tensor, Dict]
    ] = {}

    for (cfg_dataset, model, feature_type), _cfg in FEATURE_CONFIGS.items():
        if cfg_dataset != args.dataset:
            continue

        config_key = f"{model}/{feature_type}"
        print(f"  Computing trust scores for {config_key} ...")

        # Load features once per config (with optional normalization)
        real_feats, real_meta, gen_feats, gen_meta = load_features_for_dataset(
            args.dataset, model, feature_type, normalize_mode=normalize_mode
        )
        if real_feats is None or gen_feats is None:
            continue

        feature_cache[(model, feature_type)] = (
            real_feats,
            real_meta,
            gen_feats,
            gen_meta,
        )

        # For marginal models, restrict real calibration to seen combos
        filter_by_seen = "marginal" in model and args.dataset == "celeba"
        seen_combos = MARGINAL_SEEN_COMBOS if filter_by_seen else None

        trust_res = compute_trust_results_from_features(
            dataset=args.dataset,
            model=model,
            feature_type=feature_type,
            real_feats=real_feats,
            real_meta=real_meta,
            gen_feats=gen_feats,
            gen_meta=gen_meta,
            condition_keys=condition_keys,
            filter_by_seen=filter_by_seen,
            seen_combos=seen_combos,
        )
        all_results.append(trust_res)

    print(f"Computed trust results for {len(all_results)} configurations")

    # Group by (model, feature_type)
    by_config = {}
    for r in all_results:
        model = r["model"]
        feature_type = r.get("feature_type", r.get("encoder", "unknown"))
        config_key = f"{model}/{feature_type}"
        if config_key not in by_config:
            by_config[config_key] = []
        by_config[config_key].append(r)

    # Run evaluations
    ranking_results = {}
    failure_results = {}
    seen_unseen_results = {}
    alaa_results = {}
    multi_backbone = {}
    task1_results = {}
    task2_results = {}
    task3_results = {}
    task4_results = {}

    for config_key in by_config:
        config_results = by_config[config_key]
        first_result = config_results[0]
        model = first_result["model"]
        feature_type = first_result.get(
            "feature_type", first_result.get("encoder", "unknown")
        )

        print(f"\n--- Evaluating {config_key} ---")

        # Determine effective KID mode based on normalization and feature type
        effective_kid_mode = get_effective_kid_mode(normalize_mode, feature_type)
        print(
            f"  KID mode: {effective_kid_mode} (normalize={normalize_mode}, feature_type={feature_type})"
        )

        # Load features for this configuration (with optional normalization)
        # print(f"  Loading features...")
        # real_feats, real_meta, gen_feats, gen_meta = load_features_for_dataset(
        #     args.dataset, model, feature_type, normalize_mode=normalize_mode
        # )
        # Use cached features for this configuration (already loaded above)
        cache_key = (model, feature_type)
        if cache_key not in feature_cache:
            logger.warning(f"No cached features for {cache_key}; skipping.")
            continue
        real_feats, real_meta, gen_feats, gen_meta = feature_cache[cache_key]

        # Layer 1: Ranking validity
        if real_feats is not None and gen_feats is not None:
            print("  Layer 1: Ranking validity...")
            # Use effective KID mode (determined by normalization + feature type)
            ranking_results[config_key] = evaluate_ranking_validity(
                first_result,
                real_feats,
                real_meta,
                gen_feats,
                condition_keys,
                feature_type,
                kid_mode=effective_kid_mode,
            )
            # Diagnostic: only add cosine_diag if we're NOT already using cosine
            if effective_kid_mode == "standard":
                ranking_results[f"{config_key}_cosine_diag"] = (
                    evaluate_ranking_validity(
                        first_result,
                        real_feats,
                        real_meta,
                        gen_feats,
                        condition_keys,
                        feature_type,
                        kid_mode="cosine",
                    )
                )

        # Layer 2: Failure detection (marginal models only)
        if "marginal" in model:
            print("  Layer 2: Failure detection...")
            failure_results[config_key] = evaluate_failure_detection(
                first_result, args.dataset
            )

        # Seen vs Unseen sample detection (marginal models only)
        if "marginal" in model:
            print("  Seen vs Unseen sample detection...")
            seen_unseen_results[config_key] = evaluate_seen_vs_unseen_detection(
                first_result, args.dataset, model, output_dir, config_key
            )

        # Layer 3: Selective generation curves - REMOVED, replaced by decile binning (Task 4)
        # Use decile binning instead which provides cleaner KID vs score bin analysis

        # Layer 4: Alaa correlation
        print("  Layer 4: Alaa et al. correlation...")
        alaa_results[config_key] = evaluate_alaa_correlation(first_result)

        # Layer 5: Multi-backbone (per model, aggregating feature types)
        print("  Layer 5: Multi-backbone aggregation...")
        multi_backbone[config_key] = evaluate_multi_backbone(all_results, model)

        # Extension Tasks
        # Task 1: Full condition ranking + gap analysis (after Layer 1)
        if real_feats is not None and ranking_results.get(config_key):
            print("  Task 1: Full condition ranking + gap analysis...")
            task1_results[config_key] = evaluate_full_condition_ranking(
                first_result,
                ranking_results[config_key],
                args.dataset,
                condition_keys,
                output_dir,
                config_key,
            )
            diag_key = f"{config_key}_cosine_diag"
            if diag_key in ranking_results:
                print("  Task 1b: Full condition ranking (cosine-KID diagnostic)...")
                task1_results[diag_key] = evaluate_full_condition_ranking(
                    first_result,
                    ranking_results[diag_key],
                    args.dataset,
                    condition_keys,
                    output_dir,
                    diag_key,
                )

        # Task 2: 2×2 grids (optional, after Layer 1)
        if args.enable_grids and real_feats is not None:
            print("  Task 2: Creating realism/faithfulness grids...")
            model_dir = FEATURE_CONFIGS.get(
                (args.dataset, model, feature_type), [None]
            )[0]
            if model_dir:
                task2_results[config_key] = create_realism_faithfulness_grids(
                    first_result,
                    model_dir,
                    args.dataset,
                    condition_keys,
                    output_dir,
                    config_key,
                )

        # Task 3: Sample-based OOD detection (after Layer 1)
        if real_feats is not None and gen_feats is not None:
            print("  Task 3: Sample-based OOD detection (cross-fit)...")
            task3_results[config_key] = evaluate_sample_ood_detection(
                first_result,
                real_feats,
                real_meta,
                gen_feats,
                gen_meta,
                condition_keys,
                args.dataset,
                model,
                output_dir,
                config_key,
            )

        # Task 4: Decile binning with ablations (after Layer 3)
        if real_feats is not None and gen_feats is not None:
            print("  Task 4: Decile binning with ablations...")
            task4_results[config_key] = evaluate_decile_binning(
                first_result,
                real_feats,
                real_meta,
                gen_feats,
                condition_keys,
                feature_type,
                output_dir,
                config_key,
                kid_mode=effective_kid_mode,  # Use effective mode based on normalization
            )

    # Create report
    print("\nGenerating report...")
    create_report(
        args.dataset,
        all_results,
        ranking_results,
        failure_results,
        seen_unseen_results,
        alaa_results,
        multi_backbone,
        task1_results,
        task2_results,
        task3_results,
        task4_results,
        output_dir,
        normalize_mode=normalize_mode,
    )

    # Save detailed results
    torch.save(
        {
            "ranking_results": ranking_results,
            "failure_results": failure_results,
            "seen_unseen_results": seen_unseen_results,
            "alaa_results": alaa_results,
            "multi_backbone": multi_backbone,
            "task1_results": task1_results,
            "task2_results": task2_results,
            "task3_results": task3_results,
            "task4_results": task4_results,
        },
        output_dir / f"detailed_results_{args.dataset}.pt",
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    # ------------------------------------------------------------
    # Console-only transitivity sanity checks for aligned features
    # ------------------------------------------------------------
    if args.dataset == "celeba":
        # Pick a reference trust_results for each model (prefer dinov3 if available)
        for m in ["repa_full", "repa_marginal"]:
            ref = next(
                (
                    r
                    for r in all_results
                    if r["model"] == m
                    and (r.get("feature_type", r.get("encoder", "")) == "dinov3")
                ),
                None,
            )
            if ref is None:
                # fall back to any entry for that model
                ref = next((r for r in all_results if r["model"] == m), None)
            if ref is None:
                continue

            print_cosine_kid_transitivity_checks(
                dataset=args.dataset,
                model=m,
                condition_keys=condition_keys,
                ref_trust_results=ref,
                n_conditions=3,
                k=256,
                seed=0,
            )


if __name__ == "__main__":
    main()
