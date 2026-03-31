"""
Layer 1: Condition-Level Ranking Validity.

Evaluates whether trust-based ranking agrees with ground-truth quality ranking (KID).
"""

import hashlib
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr

from faithful_cond_gen.eval.trust_eval.metrics_kid import (
    calculate_kid_same_m,
    estimate_kid_null_per_condition,
)


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
        if feature_type == "aligned_mean" or feature_type.startswith("aligned_"):
            return "cosine"  # Norms weren't optimized, normalize internally
        else:
            return "standard"  # Use raw features with dimension scaling


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


def evaluate_ranking_validity(
    trust_results: Dict,
    real_feats: torch.Tensor,
    real_meta: Dict,
    gen_feats: torch.Tensor,
    condition_keys: List[str],
    feature_type: str = "dinov3",
    n_bootstrap: int = 10,
    kid_mode: str = "auto",
    use_kid_z: bool = True,
    kid_null_n_resamples: int = 100,
) -> Dict:
    """
    Layer 1: Does trust-based ranking agree with ground-truth quality ranking?

    Computes Spearman/Kendall correlation between per-condition mean trust
    and per-condition KID (z-normalized or delta).

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
        use_kid_z: If True (default), use z-normalized KID. If False, use legacy ΔKID.
        kid_null_n_resamples: Number of resamples for null distribution estimation (default: 100)
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

    # Estimate per-condition null distribution for z-KID
    kid_null_stats = {}
    if use_kid_z:
        real_feats_np = real_feats.numpy() if isinstance(real_feats, torch.Tensor) else real_feats
        real_feats_by_cond = {
            cond: real_feats_np[idx] for cond, idx in real_by_cond.items()
        }
        kid_null_stats = estimate_kid_null_per_condition(
            real_feats_by_cond,
            n_resamples=kid_null_n_resamples,
            use_cosine=effective_cosine,
            seed=42,
        )
        print(f"    Estimated KID null for {len(kid_null_stats)}/{len(real_by_cond)} conditions")

    # Compute KID per condition (z-normalized or delta)
    delta_kids = {}
    base_kids = {}
    gen_kids = {}
    kid_z_scores = {}
    kid_null_mu = {}
    kid_null_sigma = {}
    for cond in gen_by_cond:
        gen_idx = gen_by_cond[cond]
        real_idx = real_by_cond.get(cond, [])

        if len(real_idx) < 20 or len(gen_idx) < 5:
            delta_kids[cond] = np.nan
            kid_z_scores[cond] = np.nan
            continue

        gen_f = gen_feats[gen_idx].numpy()
        real_f = real_feats[real_idx].numpy()

        # Bootstrap delta KID
        k = min(len(real_idx) // 2, len(gen_idx), 500)
        if k < 5:
            delta_kids[cond] = np.nan
            kid_z_scores[cond] = np.nan
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

        # Compute z-KID if null stats available
        if use_kid_z and cond in kid_null_stats:
            mu_c, sigma_c = kid_null_stats[cond]
            kid_null_mu[cond] = mu_c
            kid_null_sigma[cond] = sigma_c
            gen_kid_mean = gen_kids[cond]
            if np.isfinite(gen_kid_mean):
                kid_z_scores[cond] = (gen_kid_mean - mu_c) / sigma_c
            else:
                kid_z_scores[cond] = np.nan
        else:
            kid_z_scores[cond] = np.nan

    # Merge with trust stats
    cond_stats["delta_kid"] = cond_stats["condition"].apply(
        lambda c: delta_kids.get(c, np.nan)
    )
    # Add real support count (for stratified correlation)
    cond_stats["n_real"] = cond_stats["condition"].apply(
        lambda c: len(real_by_cond.get(c, []))
    )
    cond_stats["kid_real_real"] = cond_stats["condition"].apply(
        lambda c: base_kids.get(c, np.nan)
    )
    cond_stats["kid_real_gen"] = cond_stats["condition"].apply(
        lambda c: gen_kids.get(c, np.nan)
    )
    cond_stats["base_kid"] = cond_stats["kid_real_real"]
    cond_stats["gen_kid"] = cond_stats["kid_real_gen"]
    # Add z-KID columns
    cond_stats["kid_z"] = cond_stats["condition"].apply(
        lambda c: kid_z_scores.get(c, np.nan)
    )
    cond_stats["kid_null_mu"] = cond_stats["condition"].apply(
        lambda c: kid_null_mu.get(c, np.nan)
    )
    cond_stats["kid_null_sigma"] = cond_stats["condition"].apply(
        lambda c: kid_null_sigma.get(c, np.nan)
    )

    # Determine primary KID metric for correlation
    if use_kid_z:
        # Use z-KID as primary metric
        valid_z = cond_stats[cond_stats["kid_z"].notna()]
        valid = valid_z if len(valid_z) >= 3 else cond_stats[cond_stats["delta_kid"].notna()]
        primary_kid_col = "kid_z" if len(valid_z) >= 3 else "delta_kid"
    else:
        valid = cond_stats[cond_stats["delta_kid"].notna()]
        primary_kid_col = "delta_kid"

    if len(valid) < 3:
        return {
            "spearman_rho": np.nan,
            "kendall_tau": np.nan,
            "n_conditions": len(valid),
            "use_kid_z": use_kid_z,
            "primary_kid_metric": primary_kid_col,
        }

    # Trust score correlation (lower trust = better, lower KID = better)
    spearman_rho, spearman_p = spearmanr(valid["trust_mean"], valid[primary_kid_col])
    kendall_tau, kendall_p = kendalltau(valid["trust_mean"], valid[primary_kid_col])

    result = {
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "kendall_tau": kendall_tau,
        "kendall_p": kendall_p,
        "n_conditions": len(valid),
        "cond_stats": cond_stats,
        "use_kid_z": use_kid_z,
        "primary_kid_metric": primary_kid_col,
        "n_conditions_with_z": len(cond_stats[cond_stats["kid_z"].notna()]),
    }

    # Also compute correlation for realism and faithfulness components separately
    rho_realism, _ = spearmanr(valid["realism_mean"], valid[primary_kid_col])
    rho_faithfulness, _ = spearmanr(valid["faithfulness_mean"], valid[primary_kid_col])
    result["spearman_rho_realism"] = rho_realism
    result["spearman_rho_faithfulness"] = rho_faithfulness

    # Also store legacy delta_kid correlations for comparison
    valid_delta = cond_stats[cond_stats["delta_kid"].notna()]
    if len(valid_delta) >= 3:
        rho_delta, _ = spearmanr(valid_delta["trust_mean"], valid_delta["delta_kid"])
        result["spearman_rho_delta_kid"] = rho_delta

    # Top-k overlap: do trust rankings match KID rankings?
    # Sort by trust (lower = better trust), sort by KID (lower = better quality)
    trust_ranking = valid.sort_values("trust_mean")["condition"].tolist()
    kid_ranking = valid.sort_values(primary_kid_col)["condition"].tolist()

    for k in [5, 10, 20]:
        if len(trust_ranking) >= k:
            result[f"topk_overlap_{k}"] = topk_overlap(
                np.array(trust_ranking), np.array(kid_ranking), k
            )

    # Stratified correlation by real support size (not gen samples)
    scores_arr = valid["trust_mean"].values
    kid_arr = valid[primary_kid_col].values
    real_support_arr = valid["n_real"].values
    result["stratified_correlation"] = stratified_correlation(
        scores_arr, kid_arr, real_support_arr
    )

    return result
