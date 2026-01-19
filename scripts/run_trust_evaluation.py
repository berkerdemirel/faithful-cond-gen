"""
Trust Score Evaluation: Extended Analysis with Selective Generation Curves.

Implements the 5 evaluation layers from the research plan:
1. Condition-level ranking validity (T̄ vs KID correlation)
2. Failure detection (PR/AUROC for predicting bad conditions)
3. Selective generation curves (KID vs acceptance rate)
4. Correlation with Alaa et al. metrics
5. Multi-backbone aggregation

Usage:
    uv run python scripts/run_trust_evaluation.py --dataset celeba
    uv run python scripts/run_trust_evaluation.py --dataset rxrx1
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ============================================================================
# Configuration
# ============================================================================

TRUST_SCORES_DIR = Path("outputs/trust_scores")
OUTPUT_DIR = Path("outputs/trust_evaluation")

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
# Helper functions
# ============================================================================


def load_trust_scores(dataset: str) -> List[Dict]:
    """Load precomputed trust scores."""
    path = TRUST_SCORES_DIR / f"trust_scores_{dataset}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Trust scores not found at {path}. Run compute_trust_scores.py first."
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def get_per_condition_stats(results: Dict) -> pd.DataFrame:
    """Aggregate scores per condition."""
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


def calculate_kid_same_m(X: np.ndarray, Y: np.ndarray) -> float:
    """Unbiased MMD^2 estimate with polynomial kernel: k(x,y)=(x·y/d + 1)^3"""
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
    n_bootstrap: int = 10,
) -> Dict:
    """
    Layer 1: Does trust-based ranking agree with ground-truth quality ranking?

    Computes Spearman/Kendall correlation between per-condition mean trust
    and per-condition delta KID.
    """
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
        for _ in range(n_bootstrap):
            perm = rng.permutation(len(real_idx))
            idx_a, idx_b = perm[:k], perm[k : 2 * k]
            real_a, real_b = real_f[idx_a], real_f[idx_b]
            gen_samp = gen_f[rng.choice(len(gen_f), k, replace=False)]

            base = calculate_kid_same_m(real_a, real_b)
            gen_kid = calculate_kid_same_m(real_a, gen_samp)
            if np.isfinite(base) and np.isfinite(gen_kid):
                deltas.append(gen_kid - base)

        delta_kids[cond] = np.mean(deltas) if deltas else np.nan

    # Merge with trust stats
    cond_stats["delta_kid"] = cond_stats["condition"].apply(
        lambda c: delta_kids.get(c, np.nan)
    )
    # Add real support count (for stratified correlation)
    cond_stats["n_real"] = cond_stats["condition"].apply(
        lambda c: len(real_by_cond.get(c, []))
    )

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
# Layer 3: Selective Generation Curves (KID vs Acceptance Rate)
# ============================================================================


def compute_selective_generation_curve(
    trust_results: Dict,
    real_feats: torch.Tensor,
    real_meta: Dict,
    gen_feats: torch.Tensor,
    condition_keys: List[str],
    acceptance_rates: List[float] = None,
    n_bootstrap: int = 10,
    k0: int = None,
) -> pd.DataFrame:
    """
    Layer 3: Does filtering by trust improve quality?

    For each acceptance rate q:
    - Keep best q% samples (lowest trust score)
    - Compute conditional KID on kept subset with FIXED sample size k0
    - Bootstrap for confidence intervals

    Args:
        k0: Fixed sample size for KID computation. If None, determined from smallest coverage.
        n_bootstrap: Number of bootstrap iterations for CI (default: 10)
    """
    if acceptance_rates is None:
        acceptance_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Use trust_updated (lower = better)
    trust_scores = trust_results["trust_updated"]

    conditions = trust_results["true_conditions"]

    # Group by condition
    gen_by_cond = {}
    for i, cond in enumerate(conditions):
        if cond not in gen_by_cond:
            gen_by_cond[cond] = []
        gen_by_cond[cond].append((i, trust_scores[i]))

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

    # Determine k0 based on smallest acceptance rate if not provided
    if k0 is None:
        min_q = min(acceptance_rates)
        min_samples_per_cond = []
        for cond, samples in gen_by_cond.items():
            real_idx = real_by_cond.get(cond, [])
            if len(real_idx) >= 10 and len(samples) >= 5:
                n_kept_at_min_q = max(1, int(len(samples) * min_q))
                k_cond = min(len(real_idx) // 2, n_kept_at_min_q, 200)
                if k_cond >= 5:
                    min_samples_per_cond.append(k_cond)
        k0 = min(min_samples_per_cond) if min_samples_per_cond else 10
        k0 = max(k0, 5)  # Ensure at least 5

    results = []

    for q in tqdm(acceptance_rates, desc="Computing selective KID curves"):
        # Bootstrap loop
        bootstrap_kids = []
        skipped_conditions = []
        n_conditions_used = 0  # Track conditions that contributed a finite KID

        for boot_iter in range(n_bootstrap):
            rng = np.random.default_rng(42 + boot_iter)
            kids = []
            conditions_contributing = 0

            for cond, samples in gen_by_cond.items():
                real_idx = real_by_cond.get(cond, [])
                if len(real_idx) < 2 * k0:
                    if boot_iter == 0:
                        skipped_conditions.append((cond, "insufficient_real"))
                    continue

                # Sort by trust score (lower = better) and keep top q%
                sorted_samples = sorted(samples, key=lambda x: x[1])
                n_keep = max(1, int(len(sorted_samples) * q))
                kept_idx = [s[0] for s in sorted_samples[:n_keep]]

                if len(kept_idx) < k0:
                    if boot_iter == 0:
                        skipped_conditions.append(
                            (cond, "insufficient_gen_after_filter")
                        )
                    continue

                # Fixed sample size k0 for KID
                gen_f = gen_feats[kept_idx].numpy()
                real_f = real_feats[real_idx].numpy()

                # Bootstrap: resample
                perm = rng.permutation(len(real_idx))
                real_a = real_f[perm[:k0]]
                gen_samp = gen_f[rng.choice(len(gen_f), k0, replace=len(gen_f) < k0)]

                try:
                    kid = calculate_kid_same_m(real_a, gen_samp)
                    if np.isfinite(kid):
                        kids.append(kid)
                        conditions_contributing += 1
                except Exception:
                    pass

            if kids:
                bootstrap_kids.append(np.mean(kids))

            # Track from first bootstrap iteration
            if boot_iter == 0:
                n_conditions_used = conditions_contributing

        # Aggregate bootstrap results
        if bootstrap_kids:
            mean_kid = np.mean(bootstrap_kids)
            std_kid = np.std(bootstrap_kids, ddof=1) if len(bootstrap_kids) > 1 else 0.0
            ci_low, ci_high = (
                np.percentile(bootstrap_kids, [2.5, 97.5])
                if len(bootstrap_kids) > 1
                else (mean_kid, mean_kid)
            )
        else:
            mean_kid = np.nan
            std_kid = np.nan
            ci_low = np.nan
            ci_high = np.nan

        # Count samples
        total_samples = sum(len(samples) for samples in gen_by_cond.values())
        total_accepted = sum(
            max(1, int(len(samples) * q)) for samples in gen_by_cond.values()
        )

        results.append(
            {
                "acceptance_rate": q,
                "mean_kid": mean_kid,
                "std_kid": std_kid,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "k0": k0,
                "n_bootstrap": n_bootstrap,
                "n_conditions_used": n_conditions_used,
                "n_skipped": len(skipped_conditions),
                "total_samples": total_samples,
                "total_accepted": total_accepted,
            }
        )

    return pd.DataFrame(results)


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
    Layer 5: Multi-backbone robustness analysis.

    Reports per-backbone aggregate statistics only.
    NOTE: Sample-level correlations across backbones removed because there's
    no guarantee samples are in the same order across different encoder caches.
    """
    # Filter to single model
    model_results = [r for r in all_results if r["model"] == model]
    if len(model_results) < 1:
        return {"n_backbones": 0}

    backbones = [r["encoder"] for r in model_results]

    # Per-backbone aggregate statistics (don't rely on same ordering)
    backbone_stats = []
    for r in model_results:
        trust = r["trust_updated"]
        realism = r["realism_global_z"]
        faithfulness = r["faithfulness_margin_z"]

        backbone_stats.append(
            {
                "backbone": r["encoder"],
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
        "n_backbones": len(backbones),
        "backbones": backbones,
        "backbone_stats": pd.DataFrame(backbone_stats),
    }

    return result


# ============================================================================
# Main
# ============================================================================


def create_report(
    dataset: str,
    all_results: List[Dict],
    ranking_results: Dict,
    failure_results: Dict,
    selective_curves: Dict,
    alaa_results: Dict,
    multi_backbone: Dict,
    output_dir: Path,
):
    """Create comprehensive markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report = []
    report.append(f"# Trust Score Evaluation Report: {dataset.upper()}\n")

    # Summary
    report.append("## Summary\n")
    report.append(f"- **Dataset**: {dataset}")
    report.append(f"- **Models**: {set(r['model'] for r in all_results)}")
    report.append(f"- **Encoders**: {set(r['encoder'] for r in all_results)}\n")

    # Layer 1: Ranking Validity
    report.append("---\n## Layer 1: Condition-Level Ranking Validity\n")
    report.append("*Does trust-based ranking correlate with ground-truth KID?*\n")

    for model in ["fullmodel", "marginalmodel"]:
        if model in ranking_results:
            r = ranking_results[model]
            report.append(f"\n### {model}\n")
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

    # Layer 2: Failure Detection
    if failure_results:
        report.append("---\n## Layer 2: Failure Detection (OOD)\n")
        report.append(
            "*Can trust score detect unseen attribute combinations? (condition-level)*\n"
        )
        report.append(f"- AUROC: **{failure_results.get('auroc', np.nan):.4f}**")
        report.append(f"- AUPRC: {failure_results.get('auprc', np.nan):.4f}")
        report.append(
            f"- N seen conditions: {failure_results.get('n_seen_conds', 0)}, N unseen conditions: {failure_results.get('n_unseen_conds', 0)}\n"
        )

    # Layer 3: Selective Generation
    if selective_curves:
        report.append("---\n## Layer 3: Selective Generation Curves\n")
        report.append(
            "*Does filtering by trust improve KID? (with bootstrap 95% CI)*\n"
        )

        for model, df in selective_curves.items():
            report.append(f"\n### {model}\n")
            if not df.empty:
                # Show key columns with formatting
                cols_to_show = [
                    "acceptance_rate",
                    "mean_kid",
                    "ci_low",
                    "ci_high",
                    "k0",
                    "n_conditions_used",
                    "n_skipped",
                ]
                cols_to_show = [c for c in cols_to_show if c in df.columns]
                report.append(df[cols_to_show].round(6).to_markdown(index=False))

                # Summary interpretation
                if len(df) > 1:
                    kid_at_10 = df[df["acceptance_rate"] == 0.1]["mean_kid"].values
                    kid_at_100 = df[df["acceptance_rate"] == 1.0]["mean_kid"].values
                    if len(kid_at_10) > 0 and len(kid_at_100) > 0:
                        improvement = kid_at_100[0] - kid_at_10[0]
                        report.append(
                            f"\n*KID improvement (100% → 10% acceptance): {improvement:.6f}*"
                        )
                report.append("\n")

    # Layer 4: Alaa Correlation
    if alaa_results:
        report.append("---\n## Layer 4: Correlation with Alaa et al. Metrics\n")
        report.append("*Does our score align with α-precision/authenticity?*\n")

        for model, results in alaa_results.items():
            report.append(f"\n### {model}\n")
            for metric, vals in results.items():
                if isinstance(vals, dict) and "spearman_rho" in vals:
                    report.append(f"- {metric}: ρ = {vals['spearman_rho']:.4f}")
            report.append("")

    # Layer 5: Multi-backbone
    if multi_backbone:
        report.append("---\n## Layer 5: Multi-Backbone Aggregate Statistics\n")
        report.append(
            "*Per-backbone aggregate scores (sample-level correlations not computed due to ordering uncertainty)*\n"
        )

        for model, results in multi_backbone.items():
            report.append(f"\n### {model}\n")
            report.append(f"- N backbones: {results.get('n_backbones', 0)}")

            if "backbone_stats" in results:
                report.append("\n**Per-backbone statistics:**\n")
                report.append(
                    results["backbone_stats"].round(4).to_markdown(index=False)
                )
            report.append("")

    # Write report
    with open(output_dir / f"TRUST_EVALUATION_{dataset}.md", "w") as f:
        f.write("\n".join(report))

    print(f"\nReport saved to {output_dir / f'TRUST_EVALUATION_{dataset}.md'}")


def load_features_for_dataset(
    dataset: str, encoder: str, model: str = "fullmodel"
) -> Tuple[torch.Tensor, Dict, torch.Tensor, Dict]:
    """Load real and generated features for a specific model."""
    real_path = (
        Path("feature_cache/real_samples") / dataset / encoder / "train_features.pt"
    )
    gen_path = Path("feature_cache/generated_samples") / dataset / model / encoder

    if not real_path.exists():
        print(f"  Warning: real features not found at {real_path}")
        return None, None, None, None

    gen_files = list(gen_path.glob("*features.pt"))
    if not gen_files:
        print(f"  Warning: no generated features in {gen_path}")
        return None, None, None, None

    data = torch.load(real_path, map_location="cpu", weights_only=False)
    real_feats, real_meta = data["features"], data.get("metadata", {})

    data = torch.load(gen_files[0], map_location="cpu", weights_only=False)
    gen_feats, gen_meta = data["features"], data.get("metadata", {})

    return real_feats, real_meta, gen_feats, gen_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="celeba", choices=["celeba", "rxrx1"]
    )
    parser.add_argument("--output-dir", type=str, default="outputs/trust_evaluation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    condition_keys = CONDITION_ATTRS.get(args.dataset, [])

    print("=" * 60)
    print("TRUST SCORE EVALUATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")

    # Load trust scores
    print("\nLoading trust scores...")
    all_results = load_trust_scores(args.dataset)
    print(f"Loaded results for {len(all_results)} configurations")

    # Group by model
    by_model = {}
    for r in all_results:
        model = r["model"]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(r)

    # Run evaluations
    ranking_results = {}
    failure_results = {}
    selective_curves = {}
    alaa_results = {}
    multi_backbone = {}

    for model in by_model:
        model_results = by_model[model]
        first_result = model_results[0]
        # Use the encoder that matches this result (not first_encoder which may differ)
        encoder_for_eval = first_result["encoder"]

        print(f"\n--- Evaluating {model} ---")

        # Load model-specific features (critical: each model has different generated samples)
        # Must use encoder matching first_result to ensure indices align
        print(f"  Loading features ({encoder_for_eval}, {model})...")
        real_feats, real_meta, gen_feats, gen_meta = load_features_for_dataset(
            args.dataset, encoder_for_eval, model
        )

        # Layer 1: Ranking validity
        if real_feats is not None and gen_feats is not None:
            print("  Layer 1: Ranking validity...")
            ranking_results[model] = evaluate_ranking_validity(
                first_result, real_feats, real_meta, gen_feats, condition_keys
            )

        # Layer 2: Failure detection (marginal only)
        if model == "marginalmodel":
            print("  Layer 2: Failure detection...")
            failure_results = evaluate_failure_detection(first_result, args.dataset)

        # Layer 3: Selective generation curves
        if real_feats is not None and gen_feats is not None:
            print("  Layer 3: Selective generation curves...")
            selective_curves[model] = compute_selective_generation_curve(
                first_result, real_feats, real_meta, gen_feats, condition_keys
            )

        # Layer 4: Alaa correlation
        print("  Layer 4: Alaa et al. correlation...")
        alaa_results[model] = evaluate_alaa_correlation(first_result)

        # Layer 5: Multi-backbone
        print("  Layer 5: Multi-backbone aggregation...")
        multi_backbone[model] = evaluate_multi_backbone(all_results, model)

    # Create report
    print("\nGenerating report...")
    create_report(
        args.dataset,
        all_results,
        ranking_results,
        failure_results,
        selective_curves,
        alaa_results,
        multi_backbone,
        output_dir,
    )

    # Save detailed results
    torch.save(
        {
            "ranking_results": ranking_results,
            "failure_results": failure_results,
            "selective_curves": selective_curves,
            "alaa_results": alaa_results,
            "multi_backbone": multi_backbone,
        },
        output_dir / f"detailed_results_{args.dataset}.pt",
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
