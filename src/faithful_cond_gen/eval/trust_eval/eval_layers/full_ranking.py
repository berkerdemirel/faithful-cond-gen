"""
Task 1: Full Condition Ranking + Gap Analysis.

Complete ranking sorted by ΔKID with correlation analysis.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import pearsonr, spearmanr

from faithful_cond_gen.eval.trust_eval.config import MARGINAL_SEEN_COMBOS


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

        # Correlations (both Spearman and Pearson for cross-reference with Layer 1)
        spearman_rho, spearman_p = spearmanr(valid["trust_mean"], valid["delta_kid"])
        pearson_rho, pearson_p = pearsonr(valid["trust_mean"], valid["delta_kid"])

        cols = ["condition_str"]
        # include these if available (they will be for your ranking_results after the patch above)
        for c in ["kid_real_real", "kid_real_gen", "delta_kid"]:
            if c in valid.columns:
                cols.append(c)
        cols += ["trust_mean", "realism_mean", "faithfulness_mean"]
        table_df = valid.sort_values("delta_kid")[cols].copy()

        # Save CSV with full details
        csv_path = output_dir / f"{dataset}_full_ranking_{config_key.replace('/', '_')}.csv"
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
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
            "pearson_rho": float(pearson_rho),
            "pearson_p": float(pearson_p),
            "gap_analysis": gap_analysis,
            "csv_path": str(csv_path),
            "n_conditions": len(valid),
            "table_df": table_df,  # For inline markdown table
        }
    else:
        return {"status": "insufficient_conditions", "n_conditions": len(valid)}
