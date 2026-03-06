"""
Task 4: Decile Binning Analysis.

KID degradation across score bins.
Supports both global binning and within-condition binning with z-KID.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from faithful_cond_gen.eval.trust_eval.metrics_kid import (
    bootstrap_kid_for_bin,
    calculate_kid_same_m,
    estimate_kid_null_per_condition,
)


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
    kid_mode: str = "auto",
    within_condition: bool = True,
    dataset: str = "",
) -> pd.DataFrame:
    """
    Task 4: Decile binning with ablations.

    For each ranking mode (trust, realism, faithfulness):
    - Sort samples by score (best→worst)
    - Split into n_bins equal-sized bins
    - For each bin:
        - Compute KID vs real (raw or z-normalized)
        - Bootstrap for confidence intervals
    - Return DataFrame with all results
    - Plot line chart with error bars

    If within_condition=True (default):
    1. For each condition c: Sort c's samples by score, split into bins
    2. For each bin index i: Merge bin i across all conditions
    3. Compute z-KID per merged bin (average per-condition z-KIDs)

    If within_condition=False (legacy):
    - Global binning: sort all samples, split into bins, compute raw KID

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
        within_condition: If True (default), bin within each condition then merge
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
    print(f"    Within-condition binning: {within_condition}")

    # Get scores
    trust_scores = trust_results["trust_updated"]
    realism_scores = trust_results["realism_global_z"]
    faithfulness_scores = trust_results["faithfulness_margin_z"]
    true_conditions = trust_results["true_conditions"]

    ranking_modes = {
        "trust": trust_scores,
        "realism": realism_scores,
        "faithfulness": faithfulness_scores,
    }

    real_feats_np = real_feats.numpy() if isinstance(real_feats, torch.Tensor) else real_feats
    gen_feats_np = gen_feats.numpy() if isinstance(gen_feats, torch.Tensor) else gen_feats

    # Group real features by condition for z-KID
    real_by_cond: Dict[Tuple, List[int]] = {}
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
        real_by_cond.setdefault(cond, []).append(i)

    # Estimate null distribution for z-KID if using within_condition
    kid_null_stats = {}
    if within_condition:
        real_feats_by_cond = {
            cond: real_feats_np[idx] for cond, idx in real_by_cond.items()
        }
        kid_null_stats = estimate_kid_null_per_condition(
            real_feats_by_cond, n_resamples=100, use_cosine=effective_cosine, seed=42
        )
        print(f"    Estimated KID null for {len(kid_null_stats)}/{len(real_by_cond)} conditions")

    # Group generated features by condition
    gen_by_cond: Dict[Tuple, List[int]] = {}
    for i, cond in enumerate(true_conditions):
        gen_by_cond.setdefault(cond, []).append(i)

    all_results = []

    for mode_name, scores in ranking_modes.items():
        if within_condition:
            # Within-condition binning
            # 1. Bin within each condition
            within_cond_bins: Dict[Tuple, Dict[int, List[int]]] = {}
            for cond, indices in gen_by_cond.items():
                indices = np.array(indices)
                cond_scores = np.array([scores[i] for i in indices])
                sorted_local = np.argsort(cond_scores)
                sorted_indices = indices[sorted_local]

                n_cond = len(sorted_indices)
                bin_size = max(1, n_cond // n_bins)

                within_cond_bins[cond] = {}
                for bi in range(n_bins):
                    start = bi * bin_size
                    end = (bi + 1) * bin_size if bi < n_bins - 1 else n_cond
                    if start < n_cond:
                        within_cond_bins[cond][bi] = sorted_indices[start:end].tolist()

            # 2. Merge bins across conditions
            for bin_idx in range(n_bins):
                merged_indices = []
                conds_in_bin = []
                kid_z_values = []

                for cond in within_cond_bins:
                    if bin_idx not in within_cond_bins[cond]:
                        continue
                    cond_bin_indices = within_cond_bins[cond][bin_idx]
                    if len(cond_bin_indices) == 0:
                        continue

                    merged_indices.extend(cond_bin_indices)
                    conds_in_bin.append(cond)

                    # Compute z-KID for this condition's contribution
                    if cond in kid_null_stats and cond in real_by_cond:
                        mu_c, sigma_c = kid_null_stats[cond]
                        gen_cond = gen_feats_np[cond_bin_indices]
                        real_cond = real_feats_np[real_by_cond[cond]]

                        k = min(len(gen_cond), len(real_cond) // 2, 100)
                        if k >= 5:
                            kid = calculate_kid_same_m(
                                real_cond[:k], gen_cond[:k], use_cosine=effective_cosine
                            )
                            if np.isfinite(kid):
                                kid_z = (kid - mu_c) / sigma_c
                                kid_z_values.append(kid_z)

                if len(merged_indices) == 0:
                    continue

                # Compute average z-KID across conditions
                avg_kid_z = np.mean(kid_z_values) if kid_z_values else np.nan

                # Also compute raw KID for the merged bin
                gen_merged = gen_feats_np[merged_indices]
                kid_stats = bootstrap_kid_for_bin(
                    gen_merged,
                    real_feats_np,
                    n_bootstrap=n_bootstrap,
                    use_cosine=effective_cosine,
                )

                # Score range
                bin_scores = np.array([scores[i] for i in merged_indices])
                score_range = (bin_scores.min(), bin_scores.max())

                all_results.append({
                    "ranking_mode": mode_name,
                    "bin_idx": bin_idx,
                    "score_min": float(score_range[0]),
                    "score_max": float(score_range[1]),
                    "mean_kid": kid_stats["mean_kid"],
                    "ci_low": kid_stats["ci_low"],
                    "ci_high": kid_stats["ci_high"],
                    "n_samples": len(merged_indices),
                    "kid_z": float(avg_kid_z) if np.isfinite(avg_kid_z) else None,
                    "n_conditions": len(conds_in_bin),
                    "n_conditions_with_z": len(kid_z_values),
                    "binning_mode": "within_condition",
                })

        else:
            # Legacy global binning
            sorted_idx = np.argsort(scores)
            bin_size = len(sorted_idx) // n_bins

            for bin_idx in range(n_bins):
                start = bin_idx * bin_size
                end = (bin_idx + 1) * bin_size if bin_idx < n_bins - 1 else len(sorted_idx)
                bin_indices = sorted_idx[start:end]

                if len(bin_indices) == 0:
                    continue

                gen_feats_bin = gen_feats_np[bin_indices]
                score_range = (scores[bin_indices].min(), scores[bin_indices].max())

                kid_stats = bootstrap_kid_for_bin(
                    gen_feats_bin,
                    real_feats_np,
                    n_bootstrap=n_bootstrap,
                    use_cosine=effective_cosine,
                )

                all_results.append({
                    "ranking_mode": mode_name,
                    "bin_idx": bin_idx,
                    "score_min": float(score_range[0]),
                    "score_max": float(score_range[1]),
                    "mean_kid": kid_stats["mean_kid"],
                    "ci_low": kid_stats["ci_low"],
                    "ci_high": kid_stats["ci_high"],
                    "n_samples": len(bin_indices),
                    "kid_z": None,
                    "n_conditions": None,
                    "n_conditions_with_z": None,
                    "binning_mode": "global",
                })

    df = pd.DataFrame(all_results)

    # Save CSV
    dataset_prefix = f"{dataset}_" if dataset else ""
    csv_path = output_dir / f"{dataset_prefix}decile_binning_{config_key.replace('/', '_')}.csv"
    df.to_csv(csv_path, index=False)

    # Determine if we have z-KID data
    has_kid_z = "kid_z" in df.columns and df["kid_z"].notna().any()

    # Plot raw KID
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

    plot_path = output_dir / f"{dataset_prefix}decile_binning_{config_key.replace('/', '_')}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot z-KID if available
    if has_kid_z:
        fig, ax = plt.subplots(figsize=(10, 6))

        for mode_name in ["trust", "realism", "faithfulness"]:
            mode_df = df[df["ranking_mode"] == mode_name]
            mode_df = mode_df[mode_df["kid_z"].notna()]
            if len(mode_df) == 0:
                continue

            x = mode_df["bin_idx"]
            y = mode_df["kid_z"]

            ax.plot(
                x,
                y,
                label=mode_name.capitalize(),
                marker="o",
                alpha=0.8,
            )

        ax.set_xlabel("Bin Index (0=best, 9=worst)")
        ax.set_ylabel("z-KID (z-normalized, higher = worse)")
        ax.set_title(f"Decile Binning: z-KID vs Ranking Mode - {config_key}")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="Real baseline (z=0)")
        ax.legend()
        ax.grid(alpha=0.3)

        plot_path_z = output_dir / f"{dataset_prefix}decile_binning_zkid_{config_key.replace('/', '_')}.png"
        fig.savefig(plot_path_z, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return df
