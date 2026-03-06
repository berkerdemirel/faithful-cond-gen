"""
Markdown report generation for trust evaluation.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


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
    task5_results: Optional[Dict] = None,
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
        task5_results: Task 5 downstream bin-selection results
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
                # Report both correlations for cross-reference with Layer 1
                report.append(
                    f"- **Spearman ρ**: {r.get('spearman_rho', np.nan):.4f} (same as Layer 1)"
                )
                report.append(f"- **Pearson ρ**: {r.get('pearson_rho', np.nan):.4f}")
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

    # Layer 3: Real vs Generated OOD Detection
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

    # Task 5: Downstream Bin-Selection Evaluation
    if task5_results:
        report.append("---\n## Task 5: Downstream Sample-Selection Evaluation\n")
        report.append(
            "*Classification accuracy by trust-score bin (16-way condition task)*\n"
        )
        report.append(
            "\n**Key Question**: Does scoring in space X help select samples for downstream task in space Y?\n"
        )

        for config_key, df in sorted(task5_results.items()):
            report.append(f"\n### {config_key}\n")
            if df is not None and not df.empty:
                # Create summary pivot table: bin_idx vs ranking_mode
                summary = (
                    df.groupby(["ranking_mode", "bin_idx"])
                    .agg(
                        {
                            "accuracy": ["mean", "std"],
                        }
                    )
                    .reset_index()
                )
                summary.columns = [
                    "ranking_mode",
                    "bin_idx",
                    "accuracy_mean",
                    "accuracy_std",
                ]

                # Highlight key findings
                for mode in ["trust", "realism", "faithfulness", "random"]:
                    mode_df = summary[summary["ranking_mode"] == mode]
                    if len(mode_df) >= 2:
                        bin0_acc = mode_df[mode_df["bin_idx"] == 0][
                            "accuracy_mean"
                        ].values
                        bin9_acc = mode_df[
                            mode_df["bin_idx"] == mode_df["bin_idx"].max()
                        ]["accuracy_mean"].values
                        if len(bin0_acc) > 0 and len(bin9_acc) > 0:
                            gap = bin0_acc[0] - bin9_acc[0]
                            report.append(
                                f"- **{mode.capitalize()}**: Bin 0 acc = {bin0_acc[0]:.4f}, Bin 9 acc = {bin9_acc[0]:.4f}, Gap = {gap:+.4f}"
                            )

                # Show trust mode table
                trust_summary = summary[summary["ranking_mode"] == "trust"]
                if not trust_summary.empty:
                    report.append("\n**Trust ranking (accuracy by bin):**\n")
                    table_df = trust_summary[
                        ["bin_idx", "accuracy_mean", "accuracy_std"]
                    ].copy()
                    table_df.columns = ["Bin", "Accuracy (mean)", "Accuracy (std)"]
                    report.append(table_df.round(4).to_markdown(index=False))

                # Extract model/feature info from dataframe
                model_name = (
                    df["model_name"].iloc[0]
                    if "model_name" in df.columns
                    else "unknown"
                )
                scoring_type = (
                    df["scoring_feature_type"].iloc[0]
                    if "scoring_feature_type" in df.columns
                    else "unknown"
                )
                downstream_type = (
                    df["downstream_feature_type"].iloc[0]
                    if "downstream_feature_type" in df.columns
                    else "unknown"
                )

                config_suffix = f"{model_name}_{scoring_type}_to_{downstream_type}"
                csv_name = f"downstream_bin_selection_{config_suffix}.csv"
                plot_name = f"downstream_bin_selection_{config_suffix}.png"
                report.append(f"\n- **Scoring space**: {scoring_type}")
                report.append(f"- **Downstream space**: {downstream_type}")
                report.append(f"- **Full CSV**: {csv_name}")
                report.append(f"- **Plot**: {plot_name}")
            report.append("")

    # Write report
    with open(output_dir / f"TRUST_EVALUATION_{dataset}.md", "w") as f:
        f.write("\n".join(report))

    print(f"\nReport saved to {output_dir / f'TRUST_EVALUATION_{dataset}.md'}")
