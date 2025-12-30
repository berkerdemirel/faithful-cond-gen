"""Ablation Script for Scorer Comparison

Runs systematic comparison of all scoring methods across:
- Different scorers (Mahalanobis, Cosine, KNN, Linear Probe, etc.)
- Different encoders (DINOv2, DINOv3, MAE, SigLIP, etc.)
- Different models (fullmodel, marginal)

Outputs comparison tables with:
- AUROC (if available)
- FPR@95 (if available)
- Spearman(score, kid_delta)
- Spearman(score, kid_rel)
- Coverage (% of conditions scored)

Usage:
    python scripts/run_ablation.py --results-dir outputs/scores --output results/ablation_report.csv

    Or use Hydra config:
    python scripts/run_ablation.py --config-name ablation
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def load_scoring_result(path: str) -> Dict:
    """Load a scoring result .pt file.

    Returns:
        Dict with keys: df, global_metrics, scorer_name, encoder_name, etc.
    """
    try:
        data = torch.load(path, map_location="cpu")
        return data
    except Exception as e:
        log.warning(f"Failed to load {path}: {e}")
        return None


def extract_metrics(data: Dict) -> Dict:
    """Extract key metrics from a scoring result.

    Returns:
        Dict with: auroc, fpr95, spearman_kid_delta, spearman_kid_rel, coverage, etc.
    """
    if data is None:
        return {}

    metrics = {}

    # Global metrics (if available)
    global_metrics = data.get("global_metrics", {})
    metrics["auroc"] = global_metrics.get("auroc", None)
    metrics["fpr95"] = global_metrics.get("fpr95", None)

    # DataFrame-based metrics
    df = data.get("df", None)
    if df is not None and isinstance(df, pd.DataFrame):
        # Spearman correlations
        if "mean_score" in df.columns and "kid_delta_mean" in df.columns:
            valid = df[["mean_score", "kid_delta_mean"]].dropna()
            if len(valid) > 2:
                corr, pval = spearmanr(valid["mean_score"], valid["kid_delta_mean"])
                metrics["spearman_kid_delta"] = corr
                metrics["spearman_kid_delta_pval"] = pval
            else:
                metrics["spearman_kid_delta"] = None
                metrics["spearman_kid_delta_pval"] = None

        if "mean_score" in df.columns and "kid_rel_mean" in df.columns:
            valid = df[["mean_score", "kid_rel_mean"]].dropna()
            if len(valid) > 2:
                corr, pval = spearmanr(valid["mean_score"], valid["kid_rel_mean"])
                metrics["spearman_kid_rel"] = corr
                metrics["spearman_kid_rel_pval"] = pval
            else:
                metrics["spearman_kid_rel"] = None
                metrics["spearman_kid_rel_pval"] = None

        # Coverage
        total_conditions = len(df)
        scored_conditions = df["mean_score"].notna().sum()
        metrics["coverage"] = scored_conditions / total_conditions if total_conditions > 0 else 0.0
        metrics["n_conditions"] = total_conditions
        metrics["n_scored"] = scored_conditions

        # Mean score statistics
        if "mean_score" in df.columns:
            scores = df["mean_score"].dropna()
            if len(scores) > 0:
                metrics["score_mean"] = scores.mean()
                metrics["score_std"] = scores.std()
                metrics["score_min"] = scores.min()
                metrics["score_max"] = scores.max()

    return metrics


def find_score_files(results_dir: str, pattern: str = "**/*_scores.pt") -> List[Path]:
    """Find all scoring result files in a directory.

    Args:
        results_dir: Directory to search
        pattern: Glob pattern for score files

    Returns:
        List of Path objects
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        log.error(f"Results directory not found: {results_dir}")
        return []

    files = list(results_path.glob(pattern))
    log.info(f"Found {len(files)} score files in {results_dir}")
    return files


def parse_filename(path: Path) -> Dict[str, str]:
    """Parse metadata from filename.

    Expected format: {dataset}_{model}_{encoder}_{scorer}_scores.pt
    Example: celeba_fullmodel_dinov2_mahalanobis_scores.pt

    Returns:
        Dict with: dataset, model, encoder, scorer
    """
    stem = path.stem  # Remove .pt
    parts = stem.split("_")

    # Try to parse
    metadata = {}
    if len(parts) >= 4:
        # Last part is "scores", second to last is scorer
        metadata["scorer"] = parts[-2]

        # Try to identify encoder (common names)
        encoder_names = ["dinov2", "dinov3", "mae", "siglip", "bioclip", "openphenom"]
        for i, part in enumerate(parts):
            if part.lower() in encoder_names:
                metadata["encoder"] = part
                # Everything before encoder is model or dataset
                if i > 0:
                    metadata["dataset"] = parts[0]
                if i > 1:
                    metadata["model"] = "_".join(parts[1:i])
                break

    # Fallback: use filename as identifier
    if "dataset" not in metadata:
        metadata["dataset"] = parts[0] if len(parts) > 0 else "unknown"
    if "model" not in metadata:
        metadata["model"] = parts[1] if len(parts) > 1 else "unknown"
    if "encoder" not in metadata:
        metadata["encoder"] = parts[2] if len(parts) > 2 else "unknown"
    if "scorer" not in metadata:
        metadata["scorer"] = parts[-1].replace("_scores", "") if len(parts) > 0 else "unknown"

    return metadata


def run_ablation(results_dir: str, output_path: str, pattern: str = "**/*_scores.pt"):
    """Run ablation study across all scoring results.

    Args:
        results_dir: Directory containing scoring results
        output_path: Path to save comparison table (CSV)
        pattern: Glob pattern for finding score files
    """
    log.info("=" * 80)
    log.info("ABLATION STUDY: Scorer Comparison")
    log.info("=" * 80)

    # Find all score files
    score_files = find_score_files(results_dir, pattern)

    if not score_files:
        log.error("No score files found. Exiting.")
        return

    # Process each file
    results = []
    for path in score_files:
        log.info(f"Processing: {path.name}")

        # Parse metadata from filename
        metadata = parse_filename(path)

        # Load data
        data = load_scoring_result(str(path))
        if data is None:
            continue

        # Extract metrics
        metrics = extract_metrics(data)

        # Combine metadata and metrics
        result = {**metadata, **metrics, "file": path.name}
        results.append(result)

    if not results:
        log.error("No valid results loaded. Exiting.")
        return

    # Create comparison DataFrame
    df = pd.DataFrame(results)

    # Reorder columns for readability
    priority_cols = [
        "dataset",
        "model",
        "encoder",
        "scorer",
        "auroc",
        "fpr95",
        "spearman_kid_delta",
        "spearman_kid_rel",
        "coverage",
        "n_conditions",
        "score_mean",
        "score_std",
    ]
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[[c for c in priority_cols if c in df.columns] + other_cols]

    # Sort by dataset, model, encoder, scorer
    sort_cols = [c for c in ["dataset", "model", "encoder", "scorer"] if c in df.columns]
    df = df.sort_values(sort_cols)

    # Save full table
    df.to_csv(output_path, index=False)
    log.info(f"\nFull results saved to: {output_path}")

    # Print summary table
    log.info("\n" + "=" * 80)
    log.info("SUMMARY TABLE")
    log.info("=" * 80)

    # Select key columns for display
    display_cols = [
        "dataset",
        "model",
        "encoder",
        "scorer",
        "auroc",
        "spearman_kid_delta",
        "spearman_kid_rel",
        "coverage",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    # Format for display
    display_df = df[display_cols].copy()
    for col in ["auroc", "spearman_kid_delta", "spearman_kid_rel", "coverage"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    print(display_df.to_string(index=False))

    # Print top performers
    log.info("\n" + "=" * 80)
    log.info("TOP PERFORMERS")
    log.info("=" * 80)

    for metric in ["auroc", "spearman_kid_delta", "spearman_kid_rel"]:
        if metric in df.columns:
            top = df.nlargest(5, metric)
            log.info(f"\nTop 5 by {metric}:")
            for idx, row in top.iterrows():
                log.info(
                    f"  {row.get('dataset', '?')}/{row.get('model', '?')}/"
                    f"{row.get('encoder', '?')}/{row.get('scorer', '?')}: {row[metric]:.4f}"
                )

    # Generate pivot tables for easier comparison
    log.info("\n" + "=" * 80)
    log.info("PIVOT TABLES")
    log.info("=" * 80)

    # Scorer vs Encoder (averaged across datasets/models)
    if all(c in df.columns for c in ["scorer", "encoder", "spearman_kid_delta"]):
        pivot = df.pivot_table(
            index="scorer", columns="encoder", values="spearman_kid_delta", aggfunc="mean"
        )
        pivot_path = output_path.replace(".csv", "_scorer_vs_encoder.csv")
        pivot.to_csv(pivot_path)
        log.info(f"\nScorer vs Encoder pivot table saved to: {pivot_path}")
        print("\nScorer vs Encoder (Spearman KID Delta):")
        print(pivot.to_string())

    log.info("\n" + "=" * 80)
    log.info(f"Ablation complete! Results saved to: {output_path}")
    log.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run ablation study for scorer comparison")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="outputs/scores",
        help="Directory containing scoring results (.pt files)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ablation_report.csv",
        help="Output path for comparison table",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*_scores.pt",
        help="Glob pattern for finding score files",
    )

    args = parser.parse_args()

    run_ablation(args.results_dir, args.output, args.pattern)


if __name__ == "__main__":
    main()
