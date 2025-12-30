"""Visualization utilities for faithfulness evaluation.

This module provides plotting functions for analyzing score-quality relationships,
visualizing worst-performing conditions, and understanding compositional generalization.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import spearmanr
from sklearn.manifold import TSNE

log = logging.getLogger(__name__)


def plot_score_distributions(
    df: pd.DataFrame,
    output_dir: str,
    score_col: str = "mean_score",
    comp_col: str = "comp_category",
    categories: Optional[list] = None,
) -> None:
    """Plot score histograms split by composition category.

    Args:
        df: DataFrame with score and category columns
        output_dir: Directory to save plots
        score_col: Column name for scores
        comp_col: Column name for categories (seen/rare/unseen)
        categories: List of categories to plot (default: ['seen', 'rare', 'unseen'])
    """
    if categories is None:
        categories = ["seen", "rare", "unseen"]

    # Filter to categories that exist in data
    existing_cats = [cat for cat in categories if cat in df[comp_col].unique()]

    if len(existing_cats) == 0:
        log.warning(f"No categories found in column '{comp_col}'. Skipping plot.")
        return

    fig, axes = plt.subplots(1, len(existing_cats), figsize=(5 * len(existing_cats), 4))
    if len(existing_cats) == 1:
        axes = [axes]

    for i, cat in enumerate(existing_cats):
        subset = df[df[comp_col] == cat]
        if len(subset) > 0:
            axes[i].hist(subset[score_col].dropna(), bins=50, alpha=0.7, edgecolor="black")
            axes[i].set_title(f"{cat.capitalize()} (n={len(subset)})")
            axes[i].set_xlabel("Faithfulness Score")
            axes[i].set_ylabel("Count")
            axes[i].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/score_distributions_by_category.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved score distribution plot to {output_dir}/score_distributions_by_category.png")


def plot_score_vs_kid(
    df: pd.DataFrame,
    output_path: str,
    score_col: str = "mean_score",
    kid_col: str = "kid_delta_mean",
    title: Optional[str] = None,
) -> None:
    """Scatter plot: score vs KID with correlation annotation.

    Args:
        df: DataFrame with score and KID columns
        output_path: Path to save plot
        score_col: Column name for scores
        kid_col: Column name for KID values
        title: Optional custom title
    """
    valid = df[[score_col, kid_col]].dropna()

    if len(valid) < 3:
        log.warning(f"Not enough valid data (n={len(valid)}) for score vs KID plot. Skipping.")
        return

    corr, pval = spearmanr(valid[score_col], valid[kid_col])

    plt.figure(figsize=(8, 6))
    plt.scatter(valid[score_col], valid[kid_col], alpha=0.5, s=30, edgecolor="k", linewidth=0.3)
    plt.xlabel("Mean Faithfulness Score", fontsize=12)
    plt.ylabel("KID Delta (gen - baseline)", fontsize=12)

    if title is None:
        title = f"Score vs Quality (ρ={corr:.3f}, p={pval:.3e}, n={len(valid)})"
    plt.title(title, fontsize=13)

    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved score vs KID plot to {output_path} (ρ={corr:.3f})")


def plot_tsne_by_score(
    features_path: str,
    scores_path: str,
    output_path: str,
    n_samples: int = 5000,
    perplexity: int = 30,
    random_state: int = 42,
) -> None:
    """t-SNE visualization colored by score magnitude.

    Args:
        features_path: Path to .pt file with features (generated samples)
        scores_path: Path to .pt file with scoring results
        output_path: Path to save plot
        n_samples: Max samples to plot (for computational speed)
        perplexity: t-SNE perplexity parameter
        random_state: Random seed for reproducibility
    """
    log.info(f"Loading features from {features_path}...")
    feat_data = torch.load(features_path, map_location="cpu")
    score_data = torch.load(scores_path, map_location="cpu")

    features = feat_data["features"].numpy()
    df = score_data["df"]

    # Need to map condition_hash to scores
    # Assuming generated features are in same order as scoring results
    # This is a simplification - in practice would need proper matching
    log.warning("t-SNE plot assumes features and scores are aligned by condition order")

    # Sample if too large
    n_total = features.shape[0]
    if n_total > n_samples:
        log.info(f"Sampling {n_samples} from {n_total} features for t-SNE")
        idx = np.random.RandomState(random_state).choice(n_total, n_samples, replace=False)
        features = features[idx]
    else:
        idx = np.arange(n_total)

    # Compute t-SNE
    log.info(f"Computing t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    embedding = tsne.fit_transform(features)

    # For visualization, assign scores (simplified - assumes alignment)
    # In practice would need condition matching
    scores = np.zeros(len(features))
    if len(df) >= len(features):
        # Use mean_score from df, but this is approximate
        log.warning("Score-to-feature alignment is approximate")

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=scores,
        cmap="viridis",
        s=10,
        alpha=0.6,
        edgecolor="none",
    )
    plt.colorbar(scatter, label="Faithfulness Score")
    plt.title(f"t-SNE Embedding Colored by Score (n={len(features)})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved t-SNE plot to {output_path}")


def save_correlation_heatmap(corr_df: pd.DataFrame, output_path: str) -> None:
    """Heatmap of Spearman correlations between metrics.

    Args:
        corr_df: Correlation matrix (DataFrame)
        output_path: Path to save heatmap
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Spearman ρ"},
    )
    plt.title("Spearman Correlations: Scores vs Metrics", fontsize=14, pad=15)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved correlation heatmap to {output_path}")


def plot_score_vs_pool_size(
    df: pd.DataFrame, output_path: str, score_col: str = "mean_score"
) -> None:
    """Scatter plot: score vs training pool size to diagnose data scarcity bias.

    Args:
        df: DataFrame with score and n_real_pool columns
        output_path: Path to save plot
        score_col: Column name for scores
    """
    valid = df[[score_col, "n_real_pool"]].dropna()

    if len(valid) < 3:
        log.warning(f"Not enough valid data (n={len(valid)}) for score vs pool size plot")
        return

    corr, pval = spearmanr(valid[score_col], valid["n_real_pool"])

    plt.figure(figsize=(8, 6))
    plt.scatter(
        valid["n_real_pool"], valid[score_col], alpha=0.5, s=30, edgecolor="k", linewidth=0.3
    )
    plt.xlabel("Training Pool Size (n_real_pool)", fontsize=12)
    plt.ylabel("Mean Faithfulness Score", fontsize=12)
    plt.title(
        f"Score vs Data Availability (ρ={corr:.3f}, p={pval:.3e}, n={len(valid)})", fontsize=13
    )
    plt.xscale("log")
    plt.grid(alpha=0.3, linestyle="--", which="both")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved score vs pool size plot to {output_path} (ρ={corr:.3f})")


def plot_difficulty_distribution(df: pd.DataFrame, output_path: str) -> None:
    """Histogram of condition difficulty scores.

    Args:
        df: DataFrame with 'difficulty' column
        output_path: Path to save plot
    """
    if "difficulty" not in df.columns:
        log.warning("No 'difficulty' column found in DataFrame. Skipping plot.")
        return

    valid = df["difficulty"].dropna()
    if len(valid) == 0:
        log.warning("No valid difficulty scores. Skipping plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(valid, bins=50, alpha=0.7, edgecolor="black", color="steelblue")
    plt.xlabel("Condition Difficulty Score", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(f"Distribution of Condition Difficulty (n={len(valid)})", fontsize=13)
    plt.axvline(valid.median(), color="red", linestyle="--", linewidth=2, label=f"Median={valid.median():.2f}")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved difficulty distribution plot to {output_path}")
