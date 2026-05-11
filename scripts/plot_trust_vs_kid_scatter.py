"""
Per-condition scatter plots: trust score vs DeltaKID.

Creates scatter plots colored by seen/unseen status for the main paper.
For RxRx1, conditions labeled 'unknown' are mapped to seen/unseen via
the held-out pairs config.

Usage:
    PYTHONPATH=src uv run python scripts/plot_trust_vs_kid_scatter.py
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from faithful_cond_gen.eval.trust_eval.config import (
    MARGINAL_SEEN_COMBOS,
    RXRX1_HELDOUT_PAIRS,
)

CELEBA_DIR = Path("outputs/trust_evaluation_celeba_v3")
RXRX1_DIR = Path("outputs/trust_evaluation_rxrx1_v3")
OUT_DIR = Path("notes/figures")

# Colors
C_SEEN = "#2171b5"
C_UNSEEN = "#cb181d"

# Configs to plot: (dataset, base_dir, config, title_suffix)
PLOT_CONFIGS = [
    ("celeba", CELEBA_DIR, "repa_marginal_dinov3", "CelebA Stress-Test (REPA, DINOv3)"),
    ("celeba", CELEBA_DIR, "vanilla_marginal_dinov3", "CelebA Stress-Test (Vanilla, DINOv3)"),
    ("rxrx1", RXRX1_DIR, "repa_marginal_dinov3", "RxRx1 Pair-Heldout (REPA, DINOv3)"),
    ("rxrx1", RXRX1_DIR, "repa_openphenom_marginal_dinov3", "RxRx1 Pair-Heldout (REPA-OpenPhenom, DINOv3)"),
]


def parse_celeba_condition(cond_str: str) -> tuple:
    vals = re.findall(r"=(\d+)", cond_str)
    return tuple(int(v) for v in vals)


def parse_rxrx1_condition(cond_str: str) -> tuple:
    vals = re.findall(r"=(\d+)", cond_str)
    return tuple(int(v) for v in vals)


def resolve_seen_unseen(df: pd.DataFrame, dataset: str) -> pd.Series:
    """Resolve seen/unseen labels, including for RxRx1 'unknown' conditions."""
    labels = df["seen_unseen"].copy()

    if dataset == "celeba":
        # Already labeled correctly
        return labels

    if dataset == "rxrx1":
        # Map 'unknown' to seen/unseen using held-out pairs
        for i, row in df.iterrows():
            if row["seen_unseen"] == "unknown":
                pair = parse_rxrx1_condition(row["condition_str"])
                if pair in RXRX1_HELDOUT_PAIRS:
                    labels.at[i] = "unseen"
                else:
                    labels.at[i] = "seen"
        return labels

    return labels


def plot_scatter(df: pd.DataFrame, dataset: str, config: str, title: str, out_path: Path):
    """Create a single scatter plot of trust vs delta_kid."""
    valid = df.dropna(subset=["delta_kid"]).copy()
    valid["label"] = resolve_seen_unseen(valid, dataset)

    trust = valid["trust_mean"].values
    kid = valid["delta_kid"].values
    rho, p = spearmanr(trust, kid)

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))

    seen_mask = valid["label"] == "seen"
    unseen_mask = valid["label"] == "unseen"

    if seen_mask.any():
        ax.scatter(
            trust[seen_mask], kid[seen_mask],
            c=C_SEEN, s=40, alpha=0.7, edgecolors="white", linewidths=0.5,
            label=f"Seen ({seen_mask.sum()})", zorder=3,
        )
    if unseen_mask.any():
        ax.scatter(
            trust[unseen_mask], kid[unseen_mask],
            c=C_UNSEEN, s=40, alpha=0.7, edgecolors="white", linewidths=0.5,
            label=f"Unseen ({unseen_mask.sum()})", zorder=3,
            marker="^",
        )

    # Trend line
    z = np.polyfit(trust, kid, 1)
    x_line = np.linspace(trust.min(), trust.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", alpha=0.5, linewidth=1)

    ax.set_xlabel("Mean Trust Score", fontsize=11)
    ax.set_ylabel(r"$\Delta$KID", fontsize=11)
    ax.set_title(title, fontsize=11)

    # Rho annotation
    ax.text(
        0.05, 0.95, f"$\\rho = {rho:.2f}$\n$n = {len(valid)}$",
        transform=ax.transAxes, fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path} (rho={rho:.4f}, n={len(valid)})")


def plot_combined_panel():
    """Create a 2-panel figure: CelebA (left) + RxRx1 (right) for the main paper."""
    configs = [
        ("celeba", CELEBA_DIR, "repa_marginal_dinov3", "CelebA Stress-Test"),
        ("rxrx1", RXRX1_DIR, "repa_marginal_dinov3", "RxRx1 Pair-Heldout"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    for ax, (dataset, base_dir, config, title) in zip(axes, configs):
        csv_path = base_dir / f"{dataset}_full_ranking_{config}.csv"
        df = pd.read_csv(csv_path)
        valid = df.dropna(subset=["delta_kid"]).copy()
        valid["label"] = resolve_seen_unseen(valid, dataset)

        trust = valid["trust_mean"].values
        kid = valid["delta_kid"].values
        rho, _ = spearmanr(trust, kid)

        seen_mask = valid["label"] == "seen"
        unseen_mask = valid["label"] == "unseen"

        if seen_mask.any():
            ax.scatter(
                trust[seen_mask], kid[seen_mask],
                c=C_SEEN, s=40, alpha=0.7, edgecolors="white", linewidths=0.5,
                label=f"Seen ({seen_mask.sum()})", zorder=3,
            )
        if unseen_mask.any():
            ax.scatter(
                trust[unseen_mask], kid[unseen_mask],
                c=C_UNSEEN, s=40, alpha=0.7, edgecolors="white", linewidths=0.5,
                label=f"Unseen ({unseen_mask.sum()})", zorder=3,
                marker="^",
            )

        # Trend line
        z = np.polyfit(trust, kid, 1)
        x_line = np.linspace(trust.min(), trust.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", alpha=0.5, linewidth=1)

        ax.set_xlabel("Mean Trust Score", fontsize=10)
        ax.set_ylabel(r"$\Delta$KID", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.text(
            0.05, 0.95, f"$\\rho = {rho:.2f}$\n$n = {len(valid)}$",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out_path = OUT_DIR / "trust_vs_kid_scatter_combined.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved combined: {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Individual plots
    for dataset, base_dir, config, title in PLOT_CONFIGS:
        csv_path = base_dir / f"{dataset}_full_ranking_{config}.csv"
        if not csv_path.exists():
            print(f"  SKIP {config}: file not found")
            continue
        out_path = OUT_DIR / f"trust_vs_kid_scatter_{dataset}_{config}.png"
        df = pd.read_csv(csv_path)
        plot_scatter(df, dataset, config, title, out_path)

    # Combined 2-panel for main paper
    print()
    plot_combined_panel()


if __name__ == "__main__":
    main()
