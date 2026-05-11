"""Per-condition FPR95 analysis scatter plot for CelebA stress-test model."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from ast import literal_eval

# Seen conditions (single-attribute or all-zero)
SEEN = {(0,0,0,0), (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)}

def load_and_prepare(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["kid_improvement_pct"])
    df["cond_tuple"] = df["condition"].apply(literal_eval)
    df["is_seen"] = df["cond_tuple"].apply(lambda t: t in SEEN)
    return df

def plot_panel(ax, df, title):
    seen = df[df["is_seen"]]
    unseen = df[~df["is_seen"]]

    # Spearman correlation
    rho, pval = spearmanr(df["acceptance_rate"], df["kid_improvement_pct"])

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)

    ax.scatter(
        unseen["acceptance_rate"], unseen["kid_improvement_pct"],
        c="#d62728", s=40, edgecolors="k", linewidths=0.4, label="Unseen", zorder=3,
    )
    ax.scatter(
        seen["acceptance_rate"], seen["kid_improvement_pct"],
        c="#1f77b4", s=40, edgecolors="k", linewidths=0.4, label="Seen", zorder=3,
    )

    ax.set_xlabel("Acceptance rate", fontsize=11)
    ax.set_ylabel("KID improvement (%)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=10)

    # Legend with Spearman rho
    pstr = f"$p$={pval:.2g}" if pval >= 0.001 else f"$p$<0.001"
    ax.legend(
        title=f"Spearman $\\rho$={rho:.2f} ({pstr})",
        title_fontsize=9, fontsize=9, loc="best", frameon=True, edgecolor="none",
        facecolor="white", framealpha=0.8,
    )

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


dinov3 = load_and_prepare(
    "outputs/trust_evaluation_celeba_v5/celeba_fpr95_per_condition_repa_marginal_dinov3.csv"
)
aligned = load_and_prepare(
    "outputs/trust_evaluation_celeba_v5/celeba_fpr95_per_condition_repa_marginal_aligned_mean.csv"
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

plot_panel(ax1, dinov3, "DINOv3 scoring")
plot_panel(ax2, aligned, "Aligned scoring")

# Shared y-axis range
ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ax1.set_ylim(ymin, ymax)
ax2.set_ylim(ymin, ymax)

fig.tight_layout()

fig.savefig("notes/figures/celeba_fpr95_per_condition_analysis.png", dpi=300, bbox_inches="tight")
fig.savefig("notes/figures/celeba_fpr95_per_condition_analysis.pdf", bbox_inches="tight")
print("Saved figures.")
