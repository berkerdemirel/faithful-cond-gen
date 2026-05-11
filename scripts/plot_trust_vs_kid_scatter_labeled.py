"""
Labeled variant of the trust-vs-KID scatter for CelebA.

Annotates each point with its 4-bit attribute code (M S B E), so one can
read off 0000 (reference) in the bottom-left and 1111 (hardest) in the
top-right. Intended for the slides.

Usage:
    PYTHONPATH=src uv run python scripts/plot_trust_vs_kid_scatter_labeled.py
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CSV_PATH = Path(
    "outputs/trust_evaluation_celeba_v7/celeba_full_ranking_repa_marginal_dinov3.csv"
)
OUT_DIR = Path("notes/figures")
OUT_PATH = OUT_DIR / "trust_vs_kid_scatter_celeba_repa_marginal_dinov3_labeled.png"

C_SEEN = "#2171b5"
C_UNSEEN = "#cb181d"

# Per-code label placement: (dx, dy) in data units. For crowded points we
# push farther and draw a thin leader line.
LABEL_OFFSETS = {
    # Low-trust cluster
    "0100": (0.08, -0.075),
    "0010": (0.00, 0.055),    # above the point
    "0000": (0.08, -0.075),
    # Trust ~0.9 (1000 below, 0001 above slightly left)
    "1000": (0.00, -0.075),   # below the point
    "0001": (-0.15, 0.075),   # above, slightly left
    # Trust ~1.1-1.3
    "0110": (-0.20, 0.055),   # top-left of the point
    "0011": (0.25, -0.135),   # below with leader, shifted right
    "0101": (0.50, -0.075),   # below-right with leader, shifted right
    # Middle cluster (~2.5-3.0)
    "1100": (-0.35, 0.035),   # closer
    "0111": (-0.15, -0.060),  # slightly right
    "1010": (0.15, -0.075),
    "1001": (0.18, 0.035),
    # Upper-right cluster
    "1101": (0.18, 0.045),
    "1110": (-0.20, 0.000),   # slightly right (closer to point)
    "1011": (0.22, -0.015),
    "1111": (-0.25, -0.015),  # closer
}

# Codes for which we draw a thin leader line to the marker.
LEADER_CODES = {"0001", "0011", "0101"}


def parse_code(cond_str: str) -> str:
    vals = re.findall(r"=(\d+)", cond_str)
    return "".join(vals)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH).dropna(subset=["delta_kid"]).copy()
    df["code"] = df["condition_str"].map(parse_code)

    trust = df["trust_mean"].values
    kid = df["delta_kid"].values
    rho, _ = spearmanr(trust, kid)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.0))

    seen_mask = df["seen_unseen"].values == "seen"
    unseen_mask = ~seen_mask

    ax.scatter(
        trust[seen_mask], kid[seen_mask],
        c=C_SEEN, s=55, alpha=0.85, edgecolors="white", linewidths=0.6,
        label=f"Seen ({seen_mask.sum()})", zorder=3,
    )
    ax.scatter(
        trust[unseen_mask], kid[unseen_mask],
        c=C_UNSEEN, s=55, alpha=0.85, edgecolors="white", linewidths=0.6,
        marker="^", label=f"Unseen ({unseen_mask.sum()})", zorder=3,
    )

    # Trend line
    z = np.polyfit(trust, kid, 1)
    x_line = np.linspace(trust.min(), trust.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="gray",
            alpha=0.5, linewidth=1, zorder=1)

    # Per-point code annotations
    for _, row in df.iterrows():
        code = row["code"]
        dx, dy = LABEL_OFFSETS.get(code, (0.20, -0.02))
        kwargs = dict(
            xy=(row["trust_mean"], row["delta_kid"]),
            xytext=(row["trust_mean"] + dx, row["delta_kid"] + dy),
            fontsize=9, fontfamily="monospace",
            color="black", zorder=4,
            ha="center", va="center",
        )
        if code in LEADER_CODES:
            kwargs["arrowprops"] = dict(
                arrowstyle="-", color="gray", lw=0.5, alpha=0.7,
                shrinkA=0, shrinkB=4,
            )
        ax.annotate(code, **kwargs)

    ax.set_xlabel("Mean trust score", fontsize=11)
    ax.set_ylabel(r"$\Delta$KID", fontsize=11)
    ax.set_title("CelebA stress-test (REPA, DINOv3): code = (M,S,B,E)",
                 fontsize=11)

    ax.text(
        0.05, 0.95, f"$\\rho = {rho:.2f}$\n$n = {len(df)}$",
        transform=ax.transAxes, fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH} (rho={rho:.4f}, n={len(df)})")


if __name__ == "__main__":
    main()
