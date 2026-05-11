"""
Method overview figure (Figure 1) for the paper.

Pipeline schematic showing offline fitting + online scoring + REPA early abstention.

Usage:
    PYTHONPATH=src uv run python scripts/plot_method_overview.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_PATH = "notes/figures/method_overview.png"
OUT_PATH_PDF = "notes/figures/method_overview.pdf"

# Colors
C_BG = "#f7f7f7"
C_OFFLINE = "#d4e6f1"      # light blue
C_ONLINE = "#fdebd0"       # light orange
C_REALISM = "#2171b5"      # blue
C_FAITHFUL = "#e67e22"     # orange
C_TRUST = "#27ae60"        # green
C_REJECT = "#cb181d"       # red
C_ACCEPT = "#2ca02c"       # green
C_REPA = "#8e44ad"         # purple
C_BOX = "#2c3e50"          # dark
C_DATA = "#7fb3d8"         # medium blue
C_MODEL = "#f0b27a"        # medium orange


def rounded_box(ax, xy, w, h, text, facecolor="white", edgecolor=C_BOX,
                fontsize=8, fontweight="normal", text_color="black",
                alpha=1.0, linestyle="-", linewidth=1.0, zorder=2):
    """Draw a rounded rectangle with centered text."""
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight=fontweight, color=text_color, zorder=zorder + 1,
    )
    return box


def arrow(ax, xy_from, xy_to, color=C_BOX, style="-|>", lw=1.2,
          connectionstyle="arc3,rad=0", linestyle="-", zorder=1):
    """Draw an arrow between two points."""
    arr = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=style, color=color,
        linewidth=lw, connectionstyle=connectionstyle,
        linestyle=linestyle, zorder=zorder,
        mutation_scale=12,
    )
    ax.add_patch(arr)
    return arr


def main():
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.8))
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-0.5, 5.2)
    ax.set_aspect("equal")
    ax.axis("off")

    bw, bh = 1.3, 0.5  # box width, height
    bw_sm = 1.0

    # ===== OFFLINE PHASE (top) =====
    # Background
    offline_bg = FancyBboxPatch(
        (-0.15, 2.7), 10.3, 2.35,
        boxstyle="round,pad=0.1", facecolor=C_OFFLINE, edgecolor="none", alpha=0.35, zorder=0,
    )
    ax.add_patch(offline_bg)
    ax.text(0.15, 4.85, "Offline: Fit scoring models on real training data",
            fontsize=9, fontweight="bold", color="#2c3e50", style="italic", zorder=5)

    # Real data
    rounded_box(ax, (0.2, 3.7), bw, bh, "Real data\n(training support)",
                facecolor=C_DATA, fontsize=7.5)

    # Encoder
    rounded_box(ax, (2.2, 3.7), bw_sm, bh, "Encoder\n$h(x)$",
                facecolor="white", fontsize=7.5)

    # L2 normalize
    rounded_box(ax, (3.8, 3.7), bw_sm, bh, "$\\ell_2$-norm\n$y = h/\\|h\\|$",
                facecolor="white", fontsize=7.5)

    # Realism model
    rounded_box(ax, (5.6, 4.2), 1.8, 0.5,
                "Global Gaussian\n$\\mathcal{N}(\\mu_{\\mathrm{real}}, \\Sigma_{\\mathrm{real}})$",
                facecolor="#d6eaf8", edgecolor=C_REALISM, fontsize=7, linewidth=1.5)

    # Faithfulness model
    rounded_box(ax, (5.6, 3.15), 1.8, 0.5,
                "Per-attribute prototypes\n$\\eta_{k,v},\\ P_k$ (shared $\\Sigma$)",
                facecolor="#fdebd0", edgecolor=C_FAITHFUL, fontsize=7, linewidth=1.5)

    # Arrows: offline flow
    arrow(ax, (1.5, 3.95), (2.2, 3.95))
    arrow(ax, (3.2, 3.95), (3.8, 3.95))
    arrow(ax, (4.8, 4.1), (5.6, 4.4), connectionstyle="arc3,rad=-0.15")
    arrow(ax, (4.8, 3.8), (5.6, 3.5), connectionstyle="arc3,rad=0.15")

    # Stored reference box (spans both models)
    rounded_box(ax, (8.2, 3.1), 1.6, 1.3, "Stored\nreference\nstatistics",
                facecolor="white", edgecolor=C_BOX, fontsize=8,
                linestyle="--", linewidth=1.2)

    # Arrows from models to stored
    arrow(ax, (7.4, 4.45), (8.2, 4.1), color=C_REALISM, style="-|>", lw=1,
          connectionstyle="arc3,rad=0.15")
    arrow(ax, (7.4, 3.4), (8.2, 3.55), color=C_FAITHFUL, style="-|>", lw=1,
          connectionstyle="arc3,rad=-0.15")

    # ===== ONLINE PHASE (bottom) =====
    online_bg = FancyBboxPatch(
        (-0.15, -0.35), 10.3, 2.85,
        boxstyle="round,pad=0.1", facecolor=C_ONLINE, edgecolor="none", alpha=0.3, zorder=0,
    )
    ax.add_patch(online_bg)
    ax.text(0.15, 2.3, "Online: Score each generated sample",
            fontsize=9, fontweight="bold", color="#2c3e50", style="italic", zorder=5)

    # Condition request
    rounded_box(ax, (0.2, 1.2), bw_sm, bh, "Condition\n$a^\\star$",
                facecolor="#e8daef", fontsize=7.5, fontweight="bold")

    # Diffusion model
    rounded_box(ax, (1.7, 1.2), 1.3, bh, "Conditional\ndiffusion",
                facecolor=C_MODEL, fontsize=7.5)

    # Generated sample
    rounded_box(ax, (3.5, 1.2), bw_sm, bh, "Generated\nsample $x$",
                facecolor="white", fontsize=7.5)

    # Encoder (online)
    rounded_box(ax, (5.0, 1.2), bw_sm, bh, "Encoder\n$y(x)$",
                facecolor="white", fontsize=7.5)

    # Scoring
    # Realism score
    rounded_box(ax, (6.5, 1.65), 1.1, 0.42, "$R(y)$",
                facecolor="#d6eaf8", edgecolor=C_REALISM, fontsize=8.5,
                fontweight="bold", linewidth=1.5)

    # Faithfulness score
    rounded_box(ax, (6.5, 0.85), 1.1, 0.42, "$F(y; a^\\star)$",
                facecolor="#fdebd0", edgecolor=C_FAITHFUL, fontsize=8.5,
                fontweight="bold", linewidth=1.5)

    # Trust score
    rounded_box(ax, (8.2, 1.15), 1.0, 0.6, "$T = R + F$",
                facecolor="#d5f5e3", edgecolor=C_TRUST, fontsize=9,
                fontweight="bold", linewidth=1.8)

    # Decision
    # Accept
    ax.text(9.8, 1.7, "Accept", fontsize=8, fontweight="bold",
            color=C_ACCEPT, ha="center", va="center")
    ax.text(9.8, 1.0, "Reject", fontsize=8, fontweight="bold",
            color=C_REJECT, ha="center", va="center")

    # Arrows: online flow
    arrow(ax, (1.2, 1.45), (1.7, 1.45))
    arrow(ax, (3.0, 1.45), (3.5, 1.45))
    arrow(ax, (4.5, 1.45), (5.0, 1.45))
    arrow(ax, (6.0, 1.6), (6.5, 1.8), connectionstyle="arc3,rad=-0.1")
    arrow(ax, (6.0, 1.3), (6.5, 1.1), connectionstyle="arc3,rad=0.1")
    arrow(ax, (7.6, 1.75), (8.2, 1.55), connectionstyle="arc3,rad=0.1")
    arrow(ax, (7.6, 1.06), (8.2, 1.35), connectionstyle="arc3,rad=-0.1")

    # Arrow from trust to accept/reject
    arrow(ax, (9.2, 1.55), (9.55, 1.65), color=C_ACCEPT, lw=1.5)
    arrow(ax, (9.2, 1.35), (9.55, 1.1), color=C_REJECT, lw=1.5)

    # Threshold line
    ax.text(9.65, 1.38, "$T \\lessgtr \\kappa$", fontsize=7, color="#555",
            ha="center", va="center")

    # Arrow from stored reference down to scoring
    arrow(ax, (9.0, 3.1), (9.0, 2.6), color=C_BOX, style="-|>", lw=1.0,
          linestyle="--")

    # ===== REPA SHORTCUT =====
    # Dashed purple arrow from inside diffusion to scoring (early abstention)
    rounded_box(ax, (2.0, 0.15), 1.6, 0.42, "REPA features\n$y_{\\ell,\\tau}(x)$",
                facecolor="#e8daef", edgecolor=C_REPA, fontsize=7,
                fontweight="bold", linewidth=1.5, linestyle="--")

    # Arrow from diffusion down to REPA
    arrow(ax, (2.35, 1.2), (2.35, 0.57), color=C_REPA, lw=1.2, linestyle="--")

    # Arrow from REPA to scoring (skip encoder)
    arrow(ax, (3.6, 0.36), (6.5, 0.95), color=C_REPA, lw=1.2, linestyle="--",
          connectionstyle="arc3,rad=-0.1")

    ax.text(4.9, 0.25, "Early abstention\n(skip encoder + remaining steps)",
            fontsize=6.5, color=C_REPA, ha="center", va="center",
            fontweight="bold", style="italic")

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_PATH, dpi=250, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PATH_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")
    print(f"Saved: {OUT_PATH_PDF}")


if __name__ == "__main__":
    main()
