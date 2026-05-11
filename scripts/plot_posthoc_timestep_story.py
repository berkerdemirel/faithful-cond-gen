"""
Early-abstention story for posthoc-mapped CelebA.

X-axis: denoising step k (0..249).
Twin Y-axes:
  (left)  DeltaKID% of the FPR95-selected subset vs random.
  (right) Per-step L2 change in the decoded image -- i.e. |decode(x_k) -
          decode(x_{k-1})|. This is a proxy for how much the image is
          still moving at step k; it falls monotonically as denoising
          settles but is not expected to hit zero.

Story: as denoising settles (per-step L2 shrinks) the trust-filtering
signal strengthens (DeltaKID% rises), saturating near the post-gen oracle.

Usage:
    PYTHONPATH=src uv run python scripts/plot_posthoc_timestep_story.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = Path("notes/figures/posthoc_timestep_story.png")

# (k, img_l2 [None if undefined], delta_kid_pct)
# k=0 has no image L2 (pure noise reference); we omit it from the L2 line.
ROWS = [
    (  0, None, 18.8),
    ( 27, 74.5, 15.5),
    ( 55, 33.7, 26.1),
    ( 83, 24.2, 33.0),
    (110, 19.3, 34.1),
    (138, 17.9, 35.5),
    (166, 16.7, 39.1),
    (193, 15.6, 43.6),
    (221, 14.7, 43.1),
    (248, 12.3, 39.4),
]

# Post-generation DINOv3 oracle (conceptually at the "end" of denoising)
ORACLE = dict(delta_kid_pct=39.3)

C_KID = "#cb181d"
C_L2 = "#2171b5"


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ks = np.array([r[0] for r in ROWS])
    kids = np.array([r[2] for r in ROWS])

    l2_ks = np.array([r[0] for r in ROWS if r[1] is not None])
    l2_vals = np.array([r[1] for r in ROWS if r[1] is not None])

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax2 = ax.twinx()

    # DeltaKID% (left, red)
    line_kid, = ax.plot(ks, kids, "s-", color=C_KID, lw=1.6, ms=6,
                        label=r"$\Delta$KID% (left)")

    # Image L2 (right, blue)
    line_l2, = ax2.plot(l2_ks, l2_vals, "o-", color=C_L2, lw=1.6, ms=6,
                        label=r"per-step image $\Delta$L2 (right)")

    # Oracle reference line on DeltaKID%
    ax.axhline(ORACLE["delta_kid_pct"], color=C_KID, ls="--",
               lw=1, alpha=0.5)
    ax.text(5, ORACLE["delta_kid_pct"] + 0.8,
            f"post-gen oracle +{ORACLE['delta_kid_pct']:.1f}%",
            color=C_KID, fontsize=9)

    # Axis styling
    ax.set_xlabel("denoising step $k$ (of 250)", fontsize=10)
    ax.set_ylabel(r"$\Delta$KID% (FPR95 subset vs. random)",
                  color=C_KID, fontsize=10)
    ax2.set_ylabel("per-step L2 change in decoded image",
                   color=C_L2, fontsize=10)
    ax.tick_params(axis="y", colors=C_KID)
    ax2.tick_params(axis="y", colors=C_L2)

    ax.set_xlim(-10, 260)
    ax.set_ylim(10, 50)
    ax2.set_ylim(0, 82)
    ax.grid(True, alpha=0.2)

    ax.legend(handles=[line_kid, line_l2], loc="center right", fontsize=9)
    ax.set_title(
        "CelebA stress-test, posthoc-mapped (whit-geom) scoring",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
