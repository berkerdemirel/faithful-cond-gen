"""Early-abstention story for posthoc-mapped RxRx1 (rxrx1_vanilla_marginal_v1).

Reads two CSVs (no recomputation):
  - --delta-csv:           per-step ΔKID% summary (built by aggregate_fpr95_delta_kid.py)
  - --image-distance-csv:  consecutive-step decoded-image L2 (built by analyze_timestep_image_distance.py)

X-axis: denoising step k.
Twin Y-axes:
  (left)  ΔKID% of FPR95-selected subset vs random per step;
          oracle row (is_oracle=1) drawn as a dashed horizontal reference.
  (right) per-step L2 change in the decoded image, |decode(x_k) - decode(x_{k-1})|.

Usage:
    PYTHONPATH=src uv run python scripts/plot_rxrx1_posthoc_timestep_story.py \\
        --delta-csv outputs/trust_evaluation_rxrx1_vanilla_marginal_ts/fpr95_delta_kid_per_step.csv \\
        --image-distance-csv outputs/gen/rxrx1_vanilla_marginal_timesteps/consecutive_image_distance.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

C_KID = "#cb181d"
C_L2 = "#2171b5"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--delta-csv", type=Path, required=True)
    p.add_argument("--image-distance-csv", type=Path, required=True)
    p.add_argument("--out", type=Path,
                   default=Path("notes/figures/rxrx1_posthoc_timestep_story.png"))
    p.add_argument("--title", default="RxRx1 stress-test, posthoc-mapped (whit-geom) scoring")
    args = p.parse_args()

    df = pd.read_csv(args.delta_csv)
    per_step = df[df["is_oracle"] == 0].sort_values("k")
    oracle_rows = df[df["is_oracle"] == 1]

    ks = per_step["k"].tolist()
    kids = per_step["delta_pct"].tolist()
    oracle = float(oracle_rows.iloc[0]["delta_pct"]) if len(oracle_rows) else None

    # Decoded-image L2 per consecutive step.
    img_dist = {}
    with open(args.image_distance_csv) as f:
        for row in csv.DictReader(f):
            img_dist[(int(row["k_from"]), int(row["k_to"]))] = float(row["mean_l2"])
    l2_ks, l2_vals = [], []
    prev = None
    for k in ks:
        if prev is not None and (prev, k) in img_dist:
            l2_ks.append(k)
            l2_vals.append(img_dist[(prev, k)])
        prev = k

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax2 = ax.twinx()

    line_kid, = ax.plot(ks, kids, "s-", color=C_KID, lw=1.6, ms=6,
                        label=r"$\Delta$KID% (left)")
    line_l2, = ax2.plot(l2_ks, l2_vals, "o-", color=C_L2, lw=1.6, ms=6,
                        label=r"per-step image $\Delta$L2 (right)")

    if oracle is not None:
        ax.axhline(oracle, color=C_KID, ls="--", lw=1, alpha=0.5)
        ax.text(5, oracle + 0.8,
                f"post-gen oracle {oracle:+.1f}%",
                color=C_KID, fontsize=9)

    ax.set_xlabel("denoising step $k$ (of 250)", fontsize=10)
    ax.set_ylabel(r"$\Delta$KID% (FPR95 subset vs. random)",
                  color=C_KID, fontsize=10)
    ax2.set_ylabel("per-step L2 change in decoded image",
                   color=C_L2, fontsize=10)
    ax.tick_params(axis="y", colors=C_KID)
    ax2.tick_params(axis="y", colors=C_L2)

    ax.set_xlim(-10, 260)
    if kids:
        lo = min(kids + ([oracle] if oracle is not None else [])) - 5
        hi = max(kids + ([oracle] if oracle is not None else [])) + 5
        ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.2)

    ax.legend(handles=[line_kid, line_l2], loc="center right", fontsize=9)
    ax.set_title(args.title, fontsize=11)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")
    print(f"Steps: {ks}")
    print(f"DeltaKID%: {[f'{v:+.1f}' for v in kids]}")
    print(f"Oracle: {oracle:+.1f}" if oracle is not None else "Oracle: missing")


if __name__ == "__main__":
    main()
