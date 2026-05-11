"""
Replot CP-feature decile downstream curves (best-combo CP config) with
trust / realism / faithfulness on the same axes.

Reads the existing per-model CSV
  outputs/cp_analysis/morphology_validation/decile_binning/decile_curves_{model}_cp.csv
and writes
  outputs/cp_analysis/morphology_validation/decile_binning/decile_curves_{model}_cp_components.png

Default model: repa_siglip_marginal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RANKING_STYLE = {
    "trust":        {"color": "#0072B2", "marker": "o", "label": "trust"},
    "realism":      {"color": "#009E73", "marker": "s", "label": "realism"},
    "faithfulness": {"color": "#D55E00", "marker": "^", "label": "faithfulness"},
}


def _plot(df: pd.DataFrame, model: str, out_path: Path, n_bins: int = 10) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    targets = [("celltype", "cell-type (4-way)"),
               ("subset",   "subset / combo (50-way)")]
    for ax, (target, pretty) in zip(axes, targets):
        # Random ±1σ band first (in background).
        rand = df[(df.target == target) & (df.ranking == "random")]
        if not rand.empty:
            agg = rand.groupby("bin_idx")["micro"].agg(["mean", "std"]).reset_index()
            ax.fill_between(agg["bin_idx"],
                            agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                            color="#888888", alpha=0.18, linewidth=0,
                            label="random ±1σ")
            ax.plot(agg["bin_idx"], agg["mean"],
                    color="#888888", linestyle="--", linewidth=1.2, alpha=0.8,
                    label="random mean")

        for rname, style in RANKING_STYLE.items():
            sub = df[(df.target == target) & (df.ranking == rname)].sort_values("bin_idx")
            if sub.empty:
                continue
            ax.plot(sub["bin_idx"], sub["micro"],
                    marker=style["marker"], color=style["color"],
                    linestyle="-", linewidth=2.0, markersize=6.5,
                    label=style["label"])

        ax.set_xlabel("Decile bin  (0 = best rank, 9 = worst)")
        ax.set_ylabel("Accuracy (gen-trained classifier, micro, on real test)")
        ax.set_title(f"{model} — {pretty}  (within-condition bins, SigLIP scoring)")
        ax.grid(linestyle=":", alpha=0.3)
        ax.set_xticks(range(n_bins))
        ax.legend(fontsize=9, loc="best", frameon=True)

    fig.suptitle(
        f"{model} — CP-feature decile curves: trust / realism / faithfulness "
        "(kept-621 all-621)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path,
                   default=Path("outputs/cp_analysis/morphology_validation/"
                                "decile_binning/decile_curves_repa_siglip_marginal_cp.csv"))
    p.add_argument("--model", type=str, default="repa_siglip_marginal")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    out_path = args.out or args.csv.with_name(args.csv.stem + "_components.png")
    _plot(df, args.model, out_path)


if __name__ == "__main__":
    main()
