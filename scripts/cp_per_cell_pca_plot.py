"""
Per-cell-type PCA scatter — where do real vs accepted vs rejected gen sit
in the cell-conditioned feature geometry?

For each cell type:
  1. Fit PCA(2) on real rows of that cell type (both arms combined).
  2. Project real, accepted gen (trust_updated ≤ t95), rejected gen through it.
  3. Render two panels (per arm), each filtered to the sirnas in that arm
     for this cell. Color = sirna; shape = {real: 'o', accepted: '^',
     rejected: 'x'}.

Output layout: one figure per (model × arm), 2×2 grid of the 4 cell types.

Reuses loaders from cp_morphology_validation.py (so --feature-space can be
cp / siglip / dinov3 without duplicating join logic).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from analyze_cp_features_rxrx1 import (  # noqa: E402
    MODEL_TO_CP_DIR,
    _load_feature_artifacts,
    resolve_models,
)
from cp_morphology_validation import (  # noqa: E402
    DEFAULT_MODELS,
    _load_gen_in_space,
    _load_real_in_space,
    _load_top_k_features,
)
from faithful_cond_gen.eval.trust_eval.subset_io import load_rxrx1_subset_arms  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


MARKER_STYLE = {
    "real":     {"marker": "o", "size": 34, "alpha": 0.80, "edgecolor": "black", "lw": 0.4},
    "accepted": {"marker": "^", "size": 30, "alpha": 0.72, "edgecolor": "black", "lw": 0.3},
    "rejected": {"marker": "x", "size": 22, "alpha": 0.40, "edgecolor": "none",  "lw": 0.0},
}


def _plot_cell_panel(
    ax,
    Z_real: np.ndarray, real_sirnas: np.ndarray,
    Z_gen_arm: np.ndarray, gen_sirnas_arm: np.ndarray, gen_accepted_arm: np.ndarray,
    sirnas_in_arm: List[int],
    evr: np.ndarray,
    cell_id: int, arm_name: str,
    n_real: int, n_acc: int, n_rej: int,
    pc_indices: Tuple[int, int] = (0, 1),
) -> None:
    pcx, pcy = pc_indices
    # Distinct-color palette. tab10 handles up to 10 sirnas per panel; tab20 for more.
    cmap = plt.get_cmap("tab10" if len(sirnas_in_arm) <= 10 else "tab20")
    color_of: Dict[int, Tuple[float, float, float, float]] = {
        s: cmap(i % cmap.N) for i, s in enumerate(sirnas_in_arm)
    }

    # Draw rejected first (faded, so accepted / real overlay on top).
    for s in sirnas_in_arm:
        c = color_of[s]
        m = (gen_sirnas_arm == s) & (~gen_accepted_arm)
        if m.any():
            st = MARKER_STYLE["rejected"]
            ax.scatter(Z_gen_arm[m, pcx], Z_gen_arm[m, pcy],
                       c=[c], marker=st["marker"], s=st["size"],
                       alpha=st["alpha"], linewidths=st["lw"])
        m = (gen_sirnas_arm == s) & (gen_accepted_arm)
        if m.any():
            st = MARKER_STYLE["accepted"]
            ax.scatter(Z_gen_arm[m, pcx], Z_gen_arm[m, pcy],
                       c=[c], marker=st["marker"], s=st["size"],
                       alpha=st["alpha"], edgecolors=st["edgecolor"], linewidths=st["lw"])
        m = (real_sirnas == s)
        if m.any():
            st = MARKER_STYLE["real"]
            ax.scatter(Z_real[m, pcx], Z_real[m, pcy],
                       c=[c], marker=st["marker"], s=st["size"],
                       alpha=st["alpha"], edgecolors=st["edgecolor"], linewidths=st["lw"])

    ax.set_xlabel(f"PC{pcx+1} ({evr[pcx]*100:.1f}%)")
    ax.set_ylabel(f"PC{pcy+1} ({evr[pcy]*100:.1f}%)")
    ax.set_title(
        f"cell {cell_id} — {arm_name} arm  "
        f"(real n={n_real}  acc={n_acc}  rej={n_rej})",
        fontsize=10,
    )
    ax.grid(linestyle=":", alpha=0.3)

    # Two-legend setup: sirna colors + marker-shape meanings.
    sirna_handles = [
        plt.Line2D([], [], marker="o", color="none",
                   markerfacecolor=color_of[s], markeredgecolor="black",
                   markeredgewidth=0.4, markersize=7, linestyle="",
                   label=f"sirna {s}")
        for s in sirnas_in_arm
    ]
    shape_handles = [
        plt.Line2D([], [], marker="o", color="gray", linestyle="",
                   markersize=7, markeredgecolor="black", markeredgewidth=0.4,
                   label="real"),
        plt.Line2D([], [], marker="^", color="gray", linestyle="",
                   markersize=7, markeredgecolor="black", markeredgewidth=0.3,
                   label="accepted gen"),
        plt.Line2D([], [], marker="x", color="gray", linestyle="",
                   markersize=7, label="rejected gen"),
    ]
    leg1 = ax.legend(handles=sirna_handles, fontsize=7, loc="upper right",
                     title="perturbation", title_fontsize=7,
                     ncol=(1 if len(sirnas_in_arm) <= 6 else 2), frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=shape_handles, fontsize=7, loc="lower right", frameon=True)


def _collect_arm_sirnas_by_cell(
    arm_set: Set[Tuple[int, int]], cell_ids: List[int],
) -> Dict[int, List[int]]:
    """cell_id → sorted list of sirnas in `arm_set` for that cell."""
    out: Dict[int, List[int]] = {c: [] for c in cell_ids}
    for (c, s) in arm_set:
        if c in out:
            out[c].append(int(s))
    for c in out:
        out[c] = sorted(set(out[c]))
    return out


def make_figures_for_model(
    model_name: str,
    X_real: np.ndarray, labels_real: Dict[str, np.ndarray],
    X_gen: np.ndarray, df_gen: pd.DataFrame, t95: float,
    seen: Set[Tuple[int, int]], unseen: Set[Tuple[int, int]],
    out_dir: Path, feature_space: str, top_k: int,
) -> None:
    cells = sorted(int(c) for c in np.unique(labels_real["cell"]))
    seen_by_cell   = _collect_arm_sirnas_by_cell(seen,   cells)
    unseen_by_cell = _collect_arm_sirnas_by_cell(unseen, cells)

    # Pre-fit PCA per cell on real of that cell (both arms combined).
    pca_by_cell: Dict[int, PCA] = {}
    Zreal_by_cell: Dict[int, np.ndarray] = {}
    real_sirnas_by_cell: Dict[int, np.ndarray] = {}
    for c in cells:
        m = labels_real["cell"] == c
        if m.sum() < 3:
            continue
        p = PCA(n_components=2, random_state=0).fit(X_real[m])
        pca_by_cell[c] = p
        Zreal_by_cell[c] = p.transform(X_real[m])
        real_sirnas_by_cell[c] = labels_real["sirna"][m]

    # Project gen (once per cell, all rows of that cell; we filter later per arm).
    Zgen_by_cell: Dict[int, np.ndarray] = {}
    gen_sirnas_by_cell: Dict[int, np.ndarray] = {}
    gen_accepted_by_cell: Dict[int, np.ndarray] = {}
    for c in cells:
        if c not in pca_by_cell:
            continue
        gm = df_gen["cell_type_id"].values == c
        if gm.sum() == 0:
            Zgen_by_cell[c] = np.zeros((0, 2))
            gen_sirnas_by_cell[c] = np.array([], dtype=int)
            gen_accepted_by_cell[c] = np.array([], dtype=bool)
            continue
        Zgen_by_cell[c] = pca_by_cell[c].transform(X_gen[gm])
        gen_sirnas_by_cell[c] = df_gen["sirna_id"].values[gm].astype(int)
        gen_accepted_by_cell[c] = (df_gen["trust_updated"].values[gm] <= t95)

    for arm_name, arm_by_cell in [("seen", seen_by_cell), ("unseen", unseen_by_cell)]:
        ncols = 2
        nrows = (len(cells) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 7 * nrows))
        axes_flat = np.array(axes).reshape(-1)

        for i, c in enumerate(cells):
            ax = axes_flat[i]
            if c not in pca_by_cell:
                ax.set_title(f"cell {c} — insufficient real")
                ax.axis("off")
                continue
            sirnas_in_arm = arm_by_cell.get(c, [])
            if not sirnas_in_arm:
                ax.set_title(f"cell {c} — no sirnas in {arm_name} arm")
                ax.axis("off")
                continue

            # Filter real to arm.
            real_mask_arm = np.isin(real_sirnas_by_cell[c], np.asarray(sirnas_in_arm))
            Z_real_cell = Zreal_by_cell[c][real_mask_arm]
            real_sirnas = real_sirnas_by_cell[c][real_mask_arm]

            # Filter gen to arm.
            gen_mask_arm = np.isin(gen_sirnas_by_cell[c], np.asarray(sirnas_in_arm))
            Z_gen_arm = Zgen_by_cell[c][gen_mask_arm]
            gen_sirnas_arm = gen_sirnas_by_cell[c][gen_mask_arm]
            gen_accepted_arm = gen_accepted_by_cell[c][gen_mask_arm]

            n_real = int(len(Z_real_cell))
            n_acc  = int(gen_accepted_arm.sum())
            n_rej  = int((~gen_accepted_arm).sum())

            evr = pca_by_cell[c].explained_variance_ratio_
            _plot_cell_panel(
                ax, Z_real_cell, real_sirnas,
                Z_gen_arm, gen_sirnas_arm, gen_accepted_arm,
                sirnas_in_arm, evr, c, arm_name, n_real, n_acc, n_rej,
            )

        for j in range(len(cells), len(axes_flat)):
            axes_flat[j].axis("off")

        title_suffix = (f"top-{top_k}" if feature_space == "cp"
                        else f"{feature_space} full")
        fig.suptitle(
            f"{model_name} — {arm_name} arm — per-cell PCA ({feature_space}, {title_suffix}, "
            f"real-fit) — colored by sirna; o real, ^ accepted, × rejected",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = out_dir / f"pca_percell_{model_name}_{arm_name}_{feature_space}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Per-cell PCA plot — real vs accepted vs rejected gen.",
    )
    p.add_argument("--cp-dir", type=Path, default=Path("/mnt/pvc/cellprofiler_outputs"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/cp_analysis"))
    p.add_argument("--encoder", type=str, default="siglip",
                   help="Trust-scoring encoder for t95 threshold (→ scores_png_<tag>.json)")
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    p.add_argument("--feature-space", type=str, default="cp",
                   choices=["cp", "siglip", "dinov3"])
    p.add_argument("--top-k", type=int, default=15,
                   help="CP top-k (only used if --feature-space=cp)")
    args = p.parse_args()

    if not args.cp_dir.exists():
        raise SystemExit(f"--cp-dir does not exist: {args.cp_dir}")
    models = resolve_models(args.models)

    out_dir = args.output_dir / "morphology_validation" / "pca_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    reduced, _, scaler = _load_feature_artifacts(args.output_dir)
    arms = load_rxrx1_subset_arms()
    seen, unseen = arms["seen"], arms["unseen"]
    top_feats = _load_top_k_features(args.output_dir, args.top_k)

    logger.info(f"Loading real in {args.feature_space} space ...")
    X_real, labels_real, gen_transformer, df_real = _load_real_in_space(
        args.cp_dir, args.output_dir, args.feature_space,
        reduced, top_feats, seen, unseen, scaler,
    )
    logger.info(f"  real shape={X_real.shape}, cells={sorted(np.unique(labels_real['cell']))}")

    for model in models:
        logger.info(f"[{model}] loading gen in {args.feature_space} space")
        X_gen, df_gen, t95 = _load_gen_in_space(
            args.cp_dir, args.output_dir, model, args.encoder, args.feature_space,
            reduced, top_feats, seen, unseen, scaler, gen_transformer,
        )
        make_figures_for_model(
            model_name=model,
            X_real=X_real, labels_real=labels_real,
            X_gen=X_gen, df_gen=df_gen, t95=t95,
            seen=seen, unseen=unseen,
            out_dir=out_dir,
            feature_space=args.feature_space, top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
