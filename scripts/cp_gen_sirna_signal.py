"""
Gen sirna signal check — does each gen model carry within-cell sirna
structure at all?

Motivation: if gen at `(cell_i, sirna_j)` and `(cell_i, sirna_k)` look
essentially the same (i.e. gen collapses sirna within a cell), then
*any* subset of gen — trust-selected, random, rejected — yields the same
sirna classifier accuracy, and "trust selection doesn't improve sirna
readout" is a vacuous conclusion. This script directly tests whether
there is anything to improve.

For each `(model × cell × arm)`:
  1. Train a 5-fold stratified CV logistic on the **real** rows of that
     cell restricted to sirnas in that arm — sirna ceiling.
  2. Same CV on **all gen** of that cell × arm.
  3. Same on **trust-accepted gen** (trust_updated ≤ t95).
  4. Same on **trust-rejected gen**.
  5. Per-feature one-way ANOVA F (sirna grouping) on each subset → mean F
     across the top-k CP features.

If gen has *no* within-cell sirna structure, gen macro ≈ chance (= 1 /
n_sirnas_in_arm) and F_gen ≪ F_real. If gen has sirna structure but
trust doesn't exploit it, accepted / rejected look similar to all gen.

Output: `outputs/cp_analysis/morphology_validation/gen_sirna_signal_within_cell.csv`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from analyze_cp_features_rxrx1 import (  # noqa: E402
    _load_feature_artifacts, resolve_models,
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


def _cv_classify(
    X: np.ndarray, y: np.ndarray,
    n_folds: int = 5, seed: int = 0, min_per_class: int = 5,
) -> Optional[Dict[str, float]]:
    """5-fold stratified CV logistic on (X, y). Drops classes with fewer than
    `min_per_class` samples. Returns None if < 2 usable classes or too few
    samples for CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    unique, counts = np.unique(y, return_counts=True)
    keep_classes = unique[counts >= min_per_class]
    if len(keep_classes) < 2:
        return None
    keep_row = np.isin(y, keep_classes)
    X = X[keep_row]; y = y[keep_row]
    min_count = int(np.unique(y, return_counts=True)[1].min())
    folds = min(n_folds, min_count)
    if folds < 2:
        return None
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=2000, solver="lbfgs", C=1.0, random_state=seed,
        )),
    ])
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    res = cross_validate(
        pipe, X, y, cv=skf,
        scoring=["balanced_accuracy", "accuracy"],
        n_jobs=1, return_train_score=False,
    )
    return {
        "macro_mean": float(res["test_balanced_accuracy"].mean()),
        "macro_std":  float(res["test_balanced_accuracy"].std()),
        "micro_mean": float(res["test_accuracy"].mean()),
        "micro_std":  float(res["test_accuracy"].std()),
        "n":          int(len(y)),
        "n_classes":  int(len(keep_classes)),
        "n_folds":    int(folds),
    }


def _feature_F(X: np.ndarray, y: np.ndarray, min_per_class: int = 5) -> Optional[np.ndarray]:
    """One-way ANOVA F per feature with sirna as the factor. Returns None if
    too few classes."""
    from sklearn.feature_selection import f_classif
    unique, counts = np.unique(y, return_counts=True)
    keep_classes = unique[counts >= min_per_class]
    if len(keep_classes) < 2:
        return None
    keep_row = np.isin(y, keep_classes)
    F, _ = f_classif(X[keep_row], y[keep_row])
    return F


def main() -> None:
    p = argparse.ArgumentParser(
        description="Within-cell sirna signal check on gen vs real."
    )
    p.add_argument("--cp-dir", type=Path, default=Path("/mnt/pvc/cellprofiler_outputs"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/cp_analysis"))
    p.add_argument("--encoder", type=str, default="siglip")
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    p.add_argument("--feature-space", type=str, default="cp",
                   choices=["cp", "siglip", "dinov3"])
    p.add_argument("--top-k", type=int, default=15)
    args = p.parse_args()

    if not args.cp_dir.exists():
        raise SystemExit(f"--cp-dir does not exist: {args.cp_dir}")
    models = resolve_models(args.models)

    out_dir = args.output_dir / "morphology_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    reduced, _, scaler = _load_feature_artifacts(args.output_dir)
    arms = load_rxrx1_subset_arms()
    seen, unseen = arms["seen"], arms["unseen"]
    top_feats = _load_top_k_features(args.output_dir, args.top_k)

    X_real, labels_real, gen_transformer, _ = _load_real_in_space(
        args.cp_dir, args.output_dir, args.feature_space,
        reduced, top_feats, seen, unseen, scaler,
    )

    rows: List[Dict] = []
    for model in models:
        logger.info(f"[{model}] loading gen in {args.feature_space} space")
        X_gen, df_gen, t95 = _load_gen_in_space(
            args.cp_dir, args.output_dir, model, args.encoder, args.feature_space,
            reduced, top_feats, seen, unseen, scaler, gen_transformer,
        )
        for cell_id in sorted(int(c) for c in np.unique(labels_real["cell"])):
            for arm_name, arm_set in [("seen", seen), ("unseen", unseen)]:
                sirnas_in_arm = sorted({s for (c, s) in arm_set if c == cell_id})
                if not sirnas_in_arm:
                    continue
                sirnas_arr = np.asarray(sirnas_in_arm)
                chance = 1.0 / len(sirnas_arr)

                # Real subset for this cell × arm
                r_mask = ((labels_real["cell"] == cell_id) &
                          np.isin(labels_real["sirna"], sirnas_arr))
                X_r = X_real[r_mask]; y_r = labels_real["sirna"][r_mask]

                # Gen subset for this cell × arm
                g_mask = ((df_gen["cell_type_id"].values == cell_id) &
                          np.isin(df_gen["sirna_id"].values, sirnas_arr))
                X_g = X_gen[g_mask]
                y_g = df_gen["sirna_id"].values[g_mask]
                trust_mask = (df_gen["trust_updated"].values[g_mask] <= t95)

                subsets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
                    "real":     (X_r, y_r),
                    "gen_all":  (X_g, y_g),
                    "gen_acc":  (X_g[trust_mask],  y_g[trust_mask]),
                    "gen_rej":  (X_g[~trust_mask], y_g[~trust_mask]),
                }
                for name, (X_s, y_s) in subsets.items():
                    res = _cv_classify(X_s, y_s)
                    F = _feature_F(X_s, y_s)
                    rows.append({
                        "model": (model if name != "real" else "—"),
                        "cell": int(cell_id), "arm": arm_name,
                        "subset": name,
                        "n_sirnas_in_arm": int(len(sirnas_arr)),
                        "chance":    chance,
                        "n":         int(len(y_s)) if res is None else res["n"],
                        "n_classes": int(len(np.unique(y_s))) if res is None else res["n_classes"],
                        "n_folds":   int(res["n_folds"]) if res else 0,
                        "macro_acc": float("nan") if res is None else res["macro_mean"],
                        "macro_std": float("nan") if res is None else res["macro_std"],
                        "micro_acc": float("nan") if res is None else res["micro_mean"],
                        "F_mean":    float("nan") if F is None else float(np.nanmean(F)),
                        "F_median":  float("nan") if F is None else float(np.nanmedian(F)),
                    })

    df = pd.DataFrame(rows)
    # Collapse the real row per (cell, arm) to one entry (it doesn't depend on model),
    # then merge per model row onto it.
    real_df = df[df["subset"] == "real"].drop_duplicates(subset=["cell", "arm", "subset"]).copy()
    real_df["model"] = "—"
    gen_df  = df[df["subset"] != "real"].copy()
    out_df  = pd.concat([real_df, gen_df], ignore_index=True)

    out_path = out_dir / f"gen_sirna_signal_within_cell_{args.feature_space}.csv"
    out_df.to_csv(out_path, index=False)

    # Compact per-(cell, arm) table showing real vs 3 gen subsets side-by-side.
    print("\n=== Within-cell sirna signal (feature_space=%s, top-%d) ===" % (args.feature_space, args.top_k))
    print("Real is ceiling. Gen subsets: all / trust-accepted / trust-rejected.")
    print("Chance = 1 / n_sirnas_in_arm (balanced-accuracy baseline).")
    # Pivot for readability.
    piv_rows = []
    for (cell_id, arm_name), sub in out_df.groupby(["cell", "arm"]):
        real_row = sub[sub.subset == "real"].iloc[0]
        for model in models:
            gall = sub[(sub.subset == "gen_all") & (sub.model == model)]
            gacc = sub[(sub.subset == "gen_acc") & (sub.model == model)]
            grej = sub[(sub.subset == "gen_rej") & (sub.model == model)]
            piv_rows.append({
                "cell": int(cell_id), "arm": arm_name, "model": model,
                "n_sirnas": int(real_row["n_sirnas_in_arm"]),
                "chance":    real_row["chance"],
                "real_macro": real_row["macro_acc"],
                "F_real":     real_row["F_mean"],
                "gen_all_macro": gall["macro_acc"].iloc[0] if len(gall) else float("nan"),
                "F_gen_all":     gall["F_mean"].iloc[0] if len(gall) else float("nan"),
                "n_all":     int(gall["n"].iloc[0]) if len(gall) else 0,
                "gen_acc_macro": gacc["macro_acc"].iloc[0] if len(gacc) else float("nan"),
                "F_gen_acc":     gacc["F_mean"].iloc[0] if len(gacc) else float("nan"),
                "n_acc":     int(gacc["n"].iloc[0]) if len(gacc) else 0,
                "gen_rej_macro": grej["macro_acc"].iloc[0] if len(grej) else float("nan"),
                "F_gen_rej":     grej["F_mean"].iloc[0] if len(grej) else float("nan"),
                "n_rej":     int(grej["n"].iloc[0]) if len(grej) else 0,
            })
    piv = pd.DataFrame(piv_rows)
    print(piv.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}" if abs(x) < 10 else f"{x:.2f}",
    ))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
