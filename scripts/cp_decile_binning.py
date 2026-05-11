"""
CP decile-binning downstream — reproduction of the DINOv3 trust-eval
`outputs/trust_evaluation_mahalanobis_dinov3_rxrx1/` decile analysis but
scoring in CP feature space.

Protocol (one per model):
  1. Rank all gen samples by each of {trust_updated, realism_z, faithfulness_z}.
  2. Split each ranking into 10 equal-size deciles (~500 per bin for n=5000 gen).
  3. For each `(ranking, bin)` train a classifier on the 500-sample gen bin
     (CP top-k features) and test on all real (CP top-k, same feature set).
  4. Targets: `celltype` (4-way) and `subset` (50-way combo).
  5. Baseline: `random` — 5 seeds of 500 random gen samples; report mean per
     decile index (flat by construction, since random draws don't depend on
     the bin index — we just fill 10 rows with 5-seed averages for direct
     side-by-side reading).

Outputs:
  outputs/cp_analysis/morphology_validation/decile_binning/
    decile_{model}_cp.csv       one CSV per model with (ranking, bin, target, acc)
    decile_{model}_cp.png       plot: celltype + subset side-by-side
    decile_summary_cp.csv       combined table across models
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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


def _train_test(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    seed: int = 0,
    pipeline: str = "standardize",
) -> Dict[str, float]:
    """Fit logistic on train, test on test. Returns macro + micro + coverage
    (fraction of test-set classes that appeared in training).

    `pipeline`:
      - "standardize" (default): StandardScaler + LogisticRegression.
      - "l2_no_scaler":           L2-normalize rows of train and test once,
        then bare LogisticRegression. Matches the trust-eval downstream
        recipe (`apply_normalization(..., 'l2')` + plain LR).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    unique_tr, counts_tr = np.unique(y_train, return_counts=True)
    keep = unique_tr[counts_tr >= 2]
    if len(keep) < 2:
        return {"macro": float("nan"), "micro": float("nan"),
                "coverage": 0.0, "n_train_classes": int(len(keep))}
    mask = np.isin(y_train, keep)

    if pipeline == "standardize":
        clf = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=2000, solver="lbfgs", C=1.0, random_state=seed,
            )),
        ])
        X_tr = X_train[mask]; X_te = X_test
    elif pipeline == "l2_no_scaler":
        def _l2(x: np.ndarray) -> np.ndarray:
            return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        X_tr = _l2(X_train[mask])
        X_te = _l2(X_test)
        clf = LogisticRegression(
            max_iter=2000, solver="lbfgs", C=1.0, random_state=seed,
        )
    else:
        raise ValueError(f"unknown pipeline: {pipeline!r}")

    clf.fit(X_tr, y_train[mask])
    y_pred = clf.predict(X_te)
    coverage = float(np.isin(np.unique(y_test), keep).mean())
    return {
        "macro":    float(balanced_accuracy_score(y_test, y_pred)),
        "micro":    float(accuracy_score(y_test, y_pred)),
        "coverage": coverage,
        "n_train_classes": int(len(keep)),
    }


def _decile_bins(scores: np.ndarray, n_bins: int = 10) -> List[np.ndarray]:
    """[legacy, global] Return list of length n_bins, each an array of indices
    ranked by ascending `scores`. Bin 0 = best (lowest score; our convention
    lower=better). NOT condition-balanced — kept for reference only."""
    order = np.argsort(scores)
    return np.array_split(order, n_bins)


def _decile_bins_within_condition(
    scores: np.ndarray,
    cell_ids: np.ndarray,
    sirna_ids: np.ndarray,
    n_bins: int = 10,
) -> List[np.ndarray]:
    """Per-(cell, sirna) balanced binning — matches the DINOv3 trust-eval
    protocol (`bin_samples_within_conditioning`). For each condition, sort
    the condition's gen by `scores` ascending and split into `n_bins`. Bin
    `i` = union of bin `i` across all conditions, so every bin is
    balanced by construction on (cell, sirna).

    Returns list of length `n_bins`; entries are arrays of global gen indices.
    """
    conds = list(zip(cell_ids.tolist(), sirna_ids.tolist()))
    cond_to_idx: Dict[Tuple[int, int], List[int]] = {}
    for i, c in enumerate(conds):
        cond_to_idx.setdefault((int(c[0]), int(c[1])), []).append(i)

    # For each condition, sort by score and split into n_bins.
    per_cond_bins: Dict[Tuple[int, int], List[np.ndarray]] = {}
    for cond, idx_list in cond_to_idx.items():
        idx_arr = np.array(idx_list, dtype=int)
        cond_scores = scores[idx_arr]
        order = np.argsort(cond_scores)
        sorted_idx = idx_arr[order]
        n_cond = len(sorted_idx)
        bin_size = max(1, n_cond // n_bins)
        cond_bins: List[np.ndarray] = []
        for b in range(n_bins):
            start = b * bin_size
            end = (b + 1) * bin_size if b < n_bins - 1 else n_cond
            cond_bins.append(sorted_idx[start:end] if start < n_cond else np.array([], dtype=int))
        per_cond_bins[cond] = cond_bins

    # Merge bin `i` across conditions.
    merged: List[np.ndarray] = []
    for b in range(n_bins):
        parts = [per_cond_bins[cond][b] for cond in per_cond_bins
                 if len(per_cond_bins[cond][b])]
        merged.append(np.concatenate(parts) if parts else np.array([], dtype=int))
    return merged


def _run_model(
    model: str,
    X_gen: np.ndarray, df_gen: pd.DataFrame,
    scores_full: Dict[str, Dict],
    X_real: np.ndarray, y_test: Dict[str, np.ndarray],
    n_bins: int, n_random_seeds: int,
) -> pd.DataFrame:
    """For one model, run all four rankings × all deciles × two targets."""
    # Pull trust / realism / faithfulness per gen row (same order as df_gen).
    trust = np.array([
        scores_full.get(Path(fn).stem, {}).get("trust_updated", np.nan)
        for fn in df_gen["FileName_DATA"]
    ], dtype=float)
    realism = np.array([
        scores_full.get(Path(fn).stem, {}).get("realism_z", np.nan)
        for fn in df_gen["FileName_DATA"]
    ], dtype=float)
    faith = np.array([
        scores_full.get(Path(fn).stem, {}).get("faithfulness_z", np.nan)
        for fn in df_gen["FileName_DATA"]
    ], dtype=float)

    # For rankings where NaN is present, drop to be safe.
    valid = np.isfinite(trust) & np.isfinite(realism) & np.isfinite(faith)
    if not valid.all():
        X_gen = X_gen[valid]
        trust = trust[valid]; realism = realism[valid]; faith = faith[valid]
        df_gen = df_gen[valid].reset_index(drop=True)

    rankings = {
        "trust":        trust,
        "realism":      realism,
        "faithfulness": faith,
    }

    rows: List[Dict] = []
    # Targets: celltype (4-way), subset = 50-way combo
    y_gen_celltype = df_gen["cell_type_id"].astype(int).values
    y_gen_combo    = (df_gen["cell_type_id"].astype(int) * 100000
                      + df_gen["sirna_id"].astype(int)).values
    y_gen_targets = {"celltype": y_gen_celltype, "subset": y_gen_combo}
    cell_ids  = df_gen["cell_type_id"].astype(int).values
    sirna_ids = df_gen["sirna_id"].astype(int).values

    # 1. Trust / realism / faithfulness: within-condition decile bins (matches
    # DINOv3 trust-eval `bin_samples_within_conditioning`). Every bin is
    # balanced on (cell, sirna) by construction.
    for rname, scores in rankings.items():
        bins = _decile_bins_within_condition(scores, cell_ids, sirna_ids, n_bins=n_bins)
        for bin_idx, idx in enumerate(bins):
            if len(idx) == 0:
                continue
            for tname in ("celltype", "subset"):
                y_tr = y_gen_targets[tname][idx]
                y_te = y_test[tname]
                res = _train_test(X_gen[idx], y_tr, X_real, y_te)
                rows.append({
                    "model": model, "ranking": rname, "bin_idx": bin_idx,
                    "target": tname, "n_train": int(len(idx)),
                    "n_test": int(len(y_te)),
                    "macro": res["macro"], "micro": res["micro"],
                    "coverage": res["coverage"],
                    "n_train_classes": res["n_train_classes"],
                    "score_min": float(scores[idx].min()),
                    "score_max": float(scores[idx].max()),
                })

    # 2. Random baseline: per-condition random partition matching the
    # within-condition decile structure. For each seed, randomly permute each
    # condition's gen indices (instead of sorting by score), then bin + merge
    # the same way. This yields n_bins random bins each balanced on (cell, sirna).
    for sd in range(n_random_seeds):
        rng = np.random.default_rng(1000 + sd)
        rnd_scores = rng.random(len(X_gen))  # uniform random "rank"
        rbins = _decile_bins_within_condition(rnd_scores, cell_ids, sirna_ids, n_bins=n_bins)
        for bin_idx, idx in enumerate(rbins):
            if len(idx) == 0:
                continue
            for tname in ("celltype", "subset"):
                res = _train_test(X_gen[idx], y_gen_targets[tname][idx], X_real, y_test[tname])
                rows.append({
                    "model": model, "ranking": "random", "bin_idx": bin_idx,
                    "target": tname, "n_train": int(len(idx)),
                    "n_test": int(len(y_test[tname])),
                    "macro": res["macro"], "micro": res["micro"],
                    "coverage": res["coverage"],
                    "n_train_classes": res["n_train_classes"],
                    "score_min": float("nan"), "score_max": float("nan"),
                    "seed": sd,
                })

    return pd.DataFrame(rows)


def _plot_model(df: pd.DataFrame, model: str, out_path: Path, n_bins: int) -> None:
    ranking_order = ["trust", "realism", "faithfulness", "random"]
    colors = {
        "trust":        "#0072B2",
        "realism":      "#009E73",
        "faithfulness": "#D55E00",
        "random":       "#888888",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, tname, pretty in [(axes[0], "celltype", "celltype (4-way)"),
                              (axes[1], "subset",   "subset / combo (50-way)")]:
        for rname in ranking_order:
            sub = df[(df.ranking == rname) & (df.target == tname)]
            if sub.empty:
                continue
            if rname == "random":
                agg = sub.groupby("bin_idx").agg(
                    macro_mean=("macro", "mean"), macro_std=("macro", "std"),
                    micro_mean=("micro", "mean"), micro_std=("micro", "std"),
                ).reset_index().sort_values("bin_idx")
                ax.errorbar(agg["bin_idx"], agg["macro_mean"], yerr=agg["macro_std"],
                            marker="o", linestyle="--", color=colors[rname],
                            capsize=3, label=f"{rname} (macro, ±std)")
                ax.errorbar(agg["bin_idx"], agg["micro_mean"], yerr=agg["micro_std"],
                            marker="^", linestyle=":", color=colors[rname], alpha=0.55,
                            capsize=3, label=f"{rname} (micro, ±std)")
            else:
                s = sub.sort_values("bin_idx")
                ax.plot(s["bin_idx"], s["macro"], marker="o", linestyle="-",
                        color=colors[rname], label=f"{rname} (macro)")
                ax.plot(s["bin_idx"], s["micro"], marker="^", linestyle=":",
                        color=colors[rname], alpha=0.55, label=f"{rname} (micro)")
        ax.set_xlabel("Decile bin  (0 = best rank)")
        ax.set_ylabel("Accuracy (gen-trained classifier on real test)")
        ax.set_title(f"{model} — {pretty}  (within-condition bins)")
        ax.grid(linestyle=":", alpha=0.3)
        ax.set_xticks(list(range(n_bins)))
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"{model} — CP top-15 decile binning (gen → real transfer)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="CP-features decile binning (gen-trained classifier → real test).",
    )
    p.add_argument("--cp-dir", type=Path, default=Path("/mnt/pvc/cellprofiler_outputs"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/cp_analysis"))
    p.add_argument("--encoder", type=str, default="siglip")
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    p.add_argument("--feature-space", type=str, default="cp",
                   choices=["cp", "siglip", "dinov3"])
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--n-random-seeds", type=int, default=5)
    args = p.parse_args()

    if not args.cp_dir.exists():
        raise SystemExit(f"--cp-dir does not exist: {args.cp_dir}")
    models = resolve_models(args.models)

    out_dir = args.output_dir / "morphology_validation" / "decile_binning"
    out_dir.mkdir(parents=True, exist_ok=True)

    reduced, _, scaler = _load_feature_artifacts(args.output_dir)
    arms = load_rxrx1_subset_arms()
    seen, unseen = arms["seen"], arms["unseen"]
    top_feats = _load_top_k_features(args.output_dir, args.top_k)

    # Real once — test set for every gen bin.
    X_real, labels_real, gen_transformer, _ = _load_real_in_space(
        args.cp_dir, args.output_dir, args.feature_space,
        reduced, top_feats, seen, unseen, scaler,
    )
    y_test = {
        "celltype": labels_real["cell"],
        "subset":   labels_real["combo"],
    }

    import json
    summaries: List[pd.DataFrame] = []
    for model in models:
        logger.info(f"[{model}] loading gen and running deciles")
        X_gen, df_gen, _ = _load_gen_in_space(
            args.cp_dir, args.output_dir, model, args.encoder, args.feature_space,
            reduced, top_feats, seen, unseen, scaler, gen_transformer,
        )
        # Need all three components, not just trust_updated. Reload full scores.
        scores_path = args.output_dir / f"rxrx1_{model.replace('_marginal','_marginal').replace('_full','_full')}" / f"scores_png_{args.encoder}.json"
        # Use the module-standard path helper instead:
        from analyze_cp_features_rxrx1 import MODEL_TO_CP_DIR
        scores_path = args.output_dir / MODEL_TO_CP_DIR[model] / f"scores_png_{args.encoder}.json"
        scores_full = json.loads(scores_path.read_text())

        df_model = _run_model(
            model, X_gen, df_gen, scores_full,
            X_real, y_test, args.n_bins, args.n_random_seeds,
        )
        out_csv = out_dir / f"decile_{model}_{args.feature_space}.csv"
        df_model.to_csv(out_csv, index=False)
        _plot_model(df_model, model, out_dir / f"decile_{model}_{args.feature_space}.png", args.n_bins)
        summaries.append(df_model)
        logger.info(f"  wrote {out_csv}")

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(out_dir / f"decile_summary_{args.feature_space}.csv", index=False)

    # Compact print: bin 0 vs bin 9 per (model, ranking, target). For random
    # (5 seeds × 10 bins), aggregate to the mean per bin_idx first.
    print("\n=== CP top-%d decile downstream (within-condition bins; bin 0 = best rank, bin 9 = worst) ===" % args.top_k)
    rows = []
    for (model, ranking, target), sub in combined.groupby(["model", "ranking", "target"]):
        if ranking == "random":
            agg = (sub.groupby("bin_idx")[["macro", "micro"]].mean()
                   .reset_index().sort_values("bin_idx"))
            sub = agg
        else:
            sub = sub.sort_values("bin_idx")
        if len(sub) < 10:
            continue
        rows.append({
            "model": model, "ranking": ranking, "target": target,
            "bin_0_macro": float(sub.iloc[0]["macro"]),
            "bin_4_macro": float(sub.iloc[4]["macro"]),
            "bin_9_macro": float(sub.iloc[-1]["macro"]),
            "spread_macro": float(sub.iloc[0]["macro"] - sub.iloc[-1]["macro"]),
            "bin_0_micro": float(sub.iloc[0]["micro"]),
            "bin_9_micro": float(sub.iloc[-1]["micro"]),
            "spread_micro": float(sub.iloc[0]["micro"] - sub.iloc[-1]["micro"]),
        })
    piv = pd.DataFrame(rows).sort_values(["model", "target", "ranking"])
    print(piv.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}" if abs(x) < 10 else f"{x:.2f}",
    ))
    print(f"\nArtifacts → {out_dir}")


if __name__ == "__main__":
    main()
