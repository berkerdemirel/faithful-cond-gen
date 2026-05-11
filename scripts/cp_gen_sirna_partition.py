"""
Gen-sirna partition sensitivity under a fixed real-trained classifier.

Question: given a real-trained CP sirna classifier (the "oracle"), does gen
sirna classification accuracy move across *any* partitioning of gen? If
every partition (trust-accepted, trust-rejected, random-stratified, all)
lands at the same accuracy, then selection cannot improve sirna readout
because the real classifier sees all gen samples as carrying the same
sirna signal. If partitions disagree, selection can in principle exploit
the variation.

Protocol, per `(model × cell × arm)`:
  1. Train a logistic classifier on **real** CP top-k restricted to the
     arm's sirnas for this cell. This is the fixed oracle.
  2. Apply the oracle to gen of the same `(cell, arm)`; get per-sample
     prediction and correctness.
  3. Compute balanced + micro accuracy under four partitions:
       - all gen
       - trust-accepted (`trust_updated ≤ t95`)
       - trust-rejected
       - combo-stratified random at matched n (5 seeds)
  4. Compute `Spearman(trust_updated, per-sample correctness)` — does trust
     rank correlate with "oracle says this sample is right"?

Output CSV: `outputs/cp_analysis/morphology_validation/gen_sirna_partition_{space}.csv`.
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
    _random_indices,
)
from faithful_cond_gen.eval.trust_eval.subset_io import load_rxrx1_subset_arms  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _fit_real_classifier(
    X: np.ndarray, y: np.ndarray, seed: int = 0, min_per_class: int = 3,
) -> Tuple[Optional[object], Optional[np.ndarray]]:
    """Fit a logistic classifier on real rows. Drops classes with too few
    samples. Returns (pipeline, kept_classes_in_y) or (None, None) if not
    enough data to train."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    unique, counts = np.unique(y, return_counts=True)
    keep = unique[counts >= min_per_class]
    if len(keep) < 2:
        return None, None
    mask = np.isin(y, keep)
    X_fit = X[mask]; y_fit = y[mask]
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=2000, solver="lbfgs", C=1.0, random_state=seed,
        )),
    ])
    pipe.fit(X_fit, y_fit)
    return pipe, keep


def _scores_on_mask(
    y_pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray,
) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
        return {"macro": float("nan"), "micro": float("nan"), "n": int(mask.sum())}
    return {
        "macro": float(balanced_accuracy_score(y_true[mask], y_pred[mask])),
        "micro": float(accuracy_score(y_true[mask], y_pred[mask])),
        "n":     int(mask.sum()),
    }


def main() -> None:
    from scipy.stats import spearmanr

    p = argparse.ArgumentParser(
        description="Does gen sirna accuracy move across partitions under a fixed real-trained classifier?"
    )
    p.add_argument("--cp-dir", type=Path, default=Path("/mnt/pvc/cellprofiler_outputs"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/cp_analysis"))
    p.add_argument("--encoder", type=str, default="siglip")
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    p.add_argument("--feature-space", type=str, default="cp",
                   choices=["cp", "siglip", "dinov3"])
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--n-random-seeds", type=int, default=5)
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

                # Train real classifier (fixed oracle for this cell × arm).
                r_mask = ((labels_real["cell"] == cell_id) &
                          np.isin(labels_real["sirna"], sirnas_arr))
                clf, kept_classes = _fit_real_classifier(X_real[r_mask], labels_real["sirna"][r_mask])
                if clf is None:
                    logger.warning(f"[{model} / cell {cell_id} / {arm_name}] skip: real too small")
                    continue

                # Score the classifier on its own training real (sanity ceiling).
                y_r = labels_real["sirna"][r_mask]
                y_r_pred = clf.predict(X_real[r_mask])
                real_macro = _scores_on_mask(y_r_pred, y_r, np.isin(y_r, kept_classes))

                # Gen for this cell × arm.
                g_mask = ((df_gen["cell_type_id"].values == cell_id) &
                          np.isin(df_gen["sirna_id"].values, sirnas_arr))
                sub = df_gen[g_mask].reset_index(drop=True)
                if len(sub) < 20:
                    continue
                X_g = X_gen[g_mask]
                y_g = sub["sirna_id"].values.astype(int)
                trust_g = sub["trust_updated"].values.astype(float)
                trust_mask = (trust_g <= t95)

                # Keep only gen rows whose class was kept in real (so prediction is meaningful).
                valid_g = np.isin(y_g, kept_classes)
                if valid_g.sum() < 20:
                    continue
                X_g = X_g[valid_g]
                sub = sub.iloc[valid_g].reset_index(drop=True)
                y_g = y_g[valid_g]
                trust_g = trust_g[valid_g]
                trust_mask = trust_mask[valid_g]

                # Predict once, partition after.
                y_g_pred = clf.predict(X_g)
                correct = (y_g_pred == y_g).astype(int)

                # Partitions: all, accepted, rejected, random-stratified.
                all_mask = np.ones(len(sub), dtype=bool)
                rand_rows: List[Dict] = []
                for sd in range(args.n_random_seeds):
                    ridx = _random_indices(sub, trust_mask, seed=sd, mode="combo_stratified")
                    rm = np.zeros(len(sub), dtype=bool); rm[ridx] = True
                    rand_rows.append({
                        "mask": rm,
                        "seed": sd,
                    })

                def _emit(subset_name: str, mask: np.ndarray, seed: Optional[int]) -> Dict:
                    sc = _scores_on_mask(y_g_pred, y_g, mask)
                    return {
                        "model": model, "cell": int(cell_id), "arm": arm_name,
                        "subset": subset_name, "seed": seed,
                        "n_sirnas_in_arm": len(sirnas_arr),
                        "n_real_train":    int(r_mask.sum()),
                        "chance":          chance,
                        "real_macro":      real_macro["macro"],   # classifier on its own real
                        "real_micro":      real_macro["micro"],
                        "gen_n":           sc["n"],
                        "gen_macro":       sc["macro"],
                        "gen_micro":       sc["micro"],
                    }

                rows.append(_emit("all", all_mask, None))
                rows.append(_emit("trust_accepted", trust_mask,  None))
                rows.append(_emit("trust_rejected", ~trust_mask, None))
                for r in rand_rows:
                    rows.append(_emit("random_stratified", r["mask"], r["seed"]))

                # Trust-correctness correlation.
                if len(np.unique(correct)) >= 2:
                    rho, pv = spearmanr(trust_g, correct)
                else:
                    rho, pv = float("nan"), float("nan")
                rows.append({
                    "model": model, "cell": int(cell_id), "arm": arm_name,
                    "subset": "_rho_trust_correct", "seed": None,
                    "n_sirnas_in_arm": len(sirnas_arr),
                    "n_real_train":    int(r_mask.sum()),
                    "chance":          chance,
                    "real_macro":      real_macro["macro"],
                    "real_micro":      real_macro["micro"],
                    "gen_n":           int(len(correct)),
                    "gen_macro":       float(rho) if np.isfinite(rho) else float("nan"),
                    "gen_micro":       float(pv)  if np.isfinite(pv)  else float("nan"),
                })

    df = pd.DataFrame(rows)
    out_path = out_dir / f"gen_sirna_partition_{args.feature_space}.csv"
    df.to_csv(out_path, index=False)

    # Compact pivoted table per (model, cell, arm).
    print("\n=== Real-trained oracle on gen partitions (CP top-%d) ===" % args.top_k)
    print("Oracle: logistic fit on real CP top-%d, restricted to arm's sirnas for each cell." % args.top_k)
    print("`random_strat` is mean across 5 combo-stratified seeds.")
    piv_rows: List[Dict] = []
    for (model, cell_id, arm_name), sub in df.groupby(["model", "cell", "arm"]):
        if sub.empty or "subset" not in sub.columns:
            continue
        def pick(s):
            r = sub[sub["subset"] == s]
            return float(r["gen_macro"].iloc[0]) if len(r) else float("nan")
        rand = sub[sub["subset"] == "random_stratified"]["gen_macro"]
        rand_mean = float(rand.mean()) if len(rand) else float("nan")
        rand_std  = float(rand.std())  if len(rand) > 1 else 0.0
        rho_row = sub[sub["subset"] == "_rho_trust_correct"]
        rho = float(rho_row["gen_macro"].iloc[0]) if len(rho_row) else float("nan")
        rho_p = float(rho_row["gen_micro"].iloc[0]) if len(rho_row) else float("nan")

        piv_rows.append({
            "model": model, "cell": int(cell_id), "arm": arm_name,
            "K":      int(sub["n_sirnas_in_arm"].iloc[0]),
            "chance": float(sub["chance"].iloc[0]),
            "real_macro": float(sub["real_macro"].iloc[0]),
            "all":           pick("all"),
            "accepted":      pick("trust_accepted"),
            "rejected":      pick("trust_rejected"),
            "random":        rand_mean,
            "rand_std":      rand_std,
            "rho(trust,correct)": rho,
            "rho_p":              rho_p,
        })
    piv = pd.DataFrame(piv_rows)
    print(piv.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}" if abs(x) < 10 else f"{x:.2f}",
    ))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
