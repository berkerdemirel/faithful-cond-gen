"""
Real-data readout ceiling — sanity baseline for per-space, per-target CP
and neural encoder morphology information.

Question answered: *given only real images, how well can a classifier recover
cell-type, combo, sirna from each feature space?* This is the ceiling; any
evaluation on generated samples must be read against it.

Five-fold stratified CV on real rows (same 2066 paired PNGs across CP /
SigLIP / DINOv3). Logistic regression pipeline with StandardScaler + optional
PCA. Reports both micro (accuracy) and macro (balanced_accuracy) — for
imbalanced fine-grained targets (sirna, combo) macro is the informative one.

Usage:
    PYTHONPATH=src uv run python scripts/cp_real_readout_ceiling.py

Output: outputs/cp_analysis/morphology_validation/real_readout_ceiling.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from analyze_cp_features_rxrx1 import _load_feature_artifacts  # noqa: E402
from cp_morphology_validation import (  # noqa: E402
    _load_real_df,
    _load_top_k_features,
)
from faithful_cond_gen.eval.trust_eval.subset_io import load_rxrx1_subset_arms  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _cv_scores_by_arm(
    X: np.ndarray, y: np.ndarray, arm_labels: np.ndarray,
    n_folds: int = 5, seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """5-fold stratified CV logistic (full 50-way). Score test accuracy per
    arm of the test row's `(cell, sirna)`, so the unseen-arm number is
    directly comparable to a trust-vs-random on gen-unseen-arm eval.
    Returns `{arm: {macro_mean, macro_std, micro_mean, micro_std, n_test_avg}}`
    for arm ∈ {'all', 'seen', 'unseen'}."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    buckets: Dict[str, Dict[str, List]] = {
        arm: {"macro": [], "micro": [], "n": []} for arm in ("all", "seen", "unseen")
    }
    for tr, te in skf.split(X, y):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=2000, solver="lbfgs", C=1.0, random_state=seed,
            )),
        ])
        pipe.fit(X[tr], y[tr])
        y_hat = pipe.predict(X[te])
        y_te  = y[te]
        arms_te = arm_labels[te]
        for arm in ("all", "seen", "unseen"):
            mask = (np.ones(len(te), dtype=bool) if arm == "all"
                    else (arms_te == arm))
            if mask.sum() == 0:
                continue
            buckets[arm]["macro"].append(balanced_accuracy_score(y_te[mask], y_hat[mask]))
            buckets[arm]["micro"].append(accuracy_score(y_te[mask], y_hat[mask]))
            buckets[arm]["n"].append(int(mask.sum()))

    out: Dict[str, Dict[str, float]] = {}
    for arm, b in buckets.items():
        if not b["macro"]:
            out[arm] = {k: float("nan") for k in
                        ("macro_mean", "macro_std", "micro_mean", "micro_std", "n_test_avg")}
            continue
        out[arm] = {
            "macro_mean": float(np.mean(b["macro"])),
            "macro_std":  float(np.std(b["macro"])),
            "micro_mean": float(np.mean(b["micro"])),
            "micro_std":  float(np.std(b["micro"])),
            "n_test_avg": float(np.mean(b["n"])),
        }
    return out


def _cv_scores(
    X: np.ndarray, y: np.ndarray,
    n_folds: int = 5, pca_n: int = 0, seed: int = 0,
) -> Tuple[float, float, float, float]:
    """5-fold stratified CV logistic regression; returns (macro_mean, macro_std,
    micro_mean, micro_std). If pca_n > 0, fit PCA inside the pipeline (no leak)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [("scale", StandardScaler())]
    if pca_n > 0:
        steps.append(("pca", PCA(n_components=min(pca_n, X.shape[1]), random_state=seed)))
    steps.append(("logistic", LogisticRegression(
        max_iter=2000, solver="lbfgs", C=1.0, random_state=seed,
    )))
    pipe = Pipeline(steps)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    res = cross_validate(
        pipe, X, y, cv=skf,
        scoring=["balanced_accuracy", "accuracy"],
        n_jobs=1, return_train_score=False,
    )
    return (float(res["test_balanced_accuracy"].mean()),
            float(res["test_balanced_accuracy"].std()),
            float(res["test_accuracy"].mean()),
            float(res["test_accuracy"].std()))


def _load_real_neural(output_dir: Path, space: str, df_real: pd.DataFrame) -> np.ndarray:
    """Load paired SigLIP / DINOv3 real features, aligned to df_real's filenames."""
    import torch
    cache = output_dir / "real_imgs" / f"{space}_from_png.pt"
    if not cache.exists():
        raise SystemExit(
            f"Missing {cache}. Run `analyze_cp_features_rxrx1.py --stage "
            f"png_features --encoder {space}` first."
        )
    d = torch.load(cache, map_location="cpu", weights_only=False)
    feats = (d["features"].numpy() if isinstance(d["features"], torch.Tensor)
             else np.asarray(d["features"])).astype(np.float32)
    fn_to_idx = {fn: i for i, fn in enumerate(d["filenames"])}
    keep = [fn_to_idx[fn] for fn in df_real["FileName_DATA"].tolist() if fn in fn_to_idx]
    if len(keep) != len(df_real):
        logger.warning(
            "[%s] only %d / %d CP real rows have paired features; truncating",
            space, len(keep), len(df_real),
        )
    return feats[keep]


def main() -> None:
    output_dir = Path("outputs/cp_analysis")
    cp_dir     = Path("/mnt/pvc/cellprofiler_outputs")
    out_dir    = output_dir / "morphology_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    reduced, _, scaler = _load_feature_artifacts(output_dir)
    arms = load_rxrx1_subset_arms()
    seen, unseen = arms["seen"], arms["unseen"]

    df_real = _load_real_df(cp_dir, reduced, seen, unseen, scaler)
    y_combo = (df_real["cell_type_id"].astype(int) * 100000
               + df_real["sirna_id"].astype(int)).values
    y_cell  = df_real["cell_type_id"].astype(int).values
    y_sirna = df_real["sirna_id"].astype(int).values
    targets = {"cell": y_cell, "combo": y_combo, "sirna": y_sirna}
    K = {"cell": len(np.unique(y_cell)),
         "combo": len(np.unique(y_combo)),
         "sirna": len(np.unique(y_sirna))}

    # --- CP spaces (already real-fit StandardScaler'd; don't re-scale) ---
    cp_spaces: Dict[str, np.ndarray] = {}
    for k in (5, 15):
        top_feats = _load_top_k_features(output_dir, k)
        cp_spaces[f"CP_top{k}"] = df_real[top_feats].values.astype(np.float32)

    # --- Neural spaces: paired real features (all targets), PCA inside pipeline ---
    X_siglip = _load_real_neural(output_dir, "siglip", df_real)
    X_dino   = _load_real_neural(output_dir, "dinov3", df_real)
    neural_spaces = {
        "SigLIP_full":  (X_siglip, 0),
        "SigLIP_PC30":  (X_siglip, 30),
        "DINOv3_full":  (X_dino,   0),
        "DINOv3_PC30":  (X_dino,   30),
    }

    rows: List[Dict] = []
    for space_name, X in cp_spaces.items():
        for tname, y in targets.items():
            logger.info("[%s × %s]", space_name, tname)
            macro_m, macro_s, micro_m, micro_s = _cv_scores(X, y, pca_n=0)
            chance = 1.0 / K[tname]
            rows.append({
                "space": space_name, "target": tname, "n_classes": K[tname],
                "n_feats": int(X.shape[1]),
                "chance": chance,
                "macro_acc": macro_m, "macro_acc_std": macro_s,
                "micro_acc": micro_m, "micro_acc_std": micro_s,
                "normalized_macro": (macro_m - chance) / max(1 - chance, 1e-12),
            })

    for space_name, (X, pca_n) in neural_spaces.items():
        for tname, y in targets.items():
            logger.info("[%s × %s]  pca_n=%d dim=%d", space_name, tname, pca_n, X.shape[1])
            macro_m, macro_s, micro_m, micro_s = _cv_scores(X, y, pca_n=pca_n)
            chance = 1.0 / K[tname]
            rows.append({
                "space": space_name, "target": tname, "n_classes": K[tname],
                "n_feats": pca_n if pca_n > 0 else int(X.shape[1]),
                "chance": chance,
                "macro_acc": macro_m, "macro_acc_std": macro_s,
                "micro_acc": micro_m, "micro_acc_std": micro_s,
                "normalized_macro": (macro_m - chance) / max(1 - chance, 1e-12),
            })

    df = pd.DataFrame(rows)
    out_path = out_dir / "real_readout_ceiling.csv"
    df.to_csv(out_path, index=False)

    print("\n=== Real-data readout ceiling (5-fold stratified CV, logistic, full-test) ===")
    print(df.to_string(index=False,
                       float_format=lambda x: f"{x:.4f}" if abs(x) < 10 else f"{x:.2f}"))
    print(f"\nWrote {out_path}")

    # -----------------------------------------------------------------------
    # Per-arm CP ceiling — same 50-way classifier, score per arm of the test
    # row's (cell, sirna). Lets the unseen-arm number be compared directly to
    # the gen-unseen-arm readout Δs.
    # -----------------------------------------------------------------------
    arm_of_row = np.array([
        "seen" if (int(c), int(s)) in seen
        else "unseen" if (int(c), int(s)) in unseen
        else "outside"
        for c, s in zip(df_real["cell_type_id"], df_real["sirna_id"])
    ])
    arm_rows: List[Dict] = []
    for k in (5, 15):
        top_feats = _load_top_k_features(output_dir, k)
        X = df_real[top_feats].values.astype(np.float32)
        for tname, y in targets.items():
            logger.info("[CP_top%d × %s] per-arm CV", k, tname)
            res = _cv_scores_by_arm(X, y, arm_of_row)
            chance = 1.0 / K[tname]
            for arm in ("all", "seen", "unseen"):
                r = res[arm]
                arm_rows.append({
                    "space": f"CP_top{k}", "target": tname,
                    "arm": arm,
                    "n_test_avg_per_fold": r["n_test_avg"],
                    "n_classes_in_test": int({
                        "all": K[tname],
                        "seen":   25 if tname == "combo" else (2 if tname == "cell" else len({s for c, s in seen})),
                        "unseen": 25 if tname == "combo" else (3 if tname == "cell" else len({s for c, s in unseen})),
                    }[arm]),
                    "chance_50way": chance,
                    "macro_acc": r["macro_mean"], "macro_acc_std": r["macro_std"],
                    "micro_acc": r["micro_mean"], "micro_acc_std": r["micro_std"],
                    "normalized_macro": (r["macro_mean"] - chance) / max(1 - chance, 1e-12),
                })
    arm_df = pd.DataFrame(arm_rows)
    arm_out = out_dir / "real_readout_ceiling_cp_by_arm.csv"
    arm_df.to_csv(arm_out, index=False)

    print("\n=== CP per-arm real ceiling (50-way classifier, test scored per arm) ===")
    print(arm_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}" if abs(x) < 10 else f"{x:.2f}",
    ))
    print(f"\nWrote {arm_out}")


if __name__ == "__main__":
    main()
