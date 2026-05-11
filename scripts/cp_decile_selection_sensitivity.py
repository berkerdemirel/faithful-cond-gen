"""
CP feature-selection sensitivity on the within-condition decile downstream.

Question: could the within-condition decile signal that exists in DINOv3
("trust bin 0 > bin 9 by ~+0.15 micro on combo") be missed in CP top-15
simply because top-15 F_combo isn't the right CP selection? Try a battery
of CP selections and k values, rerun the same decile downstream, and check
whether any of them resolve the combo signal.

Selections:
  - F_combo  (current CP protocol)
  - F_sirna  (optimize for within-cell sirna discrimination)
  - MI_combo (mutual information, captures non-linear signal)
  - LASSO    (multi-class logistic L1 coefficient mass)
  - PCA      (first-k principal components of reduced-81; linear combinations)
  - all_reduced (skip selection — use all 81 reduced features)

k ∈ {5, 15, 30, 81}. PCA is bounded by min(k, 81).

Output: single CSV + compact markdown table showing trust spread
(bin 0 − bin 9) on celltype and combo for each (selection, k, model).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from _cp_features_utils import (  # noqa: E402
    preprocess_features_fit_on_real, select_features,
)
from analyze_cp_features_rxrx1 import (  # noqa: E402
    MODEL_TO_CP_DIR, REAL_DIR,
    _load_cp_df, _load_feature_artifacts, _load_scores_png,
    is_metadata_column, parse_filename, resolve_models,
)
from cp_decile_binning import _decile_bins_within_condition, _train_test  # noqa: E402
from cp_morphology_validation import DEFAULT_MODELS  # noqa: E402
from faithful_cond_gen.eval.trust_eval.subset_io import load_rxrx1_subset_arms  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_real_reduced(
    cp_dir: Path, reduced: List[str], scaler, seen, unseen,
) -> pd.DataFrame:
    df = _load_cp_df(cp_dir / REAL_DIR / "Image.csv", reduced, seen, unseen, scaler)
    return df[df["split"].isin(["seen", "unseen"])].reset_index(drop=True)


def _compute_unfiltered_features_and_scaler(
    cp_dir: Path,
) -> Tuple[List[str], object, pd.DataFrame]:
    """All non-metadata numeric CP columns, *no* variance / outlier filter.
    Drop only var=0 (truly constant) columns since Pearson r is undefined on
    them. Fit StandardScaler on real over the surviving set."""
    arms = load_rxrx1_subset_arms()
    canonical = arms["seen"] | arms["unseen"]
    csv_path = cp_dir / REAL_DIR / "Image.csv"
    logger.info(f"[unfiltered] reading {csv_path}")
    df_real = pd.read_csv(csv_path, low_memory=False)

    numeric_cols = df_real.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if not is_metadata_column(c)]
    logger.info(f"[unfiltered]   candidate CP features (post metadata-strip): {len(feature_cols)}")

    df_real = df_real.dropna(subset=feature_cols)
    parsed = df_real["FileName_DATA"].map(parse_filename)
    mask = parsed.notna()
    df_real = df_real.loc[mask].copy()
    df_real["cell_type_id"] = parsed[mask].map(lambda t: t[0]).astype(int)
    df_real["sirna_id"]     = parsed[mask].map(lambda t: t[1]).astype(int)
    df_real = df_real[df_real.apply(
        lambda r: (int(r.cell_type_id), int(r.sirna_id)) in canonical, axis=1
    )].copy()
    logger.info(f"[unfiltered]   rows inside canonical 50-pair subset: {len(df_real)}")

    var = df_real[feature_cols].var()
    nonzero = [c for c in feature_cols if var.get(c, 0.0) > 0.0]
    dropped = len(feature_cols) - len(nonzero)
    if dropped:
        logger.info(f"[unfiltered]   dropped {dropped} truly-constant columns (var=0); kept {len(nonzero)}")

    df_real["Source"] = "real"
    df_real_scaled, scaler = preprocess_features_fit_on_real(
        df_real, nonzero, source_col="Source", real_value="real",
    )
    df_real_scaled["split"] = df_real_scaled.apply(
        lambda r: ("seen" if (int(r.cell_type_id), int(r.sirna_id)) in arms["seen"]
                   else "unseen" if (int(r.cell_type_id), int(r.sirna_id)) in arms["unseen"]
                   else "outside"),
        axis=1,
    )
    df_real_scaled = df_real_scaled[df_real_scaled["split"].isin(["seen", "unseen"])].reset_index(drop=True)
    return nonzero, scaler, df_real_scaled


def _compute_kept_features_and_scaler(
    cp_dir: Path, variance_thresh: float = 1e-5, z_thresh: float = 5.0,
) -> Tuple[List[str], object, pd.DataFrame]:
    """Recompute the post-variance+outlier-filter feature set on real rows,
    *before* the |r|>0.7 correlation prune. Fits a StandardScaler on real
    over this kept set. Returns `(kept_features, scaler, df_real_scaled)`
    where `df_real_scaled` has columns FileName_DATA, cell_type_id, sirna_id,
    split, and the scaled kept-feature values."""
    arms = load_rxrx1_subset_arms()
    canonical = arms["seen"] | arms["unseen"]

    csv_path = cp_dir / REAL_DIR / "Image.csv"
    logger.info(f"[kept_pre_corr] reading {csv_path}")
    df_real = pd.read_csv(csv_path, low_memory=False)

    numeric_cols = df_real.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if not is_metadata_column(c)]
    logger.info(f"[kept_pre_corr]   candidate CP features: {len(feature_cols)}")

    df_real = df_real.dropna(subset=feature_cols)
    parsed = df_real["FileName_DATA"].map(parse_filename)
    mask = parsed.notna()
    df_real = df_real.loc[mask].copy()
    df_real["cell_type_id"] = parsed[mask].map(lambda t: t[0]).astype(int)
    df_real["sirna_id"]     = parsed[mask].map(lambda t: t[1]).astype(int)
    df_real = df_real[df_real.apply(
        lambda r: (int(r.cell_type_id), int(r.sirna_id)) in canonical, axis=1
    )].copy()
    logger.info(f"[kept_pre_corr]   rows inside canonical 50-pair subset: {len(df_real)}")

    kept = select_features(df_real, feature_cols, variance_thresh=variance_thresh, z_thresh=z_thresh)
    logger.info(f"[kept_pre_corr]   kept (variance+outlier only): {len(kept)}")

    df_real["Source"] = "real"
    df_real_scaled, scaler = preprocess_features_fit_on_real(
        df_real, kept, source_col="Source", real_value="real",
    )
    # Tag split so we can use _load_real_reduced-compatible downstream code.
    df_real_scaled["split"] = df_real_scaled.apply(
        lambda r: ("seen" if (int(r.cell_type_id), int(r.sirna_id)) in arms["seen"]
                   else "unseen" if (int(r.cell_type_id), int(r.sirna_id)) in arms["unseen"]
                   else "outside"),
        axis=1,
    )
    df_real_scaled = df_real_scaled[df_real_scaled["split"].isin(["seen", "unseen"])].reset_index(drop=True)
    return kept, scaler, df_real_scaled


def _load_gen_reduced(
    cp_dir: Path, output_dir: Path, model: str, encoder_tag: str,
    reduced: List[str], scaler, seen, unseen,
) -> pd.DataFrame:
    scores, _thresh = _load_scores_png(output_dir / MODEL_TO_CP_DIR[model], encoder_tag)
    df = _load_cp_df(cp_dir / MODEL_TO_CP_DIR[model] / "Image.csv", reduced, seen, unseen, scaler)
    df["stem"] = df["FileName_DATA"].map(lambda x: Path(x).stem)
    for key in ("trust_updated", "realism_z", "faithfulness_z"):
        df[key] = df["stem"].map(lambda s: scores.get(s, {}).get(key))
    df = df.dropna(subset=["trust_updated"]).reset_index(drop=True)
    return df


def _dino_aligned_sirna_ranking(
    output_dir: Path, df_real: pd.DataFrame, X_real_81: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Distill DINOv3→sirna signal into CP reduced-81.

    1. Load DINOv3 real features paired with CP rows (same PNG filenames).
    2. Fit multinomial logistic (DINOv3 → sirna) on real. Its decision
       function gives an `(n_real, K_sirna)` target that encodes
       DINO's sirna-predictive direction per sample.
    3. Ridge-regress those K-dim targets from CP reduced-81 (standardized
       inside the ridge already since CP is scaled).
    4. Score each CP column by the L2 norm of its coefficient vector across
       the K sirna classes → higher = more of DINO's sirna signal resides
       on that CP column.

    Returns `(ranking_indices, scores_per_cp, diagnostics)`. Rankings are
    argsort(-scores).
    """
    import torch
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    cache = output_dir / REAL_DIR / "dinov3_from_png.pt"
    if not cache.exists():
        raise SystemExit(f"Missing {cache}; run analyze_cp_features_rxrx1 --stage png_features --encoder dinov3 first.")
    data = torch.load(cache, map_location="cpu", weights_only=False)
    dino_feats = (data["features"].numpy() if isinstance(data["features"], torch.Tensor)
                  else np.asarray(data["features"])).astype(np.float32)
    fn_to_idx = {fn: i for i, fn in enumerate(data["filenames"])}
    row_idx = [fn_to_idx[fn] for fn in df_real["FileName_DATA"].tolist() if fn in fn_to_idx]
    dino_X = dino_feats[row_idx]
    if len(dino_X) != len(X_real_81):
        raise SystemExit(
            f"Pairing failed: CP real {len(X_real_81)} vs DINO paired {len(dino_X)}."
        )

    y_sirna = df_real["sirna_id"].astype(int).values
    dino_scaler = StandardScaler().fit(dino_X)
    dino_z = dino_scaler.transform(dino_X)

    dino_clf = LogisticRegression(
        max_iter=2000, solver="lbfgs", C=1.0, random_state=0,
    ).fit(dino_z, y_sirna)
    train_acc = float((dino_clf.predict(dino_z) == y_sirna).mean())

    # Decision function: (n_real, K)
    logits = dino_clf.decision_function(dino_z)
    if logits.ndim == 1:
        logits = logits[:, None]

    # Ridge: CP reduced-81 → K-dim logits. coef_ shape: (K, 81).
    ridge = Ridge(alpha=1.0).fit(X_real_81, logits)
    pred_logits = ridge.predict(X_real_81)
    # R² per target averaged
    ss_res = np.sum((logits - pred_logits) ** 2, axis=0)
    ss_tot = np.sum((logits - logits.mean(0)) ** 2, axis=0) + 1e-12
    r2_per_class = 1.0 - ss_res / ss_tot

    col_scores = np.linalg.norm(ridge.coef_, axis=0)  # (81,)
    ranking = np.argsort(-col_scores)
    diag = {
        "dino_sirna_train_acc": train_acc,
        "mean_r2_logits":       float(np.mean(r2_per_class)),
        "median_r2_logits":     float(np.median(r2_per_class)),
        "n_classes_K":          int(logits.shape[1]),
    }
    return ranking, col_scores, diag


def _apply_cp_selection(
    X_real_81: np.ndarray, X_gen_81: np.ndarray,
    y_combo: np.ndarray, y_sirna: np.ndarray,
    criterion: str, k: int,
    dino_aligned_ranking: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return (X_real_sel, X_gen_sel, name) for the requested selection."""
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import f_classif, mutual_info_classif
    from sklearn.linear_model import LogisticRegression

    if criterion == "all_reduced":
        return X_real_81, X_gen_81, "all_reduced_81"

    k_eff = min(k, X_real_81.shape[1])

    if criterion == "PCA":
        pca = PCA(n_components=k_eff, random_state=0).fit(X_real_81)
        return pca.transform(X_real_81), pca.transform(X_gen_81), f"PCA{k_eff}"

    if criterion == "DINO_aligned_sirna":
        if dino_aligned_ranking is None:
            raise ValueError("DINO_aligned_sirna requested but ranking not provided.")
        idx = dino_aligned_ranking[:k_eff]
        return X_real_81[:, idx], X_gen_81[:, idx], f"DINO_aligned_sirna_top{k_eff}"

    if criterion == "F_combo":
        F, _ = f_classif(X_real_81, y_combo)
    elif criterion == "F_sirna":
        F, _ = f_classif(X_real_81, y_sirna)
    elif criterion == "MI_combo":
        F = mutual_info_classif(X_real_81, y_combo, random_state=0)
    elif criterion == "LASSO":
        clf = LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=2000).fit(
            X_real_81, y_combo
        )
        F = np.sum(np.abs(clf.coef_), axis=0)
    else:
        raise ValueError(f"unknown criterion: {criterion}")

    idx = np.argsort(-F)[:k_eff]
    return X_real_81[:, idx], X_gen_81[:, idx], f"{criterion}_top{k_eff}"


def _run_one_config(
    X_real: np.ndarray, X_gen: np.ndarray, df_gen: pd.DataFrame,
    labels_real: Dict[str, np.ndarray],
    n_bins: int, n_random_seeds: int,
) -> Dict[str, float]:
    """Run within-condition decile downstream on (X_real, X_gen).
    Return {celltype_trust_spread_{macro, micro}, combo_trust_spread_{macro, micro},
            celltype_random_flat_{macro, micro}, ...}.
    """
    trust_scores = df_gen["trust_updated"].values.astype(float)
    cell_ids  = df_gen["cell_type_id"].astype(int).values
    sirna_ids = df_gen["sirna_id"].astype(int).values
    y_gen_celltype = cell_ids.copy()
    y_gen_combo    = cell_ids * 100000 + sirna_ids

    # Trust: 10 within-condition bins
    bins = _decile_bins_within_condition(trust_scores, cell_ids, sirna_ids, n_bins=n_bins)
    trust_accs: Dict[str, List[float]] = {
        "celltype_macro": [], "celltype_micro": [],
        "combo_macro": [],    "combo_micro":    [],
    }
    for bin_idx, idx in enumerate(bins):
        if len(idx) == 0:
            trust_accs["celltype_macro"].append(np.nan); trust_accs["celltype_micro"].append(np.nan)
            trust_accs["combo_macro"].append(np.nan);    trust_accs["combo_micro"].append(np.nan)
            continue
        r_ct = _train_test(X_gen[idx], y_gen_celltype[idx], X_real, labels_real["cell"])
        r_cm = _train_test(X_gen[idx], y_gen_combo[idx],    X_real, labels_real["combo"])
        trust_accs["celltype_macro"].append(r_ct["macro"])
        trust_accs["celltype_micro"].append(r_ct["micro"])
        trust_accs["combo_macro"].append(r_cm["macro"])
        trust_accs["combo_micro"].append(r_cm["micro"])

    # Random: n_random_seeds within-condition random bins (averaged)
    rand_accs: Dict[str, List[List[float]]] = {
        "celltype_macro": [[] for _ in range(n_bins)],
        "celltype_micro": [[] for _ in range(n_bins)],
        "combo_macro":    [[] for _ in range(n_bins)],
        "combo_micro":    [[] for _ in range(n_bins)],
    }
    for sd in range(n_random_seeds):
        rng = np.random.default_rng(1000 + sd)
        rnd = rng.random(len(X_gen))
        rbins = _decile_bins_within_condition(rnd, cell_ids, sirna_ids, n_bins=n_bins)
        for bin_idx, idx in enumerate(rbins):
            if len(idx) == 0:
                continue
            r_ct = _train_test(X_gen[idx], y_gen_celltype[idx], X_real, labels_real["cell"])
            r_cm = _train_test(X_gen[idx], y_gen_combo[idx],    X_real, labels_real["combo"])
            rand_accs["celltype_macro"][bin_idx].append(r_ct["macro"])
            rand_accs["celltype_micro"][bin_idx].append(r_ct["micro"])
            rand_accs["combo_macro"][bin_idx].append(r_cm["macro"])
            rand_accs["combo_micro"][bin_idx].append(r_cm["micro"])

    out: Dict[str, float] = {}
    for t in ("celltype_macro", "celltype_micro", "combo_macro", "combo_micro"):
        arr = np.asarray(trust_accs[t])
        out[f"trust_bin0_{t}"] = float(arr[0])
        out[f"trust_bin9_{t}"] = float(arr[-1])
        out[f"trust_spread_{t}"] = float(arr[0] - arr[-1])
        # Flatness of random (mean across bins; flat by construction, report mean)
        rand_by_bin = np.asarray([np.mean(rand_accs[t][b]) if rand_accs[t][b] else np.nan
                                  for b in range(n_bins)])
        out[f"random_flat_{t}"]     = float(np.nanmean(rand_by_bin))
        out[f"random_bin_std_{t}"]  = float(np.nanstd(rand_by_bin))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sweep CP feature selections and report within-condition decile downstream spreads."
    )
    p.add_argument("--cp-dir", type=Path, default=Path("/mnt/pvc/cellprofiler_outputs"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/cp_analysis"))
    p.add_argument("--encoder", type=str, default="siglip")
    p.add_argument("--models", type=str, default="repa_marginal",
                   help="Comma-separated (default repa_marginal — biggest DINOv3 effect).")
    p.add_argument("--criteria", type=str,
                   default="F_combo,F_sirna,MI_combo,LASSO,PCA,DINO_aligned_sirna,all_reduced")
    p.add_argument("--k-list", type=str, default="5,15,30,81")
    p.add_argument("--feature-set", type=str, default="reduced_81",
                   choices=["reduced_81", "kept_pre_corr"],
                   help="Which CP feature set to rank over. `reduced_81` uses "
                        "top_features.json's reduced-81 (post corr-prune). "
                        "`kept_pre_corr` recomputes variance+outlier filter "
                        "only, skipping the |r|>0.7 prune.")
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--n-random-seeds", type=int, default=5)
    args = p.parse_args()

    if not args.cp_dir.exists():
        raise SystemExit(f"--cp-dir does not exist: {args.cp_dir}")
    models = resolve_models(args.models)
    criteria = [c.strip() for c in args.criteria.split(",") if c.strip()]
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]

    out_dir = args.output_dir / "morphology_validation" / "decile_binning"
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = load_rxrx1_subset_arms()
    seen, unseen = arms["seen"], arms["unseen"]

    if args.feature_set == "reduced_81":
        reduced, _, scaler = _load_feature_artifacts(args.output_dir)
        df_real = _load_real_reduced(args.cp_dir, reduced, scaler, seen, unseen)
    else:
        reduced, scaler, df_real = _compute_kept_features_and_scaler(args.cp_dir)
    logger.info(f"[main] feature_set={args.feature_set} with {len(reduced)} features")
    X_real_81 = df_real[reduced].values.astype(np.float32)
    y_real_combo = (df_real["cell_type_id"].astype(int) * 100000
                    + df_real["sirna_id"].astype(int)).values
    y_real_sirna = df_real["sirna_id"].astype(int).values
    labels_real = {
        "cell":  df_real["cell_type_id"].astype(int).values,
        "combo": y_real_combo,
        "sirna": y_real_sirna,
    }

    # DINO-distilled CP ranking (computed once from real; independent of model).
    dino_ranking: Optional[np.ndarray] = None
    dino_diag: Dict[str, float] = {}
    if "DINO_aligned_sirna" in criteria:
        logger.info("[DINO_aligned_sirna] computing ranking (one-time)")
        dino_ranking, dino_scores, dino_diag = _dino_aligned_sirna_ranking(
            args.output_dir, df_real, X_real_81,
        )
        logger.info(
            "  train acc DINO→sirna=%.3f  mean R² ridge(CP→logits)=%.3f",
            dino_diag["dino_sirna_train_acc"], dino_diag["mean_r2_logits"],
        )
        top15 = dino_ranking[:15]
        logger.info("  top-15 CP features aligned with DINO sirna signal:")
        for i, j in enumerate(top15, 1):
            logger.info("   %2d. %-60s score=%.3f", i, reduced[j], dino_scores[j])

        # Also print the F_sirna top-15 for comparison.
        from sklearn.feature_selection import f_classif
        F, _ = f_classif(X_real_81, y_real_sirna)
        f_rank = np.argsort(-F)
        logger.info("  top-15 CP by F_sirna (for comparison):")
        for i, j in enumerate(f_rank[:15], 1):
            logger.info("   %2d. %-60s F=%.2f", i, reduced[j], F[j])
        dino_set = set(dino_ranking[:15].tolist())
        f_set = set(f_rank[:15].tolist())
        overlap = dino_set & f_set
        logger.info("  overlap(top-15 DINO vs top-15 F_sirna) = %d / 15", len(overlap))

    rows: List[Dict] = []
    for model in models:
        logger.info(f"[{model}] loading gen reduced-81")
        df_gen = _load_gen_reduced(
            args.cp_dir, args.output_dir, model, args.encoder,
            reduced, scaler, seen, unseen,
        )
        X_gen_81 = df_gen[reduced].values.astype(np.float32)

        for criterion in criteria:
            # For PCA & all_reduced, k=81 is max; avoid duplicates for constants
            k_run = k_list if criterion != "all_reduced" else [X_real_81.shape[1]]
            k_run_dedup: List[int] = []
            for k in k_run:
                if criterion == "PCA":
                    k_eff = min(k, X_real_81.shape[1])
                    if k_eff in k_run_dedup:
                        continue
                    k_run_dedup.append(k_eff)
                else:
                    if k in k_run_dedup:
                        continue
                    k_run_dedup.append(k)

            for k in k_run_dedup:
                logger.info(f"  [{criterion} k={k}] selecting + running deciles")
                X_real_sel, X_gen_sel, name = _apply_cp_selection(
                    X_real_81, X_gen_81, y_real_combo, y_real_sirna, criterion, k,
                    dino_aligned_ranking=dino_ranking,
                )
                res = _run_one_config(
                    X_real_sel, X_gen_sel, df_gen, labels_real,
                    n_bins=args.n_bins, n_random_seeds=args.n_random_seeds,
                )
                rows.append({
                    "model": model,
                    "criterion": criterion,
                    "k": k,
                    "name": name,
                    "n_feats_used": int(X_real_sel.shape[1]),
                    **res,
                })

    df = pd.DataFrame(rows)
    out_path = out_dir / "decile_selection_sensitivity_cp.csv"
    df.to_csv(out_path, index=False)

    # Compact print focused on the key question: does any selection reveal
    # a meaningful trust spread on combo (DINOv3 saw +0.12 to +0.15 micro)?
    print("\n=== CP selection sensitivity — within-condition decile downstream ===")
    cols = [
        "model", "criterion", "k", "n_feats_used",
        "trust_bin0_celltype_micro", "trust_bin9_celltype_micro",
        "trust_spread_celltype_micro", "random_flat_celltype_micro",
        "trust_bin0_combo_micro",    "trust_bin9_combo_micro",
        "trust_spread_combo_micro",  "random_flat_combo_micro",
    ]
    print(df[cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}" if abs(x) < 10 else f"{x:.2f}",
    ))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
