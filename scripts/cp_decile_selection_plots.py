"""
Full decile curves for best CP configs vs DINOv3 reference.

Protocol per config:
  - Choose a CP feature set {reduced_81, kept_pre_corr} and a selection
    criterion + k (or PCA / all).
  - Within-condition bin gen by trust, realism, faithfulness, random (5 seeds).
  - For each bin, train logistic on 500 gen CP samples; test on real.
  - Plot per-bin accuracy for celltype + combo (micro), side by side.
  - Overlay the DINOv3 reference curve loaded from
    `outputs/trust_evaluation_mahalanobis_dinov3_rxrx1/`.

Output:
  outputs/cp_analysis/morphology_validation/decile_binning/
    decile_curves_{model}.png
    decile_curves_{model}.csv  (per-config per-bin accuracies)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from analyze_cp_features_rxrx1 import (  # noqa: E402
    MODEL_TO_CP_DIR, REAL_DIR,
    _load_cp_df, _load_feature_artifacts, _load_scores_png,
    resolve_models,
)
from cp_decile_binning import _decile_bins_within_condition, _train_test  # noqa: E402
from cp_decile_selection_sensitivity import (  # noqa: E402
    _apply_cp_selection,
    _compute_kept_features_and_scaler,
    _dino_aligned_sirna_ranking,
    _load_gen_reduced,
    _load_real_reduced,
)
from cp_morphology_validation import DEFAULT_MODELS  # noqa: E402
from faithful_cond_gen.eval.trust_eval.subset_io import load_rxrx1_subset_arms  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Hand-picked configs — one line per curve on the plot.
# Default: just the best-combo CP config. Set `--configs all` to keep the
# earlier 5-config comparison.
CONFIGS_BEST: List[Dict] = [
    {"label": "CP kept-621 all-621  (best combo ρ)",
     "feature_set": "kept_pre_corr", "criterion": "all_reduced", "k": 621,
     "color": "#0072B2", "linestyle": "-"},
]

CONFIGS_FULL: List[Dict] = [
    {"label": "CP reduced-81 F_combo top-15",
     "feature_set": "reduced_81", "criterion": "F_combo", "k": 15,
     "color": "#999999", "linestyle": "-"},
    {"label": "CP reduced-81 all-81",
     "feature_set": "reduced_81", "criterion": "all_reduced", "k": 81,
     "color": "#777777", "linestyle": "--"},
    {"label": "CP kept-621 F_sirna top-81",
     "feature_set": "kept_pre_corr", "criterion": "F_sirna", "k": 81,
     "color": "#009E73", "linestyle": "-"},
    {"label": "CP kept-621 DINO_aligned top-500",
     "feature_set": "kept_pre_corr", "criterion": "DINO_aligned_sirna", "k": 500,
     "color": "#CC6677", "linestyle": "-"},
    {"label": "CP kept-621 all-621  (best combo ρ)",
     "feature_set": "kept_pre_corr", "criterion": "all_reduced", "k": 621,
     "color": "#0072B2", "linestyle": "-"},
]

# real-data CV ceilings (5-fold logistic, macro) — used to annotate the plot
# so the reader can see each feature space's absolute maximum.
CEILINGS = {
    "reduced_81":    {"celltype_macro": 0.825, "combo_macro": 0.341, "sirna_macro": 0.310},
    "kept_pre_corr": None,  # filled at runtime
    "dinov3":        {"celltype_macro": 0.997, "combo_macro": 0.659, "sirna_macro": 0.636},
}


def _run_config(
    X_real: np.ndarray, X_gen: np.ndarray, df_gen: pd.DataFrame,
    labels_real: Dict[str, np.ndarray],
    n_bins: int, n_random_seeds: int,
    pipeline: str = "standardize",
) -> pd.DataFrame:
    """Return per-bin per-ranking CSV for one config. Columns: ranking,
    bin_idx, target, macro, micro, seed."""
    cell_ids  = df_gen["cell_type_id"].astype(int).values
    sirna_ids = df_gen["sirna_id"].astype(int).values
    y_gen_celltype = cell_ids.copy()
    y_gen_combo    = cell_ids * 100000 + sirna_ids
    y_gen_targets  = {"celltype": y_gen_celltype, "subset": y_gen_combo}

    rows: List[Dict] = []
    scores_by_rank = {
        "trust":        df_gen["trust_updated"].values.astype(float),
        "realism":      df_gen["realism_z"].values.astype(float),
        "faithfulness": df_gen["faithfulness_z"].values.astype(float),
    }
    for rname, scores in scores_by_rank.items():
        bins = _decile_bins_within_condition(scores, cell_ids, sirna_ids, n_bins=n_bins)
        for bin_idx, idx in enumerate(bins):
            if len(idx) == 0:
                continue
            for tname, y_test_col in [("celltype", "cell"), ("subset", "combo")]:
                r = _train_test(X_gen[idx], y_gen_targets[tname][idx],
                                X_real, labels_real[y_test_col],
                                pipeline=pipeline)
                rows.append({"ranking": rname, "bin_idx": bin_idx,
                             "target": tname, "seed": -1,
                             "macro": r["macro"], "micro": r["micro"]})

    # Random: 5 seeds of within-condition random bins.
    for sd in range(n_random_seeds):
        rng = np.random.default_rng(1000 + sd)
        rnd = rng.random(len(X_gen))
        rbins = _decile_bins_within_condition(rnd, cell_ids, sirna_ids, n_bins=n_bins)
        for bin_idx, idx in enumerate(rbins):
            if len(idx) == 0:
                continue
            for tname, y_test_col in [("celltype", "cell"), ("subset", "combo")]:
                r = _train_test(X_gen[idx], y_gen_targets[tname][idx],
                                X_real, labels_real[y_test_col],
                                pipeline=pipeline)
                rows.append({"ranking": "random", "bin_idx": bin_idx,
                             "target": tname, "seed": sd,
                             "macro": r["macro"], "micro": r["micro"]})
    return pd.DataFrame(rows)


def _compute_dinov3_reference(
    cp_dir: Path, output_dir: Path, model: str, encoder_tag: str,
    n_bins: int, n_random_seeds: int, pipeline: str,
) -> Optional[pd.DataFrame]:
    """Compute DINOv3 decile reference curves using the chosen pipeline on the
    paired canonical-50 real pool. This replaces loading the pre-computed
    trust-eval CSV (which is locked to the trust-eval pipeline recipe)."""
    import torch
    real_cache = output_dir / REAL_DIR / "dinov3_from_png.pt"
    gen_cache  = output_dir / MODEL_TO_CP_DIR[model] / "dinov3_from_png.pt"
    if not (real_cache.exists() and gen_cache.exists()):
        logger.warning(f"missing DINO caches: {real_cache} / {gen_cache}")
        return None

    rd = torch.load(real_cache, map_location="cpu", weights_only=False)
    real_feats = (rd["features"].numpy() if torch.is_tensor(rd["features"])
                  else np.asarray(rd["features"])).astype(np.float32)
    real_ct = rd["metadata"]["cell_type_id"].numpy() if torch.is_tensor(rd["metadata"]["cell_type_id"]) else np.asarray(rd["metadata"]["cell_type_id"])
    real_sr = rd["metadata"]["sirna_id"].numpy()     if torch.is_tensor(rd["metadata"]["sirna_id"])     else np.asarray(rd["metadata"]["sirna_id"])

    gd = torch.load(gen_cache, map_location="cpu", weights_only=False)
    gen_feats = (gd["features"].numpy() if torch.is_tensor(gd["features"])
                 else np.asarray(gd["features"])).astype(np.float32)
    gen_ct = gd["metadata"]["cell_type_id"].numpy() if torch.is_tensor(gd["metadata"]["cell_type_id"]) else np.asarray(gd["metadata"]["cell_type_id"])
    gen_sr = gd["metadata"]["sirna_id"].numpy()     if torch.is_tensor(gd["metadata"]["sirna_id"])     else np.asarray(gd["metadata"]["sirna_id"])
    gen_fns = list(gd["filenames"])

    # Pull trust / realism / faithfulness scores from scores_png_<encoder>.json.
    scores, _ = _load_scores_png(output_dir / MODEL_TO_CP_DIR[model], encoder_tag)
    def _col(key: str) -> np.ndarray:
        return np.array([
            scores.get(Path(fn).stem, {}).get(key, np.nan) for fn in gen_fns
        ], dtype=float)
    trust = _col("trust_updated"); realism = _col("realism_z"); faith = _col("faithfulness_z")
    valid = np.isfinite(trust) & np.isfinite(realism) & np.isfinite(faith)
    gen_feats = gen_feats[valid]; gen_ct = gen_ct[valid]; gen_sr = gen_sr[valid]
    trust = trust[valid]; realism = realism[valid]; faith = faith[valid]

    labels_real = {
        "cell":  real_ct.astype(int),
        "combo": (real_ct.astype(int) * 100000 + real_sr.astype(int)),
    }
    y_gen_targets = {
        "celltype": gen_ct.astype(int),
        "subset":   (gen_ct.astype(int) * 100000 + gen_sr.astype(int)),
    }

    rows: List[Dict] = []
    for rname, s in (("trust", trust), ("realism", realism), ("faithfulness", faith)):
        bins = _decile_bins_within_condition(s, gen_ct.astype(int), gen_sr.astype(int), n_bins=n_bins)
        for bin_idx, idx in enumerate(bins):
            if len(idx) == 0:
                continue
            for tname, y_test_col in [("celltype", "cell"), ("subset", "combo")]:
                r = _train_test(gen_feats[idx], y_gen_targets[tname][idx],
                                real_feats, labels_real[y_test_col],
                                pipeline=pipeline)
                rows.append({"ranking": rname, "bin_idx": bin_idx, "target": tname,
                             "seed": -1, "macro": r["macro"], "micro": r["micro"]})
    for sd in range(n_random_seeds):
        rng = np.random.default_rng(1000 + sd)
        rnd = rng.random(len(gen_feats))
        rbins = _decile_bins_within_condition(rnd, gen_ct.astype(int), gen_sr.astype(int), n_bins=n_bins)
        for bin_idx, idx in enumerate(rbins):
            if len(idx) == 0:
                continue
            for tname, y_test_col in [("celltype", "cell"), ("subset", "combo")]:
                r = _train_test(gen_feats[idx], y_gen_targets[tname][idx],
                                real_feats, labels_real[y_test_col],
                                pipeline=pipeline)
                rows.append({"ranking": "random", "bin_idx": bin_idx, "target": tname,
                             "seed": sd, "macro": r["macro"], "micro": r["micro"]})
    return pd.DataFrame(rows)


def _load_dinov3_reference(model: str) -> Optional[pd.DataFrame]:
    """Load the existing DINOv3 decile downstream CSV for this model."""
    base = Path("outputs/trust_evaluation_mahalanobis_dinov3_rxrx1")
    rows: List[Dict] = []
    for target_label, filename in [
        ("celltype", f"rxrx1_downstream_bin_celltype_{model}_dinov3_summary.csv"),
        ("subset",   f"rxrx1_downstream_bin_subset_{model}_dinov3_summary.csv"),
    ]:
        p = base / filename
        if not p.exists():
            logger.warning(f"DINOv3 reference missing: {p}")
            return None
        d = pd.read_csv(p)
        d["target"] = target_label
        d["micro"] = d["accuracy"]
        d["macro"] = float("nan")
        d["seed"] = -1
        d = d.rename(columns={"ranking_mode": "ranking"})
        rows.append(d[["ranking", "bin_idx", "target", "seed", "macro", "micro"]])
    return pd.concat(rows, ignore_index=True)


def _plot_single_space(
    curves: List[Tuple[str, str, str, pd.DataFrame]],
    model: str, space_label: str,
    out_path: Path, n_bins: int,
) -> None:
    """Two-panel figure (celltype + combo) for a single feature space.
    `curves`: list of (label, color, linestyle, per-bin df)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    targets = [("celltype", "cell-type (4-way)"),
               ("subset",   "subset / combo (50-way)")]
    for ax, (target, pretty) in zip(axes, targets):
        for label, color, linestyle, df in curves:
            if df is None:
                continue
            trust = df[(df.target == target) & (df.ranking == "trust")].sort_values("bin_idx")
            rand  = df[(df.target == target) & (df.ranking == "random")]
            if not trust.empty:
                ax.plot(trust["bin_idx"], trust["micro"], marker="o", color=color,
                        linestyle=linestyle, linewidth=2.2, markersize=7,
                        label=f"trust  {label}")
            if not rand.empty:
                agg = rand.groupby("bin_idx")["micro"].agg(["mean", "std"]).reset_index()
                ax.fill_between(agg["bin_idx"],
                                agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                                color=color, alpha=0.15, linewidth=0,
                                label=f"random ±1σ  {label}")
                ax.plot(agg["bin_idx"], agg["mean"],
                        color=color, linestyle="--", linewidth=1.2, alpha=0.7,
                        label=f"random mean  {label}")
        ax.set_xlabel("Decile bin  (0 = best trust / highest rank)")
        ax.set_ylabel("Accuracy (gen-trained classifier, micro, on real test)")
        ax.set_title(f"{model} — {pretty}  (within-condition bins, SigLIP trust)")
        ax.grid(linestyle=":", alpha=0.3)
        ax.set_xticks(range(n_bins))
        ax.legend(fontsize=8, loc="best", frameon=True)
    fig.suptitle(f"{model} — decile curves — {space_label}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def _plot_curves(
    configs: List[Dict],
    results: List[Tuple[str, str, pd.DataFrame]],
    dinov3_df: Optional[pd.DataFrame],
    model: str,
    out_path: Path,
    n_bins: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    targets = [("celltype", "cell-type (4-way)"),
               ("subset",   "subset / combo (50-way)")]

    for ax, (target, pretty) in zip(axes, targets):
        # CP config(s): trust curves + random bands
        for cfg, (label, color, df) in zip(configs, results):
            if df is None:
                continue
            trust = df[(df.target == target) & (df.ranking == "trust")].sort_values("bin_idx")
            rand  = df[(df.target == target) & (df.ranking == "random")]

            if not trust.empty:
                ax.plot(trust["bin_idx"], trust["micro"], marker="o", color=color,
                        linestyle=cfg.get("linestyle", "-"), linewidth=2.2, markersize=7,
                        label=f"trust  {label}")

            if not rand.empty:
                agg = rand.groupby("bin_idx")["micro"].agg(["mean", "std"]).reset_index()
                ax.fill_between(agg["bin_idx"],
                                agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                                color=color, alpha=0.15, linewidth=0,
                                label=f"random ±1σ  {label}")
                ax.plot(agg["bin_idx"], agg["mean"],
                        color=color, linestyle="--", linewidth=1.2, alpha=0.7,
                        label=f"random mean  {label}")

        # DINOv3 reference: trust curve + random mean
        if dinov3_df is not None:
            sub = dinov3_df[(dinov3_df.target == target) & (dinov3_df.ranking == "trust")].sort_values("bin_idx")
            if not sub.empty:
                ax.plot(sub["bin_idx"], sub["micro"], marker="s", color="#CC3311",
                        linestyle="-", linewidth=2.3, markersize=7,
                        label="trust  DINOv3 full (ref)")
            ref_rand = dinov3_df[(dinov3_df.target == target) & (dinov3_df.ranking == "random")]
            if not ref_rand.empty:
                vals = ref_rand.groupby("bin_idx")["micro"].mean().reset_index()
                ax.plot(vals["bin_idx"], vals["micro"],
                        color="#CC3311", linestyle="--", linewidth=1.2, alpha=0.7,
                        label="random mean  DINOv3 full")

        ax.set_xlabel("Decile bin  (0 = best trust / highest rank)")
        ax.set_ylabel("Accuracy (gen-trained classifier, micro, on real test)")
        ax.set_title(f"{model} — {pretty}  (within-condition bins, SigLIP trust)")
        ax.grid(linestyle=":", alpha=0.3)
        ax.set_xticks(range(n_bins))
        ax.legend(fontsize=8, loc="best", frameon=True)

    fig.suptitle(f"{model} — decile curves: best CP config vs DINOv3 reference", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot full decile curves for best CP configs vs DINOv3 reference."
    )
    p.add_argument("--cp-dir", type=Path, default=Path("/mnt/pvc/cellprofiler_outputs"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/cp_analysis"))
    p.add_argument("--encoder", type=str, default="siglip")
    p.add_argument("--models", type=str, default="repa_marginal,repa_siglip_marginal,vanilla_marginal")
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--n-random-seeds", type=int, default=5)
    p.add_argument("--configs", type=str, default="best",
                   choices=["best", "all"],
                   help="`best` plots only kept-621 all-621 (best combo ρ). "
                        "`all` plots 5 CP configs as before.")
    p.add_argument("--downstream-pipeline", type=str, default="standardize",
                   choices=["standardize", "l2_no_scaler"],
                   help="Downstream classifier recipe. `standardize` = "
                        "StandardScaler + LR (default). `l2_no_scaler` = "
                        "L2-normalize rows then bare LR (trust-eval recipe).")
    args = p.parse_args()
    configs = CONFIGS_BEST if args.configs == "best" else CONFIGS_FULL

    models = resolve_models(args.models)
    out_dir = args.output_dir / "morphology_validation" / "decile_binning"
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = load_rxrx1_subset_arms()
    seen, unseen = arms["seen"], arms["unseen"]

    # Prepare both feature sets once.
    logger.info("loading reduced-81 ...")
    reduced_81, _, scaler_81 = _load_feature_artifacts(args.output_dir)
    df_real_81 = _load_real_reduced(args.cp_dir, reduced_81, scaler_81, seen, unseen)
    X_real_81_81 = df_real_81[reduced_81].values.astype(np.float32)

    logger.info("loading kept_pre_corr ...")
    kept_621, scaler_621, df_real_621 = _compute_kept_features_and_scaler(args.cp_dir)
    X_real_81_621 = df_real_621[kept_621].values.astype(np.float32)

    labels_real = {
        "reduced_81":   {"cell":  df_real_81["cell_type_id"].astype(int).values,
                         "combo": (df_real_81["cell_type_id"].astype(int) * 100000
                                   + df_real_81["sirna_id"].astype(int)).values,
                         "sirna": df_real_81["sirna_id"].astype(int).values},
        "kept_pre_corr":{"cell":  df_real_621["cell_type_id"].astype(int).values,
                         "combo": (df_real_621["cell_type_id"].astype(int) * 100000
                                   + df_real_621["sirna_id"].astype(int)).values,
                         "sirna": df_real_621["sirna_id"].astype(int).values},
    }

    # Precompute DINO_aligned ranking once per feature_set.
    dino_rankings: Dict[str, np.ndarray] = {}
    need_dino_81  = any(c["feature_set"] == "reduced_81"    and c["criterion"] == "DINO_aligned_sirna" for c in configs)
    need_dino_621 = any(c["feature_set"] == "kept_pre_corr" and c["criterion"] == "DINO_aligned_sirna" for c in configs)
    if need_dino_81:
        logger.info("computing DINO_aligned ranking on reduced-81")
        r, _, _ = _dino_aligned_sirna_ranking(args.output_dir, df_real_81, X_real_81_81)
        dino_rankings["reduced_81"] = r
    if need_dino_621:
        logger.info("computing DINO_aligned ranking on kept_pre_corr")
        r, _, _ = _dino_aligned_sirna_ranking(args.output_dir, df_real_621, X_real_81_621)
        dino_rankings["kept_pre_corr"] = r

    for model in models:
        logger.info(f"[{model}]")
        gen_by_set: Dict[str, Tuple[np.ndarray, pd.DataFrame]] = {}
        if any(c["feature_set"] == "reduced_81" for c in configs):
            df_gen = _load_gen_reduced(args.cp_dir, args.output_dir, model, args.encoder,
                                       reduced_81, scaler_81, seen, unseen)
            gen_by_set["reduced_81"] = (df_gen[reduced_81].values.astype(np.float32), df_gen)
        if any(c["feature_set"] == "kept_pre_corr" for c in configs):
            df_gen = _load_gen_reduced(args.cp_dir, args.output_dir, model, args.encoder,
                                       kept_621, scaler_621, seen, unseen)
            gen_by_set["kept_pre_corr"] = (df_gen[kept_621].values.astype(np.float32), df_gen)

        results: List[Tuple[str, str, pd.DataFrame]] = []
        per_bin_rows: List[Dict] = []
        for cfg in configs:
            fs = cfg["feature_set"]
            X_real_block = X_real_81_81 if fs == "reduced_81" else X_real_81_621
            X_gen_block, df_gen = gen_by_set[fs]

            y_combo = labels_real[fs]["combo"]
            y_sirna = labels_real[fs]["sirna"]
            X_real_sel, X_gen_sel, name = _apply_cp_selection(
                X_real_block, X_gen_block, y_combo, y_sirna,
                cfg["criterion"], cfg["k"],
                dino_aligned_ranking=dino_rankings.get(fs),
            )
            logger.info(f"  {cfg['label']}  ({name}, shape real={X_real_sel.shape}, gen={X_gen_sel.shape})")
            df_res = _run_config(
                X_real_sel, X_gen_sel, df_gen, labels_real[fs],
                n_bins=args.n_bins, n_random_seeds=args.n_random_seeds,
                pipeline=args.downstream_pipeline,
            )
            df_res["config_label"] = cfg["label"]
            df_res["feature_set"]  = fs
            df_res["criterion"]    = cfg["criterion"]
            df_res["k"]            = cfg["k"]
            per_bin_rows.append(df_res)
            results.append((cfg["label"], cfg["color"], df_res))

        combined = pd.concat(per_bin_rows, ignore_index=True)
        # Drop suffix when pipeline is the default "standardize".
        suf = "" if args.downstream_pipeline == "standardize" else f"_{args.downstream_pipeline}"
        csv_path = out_dir / f"decile_curves_{model}{suf}_cp.csv"
        combined.to_csv(csv_path, index=False)

        dinov3_df = _compute_dinov3_reference(
            args.cp_dir, args.output_dir, model, args.encoder,
            n_bins=args.n_bins, n_random_seeds=args.n_random_seeds,
            pipeline=args.downstream_pipeline,
        )
        if dinov3_df is not None:
            dinov3_df.to_csv(out_dir / f"decile_curves_{model}{suf}_dinov3.csv", index=False)

        # CP-only plot
        cp_curves = [(cfg["label"], cfg["color"], cfg.get("linestyle", "-"), df)
                     for cfg, (_, _, df) in zip(configs, results)]
        _plot_single_space(
            cp_curves, model,
            space_label=f"CP features (pipeline={args.downstream_pipeline})",
            out_path=out_dir / f"decile_curves_{model}{suf}_cp.png",
            n_bins=args.n_bins,
        )
        # DINO-only plot
        if dinov3_df is not None:
            dino_curves = [("DINOv3 full (1024-d)", "#CC3311", "-", dinov3_df)]
            _plot_single_space(
                dino_curves, model,
                space_label=f"DINOv3 features (pipeline={args.downstream_pipeline})",
                out_path=out_dir / f"decile_curves_{model}{suf}_dinov3.png",
                n_bins=args.n_bins,
            )


if __name__ == "__main__":
    main()
