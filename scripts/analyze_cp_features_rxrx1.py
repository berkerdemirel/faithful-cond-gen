"""
CellProfiler feature analysis for rxrx1 — answers six questions on the unseen split.

  Q1. How many gen samples pass FPR@95?
      → --stage png_scores        (acceptance + thresholds per model)
  Q2. Do trust-selected gen recover real perturbation directions?
      → --stage ate_cross         (22×22 cross-combo Pearson r, specificity vs random)
  Q3. Does the trust score rank gen by morphology L2 to real?
      → --stage trust_ladder      (per-combo trust quartile → centroid L2 to real)
  Q4. Does the per-combo trust rank identify the right real combo?
      → --stage perturbation_corr (25×25 Pearson r on means, top-1 acc + specificity)
  Q5. Does the distributional match to real improve with trust rank?
      → --stage mmd_bins          (per-combo RBF MMD² per bin, median across combos)
  Q6. Can a real-trained kNN classify trust-selected gen as the right combo?
      → --stage knn_conditioning  (train on real unseen, predict on gen_bin, per-combo acc)

  --stage summary prints all six tables in markdown.

Pipeline (run in order on a fresh `outputs/cp_analysis/`):
  1. --stage features           # CP feature selection on real, top-k by ANOVA F
  2. --stage png_features       # DINOv3 mean-patch on the exact PNGs CP processed
  3. --stage png_scores         # Fit Mahalanobis on legacy real, score PNG gen
  4. --stage ate_cross          # Q2
  5. --stage trust_ladder       # Q3
  6. --stage perturbation_corr  # Q4
  7. --stage mmd_bins           # Q5
  8. --stage knn_conditioning   # Q6
  9. --stage summary            # All six tables
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # for _cp_features_utils

from _cp_features_utils import (  # noqa: E402
    preprocess_features_fit_on_real,
    remove_highly_correlated_features,
    select_features,
)
from faithful_cond_gen.eval.trust_eval.condition_utils import (  # noqa: E402
    filter_feats_and_meta_by_seen_combos,
)
from faithful_cond_gen.eval.trust_eval.config import (  # noqa: E402
    CONDITION_ATTRS,
    RXRX1_HELDOUT_PAIRS,
)
from faithful_cond_gen.eval.trust_eval.feature_io import (  # noqa: E402
    apply_normalization,
    load_features_for_dataset,
)
from faithful_cond_gen.eval.trust_eval.metrics_kid import calculate_kid_same_m  # noqa: E402
from faithful_cond_gen.eval.trust_eval.scoring_core import (  # noqa: E402
    fit_trust_scoring_components,
    score_trust_from_components,
)
from faithful_cond_gen.eval.trust_eval.subset_io import load_rxrx1_subset_arms  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Trust-eval model key → CellProfiler / RGB-PNG subdir name.
MODEL_TO_CP_DIR: Dict[str, str] = {
    "vanilla_full":        "rxrx1_vanilla_full",
    "vanilla_marginal":    "rxrx1_vanilla_marginal",
    "repa_full":           "rxrx1_repa_full",
    "repa_marginal":       "rxrx1_repa_marginal",
    "repa_siglip_full":    "rxrx1_repa_siglip_full",
    "repa_siglip_marginal":"rxrx1_repa_siglip_marginal",
}
REAL_DIR = "real_imgs"

# PNG filenames like "cell0_sirna1110_0.png" — parse cell, sirna, sample index.
_FILENAME_RE = re.compile(r"^cell(\d+)_sirna(\d+)_(\d+)\.(png|pt)$")

# Column names in CP Image.csv that are bookkeeping rather than measurements.
_META_SUBSTRINGS = (
    "metadata", "filename", "pathname", "imagenumber", "objectnumber",
    "executiontime", "moduleerror", "series", "frame",
    "group_index", "group_number", "url",
)


def parse_filename(fn: str) -> Optional[Tuple[int, int, int]]:
    m = _FILENAME_RE.match(fn.strip())
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def is_metadata_column(col: str) -> bool:
    low = col.lower()
    return any(s in low for s in _META_SUBSTRINGS)


def resolve_models(selector: str) -> List[str]:
    if selector == "all":
        return list(MODEL_TO_CP_DIR.keys())
    keys = [k.strip() for k in selector.split(",") if k.strip()]
    bad = [k for k in keys if k not in MODEL_TO_CP_DIR]
    if bad:
        raise SystemExit(f"Unknown model key(s): {bad}. Valid: {list(MODEL_TO_CP_DIR.keys())}")
    return keys


# ---------------------------------------------------------------------------
# Shared helpers: loading artifacts + CP rows
# ---------------------------------------------------------------------------


def _load_feature_artifacts(output_dir: Path) -> Tuple[List[str], List[str], "StandardScaler"]:  # noqa: F821
    reduced_path = output_dir / "reduced_features.json"
    scaler_path  = output_dir / "scaler.pkl"
    top_path     = output_dir / "top_features.json"
    if not reduced_path.exists() or not scaler_path.exists() or not top_path.exists():
        raise SystemExit(
            f"Missing feature artifacts in {output_dir}. Run --stage features first."
        )
    reduced = json.loads(reduced_path.read_text())["features"]
    top     = [row["feature"] for row in json.loads(top_path.read_text())["features"]]
    with open(scaler_path, "rb") as fh:
        bundle = pickle.load(fh)
    return reduced, top, bundle["scaler"]


def _load_cp_df(
    csv_path: Path,
    reduced: List[str],
    seen: Set[Tuple[int, int]],
    unseen: Set[Tuple[int, int]],
    scaler,
) -> pd.DataFrame:
    """Load an Image.csv, keep only reduced features + FileName_DATA, parse split, scale.
    Returns rows with columns: FileName_DATA, cell_type_id, sirna_id, sample_idx,
    split ∈ {seen, unseen, outside}, and the scaled numeric features.
    """
    usecols = ["FileName_DATA"] + reduced
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    parsed = df["FileName_DATA"].map(parse_filename)
    mask = parsed.notna()
    df = df.loc[mask].copy()
    df["cell_type_id"] = parsed[mask].map(lambda t: t[0]).astype(int)
    df["sirna_id"]     = parsed[mask].map(lambda t: t[1]).astype(int)
    df["sample_idx"]   = parsed[mask].map(lambda t: t[2]).astype(int)

    def _split(row):
        k = (int(row.cell_type_id), int(row.sirna_id))
        return "seen" if k in seen else "unseen" if k in unseen else "outside"
    df["split"] = df.apply(_split, axis=1)

    df = df.dropna(subset=reduced).copy()
    df[reduced] = scaler.transform(df[reduced])
    return df


# ---------------------------------------------------------------------------
# STAGE 1: features (CP feature selection on real, top-k by ANOVA F)
# ---------------------------------------------------------------------------


def stage_features(
    cp_dir: Path,
    output_dir: Path,
    variance_thresh: float,
    z_thresh: float,
    corr_thresh: float,
    top_k: int,
) -> None:
    """
    On real CP rows only:
      1. drop metadata/non-numeric columns
      2. drop features with low variance or |z| > z_thresh (outlier columns)
      3. drop one of each highly-correlated pair (|r| > corr_thresh)
      4. fit StandardScaler on what remains
      5. pick top-k most discriminative CP features by ANOVA F over the
         (cell, sirna) combo factor. Log F_cell and F_sirna for interpretability.

    Outputs:
      outputs/cp_analysis/reduced_features.json
      outputs/cp_analysis/scaler.pkl
      outputs/cp_analysis/top_features.json
    """
    from sklearn.feature_selection import f_classif

    output_dir.mkdir(parents=True, exist_ok=True)
    arms = load_rxrx1_subset_arms()
    canonical = arms["seen"] | arms["unseen"]

    real_csv = cp_dir / REAL_DIR / "Image.csv"
    logger.info(f"Loading real CP features from {real_csv}")
    df_real = pd.read_csv(real_csv, low_memory=False)

    numeric_cols = df_real.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if not is_metadata_column(c)]
    logger.info(f"  candidate CP features: {len(feature_cols)}")

    # Drop rows with NaN in candidate cols and filter to canonical 50-pair subset.
    df_real = df_real.dropna(subset=feature_cols)
    parsed = df_real["FileName_DATA"].map(parse_filename)
    mask = parsed.notna()
    df_real = df_real.loc[mask].copy()
    df_real["cell_type_id"] = parsed[mask].map(lambda t: t[0]).astype(int)
    df_real["sirna_id"]     = parsed[mask].map(lambda t: t[1]).astype(int)
    df_real = df_real[df_real.apply(
        lambda r: (int(r.cell_type_id), int(r.sirna_id)) in canonical, axis=1
    )].copy()
    logger.info(f"  rows inside canonical 50-pair subset: {len(df_real)}")

    kept = select_features(df_real, feature_cols, variance_thresh=variance_thresh, z_thresh=z_thresh)
    reduced = remove_highly_correlated_features(df_real, kept, corr_thresh=corr_thresh)
    logger.info(f"  variance+outlier → {len(kept)}  corr-prune → {len(reduced)}")

    df_real["Source"] = "real"
    df_real_scaled, scaler = preprocess_features_fit_on_real(
        df_real, reduced, source_col="Source", real_value="real"
    )

    # Top-k by ANOVA F with (cell × sirna) combo as the single-factor class label.
    combo_label = (
        df_real_scaled["cell_type_id"].astype(int) * 100000
        + df_real_scaled["sirna_id"].astype(int)
    ).values
    X = df_real_scaled[reduced].values
    F_combo, _ = f_classif(X, combo_label)
    F_cell,  _ = f_classif(X, df_real_scaled["cell_type_id"].astype(int).values)
    F_sirna, _ = f_classif(X, df_real_scaled["sirna_id"].astype(int).values)
    order = np.argsort(-F_combo)[:top_k]
    top = [
        {
            "feature": reduced[i],
            "F_combo": float(F_combo[i]),
            "F_cell":  float(F_cell[i]),
            "F_sirna": float(F_sirna[i]),
        }
        for i in order
    ]

    (output_dir / "reduced_features.json").write_text(
        json.dumps({"features": reduced, "n": len(reduced)}, indent=2)
    )
    with open(output_dir / "scaler.pkl", "wb") as fh:
        pickle.dump({"feature_names": reduced, "scaler": scaler}, fh)
    (output_dir / "top_features.json").write_text(json.dumps({
        "top_k": top_k,
        "method": "anova",
        "features": top,
        "n_combos_in_real": int(len(set(map(int, combo_label)))),
        "median_F_combo":   float(np.nanmedian(F_combo)),
        "median_F_cell":    float(np.nanmedian(F_cell)),
        "median_F_sirna":   float(np.nanmedian(F_sirna)),
    }, indent=2))

    print("\n" + "=" * 100)
    print(f"Top {top_k} CP features (ranked by F_combo; higher = separates combos more)")
    print("=" * 100)
    for i, row in enumerate(top, 1):
        print(
            f"  {i:>2}. {row['feature']:<70s}  "
            f"F_combo={row['F_combo']:7.1f}  F_cell={row['F_cell']:7.1f}  F_sirna={row['F_sirna']:7.1f}"
        )
    print(f"\nArtifacts written under: {output_dir}")


# ---------------------------------------------------------------------------
# Encoder selection — dinov3 vs siglip. Same PNG, different representation space.
# ---------------------------------------------------------------------------

# encoder_tag → REPAEncoder encoder_name. CP scoring (--encoder cp) reads
# CP features from Image.csv directly and does not use a REPA encoder.
ENCODER_NAMES: Dict[str, str] = {
    "dinov3": "dinov3-vit-l",
    "siglip": "siglip",
}
ALL_ENCODER_TAGS: List[str] = list(ENCODER_NAMES.keys()) + ["cp"]


def _tag_path(output_dir: Path, base: str, encoder_tag: str) -> Path:
    """Append _{encoder_tag} before the file extension (e.g. scores_png.json → scores_png_siglip.json)."""
    p = Path(base)
    return output_dir / f"{p.stem}_{encoder_tag}{p.suffix}"


# ---------------------------------------------------------------------------
# STAGE 2: png_features — extract encoder features on the fly from the PNGs CP processed
# ---------------------------------------------------------------------------


def _extract_encoder_from_pngs(
    rgb_dir: Path,
    cache_path: Path,
    batch_size: int,
    device: str,
    encoder_name: str,
) -> None:
    """
    Load every PNG in `rgb_dir`, run REPAEncoder(encoder_name), mean-pool patches,
    and save (features, filenames, metadata) to `cache_path`. Correspondence to
    CellProfiler rows is guaranteed: both read the same PNG byte-for-byte.
    """
    from faithful_cond_gen.model.repa_encoder import REPAEncoder

    png_paths = sorted(glob.glob(str(rgb_dir / "*.png")))
    if not png_paths:
        raise SystemExit(f"No PNGs in {rgb_dir}")
    logger.info(f"Encoder {encoder_name!r}: extracting {len(png_paths)} PNGs from {rgb_dir}")

    enc = REPAEncoder(
        encoder_name=encoder_name, resolution=256, in_channels=3,
        target_grid=16, device=device,
    )
    enc.eval()

    feats_blocks: List[torch.Tensor] = []
    filenames, cell_ids, sirna_ids, sample_idx = [], [], [], []
    for start in range(0, len(png_paths), batch_size):
        batch = png_paths[start:start + batch_size]
        imgs = []
        for p in batch:
            arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
        x = torch.stack(imgs, 0).to(device)
        with torch.no_grad():
            tokens = enc(x)                   # (B, 256, D)
            feats = tokens.mean(dim=1).cpu()  # (B, D)
        feats_blocks.append(feats)
        for p in batch:
            fn = Path(p).name
            parsed = parse_filename(fn)
            if parsed is None:
                raise ValueError(f"Unparseable filename: {fn}")
            c, s, idx = parsed
            filenames.append(fn)
            cell_ids.append(c); sirna_ids.append(s); sample_idx.append(idx)

    features = torch.cat(feats_blocks, 0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "features": features,
        "metadata": {
            "cell_type_id": torch.tensor(cell_ids, dtype=torch.long),
            "sirna_id":     torch.tensor(sirna_ids, dtype=torch.long),
            "sample_idx":   torch.tensor(sample_idx, dtype=torch.long),
        },
        "filenames": filenames,
        "encoder_name": f"{encoder_name}_meanpatch",
        "feature_dim": int(features.shape[1]),
        "source_dir": str(rgb_dir),
    }, cache_path)
    logger.info(f"  Saved features {tuple(features.shape)} → {cache_path}")
    del enc
    torch.cuda.empty_cache()


def stage_png_features(
    rgb_root: Path,
    output_dir: Path,
    models: List[str],
    batch_size: int,
    device: str,
    force: bool,
    encoder_tag: str,
) -> None:
    if encoder_tag == "cp":
        logger.info(
            "CP encoder reads from CellProfiler Image.csv directly; "
            "no png_features extraction needed. Skipping."
        )
        return
    encoder_name = ENCODER_NAMES[encoder_tag]
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"{encoder_tag}_from_png.pt"
    real_cache = output_dir / REAL_DIR / cache_name
    if real_cache.exists() and not force:
        logger.info(f"Real cache exists: {real_cache}; skip (use --force to redo)")
    else:
        _extract_encoder_from_pngs(rgb_root / REAL_DIR, real_cache, batch_size, device, encoder_name)
    for model in models:
        cp_name = MODEL_TO_CP_DIR[model]
        gen_cache = output_dir / cp_name / cache_name
        if gen_cache.exists() and not force:
            logger.info(f"Gen cache exists for {model}: {gen_cache}; skip")
            continue
        _extract_encoder_from_pngs(rgb_root / cp_name, gen_cache, batch_size, device, encoder_name)


# ---------------------------------------------------------------------------
# STAGE 3: png_scores — refit trust on legacy real pool, score PNG gen
# ---------------------------------------------------------------------------


def _load_real_pool_for_encoder(
    encoder_tag: str,
    model: str,
    dataset: str,
    normalize_mode: str,
    cp_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[Optional[torch.Tensor], Optional[Dict]]:
    """Encoder-agnostic real calibration pool loader.

    dinov3 → goes through `load_features_for_dataset` which prefers the
             `_subset_scoring.pt` sibling for rxrx1.
    siglip → loads `outputs/real_rxrx1_siglip_meanpatch/train_features.pt`
             directly and applies the sirna-column subset filter (same role
             the `_subset_scoring` sibling plays for dinov3).
    cp     → loads real CellProfiler rows from `<cp_dir>/real_imgs/Image.csv`,
             filters to the canonical 50-pair subset, and returns the 81
             reduced features (already scaled by the real-fit StandardScaler).
             No further L2 normalization — CP features are already standardized.
    """
    if encoder_tag == "dinov3":
        real_feats, real_meta, _, _ = load_features_for_dataset(
            dataset, model, "dinov3", normalize_mode=normalize_mode
        )
        return real_feats, real_meta

    if encoder_tag == "siglip" and dataset == "rxrx1":
        from faithful_cond_gen.eval.trust_eval.subset_io import filter_rxrx1_real_to_scoring_pool

        real_path = Path("outputs/real_rxrx1_siglip_meanpatch/train_features.pt")
        if not real_path.exists():
            logger.warning(f"Real siglip features not found at {real_path}")
            return None, None
        data = torch.load(real_path, map_location="cpu", weights_only=False)
        feats = data["features"]
        meta = data.get("metadata", {})
        feats, meta = filter_rxrx1_real_to_scoring_pool(feats, meta)
        feats = apply_normalization(feats, normalize_mode, "real_siglip_scoring")
        return feats, meta

    if encoder_tag == "cp" and dataset == "rxrx1":
        if cp_dir is None or output_dir is None:
            raise SystemExit("cp_dir and output_dir required for --encoder cp")
        reduced, _, scaler = _load_feature_artifacts(output_dir)
        arms = load_rxrx1_subset_arms()
        seen, unseen = arms["seen"], arms["unseen"]
        df = _load_cp_df(cp_dir / REAL_DIR / "Image.csv", reduced, seen, unseen, scaler)
        df = df[df["split"].isin(["seen", "unseen"])]
        feats = torch.tensor(df[reduced].values, dtype=torch.float32)
        meta = {
            "cell_type_id": torch.tensor(df["cell_type_id"].values, dtype=torch.long),
            "sirna_id":     torch.tensor(df["sirna_id"].values, dtype=torch.long),
        }
        return feats, meta

    raise SystemExit(f"Unsupported (encoder_tag, dataset) = ({encoder_tag}, {dataset})")


def stage_png_scores(
    output_dir: Path,
    models: List[str],
    encoder_tag: str,
    dataset: str = "rxrx1",
    normalize_mode: str = "l2",
    cp_dir: Optional[Path] = None,
) -> None:
    """
    Fit Mahalanobis components on the real calibration pool for the chosen
    encoder (dinov3 → legacy `train_features_subset_scoring.pt`; siglip →
    `outputs/real_rxrx1_siglip_meanpatch/train_features.pt` + sirna-subset
    filter; cp → CellProfiler Image.csv for real, reduced-81 + real-fit scaler,
    no L2 normalization). Score each model's gen using PNG-extracted features
    (for REPA encoders) or directly from CP Image.csv (for --encoder cp).

    Outputs per model (filenames are encoder-tagged):
      outputs/cp_analysis/<cp_dir>/scores_png_<tag>.json
      outputs/cp_analysis/<cp_dir>/thresholds_png_<tag>.json
    """
    condition_keys = CONDITION_ATTRS[dataset]
    # CP features are already scaled; don't re-normalize. REPA encoders stay L2.
    effective_norm = "none" if encoder_tag == "cp" else normalize_mode

    cp_scoring = encoder_tag == "cp"
    if cp_scoring:
        if cp_dir is None:
            raise SystemExit("--cp-dir required when --encoder cp")
        reduced_cp, _, scaler_cp = _load_feature_artifacts(output_dir)
        arms_cp = load_rxrx1_subset_arms()
        seen_cp, unseen_cp = arms_cp["seen"], arms_cp["unseen"]
    else:
        gen_cache_name = f"{encoder_tag}_from_png.pt"

    for model in models:
        cp_name = MODEL_TO_CP_DIR[model]
        model_out = output_dir / cp_name
        model_out.mkdir(parents=True, exist_ok=True)

        if cp_scoring:
            df_gen = _load_cp_df(
                cp_dir / cp_name / "Image.csv", reduced_cp, seen_cp, unseen_cp, scaler_cp
            )
            df_gen = df_gen[df_gen["split"].isin(["seen", "unseen"])].reset_index(drop=True)
            gen_feats: torch.Tensor = torch.tensor(df_gen[reduced_cp].values, dtype=torch.float32)
            gen_meta = {
                "cell_type_id": torch.tensor(df_gen["cell_type_id"].values, dtype=torch.long),
                "sirna_id":     torch.tensor(df_gen["sirna_id"].values, dtype=torch.long),
            }
            gen_filenames: List[str] = df_gen["FileName_DATA"].tolist()
        else:
            gen_cache = model_out / gen_cache_name
            if not gen_cache.exists():
                logger.warning(f"[{model}] PNG gen cache missing at {gen_cache}; skip")
                continue
            gdata = torch.load(gen_cache, map_location="cpu", weights_only=False)
            gen_feats = gdata["features"]
            gen_meta = gdata["metadata"]
            gen_filenames = gdata["filenames"]
            gen_feats = apply_normalization(gen_feats, effective_norm, f"{model}_png_{encoder_tag}")

        real_feats, real_meta = _load_real_pool_for_encoder(
            encoder_tag, model, dataset, effective_norm,
            cp_dir=cp_dir, output_dir=output_dir,
        )
        if real_feats is None:
            logger.error(f"[{model}] real features not found for encoder={encoder_tag}; skip")
            continue

        # For marginal models, restrict calibration to the model's seen support.
        if "marginal" in model:
            ct = real_meta["cell_type_id"]
            sr = real_meta["sirna_id"]
            ct_list = ct.tolist() if hasattr(ct, "tolist") else list(ct)
            sr_list = sr.tolist() if hasattr(sr, "tolist") else list(sr)
            all_combos = {(int(a), int(b)) for a, b in zip(ct_list, sr_list)}
            seen_combos = all_combos - RXRX1_HELDOUT_PAIRS
            calib_feats, calib_meta = filter_feats_and_meta_by_seen_combos(
                real_feats, real_meta, condition_keys, seen_combos
            )
        else:
            calib_feats, calib_meta = real_feats, real_meta
        logger.info(
            f"[{model}] calibration n={len(calib_feats)} "
            f"({'marginal: seen-only' if 'marginal' in model else 'full: all'})"
        )

        components = fit_trust_scoring_components(calib_feats, calib_meta, condition_keys)
        _, _, real_trust = score_trust_from_components(calib_feats, calib_meta, components)
        realism_z, faith_z, gen_trust = score_trust_from_components(
            gen_feats, gen_meta, components
        )

        real_trust = np.asarray(real_trust, dtype=float)
        real_trust = real_trust[np.isfinite(real_trust)]
        thresholds = {
            "P50": float(np.percentile(real_trust, 50)),
            "P75": float(np.percentile(real_trust, 75)),
            "P90": float(np.percentile(real_trust, 90)),
            "P95": float(np.percentile(real_trust, 95)),
            "n_calib": int(len(real_trust)),
            "source": f"png_gen + real_{encoder_tag}",
            "encoder": encoder_tag,
        }
        (model_out / f"thresholds_png_{encoder_tag}.json").write_text(
            json.dumps(thresholds, indent=2)
        )

        gen_trust_arr = np.asarray(gen_trust, dtype=float)
        realism_arr   = np.asarray(realism_z, dtype=float)
        faith_arr     = np.asarray(faith_z, dtype=float)
        ct_g = (gen_meta["cell_type_id"].tolist()
                if isinstance(gen_meta["cell_type_id"], torch.Tensor)
                else list(gen_meta["cell_type_id"]))
        sr_g = (gen_meta["sirna_id"].tolist()
                if isinstance(gen_meta["sirna_id"], torch.Tensor)
                else list(gen_meta["sirna_id"]))
        scores = {}
        for i, fn in enumerate(gen_filenames):
            stem = Path(fn).stem
            scores[stem] = {
                "trust_updated":  float(gen_trust_arr[i]) if np.isfinite(gen_trust_arr[i]) else None,
                "realism_z":      float(realism_arr[i])   if np.isfinite(realism_arr[i])   else None,
                "faithfulness_z": float(faith_arr[i])     if np.isfinite(faith_arr[i])     else None,
                "cell_type_id":   int(ct_g[i]),
                "sirna_id":       int(sr_g[i]),
            }
        (model_out / f"scores_png_{encoder_tag}.json").write_text(
            json.dumps(scores, indent=2)
        )

        print(
            f"[{model}] ({encoder_tag}) thresholds P50={thresholds['P50']:.3f} P75={thresholds['P75']:.3f} "
            f"P90={thresholds['P90']:.3f} P95={thresholds['P95']:.3f}  n_calib={thresholds['n_calib']}"
        )


def _load_scores_png(model_out: Path, encoder_tag: str) -> Tuple[Dict, Dict]:
    scores_path = model_out / f"scores_png_{encoder_tag}.json"
    th_path     = model_out / f"thresholds_png_{encoder_tag}.json"
    if not scores_path.exists() or not th_path.exists():
        raise SystemExit(
            f"Missing {scores_path} or {th_path}. "
            f"Run --stage png_features then --stage png_scores with --encoder {encoder_tag}."
        )
    return json.loads(scores_path.read_text()), json.loads(th_path.read_text())


# ---------------------------------------------------------------------------
# STAGE 4: ate_cross — 22×22 cross-combo Pearson r matrix at FPR@95
# ---------------------------------------------------------------------------


def _sample_random_same_n(df: pd.DataFrame, n_per_combo: Dict[Tuple[int, int], int], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keep: List[int] = []
    for combo, sub in df.groupby(["cell_type_id", "sirna_id"], sort=False):
        k = int(n_per_combo.get(combo, 0))
        if k <= 0:
            continue
        k = min(k, len(sub))
        idx = rng.choice(sub.index.values, size=k, replace=False)
        keep.extend(idx.tolist())
    return df.loc[keep].copy()


def _build_cross_r_matrix(
    delta_real: Dict[Tuple[int, int], np.ndarray],
    delta_gen:  Dict[Tuple[int, int], np.ndarray],
    keys: List[Tuple[int, int]],
) -> np.ndarray:
    """n×n matrix M[i, j] = pearsonr(Δ_real[k_i], Δ_gen[k_j])."""
    from scipy.stats import pearsonr

    n = len(keys)
    M = np.full((n, n), np.nan, dtype=float)
    for i, ki in enumerate(keys):
        dr = delta_real.get(ki)
        if dr is None or np.isclose(np.std(dr), 0):
            continue
        for j, kj in enumerate(keys):
            dg = delta_gen.get(kj)
            if dg is None or np.isclose(np.std(dg), 0):
                continue
            r, _ = pearsonr(dr, dg)
            M[i, j] = float(r)
    return M


def _cross_matrix_summary(M: np.ndarray) -> Dict[str, float]:
    if M.ndim != 2 or M.shape[0] == 0:
        return {"median_diag": float("nan"), "median_offdiag": float("nan"), "specificity": float("nan")}
    n = M.shape[0]
    diag = np.diag(M)
    off = M[~np.eye(n, dtype=bool)]
    diag_med = float(np.nanmedian(diag))
    off_med  = float(np.nanmedian(off))
    return {"median_diag": diag_med, "median_offdiag": off_med, "specificity": diag_med - off_med}


def stage_ate_cross(
    cp_dir: Path,
    output_dir: Path,
    models: List[str],
    control_sirna: int,
    n_random_seeds: int,
    encoder_tag: str,
) -> None:
    """
    ATE direction recovery with cross-combo specificity null.

    For each unseen treatment combo i (22 total across cells {0, 1, 2}):
      Δ_real[i]  = mean_real(combo i) − mean_real(cell of i, sirna=control)
      Δ_trust[j] = mean_gen_FPR95(combo j) − mean_gen_FPR95(cell of j, sirna=control)
      Δ_rand[j] = same with random-selected gen at matched per-combo n (5 seeds).

    All Δ vectors are computed on the **top-k discriminative CP features**.
    M_trust[i, j] = pearsonr(Δ_real[i], Δ_trust[j]).
    Specificity = median(diag) − median(off-diag).

    Outputs:
      outputs/cp_analysis/<cp_dir>/ate_cross_fpr95.npz
      outputs/cp_analysis/ate_cross_fpr95_summary.csv
      outputs/cp_analysis/headline_ate_cross_fpr95.png
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    reduced, top, scaler = _load_feature_artifacts(output_dir)
    features = top  # top-k discriminative
    arms = load_rxrx1_subset_arms()
    seen, unseen = arms["seen"], arms["unseen"]

    df_real_all = _load_cp_df(
        cp_dir / REAL_DIR / "Image.csv", reduced, seen, unseen, scaler
    )

    valid_cells = sorted(
        c for (c, s) in unseen if s == control_sirna
        and any(ac == c and asr != control_sirna for (ac, asr) in unseen)
    )

    summary_rows: List[Dict] = []
    for model in models:
        cp_name = MODEL_TO_CP_DIR[model]
        model_out = output_dir / cp_name
        scores, thresholds = _load_scores_png(model_out, encoder_tag)
        t_95 = float(thresholds["P95"])

        df_gen = _load_cp_df(
            cp_dir / cp_name / "Image.csv", reduced, seen, unseen, scaler
        )
        df_gen["stem"] = df_gen["FileName_DATA"].map(lambda x: Path(x).stem)
        df_gen["trust_updated"] = df_gen["stem"].map(
            lambda s: scores.get(s, {}).get("trust_updated")
        )
        df_gen = df_gen.dropna(subset=["trust_updated"]).copy()
        df_trust = df_gen[df_gen["trust_updated"] <= t_95].copy()
        logger.info(
            f"[{model}] FPR@95 accept {len(df_trust)}/{len(df_gen)} "
            f"({100 * len(df_trust) / max(1, len(df_gen)):.1f}%)"
        )

        def _mean_vec(df: pd.DataFrame, cell: int, sirna: int) -> Optional[np.ndarray]:
            sub = df[(df["cell_type_id"] == cell) & (df["sirna_id"] == sirna)]
            if len(sub) == 0:
                return None
            return sub[features].mean(axis=0).values

        # Treatment combos per valid cell.
        treat_combos: List[Tuple[int, int]] = []
        for cell in valid_cells:
            treat_combos.extend(
                (c, s) for (c, s) in sorted(unseen) if c == cell and s != control_sirna
            )

        # Real Δ vectors — deterministic.
        real_ctrl_by_cell = {c: _mean_vec(df_real_all, c, control_sirna) for c in valid_cells}
        delta_real: Dict[Tuple[int, int], np.ndarray] = {}
        evaluable_combos: List[Tuple[int, int]] = []
        for (c, s) in treat_combos:
            rv = _mean_vec(df_real_all, c, s)
            if rv is None or real_ctrl_by_cell[c] is None:
                continue
            delta_real[(c, s)] = rv - real_ctrl_by_cell[c]
            evaluable_combos.append((c, s))

        # Trust-side Δ vectors at FPR@95.
        trust_ctrl_by_cell = {c: _mean_vec(df_trust, c, control_sirna) for c in valid_cells}
        delta_trust: Dict[Tuple[int, int], np.ndarray] = {}
        for (c, s) in evaluable_combos:
            if trust_ctrl_by_cell.get(c) is None:
                continue
            gv = _mean_vec(df_trust, c, s)
            if gv is None:
                continue
            delta_trust[(c, s)] = gv - trust_ctrl_by_cell[c]
        trust_keys = [k for k in evaluable_combos if k in delta_trust]

        # Random baselines at matched per-combo n.
        n_per_combo = df_trust.groupby(["cell_type_id", "sirna_id"]).size().to_dict()
        rand_matrices: List[np.ndarray] = []
        rand_summaries: List[Dict[str, float]] = []
        for sd in range(n_random_seeds):
            df_rand = _sample_random_same_n(df_gen, n_per_combo, seed=sd)
            ctrl_by_cell = {c: _mean_vec(df_rand, c, control_sirna) for c in valid_cells}
            delta_rand: Dict[Tuple[int, int], np.ndarray] = {}
            for (c, s) in evaluable_combos:
                if ctrl_by_cell.get(c) is None:
                    continue
                gv = _mean_vec(df_rand, c, s)
                if gv is None:
                    continue
                delta_rand[(c, s)] = gv - ctrl_by_cell[c]
            rand_keys = [k for k in evaluable_combos if k in delta_rand]
            M = _build_cross_r_matrix(delta_real, delta_rand, rand_keys)
            rand_matrices.append(M)
            s = _cross_matrix_summary(M)
            s["n_combos"] = len(rand_keys)
            rand_summaries.append(s)

        M_trust = _build_cross_r_matrix(delta_real, delta_trust, trust_keys)
        trust_sum = _cross_matrix_summary(M_trust)

        def _agg(attr: str) -> Tuple[float, float]:
            vals = [r[attr] for r in rand_summaries if np.isfinite(r[attr])]
            if not vals:
                return float("nan"), float("nan")
            return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        rand_diag_mean, rand_diag_std = _agg("median_diag")
        rand_off_mean,  rand_off_std  = _agg("median_offdiag")
        rand_spec_mean, rand_spec_std = _agg("specificity")

        np.savez(
            model_out / f"ate_cross_fpr95_{encoder_tag}.npz",
            M_trust=M_trust,
            M_rand_stack=np.stack(rand_matrices, axis=0) if rand_matrices else np.zeros((0,)),
            trust_keys=np.asarray(trust_keys),
            evaluable_combos=np.asarray(evaluable_combos),
        )

        summary_rows.append({
            "model": model,
            "t_95": t_95,
            "n_accepted_gen": int(len(df_trust)),
            "n_total_gen":    int(len(df_gen)),
            "acceptance_rate": float(len(df_trust) / max(1, len(df_gen))),
            "n_evaluable_combos":  len(evaluable_combos),
            "n_trust_combos":      len(trust_keys),
            "trust_median_diag":    trust_sum["median_diag"],
            "trust_median_offdiag": trust_sum["median_offdiag"],
            "trust_specificity":    trust_sum["specificity"],
            "rand_median_diag_mean":    rand_diag_mean,
            "rand_median_diag_std":     rand_diag_std,
            "rand_median_offdiag_mean": rand_off_mean,
            "rand_median_offdiag_std":  rand_off_std,
            "rand_specificity_mean":    rand_spec_mean,
            "rand_specificity_std":     rand_spec_std,
            "spec_delta_trust_minus_rand":
                trust_sum["specificity"] - rand_spec_mean if np.isfinite(rand_spec_mean) else float("nan"),
        })
        print(
            f"[{model}] accept={len(df_trust)}/{len(df_gen)}  "
            f"trust spec={trust_sum['specificity']:+.3f}  "
            f"rand spec={rand_spec_mean:+.3f}±{rand_spec_std:.3f}  "
            f"Δspec={trust_sum['specificity'] - rand_spec_mean:+.3f}"
        )

    if not summary_rows:
        return
    pd.DataFrame(summary_rows).to_csv(
        output_dir / f"ate_cross_fpr95_summary_{encoder_tag}.csv", index=False
    )
    _plot_ate_cross_summary(output_dir, encoder_tag)


def _plot_ate_cross_summary(output_dir: Path, encoder_tag: str) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(
        output_dir / f"ate_cross_fpr95_summary_{encoder_tag}.csv"
    ).sort_values("model")
    if df.empty:
        return
    COLOR_TRUST = "#0072B2"
    COLOR_RAND  = "#D55E00"

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(df))
    w = 0.38

    ax = axes[0]
    ax.bar(x - w/2, df["trust_specificity"], width=w, color=COLOR_TRUST,
           edgecolor="black", linewidth=0.7, label="Trust-selected (FPR@95)")
    ax.bar(x + w/2, df["rand_specificity_mean"], width=w,
           yerr=df["rand_specificity_std"], capsize=3,
           color=COLOR_RAND, edgecolor="black", linewidth=0.7, alpha=0.85,
           label="Random (matched n, 5 seeds)")
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.set_ylabel("Specificity = median(diag) − median(off-diag) of cross-combo r matrix")
    ax.set_title("ATE specificity on unseen (higher = combo-specific direction recovery)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=9, frameon=True)

    ax = axes[1]
    ax.bar(x - w/2, df["trust_median_diag"], width=w, color=COLOR_TRUST,
           edgecolor="black", linewidth=0.7, label="Trust (median diag)")
    ax.bar(x + w/2, df["rand_median_diag_mean"], width=w,
           yerr=df["rand_median_diag_std"], capsize=3,
           color=COLOR_RAND, edgecolor="black", linewidth=0.7, alpha=0.85,
           label="Random (median diag)")
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.set_ylabel("median same-combo r(Δ_real, Δ_gen)")
    ax.set_title("Same-combo r only (no cross-null) — for comparison")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=9, frameon=True)

    fig.tight_layout()
    path = output_dir / f"headline_ate_cross_fpr95_{encoder_tag}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# STAGE 5: trust_ladder — per-combo trust quartile → centroid distance to real
# ---------------------------------------------------------------------------


def stage_trust_ladder(
    cp_dir: Path,
    output_dir: Path,
    models: List[str],
    encoder_tag: str,
    n_quartiles: int = 4,
    min_per_quartile: int = 5,
) -> None:
    """
    Per unseen combo, bin gen samples by per-combo trust quartile (Q1 = best trust,
    Q{n_quartiles} = worst). For each quartile compute L2 distance from its centroid
    on the top-k discriminative CP features to the real-combo centroid. Aggregate
    mean ± bootstrap 95% CI across the 25 unseen combos.

    If the score rank-orders correctly, distance should rise from Q1 to Q{n}.

    Outputs:
      outputs/cp_analysis/<cp_dir>/trust_ladder_unseen.csv
      outputs/cp_analysis/<cp_dir>/trust_ladder_unseen.png
      outputs/cp_analysis/headline_trust_ladder_unseen.png
    """
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    reduced, top, scaler = _load_feature_artifacts(output_dir)
    features = top
    arms = load_rxrx1_subset_arms()
    seen, unseen_arm = arms["seen"], arms["unseen"]

    df_real_all = _load_cp_df(
        cp_dir / REAL_DIR / "Image.csv", reduced, seen, unseen_arm, scaler
    )

    model_to_agg: Dict[str, pd.DataFrame] = {}
    for model in models:
        cp_name = MODEL_TO_CP_DIR[model]
        model_out = output_dir / cp_name
        scores, _ = _load_scores_png(model_out, encoder_tag)

        df_gen = _load_cp_df(
            cp_dir / cp_name / "Image.csv", reduced, seen, unseen_arm, scaler
        )
        df_gen["stem"] = df_gen["FileName_DATA"].map(lambda x: Path(x).stem)
        df_gen["trust_updated"] = df_gen["stem"].map(
            lambda s: scores.get(s, {}).get("trust_updated")
        )
        df_gen = df_gen.dropna(subset=["trust_updated"]).copy()
        df_gen_unseen = df_gen[df_gen["split"] == "unseen"]

        rows = []
        for (c, s) in sorted(unseen_arm):
            real_mean = (
                df_real_all[(df_real_all["cell_type_id"] == c) & (df_real_all["sirna_id"] == s)]
                [features].mean(axis=0).values
            )
            if np.any(np.isnan(real_mean)):
                continue
            sub = df_gen_unseen[
                (df_gen_unseen["cell_type_id"] == c) & (df_gen_unseen["sirna_id"] == s)
            ].copy()
            if len(sub) < n_quartiles * min_per_quartile:
                continue
            sub["_rank"] = sub["trust_updated"].rank(method="first", ascending=True)
            sub["_q"] = pd.qcut(sub["_rank"], n_quartiles, labels=list(range(1, n_quartiles + 1)))
            for q in range(1, n_quartiles + 1):
                qsub = sub[sub["_q"] == q]
                if len(qsub) < min_per_quartile:
                    continue
                d = float(np.linalg.norm(qsub[features].mean(axis=0).values - real_mean))
                rows.append({
                    "model": model, "cell_type_id": c, "sirna_id": s,
                    "quartile": q, "n_in_q": int(len(qsub)), "distance": d,
                })
        ldf = pd.DataFrame(rows)
        ldf.to_csv(model_out / f"trust_ladder_unseen_{encoder_tag}.csv", index=False)

        rng = np.random.default_rng(0)
        agg_rows = []
        for q in range(1, n_quartiles + 1):
            vals = ldf.loc[ldf["quartile"] == q, "distance"].astype(float).values
            if len(vals) == 0:
                agg_rows.append({"quartile": q, "n_combos": 0,
                                 "mean_dist": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")})
                continue
            boots = [float(np.mean(rng.choice(vals, size=len(vals), replace=True))) for _ in range(1000)]
            agg_rows.append({
                "quartile": q, "n_combos": int(len(vals)),
                "mean_dist": float(np.mean(vals)),
                "ci_lo": float(np.percentile(boots, 2.5)),
                "ci_hi": float(np.percentile(boots, 97.5)),
            })
        adf = pd.DataFrame(agg_rows)
        model_to_agg[model] = adf

        fig, ax = plt.subplots(figsize=(7, 5))
        xs = adf["quartile"].values
        ys = adf["mean_dist"].values
        yerr_lo = ys - adf["ci_lo"].values
        yerr_hi = adf["ci_hi"].values - ys
        ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt="o-", color="#0072B2",
                    linewidth=2, capsize=4, markersize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([
            "Q1\n(best)" if q == 1 else "Q4\n(worst)" if q == n_quartiles else f"Q{q}"
            for q in xs
        ])
        ax.set_xlabel("Per-combo trust quartile")
        ax.set_ylabel("L2 distance to real-combo centroid (top-k discriminative features)")
        ax.set_title(f"{model} — trust ladder (n_combos ≈ {adf['n_combos'].mean():.0f})")
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(
            model_out / f"trust_ladder_unseen_{encoder_tag}.png",
            dpi=200, bbox_inches="tight",
        )
        plt.close(fig)

        print(
            f"[{model}] "
            + "  ".join(f"Q{row.quartile}={row.mean_dist:.3f}" for row in adf.itertuples(index=False))
        )

    # Overlay headline.
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = plt.cm.tab10.colors
    for i, (model, adf) in enumerate(model_to_agg.items()):
        xs = adf["quartile"].values; ys = adf["mean_dist"].values
        ax.plot(xs, ys, "o-", color=palette[i % 10], linewidth=2, markersize=8, label=model)
    ax.set_xticks(range(1, n_quartiles + 1))
    ax.set_xticklabels([
        "Q1\n(best trust)" if q == 1 else f"Q{n_quartiles}\n(worst trust)"
        if q == n_quartiles else f"Q{q}" for q in range(1, n_quartiles + 1)
    ])
    ax.set_xlabel("Per-combo trust quartile")
    ax.set_ylabel("L2 distance to real-combo centroid (top-k discriminative features)")
    ax.set_title("Trust ladder on unseen — mean across 25 treatment combos")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=9, frameon=True)
    fig.tight_layout()
    headline_path = output_dir / f"headline_trust_ladder_unseen_{encoder_tag}.png"
    fig.savefig(headline_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {headline_path}")


# ---------------------------------------------------------------------------
# Shared helper for Q4/Q5/Q6: bin unseen gen by per-combo trust quartile
# ---------------------------------------------------------------------------


def _gen_unseen_with_trust_and_bins(
    cp_dir: Path,
    output_dir: Path,
    model: str,
    reduced: List[str],
    scaler,
    seen: Set[Tuple[int, int]],
    unseen_arm: Set[Tuple[int, int]],
    n_quartiles: int,
    encoder_tag: str,
) -> pd.DataFrame:
    """Load unseen gen CP rows + trust score, tag each row with per-combo trust quartile.

    Q1 = lowest trust_updated (best), Q{n_quartiles} = highest (worst).
    Combos with < n_quartiles samples are dropped from the output.
    """
    cp_name = MODEL_TO_CP_DIR[model]
    model_out = output_dir / cp_name
    scores, _ = _load_scores_png(model_out, encoder_tag)

    df_gen = _load_cp_df(cp_dir / cp_name / "Image.csv", reduced, seen, unseen_arm, scaler)
    df_gen["stem"] = df_gen["FileName_DATA"].map(lambda x: Path(x).stem)
    df_gen["trust_updated"] = df_gen["stem"].map(
        lambda s: scores.get(s, {}).get("trust_updated")
    )
    df_gen = df_gen.dropna(subset=["trust_updated"]).copy()
    df_gen_unseen = df_gen[df_gen["split"] == "unseen"].copy()

    parts: List[pd.DataFrame] = []
    for (_, _), sub in df_gen_unseen.groupby(["cell_type_id", "sirna_id"], sort=False):
        if len(sub) < n_quartiles:
            continue
        sub = sub.copy()
        sub["_rank"] = sub["trust_updated"].rank(method="first", ascending=True)
        sub["_q"] = pd.qcut(
            sub["_rank"], n_quartiles, labels=list(range(1, n_quartiles + 1))
        ).astype(int)
        parts.append(sub)
    if not parts:
        return df_gen_unseen.assign(_q=pd.Series(dtype=int))
    return pd.concat(parts, axis=0, ignore_index=True)


def _iter_bins(df_binned: pd.DataFrame, n_quartiles: int) -> List[Tuple[str, pd.DataFrame]]:
    """Yield (label, subset) for Q1..Q{n} plus an 'all' anchor (no trust filtering)."""
    out: List[Tuple[str, pd.DataFrame]] = []
    for q in range(1, n_quartiles + 1):
        out.append((f"Q{q}", df_binned[df_binned["_q"] == q]))
    out.append(("all", df_binned))
    return out


# ---------------------------------------------------------------------------
# STAGE 6: perturbation_corr — 25×25 Pearson r matrix of mean CP vectors per bin
# ---------------------------------------------------------------------------


def stage_perturbation_corr(
    cp_dir: Path,
    output_dir: Path,
    models: List[str],
    encoder_tag: str,
    n_quartiles: int = 4,
    min_gen_per_bin: int = 5,
    min_real_per_combo: int = 3,
) -> None:
    """
    Q4: Per bin b ∈ {Q1..Q{n}, all}, build a 25×25 matrix
      C_b[i, j] = pearsonr(μ_gen_bin[combo_i], μ_real[combo_j])
    on the top-k discriminative CP features, over the 25 unseen combos.

    Report per bin:
      - top-1 accuracy: fraction of combos where argmax_j C[i, j] == i
      - combo-specificity: median(diag) − median(off-diag)

    Expectation: Q1 has highest top-1 / specificity, Q{n} lowest.
    """
    from scipy.stats import pearsonr

    output_dir.mkdir(parents=True, exist_ok=True)
    reduced, top, scaler = _load_feature_artifacts(output_dir)
    features = top
    arms = load_rxrx1_subset_arms()
    seen, unseen_arm = arms["seen"], arms["unseen"]

    df_real_all = _load_cp_df(cp_dir / REAL_DIR / "Image.csv", reduced, seen, unseen_arm, scaler)
    real_combos = sorted(unseen_arm)

    real_means: Dict[Tuple[int, int], np.ndarray] = {}
    for (c, s) in real_combos:
        sub = df_real_all[(df_real_all["cell_type_id"] == c) & (df_real_all["sirna_id"] == s)]
        if len(sub) < min_real_per_combo:
            continue
        real_means[(c, s)] = sub[features].mean(axis=0).values

    keys = [k for k in real_combos if k in real_means]
    n = len(keys)

    summary_rows: List[Dict] = []
    for model in models:
        cp_name = MODEL_TO_CP_DIR[model]
        model_out = output_dir / cp_name
        df_bin = _gen_unseen_with_trust_and_bins(
            cp_dir, output_dir, model, reduced, scaler, seen, unseen_arm, n_quartiles, encoder_tag
        )

        per_bin_matrices: Dict[str, np.ndarray] = {}
        for bin_label, df_b in _iter_bins(df_bin, n_quartiles):
            gen_means: Dict[Tuple[int, int], np.ndarray] = {}
            for (c, s) in keys:
                sub = df_b[(df_b["cell_type_id"] == c) & (df_b["sirna_id"] == s)]
                if len(sub) < min_gen_per_bin:
                    continue
                gen_means[(c, s)] = sub[features].mean(axis=0).values

            C = np.full((n, n), np.nan, dtype=float)
            for i, ki in enumerate(keys):
                gv = gen_means.get(ki)
                if gv is None or np.isclose(np.std(gv), 0):
                    continue
                for j, kj in enumerate(keys):
                    rv = real_means.get(kj)
                    if rv is None or np.isclose(np.std(rv), 0):
                        continue
                    r, _ = pearsonr(gv, rv)
                    C[i, j] = float(r)
            per_bin_matrices[bin_label] = C

            valid_rows = [i for i in range(n) if np.isfinite(C[i]).any()]
            top1 = sum(1 for i in valid_rows if int(np.nanargmax(C[i])) == i)
            top1_acc = float(top1 / len(valid_rows)) if valid_rows else float("nan")

            diag = np.diag(C)
            off = C[~np.eye(n, dtype=bool)]
            diag_med = float(np.nanmedian(diag))
            off_med = float(np.nanmedian(off))

            summary_rows.append({
                "model": model,
                "bin": bin_label,
                "n_gen_combos": int(len(gen_means)),
                "n_eval_rows": int(len(valid_rows)),
                "top1_acc": top1_acc,
                "median_diag": diag_med,
                "median_offdiag": off_med,
                "specificity": diag_med - off_med,
            })

        np.savez(
            model_out / f"perturbation_corr_unseen_{encoder_tag}.npz",
            keys=np.asarray(keys),
            **{f"C_{b}": M for b, M in per_bin_matrices.items()},
        )

        by_bin = {r["bin"]: r for r in summary_rows if r["model"] == model}
        acc_str = "  ".join(
            f"{b}={by_bin[b]['top1_acc']:.2f}"
            for b in [f"Q{q}" for q in range(1, n_quartiles + 1)] + ["all"]
            if b in by_bin
        )
        print(f"[{model}] top-1 acc: {acc_str}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            output_dir / f"perturbation_corr_summary_{encoder_tag}.csv", index=False
        )


# ---------------------------------------------------------------------------
# STAGE 7: mmd_bins — RBF MMD² between gen_bin and real per combo per bin
# ---------------------------------------------------------------------------


def stage_mmd_bins(
    cp_dir: Path,
    output_dir: Path,
    models: List[str],
    encoder_tag: str,
    n_quartiles: int = 4,
    min_gen_per_bin: int = 5,
    min_real_per_combo: int = 5,
    n_bootstrap: int = 1000,
) -> None:
    """
    Q5: Per bin b ∈ {Q1..Q{n}, all}, per unseen combo (c, s), compute a Gaussian-RBF
    MMD² between gen_bin[c, s] and real[c, s] on top-k CP features. Bandwidth = median
    pairwise L2 on the real side (median-heuristic). Aggregate median MMD² across the
    25 unseen combos per bin + 95% bootstrap CI.

    Expectation: Q1 smallest median MMD², Q{n} largest.
    """
    from scipy.spatial.distance import cdist

    output_dir.mkdir(parents=True, exist_ok=True)
    reduced, top, scaler = _load_feature_artifacts(output_dir)
    features = top
    arms = load_rxrx1_subset_arms()
    seen, unseen_arm = arms["seen"], arms["unseen"]

    df_real_all = _load_cp_df(cp_dir / REAL_DIR / "Image.csv", reduced, seen, unseen_arm, scaler)

    def _median_heuristic(X: np.ndarray) -> float:
        if len(X) < 2:
            return 1.0
        iu = np.triu_indices(len(X), k=1)
        vals = cdist(X, X, "euclidean")[iu]
        m = float(np.median(vals))
        return m if m > 0 else 1.0

    def _rbf_mmd2(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
        gamma = 1.0 / (2.0 * sigma * sigma)
        Kxx = np.exp(-gamma * cdist(X, X, "sqeuclidean"))
        Kyy = np.exp(-gamma * cdist(Y, Y, "sqeuclidean"))
        Kxy = np.exp(-gamma * cdist(X, Y, "sqeuclidean"))
        return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())

    summary_rows: List[Dict] = []
    per_model_rows: Dict[str, List[Dict]] = {}
    for model in models:
        cp_name = MODEL_TO_CP_DIR[model]
        model_out = output_dir / cp_name
        df_bin = _gen_unseen_with_trust_and_bins(
            cp_dir, output_dir, model, reduced, scaler, seen, unseen_arm, n_quartiles, encoder_tag
        )

        model_rows: List[Dict] = []
        for bin_label, df_b in _iter_bins(df_bin, n_quartiles):
            mmd_vals: List[float] = []
            for (c, s) in sorted(unseen_arm):
                real_sub = df_real_all[
                    (df_real_all["cell_type_id"] == c) & (df_real_all["sirna_id"] == s)
                ][features].values
                gen_sub = df_b[
                    (df_b["cell_type_id"] == c) & (df_b["sirna_id"] == s)
                ][features].values
                if len(real_sub) < min_real_per_combo or len(gen_sub) < min_gen_per_bin:
                    continue
                sigma = _median_heuristic(real_sub)
                mmd = _rbf_mmd2(gen_sub, real_sub, sigma)
                mmd_vals.append(mmd)
                model_rows.append({
                    "model": model, "bin": bin_label,
                    "cell_type_id": c, "sirna_id": s,
                    "n_gen": int(len(gen_sub)), "n_real": int(len(real_sub)),
                    "mmd2": mmd,
                })

            if not mmd_vals:
                continue
            vals = np.asarray(mmd_vals, dtype=float)
            rng = np.random.default_rng(0)
            boots = [
                float(np.median(rng.choice(vals, size=len(vals), replace=True)))
                for _ in range(n_bootstrap)
            ]
            summary_rows.append({
                "model": model,
                "bin": bin_label,
                "n_combos": int(len(vals)),
                "median_mmd2": float(np.median(vals)),
                "mean_mmd2": float(np.mean(vals)),
                "ci_lo": float(np.percentile(boots, 2.5)),
                "ci_hi": float(np.percentile(boots, 97.5)),
            })
        per_model_rows[model] = model_rows
        pd.DataFrame(model_rows).to_csv(
            model_out / f"mmd_bins_unseen_{encoder_tag}.csv", index=False
        )

        by_bin = {r["bin"]: r for r in summary_rows if r["model"] == model}
        s = "  ".join(
            f"{b}={by_bin[b]['median_mmd2']:.3f}"
            for b in [f"Q{q}" for q in range(1, n_quartiles + 1)] + ["all"]
            if b in by_bin
        )
        print(f"[{model}] median MMD²: {s}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            output_dir / f"mmd_bins_summary_{encoder_tag}.csv", index=False
        )


# ---------------------------------------------------------------------------
# STAGE 8: knn_conditioning — real-trained kNN classifies gen_bin per combo
# ---------------------------------------------------------------------------


def stage_knn_conditioning(
    cp_dir: Path,
    output_dir: Path,
    models: List[str],
    encoder_tag: str,
    n_quartiles: int = 4,
    k_neighbors: int = 5,
    min_gen_per_bin: int = 5,
    n_bootstrap: int = 1000,
) -> None:
    """
    Q6: Train kNN on all real unseen CP rows with label = cell × 100000 + sirna.
    For each bin b ∈ {Q1..Q{n}, all}, predict combo on gen_bin samples. Report
    per-combo accuracy, mean across 25 unseen combos, pooled sample-level accuracy,
    and 95% bootstrap CI on per-combo accuracy.

    Expectation: Q1 highest per-combo accuracy, Q{n} lowest.
    """
    from sklearn.neighbors import KNeighborsClassifier

    output_dir.mkdir(parents=True, exist_ok=True)
    reduced, top, scaler = _load_feature_artifacts(output_dir)
    features = top
    arms = load_rxrx1_subset_arms()
    seen, unseen_arm = arms["seen"], arms["unseen"]

    df_real_all = _load_cp_df(cp_dir / REAL_DIR / "Image.csv", reduced, seen, unseen_arm, scaler)
    df_real_unseen = df_real_all[df_real_all["split"] == "unseen"].copy()
    combo_label = (
        df_real_unseen["cell_type_id"].astype(int) * 100000
        + df_real_unseen["sirna_id"].astype(int)
    ).values
    X_train = df_real_unseen[features].values
    knn = KNeighborsClassifier(n_neighbors=k_neighbors).fit(X_train, combo_label)

    summary_rows: List[Dict] = []
    for model in models:
        cp_name = MODEL_TO_CP_DIR[model]
        model_out = output_dir / cp_name
        df_bin = _gen_unseen_with_trust_and_bins(
            cp_dir, output_dir, model, reduced, scaler, seen, unseen_arm, n_quartiles, encoder_tag
        )

        per_combo_rows: List[Dict] = []
        for bin_label, df_b in _iter_bins(df_bin, n_quartiles):
            combo_accs: List[float] = []
            total_correct = total_preds = 0
            for (c, s) in sorted(unseen_arm):
                sub = df_b[(df_b["cell_type_id"] == c) & (df_b["sirna_id"] == s)]
                if len(sub) < min_gen_per_bin:
                    continue
                y_true = c * 100000 + s
                preds = knn.predict(sub[features].values)
                acc = float(np.mean(preds == y_true))
                combo_accs.append(acc)
                total_correct += int(np.sum(preds == y_true))
                total_preds += len(preds)
                per_combo_rows.append({
                    "model": model, "bin": bin_label,
                    "cell_type_id": c, "sirna_id": s,
                    "n_gen": int(len(sub)), "accuracy": acc,
                })

            if not combo_accs:
                continue
            vals = np.asarray(combo_accs, dtype=float)
            rng = np.random.default_rng(0)
            boots = [
                float(np.mean(rng.choice(vals, size=len(vals), replace=True)))
                for _ in range(n_bootstrap)
            ]
            summary_rows.append({
                "model": model,
                "bin": bin_label,
                "n_combos": int(len(vals)),
                "mean_combo_acc": float(np.mean(vals)),
                "pooled_acc": float(total_correct / total_preds) if total_preds else float("nan"),
                "ci_lo": float(np.percentile(boots, 2.5)),
                "ci_hi": float(np.percentile(boots, 97.5)),
            })
        pd.DataFrame(per_combo_rows).to_csv(
            model_out / f"knn_conditioning_unseen_{encoder_tag}.csv", index=False
        )

        by_bin = {r["bin"]: r for r in summary_rows if r["model"] == model}
        s = "  ".join(
            f"{b}={by_bin[b]['mean_combo_acc']:.2f}"
            for b in [f"Q{q}" for q in range(1, n_quartiles + 1)] + ["all"]
            if b in by_bin
        )
        print(f"[{model}] kNN (k={k_neighbors}) mean-combo acc: {s}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            output_dir / f"knn_conditioning_summary_{encoder_tag}.csv", index=False
        )


# ---------------------------------------------------------------------------
# STAGE kid_arms: polynomial-kernel KID on CP top-k features, trust vs random,
# split by seen / unseen arm
# ---------------------------------------------------------------------------


def _hamming(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return int(a[0] != b[0]) + int(a[1] != b[1])


def _condmatched_real_indices(
    gen_keys: List[Tuple[int, int]],
    real_by_cond: Dict[Tuple[int, int], List[int]],
    seed: int,
) -> np.ndarray:
    """For each gen label `(c, s)`, draw 1 real index from the same `(c, s)`
    pool. If absent, fall back to the nearest pool by Hamming distance."""
    rng = np.random.default_rng(seed)
    all_keys = list(real_by_cond.keys())
    out: List[int] = []
    for k in gen_keys:
        pool = real_by_cond.get(k)
        if not pool:
            best = sorted(all_keys, key=lambda kk: (_hamming(k, kk), kk))
            for kk in best:
                pool = real_by_cond.get(kk)
                if pool:
                    break
        if not pool:
            continue
        out.append(int(rng.choice(pool)))
    return np.asarray(out, dtype=int)


def _kid_condmatched_bootstrap(
    gen_np: np.ndarray,
    gen_keys: List[Tuple[int, int]],
    real_np: np.ndarray,
    real_by_cond: Dict[Tuple[int, int], List[int]],
    k_subset: int = 500,
    n_boot: int = 10,
    seed: int = 42,
) -> Tuple[float, float]:
    """Condition-matched KID bootstrap (legacy protocol).

    For each of `n_boot` draws: rebuild a 1:1 condition-matched real index set
    from `real_by_cond`, take a random permutation of `k_subset` gen indices,
    compute `calculate_kid_same_m` with `use_cosine=True`. Return mean, std.
    """
    if len(gen_np) < 10 or len(real_np) < 10:
        return float("nan"), float("nan")
    vals: List[float] = []
    for b in range(n_boot):
        rp = _condmatched_real_indices(gen_keys, real_by_cond, seed=seed + 1100 + b)
        if len(rp) < 10:
            continue
        eff = min(k_subset, len(gen_np), len(rp))
        rng = np.random.default_rng(seed + b)
        gi = rng.permutation(len(gen_np))[:eff]
        ri = rng.permutation(len(rp))[:eff]
        vals.append(calculate_kid_same_m(gen_np[gi], real_np[rp][ri], use_cosine=True))
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def _load_real_pool_for_kid(
    encoder_tag: str,
    cp_dir: Path,
    output_dir: Path,
) -> Dict[str, Tuple[np.ndarray, Dict[Tuple[int, int], List[int]]]]:
    """Per-arm real features + `(cell, sirna) → indices` map, in the encoder's
    native feature space, for condition-matched KID.

      dinov3/siglip → train pool for seen-arm, val pool for unseen-arm (train
                      doesn't contain unseen-arm canonical pairs by design).
      cp            → reduced-81 CP features from `real_imgs/Image.csv` (50
                      pairs total, both arms present).
    """
    from faithful_cond_gen.eval.trust_eval.subset_io import load_rxrx1_subset_arms

    arms = load_rxrx1_subset_arms()
    seen_set, unseen_set = arms["seen"], arms["unseen"]

    def _arm_from_tensor(feats: torch.Tensor, meta: Dict, arm_set) -> Tuple[np.ndarray, Dict]:
        ct = meta["cell_type_id"].tolist() if hasattr(meta["cell_type_id"], "tolist") else list(meta["cell_type_id"])
        sr = meta["sirna_id"].tolist() if hasattr(meta["sirna_id"], "tolist") else list(meta["sirna_id"])
        keep = np.array([(int(a), int(b)) in arm_set for a, b in zip(ct, sr)])
        f = feats[keep].numpy() if isinstance(feats, torch.Tensor) else np.asarray(feats)[keep]
        by_cond: Dict[Tuple[int, int], List[int]] = {}
        idx = 0
        for a, b, kp in zip(ct, sr, keep):
            if not kp:
                continue
            by_cond.setdefault((int(a), int(b)), []).append(idx)
            idx += 1
        return f, by_cond

    if encoder_tag in ("dinov3", "siglip"):
        base = "real_rxrx1_dinov3_meanpatch" if encoder_tag == "dinov3" else "real_rxrx1_siglip_meanpatch"
        train_path = Path(f"outputs/{base}/train_features.pt")
        val_path = Path(f"outputs/{base}/val_features.pt")
        if not train_path.exists():
            raise SystemExit(f"Missing {train_path}")
        d_tr = torch.load(train_path, map_location="cpu", weights_only=False)
        f_seen, by_seen = _arm_from_tensor(d_tr["features"], d_tr["metadata"], seen_set)
        if val_path.exists():
            d_va = torch.load(val_path, map_location="cpu", weights_only=False)
            f_unseen, by_unseen = _arm_from_tensor(d_va["features"], d_va["metadata"], unseen_set)
        else:
            f_unseen, by_unseen = np.zeros((0, f_seen.shape[1]), dtype=f_seen.dtype), {}
        return {"seen": (f_seen, by_seen), "unseen": (f_unseen, by_unseen)}

    if encoder_tag == "cp":
        reduced, top, scaler = _load_feature_artifacts(output_dir)
        df = _load_cp_df(cp_dir / REAL_DIR / "Image.csv", reduced, seen_set, unseen_set, scaler)
        out: Dict[str, Tuple[np.ndarray, Dict]] = {}
        for arm_name, arm_set in (("seen", seen_set), ("unseen", unseen_set)):
            sub = df[df["split"] == arm_name].reset_index(drop=True)
            feats = sub[top].values.astype(np.float32)  # top-k discriminative
            by_cond: Dict[Tuple[int, int], List[int]] = {}
            for i, row in sub.iterrows():
                by_cond.setdefault((int(row.cell_type_id), int(row.sirna_id)), []).append(int(i))
            out[arm_name] = (feats, by_cond)
        return out

    raise SystemExit(f"Unsupported encoder_tag for kid_arms: {encoder_tag}")


def _load_gen_features_for_kid(
    encoder_tag: str,
    cp_dir: Path,
    output_dir: Path,
    model: str,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[str]]:
    """Gen features in the encoder's native space + per-sample (cell, sirna)
    keys + filenames (for joining with the trust-score JSON)."""
    cp_name = MODEL_TO_CP_DIR[model]
    model_out = output_dir / cp_name
    if encoder_tag == "cp":
        reduced, top, scaler = _load_feature_artifacts(output_dir)
        arms = load_rxrx1_subset_arms()
        df = _load_cp_df(
            cp_dir / cp_name / "Image.csv", reduced, arms["seen"], arms["unseen"], scaler
        )
        feats = df[top].values.astype(np.float32)  # top-k discriminative
        keys = [(int(c), int(s)) for c, s in zip(df["cell_type_id"], df["sirna_id"])]
        fns = df["FileName_DATA"].tolist()
        return feats, keys, fns
    gen_cache = model_out / f"{encoder_tag}_from_png.pt"
    g = torch.load(gen_cache, map_location="cpu", weights_only=False)
    feats = g["features"].numpy() if isinstance(g["features"], torch.Tensor) else np.asarray(g["features"])
    ct = g["metadata"]["cell_type_id"].tolist()
    sr = g["metadata"]["sirna_id"].tolist()
    keys = [(int(a), int(b)) for a, b in zip(ct, sr)]
    fns = g["filenames"]
    return feats, keys, fns


def stage_kid_arms(
    cp_dir: Path,
    output_dir: Path,
    models: List[str],
    encoder_tag: str,
    n_random_seeds: int = 5,
    kid_subset_size: int = 500,
    kid_n_boot: int = 10,
    kid_feature_space: Optional[str] = None,
) -> None:
    """
    Condition-matched KID (legacy protocol: 1:1 draw per accepted gen at the
    same (cell, sirna), Hamming fallback, cosine kernel `(x·y + 1)^3` via
    `metrics_kid.calculate_kid_same_m(use_cosine=True)`, `kid_n_boot`
    permutation bootstraps of `kid_subset_size`).

    `encoder_tag` chooses the trust-scoring space (→ `scores_png_<tag>.json`
    and FPR@95 acceptance). `kid_feature_space` chooses the feature space in
    which KID itself is computed (→ real/gen pool used for the MMD²). Default
    is `kid_feature_space = encoder_tag`. Setting them differently decouples
    selection from evaluation — e.g. `--encoder siglip --kid-feature-space cp`
    selects gen with SigLIP trust then measures morphology in CP space.

    Arm split: seen vs unseen. For dinov3/siglip feature space the seen-arm
    real comes from `train_features.pt`, unseen-arm real from `val_features.pt`
    (train-feature extraction excludes canonical unseen pairs by design). CP
    feature space reads both arms from `real_imgs/Image.csv`.

    Output: `outputs/cp_analysis/kid_arms_summary_<encoder_tag>_in_<kid_feature_space>.csv`
    (filename suffixed with `_in_<kid_feature_space>` when it differs from the
    scoring encoder; otherwise falls back to `kid_arms_summary_<encoder_tag>.csv`
    for back-compat with the existing legacy reproductions).
    """
    from faithful_cond_gen.eval.trust_eval.subset_io import load_rxrx1_subset_arms

    output_dir.mkdir(parents=True, exist_ok=True)
    kid_space = kid_feature_space or encoder_tag
    real_by_arm = _load_real_pool_for_kid(kid_space, cp_dir, output_dir)
    arms = load_rxrx1_subset_arms()
    arm_sets = {"seen": arms["seen"], "unseen": arms["unseen"]}

    summary_rows: List[Dict] = []
    for model in models:
        cp_name = MODEL_TO_CP_DIR[model]
        model_out = output_dir / cp_name
        scores, thresholds = _load_scores_png(model_out, encoder_tag)
        t95 = float(thresholds["P95"])

        gen_feats, gen_keys, gen_fns = _load_gen_features_for_kid(
            kid_space, cp_dir, output_dir, model
        )
        gen_trust = np.array([
            scores.get(Path(fn).stem, {}).get("trust_updated", np.nan) for fn in gen_fns
        ], dtype=float)

        for arm_name in ("seen", "unseen"):
            real_np, real_by_cond = real_by_arm.get(arm_name, (np.zeros((0, 0)), {}))
            if len(real_np) < 10:
                logger.warning(f"[{model}] {arm_name}: real pool too small ({len(real_np)}); skip")
                continue
            arm_mask = np.array([k in arm_sets[arm_name] for k in gen_keys])
            gen_arm_np = gen_feats[arm_mask]
            gen_arm_keys = [gen_keys[i] for i in np.where(arm_mask)[0]]
            trust_arm = gen_trust[arm_mask]

            acc_idx = np.where(trust_arm <= t95)[0]
            n_trust = len(acc_idx)
            if n_trust < 10:
                logger.warning(f"[{model}] {arm_name}: n_trust={n_trust} too small; skip")
                continue

            trust_gen = gen_arm_np[acc_idx]
            trust_keys = [gen_arm_keys[i] for i in acc_idx]
            kid_trust, _ = _kid_condmatched_bootstrap(
                trust_gen, trust_keys, real_np, real_by_cond,
                k_subset=kid_subset_size, n_boot=kid_n_boot, seed=42,
            )

            rand_kids: List[float] = []
            for sd in range(n_random_seeds):
                rng = np.random.default_rng(sd)
                idx = rng.choice(len(gen_arm_np), size=n_trust, replace=False)
                rand_gen = gen_arm_np[idx]
                rand_keys = [gen_arm_keys[i] for i in idx]
                km, _ = _kid_condmatched_bootstrap(
                    rand_gen, rand_keys, real_np, real_by_cond,
                    k_subset=kid_subset_size, n_boot=kid_n_boot, seed=2000 + sd,
                )
                rand_kids.append(km)
            rand_kids = [v for v in rand_kids if np.isfinite(v)]
            rand_mean = float(np.mean(rand_kids)) if rand_kids else float("nan")
            rand_std = float(np.std(rand_kids, ddof=1) if len(rand_kids) > 1 else 0.0)

            summary_rows.append({
                "model": model,
                "arm": arm_name,
                "scoring_encoder": encoder_tag,
                "kid_feature_space": kid_space,
                "n_real": int(len(real_np)),
                "n_gen_arm": int(arm_mask.sum()),
                "n_trust": int(n_trust),
                "subset_size": int(min(kid_subset_size, n_trust, len(real_np))),
                "kid_trust": kid_trust,
                "kid_random_mean": rand_mean,
                "kid_random_std": rand_std,
                "delta_trust_minus_random": kid_trust - rand_mean,
            })
            print(
                f"[{model}] {arm_name:6s} n_trust={n_trust:4d}  "
                f"KID trust={kid_trust:+.4f}  KID random={rand_mean:+.4f}±{rand_std:.4f}  "
                f"Δ={kid_trust - rand_mean:+.4f}"
            )

    if summary_rows:
        suffix = encoder_tag if kid_space == encoder_tag else f"{encoder_tag}_in_{kid_space}"
        pd.DataFrame(summary_rows).to_csv(
            output_dir / f"kid_arms_summary_{suffix}.csv", index=False
        )


# ---------------------------------------------------------------------------
# STAGE selection_audit — diagnostics A–F on the CP feature-selection pipeline
# ---------------------------------------------------------------------------


def _loo_knn_accuracy(X: np.ndarray, y: np.ndarray, k_list: List[int]) -> Dict[int, float]:
    """Leave-one-out k-NN classification accuracy for each k in `k_list`."""
    from sklearn.neighbors import NearestNeighbors
    if len(X) < max(k_list) + 1:
        return {k: float("nan") for k in k_list}
    nbrs = NearestNeighbors(n_neighbors=max(k_list) + 1, metric="euclidean").fit(X)
    _, idx = nbrs.kneighbors(X)
    neigh = y[idx[:, 1:]]  # column 0 is the self-neighbor
    out: Dict[int, float] = {}
    for k in k_list:
        nk = neigh[:, :k]
        preds = np.array([np.bincount(row).argmax() for row in nk])
        out[k] = float((preds == y).mean())
    return out


def _frechet_distance(mu_a: np.ndarray, cov_a: np.ndarray,
                      mu_b: np.ndarray, cov_b: np.ndarray) -> float:
    """Fréchet / Wasserstein-2 between two Gaussians."""
    from scipy.linalg import sqrtm
    diff = mu_a - mu_b
    prod = cov_a @ cov_b
    cov_sqrt = sqrtm(prod)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    return float(diff @ diff + np.trace(cov_a + cov_b - 2.0 * cov_sqrt))


def _condmatched_metrics_bootstrap(
    gen_np: np.ndarray,
    gen_keys: List[Tuple[int, int]],
    real_np: np.ndarray,
    real_by_cond: Dict[Tuple[int, int], List[int]],
    k_subset: int = 500,
    n_boot: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap mean-L2, covariance-Frobenius, Fréchet, and cubic-cosine KID
    under the condition-matched real draw. All four metrics share the same
    (gi, ri) samples per bootstrap so they're directly comparable."""
    if len(gen_np) < 10 or len(real_np) < 10:
        return {k: float("nan") for k in (
            "mean", "mean_std", "cov", "cov_std",
            "frechet", "frechet_std", "kid", "kid_std",
        )}
    m_mean: List[float] = []
    m_cov:  List[float] = []
    m_frec: List[float] = []
    m_kid:  List[float] = []
    for b in range(n_boot):
        rp = _condmatched_real_indices(gen_keys, real_by_cond, seed=seed + 1100 + b)
        if len(rp) < 10:
            continue
        eff = min(k_subset, len(gen_np), len(rp))
        rng = np.random.default_rng(seed + b)
        gi = rng.permutation(len(gen_np))[:eff]
        ri = rng.permutation(len(rp))[:eff]
        g = gen_np[gi]
        r = real_np[rp][ri]
        mu_g, mu_r = g.mean(0), r.mean(0)
        cov_g, cov_r = np.cov(g.T), np.cov(r.T)
        if cov_g.ndim == 0:
            cov_g = cov_g.reshape(1, 1); cov_r = cov_r.reshape(1, 1)
        m_mean.append(float(np.linalg.norm(mu_g - mu_r)))
        m_cov.append(float(np.linalg.norm(cov_g - cov_r, ord="fro")))
        try:
            m_frec.append(_frechet_distance(mu_g, cov_g, mu_r, cov_r))
        except Exception:
            m_frec.append(float("nan"))
        m_kid.append(calculate_kid_same_m(g, r, use_cosine=True))

    def _mv(vals: List[float]) -> Tuple[float, float]:
        arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
        if len(arr) == 0:
            return float("nan"), float("nan")
        return float(arr.mean()), float(arr.std())

    mean_m, mean_s = _mv(m_mean)
    cov_m,  cov_s  = _mv(m_cov)
    frec_m, frec_s = _mv(m_frec)
    kid_m,  kid_s  = _mv(m_kid)
    return {
        "mean": mean_m, "mean_std": mean_s,
        "cov": cov_m,   "cov_std":  cov_s,
        "frechet": frec_m, "frechet_std": frec_s,
        "kid": kid_m,   "kid_std":  kid_s,
    }


def _trust_vs_random_metrics(
    gen_np: np.ndarray,
    gen_trust: np.ndarray,
    gen_keys: List[Tuple[int, int]],
    real_np: np.ndarray,
    real_by_cond: Dict[Tuple[int, int], List[int]],
    t95: float,
    n_random_seeds: int,
    kid_subset_size: int,
    kid_n_boot: int,
) -> Dict[str, float]:
    """For one (model, arm, feature-subspace): compute trust metrics, random
    baseline (n_random_seeds seeds at matched n), and Δ for each of the four
    metrics (mean-L2, cov-Frob, Fréchet, cubic-cosine KID)."""
    acc_idx = np.where(np.asarray(gen_trust) <= t95)[0]
    n_trust = int(len(acc_idx))
    out: Dict[str, float] = {"n_trust": n_trust}
    if n_trust < 10:
        return out
    trust_np = gen_np[acc_idx]
    trust_keys = [gen_keys[i] for i in acc_idx]
    trust_m = _condmatched_metrics_bootstrap(
        trust_np, trust_keys, real_np, real_by_cond,
        k_subset=kid_subset_size, n_boot=kid_n_boot, seed=42,
    )

    rand_agg: Dict[str, List[float]] = {k: [] for k in ("mean", "cov", "frechet", "kid")}
    for sd in range(n_random_seeds):
        rng = np.random.default_rng(sd)
        rix = rng.choice(len(gen_np), size=n_trust, replace=False)
        rand_np = gen_np[rix]
        rand_keys = [gen_keys[i] for i in rix]
        rm = _condmatched_metrics_bootstrap(
            rand_np, rand_keys, real_np, real_by_cond,
            k_subset=kid_subset_size, n_boot=kid_n_boot, seed=2000 + sd,
        )
        for key in rand_agg:
            if np.isfinite(rm[key]):
                rand_agg[key].append(rm[key])
    for metric in ("mean", "cov", "frechet", "kid"):
        t = trust_m[metric]
        r = rand_agg[metric]
        r_mean = float(np.mean(r)) if r else float("nan")
        r_std  = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
        out[f"{metric}_trust"]       = t
        out[f"{metric}_rand_mean"]   = r_mean
        out[f"{metric}_rand_std"]    = r_std
        out[f"{metric}_delta"]       = t - r_mean if np.isfinite(r_mean) else float("nan")
    return out


def _build_real_by_cond(
    df: pd.DataFrame, feats: List[str], arm_name: str,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], List[int]]]:
    sub = df[df["split"] == arm_name].reset_index(drop=True)
    X = sub[feats].values.astype(np.float32)
    by_cond: Dict[Tuple[int, int], List[int]] = {}
    for i in range(len(sub)):
        by_cond.setdefault(
            (int(sub.cell_type_id.iat[i]), int(sub.sirna_id.iat[i])), []
        ).append(i)
    return X, by_cond


def _load_gen_reduced81(
    cp_dir: Path, output_dir: Path, model: str, encoder_tag: str,
    reduced: List[str], seen: Set[Tuple[int, int]], unseen: Set[Tuple[int, int]], scaler,
) -> Tuple[Dict[str, Dict], float]:
    """Per-arm gen features (reduced-81), trust scores, and the FPR@95 threshold."""
    cp_name = MODEL_TO_CP_DIR[model]
    scores, thresholds = _load_scores_png(output_dir / cp_name, encoder_tag)
    t95 = float(thresholds["P95"])
    df = _load_cp_df(cp_dir / cp_name / "Image.csv", reduced, seen, unseen, scaler)
    df["stem"] = df["FileName_DATA"].map(lambda x: Path(x).stem)
    df["trust_updated"] = df["stem"].map(lambda s: scores.get(s, {}).get("trust_updated"))
    df = df.dropna(subset=["trust_updated"]).reset_index(drop=True)
    per_arm: Dict[str, Dict] = {}
    for arm_name in ("seen", "unseen"):
        sub = df[df["split"] == arm_name].reset_index(drop=True)
        per_arm[arm_name] = {
            "X":     sub[reduced].values.astype(np.float32),
            "trust": sub["trust_updated"].values.astype(float),
            "keys":  [(int(c), int(s)) for c, s in zip(sub["cell_type_id"], sub["sirna_id"])],
            "fns":   sub["FileName_DATA"].tolist(),
        }
    return per_arm, t95


def _cross_space_predictability(
    cp_real_X: np.ndarray,      # (N, 30) CP top-30 on paired real rows
    siglip_real_X: np.ndarray,  # (N, D) SigLIP on the same paired real rows
    output_path: Path,
) -> Dict[str, float]:
    """F1: how much of each space does the other linearly explain?
    Ridge regression with 5-fold CV R² in both directions + PCA variants."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    def _cv_r2(X: np.ndarray, Y: np.ndarray, n_splits: int = 5) -> float:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        r2s: List[float] = []
        for tr, te in kf.split(X):
            reg = Ridge(alpha=1.0).fit(X[tr], Y[tr])
            pred = reg.predict(X[te])
            ss_res = float(np.sum((Y[te] - pred) ** 2))
            ss_tot = float(np.sum((Y[te] - Y[te].mean(0)) ** 2))
            r2s.append(1.0 - ss_res / max(ss_tot, 1e-12))
        return float(np.mean(r2s))

    # Standardize both blocks before regressing (centered Ridge).
    cp_m, cp_s = cp_real_X.mean(0), cp_real_X.std(0) + 1e-8
    sg_m, sg_s = siglip_real_X.mean(0), siglip_real_X.std(0) + 1e-8
    cp_z = (cp_real_X - cp_m) / cp_s
    sg_z = (siglip_real_X - sg_m) / sg_s

    # To keep the two directions symmetric in dim, compare CP ↔ first-30 SigLIP PCs.
    sg_pcs = PCA(n_components=30).fit_transform(sg_z)

    r2_cp_from_sg_full = _cv_r2(sg_z,   cp_z)
    r2_cp_from_sg_30   = _cv_r2(sg_pcs, cp_z)
    r2_sg_from_cp_full = _cv_r2(cp_z,   sg_z)
    r2_sg_from_cp_30   = _cv_r2(cp_z,   sg_pcs)

    out = {
        "n_paired":               int(len(cp_real_X)),
        "cp_dim":                 int(cp_real_X.shape[1]),
        "siglip_dim":             int(siglip_real_X.shape[1]),
        "r2_cp_from_siglip_full": float(r2_cp_from_sg_full),
        "r2_cp_from_siglip_30pc": float(r2_cp_from_sg_30),
        "r2_siglip_from_cp_full": float(r2_sg_from_cp_full),
        "r2_siglip30pc_from_cp": float(r2_sg_from_cp_30),
    }
    output_path.write_text(json.dumps(out, indent=2))
    return out


def _image_level_stats(png_path: Path) -> Dict[str, float]:
    """Per-image stats that are *not* cellular morphology — sharpness,
    intensity mean/std, channel balance. Used to probe whether SigLIP-trust
    within a fixed (c, s) correlates with these (would suggest SigLIP is
    partly ranking by image-level artifact cleanliness)."""
    from scipy.ndimage import laplace
    arr = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.float32)
    gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    lap_var = float(laplace(gray).var())
    return {
        "mean_intensity": float(arr.mean()),
        "std_intensity":  float(arr.std()),
        "laplacian_var":  lap_var,
        "mean_R":         float(arr[..., 0].mean()),
        "mean_G":         float(arr[..., 1].mean()),
        "mean_B":         float(arr[..., 2].mean()),
    }


def _f2_within_combo_trust_vs_image_stats(
    rgb_root: Path, cp_dir: Path, output_dir: Path, model: str, encoder_tag: str,
    reduced: List[str], seen: Set[Tuple[int, int]], unseen: Set[Tuple[int, int]], scaler,
    out_path: Path,
    min_per_combo: int = 20,
    cache_path: Optional[Path] = None,
) -> Dict[str, float]:
    """F2: within each (c, s) combo in the unseen arm, split gen into upper and
    lower halves by SigLIP trust and compare image-level stats. Return the
    median (upper − lower) difference across combos."""
    from scipy.stats import spearmanr

    scores, thresholds = _load_scores_png(output_dir / MODEL_TO_CP_DIR[model], encoder_tag)
    df = _load_cp_df(
        cp_dir / MODEL_TO_CP_DIR[model] / "Image.csv", reduced, seen, unseen, scaler
    )
    df["stem"] = df["FileName_DATA"].map(lambda x: Path(x).stem)
    df["trust_updated"] = df["stem"].map(lambda s: scores.get(s, {}).get("trust_updated"))
    df = df.dropna(subset=["trust_updated"]).reset_index(drop=True)
    df_u = df[df["split"] == "unseen"].reset_index(drop=True)

    # Cache image stats (slow part) to avoid re-decoding every run.
    if cache_path is not None and cache_path.exists():
        stats_df = pd.read_parquet(cache_path)
    else:
        rows: List[Dict] = []
        rgb_dir = rgb_root / MODEL_TO_CP_DIR[model]
        for fn in df_u["FileName_DATA"].tolist():
            s = _image_level_stats(rgb_dir / fn)
            s["FileName_DATA"] = fn
            rows.append(s)
        stats_df = pd.DataFrame(rows)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            stats_df.to_parquet(cache_path)
    df_u = df_u.merge(stats_df, on="FileName_DATA", how="left")

    stat_cols = [c for c in ("mean_intensity", "std_intensity", "laplacian_var",
                             "mean_B", "mean_G", "mean_R") if c in df_u.columns]
    rows: List[Dict] = []
    for (c, s), sub in df_u.groupby(["cell_type_id", "sirna_id"]):
        if len(sub) < min_per_combo:
            continue
        med = sub["trust_updated"].median()
        low = sub[sub["trust_updated"] <= med]  # better trust
        hi  = sub[sub["trust_updated"] >  med]
        if len(low) < 5 or len(hi) < 5:
            continue
        r: Dict = {"cell": int(c), "sirna": int(s), "n_low": int(len(low)), "n_hi": int(len(hi))}
        for col in stat_cols:
            r[f"delta_{col}"] = float(low[col].mean() - hi[col].mean())
            rho, _ = spearmanr(sub["trust_updated"].values, sub[col].values)
            r[f"rho_trust_{col}"] = float(rho) if np.isfinite(rho) else float("nan")
        rows.append(r)
    res_df = pd.DataFrame(rows)
    res_df.to_csv(out_path, index=False)
    summary = {
        "model": model, "n_combos": int(len(res_df)),
        **{f"median_{c}": float(res_df[c].median()) for c in res_df.columns if c.startswith(("delta_", "rho_"))},
    }
    return summary


def stage_selection_audit(
    cp_dir: Path,
    output_dir: Path,
    rgb_root: Path,
    models: List[str],
    encoder_tag: str,
    k_sweep: Tuple[int, ...] = (5, 10, 15, 20, 25, 30, 40, 50, 81),
    k_fixed: int = 20,
    n_bootstrap: int = 50,
    kid_subset_size: int = 500,
    kid_n_boot: int = 10,
    n_random_seeds: int = 5,
    knn_k_list: Tuple[int, ...] = (1, 5, 10),
    run_f1: bool = True,
    run_f2: bool = True,
) -> None:
    """Audit the CP feature-selection pipeline: A k-sweep, B criterion
    comparison, C top-30 redundancy, D bootstrap stability, E metric
    sensitivity, F1 cross-space predictability, F2 image-level confounders."""
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import f_classif, mutual_info_classif
    from sklearn.linear_model import LogisticRegression

    out = output_dir / "selection_audit"
    out.mkdir(parents=True, exist_ok=True)

    reduced, _saved_top, scaler = _load_feature_artifacts(output_dir)
    arms = load_rxrx1_subset_arms()
    seen, unseen = arms["seen"], arms["unseen"]

    # --- Real rows in reduced-81 ---
    df_real = _load_cp_df(cp_dir / REAL_DIR / "Image.csv", reduced, seen, unseen, scaler)
    df_real = df_real[df_real["split"].isin(["seen", "unseen"])].reset_index(drop=True)
    X_real_81 = df_real[reduced].values.astype(np.float32)
    combo_label = (df_real["cell_type_id"].astype(int) * 100000
                   + df_real["sirna_id"].astype(int)).values
    cell_label  = df_real["cell_type_id"].astype(int).values
    sirna_label = df_real["sirna_id"].astype(int).values

    # --- Rankings on reduced-81 ---
    F_combo, _ = f_classif(X_real_81, combo_label)
    F_cell,  _ = f_classif(X_real_81, cell_label)
    F_sirna, _ = f_classif(X_real_81, sirna_label)
    logger.info("[audit] mutual_info_classif on reduced-81 × combo ...")
    MI_combo = mutual_info_classif(X_real_81, combo_label, random_state=0)
    logger.info("[audit] LASSO-logistic on reduced-81 × combo ...")
    lasso = LogisticRegression(
        penalty="l1", solver="saga", C=0.1, max_iter=2000, n_jobs=-1,
    )
    lasso.fit(X_real_81, combo_label)
    lasso_mass = np.sum(np.abs(lasso.coef_), axis=0)

    rank_map = {
        "F_combo": np.argsort(-F_combo),
        "F_sirna": np.argsort(-F_sirna),
        "MI_combo": np.argsort(-MI_combo),
        "LASSO": np.argsort(-lasso_mass),
    }

    # --- Real-arm pools in reduced-81 ---
    real_by_arm_81: Dict[str, Tuple[np.ndarray, Dict]] = {}
    for arm_name in ("seen", "unseen"):
        real_by_arm_81[arm_name] = _build_real_by_cond(df_real, reduced, arm_name)

    # --- Per-model gen in reduced-81 ---
    gen_per_model: Dict[str, Tuple[Dict, float]] = {}
    for model in models:
        gen_per_model[model] = _load_gen_reduced81(
            cp_dir, output_dir, model, encoder_tag, reduced, seen, unseen, scaler
        )

    # =========================================================================
    # A. k-sweep of F_combo-ranked top-k
    # =========================================================================
    logger.info(f"[A] k-sweep over {list(k_sweep)}")
    rows_knn_A: List[Dict] = []
    rows_kid_A: List[Dict] = []
    for k in k_sweep:
        cols = rank_map["F_combo"][:k]
        Xr = X_real_81[:, cols]
        knn = _loo_knn_accuracy(Xr, combo_label, list(knn_k_list))
        rows_knn_A.append({"k": int(k), **{f"knn_{kk}": knn[kk] for kk in knn_k_list}})
        for model, (per_arm, t95) in gen_per_model.items():
            for arm_name in ("seen", "unseen"):
                real_np_full, real_by_cond = real_by_arm_81[arm_name]
                if len(real_np_full) < 10:
                    continue
                real_np = real_np_full[:, cols]
                g = per_arm[arm_name]
                gen_np = g["X"][:, cols]
                res = _trust_vs_random_metrics(
                    gen_np, g["trust"], g["keys"], real_np, real_by_cond, t95,
                    n_random_seeds, kid_subset_size, kid_n_boot,
                )
                rows_kid_A.append({"k": int(k), "model": model, "arm": arm_name, **res})
    pd.DataFrame(rows_knn_A).to_csv(out / "A_knn_vs_k.csv", index=False)
    pd.DataFrame(rows_kid_A).to_csv(out / "A_kid_vs_k.csv", index=False)

    # =========================================================================
    # B. Criterion comparison at k = k_fixed (F_combo, F_sirna, MI, LASSO, PCA)
    # =========================================================================
    logger.info(f"[B] criterion comparison at k={k_fixed}")
    rows_knn_B: List[Dict] = []
    rows_kid_B: List[Dict] = []
    for name, rank in rank_map.items():
        cols = rank[:k_fixed]
        Xr = X_real_81[:, cols]
        knn = _loo_knn_accuracy(Xr, combo_label, list(knn_k_list))
        rows_knn_B.append({"criterion": name, **{f"knn_{kk}": knn[kk] for kk in knn_k_list}})
        for model, (per_arm, t95) in gen_per_model.items():
            for arm_name in ("seen", "unseen"):
                real_np_full, real_by_cond = real_by_arm_81[arm_name]
                real_np = real_np_full[:, cols]
                g = per_arm[arm_name]
                gen_np = g["X"][:, cols]
                res = _trust_vs_random_metrics(
                    gen_np, g["trust"], g["keys"], real_np, real_by_cond, t95,
                    n_random_seeds, kid_subset_size, kid_n_boot,
                )
                rows_kid_B.append(
                    {"criterion": name, "model": model, "arm": arm_name, **res}
                )
    # PCA as a separate criterion (linear combinations of all 81).
    pca = PCA(n_components=k_fixed).fit(X_real_81)
    Zr = pca.transform(X_real_81)
    knn = _loo_knn_accuracy(Zr, combo_label, list(knn_k_list))
    rows_knn_B.append({"criterion": "PCA", **{f"knn_{kk}": knn[kk] for kk in knn_k_list}})
    for model, (per_arm, t95) in gen_per_model.items():
        for arm_name in ("seen", "unseen"):
            real_np_full, real_by_cond = real_by_arm_81[arm_name]
            real_np = pca.transform(real_np_full)
            g = per_arm[arm_name]
            gen_np = pca.transform(g["X"])
            res = _trust_vs_random_metrics(
                gen_np, g["trust"], g["keys"], real_np, real_by_cond, t95,
                n_random_seeds, kid_subset_size, kid_n_boot,
            )
            rows_kid_B.append(
                {"criterion": "PCA", "model": model, "arm": arm_name, **res}
            )
    pd.DataFrame(rows_knn_B).to_csv(out / "B_knn_criterion.csv", index=False)
    pd.DataFrame(rows_kid_B).to_csv(out / "B_kid_criterion.csv", index=False)

    # =========================================================================
    # C. Redundancy of the current top-30 (F_combo)
    # =========================================================================
    logger.info("[C] redundancy of F_combo top-30")
    top30_cols = rank_map["F_combo"][:30]
    R = np.corrcoef(X_real_81[:, top30_cols].T)
    eigvals = np.sort(np.linalg.eigvalsh(R))[::-1]
    eigvals = np.clip(eigvals, 0.0, None)
    cum = np.cumsum(eigvals) / max(np.sum(eigvals), 1e-12)
    eff_rank_95 = int(np.searchsorted(cum, 0.95) + 1)
    abs_r = np.abs(R)
    iu = np.triu_indices_from(abs_r, k=1)
    upper = abs_r[iu]
    c_summary = {
        "n_features":   30,
        "eff_rank_95":  eff_rank_95,
        "median_abs_r": float(np.median(upper)),
        "max_abs_r":    float(np.max(upper)),
        "pairs_gt_0.5": int((upper > 0.5).sum()),
        "pairs_gt_0.7": int((upper > 0.7).sum()),
        "top_eigvals":  [float(x) for x in eigvals[:10]],
    }
    (out / "C_redundancy.json").write_text(json.dumps(c_summary, indent=2))
    np.save(out / "C_corr_top30.npy", R)

    # =========================================================================
    # D. Bootstrap stability of F_combo top-30
    # =========================================================================
    logger.info(f"[D] bootstrap stability ({n_bootstrap} resamples)")
    rng = np.random.default_rng(0)
    member = np.zeros(len(reduced), dtype=int)
    F_stack = np.zeros((n_bootstrap, len(reduced)), dtype=float)
    N = len(X_real_81)
    for b in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        Fb, _ = f_classif(X_real_81[idx], combo_label[idx])
        F_stack[b] = Fb
        member[np.argsort(-Fb)[:30]] += 1
    stab_rows = []
    for j in range(len(reduced)):
        stab_rows.append({
            "feature":  reduced[j],
            "rank_F_combo":   int(np.where(rank_map["F_combo"] == j)[0][0]) + 1,
            "F_combo_mean":   float(F_stack[:, j].mean()),
            "F_combo_std":    float(F_stack[:, j].std()),
            "top30_freq":     float(member[j] / n_bootstrap),
        })
    pd.DataFrame(stab_rows).sort_values("rank_F_combo").to_csv(
        out / "D_stability.csv", index=False
    )

    # =========================================================================
    # E. Metric-sensitivity panel on the current top-30 (already computed in
    # A at k=30). Save a tidy table with the four metrics side-by-side.
    # =========================================================================
    df_A = pd.DataFrame(rows_kid_A)
    e_df = df_A[df_A["k"] == 30].copy()
    e_cols = ["model", "arm", "n_trust"] + [
        f"{m}_{s}" for m in ("mean", "cov", "frechet", "kid")
        for s in ("trust", "rand_mean", "rand_std", "delta")
    ]
    e_df[[c for c in e_cols if c in e_df.columns]].to_csv(
        out / "E_metric_sensitivity_top30.csv", index=False
    )

    # =========================================================================
    # F1. Cross-space predictability on paired real rows
    # =========================================================================
    if run_f1:
        siglip_cache = output_dir / REAL_DIR / "siglip_from_png.pt"
        if not siglip_cache.exists():
            logger.warning(
                f"[F1] siglip real cache missing at {siglip_cache}; "
                "run --stage png_features --encoder siglip first. Skipping F1."
            )
        else:
            logger.info("[F1] cross-space predictability on paired real rows")
            sdata = torch.load(siglip_cache, map_location="cpu", weights_only=False)
            sfeats = sdata["features"].numpy() if isinstance(sdata["features"], torch.Tensor) else sdata["features"]
            sfilenames = list(sdata["filenames"])
            fn_to_sidx = {fn: i for i, fn in enumerate(sfilenames)}
            join_idx = [fn_to_sidx.get(fn) for fn in df_real["FileName_DATA"].tolist()]
            valid = np.array([i is not None for i in join_idx])
            if valid.sum() < 100:
                logger.warning(f"[F1] too few paired rows ({valid.sum()}); skipping")
            else:
                cp30 = X_real_81[valid][:, top30_cols]
                sgv  = sfeats[[i for i in join_idx if i is not None]]
                _cross_space_predictability(cp30, sgv, out / "F1_cross_space.json")

    # =========================================================================
    # F2. Image-level confounders — SigLIP-trust vs non-morphological image stats
    # =========================================================================
    if run_f2:
        logger.info("[F2] image-level confounder check (unseen arm, per model)")
        f2_rows: List[Dict] = []
        for model in models:
            cache_path = out / "f2_image_stats" / f"{MODEL_TO_CP_DIR[model]}.parquet"
            summary = _f2_within_combo_trust_vs_image_stats(
                rgb_root=rgb_root, cp_dir=cp_dir, output_dir=output_dir,
                model=model, encoder_tag=encoder_tag,
                reduced=reduced, seen=seen, unseen=unseen, scaler=scaler,
                out_path=out / f"F2_{MODEL_TO_CP_DIR[model]}.csv",
                cache_path=cache_path,
            )
            f2_rows.append(summary)
        if f2_rows:
            pd.DataFrame(f2_rows).to_csv(out / "F2_summary.csv", index=False)

    # Small console preview of the headline numbers.
    print("\n" + "=" * 80)
    print("[audit] A_knn_vs_k.csv (LOO kNN on real, top-k by F_combo):")
    print(pd.DataFrame(rows_knn_A).to_string(index=False))
    print("\n[audit] B_knn_criterion.csv (at k={}):".format(k_fixed))
    print(pd.DataFrame(rows_knn_B).to_string(index=False))
    print("\n[audit] C_redundancy.json:", c_summary)
    print(f"\n[audit] all artifacts → {out}")


# ---------------------------------------------------------------------------
# STAGE 9: summary — print the 6 tables in markdown
# ---------------------------------------------------------------------------


def _print_bin_table(csv_path: Path, title: str, caption: str, value_col: str,
                     fmt: str = ".2f", n_quartiles: int = 4) -> None:
    """Render a model × bin table from a per-bin summary CSV."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    bin_order = [f"Q{q}" for q in range(1, n_quartiles + 1)] + ["all"]
    pivot = df.pivot_table(index="model", columns="bin", values=value_col, aggfunc="first")
    pivot = pivot.reindex(columns=[b for b in bin_order if b in pivot.columns])
    pivot = pivot.sort_index()

    print(f"\n## {title}\n")
    print(f"*{caption}*\n")
    cols = list(pivot.columns)
    header = "| model | " + " | ".join(cols) + f" | **Q{n_quartiles} − Q1** |"
    sep = "|---|" + "|".join(["---:"] * len(cols)) + "|---:|"
    print(header); print(sep)
    for model, row in pivot.iterrows():
        qlast = row.get(f"Q{n_quartiles}"); q1 = row.get("Q1")
        delta = (qlast - q1) if (pd.notna(qlast) and pd.notna(q1)) else float("nan")
        cells = " | ".join(
            f"{row[c]:{fmt}}" if pd.notna(row.get(c)) else "—" for c in cols
        )
        delta_str = f"**{delta:+{fmt}}**" if pd.notna(delta) else "—"
        print(f"| {model} | {cells} | {delta_str} |")


def stage_summary(output_dir: Path, models: List[str], encoder_tag: str) -> None:
    """Print Q1–Q6 tables consolidated from the per-model artifacts."""
    print(f"\n# CP-feature summary (encoder = {encoder_tag})\n")
    ate_csv = output_dir / f"ate_cross_fpr95_summary_{encoder_tag}.csv"
    if not ate_csv.exists():
        raise SystemExit(f"Missing {ate_csv}. Run --stage ate_cross --encoder {encoder_tag} first.")
    ate = pd.read_csv(ate_csv).sort_values("model")

    # Q1 acceptance broken down by seen / unseen gen combo membership.
    arms = load_rxrx1_subset_arms()
    seen_set, unseen_set = arms["seen"], arms["unseen"]

    accept_rows: List[Dict] = []
    for model in sorted(ate["model"].tolist()):
        cp = MODEL_TO_CP_DIR[model]
        scores, thresholds = _load_scores_png(output_dir / cp, encoder_tag)
        t_95 = float(thresholds["P95"])
        n_seen_tot = n_unseen_tot = n_seen_acc = n_unseen_acc = 0
        for s in scores.values():
            k = (int(s["cell_type_id"]), int(s["sirna_id"]))
            in_seen = k in seen_set
            in_unseen = k in unseen_set
            if not (in_seen or in_unseen):
                continue
            accepted = s.get("trust_updated") is not None and s["trust_updated"] <= t_95
            if in_seen:
                n_seen_tot += 1;  n_seen_acc += int(accepted)
            else:
                n_unseen_tot += 1; n_unseen_acc += int(accepted)
        accept_rows.append({
            "model": model,
            "n_seen_acc": n_seen_acc, "n_seen_tot": n_seen_tot,
            "n_unseen_acc": n_unseen_acc, "n_unseen_tot": n_unseen_tot,
        })

    print("\n## Table 1 — How many unseen gen samples pass FPR@95?\n")
    print("*Only unseen combos count — that is the compositional-shift regime we care about.*\n")
    print("| model | unseen accepted / total | unseen % |")
    print("|---|---:|---:|")
    for r in accept_rows:
        rate_u = 100 * r["n_unseen_acc"] / max(1, r["n_unseen_tot"])
        print(
            f"| {r['model']} | {r['n_unseen_acc']} / {r['n_unseen_tot']} | {rate_u:.1f} % |"
        )

    print("\n## Table 2 — Does trust selection recover the *right* perturbation direction?\n")
    print(
        "*Unseen only. Trust = gen samples that pass FPR@95. "
        "Random = same per-combo n drawn uniformly from that combo's gen pool (5 seeds).*\n"
    )
    print(
        "| model | n combos used (of 22) | combo-specificity (trust) "
        "| combo-specificity (random) | **trust − random** |"
    )
    print("|---|---:|---:|---:|---:|")
    for _, r in ate.iterrows():
        delta = r["spec_delta_trust_minus_rand"]
        tag = "trust better" if delta > 0 else ("tie" if abs(delta) < 0.02 else "trust worse")
        print(
            f"| {r['model']} | {int(r['n_trust_combos'])}/{int(r['n_evaluable_combos'])} "
            f"| {r['trust_specificity']:+.3f} "
            f"| {r['rand_specificity_mean']:+.3f} ± {r['rand_specificity_std']:.3f} "
            f"| **{delta:+.3f} → {tag}** |"
        )

    # Q3 trust ladder.
    print("\n## Table 3 — Does the trust score rank gen samples by morphology similarity to real?\n")
    print(
        "*Unseen only. Per-combo quartiles: Q1 = lowest trust_updated (best), "
        "Q4 = highest (worst). Distance = L2 between the quartile's mean CP-feature vector "
        "and the real-combo mean on the top-k discriminative features.*\n"
    )
    rows = []
    for model in models:
        cp = MODEL_TO_CP_DIR[model]
        ladder_csv = output_dir / cp / f"trust_ladder_unseen_{encoder_tag}.csv"
        if not ladder_csv.exists():
            continue
        d = pd.read_csv(ladder_csv)
        g = d.groupby("quartile")["distance"].mean()
        if len(g) < 4:
            continue
        rows.append({
            "model": model,
            "Q1": g.get(1), "Q2": g.get(2), "Q3": g.get(3), "Q4": g.get(4),
            "Q4_minus_Q1": g.get(4) - g.get(1),
        })
    if rows:
        qdf = pd.DataFrame(rows).sort_values("model")
        print("| model | Q1 (best trust) | Q2 | Q3 | Q4 (worst trust) | **Q4 − Q1** |")
        print("|---|---:|---:|---:|---:|---:|")
        for _, r in qdf.iterrows():
            print(
                f"| {r['model']} | {r['Q1']:.2f} | {r['Q2']:.2f} | {r['Q3']:.2f} | {r['Q4']:.2f} "
                f"| **{r['Q4_minus_Q1']:+.2f}** |"
            )
    print()

    _print_bin_table(
        output_dir / f"perturbation_corr_summary_{encoder_tag}.csv",
        title="Table 4 — Does per-combo trust rank identify the right real combo?",
        caption=(
            "Unseen only. Top-1 accuracy = fraction of 25 combos where the gen-bin mean "
            "correlates highest with its own real-combo mean (argmax over columns). "
            "Q1 = best trust; 'all' = no trust filtering (anchor)."
        ),
        value_col="top1_acc",
    )

    _print_bin_table(
        output_dir / f"mmd_bins_summary_{encoder_tag}.csv",
        title="Table 5 — Does distributional match to real improve with trust rank?",
        caption=(
            "Unseen only. Gaussian-RBF MMD² between gen_bin[c, s] and real[c, s] "
            "on top-k discriminative features. Median across the 25 combos. "
            "Lower = better distributional match."
        ),
        value_col="median_mmd2",
        fmt=".3f",
    )

    _print_bin_table(
        output_dir / f"knn_conditioning_summary_{encoder_tag}.csv",
        title="Table 6 — Can a real-trained kNN classify trust-selected gen as the right combo?",
        caption=(
            "Unseen only. kNN (k = 5) trained on all real unseen CP rows; predicts "
            "(cell, sirna) on gen_bin. Mean of per-combo accuracy across 25 combos."
        ),
        value_col="mean_combo_acc",
    )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CellProfiler feature analysis for rxrx1 with trust-score selection",
    )
    p.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=[
            "features", "png_features", "png_scores",
            "ate_cross", "trust_ladder",
            "perturbation_corr", "mmd_bins", "knn_conditioning",
            "kid_arms",
            "selection_audit",
            "summary",
        ],
        help="Pipeline stage to run.",
    )
    p.add_argument(
        "--cp-dir",
        type=Path,
        default=Path("/mnt/pvc/cellprofiler_outputs"),
        help="Directory with real_imgs/ + rxrx1_*/ CP output subdirs.",
    )
    p.add_argument(
        "--rgb-root",
        type=Path,
        default=Path("/mnt/pvc/rgb_imgs_rxrx1"),
        help="Directory with the PNGs CellProfiler processed.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cp_analysis"),
        help="Where to write all pipeline artifacts.",
    )
    p.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model keys or 'all'. "
             f"Valid: {','.join(MODEL_TO_CP_DIR.keys())}.",
    )
    # features stage
    p.add_argument("--variance-thresh", type=float, default=1e-5)
    p.add_argument("--z-thresh", type=float, default=5.0)
    p.add_argument("--corr-thresh", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=30)
    # png_features
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--force", action="store_true", help="Re-run png_features even if cache exists.")
    # ate_cross
    p.add_argument("--control-sirna", type=int, default=1138)
    p.add_argument("--n-random-seeds", type=int, default=5)
    # selection_audit
    p.add_argument("--k-fixed", type=int, default=20, help="k for criterion comparison (B)")
    p.add_argument("--n-bootstrap", type=int, default=50, help="bootstrap resamples for stability (D)")
    p.add_argument("--skip-f1", action="store_true", help="skip F1 cross-space predictability")
    p.add_argument("--skip-f2", action="store_true", help="skip F2 image-level confounder check")
    # binned analyses (Q4/Q5/Q6)
    p.add_argument("--n-quartiles", type=int, default=4)
    p.add_argument("--k-neighbors", type=int, default=5)
    # kid_arms: decouple scoring encoder from KID feature space
    p.add_argument(
        "--kid-feature-space",
        type=str,
        default=None,
        choices=[None, *ALL_ENCODER_TAGS],
        help=(
            "Feature space in which kid_arms computes KID. Defaults to --encoder "
            "(KID in the same space as scoring). Set e.g. to 'cp' while "
            "--encoder=siglip to select with SigLIP trust and measure morphology in CP."
        ),
    )
    # representation space for trust scoring
    p.add_argument(
        "--encoder",
        type=str,
        default="dinov3",
        choices=ALL_ENCODER_TAGS,
        help=(
            "Trust-scoring feature space. 'dinov3'/'siglip' use REPAEncoder on "
            "PNGs (requires --stage png_features first). 'cp' reads CP features "
            "from Image.csv directly — no png_features extraction needed."
        ),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    models = resolve_models(args.models)
    if not args.cp_dir.exists():
        raise SystemExit(f"--cp-dir does not exist: {args.cp_dir}")

    if args.stage == "features":
        stage_features(
            cp_dir=args.cp_dir, output_dir=args.output_dir,
            variance_thresh=args.variance_thresh, z_thresh=args.z_thresh,
            corr_thresh=args.corr_thresh, top_k=args.top_k,
        )
    elif args.stage == "png_features":
        stage_png_features(
            rgb_root=args.rgb_root, output_dir=args.output_dir, models=models,
            batch_size=args.batch_size, device=args.device, force=args.force,
            encoder_tag=args.encoder,
        )
    elif args.stage == "png_scores":
        stage_png_scores(
            output_dir=args.output_dir, models=models,
            encoder_tag=args.encoder, cp_dir=args.cp_dir,
        )
    elif args.stage == "ate_cross":
        stage_ate_cross(
            cp_dir=args.cp_dir, output_dir=args.output_dir, models=models,
            control_sirna=args.control_sirna, n_random_seeds=args.n_random_seeds,
            encoder_tag=args.encoder,
        )
    elif args.stage == "trust_ladder":
        stage_trust_ladder(
            cp_dir=args.cp_dir, output_dir=args.output_dir, models=models,
            encoder_tag=args.encoder,
        )
    elif args.stage == "perturbation_corr":
        stage_perturbation_corr(
            cp_dir=args.cp_dir, output_dir=args.output_dir, models=models,
            encoder_tag=args.encoder, n_quartiles=args.n_quartiles,
        )
    elif args.stage == "mmd_bins":
        stage_mmd_bins(
            cp_dir=args.cp_dir, output_dir=args.output_dir, models=models,
            encoder_tag=args.encoder, n_quartiles=args.n_quartiles,
        )
    elif args.stage == "knn_conditioning":
        stage_knn_conditioning(
            cp_dir=args.cp_dir, output_dir=args.output_dir, models=models,
            encoder_tag=args.encoder, n_quartiles=args.n_quartiles,
            k_neighbors=args.k_neighbors,
        )
    elif args.stage == "kid_arms":
        stage_kid_arms(
            cp_dir=args.cp_dir, output_dir=args.output_dir, models=models,
            encoder_tag=args.encoder,
            kid_feature_space=args.kid_feature_space,
        )
    elif args.stage == "selection_audit":
        stage_selection_audit(
            cp_dir=args.cp_dir, output_dir=args.output_dir, rgb_root=args.rgb_root,
            models=models, encoder_tag=args.encoder,
            k_fixed=args.k_fixed, n_bootstrap=args.n_bootstrap,
            n_random_seeds=args.n_random_seeds,
            run_f1=not args.skip_f1, run_f2=not args.skip_f2,
        )
    elif args.stage == "summary":
        stage_summary(output_dir=args.output_dir, models=models, encoder_tag=args.encoder)


if __name__ == "__main__":
    main()
