"""
CP morphology validation — condition-matched evaluation on CP top-15.

Audit finding (`notes/cp_selection_audit.md`) set k=15 as the stable,
informativeness-peak cut for CP feature evaluation. This script uses CP
top-15 as the morphology yardstick and produces three targeted analyses:

  --stage centroid         Per-gen-sample L2 to matched-real (c, s) centroid.
                           Per-model Δd = d_trust − d_random with stratified
                           bootstrap CI. Sensitivity sweep k ∈ {5, 15, 20}.
  --stage rr_ratio         Per-gen-sample r(x) = ||x − μ_real(c,s)|| / μ_RR(c,s),
                           where μ_RR is the expected real-to-real-centroid
                           distance estimated by random half-split Monte
                           Carlo (T repeats). Reports r̄_trust, r̄_rand, Δr,
                           ratio, with a real LOO anchor calibrating to ≈1.
  --stage readout          Real-trained CP classifiers (logistic, ridge,
                           kNN-10) on top-15; accuracy on gen split by
                           {all_gen, random_matched-5-seeds, trust-selected}
                           against {combo, cell, sirna} targets.
  --stage feature_deltas   Per-feature condition-matched |μ_gen − μ_real|;
                           improvement = random − trust per feature. Paired
                           bar plot per model + summary CSV.
  --stage all              Run the three above.

Reuses:
  scripts/analyze_cp_features_rxrx1.py helpers (feature artifacts, CP df
  loader, trust-score loader, model directory map).

Output: `outputs/cp_analysis/morphology_validation/`.
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

import torch  # noqa: E402

from analyze_cp_features_rxrx1 import (  # noqa: E402
    MODEL_TO_CP_DIR,
    REAL_DIR,
    _load_cp_df,
    _load_feature_artifacts,
    _load_scores_png,
    parse_filename,
    resolve_models,
)
from faithful_cond_gen.eval.trust_eval.subset_io import load_rxrx1_subset_arms  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODELS = ["vanilla_marginal", "repa_marginal", "repa_siglip_marginal"]


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------


def _load_top_k_features(output_dir: Path, k: int) -> List[str]:
    """Return the first `k` features from `top_features.json` (F_combo rank)."""
    top = json.loads((output_dir / "top_features.json").read_text())["features"]
    if k > len(top):
        raise SystemExit(
            f"top_features.json only has {len(top)} features; requested k={k}."
            f" Rerun --stage features with --top-k >= {k}."
        )
    return [row["feature"] for row in top[:k]]


def _load_real_df(
    cp_dir: Path,
    reduced: List[str],
    seen: set,
    unseen: set,
    scaler,
) -> pd.DataFrame:
    df = _load_cp_df(cp_dir / REAL_DIR / "Image.csv", reduced, seen, unseen, scaler)
    return df[df["split"].isin(["seen", "unseen"])].reset_index(drop=True)


def _load_gen_df(
    cp_dir: Path,
    output_dir: Path,
    model: str,
    encoder_tag: str,
    reduced: List[str],
    seen: set,
    unseen: set,
    scaler,
) -> Tuple[pd.DataFrame, float]:
    scores, thresholds = _load_scores_png(output_dir / MODEL_TO_CP_DIR[model], encoder_tag)
    t95 = float(thresholds["P95"])
    df = _load_cp_df(cp_dir / MODEL_TO_CP_DIR[model] / "Image.csv", reduced, seen, unseen, scaler)
    df["stem"] = df["FileName_DATA"].map(lambda x: Path(x).stem)
    df["trust_updated"] = df["stem"].map(lambda s: scores.get(s, {}).get("trust_updated"))
    df = df.dropna(subset=["trust_updated"]).reset_index(drop=True)
    return df, t95


def _combo_centroids(df_real: pd.DataFrame, features: List[str]) -> Dict[Tuple[int, int], np.ndarray]:
    """Real per-(c, s) centroid on `features`. Uses Hamming-nearest fallback for
    gen combos without any real (shouldn't happen inside the canonical 50 but
    keeps us safe)."""
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for (c, s), sub in df_real.groupby(["cell_type_id", "sirna_id"]):
        out[(int(c), int(s))] = sub[features].mean(axis=0).values.astype(np.float32)
    return out


def _real_real_stats(
    df_real: pd.DataFrame,
    features: List[str],
    n_repeats: int = 10,
    seed: int = 42,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """Per-(c, s) (μ_RR, σ_RR) of the real-to-real-centroid distance under
    random half-split Monte Carlo. For each repeat: split reals into disjoint
    halves A, B; take the centroid of A and record ||x − μ_A|| for every
    x ∈ B. Pool all such distances (T · n/2 per combo) and report mean+std.
    Fallback to (mean, std) of distances to the full centroid for combos
    with fewer than 4 reals."""
    rng = np.random.default_rng(seed)
    mu_out: Dict[Tuple[int, int], float] = {}
    sd_out: Dict[Tuple[int, int], float] = {}
    for (c, s), sub in df_real.groupby(["cell_type_id", "sirna_id"]):
        X = sub[features].values.astype(np.float32)
        n = len(X)
        if n < 4:
            mu = X.mean(axis=0)
            ds = np.linalg.norm(X - mu, axis=1)
            mu_out[(int(c), int(s))] = float(ds.mean())
            sd_out[(int(c), int(s))] = float(ds.std(ddof=1)) if n > 1 else 0.0
            continue
        h = n // 2
        all_dists: List[np.ndarray] = []
        for _ in range(n_repeats):
            perm = rng.permutation(n)
            A = X[perm[:h]]
            B = X[perm[h:2 * h]]
            mu_A = A.mean(axis=0)
            all_dists.append(np.linalg.norm(B - mu_A, axis=1))
        flat = np.concatenate(all_dists)
        mu_out[(int(c), int(s))] = float(flat.mean())
        sd_out[(int(c), int(s))] = float(flat.std(ddof=1))
    return mu_out, sd_out


def _real_real_mu(
    df_real: pd.DataFrame,
    features: List[str],
    n_repeats: int = 10,
    seed: int = 42,
) -> Dict[Tuple[int, int], float]:
    """Backwards-compatible wrapper: returns only the per-combo μ_RR dict."""
    mu, _ = _real_real_stats(df_real, features, n_repeats=n_repeats, seed=seed)
    return mu


def _hamming(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return int(a[0] != b[0]) + int(a[1] != b[1])


def _centroid_fallback(
    k: Tuple[int, int], centroids: Dict[Tuple[int, int], np.ndarray]
) -> Optional[np.ndarray]:
    if k in centroids:
        return centroids[k]
    best = sorted(centroids.keys(), key=lambda kk: (_hamming(k, kk), kk))
    return centroids[best[0]] if best else None


def _random_indices(
    sub: pd.DataFrame, trust_mask: np.ndarray, seed: int, mode: str,
) -> np.ndarray:
    """Return row indices into `sub` for a random baseline.
    mode='combo_stratified': for each (c, s) present in trust, draw the same
      count from all gen at that combo (without replacement). Neutralizes
      combo-frequency confounding between trust and random.
    mode='uniform': draw n_trust indices uniformly across the arm.
    """
    n_trust = int(trust_mask.sum())
    if mode == "uniform":
        return np.random.default_rng(seed).choice(
            np.arange(len(sub)), size=n_trust, replace=False
        )
    if mode == "combo_stratified":
        rng = np.random.default_rng(seed)
        trust_df = sub[trust_mask]
        counts = trust_df.groupby(["cell_type_id", "sirna_id"]).size()
        keep: List[int] = []
        for (c, s), n in counts.items():
            pool = sub.index[(sub.cell_type_id == c) & (sub.sirna_id == s)].values
            k = min(int(n), len(pool))
            if k <= 0:
                continue
            keep.extend(rng.choice(pool, size=k, replace=False).tolist())
        return np.asarray(keep, dtype=int)
    raise ValueError(f"unknown random mode: {mode}")


# ---------------------------------------------------------------------------
# Stage 1: centroid distance
# ---------------------------------------------------------------------------


def _per_sample_centroid_distance(
    X: np.ndarray,
    keys: List[Tuple[int, int]],
    centroids: Dict[Tuple[int, int], np.ndarray],
) -> np.ndarray:
    diffs = np.empty(len(X), dtype=np.float32)
    for i, k in enumerate(keys):
        mu = _centroid_fallback(k, centroids)
        diffs[i] = float(np.linalg.norm(X[i] - mu)) if mu is not None else np.nan
    return diffs


def _stratified_delta_bootstrap(
    d_trust: np.ndarray,
    d_rand_pooled: np.ndarray,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_t = len(d_trust); n_r = len(d_rand_pooled)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        ti = rng.integers(0, n_t, size=n_t)
        ri = rng.integers(0, n_r, size=n_r)
        boots[b] = float(d_trust[ti].mean() - d_rand_pooled[ri].mean())
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def stage_centroid(
    cp_dir: Path,
    output_dir: Path,
    models: List[str],
    encoder_tag: str,
    k_list: List[int],
    n_random_seeds: int,
    n_boot: int,
    out_dir: Path,
    random_mode: str = "combo_stratified",
    feature_set: str = "top_k",
) -> None:
    arms = load_rxrx1_subset_arms()
    seen, unseen = arms["seen"], arms["unseen"]

    if feature_set == "kept_pre_corr":
        # Pre-corr-prune: ~621 CP features after variance + |z|≤5 filter only.
        # Refit scaler on this wider set; ignore k_list and use all 621.
        from cp_decile_selection_sensitivity import _compute_kept_features_and_scaler
        kept, scaler, df_real_all = _compute_kept_features_and_scaler(cp_dir)
        reduced = kept
        k_iter: List[Tuple[int, List[str]]] = [(len(kept), kept)]
    else:
        reduced, _, scaler = _load_feature_artifacts(output_dir)
        df_real_all = _load_real_df(cp_dir, reduced, seen, unseen, scaler)
        k_iter = [(k, _load_top_k_features(output_dir, k)) for k in k_list]

    rows: List[Dict] = []
    for k, top_feats in k_iter:
        centroids = _combo_centroids(df_real_all, top_feats)
        for model in models:
            df_gen, t95 = _load_gen_df(
                cp_dir, output_dir, model, encoder_tag, reduced, seen, unseen, scaler
            )
            for arm_name in ("seen", "unseen"):
                sub = df_gen[df_gen["split"] == arm_name].reset_index(drop=True)
                if sub.empty:
                    continue
                X = sub[top_feats].values.astype(np.float32)
                keys = [(int(c), int(s))
                        for c, s in zip(sub["cell_type_id"], sub["sirna_id"])]
                d_all = _per_sample_centroid_distance(X, keys, centroids)

                trust_mask = (sub["trust_updated"] <= t95).values
                d_trust = d_all[trust_mask]
                d_trust = d_trust[np.isfinite(d_trust)]
                n_trust = int(trust_mask.sum())
                if n_trust < 10:
                    continue

                # Random baseline: 5 seeds, matched per-combo count (or
                # uniform across arm if random_mode == "uniform"). Pool the
                # distances so the bootstrap samples from the joint random
                # pool.
                rand_pool: List[np.ndarray] = []
                for sd in range(n_random_seeds):
                    ridx = _random_indices(sub, trust_mask, seed=sd, mode=random_mode)
                    rand_pool.append(d_all[ridx])
                d_rand_pooled = np.concatenate(rand_pool)
                d_rand_pooled = d_rand_pooled[np.isfinite(d_rand_pooled)]

                delta = float(d_trust.mean() - d_rand_pooled.mean())
                ci_lo, ci_hi = _stratified_delta_bootstrap(
                    d_trust, d_rand_pooled, n_boot=n_boot, seed=42,
                )
                verdict = (
                    "selected closer"  if ci_hi < 0 else
                    "selected farther" if ci_lo > 0 else
                    "no significant Δ"
                )
                rows.append({
                    "k": int(k),
                    "model": model, "arm": arm_name,
                    "random_mode": random_mode,
                    "n_trust": n_trust,
                    "n_rand_per_seed": int(len(rand_pool[0])) if rand_pool else 0,
                    "mean_d_trust": float(d_trust.mean()),
                    "mean_d_random": float(d_rand_pooled.mean()),
                    "delta_d": delta,
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "verdict": verdict,
                })
    df = pd.DataFrame(rows)
    # Suffix the filename by mode + feature_set so multiple runs can coexist.
    fs_suf = "" if feature_set == "top_k" else f"_{feature_set}"
    out_path = out_dir / f"centroid_distance_{random_mode}{fs_suf}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n=== Centroid distance (feature_set={feature_set}, random_mode={random_mode}) ===")
    print(df.to_string(index=False))
    print(f"\nWrote {out_path}")


# ---------------------------------------------------------------------------
# Stage 1b: real-real-normalized ratio  r(x) = d(x) / μ_RR(c,s)
# ---------------------------------------------------------------------------


def _real_loo_distances(
    df_real: pd.DataFrame,
    features: List[str],
    centroids: Dict[Tuple[int, int], np.ndarray],
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Per-real-sample distance to the LOO centroid at its own (c,s).
    Uses the algebraic shortcut μ_LOO = (μ * n − x) / (n − 1) so we only need
    the full centroid + per-combo n. Returns (distances, keys)."""
    counts = df_real.groupby(["cell_type_id", "sirna_id"]).size().to_dict()
    keys: List[Tuple[int, int]] = []
    out: List[float] = []
    X_full = df_real[features].values.astype(np.float32)
    cs = list(zip(df_real["cell_type_id"].astype(int).values,
                  df_real["sirna_id"].astype(int).values))
    for i, (c, s) in enumerate(cs):
        n = int(counts.get((c, s), 0))
        mu = centroids.get((c, s))
        if mu is None or n <= 1:
            continue
        mu_loo = (mu * n - X_full[i]) / (n - 1)
        out.append(float(np.linalg.norm(X_full[i] - mu_loo)))
        keys.append((c, s))
    return np.asarray(out, dtype=np.float32), keys


def stage_rr_ratio(
    cp_dir: Path,
    output_dir: Path,
    models: List[str],
    encoder_tag: str,
    k_list: List[int],
    n_random_seeds: int,
    n_boot: int,
    out_dir: Path,
    random_mode: str = "combo_stratified",
    feature_set: str = "kept_pre_corr",
    n_split_repeats: int = 10,
    seed: int = 42,
) -> None:
    """Per-gen-sample r(x) = ||x − μ_real(c,s)|| / μ_RR(c,s) where μ_RR is the
    expected real-to-real-centroid distance (random half-split Monte Carlo,
    `n_split_repeats` repeats). Reports r̄_trust, r̄_random, Δr, ratio with
    bootstrap CI per (model, arm), and a real-LOO anchor row per arm that
    calibrates to r̄ ≈ 1."""
    arms = load_rxrx1_subset_arms()
    seen, unseen = arms["seen"], arms["unseen"]
    arm_combos = {"seen": seen, "unseen": unseen}

    if feature_set == "kept_pre_corr":
        from cp_decile_selection_sensitivity import _compute_kept_features_and_scaler
        kept, scaler, df_real_all = _compute_kept_features_and_scaler(cp_dir)
        reduced = kept
        k_iter: List[Tuple[int, List[str]]] = [(len(kept), kept)]
    else:
        reduced, _, scaler = _load_feature_artifacts(output_dir)
        df_real_all = _load_real_df(cp_dir, reduced, seen, unseen, scaler)
        k_iter = [(k, _load_top_k_features(output_dir, k)) for k in k_list]

    rows_r: List[Dict] = []
    rows_z: List[Dict] = []
    for k, top_feats in k_iter:
        centroids = _combo_centroids(df_real_all, top_feats)
        mu_RR, sd_RR = _real_real_stats(
            df_real_all, top_feats, n_repeats=n_split_repeats, seed=seed,
        )

        # Real LOO anchor — should land at r̄ ≈ 1, z̄ ≈ 0 with std(z) ≈ 1.
        d_real, keys_real = _real_loo_distances(df_real_all, top_feats, centroids)
        denom_real_mu = np.asarray([mu_RR.get(key, np.nan) for key in keys_real],
                                   dtype=np.float32)
        denom_real_sd = np.asarray([sd_RR.get(key, np.nan) for key in keys_real],
                                   dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_real = d_real / denom_real_mu
            z_real = (d_real - denom_real_mu) / denom_real_sd
        r_real = np.where(np.isfinite(r_real), r_real, np.nan)
        z_real = np.where(np.isfinite(z_real), z_real, np.nan)
        for arm_name, combos in arm_combos.items():
            mask = np.asarray([key in combos for key in keys_real])
            r_arm = r_real[mask]; r_arm = r_arm[np.isfinite(r_arm)]
            z_arm = z_real[mask]; z_arm = z_arm[np.isfinite(z_arm)]
            if len(r_arm) == 0:
                continue
            rows_r.append({
                "k": int(k), "model": "real_LOO_anchor", "arm": arm_name,
                "random_mode": "—",
                "n_trust": int(len(r_arm)), "n_rand_per_seed": 0,
                "r_trust": float(r_arm.mean()), "r_random": float("nan"),
                "delta_r": float("nan"),
                "ratio_trust_over_random": float("nan"),
                "pct_closer_vs_random": float("nan"),
                "pct_excess_gap_closed": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "verdict": "anchor (≈1.0 expected)",
            })
            rows_z.append({
                "k": int(k), "model": "real_LOO_anchor", "arm": arm_name,
                "random_mode": "—",
                "n_trust": int(len(z_arm)), "n_rand_per_seed": 0,
                "z_trust_mean": float(z_arm.mean()),
                "z_trust_std":  float(z_arm.std(ddof=1)) if len(z_arm) > 1 else 0.0,
                "z_random_mean": float("nan"), "z_random_std": float("nan"),
                "delta_z": float("nan"),
                "pct_z_closer_vs_random": float("nan"),
                "pct_z_excess_gap_closed": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "verdict": "anchor (mean≈0, std≈1 expected)",
            })

        for model in models:
            df_gen, t95 = _load_gen_df(
                cp_dir, output_dir, model, encoder_tag, reduced, seen, unseen, scaler,
            )
            for arm_name in ("seen", "unseen"):
                sub = df_gen[df_gen["split"] == arm_name].reset_index(drop=True)
                if sub.empty:
                    continue
                X = sub[top_feats].values.astype(np.float32)
                keys = [(int(c), int(s))
                        for c, s in zip(sub["cell_type_id"], sub["sirna_id"])]
                d_all = _per_sample_centroid_distance(X, keys, centroids)
                denom_mu = np.asarray([mu_RR.get(key, np.nan) for key in keys],
                                      dtype=np.float32)
                denom_sd = np.asarray([sd_RR.get(key, np.nan) for key in keys],
                                      dtype=np.float32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    r_all = d_all / denom_mu
                    z_all = (d_all - denom_mu) / denom_sd
                r_all = np.where(np.isfinite(r_all), r_all, np.nan)
                z_all = np.where(np.isfinite(z_all), z_all, np.nan)

                trust_mask = (sub["trust_updated"] <= t95).values
                n_trust = int(trust_mask.sum())
                if n_trust < 10:
                    continue
                r_trust = r_all[trust_mask]; r_trust = r_trust[np.isfinite(r_trust)]
                z_trust = z_all[trust_mask]; z_trust = z_trust[np.isfinite(z_trust)]

                rand_idx_by_seed = [
                    _random_indices(sub, trust_mask, seed=sd, mode=random_mode)
                    for sd in range(n_random_seeds)
                ]
                r_rand_pooled = np.concatenate([r_all[ri] for ri in rand_idx_by_seed])
                z_rand_pooled = np.concatenate([z_all[ri] for ri in rand_idx_by_seed])
                r_rand_pooled = r_rand_pooled[np.isfinite(r_rand_pooled)]
                z_rand_pooled = z_rand_pooled[np.isfinite(z_rand_pooled)]

                # ---- ratio table ----
                mean_t = float(r_trust.mean()); mean_r = float(r_rand_pooled.mean())
                delta_r = mean_t - mean_r
                ratio = mean_t / mean_r if mean_r > 0 else float("nan")
                pct_close = (
                    -delta_r / mean_r * 100.0 if mean_r > 0 else float("nan")
                )
                excess = mean_r - 1.0
                pct_gap_closed = (
                    -delta_r / excess * 100.0 if excess > 1e-6 else float("nan")
                )
                ci_lo_r, ci_hi_r = _stratified_delta_bootstrap(
                    r_trust, r_rand_pooled, n_boot=n_boot, seed=42,
                )
                verdict_r = (
                    "selected closer"  if ci_hi_r < 0 else
                    "selected farther" if ci_lo_r > 0 else
                    "no significant Δ"
                )
                rows_r.append({
                    "k": int(k), "model": model, "arm": arm_name,
                    "random_mode": random_mode,
                    "n_trust": n_trust,
                    "n_rand_per_seed": int(len(rand_idx_by_seed[0])),
                    "r_trust": mean_t, "r_random": mean_r,
                    "delta_r": delta_r,
                    "ratio_trust_over_random": ratio,
                    "pct_closer_vs_random": pct_close,
                    "pct_excess_gap_closed": pct_gap_closed,
                    "ci_lo": ci_lo_r, "ci_hi": ci_hi_r,
                    "verdict": verdict_r,
                })

                # ---- z-score table ----
                z_t_mean = float(z_trust.mean()); z_t_std = float(z_trust.std(ddof=1))
                z_r_mean = float(z_rand_pooled.mean())
                z_r_std  = float(z_rand_pooled.std(ddof=1))
                delta_z = z_t_mean - z_r_mean
                pct_close_z = (
                    -delta_z / z_r_mean * 100.0 if z_r_mean > 1e-6 else float("nan")
                )
                # Excess-over-real-anchor framing for z (anchor mean is ≈ 0,
                # so excess ≈ z_random itself).
                pct_gap_closed_z = pct_close_z  # same, retained for symmetry
                ci_lo_z, ci_hi_z = _stratified_delta_bootstrap(
                    z_trust, z_rand_pooled, n_boot=n_boot, seed=42,
                )
                verdict_z = (
                    "selected closer"  if ci_hi_z < 0 else
                    "selected farther" if ci_lo_z > 0 else
                    "no significant Δ"
                )
                rows_z.append({
                    "k": int(k), "model": model, "arm": arm_name,
                    "random_mode": random_mode,
                    "n_trust": n_trust,
                    "n_rand_per_seed": int(len(rand_idx_by_seed[0])),
                    "z_trust_mean": z_t_mean, "z_trust_std": z_t_std,
                    "z_random_mean": z_r_mean, "z_random_std": z_r_std,
                    "delta_z": delta_z,
                    "pct_z_closer_vs_random": pct_close_z,
                    "pct_z_excess_gap_closed": pct_gap_closed_z,
                    "ci_lo": ci_lo_z, "ci_hi": ci_hi_z,
                    "verdict": verdict_z,
                })

    df_r = pd.DataFrame(rows_r)
    df_z = pd.DataFrame(rows_z)
    fs_suf = "" if feature_set == "top_k" else f"_{feature_set}"
    out_path_r = out_dir / f"rr_ratio_{random_mode}{fs_suf}.csv"
    out_path_z = out_dir / f"rr_zscore_{random_mode}{fs_suf}.csv"
    df_r.to_csv(out_path_r, index=False)
    df_z.to_csv(out_path_z, index=False)
    print(
        f"\n=== rr_ratio (feature_set={feature_set}, random_mode={random_mode}, "
        f"T={n_split_repeats}) ==="
    )
    print(df_r.to_string(index=False))
    print(
        f"\n=== rr_zscore (feature_set={feature_set}, random_mode={random_mode}, "
        f"T={n_split_repeats}) ==="
    )
    print(df_z.to_string(index=False))
    print(f"\nWrote {out_path_r}\nWrote {out_path_z}")


# ---------------------------------------------------------------------------
# Stage 2: readout (CP classifiers trained on real, evaluated on gen)
# ---------------------------------------------------------------------------


def _load_real_in_space(
    cp_dir: Path,
    output_dir: Path,
    space: str,
    reduced: List[str],
    top_feats: List[str],
    seen: set,
    unseen: set,
    scaler,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[object], pd.DataFrame]:
    """Load real features in {cp, siglip, dinov3} space, paired by filename to
    CP rows (so all spaces operate on the same 2066 PNGs).

    Returns `(X_scaled, {target: labels}, transformer_for_gen, df_real)`.
    CP: X is already scaler-applied; transformer is None. SigLIP/DINOv3: fit a
    new StandardScaler on real features and return it for gen transform.
    """
    df_real = _load_real_df(cp_dir, reduced, seen, unseen, scaler)
    labels = {
        "combo": (df_real["cell_type_id"].astype(int) * 100000
                  + df_real["sirna_id"].astype(int)).values,
        "cell":  df_real["cell_type_id"].astype(int).values,
        "sirna": df_real["sirna_id"].astype(int).values,
    }
    if space == "cp":
        X = df_real[top_feats].values.astype(np.float32)
        return X, labels, None, df_real

    cache = output_dir / REAL_DIR / f"{space}_from_png.pt"
    if not cache.exists():
        raise SystemExit(
            f"Missing {cache}. Run `--stage png_features --encoder {space}` in "
            f"analyze_cp_features_rxrx1 first."
        )
    data = torch.load(cache, map_location="cpu", weights_only=False)
    feats = (data["features"].numpy() if isinstance(data["features"], torch.Tensor)
             else np.asarray(data["features"])).astype(np.float32)
    fns = list(data["filenames"])
    fn_to_idx = {fn: i for i, fn in enumerate(fns)}

    # Intersect with CP real rows — same 2066 PNGs should match 1:1.
    row_idx: List[int] = []
    keep_mask: List[bool] = []
    for fn in df_real["FileName_DATA"].tolist():
        i = fn_to_idx.get(fn)
        if i is None:
            keep_mask.append(False)
            continue
        row_idx.append(i)
        keep_mask.append(True)
    keep_mask_arr = np.asarray(keep_mask)
    df_real = df_real[keep_mask_arr].reset_index(drop=True)
    for k in labels:
        labels[k] = labels[k][keep_mask_arr]
    X = feats[row_idx]

    from sklearn.preprocessing import StandardScaler
    real_scaler = StandardScaler().fit(X)
    X_scaled = real_scaler.transform(X).astype(np.float32)
    return X_scaled, labels, real_scaler, df_real


def _load_gen_in_space(
    cp_dir: Path,
    output_dir: Path,
    model: str,
    encoder_tag: str,
    space: str,
    reduced: List[str],
    top_feats: List[str],
    seen: set,
    unseen: set,
    scaler,
    gen_transformer: Optional[object],
) -> Tuple[np.ndarray, pd.DataFrame, float]:
    """Load gen features in {cp, siglip, dinov3} space with a joined DataFrame
    carrying `cell_type_id, sirna_id, split, trust_updated, FileName_DATA` for
    each row. `gen_transformer` is the StandardScaler fit on real (only used
    for non-CP spaces)."""
    cp_name = MODEL_TO_CP_DIR[model]
    scores, thresholds = _load_scores_png(output_dir / cp_name, encoder_tag)
    t95 = float(thresholds["P95"])

    if space == "cp":
        df, t95 = _load_gen_df(cp_dir, output_dir, model, encoder_tag, reduced, seen, unseen, scaler)
        X = df[top_feats].values.astype(np.float32)  # already scaled
        return X, df, t95

    cache = output_dir / cp_name / f"{space}_from_png.pt"
    if not cache.exists():
        raise SystemExit(
            f"Missing {cache}. Run `--stage png_features --encoder {space}` first."
        )
    data = torch.load(cache, map_location="cpu", weights_only=False)
    feats = (data["features"].numpy() if isinstance(data["features"], torch.Tensor)
             else np.asarray(data["features"])).astype(np.float32)
    fns = list(data["filenames"])

    # Parse labels + attach trust score per row.
    rows: List[Dict] = []
    feat_indices: List[int] = []
    for i, fn in enumerate(fns):
        parsed = parse_filename(fn)
        if parsed is None:
            continue
        c, s, _idx = parsed
        trust = scores.get(Path(fn).stem, {}).get("trust_updated")
        if trust is None:
            continue
        arm = ("seen" if (c, s) in seen
               else "unseen" if (c, s) in unseen
               else "outside")
        rows.append({
            "FileName_DATA": fn,
            "cell_type_id": int(c), "sirna_id": int(s),
            "split": arm, "trust_updated": float(trust),
        })
        feat_indices.append(i)
    df = pd.DataFrame(rows).reset_index(drop=True)
    X = feats[feat_indices]
    assert gen_transformer is not None, f"gen_transformer required for space={space}"
    X_scaled = gen_transformer.transform(X).astype(np.float32)
    return X_scaled, df, t95


def _fit_classifiers(X: np.ndarray, y: np.ndarray, seed: int = 0) -> Dict[str, object]:
    """Fit three CP classifiers on real rows. Return a dict of fitted models."""
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.neighbors import KNeighborsClassifier

    models: Dict[str, object] = {}
    models["logistic"] = LogisticRegression(
        max_iter=2000, C=1.0, solver="lbfgs", n_jobs=-1, random_state=seed,
    ).fit(X, y)
    models["ridge"] = RidgeClassifier(alpha=1.0, random_state=seed).fit(X, y)
    models["knn10"] = KNeighborsClassifier(n_neighbors=10, n_jobs=-1).fit(X, y)
    return models


def _accuracy(clf, X: np.ndarray, y: np.ndarray) -> float:
    return float((clf.predict(X) == y).mean())


def stage_readout(
    cp_dir: Path,
    output_dir: Path,
    models: List[str],
    encoder_tag: str,
    top_k: int,
    n_random_seeds: int,
    out_dir: Path,
    random_mode: str = "combo_stratified",
    feature_space: str = "cp",
) -> None:
    reduced, _, scaler = _load_feature_artifacts(output_dir)
    arms = load_rxrx1_subset_arms()
    seen, unseen = arms["seen"], arms["unseen"]

    top_feats = _load_top_k_features(output_dir, top_k)
    X_real, labels_real, gen_transformer, _ = _load_real_in_space(
        cp_dir, output_dir, feature_space, reduced, top_feats, seen, unseen, scaler,
    )
    targets: Dict[str, np.ndarray] = {
        "combo": labels_real["combo"],
        "cell":  labels_real["cell"],
        "sirna": labels_real["sirna"],
    }
    logger.info(
        "[readout] fitting classifiers on real in %s space (n=%d, dim=%d)",
        feature_space, X_real.shape[0], X_real.shape[1],
    )
    clf_by_target: Dict[str, Dict[str, object]] = {
        t: _fit_classifiers(X_real, y) for t, y in targets.items()
    }

    rows: List[Dict] = []
    for model in models:
        X_gen_full, df_gen, t95 = _load_gen_in_space(
            cp_dir, output_dir, model, encoder_tag, feature_space,
            reduced, top_feats, seen, unseen, scaler, gen_transformer,
        )
        for arm_name in ("seen", "unseen"):
            arm_idx = np.where(df_gen["split"].values == arm_name)[0]
            if len(arm_idx) == 0:
                continue
            sub = df_gen.iloc[arm_idx].reset_index(drop=True)
            X_gen = X_gen_full[arm_idx]
            y_true = {
                "combo": (sub["cell_type_id"].astype(int) * 100000
                          + sub["sirna_id"].astype(int)).values,
                "cell":  sub["cell_type_id"].astype(int).values,
                "sirna": sub["sirna_id"].astype(int).values,
            }
            trust_mask = (sub["trust_updated"] <= t95).values
            n_trust = int(trust_mask.sum())
            if n_trust < 10:
                continue

            # 5 random draws matched to n_trust (per-combo if stratified).
            rand_idx_by_seed = [
                _random_indices(sub, trust_mask, seed=sd, mode=random_mode)
                for sd in range(n_random_seeds)
            ]

            for tname, classifiers in clf_by_target.items():
                for clfname, clf in classifiers.items():
                    acc_all     = _accuracy(clf, X_gen, y_true[tname])
                    acc_trust   = _accuracy(clf, X_gen[trust_mask], y_true[tname][trust_mask])
                    acc_rand_vals = [
                        _accuracy(clf, X_gen[ri], y_true[tname][ri])
                        for ri in rand_idx_by_seed
                    ]
                    acc_rand_mean = float(np.mean(acc_rand_vals))
                    acc_rand_std  = float(np.std(acc_rand_vals, ddof=1)) \
                        if len(acc_rand_vals) > 1 else 0.0
                    rows.append({
                        "model": model, "arm": arm_name,
                        "feature_space": feature_space,
                        "random_mode": random_mode,
                        "classifier": clfname, "target": tname,
                        "n_trust": n_trust, "n_total_gen": int(len(sub)),
                        "acc_all_gen":     acc_all,
                        "acc_random_mean": acc_rand_mean,
                        "acc_random_std":  acc_rand_std,
                        "acc_trust":       acc_trust,
                        "delta_trust_vs_random": acc_trust - acc_rand_mean,
                    })

    df = pd.DataFrame(rows)
    out_path = out_dir / f"readout_accuracy_{random_mode}_{feature_space}.csv"
    df.to_csv(out_path, index=False)
    # Print a compact per-model table (logistic, unseen arm).
    print("\n=== Readout accuracy (space=%s, logistic, unseen, random_mode=%s) ===" %
          (feature_space, random_mode))
    view = df[(df.classifier == "logistic") & (df.arm == "unseen")]
    cols = ["model", "target", "n_trust",
            "acc_all_gen", "acc_random_mean", "acc_random_std", "acc_trust",
            "delta_trust_vs_random"]
    print(view[cols].to_string(index=False))
    print(f"\nWrote {out_path}")


# ---------------------------------------------------------------------------
# Stage 3: per-feature closer-to-real
# ---------------------------------------------------------------------------


def _condmatched_feature_errors(
    gen_df: pd.DataFrame, real_df: pd.DataFrame, features: List[str],
) -> np.ndarray:
    """For each (c, s) combo present in gen, compute |mean(gen_cs) - mean(real_cs)|
    per feature, then mean across combos. Returns (n_features,) array."""
    errs: List[np.ndarray] = []
    for (c, s), sub_gen in gen_df.groupby(["cell_type_id", "sirna_id"]):
        sub_real = real_df[(real_df["cell_type_id"] == c) & (real_df["sirna_id"] == s)]
        if sub_real.empty:
            continue
        diff = np.abs(sub_gen[features].mean(axis=0).values
                      - sub_real[features].mean(axis=0).values)
        errs.append(diff.astype(np.float64))
    if not errs:
        return np.full(len(features), np.nan)
    return np.vstack(errs).mean(axis=0)


def stage_feature_deltas(
    cp_dir: Path,
    output_dir: Path,
    models: List[str],
    encoder_tag: str,
    top_k: int,
    n_random_seeds: int,
    out_dir: Path,
    random_mode: str = "combo_stratified",
) -> None:
    import matplotlib.pyplot as plt

    reduced, _, scaler = _load_feature_artifacts(output_dir)
    arms = load_rxrx1_subset_arms()
    seen, unseen = arms["seen"], arms["unseen"]

    top_feats = _load_top_k_features(output_dir, top_k)
    df_real = _load_real_df(cp_dir, reduced, seen, unseen, scaler)

    rows: List[Dict] = []
    for model in models:
        df_gen, t95 = _load_gen_df(
            cp_dir, output_dir, model, encoder_tag, reduced, seen, unseen, scaler
        )
        for arm_name in ("seen", "unseen"):
            sub = df_gen[df_gen["split"] == arm_name].reset_index(drop=True)
            if sub.empty:
                continue
            trust_mask = (sub["trust_updated"] <= t95).values
            trust_sub = sub[trust_mask]
            n_trust = int(len(trust_sub))
            if n_trust < 10:
                continue

            err_trust = _condmatched_feature_errors(trust_sub, df_real, top_feats)

            rand_stack: List[np.ndarray] = []
            for sd in range(n_random_seeds):
                ridx = _random_indices(sub, trust_mask, seed=sd, mode=random_mode)
                rand_sub = sub.iloc[ridx]
                rand_stack.append(_condmatched_feature_errors(rand_sub, df_real, top_feats))
            err_rand = np.vstack(rand_stack).mean(axis=0)
            err_rand_std = np.vstack(rand_stack).std(axis=0)

            improvement = err_rand - err_trust  # positive → trust is closer to real

            for i, feat in enumerate(top_feats):
                rows.append({
                    "model": model, "arm": arm_name, "feature": feat,
                    "random_mode": random_mode,
                    "rank": i + 1,
                    "err_trust": float(err_trust[i]),
                    "err_random_mean": float(err_rand[i]),
                    "err_random_std":  float(err_rand_std[i]),
                    "improvement": float(improvement[i]),
                })

    df = pd.DataFrame(rows)
    out_path = out_dir / f"feature_deltas_{random_mode}.csv"
    df.to_csv(out_path, index=False)

    # Paired bar plot per model, unseen arm.
    for arm_name in ("seen", "unseen"):
        arm_df = df[df.arm == arm_name]
        if arm_df.empty:
            continue
        present = [m for m in models if (arm_df.model == m).any()]
        n = len(present)
        if n == 0:
            continue
        fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, model in zip(axes, present):
            sub = arm_df[arm_df.model == model].sort_values("rank")
            xs = np.arange(len(sub))
            ys = sub["improvement"].values
            colors = ["#0072B2" if v >= 0 else "#D55E00" for v in ys]
            ax.bar(xs, ys, color=colors, edgecolor="black", linewidth=0.6)
            ax.axhline(0.0, color="gray", linestyle=":", alpha=0.6)
            ax.set_ylabel("err(random) − err(trust)")
            n_pos = int((ys > 0).sum())
            ax.set_title(
                f"{model} — {arm_name} arm ({n_pos}/{len(ys)} features improved, "
                f"mean improvement = {ys.mean():+.3f})"
            )
            ax.grid(axis="y", linestyle=":", alpha=0.3)
        axes[-1].set_xticks(np.arange(len(top_feats)))
        axes[-1].set_xticklabels(top_feats, rotation=75, ha="right", fontsize=8)
        fig.tight_layout()
        fig_path = out_dir / f"feature_deltas_{random_mode}_{arm_name}.png"
        fig.savefig(fig_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {fig_path}")

    # Compact per-model summary.
    print("\n=== Feature-level improvement (condition-matched, top-%d) ===" % top_k)
    for arm_name in ("unseen",):
        print(f"\n-- arm = {arm_name} --")
        for model in models:
            sub = df[(df.model == model) & (df.arm == arm_name)]
            if sub.empty:
                continue
            n_pos = int((sub.improvement > 0).sum())
            mean_imp = float(sub.improvement.mean())
            median_imp = float(sub.improvement.median())
            print(
                f"  {model:<22s}  improved {n_pos:>2d}/{len(sub)}  "
                f"mean={mean_imp:+.3f}  median={median_imp:+.3f}"
            )
    print(f"\nWrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CP morphology validation — condition-matched on CP top-15."
    )
    p.add_argument(
        "--stage", type=str, required=True,
        choices=["centroid", "rr_ratio", "readout", "feature_deltas",
                 "feature_correlations", "all"],
    )
    p.add_argument("--cp-dir", type=Path, default=Path("/mnt/pvc/cellprofiler_outputs"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/cp_analysis"))
    p.add_argument("--encoder", type=str, default="siglip",
                   help="Trust-scoring encoder tag (→ scores_png_<tag>.json)")
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                   help="Comma-separated model keys. Default: 3 marginals.")
    p.add_argument("--top-k", type=int, default=15,
                   help="Primary CP top-k (default 15 per audit; top_features.json must have >= top-k).")
    p.add_argument("--k-list", type=str, default="5,15,20",
                   help="Comma-separated k values for --stage centroid sensitivity sweep.")
    p.add_argument("--n-random-seeds", type=int, default=5)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument(
        "--feature-space", type=str, default="cp",
        choices=["cp", "siglip", "dinov3"],
        help=("Feature space for the real classifier / gen evaluation in "
              "--stage readout. CP uses top-k from top_features.json; "
              "siglip/dinov3 load from outputs/cp_analysis/real_imgs/ + "
              "rxrx1_<model>/ caches (requires `analyze_cp_features_rxrx1.py "
              "--stage png_features --encoder <tag>` to have run)."),
    )
    p.add_argument(
        "--corr-feature-set", type=str, default="kept_pre_corr",
        choices=["kept_pre_corr", "unfiltered"],
        help="Feature set for --stage feature_correlations. `kept_pre_corr` "
             "applies variance + |z|≤5 filter (621 features). `unfiltered` "
             "uses all non-metadata columns with var > 0 (~2300+).",
    )
    p.add_argument(
        "--centroid-feature-set", type=str, default="top_k",
        choices=["top_k", "kept_pre_corr"],
        help="Feature set for --stage centroid. `top_k` uses the F_combo "
             "top-k saved in top_features.json (k from --k-list). "
             "`kept_pre_corr` uses all ~621 CP features after variance + "
             "|z|≤5 outlier filter only (no |r|>0.7 corr-prune); ignores "
             "--k-list. Output goes to centroid_distance_{mode}_kept_pre_corr.csv.",
    )
    p.add_argument(
        "--rr-feature-set", type=str, default="kept_pre_corr",
        choices=["top_k", "kept_pre_corr"],
        help="Feature set for --stage rr_ratio. Defaults to kept_pre_corr "
             "(matches the paper's main centroid table).",
    )
    p.add_argument(
        "--rr-split-repeats", type=int, default=10,
        help="Number of random half-split repeats for the μ_RR(c,s) estimate "
             "in --stage rr_ratio.",
    )
    p.add_argument(
        "--random-mode", type=str, default="combo_stratified",
        choices=["combo_stratified", "uniform"],
        help=("How to draw the random baseline. 'combo_stratified' (default): "
              "for each (cell, sirna) present in trust, draw the same count "
              "from gen at that same combo — removes combo-frequency "
              "confounding. 'uniform': draw uniformly across the arm (kept "
              "for backward-compat / fallback)."),
    )
    return p


# ---------------------------------------------------------------------------
# Stage 4: feature_correlations — per-feature Pearson r between
# {trust gen, random gen} and real, computed across condition means
# ---------------------------------------------------------------------------


def _per_combo_means(
    df: pd.DataFrame, features: List[str], combos: List[Tuple[int, int]],
) -> np.ndarray:
    """Return (len(combos), len(features)) — per-combo column means; NaN where
    the combo has no rows in df."""
    out = np.full((len(combos), len(features)), np.nan, dtype=np.float64)
    for i, (c, s) in enumerate(combos):
        sub = df[(df["cell_type_id"] == c) & (df["sirna_id"] == s)]
        if len(sub) == 0:
            continue
        out[i] = sub[features].mean(axis=0).values
    return out


def _per_feature_pearson(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson r per column between A and B, rows aligned. Drops rows with any NaN
    in either matrix. Returns (n_features,) array."""
    valid = ~(np.isnan(A).any(axis=1) | np.isnan(B).any(axis=1))
    A = A[valid]; B = B[valid]
    if len(A) < 3:
        return np.full(A.shape[1], np.nan)
    A_c = A - A.mean(axis=0)
    B_c = B - B.mean(axis=0)
    num = (A_c * B_c).sum(axis=0)
    den = np.sqrt((A_c ** 2).sum(axis=0) * (B_c ** 2).sum(axis=0)) + 1e-12
    return num / den


def stage_feature_correlations(
    cp_dir: Path, output_dir: Path, models: List[str], encoder_tag: str,
    n_random_seeds: int, out_dir: Path,
    random_mode: str = "combo_stratified",
    feature_set: str = "kept_pre_corr",
) -> None:
    """For each (model, arm), per CP feature compute Pearson r over the arm's
    condition means between {trust-gen, random-gen} and real. Report top/bottom
    10 by r_trust and r_random plus the full per-feature CSV."""
    from cp_decile_selection_sensitivity import (
        _compute_kept_features_and_scaler,
        _compute_unfiltered_features_and_scaler,
    )

    arms = load_rxrx1_subset_arms()
    seen, unseen = arms["seen"], arms["unseen"]
    arm_sets = {"seen": seen, "unseen": unseen}

    if feature_set == "unfiltered":
        kept, scaler, df_real_all = _compute_unfiltered_features_and_scaler(cp_dir)
    else:
        kept, scaler, df_real_all = _compute_kept_features_and_scaler(cp_dir)
    logger.info(f"[feature_correlations] using {feature_set} ({len(kept)} features)")

    summary_rows: List[Dict] = []
    full_rows:    List[Dict] = []

    for model in models:
        df_gen, t95 = _load_gen_df(
            cp_dir, output_dir, model, encoder_tag, kept, seen, unseen, scaler,
        )
        for arm_name in ("seen", "unseen"):
            sub = df_gen[df_gen["split"] == arm_name].reset_index(drop=True)
            if sub.empty:
                continue
            arm_combos = sorted(arm_sets[arm_name])
            real_arm = df_real_all[df_real_all["split"] == arm_name]
            real_means = _per_combo_means(real_arm, kept, arm_combos)

            trust_mask = (sub["trust_updated"] <= t95).values
            trust_sub = sub[trust_mask].reset_index(drop=True)
            n_trust = int(len(trust_sub))
            if n_trust < 10:
                continue
            trust_means = _per_combo_means(trust_sub, kept, arm_combos)

            r_trust = _per_feature_pearson(trust_means, real_means)

            # Random baseline: n_random_seeds × per-feature Pearson; average.
            r_random_per_seed = []
            for sd in range(n_random_seeds):
                ridx = _random_indices(sub, trust_mask, seed=sd, mode=random_mode)
                rand_sub = sub.iloc[ridx].reset_index(drop=True)
                rand_means = _per_combo_means(rand_sub, kept, arm_combos)
                r_random_per_seed.append(_per_feature_pearson(rand_means, real_means))
            r_random = np.nanmean(np.vstack(r_random_per_seed), axis=0)

            r_diff = r_trust - r_random

            # Top/bottom 10 by signed r (trust ranking, random ranking),
            # plus bot-10 by |r| ascending — the "uninformative" view.
            # NaN r (zero-variance features) treated as |r|=0 so they sort first.
            r_trust_clean  = np.where(np.isfinite(r_trust),  r_trust,  np.nan)
            r_random_clean = np.where(np.isfinite(r_random), r_random, np.nan)
            order_trust  = np.argsort(-r_trust_clean)  # NaNs go to end
            order_random = np.argsort(-r_random_clean)
            abs_r_trust_safe  = np.where(np.isnan(r_trust_clean),  0.0, np.abs(r_trust_clean))
            abs_r_random_safe = np.where(np.isnan(r_random_clean), 0.0, np.abs(r_random_clean))
            order_abs_trust   = np.argsort(abs_r_trust_safe)   # ascending → uninformative first
            order_abs_random  = np.argsort(abs_r_random_safe)

            def _emit(rank_label: str, rank_arr: np.ndarray) -> None:
                for rank, j in enumerate(rank_arr, 1):
                    summary_rows.append({
                        "model": model, "arm": arm_name, "rank_label": rank_label,
                        "rank": rank, "feature": kept[j],
                        "r_trust": float(r_trust[j]), "r_random": float(r_random[j]),
                        "r_diff_trust_minus_random": float(r_diff[j]),
                    })

            _emit("top10_r_trust",     order_trust[:10])
            _emit("bot10_r_trust",     order_trust[-10:][::-1])
            _emit("top10_r_random",    order_random[:10])
            _emit("bot10_r_random",    order_random[-10:][::-1])
            _emit("uninformative_trust",  order_abs_trust[:10])   # |r_trust| smallest
            _emit("uninformative_random", order_abs_random[:10])  # |r_random| smallest

            # Full per-feature dump for downstream use.
            for j in range(len(kept)):
                full_rows.append({
                    "model": model, "arm": arm_name, "feature": kept[j],
                    "r_trust":   float(r_trust[j]),
                    "r_random":  float(r_random[j]),
                    "r_diff":    float(r_diff[j]),
                })

    summ_df = pd.DataFrame(summary_rows)
    full_df = pd.DataFrame(full_rows)
    rmode_suf = "" if random_mode == "combo_stratified" else f"_{random_mode}"
    fs_tag = feature_set
    summ_path = out_dir / f"feature_corr_summary_{fs_tag}{rmode_suf}.csv"
    full_path = out_dir / f"feature_corr_full_{fs_tag}{rmode_suf}.csv"
    summ_df.to_csv(summ_path, index=False)
    full_df.to_csv(full_path, index=False)

    print(f"\n=== Per-feature Pearson r ({feature_set}, {len(kept)} features, random_mode={random_mode}) ===")
    for (model, arm_name), sub in summ_df.groupby(["model", "arm"]):
        print(f"\n-- {model} / {arm_name} --")
        for rank_label in ("top10_r_trust", "bot10_r_trust",
                           "top10_r_random", "bot10_r_random",
                           "uninformative_trust", "uninformative_random"):
            print(f"  {rank_label}:")
            sl = sub[sub["rank_label"] == rank_label].sort_values("rank")
            for _, row in sl.iterrows():
                print(f"    {row['rank']:>2}.  r_trust={row['r_trust']:+.3f}  "
                      f"r_random={row['r_random']:+.3f}  "
                      f"Δ={row['r_diff_trust_minus_random']:+.3f}  {row['feature']}")
    print(f"\nWrote {summ_path}\nWrote {full_path}")


def main() -> None:
    args = build_parser().parse_args()
    if not args.cp_dir.exists():
        raise SystemExit(f"--cp-dir does not exist: {args.cp_dir}")
    models = resolve_models(args.models)

    out_dir = args.output_dir / "morphology_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]

    if args.stage in ("centroid", "all"):
        stage_centroid(
            cp_dir=args.cp_dir, output_dir=args.output_dir, models=models,
            encoder_tag=args.encoder,
            k_list=k_list,
            n_random_seeds=args.n_random_seeds, n_boot=args.n_boot,
            out_dir=out_dir, random_mode=args.random_mode,
            feature_set=args.centroid_feature_set,
        )
    if args.stage in ("rr_ratio", "all"):
        stage_rr_ratio(
            cp_dir=args.cp_dir, output_dir=args.output_dir, models=models,
            encoder_tag=args.encoder,
            k_list=k_list,
            n_random_seeds=args.n_random_seeds, n_boot=args.n_boot,
            out_dir=out_dir, random_mode=args.random_mode,
            feature_set=args.rr_feature_set,
            n_split_repeats=args.rr_split_repeats,
        )
    if args.stage in ("readout", "all"):
        stage_readout(
            cp_dir=args.cp_dir, output_dir=args.output_dir, models=models,
            encoder_tag=args.encoder, top_k=args.top_k,
            n_random_seeds=args.n_random_seeds, out_dir=out_dir,
            random_mode=args.random_mode, feature_space=args.feature_space,
        )
    if args.stage in ("feature_deltas", "all"):
        stage_feature_deltas(
            cp_dir=args.cp_dir, output_dir=args.output_dir, models=models,
            encoder_tag=args.encoder, top_k=args.top_k,
            n_random_seeds=args.n_random_seeds, out_dir=out_dir,
            random_mode=args.random_mode,
        )
    if args.stage in ("feature_correlations", "all"):
        stage_feature_correlations(
            cp_dir=args.cp_dir, output_dir=args.output_dir, models=models,
            encoder_tag=args.encoder,
            n_random_seeds=args.n_random_seeds, out_dir=out_dir,
            random_mode=args.random_mode,
            feature_set=args.corr_feature_set,
        )


if __name__ == "__main__":
    main()
