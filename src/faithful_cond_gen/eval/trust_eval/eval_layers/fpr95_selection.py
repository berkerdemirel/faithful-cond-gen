"""
Task 4: FPR@95 Selection + z-KID Evaluation.

Evaluate quality of generated samples that pass at FPR@95 threshold.
Now with:
- Per-condition FPR for ALL datasets (not just CelebA)
- Multi-threshold (P50/P75/P90/P95)
- Bootstrap CIs for per-condition KID
- Per-condition scatter plots and bar charts
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from faithful_cond_gen.eval.trust_eval.condition_utils import (
    filter_feats_and_meta_by_seen_combos,
)
from faithful_cond_gen.eval.trust_eval.metrics_kid import (
    calculate_kid_same_m,
    estimate_kid_null_per_condition,
)
from faithful_cond_gen.eval.trust_eval.scoring_core import (
    compute_real_sample_scores,
    fit_trust_scoring_components,
)

# Constants for per-condition evaluation
PCTS = [50, 75, 90, 95]
N_BOOT = 10
N_RAND = 10
KID_K = 500
COMPONENTS = ["trust", "realism", "faithfulness"]
COLORS = {"trust": "#1f77b4", "realism": "#ff7f0e", "faithfulness": "#2ca02c"}


def _get_cond(meta, i, condition_keys):
    return tuple(int(meta[k][i]) for k in condition_keys)


def _group_by_cond(meta, n, condition_keys):
    out = {}
    for i in range(n):
        c = _get_cond(meta, i, condition_keys)
        out.setdefault(c, []).append(i)
    return out


def _hamming(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def _build_calib_fallback(
    gen_conds: List[Tuple[int, ...]],
    real_score_by_cond: Dict[Tuple[int, ...], List[int]],
    seen_combos: Optional[Set[Tuple[int, ...]]],
) -> Dict[Tuple[int, ...], List[Tuple[int, ...]]]:
    """Map each gen cond -> list of calib conds for p95 threshold.

    Full models (seen_combos=None): each gen cond maps to [itself].

    Marginal models: seen gen conds map to [itself]. Unseen gen conds map
    to **all** seen conds tied at the minimum Hamming distance — the
    per-cond selection is then run once per calib cond and the KID results
    are averaged (no sample-size tiebreak).

    Only seen combos with calibration support are considered.
    """
    if not seen_combos:
        return {c: [c] for c in gen_conds}

    usable_seen = [s for s in seen_combos if len(real_score_by_cond.get(s, [])) > 0]
    fallback: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
    for c in gen_conds:
        if c in seen_combos and len(real_score_by_cond.get(c, [])) > 0:
            fallback[c] = [c]
            continue
        if not usable_seen:
            fallback[c] = [c]
            continue
        dists = {s: _hamming(c, s) for s in usable_seen}
        d_min = min(dists.values())
        fallback[c] = sorted(s for s, d in dists.items() if d == d_min)
    return fallback


def _bootstrap_kid(gen_feats, real_feats, k=KID_K, n_boot=N_BOOT, seed=42):
    """Bootstrap KID estimate. Returns (mean, std)."""
    eff_k = min(k, len(gen_feats), len(real_feats))
    if eff_k < 10:
        return np.nan, np.nan
    vals = []
    for b in range(n_boot):
        rng = np.random.default_rng(seed + b)
        gi = rng.permutation(len(gen_feats))[:eff_k]
        ri = rng.permutation(len(real_feats))[:eff_k]
        kv = calculate_kid_same_m(gen_feats[gi], real_feats[ri], use_cosine=True)
        if np.isfinite(kv):
            vals.append(kv)
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, np.nan)


def _fpr_per_cond(gen_score, real_score, real_meta, kid_gen, kid_real,
                  gen_by_cond, real_kid_by_cond, real_score_by_cond,
                  condition_keys, pct=95, seed=42,
                  calib_fallback: Optional[Dict[Tuple[int, ...], List[Tuple[int, ...]]]] = None):
    """Per-condition FPR at given percentile threshold.

    For each gen cond we look up a list of calib conds in calib_fallback.
    Seen gen conds have a single-entry list; unseen gen conds have every
    seen cond tied at the minimum Hamming distance. The selection is run
    once per calib cond (each gives its own threshold, its own accepted
    set, and its own bootstrap KID_acc/KID_rand); the reported per-cond
    result is the mean of those sub-results. KID pools stay same-cond
    regardless of calibration source.

    Returns dict[cond_str] -> {n_accepted, n_total, threshold, calib_cond,
    n_calib_used, kid_acc, kid_rand, delta, ci_*}.
    """
    out = {}
    for cond in sorted(gen_by_cond.keys()):
        gen_idx_c = gen_by_cond[cond]
        calib_list = calib_fallback.get(cond, [cond]) if calib_fallback else [cond]
        real_kid_idx_c = real_kid_by_cond.get(cond, [])

        if len(gen_idx_c) < 20 or len(real_kid_idx_c) < 20:
            continue

        sub = []
        for cc in calib_list:
            real_score_idx_cc = real_score_by_cond.get(cc, [])
            if len(real_score_idx_cc) < 20:
                continue
            rs_c = real_score[real_score_idx_cc]
            valid_r = np.isfinite(rs_c)
            if np.sum(valid_r) < 20:
                continue
            threshold = float(np.percentile(rs_c[valid_r], pct))

            accept_idx = [i for i in gen_idx_c
                          if np.isfinite(gen_score[i]) and gen_score[i] <= threshold]
            if len(accept_idx) < 10:
                continue

            eff_k = min(KID_K, len(accept_idx), len(real_kid_idx_c))
            if eff_k < 10:
                continue

            h = (hash(cond) + hash(cc)) % 10000

            acc_vals = []
            for b in range(N_BOOT):
                rng = np.random.default_rng(seed + 4000 + h + b)
                gi = rng.choice(accept_idx, eff_k, replace=False)
                ri = rng.choice(real_kid_idx_c, eff_k, replace=False)
                kv = calculate_kid_same_m(kid_gen[gi], kid_real[ri], use_cosine=True)
                if np.isfinite(kv):
                    acc_vals.append(kv)

            rand_vals = []
            for rep in range(N_RAND):
                rng = np.random.default_rng(seed + 5000 + h + rep)
                n_draw = min(len(accept_idx), len(gen_idx_c))
                ridx = rng.choice(gen_idx_c, n_draw, replace=False)
                eff_k_r = min(KID_K, len(ridx), len(real_kid_idx_c))
                if eff_k_r < 10:
                    continue
                gi = rng.choice(ridx, eff_k_r, replace=False)
                ri = rng.choice(real_kid_idx_c, eff_k_r, replace=False)
                kv = calculate_kid_same_m(kid_gen[gi], kid_real[ri], use_cosine=True)
                if np.isfinite(kv):
                    rand_vals.append(kv)

            if not acc_vals or not rand_vals:
                continue

            kid_acc_m = float(np.mean(acc_vals))
            kid_acc_s = float(np.std(acc_vals))
            kid_rand_m = float(np.mean(rand_vals))
            kid_rand_s = float(np.std(rand_vals))
            sub.append(dict(
                calib_cond=cc,
                threshold=threshold,
                n_accepted=len(accept_idx),
                kid_acc_m=kid_acc_m, kid_acc_s=kid_acc_s,
                kid_rand_m=kid_rand_m, kid_rand_s=kid_rand_s,
            ))

        if not sub:
            continue

        # Average across all sub-results (one per min-Hamming calib cond).
        kid_acc_m = float(np.mean([s["kid_acc_m"] for s in sub]))
        kid_acc_s = float(np.mean([s["kid_acc_s"] for s in sub]))
        kid_rand_m = float(np.mean([s["kid_rand_m"] for s in sub]))
        kid_rand_s = float(np.mean([s["kid_rand_s"] for s in sub]))
        threshold = float(np.mean([s["threshold"] for s in sub]))
        n_accepted = int(round(float(np.mean([s["n_accepted"] for s in sub]))))
        delta = kid_rand_m - kid_acc_m

        ci_lower = delta - 1.96 * np.sqrt(kid_acc_s**2 + kid_rand_s**2) if np.isfinite(kid_acc_s) and np.isfinite(kid_rand_s) else np.nan
        ci_upper = delta + 1.96 * np.sqrt(kid_acc_s**2 + kid_rand_s**2) if np.isfinite(kid_acc_s) and np.isfinite(kid_rand_s) else np.nan
        improvement_pct = (delta / kid_rand_m * 100) if np.isfinite(kid_rand_m) and kid_rand_m != 0 else np.nan

        calib_strs = [str(s["calib_cond"]) for s in sub]
        out[str(cond)] = dict(
            n_accepted=n_accepted,
            n_total=len(gen_idx_c),
            acceptance_rate=n_accepted / len(gen_idx_c),
            threshold=threshold,
            calib_cond=" | ".join(calib_strs),
            calib_borrowed=bool(len(calib_list) != 1 or calib_list[0] != cond),
            n_calib_used=len(sub),
            kid_acc=(kid_acc_m, kid_acc_s),
            kid_rand=(kid_rand_m, kid_rand_s),
            delta=delta,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            improvement_pct=improvement_pct,
        )
    return out


def _plot_percond_scatter(percond_by_comp, model, output_dir, config_key, dataset, pct=95):
    """Per-condition scatter plot: KID(random) vs KID(accepted). Points below diagonal = improvement."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ci, comp in enumerate(COMPONENTS):
        ax = axes[ci]
        data = percond_by_comp.get(comp, {})
        acc, rand = [], []
        for cond_str in sorted(data.keys()):
            r = data[cond_str]
            if np.isfinite(r["kid_acc"][0]) and np.isfinite(r["kid_rand"][0]):
                acc.append(r["kid_acc"][0])
                rand.append(r["kid_rand"][0])
        if not acc:
            ax.set_title(f"{comp.capitalize()} (no data)")
            continue
        ax.scatter(rand, acc, c=COLORS[comp], alpha=0.7, s=50, zorder=3)
        mn = min(min(acc), min(rand))
        mx = max(max(acc), max(rand))
        pad = (mx - mn) * 0.1 + 1e-6
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], 'k--', alpha=0.3, linewidth=1)
        n_better = sum(1 for a, r in zip(acc, rand) if a < r)
        ax.set_xlabel("KID (random)")
        ax.set_ylabel("KID (accepted)")
        ax.set_title(f"{comp.capitalize()} ({n_better}/{len(acc)} improved)")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Per-Cond FPR@{pct}: {model} [{dataset}]", fontsize=12)
    fig.tight_layout()
    safe_key = config_key.replace("/", "_")
    path = output_dir / f"{dataset}_fpr95_percond_{safe_key}_P{pct}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _plot_fpr_bar(fpr_global_by_comp, model, output_dir, config_key, dataset):
    """Bar chart: Accepted vs Random at each threshold (P50/P75/P90/P95) x 3 components."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ci, comp in enumerate(COMPONENTS):
        ax = axes[ci]
        data = fpr_global_by_comp.get(comp, {})
        pcts_valid, acc_m, acc_s, rand_m, rand_s = [], [], [], [], []
        for pct in PCTS:
            r = data.get(pct)
            if r is None:
                continue
            pcts_valid.append(pct)
            acc_m.append(r["kid_acc"][0]); acc_s.append(r["kid_acc"][1])
            rand_m.append(r["kid_rand"][0]); rand_s.append(r["kid_rand"][1])
        if not pcts_valid:
            ax.set_title(f"{comp.capitalize()} (no data)")
            continue
        x = np.arange(len(pcts_valid))
        w = 0.35
        ax.bar(x - w/2, acc_m, w, yerr=acc_s, label="Accepted",
               color=COLORS[comp], alpha=0.8, capsize=3)
        ax.bar(x + w/2, rand_m, w, yerr=rand_s, label="Random",
               color="gray", alpha=0.6, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"P{p}" for p in pcts_valid])
        ax.set_title(comp.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    axes[0].set_ylabel("KID (lower = better)")
    fig.suptitle(f"FPR Selection: {model} [{dataset}]", fontsize=12)
    fig.tight_layout()
    safe_key = config_key.replace("/", "_")
    path = output_dir / f"{dataset}_fpr95_bar_{safe_key}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _build_condmatched_real(gen_indices, gen_meta, real_by_cond, condition_keys, seed):
    """Build condition-matched real pool. Returns (real_indices, cond_hist)."""
    cond_hist = {}
    for i in gen_indices:
        c = _get_cond(gen_meta, i, condition_keys)
        cond_hist[c] = cond_hist.get(c, 0) + 1

    rng = np.random.default_rng(seed)
    pool = []
    for c in sorted(cond_hist.keys()):
        rc = real_by_cond.get(c, [])
        if not rc:
            continue
        n = cond_hist[c]
        pool.extend(rng.choice(rc, n, replace=(len(rc) < n)))
    return np.array(pool, dtype=int), cond_hist


def _build_condmatched_capped(gen_indices, gen_meta, real_by_cond, condition_keys, seed):
    """Condition-matched pair preserving the gen histogram, no real oversampling.

    Let n_g_c = |gen ∩ cond c|, n_r_c = |real of cond c|. We compute a single
    global scale s = min over conds (with n_g_c > 0 and n_r_c > 0) of
    n_r_c / n_g_c, clipped to [0, 1]. Per cond, we keep round(n_g_c * s) from
    gen and the same count from real (both without replacement).

    This preserves the relative per-cond weights of the input gen pool but
    shrinks the total to whatever the rarest real cell allows, so the real
    side is never duplicated. Conditions with no reals are dropped from both.

    Returns (gen_indices_kept, real_indices_matched, cond_counts_kept).
    """
    g_by_cond: Dict[Tuple, List[int]] = {}
    for gi in gen_indices:
        c = _get_cond(gen_meta, gi, condition_keys)
        g_by_cond.setdefault(c, []).append(int(gi))

    # Keep only conds with real support
    ratios = []
    for c, g in g_by_cond.items():
        r = real_by_cond.get(c, [])
        if g and r:
            ratios.append(len(r) / len(g))
    if not ratios:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), {}

    s = min(1.0, min(ratios))

    rng = np.random.default_rng(seed)
    gen_kept = []
    real_matched = []
    counts: Dict[Tuple, int] = {}
    for c in sorted(g_by_cond.keys()):
        g = g_by_cond[c]
        r = real_by_cond.get(c, [])
        if not r:
            continue
        n = int(round(len(g) * s))
        if n <= 0:
            continue
        n = min(n, len(g), len(r))
        g_sub = rng.choice(g, n, replace=False) if n < len(g) else np.asarray(g)
        r_sub = rng.choice(r, n, replace=False) if n < len(r) else np.asarray(r)
        gen_kept.extend(int(x) for x in g_sub)
        real_matched.extend(int(x) for x in r_sub)
        counts[c] = n
    return (
        np.array(gen_kept, dtype=int),
        np.array(real_matched, dtype=int),
        counts,
    )


def _fpr_global(gen_score, real_score, kid_gen, kid_real,
                gen_meta, gen_by_cond, real_by_cond, condition_keys, seed=42,
                cap_real_pool: bool = False):
    """FPR at each percentile (global). Returns dict[pct] -> result.

    Random baseline: truly random gen (same N) with its OWN condition-matched
    real pool, so the comparison captures both condition-selection and
    within-condition quality effects.

    When cap_real_pool=False (default) the real pool is sampled with
    replacement per cond to match the gen histogram exactly (legacy path).
    When True, both gen and real are proportionally downscaled so no real
    sample is reused — see _build_condmatched_capped.
    """
    n_gen = len(gen_score)
    valid_real = np.isfinite(real_score)
    out = {}
    for pct in PCTS:
        threshold = float(np.percentile(real_score[valid_real], pct))
        accept_idx = np.where(np.isfinite(gen_score) & (gen_score <= threshold))[0]
        if len(accept_idx) < 20:
            out[pct] = None
            continue

        if cap_real_pool:
            g_acc, rp_acc, _ = _build_condmatched_capped(
                accept_idx, gen_meta, real_by_cond, condition_keys,
                seed=seed + 1000 + pct)
        else:
            rp_acc, _ = _build_condmatched_real(
                accept_idx, gen_meta, real_by_cond, condition_keys,
                seed=seed + 1000 + pct)
            g_acc = accept_idx
        if len(rp_acc) < 10:
            out[pct] = None
            continue
        kid_acc_m, kid_acc_s = _bootstrap_kid(
            kid_gen[g_acc], kid_real[rp_acc],
            k=KID_K, n_boot=N_BOOT, seed=seed + 1100 + pct)

        # Random: truly random gen (same N) vs condition-matched real.
        n_acc = len(accept_idx)
        rand_vals = []
        for rep in range(N_RAND):
            rng = np.random.default_rng(seed + 2000 + pct * 100 + rep)
            rand_idx = rng.choice(n_gen, n_acc, replace=False)
            if cap_real_pool:
                g_rand, rp_rand, _ = _build_condmatched_capped(
                    rand_idx, gen_meta, real_by_cond, condition_keys,
                    seed=seed + 3000 + pct * 100 + rep)
            else:
                rp_rand, _ = _build_condmatched_real(
                    rand_idx, gen_meta, real_by_cond, condition_keys,
                    seed=seed + 3000 + pct * 100 + rep)
                g_rand = rand_idx
            if len(rp_rand) < 10:
                continue
            km, _ = _bootstrap_kid(
                kid_gen[g_rand], kid_real[rp_rand],
                k=KID_K, n_boot=1, seed=seed + 4000 + rep)
            if np.isfinite(km):
                rand_vals.append(km)

        kid_rand_m = float(np.mean(rand_vals)) if rand_vals else np.nan
        kid_rand_s = float(np.std(rand_vals)) if rand_vals else np.nan

        out[pct] = dict(
            n_accepted=int(len(accept_idx)),
            n_accepted_after_cap=int(len(g_acc)),
            n_total=int(np.sum(np.isfinite(gen_score))),
            threshold=float(threshold),
            kid_acc=(kid_acc_m, kid_acc_s),
            kid_rand=(kid_rand_m, kid_rand_s),
            delta=kid_rand_m - kid_acc_m,
        )
    return out


def evaluate_fpr95_selection(
    trust_results: Dict,
    real_feats: torch.Tensor,
    real_meta: Dict,
    gen_feats: torch.Tensor,
    gen_meta: Dict,
    condition_keys: List[str],
    dataset: str,
    model: str,
    output_dir: Path,
    config_key: str,
    kid_real_feats: torch.Tensor = None,
    kid_gen_feats: torch.Tensor = None,
    kid_real_meta: Dict = None,
    kid_mode: str = "auto",
    feature_type: str = "dinov3",
    use_kid_z: bool = False,
    seed: int = 42,
    n_random_repeats: int = 5,
    seen_combos: set = None,
    cap_real_pool: bool = False,
) -> Dict:
    """
    Evaluate quality of generated samples that pass at FPR@95.

    Now includes:
    - Per-condition FPR for ALL datasets (CelebA + RxRx1)
    - Multi-threshold (P50/P75/P90/P95) global and per-condition
    - Bootstrap CIs for all KID estimates
    - Per-condition scatter plots and bar charts
    """
    rng = np.random.default_rng(seed)

    if kid_real_feats is None:
        kid_real_feats = real_feats
    if kid_gen_feats is None:
        kid_gen_feats = gen_feats

    # Get generated sample scores
    gen_scores_trust = trust_results["trust_updated"]
    gen_scores_realism = trust_results["realism_global_z"]
    gen_scores_faithfulness = trust_results["faithfulness_margin_z"]
    gen_conditions = trust_results["true_conditions"]
    n_gen = len(gen_scores_trust)

    # For marginal models, restrict calibration to seen combos
    calib_feats, calib_meta = filter_feats_and_meta_by_seen_combos(
        real_feats, real_meta, condition_keys, seen_combos
    )

    # Fit scoring on calibration set and score both gen and real
    components = fit_trust_scoring_components(
        calib_feats, calib_meta, condition_keys,
    )

    from faithful_cond_gen.eval.trust_eval.scoring_core import score_trust_from_components
    real_realism, real_faithfulness, real_trust = score_trust_from_components(
        calib_feats, calib_meta, components
    )

    # Convert KID features to numpy (L2-normed)
    kid_real_np = kid_real_feats.numpy() if isinstance(kid_real_feats, torch.Tensor) else kid_real_feats
    kid_gen_np = kid_gen_feats.numpy() if isinstance(kid_gen_feats, torch.Tensor) else kid_gen_feats

    # Build condition groupings for KID (DINO space) and scoring space
    gen_by_cond: Dict[Tuple, List[int]] = {}
    for i in range(n_gen):
        gen_by_cond.setdefault(gen_conditions[i], []).append(i)

    # Real grouping in KID space (full dataset, not just calibration)
    # Use kid_real_meta if provided (needed when KID real has different N than scoring real)
    n_kid_real = len(kid_real_np)
    if kid_real_meta is None:
        kid_real_meta = real_meta
    kid_real_by_cond: Dict[Tuple, List[int]] = {}
    for i in range(n_kid_real):
        c = tuple(
            int(kid_real_meta[k][i].item() if isinstance(kid_real_meta[k][i], torch.Tensor) else kid_real_meta[k][i])
            for k in condition_keys
        )
        kid_real_by_cond.setdefault(c, []).append(i)

    # Real grouping in scoring space (calibration set)
    n_calib = len(calib_feats)
    real_score_by_cond: Dict[Tuple, List[int]] = {}
    for i in range(n_calib):
        c = tuple(
            int(calib_meta[k][i].item() if isinstance(calib_meta[k][i], torch.Tensor) else calib_meta[k][i])
            for k in condition_keys
        )
        real_score_by_cond.setdefault(c, []).append(i)

    # Score arrays: {component: gen_scores, real_scores}
    score_arrays = {
        "trust": (gen_scores_trust, real_trust),
        "realism": (gen_scores_realism, real_realism),
        "faithfulness": (gen_scores_faithfulness, real_faithfulness),
    }

    # --- Global FPR at P95 (legacy, for backward compat / summary) ---
    # NOTE: trust may have zero-acceptance in spaces where real/gen distributions
    # are disjoint (e.g. rxrx1 posthoc_mapped). We still run the multi-threshold
    # section below so realism/faithfulness components are reported even when
    # trust degenerates. Legacy fields are set to None in that case.
    valid_real = np.isfinite(real_trust)
    results: Dict = {"status": "success"}
    accept_indices = None
    if valid_real.sum() < 10:
        results.update({
            "status": "insufficient_real_samples",
            "n_valid_real": int(valid_real.sum()),
            "threshold_95": None,
            "n_accepted": 0,
            "acceptance_rate": 0.0,
            "n_conditions_in_accepted": 0,
            "kid_sample_size": 0,
            "kid_raw_accepted": None,
            "kid_raw_accepted_std": None,
            "n_accept_repeats": 0,
            "kid_raw_random": None,
            "kid_raw_random_std": None,
            "n_random_repeats": 0,
        })
        t_95 = None
        n_accepted = 0
    else:
        t_95 = float(np.percentile(real_trust[valid_real], 95))
        valid_gen = np.isfinite(gen_scores_trust)
        accept_mask = valid_gen & (gen_scores_trust <= t_95)
        n_accepted = int(accept_mask.sum())

    if t_95 is not None and n_accepted < 10:
        results.update({
            "status": "insufficient_accepted",
            "threshold_95": t_95,
            "n_accepted": n_accepted,
            "acceptance_rate": n_accepted / n_gen,
            "n_conditions_in_accepted": 0,
            "kid_sample_size": 0,
            "kid_raw_accepted": None,
            "kid_raw_accepted_std": None,
            "n_accept_repeats": 0,
            "kid_raw_random": None,
            "kid_raw_random_std": None,
            "n_random_repeats": 0,
        })
        accept_indices = None
    elif t_95 is not None:
        accept_indices = np.where(accept_mask)[0]

    if accept_indices is not None:
        # Mirror the random baseline protocol: redraw the condition-matched real
        # pool inside the rep loop with a different seed each rep. This is the
        # only source of variance when n_accept <= kid_sample_size, otherwise the
        # permutation subsampling on the gen side adds further variance.
        kid_sample_size_probe = min(len(accept_indices), 500)
        kid_accept_values = []
        cond_hist = {}
        n_accept_repeats = n_random_repeats
        for rep in range(n_accept_repeats):
            rep_rng = np.random.default_rng(seed + 100 + rep)
            if cap_real_pool:
                g_acc_used, match_real_indices, cond_hist_rep = _build_condmatched_capped(
                    accept_indices, gen_meta, kid_real_by_cond, condition_keys,
                    seed=seed + 700 + rep)
            else:
                match_real_indices, cond_hist_rep = _build_condmatched_real(
                    accept_indices, gen_meta, kid_real_by_cond, condition_keys,
                    seed=seed + 700 + rep)
                g_acc_used = accept_indices
            if rep == 0:
                cond_hist = cond_hist_rep
            gen_accept_all = kid_gen_np[g_acc_used]
            real_match_all = kid_real_np[match_real_indices]
            ks = min(len(gen_accept_all), len(real_match_all), 500)
            if ks < 10:
                continue
            perm_g = rep_rng.permutation(len(gen_accept_all))[:ks]
            perm_r = rep_rng.permutation(len(real_match_all))[:ks]
            kid_val = calculate_kid_same_m(gen_accept_all[perm_g], real_match_all[perm_r], use_cosine=True)
            if np.isfinite(kid_val):
                kid_accept_values.append(kid_val)
        kid_sample_size = kid_sample_size_probe

        kid_accept = float(np.mean(kid_accept_values)) if kid_accept_values else np.nan
        kid_accept_std = float(np.std(kid_accept_values)) if len(kid_accept_values) > 1 else np.nan

        # Random baseline: truly random gen (same N) with its condition-matched real,
        # sharing the same cap/replacement regime as the accepted path.
        kid_random_values = []
        for rep in range(n_random_repeats):
            rep_rng = np.random.default_rng(seed + rep + 1)
            random_indices = rep_rng.choice(n_gen, n_accepted, replace=False)
            if cap_real_pool:
                g_rand_used, rand_real_indices, _ = _build_condmatched_capped(
                    random_indices, gen_meta, kid_real_by_cond, condition_keys,
                    seed=seed + 500 + rep)
            else:
                rand_real_indices, _ = _build_condmatched_real(
                    random_indices, gen_meta, kid_real_by_cond, condition_keys,
                    seed=seed + 500 + rep)
                g_rand_used = random_indices
            gen_random = kid_gen_np[g_rand_used]
            real_random_match = kid_real_np[rand_real_indices] if len(rand_real_indices) > 0 else np.array([])
            ks = min(len(gen_random), len(real_random_match), 500)
            if ks >= 10:
                perm_rand = rep_rng.permutation(len(gen_random))[:ks]
                perm_r = rep_rng.permutation(len(real_random_match))[:ks]
                kid_rand = calculate_kid_same_m(gen_random[perm_rand], real_random_match[perm_r], use_cosine=True)
                if np.isfinite(kid_rand):
                    kid_random_values.append(kid_rand)

        kid_random = float(np.mean(kid_random_values)) if kid_random_values else np.nan
        kid_random_std = float(np.std(kid_random_values)) if len(kid_random_values) > 1 else np.nan

        results.update({
            "status": "success",
            "threshold_95": t_95,
            "n_accepted": n_accepted,
            "acceptance_rate": n_accepted / n_gen,
            "n_conditions_in_accepted": len(cond_hist),
            "kid_sample_size": kid_sample_size,
            "kid_raw_accepted": float(kid_accept) if np.isfinite(kid_accept) else None,
            "kid_raw_accepted_std": float(kid_accept_std) if np.isfinite(kid_accept_std) else None,
            "n_accept_repeats": n_accept_repeats,
            "kid_raw_random": float(kid_random) if np.isfinite(kid_random) else None,
            "kid_raw_random_std": float(kid_random_std) if np.isfinite(kid_random_std) else None,
            "n_random_repeats": n_random_repeats,
        })

    # --- Multi-threshold global FPR (P50/P75/P90/P95) per component ---
    print(f"    Multi-threshold FPR (global):")
    fpr_global_by_comp = {}
    for comp in COMPONENTS:
        gen_sc, real_sc = score_arrays[comp]
        fpr_global_by_comp[comp] = _fpr_global(
            gen_sc, real_sc, kid_gen_np, kid_real_np,
            gen_meta, gen_by_cond, kid_real_by_cond, condition_keys, seed=seed,
            cap_real_pool=cap_real_pool)
        for pct in PCTS:
            r = fpr_global_by_comp[comp].get(pct)
            if r is None:
                continue
            marker = "+" if r['delta'] > 0 else "-"
            print(f"      {comp:15s} P{pct:2d}: n={r['n_accepted']:5d}/{r['n_total']:5d}  "
                  f"acc={r['kid_acc'][0]:.5f}+/-{r['kid_acc'][1]:.5f}  "
                  f"rand={r['kid_rand'][0]:.5f}+/-{r['kid_rand'][1]:.5f}  "
                  f"D={r['delta']:+.5f} {marker}")
    results["fpr_global_multithreshold"] = fpr_global_by_comp

    # Bar chart for global multi-threshold
    bar_path = _plot_fpr_bar(fpr_global_by_comp, model, output_dir, config_key, dataset)
    print(f"    -> {bar_path}")

    # --- Per-condition FPR at P95 (for ALL datasets) ---
    # For marginal models, unseen gen conds borrow the p95 threshold from the
    # nearest seen cond (Hamming, tie -> largest calibration-side support).
    # Full models (seen_combos=None) keep the identity map.
    calib_fallback = _build_calib_fallback(
        list(gen_by_cond.keys()), real_score_by_cond, seen_combos
    )
    n_borrowed = sum(1 for c, lst in calib_fallback.items() if not (len(lst) == 1 and lst[0] == c))
    if n_borrowed:
        borrow_sizes = [len(lst) for c, lst in calib_fallback.items()
                        if not (len(lst) == 1 and lst[0] == c)]
        print(f"    Calib fallback: {n_borrowed}/{len(calib_fallback)} gen conds "
              f"borrow p95 threshold from min-Hamming seen conds "
              f"(avg #calib={np.mean(borrow_sizes):.1f}, max={max(borrow_sizes)})")

    print(f"    Per-condition FPR@95:")
    percond_by_comp = {}
    percond_summary: Dict = {}
    for comp in COMPONENTS:
        gen_sc, real_sc = score_arrays[comp]
        percond_by_comp[comp] = _fpr_per_cond(
            gen_sc, real_sc, calib_meta,
            kid_gen_np, kid_real_np,
            gen_by_cond, kid_real_by_cond, real_score_by_cond,
            condition_keys, pct=95, seed=seed,
            calib_fallback=calib_fallback)
        rows = list(percond_by_comp[comp].values())
        vals = [r["delta"] for r in rows if np.isfinite(r["delta"])]
        n_better = sum(1 for v in vals if v > 0)
        mean_d = np.mean(vals) if vals else np.nan
        # w-ΔKID%: sample-weighted mean of per-cond improvement_pct (weight = n_accepted)
        pair = [(r["improvement_pct"], r["n_accepted"]) for r in rows
                if np.isfinite(r.get("improvement_pct", np.nan))]
        total_acc = sum(w for _, w in pair)
        w_imp = (sum(p * w for p, w in pair) / total_acc) if total_acc > 0 else float("nan")
        mean_imp = float(np.mean([p for p, _ in pair])) if pair else float("nan")
        percond_summary[comp] = dict(
            n_conds=len(rows),
            improved=n_better,
            mean_delta=float(mean_d) if np.isfinite(mean_d) else float("nan"),
            mean_improvement_pct=mean_imp,
            w_improvement_pct=float(w_imp) if np.isfinite(w_imp) else float("nan"),
            total_accepted=int(total_acc),
        )
        print(f"      {comp:15s}: {n_better}/{len(vals)} conds improved, "
              f"mean D={mean_d:+.5f}, w-ΔKID%={w_imp:+.2f}")
    results["per_condition_fpr95"] = percond_by_comp
    results["per_condition_fpr95_summary"] = percond_summary

    # Scatter plot for per-condition P95
    scatter_path = _plot_percond_scatter(percond_by_comp, model, output_dir, config_key, dataset, pct=95)
    print(f"    -> {scatter_path}")

    # --- Build per-condition breakdown table (flat, for CSV) ---
    per_cond_rows = []
    for comp in COMPONENTS:
        for cond_str, r in sorted(percond_by_comp.get(comp, {}).items()):
            per_cond_rows.append({
                "component": comp,
                "condition": cond_str,
                "calib_cond": r.get("calib_cond", cond_str),
                "calib_borrowed": r.get("calib_borrowed", False),
                "n_calib_used": r.get("n_calib_used", 1),
                "n_accepted": r["n_accepted"],
                "n_total": r["n_total"],
                "acceptance_rate": r["acceptance_rate"],
                "kid_accepted": r["kid_acc"][0],
                "kid_accepted_std": r["kid_acc"][1],
                "kid_random": r["kid_rand"][0],
                "kid_random_std": r["kid_rand"][1],
                "delta": r["delta"],
                "ci_lower": r["ci_lower"],
                "ci_upper": r["ci_upper"],
                "improvement_pct": r["improvement_pct"],
            })
    results["per_condition_breakdown"] = per_cond_rows

    # Save CSVs
    safe_key = config_key.replace("/", "_")
    csv_path = output_dir / f"{dataset}_fpr95_selection_{safe_key}.csv"
    summary_df = pd.DataFrame([{k: v for k, v in results.items()
                                 if k not in ("per_condition_breakdown", "per_condition_fpr95",
                                              "per_condition_fpr95_summary",
                                              "fpr_global_multithreshold")}])
    summary_df.to_csv(csv_path, index=False)

    per_cond_path = output_dir / f"{dataset}_fpr95_per_condition_{safe_key}.csv"
    if per_cond_rows:
        pd.DataFrame(per_cond_rows).to_csv(per_cond_path, index=False)
        print(f"    Per-condition CSV: {per_cond_path}")

    # Summary print
    print(
        f"    FPR@95 acceptance rate: {results['acceptance_rate']:.2%} "
        f"({results['n_accepted']}/{n_gen}), "
        f"KID subsample={results['kid_sample_size']}"
    )
    if results["kid_raw_accepted"] is not None:
        acc_std = f" +/- {results['kid_raw_accepted_std']:.4f}" if results.get("kid_raw_accepted_std") else ""
        print(f"    KID (accepted): {results['kid_raw_accepted']:.4f}{acc_std}")
    if results["kid_raw_random"] is not None:
        std_str = f" +/- {results['kid_raw_random_std']:.4f}" if results.get("kid_raw_random_std") else ""
        print(f"    KID (random baseline): {results['kid_raw_random']:.4f}{std_str}")

    return results
