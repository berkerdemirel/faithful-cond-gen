"""
Task 4: FPR@95 Selection + z-KID Evaluation.

Evaluate quality of generated samples that pass at FPR@95 threshold.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from faithful_cond_gen.eval.trust_eval.metrics_kid import (
    calculate_kid_same_m,
    estimate_kid_null_per_condition,
)
from faithful_cond_gen.eval.trust_eval.scoring_core import (
    compute_real_sample_scores,
    fit_trust_scoring_components,
)


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
    kid_mode: str = "auto",
    feature_type: str = "dinov3",
    scoring_method: str = "mahalanobis",
    use_kid_z: bool = False,
    seed: int = 42,
    n_random_repeats: int = 5,
) -> Dict:
    """
    Evaluate quality of generated samples that pass at FPR@95.

    1. Compute threshold t = 95th percentile of real scores (in scoring space)
    2. Build G_accept = {g : score(g) <= t}
    3. Build R_match with same condition histogram
    4. Compute KID(G_accept, R_match) in KID space - raw or z-normalized
    5. Compare to random baseline (repeated n_random_repeats times for CIs)

    Args:
        trust_results: Dict with trust scores for generated samples
        real_feats: Real features in scoring space (for threshold calibration)
        real_meta: Real metadata dict
        gen_feats: Generated features in scoring space (unused, kept for API compat)
        gen_meta: Generated metadata dict
        condition_keys: List of condition attribute names
        dataset: Dataset name
        model: Model name
        output_dir: Output directory for results
        config_key: Configuration key for naming
        kid_real_feats: Real features for KID computation (defaults to real_feats)
        kid_gen_feats: Generated features for KID computation (defaults to gen_feats)
        kid_mode: KID computation mode
        feature_type: Feature type for KID mode determination
        scoring_method: Scoring method used ("mahalanobis" or "knn")
        use_kid_z: If True, report z-normalized KID. If False (default), report raw KID.
        seed: Random seed
        n_random_repeats: Number of random baseline draws for confidence intervals

    Returns:
        Dict with FPR@95 selection results
    """
    rng = np.random.default_rng(seed)

    # Fall back to scoring features if separate KID features not provided
    if kid_real_feats is None:
        kid_real_feats = real_feats
    if kid_gen_feats is None:
        kid_gen_feats = gen_feats

    # Get generated sample scores
    gen_scores = trust_results["trust_updated"]
    gen_conditions = trust_results["true_conditions"]
    n_gen = len(gen_scores)

    # Compute scores for real samples using same components
    # Need to fit components first, then score real samples
    components = fit_trust_scoring_components(
        real_feats, real_meta, condition_keys,
        scoring_method=scoring_method,
    )

    from faithful_cond_gen.eval.trust_eval.scoring_core import score_trust_from_components
    real_realism, real_faithfulness, real_scores = score_trust_from_components(
        real_feats, real_meta, components
    )

    # Compute threshold at 95th percentile of real scores
    valid_real = np.isfinite(real_scores)
    if valid_real.sum() < 10:
        return {"status": "insufficient_real_samples", "n_valid_real": int(valid_real.sum())}

    t_95 = float(np.percentile(real_scores[valid_real], 95))

    # Build G_accept: generated samples with score <= t_95
    valid_gen = np.isfinite(gen_scores)
    accept_mask = valid_gen & (gen_scores <= t_95)
    n_accepted = int(accept_mask.sum())

    if n_accepted < 10:
        return {
            "status": "insufficient_accepted",
            "threshold_95": t_95,
            "n_accepted": n_accepted,
            "acceptance_rate": n_accepted / n_gen,
        }

    accept_indices = np.where(accept_mask)[0]
    accept_conditions = [gen_conditions[i] for i in accept_indices]

    # Build condition histogram for accepted samples
    cond_hist: Dict[Tuple, int] = {}
    for cond in accept_conditions:
        cond_hist[cond] = cond_hist.get(cond, 0) + 1

    # Group real samples by condition
    real_by_cond: Dict[Tuple, List[int]] = {}
    n_real = len(real_feats)
    for i in range(n_real):
        cond = tuple(
            int(
                real_meta[k][i].item()
                if isinstance(real_meta[k][i], torch.Tensor)
                else real_meta[k][i]
            )
            for k in condition_keys
        )
        real_by_cond.setdefault(cond, []).append(i)

    # Build R_match with same condition histogram
    match_real_indices = []
    for cond, count in cond_hist.items():
        real_cond_indices = real_by_cond.get(cond, [])
        if len(real_cond_indices) >= count:
            sampled = rng.choice(real_cond_indices, size=count, replace=False)
        else:
            # Not enough real samples, sample with replacement
            sampled = rng.choice(real_cond_indices, size=count, replace=True) if real_cond_indices else []
        match_real_indices.extend(sampled)

    match_real_indices = np.array(match_real_indices, dtype=int)

    # Determine effective cosine mode for KID
    if kid_mode == "auto":
        use_cosine = feature_type == "aligned_mean"
    elif kid_mode == "cosine":
        use_cosine = True
    else:
        use_cosine = False

    # Use KID features (DINO space) for distributional quality measurement
    kid_real_np = kid_real_feats.numpy() if isinstance(kid_real_feats, torch.Tensor) else kid_real_feats
    kid_gen_np = kid_gen_feats.numpy() if isinstance(kid_gen_feats, torch.Tensor) else kid_gen_feats

    # Estimate null distribution for z-KID (only if use_kid_z=True)
    kid_null_stats = {}
    agg_mu, agg_sigma = 0.0, 1.0
    if use_kid_z:
        real_feats_by_cond = {cond: kid_real_np[idx] for cond, idx in real_by_cond.items()}
        kid_null_stats = estimate_kid_null_per_condition(
            real_feats_by_cond, n_resamples=100, use_cosine=use_cosine, seed=seed
        )
        if kid_null_stats:
            null_mus = [s[0] for s in kid_null_stats.values()]
            null_sigmas = [s[1] for s in kid_null_stats.values()]
            agg_mu = np.mean(null_mus)
            agg_sigma = np.mean(null_sigmas)

    # Compute KID for accepted samples (in KID feature space)
    gen_accept = kid_gen_np[accept_indices]
    real_match = kid_real_np[match_real_indices]

    # Compute raw KID
    k = min(len(gen_accept), len(real_match), 500)
    if k >= 10:
        kid_accept = calculate_kid_same_m(
            gen_accept[:k], real_match[:k], use_cosine=use_cosine
        )
    else:
        kid_accept = np.nan

    # Compute z-KID using aggregated null (only if use_kid_z=True)
    kid_z_accept = np.nan
    if use_kid_z and kid_null_stats and np.isfinite(kid_accept) and agg_sigma > 1e-10:
        kid_z_accept = (kid_accept - agg_mu) / agg_sigma

    # Random baseline: repeat n_random_repeats times for confidence intervals
    gen_by_cond: Dict[Tuple, List[int]] = {}
    for i in range(n_gen):
        gen_by_cond.setdefault(gen_conditions[i], []).append(i)

    kid_random_values = []
    kid_z_random_values = []
    for rep in range(n_random_repeats):
        rep_rng = np.random.default_rng(seed + rep + 1)
        random_indices = []
        for cond, count in cond_hist.items():
            gen_cond_indices = gen_by_cond.get(cond, [])
            if len(gen_cond_indices) >= count:
                sampled = rep_rng.choice(gen_cond_indices, size=count, replace=False)
            else:
                sampled = list(gen_cond_indices) if gen_cond_indices else []
            random_indices.extend(sampled)

        random_indices = np.array(random_indices, dtype=int)
        gen_random = kid_gen_np[random_indices] if len(random_indices) > 0 else np.array([])

        if len(gen_random) >= 10 and len(real_match) >= 10:
            k_rand = min(len(gen_random), len(real_match), 500)
            kid_rand = calculate_kid_same_m(
                gen_random[:k_rand], real_match[:k_rand], use_cosine=use_cosine
            )
            if np.isfinite(kid_rand):
                kid_random_values.append(kid_rand)
                if use_kid_z and kid_null_stats and agg_sigma > 1e-10:
                    kid_z_random_values.append((kid_rand - agg_mu) / agg_sigma)

    kid_random = float(np.mean(kid_random_values)) if kid_random_values else np.nan
    kid_random_std = float(np.std(kid_random_values)) if len(kid_random_values) > 1 else np.nan
    kid_z_random = float(np.mean(kid_z_random_values)) if kid_z_random_values else np.nan
    kid_z_random_std = float(np.std(kid_z_random_values)) if len(kid_z_random_values) > 1 else np.nan

    results = {
        "status": "success",
        "threshold_95": t_95,
        "n_accepted": n_accepted,
        "acceptance_rate": n_accepted / n_gen,
        "n_conditions_in_accepted": len(cond_hist),
        "kid_raw_accepted": float(kid_accept) if np.isfinite(kid_accept) else None,
        "kid_z_accepted": float(kid_z_accept) if np.isfinite(kid_z_accept) else None,
        "kid_raw_random": float(kid_random) if np.isfinite(kid_random) else None,
        "kid_raw_random_std": float(kid_random_std) if np.isfinite(kid_random_std) else None,
        "kid_z_random": float(kid_z_random) if np.isfinite(kid_z_random) else None,
        "kid_z_random_std": float(kid_z_random_std) if np.isfinite(kid_z_random_std) else None,
        "n_random_repeats": n_random_repeats,
        "null_mu_agg": float(agg_mu) if kid_null_stats else None,
        "null_sigma_agg": float(agg_sigma) if kid_null_stats else None,
        "n_conditions_with_null": len(kid_null_stats),
    }

    # Per-condition breakdown
    per_cond_results = []
    for cond, count in cond_hist.items():
        per_cond_results.append({
            "condition": str(cond),
            "n_accepted": count,
            "n_real_available": len(real_by_cond.get(cond, [])),
        })
    results["per_condition_breakdown"] = per_cond_results

    # Save results
    csv_path = output_dir / f"{dataset}_fpr95_selection_{config_key.replace('/', '_')}.csv"
    df = pd.DataFrame([{k: v for k, v in results.items() if k != "per_condition_breakdown"}])
    df.to_csv(csv_path, index=False)

    # Save per-condition breakdown
    per_cond_path = output_dir / f"{dataset}_fpr95_per_condition_{config_key.replace('/', '_')}.csv"
    pd.DataFrame(per_cond_results).to_csv(per_cond_path, index=False)

    print(f"    FPR@95 acceptance rate: {results['acceptance_rate']:.2%} ({n_accepted}/{n_gen})")
    if use_kid_z:
        if results["kid_z_accepted"] is not None:
            print(f"    z-KID (accepted): {results['kid_z_accepted']:.4f}")
        if results["kid_z_random"] is not None:
            std_str = f" ± {results['kid_z_random_std']:.4f}" if results.get("kid_z_random_std") else ""
            print(f"    z-KID (random baseline): {results['kid_z_random']:.4f}{std_str} (n={n_random_repeats})")
    else:
        if results["kid_raw_accepted"] is not None:
            print(f"    KID (accepted): {results['kid_raw_accepted']:.4f}")
        if results["kid_raw_random"] is not None:
            std_str = f" ± {results['kid_raw_random_std']:.4f}" if results.get("kid_raw_random_std") else ""
            print(f"    KID (random baseline): {results['kid_raw_random']:.4f}{std_str} (n={n_random_repeats})")

    return results
