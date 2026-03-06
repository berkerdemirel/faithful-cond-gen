"""
KID (Kernel Inception Distance) computation utilities.

This module consolidates KID computation from both run_trust_evaluation.py
and trust_eval_extensions.py. The calculate_kid_same_m function preserves
exact behavior from trust_eval_extensions.py (which has kid_mode parameter).
"""

from typing import Dict, Tuple

import numpy as np


def calculate_kid_same_m(
    X: np.ndarray,
    Y: np.ndarray,
    use_cosine: bool = False,
    kid_mode: str = None,
) -> float:
    """
    Unbiased MMD^2 estimate with polynomial kernel.

    Args:
        X, Y: Feature arrays (same size)
        use_cosine: If True, L2-normalize features and use k(x,y)=(x·y + 1)^3
                   If False, use standard k(x,y)=(x·y/d + 1)^3
                   This parameter is used when kid_mode is None.
        kid_mode: Optional mode override. One of:
                  - "auto": use_cosine determined by caller (uses use_cosine param)
                  - "standard": force use_cosine=False
                  - "cosine": force use_cosine=True
                  - None: use the use_cosine parameter directly

    Note:
        The kid_mode parameter takes precedence over use_cosine when specified
        (except for "auto" which defers to use_cosine).

    Returns:
        Unbiased MMD^2 estimate (float). Returns np.nan if m < 2.
    """
    # Resolve effective use_cosine based on kid_mode
    if kid_mode == "cosine":
        use_cosine = True
    elif kid_mode == "standard":
        use_cosine = False
    # For "auto" or None, use the use_cosine parameter as-is

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"KID expects equal sample sizes, got {X.shape[0]} vs {Y.shape[0]}"
        )

    m = X.shape[0]
    if m < 2:
        return np.nan

    if use_cosine:
        # L2-normalize features for cosine-based kernel
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
        # Cosine kernel: k(x,y) = (x·y + 1)^3
        Kxx = (X @ X.T + 1.0) ** 3
        Kyy = (Y @ Y.T + 1.0) ** 3
        Kxy = (X @ Y.T + 1.0) ** 3
    else:
        # Standard kernel: k(x,y) = (x·y/d + 1)^3
        dim = X.shape[1]
        Kxx = (X @ X.T / dim + 1.0) ** 3
        Kyy = (Y @ Y.T / dim + 1.0) ** 3
        Kxy = (X @ Y.T / dim + 1.0) ** 3

    kxx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    kyy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    kxy = Kxy.sum() / (m * m)

    return float(kxx + kyy - 2.0 * kxy)


def bootstrap_kid_for_bin(
    gen_feats_bin: np.ndarray,
    real_feats: np.ndarray,
    n_bootstrap: int = 10,
    k: int = None,
    use_cosine: bool = False,
) -> Dict:
    """
    Bootstrap KID computation for a bin of generated features.

    Args:
        gen_feats_bin: Generated features in this bin (M, D)
        real_feats: All real features (N, D)
        n_bootstrap: Number of bootstrap iterations
        k: Sample size for KID computation (if None, use min(M, N//2, 500))
        use_cosine: If True, use cosine-based kernel (for aligned_mean features)

    Returns:
        Dict with mean_kid, ci_low, ci_high
    """
    if k is None:
        k = min(len(gen_feats_bin), len(real_feats) // 2, 500)

    k = max(k, 5)  # Minimum 5 samples

    if len(gen_feats_bin) < k or len(real_feats) < k:
        return {"mean_kid": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    kids = []
    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        # Resample both gen and real
        gen_idx = rng.choice(len(gen_feats_bin), k, replace=len(gen_feats_bin) < k)
        real_idx = rng.choice(len(real_feats), k, replace=False)

        gen_sample = gen_feats_bin[gen_idx]
        real_sample = real_feats[real_idx]

        try:
            kid = calculate_kid_same_m(real_sample, gen_sample, use_cosine=use_cosine)
            if np.isfinite(kid):
                kids.append(kid)
        except Exception:
            continue

    if len(kids) == 0:
        return {"mean_kid": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    mean_kid = np.mean(kids)
    if len(kids) > 1:
        ci_low, ci_high = np.percentile(kids, [2.5, 97.5])
    else:
        ci_low = ci_high = mean_kid

    return {
        "mean_kid": float(mean_kid),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def estimate_kid_null_per_condition(
    real_feats_by_cond: Dict[Tuple, np.ndarray],
    n_resamples: int = 100,
    use_cosine: bool = False,
    seed: int = 42,
    max_samples_per_split: int = 500,
) -> Dict[Tuple, Tuple[float, float]]:
    """
    Estimate per-condition KID null distribution via real-real resampling.

    For each condition, repeatedly split real samples into two halves and
    compute KID between them. The resulting distribution provides (μ_c, σ_c)
    for z-scoring: KID_z(c) = (KID(G_c, R_c) - μ_c) / σ_c

    Args:
        real_feats_by_cond: Dict mapping condition tuple to real features array (N_c, D)
        n_resamples: Number of bootstrap resamples per condition
        use_cosine: If True, use cosine-based kernel
        seed: Random seed for reproducibility
        max_samples_per_split: Maximum samples per KID split (caps kernel size for speed)

    Returns:
        Dict mapping condition tuple to (mu_c, sigma_c) for z-scoring.
        Conditions with < 10 samples or degenerate std are excluded.
    """
    rng = np.random.default_rng(seed)
    null_stats = {}

    for cond, real_feats in real_feats_by_cond.items():
        n_c = len(real_feats)
        if n_c < 10:
            # Too few samples for reliable null estimation
            continue

        # Cap the sample size for efficiency (KID kernel is O(n²))
        half_size = min(n_c // 2, max_samples_per_split)
        kids = []

        for _ in range(n_resamples):
            # Random split - sample half_size from each half
            perm = rng.permutation(n_c)
            idx_a = perm[:half_size]
            idx_b = perm[n_c // 2 : n_c // 2 + half_size]
            real_a = real_feats[idx_a]
            real_b = real_feats[idx_b]

            kid = calculate_kid_same_m(real_a, real_b, use_cosine=use_cosine)
            if np.isfinite(kid):
                kids.append(kid)

        if len(kids) < 10:
            # Not enough valid resamples
            continue

        mu_c = np.mean(kids)
        sigma_c = np.std(kids, ddof=1)

        if sigma_c < 1e-10:
            # Degenerate (constant) distribution
            continue

        null_stats[cond] = (float(mu_c), float(sigma_c))

    return null_stats
