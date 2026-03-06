"""
Core scoring functions for trust evaluation.

Contains Mahalanobis and kNN-based scoring, calibration, and the main
compute_trust_results_from_features function.
"""

import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

from faithful_cond_gen.eval.trust_eval.condition_utils import (
    filter_feats_and_meta_by_seen_combos,
    get_condition_key,
)
from faithful_cond_gen.eval.trust_eval.config import (
    MARGINAL_SEEN_COMBOS,
    MIXTURE_COMPONENTS,
)


# ============================================================================
# Feature Normalization
# ============================================================================


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """L2 normalize features."""
    return features / (features.norm(dim=1, keepdim=True) + 1e-12)


def compute_mahalanobis(
    x: torch.Tensor, mu: torch.Tensor, precision: torch.Tensor
) -> torch.Tensor:
    """Compute Mahalanobis distance."""
    centered = x - mu.unsqueeze(0)
    term = torch.einsum("bd,de->be", centered, precision)
    dist = torch.sum(term * centered, dim=1)
    return dist


def zscore(x: np.ndarray, mean: float, std: float, eps: float = 1e-12) -> np.ndarray:
    """Z-score normalization."""
    return (x - mean) / (std + eps)


# ============================================================================
# Global Stats Fitting (Mahalanobis)
# ============================================================================


def fit_global_stats(
    features: torch.Tensor,
    regularization: float = 1e-5,
    min_samples: int = 2,
) -> Dict:
    """Fit global Gaussian (mu, precision) on features."""
    features = normalize_features(features)
    N, D = features.shape

    mu = features.mean(dim=0)

    if N >= min_samples:
        feats_np = features.numpy()
        lw = LedoitWolf()
        try:
            cov_np = lw.fit(feats_np).covariance_
            cov = torch.from_numpy(cov_np).float()
            shrinkage = float(lw.shrinkage_)
        except Exception:
            cov = torch.eye(D)
            shrinkage = 1.0
    else:
        cov = torch.eye(D)
        shrinkage = 1.0

    cov_reg = cov + regularization * torch.eye(D)
    try:
        L = torch.linalg.cholesky(cov_reg)
        precision = torch.cholesky_inverse(L)
    except RuntimeError:
        precision = torch.linalg.pinv(cov_reg)

    return {
        "mu": mu,
        "precision": precision,
        "n_samples": int(N),
        "shrinkage": shrinkage,
    }


def fit_factorized_stats(
    features: torch.Tensor,
    metadata: Dict,
    condition_keys: List[str],
    regularization: float = 1e-5,
    min_samples: int = 2,
    use_shared_cov: bool = True,
) -> Dict[str, Dict[int, Dict]]:
    """
    Fit per-attribute Gaussians with LDA-style pooled within-class covariance.

    For each attribute, computes:
    - Per-value means (μ_v)
    - Shared within-class covariance (pooled across all values)

    This ensures Mahalanobis distances are on the same scale across attribute
    values, making margin computation (E_true - E_other) fair.

    Args:
        features: L2-normalized features (N, D)
        metadata: Dict with attribute values
        condition_keys: List of attribute names
        regularization: Regularization for covariance
        min_samples: Minimum samples per class
        use_shared_cov: If True (default), use LDA-style pooled covariance.
                       If False, use per-class covariance (legacy behavior).
    """
    features = normalize_features(features)
    N, D = features.shape

    # Group samples by attribute value
    attr_groups = {k: {} for k in condition_keys}
    for i in range(N):
        for k in condition_keys:
            val = int(
                metadata[k][i].item()
                if isinstance(metadata[k][i], torch.Tensor)
                else metadata[k][i]
            )
            attr_groups[k].setdefault(val, []).append(i)

    stats = {}
    for attr_key in condition_keys:
        stats[attr_key] = {}
        values = list(attr_groups[attr_key].keys())

        if use_shared_cov and len(values) >= 2:
            # LDA-style: Compute pooled within-class covariance
            # S_w = (1/N) * Σ_c Σ_{i in c} (x_i - μ_c)(x_i - μ_c)^T
            pooled_cov = torch.zeros(D, D)
            total_samples = 0

            # First pass: compute per-class means
            class_means = {}
            for val in values:
                idx = attr_groups[attr_key][val]
                if len(idx) >= min_samples:
                    class_means[val] = features[idx].mean(dim=0)

            # Second pass: accumulate within-class scatter
            for val in values:
                idx = attr_groups[attr_key][val]
                if val not in class_means:
                    continue
                feats = features[idx]
                mu = class_means[val]
                centered = feats - mu.unsqueeze(0)
                # Accumulate scatter matrix
                pooled_cov += centered.T @ centered
                total_samples += len(idx)

            if total_samples > D:
                # Normalize by total samples
                pooled_cov = pooled_cov / total_samples

                # Apply Ledoit-Wolf shrinkage to pooled covariance
                try:
                    lw = LedoitWolf()
                    # Fit LW on residuals to get shrinkage estimate
                    all_residuals = []
                    for val in values:
                        if val not in class_means:
                            continue
                        idx = attr_groups[attr_key][val]
                        feats = features[idx]
                        mu = class_means[val]
                        all_residuals.append((feats - mu.unsqueeze(0)).numpy())
                    all_residuals = np.concatenate(all_residuals, axis=0)
                    lw.fit(all_residuals)
                    shrinkage = lw.shrinkage_
                    # Apply shrinkage: (1-α)*S + α*tr(S)/d*I
                    trace_div_d = pooled_cov.trace() / D
                    pooled_cov = (
                        1 - shrinkage
                    ) * pooled_cov + shrinkage * trace_div_d * torch.eye(D)
                except Exception:
                    pass  # Use unshrunken pooled cov

                # Regularize and invert
                cov_reg = pooled_cov + regularization * torch.eye(D)
                try:
                    L = torch.linalg.cholesky(cov_reg)
                    shared_precision = torch.cholesky_inverse(L)
                except RuntimeError:
                    shared_precision = torch.linalg.pinv(cov_reg)
            else:
                shared_precision = torch.eye(D) / (regularization + 1e-3)

            # Store stats with shared precision
            for val in values:
                idx = attr_groups[attr_key][val]
                n = len(idx)
                mu = features[idx].mean(dim=0) if n > 0 else torch.zeros(D)
                stats[attr_key][val] = {
                    "mu": mu,
                    "precision": shared_precision,
                    "n_samples": n,
                    "shared_cov": True,
                }
        else:
            # Legacy: per-class covariance (not recommended for margin computation)
            for val in values:
                idx = attr_groups[attr_key][val]
                feats = features[idx]
                n = len(idx)
                mu = feats.mean(dim=0)

                if n >= min_samples:
                    lw = LedoitWolf()
                    try:
                        cov_np = lw.fit(feats.numpy()).covariance_
                        cov = torch.from_numpy(cov_np).float()
                    except Exception:
                        cov = torch.eye(D)
                    cov_reg = cov + regularization * torch.eye(D)
                    try:
                        L = torch.linalg.cholesky(cov_reg)
                        precision = torch.cholesky_inverse(L)
                    except RuntimeError:
                        precision = torch.linalg.pinv(cov_reg)
                else:
                    precision = torch.eye(D) / (regularization + 1e-3)

                stats[attr_key][val] = {
                    "mu": mu,
                    "precision": precision,
                    "n_samples": n,
                    "shared_cov": False,
                }

    return stats


# ============================================================================
# Mixture Mahalanobis (for per-component realism)
# ============================================================================


def get_mixture_component_key(
    dataset: str,
    model: str,
    metadata: Dict,
    condition_keys: List[str],
    sample_idx: int,
) -> Tuple:
    """
    Get the mixture component key for a sample based on dataset/model config.

    For CelebA full: full condition tuple (16 combos)
    For CelebA marginal: condition tuple if in MARGINAL_SEEN_COMBOS (5 combos)
    For RxRx1: cell_type_id only (4 cell types)
    """
    model_type = "marginal" if "marginal" in model else "full"
    config_key = (dataset, model_type)
    component_strategy = MIXTURE_COMPONENTS.get(config_key, "condition")

    if component_strategy == "cell_type":
        # RxRx1: group by cell type only
        cell_type = int(
            metadata["cell_type_id"][sample_idx].item()
            if isinstance(metadata["cell_type_id"][sample_idx], torch.Tensor)
            else metadata["cell_type_id"][sample_idx]
        )
        return (cell_type,)
    elif component_strategy == "seen_condition":
        # CelebA marginal: only 5 seen combos
        cond = tuple(
            int(
                metadata[k][sample_idx].item()
                if isinstance(metadata[k][sample_idx], torch.Tensor)
                else metadata[k][sample_idx]
            )
            for k in condition_keys
        )
        return cond if cond in MARGINAL_SEEN_COMBOS else None
    else:
        # Default: full condition tuple
        return tuple(
            int(
                metadata[k][sample_idx].item()
                if isinstance(metadata[k][sample_idx], torch.Tensor)
                else metadata[k][sample_idx]
            )
            for k in condition_keys
        )


def fit_mixture_stats(
    features: torch.Tensor,
    metadata: Dict,
    condition_keys: List[str],
    dataset: str,
    model: str,
    regularization: float = 1e-5,
    min_samples_per_component: int = 10,
) -> Dict:
    """
    Fit mixture-of-Gaussians with shared (LDA-style pooled) covariance for realism.

    Args:
        features: L2-normalized features (N, D)
        metadata: Metadata dict with condition keys
        condition_keys: List of attribute names
        dataset: Dataset name ("celeba" or "rxrx1")
        model: Model name (used to determine component grouping)
        regularization: Regularization for covariance
        min_samples_per_component: Minimum samples to include a component

    Returns:
        Dict with:
        - "mus": Dict[component_key -> mean vector (D,)]
        - "precision": shared precision matrix (D, D)
        - "component_keys": list of valid component keys
        - "n_components": number of components
        - "n_samples_per_component": Dict[component_key -> n_samples]
    """
    features = normalize_features(features)
    N, D = features.shape

    # Group samples by mixture component
    component_groups: Dict[Tuple, List[int]] = {}
    for i in range(N):
        comp_key = get_mixture_component_key(dataset, model, metadata, condition_keys, i)
        if comp_key is not None:  # None means not in seen combos for marginal
            component_groups.setdefault(comp_key, []).append(i)

    # Filter to components with enough samples
    valid_components = {
        k: v for k, v in component_groups.items() if len(v) >= min_samples_per_component
    }

    if len(valid_components) < 1:
        # Fallback to global Gaussian
        return fit_global_stats(features, regularization)

    # Compute per-component means
    mus = {}
    for comp_key, idx_list in valid_components.items():
        mus[comp_key] = features[idx_list].mean(dim=0)

    # Compute pooled within-component covariance (LDA-style)
    pooled_cov = torch.zeros(D, D)
    total_samples = 0

    for comp_key, idx_list in valid_components.items():
        feats = features[idx_list]
        mu = mus[comp_key]
        centered = feats - mu.unsqueeze(0)
        pooled_cov += centered.T @ centered
        total_samples += len(idx_list)

    if total_samples > D:
        pooled_cov = pooled_cov / total_samples

        # Apply Ledoit-Wolf shrinkage
        try:
            all_residuals = []
            for comp_key, idx_list in valid_components.items():
                feats = features[idx_list]
                mu = mus[comp_key]
                all_residuals.append((feats - mu.unsqueeze(0)).numpy())
            all_residuals = np.concatenate(all_residuals, axis=0)
            lw = LedoitWolf()
            lw.fit(all_residuals)
            shrinkage = lw.shrinkage_
            trace_div_d = pooled_cov.trace() / D
            pooled_cov = (1 - shrinkage) * pooled_cov + shrinkage * trace_div_d * torch.eye(D)
        except Exception:
            pass  # Use unshrunken pooled cov

        # Regularize and invert
        cov_reg = pooled_cov + regularization * torch.eye(D)
        try:
            L = torch.linalg.cholesky(cov_reg)
            precision = torch.cholesky_inverse(L)
        except RuntimeError:
            precision = torch.linalg.pinv(cov_reg)
    else:
        precision = torch.eye(D) / (regularization + 1e-3)

    return {
        "mus": mus,
        "precision": precision,
        "component_keys": list(valid_components.keys()),
        "n_components": len(valid_components),
        "n_samples_per_component": {k: len(v) for k, v in valid_components.items()},
    }


def compute_real_calibration_for_mixture_energy(
    real_feats: torch.Tensor,
    real_meta: Dict,
    condition_keys: List[str],
    mixture_stats: Dict,
    dataset: str,
    model: str,
    batch_size: int = 2000,
) -> Tuple[float, float]:
    """
    Compute mean/std of mixture Mahalanobis energy on real samples.

    For each real sample, computes min_k (x - mu_k)^T P (x - mu_k) and returns
    calibration stats (mean, std) for z-scoring.
    """
    real_feats = normalize_features(real_feats)
    N = len(real_feats)

    mus = mixture_stats["mus"]
    P = mixture_stats["precision"]
    component_keys = mixture_stats["component_keys"]
    K = len(component_keys)

    if K == 0:
        return 0.0, 1.0

    # Stack means for efficient computation
    mus_stacked = torch.stack([mus[k] for k in component_keys], dim=0)  # (K, D)

    energies = np.zeros(N, dtype=np.float64)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x = real_feats[start:end]  # (B, D)

        # Compute distance to each component
        centered = x.unsqueeze(1) - mus_stacked.unsqueeze(0)  # (B, K, D)
        term = torch.einsum("bkd,de->bke", centered, P)  # (B, K, D)
        dists = torch.sum(term * centered, dim=2).numpy()  # (B, K)

        # Take minimum across components (hard assignment)
        energies[start:end] = dists.min(axis=1)

    mean = float(np.mean(energies)) if N > 0 else 0.0
    std = float(np.std(energies, ddof=1) if N > 1 else 1.0)
    return mean, std


def compute_mixture_realism_z(
    feats: torch.Tensor,
    metadata: Dict,
    condition_keys: List[str],
    mixture_stats: Dict,
    real_E_mean: float,
    real_E_std: float,
    dataset: str,
    model: str,
    batch_size: int = 2000,
    two_sided: bool = True,
    assignment: str = "hard",
) -> np.ndarray:
    """
    Compute mixture realism z-scores.

    e_k(x) = (x - μ_k)^T P (x - μ_k)
    e(x) = min_k e_k(x)  [hard assignment]
    z = (e - mean_E) / std_E

    Args:
        feats: Features to score (N, D)
        metadata: Metadata dict
        condition_keys: List of attribute names
        mixture_stats: Dict from fit_mixture_stats
        real_E_mean, real_E_std: Calibration from real samples
        dataset, model: For component key computation
        two_sided: If True (default), return z² for two-sided detection
        assignment: "hard" (min) or "soft" (log-sum-exp, not implemented)

    Returns:
        Array of z-scores (N,)
    """
    feats = normalize_features(feats)
    N = len(feats)

    mus = mixture_stats["mus"]
    P = mixture_stats["precision"]
    component_keys = mixture_stats["component_keys"]
    K = len(component_keys)

    if K == 0:
        return np.zeros(N, dtype=np.float64)

    # Stack means
    mus_stacked = torch.stack([mus[k] for k in component_keys], dim=0)  # (K, D)

    out = np.zeros(N, dtype=np.float64)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x = feats[start:end]  # (B, D)

        # Distance to each component
        centered = x.unsqueeze(1) - mus_stacked.unsqueeze(0)  # (B, K, D)
        term = torch.einsum("bkd,de->bke", centered, P)  # (B, K, D)
        dists = torch.sum(term * centered, dim=2).numpy()  # (B, K)

        # Hard assignment: min across components
        e = dists.min(axis=1)
        z = zscore(e, real_E_mean, real_E_std)

        if two_sided:
            out[start:end] = z**2
        else:
            out[start:end] = z

    return out


# ============================================================================
# Real Calibration
# ============================================================================


def compute_real_calibration_for_global_energy(
    real_feats: torch.Tensor,
    global_stats: Dict[str, Any],
    batch_size: int = 2000,
) -> Tuple[float, float]:
    """Compute mean/std of global Mahalanobis energy on real samples."""
    real_feats = normalize_features(real_feats)
    N = len(real_feats)

    mu = global_stats["mu"]
    P = global_stats["precision"]

    energies = np.zeros(N, dtype=np.float64)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        e = compute_mahalanobis(real_feats[start:end], mu, P).numpy()
        energies[start:end] = e.astype(np.float64)

    mean = float(np.mean(energies)) if N > 0 else 0.0
    std = float(np.std(energies, ddof=1) if N > 1 else 1.0)
    return mean, std


def compute_real_calibration_for_factorized_margins(
    real_feats: torch.Tensor,
    real_meta: Dict,
    factorized_stats: Dict[str, Dict[int, Dict[str, Any]]],
    condition_keys: List[str],
    batch_size: int = 2000,
) -> Dict[str, Tuple[float, float]]:
    """
    For each attribute k, compute mean/std of real margin:
      m_k(x) = E_k(x; true) - min_{v!=true} E_k(x; v)
    over real samples. Used for z-scoring margins.
    """
    real_feats = normalize_features(real_feats)
    N = len(real_feats)

    calib: Dict[str, Tuple[float, float]] = {}
    for attr_key in condition_keys:
        values = sorted(factorized_stats[attr_key].keys())
        V = len(values)
        mus = torch.stack(
            [factorized_stats[attr_key][v]["mu"] for v in values]
        )  # (V, D)
        Ps = torch.stack(
            [factorized_stats[attr_key][v]["precision"] for v in values]
        )  # (V, D, D)

        true_idx = np.zeros(N, dtype=int)
        for i in range(N):
            val = int(
                real_meta[attr_key][i].item()
                if isinstance(real_meta[attr_key][i], torch.Tensor)
                else real_meta[attr_key][i]
            )
            true_idx[i] = values.index(val) if val in values else -1

        margins: List[float] = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x = real_feats[start:end]  # (B, D)

            centered = x.unsqueeze(1) - mus.unsqueeze(0)  # (B, V, D)
            term1 = torch.einsum("bvd,vde->bve", centered, Ps)  # (B, V, D)
            dists = torch.sum(term1 * centered, dim=2).numpy()  # (B, V)

            for bi, gi in enumerate(range(start, end)):
                ti = true_idx[gi]
                if ti < 0 or V <= 1:
                    continue
                true_d = dists[bi, ti]
                mask = np.ones(V, dtype=bool)
                mask[ti] = False
                min_other = dists[bi, mask].min()
                margins.append(float(true_d - min_other))

        m = np.asarray(margins, dtype=np.float64)
        if len(m) == 0:
            calib[attr_key] = (0.0, 1.0)
        else:
            calib[attr_key] = (
                float(m.mean()),
                float(m.std(ddof=1) if m.size > 1 else 1.0),
            )

    return calib


# ============================================================================
# Realism and Faithfulness Scoring (Mahalanobis)
# ============================================================================


def compute_global_realism_z(
    feats: torch.Tensor,
    global_stats: Dict[str, Any],
    real_E_mean: float,
    real_E_std: float,
    batch_size: int = 2000,
    two_sided: bool = True,
) -> np.ndarray:
    """
    R(x) = z(E_global(x)) using real calibration.

    If two_sided=True (default), returns z² which detects both over-typical (z<0)
    and under-typical (z>0) samples. This is critical for aligned features which
    tend to be over-concentrated toward the real mean (H2 hypothesis).

    If two_sided=False, returns raw z (one-sided, original behavior).
    """
    feats = normalize_features(feats)
    N = len(feats)

    mu = global_stats["mu"]
    P = global_stats["precision"]

    out = np.zeros(N, dtype=np.float64)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        e = compute_mahalanobis(feats[start:end], mu, P).numpy()
        z = zscore(e, real_E_mean, real_E_std)
        if two_sided:
            out[start:end] = z**2  # Two-sided: penalize both over- and under-typical
        else:
            out[start:end] = z  # One-sided: original behavior
    return out


def compute_factorized_faithfulness_margin_z(
    feats: torch.Tensor,
    metadata: Dict,
    factorized_stats: Dict[str, Dict[int, Dict[str, Any]]],
    condition_keys: List[str],
    margin_calib: Dict[str, Tuple[float, float]],
    batch_size: int = 1000,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    For each sample and each attribute k:
      margin_k(x) = E_k(x; true) - min_{v!=true} E_k(x; v)
      zmargin_k(x) = z(margin_k(x)) using real calibration

    Faithfulness:
      F(x) = (1/K) * sum_k zmargin_k(x)

    Returns:
      - F: (N,)
      - per_attr_margin: raw margin arrays per attribute
    """
    feats = normalize_features(feats)
    N = len(feats)
    K = len(condition_keys)

    zsum = np.zeros(N, dtype=np.float64)
    per_attr_margin: Dict[str, np.ndarray] = {}

    for attr_key in condition_keys:
        values = sorted(factorized_stats[attr_key].keys())
        V = len(values)

        mus = torch.stack(
            [factorized_stats[attr_key][v]["mu"] for v in values]
        )  # (V, D)
        Ps = torch.stack(
            [factorized_stats[attr_key][v]["precision"] for v in values]
        )  # (V, D, D)

        true_idx = np.zeros(N, dtype=int)
        for i in range(N):
            val = int(
                metadata[attr_key][i].item()
                if isinstance(metadata[attr_key][i], torch.Tensor)
                else metadata[attr_key][i]
            )
            true_idx[i] = values.index(val) if val in values else -1

        margins = np.zeros(N, dtype=np.float64)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x = feats[start:end]  # (B, D)

            centered = x.unsqueeze(1) - mus.unsqueeze(0)  # (B, V, D)
            term1 = torch.einsum("bvd,vde->bve", centered, Ps)  # (B, V, D)
            dists = torch.sum(term1 * centered, dim=2).numpy()  # (B, V)

            for bi, gi in enumerate(range(start, end)):
                ti = true_idx[gi]
                if ti < 0 or V <= 1:
                    margins[gi] = 0.0
                    continue
                true_d = dists[bi, ti]
                mask = np.ones(V, dtype=bool)
                mask[ti] = False
                min_other = dists[bi, mask].min()
                margins[gi] = float(true_d - min_other)

        m_mean, m_std = margin_calib.get(attr_key, (0.0, 1.0))
        zsum += zscore(margins, m_mean, m_std)
        per_attr_margin[attr_key] = margins

    F = zsum / max(1, K)
    return F, per_attr_margin


# ============================================================================
# Alpha-Precision and Beta-Recall (Alaa et al.)
# ============================================================================


def compute_alpha_precision_scores(
    gen_features: torch.Tensor,
    real_features: torch.Tensor,
    alphas: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute α-precision curve in a fixed feature space."""
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 10, dtype=np.float64)
    else:
        alphas = np.asarray(alphas, dtype=np.float64)

    gen = normalize_features(gen_features).cpu().numpy().astype(np.float64)
    real = normalize_features(real_features).cpu().numpy().astype(np.float64)

    center = real.mean(axis=0)
    real_d = np.linalg.norm(real - center[None, :], axis=1)
    gen_d = np.linalg.norm(gen - center[None, :], axis=1)

    radii = np.quantile(real_d, alphas)
    inside = gen_d[:, None] <= radii[None, :]
    P_alpha = inside.mean(axis=0).astype(np.float64)

    return {
        "alphas": alphas,
        "radii": radii.astype(np.float64),
        "P_alpha": P_alpha,
        "gen_center_dist": gen_d.astype(np.float64),
        "real_center_dist": real_d.astype(np.float64),
    }


def compute_beta_recall_scores(
    gen_features: torch.Tensor,
    real_features: torch.Tensor,
    betas: Optional[np.ndarray] = None,
    k: int = 5,
) -> Dict[str, np.ndarray]:
    """Compute β-recall curve in a fixed feature space."""
    if betas is None:
        betas = np.linspace(0.0, 1.0, 10, dtype=np.float64)
    else:
        betas = np.asarray(betas, dtype=np.float64)

    gen = normalize_features(gen_features).cpu().numpy().astype(np.float64)
    real = normalize_features(real_features).cpu().numpy().astype(np.float64)

    n_real = real.shape[0]
    n_gen = gen.shape[0]
    if n_real < (k + 1):
        raise ValueError(
            f"Need at least k+1 real samples for kNN (got n_real={n_real}, k={k})"
        )
    if n_gen < 1:
        raise ValueError(f"Need at least 1 generated sample (got n_gen={n_gen})")

    nn_real = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn_real.fit(real)
    dists_real, _ = nn_real.kneighbors(real, return_distance=True)
    tau = dists_real[:, k].astype(np.float64)

    c_g = gen.mean(axis=0)
    gen_center_dist = np.linalg.norm(gen - c_g[None, :], axis=1)
    radii = np.quantile(gen_center_dist, betas).astype(np.float64)

    R_beta = np.zeros_like(betas, dtype=np.float64)

    for t, r_beta in enumerate(radii):
        mask = gen_center_dist <= r_beta
        idx = np.where(mask)[0]
        if idx.size == 0:
            R_beta[t] = 0.0
            continue
        gen_typ = gen[idx]

        nn_gen = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn_gen.fit(gen_typ)
        d_rg, _ = nn_gen.kneighbors(real, return_distance=True)
        delta = d_rg[:, 0].astype(np.float64)

        R_beta[t] = (delta <= tau).mean()

    return {
        "betas": betas,
        "radii": radii,
        "R_beta": R_beta,
        "real_knn_radius": tau,
        "gen_center_dist": gen_center_dist.astype(np.float64),
    }


def compute_authenticity_scores(
    gen_features: torch.Tensor,
    real_features: torch.Tensor,
) -> np.ndarray:
    """Compute Alaa et al. authenticity scores (1=authentic, 0=potentially memorized)."""
    gen = normalize_features(gen_features).cpu().numpy().astype(np.float64)
    real = normalize_features(real_features).cpu().numpy().astype(np.float64)

    nn_real = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn_real.fit(real)
    distances_real, _ = nn_real.kneighbors(real)
    real_nn_distances = distances_real[:, 1]

    nn_gen = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn_gen.fit(real)
    distances_gen, indices_gen = nn_gen.kneighbors(gen)
    gen_to_real_distances = distances_gen[:, 0]
    nearest_real_indices = indices_gen[:, 0]

    authenticity = (
        gen_to_real_distances >= real_nn_distances[nearest_real_indices]
    ).astype(np.float64)
    return authenticity


# ============================================================================
# kNN-Based Scoring (Alternative to Mahalanobis)
# ============================================================================


def fit_knn_global_stats(
    features: torch.Tensor,
    k: int = 10,
    metric: str = "cosine",
) -> Dict:
    """
    Fit global kNN model on real features for realism scoring.

    Args:
        features: Real features (N, D)
        k: Number of neighbors for kNN
        metric: Distance metric ("cosine" or "euclidean")

    Returns:
        Dict with:
        - nn_model: Fitted NearestNeighbors model
        - k: Number of neighbors
        - metric: Distance metric used
        - n_samples: Number of samples fitted
    """
    features_norm = normalize_features(features).numpy().astype(np.float64)
    nn_model = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="auto")
    nn_model.fit(features_norm)

    return {
        "nn_model": nn_model,
        "k": k,
        "metric": metric,
        "n_samples": len(features),
    }


def compute_real_calibration_for_global_knn(
    real_feats: torch.Tensor,
    knn_stats: Dict,
    batch_size: int = 2000,
) -> Tuple[float, float]:
    """
    Compute mean/std of kNN radius on real samples (leave-one-out).

    For each real sample, compute distance to k-th nearest neighbor
    (excluding self), then return mean and std for z-scoring.
    """
    real_norm = normalize_features(real_feats).numpy().astype(np.float64)
    N = len(real_norm)
    k = knn_stats["k"]

    # Leave-one-out: query k+1 neighbors and skip self
    nn_model = NearestNeighbors(
        n_neighbors=k + 1, metric=knn_stats["metric"], algorithm="auto"
    )
    nn_model.fit(real_norm)

    radii = np.zeros(N, dtype=np.float64)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        dists, _ = nn_model.kneighbors(real_norm[start:end])
        # Skip self (index 0), take k-th neighbor (index k)
        radii[start:end] = dists[:, k]

    mean = float(np.mean(radii))
    std = float(np.std(radii, ddof=1) if N > 1 else 1.0)
    return mean, std


def compute_global_realism_knn_z(
    feats: torch.Tensor,
    real_feats: torch.Tensor,
    knn_stats: Dict,
    real_knn_mean: float,
    real_knn_std: float,
    batch_size: int = 2000,
    two_sided: bool = False,
) -> np.ndarray:
    """
    kNN-based realism z-score.

    R_knn(x) = z(knn_radius(x; real)) using real calibration.

    Args:
        feats: Features to score (N, D)
        real_feats: Real features for kNN lookup (M, D)
        knn_stats: Dict with k, metric from fit_knn_global_stats
        real_knn_mean, real_knn_std: Calibration from compute_real_calibration_for_global_knn
        two_sided: If True, return z² (penalize both too close and too far)

    Returns:
        Array of shape (N,) with z-scores
    """
    feats_norm = normalize_features(feats).numpy().astype(np.float64)
    real_norm = normalize_features(real_feats).numpy().astype(np.float64)
    N = len(feats_norm)
    k = knn_stats["k"]

    # Fit on real, query with gen
    nn_model = NearestNeighbors(
        n_neighbors=k, metric=knn_stats["metric"], algorithm="auto"
    )
    nn_model.fit(real_norm)

    out = np.zeros(N, dtype=np.float64)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        dists, _ = nn_model.kneighbors(feats_norm[start:end])
        # Take k-th neighbor distance (index k-1)
        radii = dists[:, k - 1]
        z = zscore(radii, real_knn_mean, real_knn_std)
        if two_sided:
            out[start:end] = z**2
        else:
            out[start:end] = z
    return out


def fit_knn_factorized_stats(
    features: torch.Tensor,
    metadata: Dict,
    condition_keys: List[str],
    k: int = 10,
    metric: str = "cosine",
    min_samples: int = 10,
) -> Dict[str, Dict[int, Dict]]:
    """
    Fit per-attribute kNN models for faithfulness scoring.

    For each attribute value, fit a kNN model on the subset of real samples
    with that attribute value.

    Args:
        features: Real features (N, D)
        metadata: Dict with attribute values
        condition_keys: List of attribute names
        k: Number of neighbors for kNN
        metric: Distance metric
        min_samples: Minimum samples per class to fit

    Returns:
        Dict[attr_key][value] = {nn_model, k, metric, n_samples}
    """
    features_norm = normalize_features(features).numpy().astype(np.float64)
    N = len(features_norm)

    # Group samples by attribute value
    attr_groups = {key: {} for key in condition_keys}
    for i in range(N):
        for key in condition_keys:
            val = int(
                metadata[key][i].item()
                if isinstance(metadata[key][i], torch.Tensor)
                else metadata[key][i]
            )
            attr_groups[key].setdefault(val, []).append(i)

    stats = {}
    for attr_key in condition_keys:
        stats[attr_key] = {}
        for val, idx_list in attr_groups[attr_key].items():
            idx = np.array(idx_list, dtype=int)
            n = len(idx)

            if n >= min_samples:
                subset = features_norm[idx]
                nn_model = NearestNeighbors(
                    n_neighbors=min(k, n - 1), metric=metric, algorithm="auto"
                )
                nn_model.fit(subset)
                stats[attr_key][val] = {
                    "nn_model": nn_model,
                    "indices": idx,  # For leave-one-out
                    "k": min(k, n - 1),
                    "metric": metric,
                    "n_samples": n,
                }
            else:
                stats[attr_key][val] = {
                    "nn_model": None,
                    "indices": idx,
                    "k": k,
                    "metric": metric,
                    "n_samples": n,
                }

    return stats


def _iter_batches(n: int, bs: int):
    """Batch iterator helper - extracted to module level to avoid duplication."""
    for s in range(0, n, bs):
        yield s, min(s + bs, n)


def compute_real_calibration_for_factorized_knn_margins(
    real_feats: torch.Tensor,
    real_meta: Dict,
    knn_factorized_stats: Dict[str, Dict[int, Dict]],
    condition_keys: List[str],
    batch_size: int = 2000,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute mean/std of kNN margin on real samples for each attribute.

    kNN margin = knn_dist(x; true_class) - knn_dist(x; nearest_other_class)

    Smaller margin = closer to true class than other classes = good.

    NOTE:
        This implementation is exact but avoids rebuilding a leave-one-out kNN
        graph for every sample. For each (attribute, value) class, it fits one
        kNN model on the class subset, queries the subset against itself with
        k+1 neighbors, and removes each sample's self-match by index. This is
        equivalent to per-sample leave-one-out for the k-th neighbor distance.
    """
    real_norm = normalize_features(real_feats).numpy().astype(np.float64)
    N = len(real_norm)

    calib: Dict[str, Tuple[float, float]] = {}

    for attr_key in condition_keys:
        values = sorted(knn_factorized_stats[attr_key].keys())
        V = len(values)

        if V < 2:
            calib[attr_key] = (0.0, 1.0)
            continue

        # Precompute true attribute value for each real sample once.
        true_vals = np.empty(N, dtype=np.int64)
        for i in range(N):
            true_vals[i] = int(
                real_meta[attr_key][i].item()
                if isinstance(real_meta[attr_key][i], torch.Tensor)
                else real_meta[attr_key][i]
            )

        margins = np.full(N, np.nan, dtype=np.float64)

        # Group global indices by their true attribute value (only values present in stats).
        rows_by_val: Dict[int, np.ndarray] = {}
        for v in values:
            rows = np.where(true_vals == v)[0]
            if rows.size > 0:
                rows_by_val[v] = rows

        for true_val in values:
            true_stats = knn_factorized_stats[attr_key][true_val]
            rows = rows_by_val.get(true_val)
            if rows is None or rows.size == 0:
                continue
            if true_stats["nn_model"] is None:
                continue

            true_idx = np.asarray(true_stats["indices"], dtype=int)
            k_true = int(true_stats["k"])
            if k_true <= 0:
                continue

            # Map global index -> local row within the class subset.
            local_pos = {gi: li for li, gi in enumerate(true_idx.tolist())}
            local_rows = np.array(
                [local_pos.get(int(gi), -1) for gi in rows], dtype=int
            )
            valid_rows_mask = local_rows >= 0
            if not np.any(valid_rows_mask):
                continue

            rows = rows[valid_rows_mask]
            local_rows = local_rows[valid_rows_mask]
            subset = real_norm[true_idx]
            n_subset = subset.shape[0]

            # Need at least k_true+1 points in the class subset to exclude self and still
            # have k_true neighbors. This should normally hold when nn_model is present,
            # but keep the check for robustness.
            if n_subset <= k_true:
                continue

            # Exact leave-one-out kth radius for all needed rows in this class using one fit.
            n_query = min(k_true + 1, n_subset)
            nn_self = NearestNeighbors(
                n_neighbors=n_query,
                metric=true_stats["metric"],
                algorithm="auto",
            )
            nn_self.fit(subset)

            d_true = np.full(rows.shape[0], np.nan, dtype=np.float64)
            # Query only the rows we need (usually all rows in this class for calibration).
            for start, end in _iter_batches(len(rows), batch_size):
                q = subset[local_rows[start:end]]
                dists, inds = nn_self.kneighbors(q, return_distance=True)
                # Remove self-match by identity (index in subset), not by distance,
                # to remain exact even when duplicate vectors exist.
                for bi in range(end - start):
                    row_local = int(local_rows[start + bi])
                    keep = inds[bi] != row_local
                    if np.count_nonzero(keep) < k_true:
                        continue
                    d_true[start + bi] = float(dists[bi][keep][k_true - 1])

            # Distances to other classes: batch query each other class once, then take min.
            min_other = np.full(rows.shape[0], np.inf, dtype=np.float64)
            q_all = subset[local_rows]
            for other_val in values:
                if other_val == true_val:
                    continue
                other_stats = knn_factorized_stats[attr_key][other_val]
                if other_stats["nn_model"] is None:
                    continue

                k_other = int(other_stats["k"])
                if k_other <= 0:
                    continue

                d_other_vec = np.empty(rows.shape[0], dtype=np.float64)
                for start, end in _iter_batches(len(rows), batch_size):
                    d_other, _ = other_stats["nn_model"].kneighbors(
                        q_all[start:end], return_distance=True
                    )
                    d_other_vec[start:end] = d_other[:, k_other - 1]
                min_other = np.minimum(min_other, d_other_vec)

            valid = np.isfinite(d_true) & np.isfinite(min_other)
            if np.any(valid):
                margins[rows[valid]] = d_true[valid] - min_other[valid]

        m = margins[np.isfinite(margins)]
        if len(m) == 0:
            calib[attr_key] = (0.0, 1.0)
        else:
            calib[attr_key] = (
                float(m.mean()),
                float(m.std(ddof=1) if len(m) > 1 else 1.0),
            )

    return calib


def compute_factorized_faithfulness_knn_margin_z(
    feats: torch.Tensor,
    metadata: Dict,
    real_feats: torch.Tensor,
    real_meta: Dict,
    knn_factorized_stats: Dict[str, Dict[int, Dict]],
    condition_keys: List[str],
    margin_calib: Dict[str, Tuple[float, float]],
    k: int = 10,
    metric: str = "cosine",
    batch_size: int = 1000,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    kNN-based faithfulness margin z-score.

    For each sample and attribute:
      margin_k(x) = knn_dist(x; true_class) - knn_dist(x; nearest_other_class)
      zmargin_k(x) = z(margin_k(x)) using real calibration

    Faithfulness = (1/K) * sum_k zmargin_k(x)

    Returns:
        - F: (N,) faithfulness z-scores
        - per_attr_margin: raw margin arrays per attribute

    NOTE:
        Exact implementation using batched kNN queries. It avoids the previous
        O(N * V) Python-level sample loop by querying each class model on whole
        batches and assembling margins from those distances.
    """
    feats_norm = normalize_features(feats).numpy().astype(np.float64)
    # Keep normalization for parity with the existing API (real_feats/real_meta are unused
    # for scoring but kept in the signature for compatibility).
    _ = normalize_features(real_feats).numpy().astype(np.float64)
    N = len(feats_norm)
    K = len(condition_keys)

    zsum = np.zeros(N, dtype=np.float64)
    per_attr_margin: Dict[str, np.ndarray] = {}

    for attr_key in condition_keys:
        values = sorted(knn_factorized_stats[attr_key].keys())
        V = len(values)

        margins = np.zeros(N, dtype=np.float64)

        if V < 2:
            per_attr_margin[attr_key] = margins
            continue

        value_to_col = {v: j for j, v in enumerate(values)}

        # True class per sample for this attribute.
        true_idx = np.full(N, -1, dtype=np.int64)
        for i in range(N):
            v = int(
                metadata[attr_key][i].item()
                if isinstance(metadata[attr_key][i], torch.Tensor)
                else metadata[attr_key][i]
            )
            true_idx[i] = value_to_col.get(v, -1)

        # Distances to each class model (k-th NN radius wrt that class), computed in batches.
        dmat = np.full((N, V), np.inf, dtype=np.float64)
        for col, v in enumerate(values):
            stats_v = knn_factorized_stats[attr_key][v]
            nn_model = stats_v["nn_model"]
            k_v = int(stats_v.get("k", 0))
            if nn_model is None or k_v <= 0:
                continue

            for start, end in _iter_batches(N, batch_size):
                dists, _ = nn_model.kneighbors(
                    feats_norm[start:end], return_distance=True
                )
                dmat[start:end, col] = dists[:, k_v - 1]

        valid_true = true_idx >= 0
        if np.any(valid_true):
            rows = np.where(valid_true)[0]
            cols = true_idx[rows]
            d_true = dmat[rows, cols]

            d_other = dmat[rows].copy()
            d_other[np.arange(len(rows)), cols] = np.inf
            min_other = d_other.min(axis=1)

            valid_margin = np.isfinite(d_true) & np.isfinite(min_other)
            margins[rows[valid_margin]] = d_true[valid_margin] - min_other[valid_margin]
            # Invalid margins remain 0.0 (backward-compatible behavior).

        m_mean, m_std = margin_calib.get(attr_key, (0.0, 1.0))
        zsum += zscore(margins, m_mean, m_std)
        per_attr_margin[attr_key] = margins

    F = zsum / max(1, K)
    return F, per_attr_margin


def fit_knn_scoring_components(
    calib_feats: torch.Tensor,
    calib_meta: Dict,
    condition_keys: List[str],
    k: int = 10,
    metric: str = "cosine",
) -> Dict[str, Any]:
    """
    Fit all kNN-based components for trust scoring.

    Analogous to fit_trust_scoring_components but using kNN instead of Mahalanobis.
    """
    knn_global_stats = fit_knn_global_stats(calib_feats, k=k, metric=metric)
    knn_factorized_stats = fit_knn_factorized_stats(
        calib_feats, calib_meta, condition_keys, k=k, metric=metric
    )
    real_knn_mean, real_knn_std = compute_real_calibration_for_global_knn(
        calib_feats, knn_global_stats
    )
    margin_calib = compute_real_calibration_for_factorized_knn_margins(
        calib_feats, calib_meta, knn_factorized_stats, condition_keys
    )

    return {
        "scoring_method": "knn",
        "knn_global_stats": knn_global_stats,
        "knn_factorized_stats": knn_factorized_stats,
        "real_knn_mean": real_knn_mean,
        "real_knn_std": real_knn_std,
        "margin_calib": margin_calib,
        "condition_keys": condition_keys,
        "calib_feats": calib_feats,  # Needed for kNN queries
        "calib_meta": calib_meta,
        "k": k,
        "metric": metric,
    }


def score_trust_from_knn_components(
    feats: torch.Tensor,
    meta: Dict,
    components: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Score features using pre-fit kNN-based trust scoring components.

    Returns:
        Tuple of (realism_z, faithfulness_z, trust_z) arrays of shape (N,)
    """
    knn_global_stats = components["knn_global_stats"]
    knn_factorized_stats = components["knn_factorized_stats"]
    real_knn_mean = components["real_knn_mean"]
    real_knn_std = components["real_knn_std"]
    margin_calib = components["margin_calib"]
    condition_keys = components["condition_keys"]
    calib_feats = components["calib_feats"]
    calib_meta = components["calib_meta"]

    realism_z = compute_global_realism_knn_z(
        feats, calib_feats, knn_global_stats, real_knn_mean, real_knn_std
    )
    faithfulness_z, _ = compute_factorized_faithfulness_knn_margin_z(
        feats,
        meta,
        calib_feats,
        calib_meta,
        knn_factorized_stats,
        condition_keys,
        margin_calib,
        k=components["k"],
        metric=components["metric"],
    )
    trust_z = realism_z + faithfulness_z

    return realism_z, faithfulness_z, trust_z


# ============================================================================
# Main Trust Scoring API
# ============================================================================


def compute_trust_results_from_features(
    dataset: str,
    model: str,
    feature_type: str,
    real_feats: torch.Tensor,
    real_meta: Dict,
    gen_feats: torch.Tensor,
    gen_meta: Dict,
    condition_keys: List[str],
    filter_by_seen: bool = False,
    seen_combos: Optional[Set[Tuple[int, ...]]] = None,
    use_shared_cov: Optional[bool] = None,
    scoring_method: str = "mahalanobis",
    knn_k: int = 10,
) -> Dict:
    """
    Compute the full trust_results dict (same schema as outputs/trust_scores/*.pt), without any I/O.

    Args:
        scoring_method: "mahalanobis" (default) or "knn"
        knn_k: Number of neighbors for kNN scoring (only used if scoring_method="knn")
    """
    if use_shared_cov is None:
        # Default to LDA-style shared covariance for fair margin comparison
        use_shared_cov = True

    # Optionally restrict real stats/calibration to the model's seen training support (marginal models).
    real_used_feats, real_used_meta = filter_feats_and_meta_by_seen_combos(
        real_feats, real_meta, condition_keys, seen_combos if filter_by_seen else None
    )

    if scoring_method == "knn":
        # kNN-based scoring
        knn_global_stats = fit_knn_global_stats(
            real_used_feats, k=knn_k, metric="cosine"
        )
        knn_factorized_stats = fit_knn_factorized_stats(
            real_used_feats, real_used_meta, condition_keys, k=knn_k, metric="cosine"
        )
        real_knn_mean, real_knn_std = compute_real_calibration_for_global_knn(
            real_used_feats, knn_global_stats
        )
        margin_calib = compute_real_calibration_for_factorized_knn_margins(
            real_used_feats, real_used_meta, knn_factorized_stats, condition_keys
        )

        # Score generated samples using kNN
        realism_global_z = compute_global_realism_knn_z(
            gen_feats,
            real_used_feats,
            knn_global_stats,
            real_knn_mean,
            real_knn_std,
            two_sided=False,
        )
        faithfulness_margin_z, _ = compute_factorized_faithfulness_knn_margin_z(
            gen_feats,
            gen_meta,
            real_used_feats,
            real_used_meta,
            knn_factorized_stats,
            condition_keys,
            margin_calib,
            k=knn_k,
            metric="cosine",
        )
        trust_updated = realism_global_z + faithfulness_margin_z

        # Store kNN-specific stats for summary
        global_stats_summary = {
            "n_samples": knn_global_stats["n_samples"],
            "scoring_method": "knn",
            "knn_k": knn_k,
            "real_knn_mean": real_knn_mean,
            "real_knn_std": real_knn_std,
        }
    else:
        # Mahalanobis-based scoring (default)
        factorized_stats = fit_factorized_stats(
            real_used_feats,
            real_used_meta,
            condition_keys,
            regularization=1e-5,
            use_shared_cov=bool(use_shared_cov),
        )
        global_stats = fit_global_stats(real_used_feats, regularization=1e-5)
        real_E_mean, real_E_std = compute_real_calibration_for_global_energy(
            real_used_feats, global_stats
        )
        margin_calib = compute_real_calibration_for_factorized_margins(
            real_used_feats, real_used_meta, factorized_stats, condition_keys
        )

        # Score generated samples using Mahalanobis
        realism_global_z = compute_global_realism_z(
            gen_feats, global_stats, real_E_mean, real_E_std, two_sided=False
        )
        faithfulness_margin_z, _ = compute_factorized_faithfulness_margin_z(
            gen_feats, gen_meta, factorized_stats, condition_keys, margin_calib
        )
        trust_updated = realism_global_z + faithfulness_margin_z

        global_stats_summary = {
            "n_samples": global_stats["n_samples"],
            "scoring_method": "mahalanobis",
            "shrinkage": global_stats.get("shrinkage", float("nan")),
            "real_E_mean": real_E_mean,
            "real_E_std": real_E_std,
        }

    # Alaa et al. metrics (computed in the same feature space)
    alpha_res = compute_alpha_precision_scores(
        gen_feats, real_used_feats, alphas=np.linspace(0, 1, 11, dtype=np.float64)
    )
    beta_res = compute_beta_recall_scores(
        gen_feats, real_used_feats, betas=np.linspace(0, 1, 11, dtype=np.float64), k=5
    )
    authenticity = compute_authenticity_scores(gen_feats, real_used_feats)

    true_conditions = [
        get_condition_key(gen_meta, condition_keys, i) for i in range(len(gen_feats))
    ]

    return {
        "dataset": dataset,
        "model": model,
        "feature_type": feature_type,
        "n_samples": len(gen_feats),
        "n_real_used_for_stats": int(len(real_used_feats)),
        "alpha_grid": alpha_res["alphas"],
        "P_alpha": alpha_res["P_alpha"],
        "gen_center_dist_real": alpha_res["gen_center_dist"],
        "real_center_dist_real": alpha_res["real_center_dist"],
        "alpha_radii": alpha_res["radii"],
        "beta_grid": beta_res["betas"],
        "R_beta": beta_res["R_beta"],
        "beta_radii": beta_res["radii"],
        "real_knn_radius": beta_res["real_knn_radius"],
        "gen_center_dist_gen": beta_res["gen_center_dist"],
        "authenticity": authenticity,
        "true_conditions": true_conditions,
        "realism_global_z": realism_global_z,
        "faithfulness_margin_z": faithfulness_margin_z,
        "trust_updated": trust_updated,
        "global_stats_summary": global_stats_summary,
        "margin_calib": margin_calib,
    }


def fit_trust_scoring_components(
    calib_feats: torch.Tensor,
    calib_meta: Dict,
    condition_keys: List[str],
    regularization: float = 1e-5,
    use_shared_cov: bool = True,
    scoring_method: str = "mahalanobis",
    knn_k: int = 10,
) -> Dict[str, Any]:
    """
    Fit all components needed for trust scoring: Gaussian stats + calibration.

    This function fits:
    - Global Gaussian (mu, precision) for realism scoring
    - Per-attribute Gaussians for faithfulness margin scoring
    - Calibration mean/std for z-scoring (computed on the same calib set)

    Args:
        calib_feats: Features to fit on (N, D)
        calib_meta: Metadata dict with condition keys
        condition_keys: List of attribute names
        regularization: Regularization for covariance
        use_shared_cov: Use shared covariance across attribute values
        scoring_method: "mahalanobis" (default) or "knn"
        knn_k: Number of neighbors for kNN scoring

    Returns:
        Dict with all fitted components
    """
    if scoring_method == "knn":
        return fit_knn_scoring_components(
            calib_feats, calib_meta, condition_keys, k=knn_k, metric="cosine"
        )

    # Mahalanobis-based (default)
    factorized_stats = fit_factorized_stats(
        calib_feats,
        calib_meta,
        condition_keys,
        regularization=regularization,
        use_shared_cov=use_shared_cov,
    )
    global_stats = fit_global_stats(calib_feats, regularization=regularization)
    real_E_mean, real_E_std = compute_real_calibration_for_global_energy(
        calib_feats, global_stats
    )
    margin_calib = compute_real_calibration_for_factorized_margins(
        calib_feats, calib_meta, factorized_stats, condition_keys
    )

    return {
        "scoring_method": "mahalanobis",
        "global_stats": global_stats,
        "factorized_stats": factorized_stats,
        "real_E_mean": real_E_mean,
        "real_E_std": real_E_std,
        "margin_calib": margin_calib,
        "condition_keys": condition_keys,
    }


def score_trust_from_components(
    feats: torch.Tensor,
    meta: Dict,
    components: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Score features using pre-fit trust scoring components.

    Args:
        feats: Features to score (N, D)
        meta: Metadata dict with condition keys
        components: Dict from fit_trust_scoring_components()

    Returns:
        Tuple of (realism_z, faithfulness_z, trust_z) arrays of shape (N,)
    """
    scoring_method = components.get("scoring_method", "mahalanobis")

    if scoring_method == "knn":
        return score_trust_from_knn_components(feats, meta, components)

    # Mahalanobis-based (default)
    global_stats = components["global_stats"]
    factorized_stats = components["factorized_stats"]
    real_E_mean = components["real_E_mean"]
    real_E_std = components["real_E_std"]
    margin_calib = components["margin_calib"]
    condition_keys = components["condition_keys"]

    realism_z = compute_global_realism_z(
        feats, global_stats, real_E_mean, real_E_std, two_sided=False
    )
    faithfulness_z, _ = compute_factorized_faithfulness_margin_z(
        feats, meta, factorized_stats, condition_keys, margin_calib
    )
    trust_z = realism_z + faithfulness_z

    return realism_z, faithfulness_z, trust_z


def compute_real_sample_scores(
    real_feats: torch.Tensor,
    real_meta: Dict,
    condition_keys: List[str],
    filter_by_seen: bool = False,
    seen_combos: set = None,
    components: Optional[Dict[str, Any]] = None,
    scoring_method: str = "mahalanobis",
    knn_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute realism_z, faithfulness_z, and trust_z for real samples.

    This uses the exact same scoring pipeline as generated samples:
      - pick a calibration real subset (optionally restricted to seen combos)
      - fit global + factorized stats on that subset
      - calibrate energy + margins on that subset
      - score the provided real_feats/real_meta against that calibration

    Args:
        real_feats: Real features to score (N, D)
        real_meta: Metadata dict with condition keys
        condition_keys: List of attribute names
        filter_by_seen: If True, only use seen combos for calibration
        seen_combos: Set of seen condition tuples (for marginal models)
        components: Optional pre-fit components. If provided, uses these
                   instead of fitting on real_feats (avoids calibration bias).
        scoring_method: "mahalanobis" (default) or "knn"
        knn_k: Number of neighbors for kNN scoring

    Returns:
        (realism_z, faithfulness_z, trust_z) arrays of shape (N,)
    """
    if components is not None:
        # Use pre-fit components (cross-fit scenario)
        return score_trust_from_components(real_feats, real_meta, components)

    # Fit on filtered subset (original behavior, has calibration bias)
    calib_feats, calib_meta = filter_feats_and_meta_by_seen_combos(
        real_feats, real_meta, condition_keys, seen_combos if filter_by_seen else None
    )

    use_shared_cov = True  # LDA-style shared covariance for fair margin comparison
    components = fit_trust_scoring_components(
        calib_feats,
        calib_meta,
        condition_keys,
        regularization=1e-5,
        use_shared_cov=use_shared_cov,
        scoring_method=scoring_method,
        knn_k=knn_k,
    )

    return score_trust_from_components(real_feats, real_meta, components)


def dedupe_generated(
    gen_feats: torch.Tensor,
    gen_meta: Dict,
    trust_results: Dict,
    precision: int = 6,
) -> Tuple[torch.Tensor, Dict, Dict, Dict]:
    """
    Deduplicate identical generated samples based on feature hashing.

    Args:
        gen_feats: Generated features (N, D)
        gen_meta: Metadata dict with condition keys
        trust_results: Dict with per-sample arrays (trust_updated, realism_global_z, etc.)
        precision: Decimal places to round features before hashing (default 6)

    Returns:
        Tuple of (deduped_gen_feats, deduped_gen_meta, deduped_trust_results, stats)
        where stats contains: original_n, deduped_n, n_removed, dedupe_key
    """
    N = gen_feats.shape[0]

    # Hash each feature vector
    seen_hashes = set()
    keep_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        feat = gen_feats[i]
        h = hashlib.md5(np.round(feat.numpy(), precision).tobytes()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            keep_mask[i] = True

    # Apply keep_mask to gen_feats
    keep_indices = np.where(keep_mask)[0]
    deduped_gen_feats = gen_feats[keep_indices]

    # Apply keep_mask to gen_meta (handle both tensor and list/array)
    deduped_gen_meta = {}
    for k, v in gen_meta.items():
        if isinstance(v, torch.Tensor):
            deduped_gen_meta[k] = v[keep_indices]
        elif isinstance(v, np.ndarray):
            deduped_gen_meta[k] = v[keep_mask]
        elif isinstance(v, list):
            deduped_gen_meta[k] = [v[i] for i in range(N) if keep_mask[i]]
        else:
            deduped_gen_meta[k] = v

    # Apply keep_mask to trust_results
    deduped_trust_results = {}
    trust_keys = [
        "trust_updated",
        "realism_global_z",
        "faithfulness_margin_z",
        "authenticity",
        "gen_center_dist_real",
        "true_conditions",
    ]

    for k in trust_keys:
        if k not in trust_results:
            continue
        v = trust_results[k]
        if isinstance(v, torch.Tensor):
            deduped_trust_results[k] = v[keep_indices]
        elif isinstance(v, np.ndarray):
            deduped_trust_results[k] = v[keep_mask]
        elif isinstance(v, list):
            deduped_trust_results[k] = [v[i] for i in range(N) if keep_mask[i]]
        else:
            deduped_trust_results[k] = v

    # Copy over any other keys from trust_results unchanged
    for k, v in trust_results.items():
        if k not in deduped_trust_results:
            deduped_trust_results[k] = v

    # Build stats
    deduped_n = int(keep_mask.sum())
    stats = {
        "original_n": N,
        "deduped_n": deduped_n,
        "n_removed": N - deduped_n,
        "dedupe_key": f"md5_round{precision}",
    }

    return deduped_gen_feats, deduped_gen_meta, deduped_trust_results, stats
