"""
Helper utilities for trust evaluation extensions.

Provides:
- Image path mapping from conditions
- Grid creation utilities
- Real sample scoring
- Bootstrap KID computation
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

# ============================================================================
# Image Path Mapping
# ============================================================================


def condition_to_signature(condition: tuple, condition_keys: List[str]) -> str:
    """
    Convert condition tuple to alphabetically sorted filename signature.

    Example:
        condition = (1, 0, 0, 1)  # (Male, Smiling, Blond_Hair, Eyeglasses)
        condition_keys = ["Male", "Smiling", "Blond_Hair", "Eyeglasses"]
        -> "Blond_Hair0_Eyeglasses1_Male1_Smiling0"
    """
    # Create list of (key, value) pairs
    pairs = list(zip(condition_keys, condition))

    # Sort alphabetically by key
    pairs_sorted = sorted(pairs, key=lambda x: x[0])

    # Format as key0_key1_...
    parts = [f"{k}{v}" for k, v in pairs_sorted]
    return "_".join(parts)


def get_image_path(
    condition: tuple,
    idx: int,
    model_dir: str,
    condition_keys: List[str],
) -> Path:
    """
    Map condition tuple and index to image file path.

    Args:
        condition: Condition values tuple
        idx: Sample index within condition
        model_dir: Model directory name (e.g., "celeba_vanilla_full")
        condition_keys: Attribute names

    Returns:
        Path to image file
    """
    signature = condition_to_signature(condition, condition_keys)
    filename = f"{signature}_{idx}.png"
    return Path(f"outputs/gen/{model_dir}/images/{filename}")


# ============================================================================
# Image Grid Creation
# ============================================================================


def create_image_grid(
    image_paths: List[Path],
    titles: List[str],
    scores: List[Tuple[float, float]] = None,
) -> Image.Image:
    """
    Create a 2×2 grid of images with titles and optional scores.

    Args:
        image_paths: List of 4 image paths (top-left, top-right, bottom-left, bottom-right)
        titles: List of 4 titles (e.g., ["Good R + Good F", "Good R + Bad F", ...])
        scores: Optional list of 4 (realism_z, faithfulness_z) tuples

    Returns:
        PIL Image with 2×2 grid
    """
    if len(image_paths) != 4 or len(titles) != 4:
        raise ValueError("Need exactly 4 images and 4 titles for 2×2 grid")

    # Load images (or use placeholder if missing)
    images = []
    for path in image_paths:
        if path.exists():
            img = Image.open(path).convert("RGB")
        else:
            # Create placeholder
            img = Image.new("RGB", (256, 256), color=(200, 200, 200))
            draw = ImageDraw.Draw(img)
            draw.text((128, 128), "Missing", fill=(100, 100, 100), anchor="mm")
        images.append(img)

    # Resize to consistent size
    img_size = 256
    images = [
        img.resize((img_size, img_size), Image.Resampling.LANCZOS) for img in images
    ]

    # Create grid canvas (2×2 + margins for text)
    margin = 60
    grid_width = 2 * img_size + 3 * margin
    grid_height = 2 * img_size + 4 * margin
    grid = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))

    # Paste images
    positions = [
        (margin, margin),  # top-left
        (2 * margin + img_size, margin),  # top-right
        (margin, 2 * margin + img_size),  # bottom-left
        (2 * margin + img_size, 2 * margin + img_size),  # bottom-right
    ]

    for img, pos in zip(images, positions):
        grid.paste(img, pos)

    # Add titles and scores
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, (title, pos) in enumerate(zip(titles, positions)):
        text_x = pos[0] + img_size // 2
        text_y = pos[1] + img_size + 10

        # Draw title
        draw.text((text_x, text_y), title, fill=(0, 0, 0), anchor="mt", font=font)

        # Draw scores if provided
        if scores is not None:
            r_z, f_z = scores[i]
            score_text = f"R:{r_z:.2f} F:{f_z:.2f}"
            draw.text(
                (text_x, text_y + 20),
                score_text,
                fill=(100, 100, 100),
                anchor="mt",
                font=font,
            )

    return grid


# ============================================================================
# Real Sample Scoring
# ============================================================================


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """L2 normalize features (same as compute_trust_scores.py)."""
    return features / (features.norm(dim=1, keepdim=True) + 1e-12)


def compute_mahalanobis(
    x: torch.Tensor, mu: torch.Tensor, precision: torch.Tensor
) -> torch.Tensor:
    """Compute Mahalanobis distance (same as compute_trust_scores.py)."""
    centered = x - mu.unsqueeze(0)
    term = torch.einsum("bd,de->be", centered, precision)
    dist = torch.sum(term * centered, dim=1)
    return dist


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
                    pooled_cov = (1 - shrinkage) * pooled_cov + shrinkage * trace_div_d * torch.eye(D)
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


def zscore(x: np.ndarray, mean: float, std: float, eps: float = 1e-12) -> np.ndarray:
    """Z-score normalization."""
    return (x - mean) / (std + eps)


def get_condition_key(metadata: Dict, keys: List[str], idx: int) -> Tuple[int, ...]:
    """Extract a joint condition tuple from metadata at index idx."""
    return tuple(
        int(
            metadata[k][idx].item()
            if isinstance(metadata[k][idx], torch.Tensor)
            else metadata[k][idx]
        )
        for k in keys
    )


def _filter_feats_and_meta_by_seen_combos(
    feats: torch.Tensor,
    meta: Dict,
    condition_keys: List[str],
    seen_combos: Optional[Set[Tuple[int, ...]]],
) -> Tuple[torch.Tensor, Dict]:
    """Keep only rows whose joint condition is in seen_combos."""
    if not seen_combos:
        return feats, meta

    N = len(feats)
    keep_list = [
        get_condition_key(meta, condition_keys, i) in seen_combos for i in range(N)
    ]
    keep = torch.tensor(keep_list, dtype=torch.bool)

    feats_f = feats[keep]
    meta_f: Dict = {}
    for k, v in meta.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == N:
            meta_f[k] = v[keep]
        elif isinstance(v, list) and len(v) == N:
            meta_f[k] = [vv for vv, kk in zip(v, keep_list) if kk]
        else:
            meta_f[k] = v

    return feats_f, meta_f


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


def compute_global_realism_z(
    feats: torch.Tensor,
    global_stats: Dict[str, Any],
    real_E_mean: float,
    real_E_std: float,
    batch_size: int = 2000,
) -> np.ndarray:
    """R(x) = z(E_global(x)) using real calibration."""
    feats = normalize_features(feats)
    N = len(feats)

    mu = global_stats["mu"]
    P = global_stats["precision"]

    out = np.zeros(N, dtype=np.float64)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        e = compute_mahalanobis(feats[start:end], mu, P).numpy()
        out[start:end] = zscore(e, real_E_mean, real_E_std)
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
        # Return NaNs rather than hard-failing the eval pipeline
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
) -> Dict:
    """Compute the full trust_results dict (same schema as outputs/trust_scores/*.pt), without any I/O."""
    if use_shared_cov is None:
        # Default to LDA-style shared covariance for fair margin comparison
        use_shared_cov = True

    # Optionally restrict real stats/calibration to the model's seen training support (marginal models).
    real_used_feats, real_used_meta = _filter_feats_and_meta_by_seen_combos(
        real_feats, real_meta, condition_keys, seen_combos if filter_by_seen else None
    )

    # Fit stats + calibration on the chosen real subset
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

    # Score generated samples
    realism_global_z = compute_global_realism_z(
        gen_feats, global_stats, real_E_mean, real_E_std
    )
    faithfulness_margin_z, _ = compute_factorized_faithfulness_margin_z(
        gen_feats, gen_meta, factorized_stats, condition_keys, margin_calib
    )
    trust_updated = realism_global_z + faithfulness_margin_z

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
        "global_stats_summary": {
            "n_samples": global_stats["n_samples"],
            "shrinkage": global_stats.get("shrinkage", float("nan")),
            "real_E_mean": real_E_mean,
            "real_E_std": real_E_std,
        },
        "margin_calib": margin_calib,
    }


def fit_trust_scoring_components(
    calib_feats: torch.Tensor,
    calib_meta: Dict,
    condition_keys: List[str],
    regularization: float = 1e-5,
    use_shared_cov: bool = True,
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

    Returns:
        Dict with all fitted components:
        - global_stats: Dict with mu, precision
        - factorized_stats: Dict of per-attribute stats
        - real_E_mean, real_E_std: Global energy calibration
        - margin_calib: Per-attribute margin calibration
    """
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
    global_stats = components["global_stats"]
    factorized_stats = components["factorized_stats"]
    real_E_mean = components["real_E_mean"]
    real_E_std = components["real_E_std"]
    margin_calib = components["margin_calib"]
    condition_keys = components["condition_keys"]

    realism_z = compute_global_realism_z(feats, global_stats, real_E_mean, real_E_std)
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

    Returns:
        (realism_z, faithfulness_z, trust_z) arrays of shape (N,)
    """
    if components is not None:
        # Use pre-fit components (cross-fit scenario)
        return score_trust_from_components(real_feats, real_meta, components)

    # Fit on filtered subset (original behavior, has calibration bias)
    calib_feats, calib_meta = _filter_feats_and_meta_by_seen_combos(
        real_feats, real_meta, condition_keys, seen_combos if filter_by_seen else None
    )

    use_shared_cov = True  # LDA-style shared covariance for fair margin comparison
    components = fit_trust_scoring_components(
        calib_feats,
        calib_meta,
        condition_keys,
        regularization=1e-5,
        use_shared_cov=use_shared_cov,
    )

    return score_trust_from_components(real_feats, real_meta, components)

    # """
    # Compute realism_z, faithfulness_z, and trust_z for real samples.

    # Uses the same scoring logic as generated samples:
    # - Fit stats on real features
    # - Compute calibration on real features
    # - Score real samples

    # Args:
    #     real_feats: Real features (N, D)
    #     real_meta: Metadata dict with condition keys
    #     condition_keys: List of attribute names
    #     filter_by_seen: If True, only use seen combos for calibration
    #     seen_combos: Set of seen condition tuples (for marginal models)

    # Returns:
    #     (realism_z, faithfulness_z, trust_z) arrays of shape (N,)
    # """
    # N = len(real_feats)

    # # Filter to seen combos if requested
    # if filter_by_seen and seen_combos is not None:
    #     seen_mask = np.zeros(N, dtype=bool)
    #     for i in range(N):
    #         cond = tuple(
    #             int(
    #                 real_meta[k][i].item()
    #                 if isinstance(real_meta[k][i], torch.Tensor)
    #                 else real_meta[k][i]
    #             )
    #             for k in condition_keys
    #         )
    #         if cond in seen_combos:
    #             seen_mask[i] = True

    #     calib_feats = real_feats[seen_mask]
    #     calib_meta = {k: real_meta[k][seen_mask] for k in condition_keys}
    # else:
    #     calib_feats = real_feats
    #     calib_meta = real_meta

    # # Fit global stats on calibration set
    # global_stats = fit_global_stats(calib_feats)

    # # Compute global energy on calibration set for normalization
    # calib_feats_norm = normalize_features(calib_feats)
    # mu = global_stats["mu"]
    # P = global_stats["precision"]
    # calib_energy = compute_mahalanobis(calib_feats_norm, mu, P).numpy()
    # E_mean = float(np.mean(calib_energy))
    # E_std = float(np.std(calib_energy, ddof=1) if len(calib_energy) > 1 else 1.0)

    # # Fit factorized stats on calibration set
    # factorized_stats = fit_factorized_stats(calib_feats, calib_meta, condition_keys)

    # # Compute margin calibration on calibration set
    # margin_calib = {}
    # for attr_key in condition_keys:
    #     values = sorted(factorized_stats[attr_key].keys())
    #     V = len(values)
    #     mus = torch.stack([factorized_stats[attr_key][v]["mu"] for v in values])
    #     Ps = torch.stack([factorized_stats[attr_key][v]["precision"] for v in values])

    #     margins = []
    #     calib_feats_norm = normalize_features(calib_feats)
    #     for i in range(len(calib_feats_norm)):
    #         val = int(
    #             calib_meta[attr_key][i].item()
    #             if isinstance(calib_meta[attr_key][i], torch.Tensor)
    #             else calib_meta[attr_key][i]
    #         )
    #         if val not in values or V <= 1:
    #             continue

    #         ti = values.index(val)
    #         x = calib_feats_norm[i : i + 1]

    #         centered = x.unsqueeze(1) - mus.unsqueeze(0)
    #         term1 = torch.einsum("bvd,vde->bve", centered, Ps)
    #         dists = torch.sum(term1 * centered, dim=2).numpy()[0]

    #         true_d = dists[ti]
    #         mask = np.ones(V, dtype=bool)
    #         mask[ti] = False
    #         min_other = dists[mask].min()
    #         margins.append(true_d - min_other)

    #     m = np.array(margins, dtype=np.float64)
    #     if len(m) == 0:
    #         margin_calib[attr_key] = (0.0, 1.0)
    #     else:
    #         margin_calib[attr_key] = (
    #             float(m.mean()),
    #             float(m.std(ddof=1) if len(m) > 1 else 1.0),
    #         )

    # # Score all real samples
    # real_feats_norm = normalize_features(real_feats)

    # # Realism z-scores
    # real_energy = compute_mahalanobis(real_feats_norm, mu, P).numpy()
    # realism_z = zscore(real_energy, E_mean, E_std)

    # # Faithfulness z-scores
    # K = len(condition_keys)
    # faithfulness_z = np.zeros(N, dtype=np.float64)

    # for attr_key in condition_keys:
    #     values = sorted(factorized_stats[attr_key].keys())
    #     V = len(values)
    #     mus = torch.stack([factorized_stats[attr_key][v]["mu"] for v in values])
    #     Ps = torch.stack([factorized_stats[attr_key][v]["precision"] for v in values])

    #     margins = np.zeros(N, dtype=np.float64)

    #     for i in range(N):
    #         val = int(
    #             real_meta[attr_key][i].item()
    #             if isinstance(real_meta[attr_key][i], torch.Tensor)
    #             else real_meta[attr_key][i]
    #         )
    #         if val not in values or V <= 1:
    #             margins[i] = 0.0
    #             continue

    #         ti = values.index(val)
    #         x = real_feats_norm[i : i + 1]

    #         centered = x.unsqueeze(1) - mus.unsqueeze(0)
    #         term1 = torch.einsum("bvd,vde->bve", centered, Ps)
    #         dists = torch.sum(term1 * centered, dim=2).numpy()[0]

    #         true_d = dists[ti]
    #         mask = np.ones(V, dtype=bool)
    #         mask[ti] = False
    #         min_other = dists[mask].min()
    #         margins[i] = true_d - min_other

    #     m_mean, m_std = margin_calib[attr_key]
    #     faithfulness_z += zscore(margins, m_mean, m_std)

    # faithfulness_z /= max(1, K)

    # # Trust = realism + faithfulness
    # trust_z = realism_z + faithfulness_z

    # return realism_z, faithfulness_z, trust_z


# ============================================================================
# Bootstrap KID for Decile Binning
# ============================================================================


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


# ============================================================================
# Deduplication Helper
# ============================================================================


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
