"""
Compute Trust Scores for Conditional Generative Model Evaluation.

Implements a tuning-free decomposition:
- Realism: Global Mahalanobis energy z-scored against real distribution
- Faithfulness: Factorized per-attribute margins z-scored against real
  margin_k(x) = E_k(x; true_value) - min_{v != true} E_k(x; v)
- Trust: T(x) = realism_z(x) + faithfulness_z(x)  [lower = better]

Also computes Alaa et al. (ICML 2022) metrics:
- α-precision curve (coverage of real support)
- β-recall curve (mode coverage)
- Authenticity (kNN memorization check)

Usage:
    uv run python scripts/compute_trust_scores.py --dataset celeba
    uv run python scripts/compute_trust_scores.py --dataset rxrx1
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path("outputs/trust_scores")

# Path mappings for generated samples
# Maps (dataset, model) -> directory name under outputs/gen/
GEN_PATH_MAP = {
    ("celeba", "fullmodel"): "celeba_vanilla_full",
    ("celeba", "marginalmodel"): "celeba_vanilla_marginal",
    ("celeba", "repamodel"): "celeba_repa_full",
    ("celeba", "repa_marginalmodel"): "celeba_repa_marginal",
    ("rxrx1", "fullmodel"): "rxrx1_vanilla_full",
    ("rxrx1", "marginalmodel"): "rxrx1_vanilla_marginal",
}

# Condition attributes per dataset
CONDITION_ATTRS = {
    "celeba": ["Male", "Smiling", "Blond_Hair", "Eyeglasses"],
    "rxrx1": ["cell_type_id", "sirna_id"],
}

ENCODERS = {
    "celeba": ["dinov2", "dinov3", "mae", "siglip"],
    "rxrx1": ["dinov2", "dinov3", "mae", "siglip", "bioclip", "openphenom"],
}

# to filter accordingly for compositional generalization
MARGINAL_UNSEEN_COMBOS = {
    "celeba": {
        (1, 1, 1, 1),
        (1, 1, 1, 0),
        (1, 1, 0, 1),
        (1, 0, 1, 1),
        (0, 1, 1, 1),
        (1, 1, 0, 0),
        (1, 0, 1, 0),
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        (0, 1, 0, 1),
        (0, 0, 1, 1),
    },
    "rxrx1": {
        (1, 1138),
        (1, 1109),
        (1, 1116),
        (1, 1115),
        (1, 1108),
        (1, 1137),
        (1, 1134),
        (1, 1111),
        (1, 1117),
        (1, 1126),
        (1, 1129),
        (1, 1125),
        (1, 1121),
        (1, 1136),
        (1, 1135),
        (1, 1124),
        (1, 1128),
        (1, 1113),
        (1, 1131),
        (1, 1123),
        (1, 1118),
        (1, 1110),
        (1, 1122),
        (1, 1127),
        (1, 1133),
        (1, 1130),
        (1, 1119),
        (1, 1120),
        (1, 1132),
        (1, 1112),
        (1, 1114),
        (2, 1138),
        (0, 1138),
        (0, 1115),
        (0, 1110),
        (0, 1133),
        (0, 1112),
        (0, 1113),
        (0, 1114),
        (0, 1125),
        (0, 1127),
        (0, 1126),
        (0, 1128),
        (0, 1129),
        (0, 1130),
        (2, 1133),
        (0, 1131),
        (0, 1132),
        (0, 1108),
        (0, 1116),
        (0, 1117),
        (0, 1123),
        (0, 1122),
        (0, 1121),
        (0, 1120),
        (0, 1119),
        (2, 1128),
        (2, 1127),
        (2, 1126),
        (0, 1124),
        (0, 1109),
        (2, 1110),
        (2, 1108),
        (2, 1131),
        (2, 1130),
        (2, 1109),
        (2, 1129),
        (2, 1120),
        (2, 1125),
        (2, 1116),
        (2, 1117),
        (2, 1122),
        (2, 1123),
        (2, 1112),
        (0, 1111),
        (2, 1118),
        (2, 1113),
        (2, 1121),
        (2, 1124),
        (2, 1137),
        (2, 1136),
        (2, 1135),
        (2, 1115),
        (2, 1132),
        (0, 1137),
        (0, 1134),
        (0, 1136),
        (0, 1135),
        (2, 1134),
        (2, 1119),
        (2, 1111),
        (2, 1114),
        (0, 1118),
        (1, 363),
        (1, 364),
        (1, 349),
        (1, 350),
        (1, 351),
        (1, 352),
        (1, 401),
    },
}
# ============================================================================
# Helper functions
# ============================================================================


def load_features(path: Path) -> Tuple[torch.Tensor, Dict]:
    """Load features and metadata from a .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    features = data["features"]
    metadata = data.get("metadata", {})
    return features, metadata


def get_condition_key(metadata: Dict, keys: List[str], idx: int) -> tuple:
    """Get condition tuple for a sample."""
    return tuple(
        int(
            metadata[k][idx].item()
            if isinstance(metadata[k][idx], torch.Tensor)
            else metadata[k][idx]
        )
        for k in keys
    )


def filter_real_by_model(
    dataset: str,
    model: str,
    real_feats: torch.Tensor,
    real_meta: Dict,
    condition_keys: List[str],
) -> Tuple[torch.Tensor, Dict]:
    """
    For marginalmodel, restrict real train stats to the model-available training distribution:
    keep samples whose joint condition is NOT in the held-out (unseen) set.
    """
    if model != "marginalmodel":
        return real_feats, real_meta

    unseen = MARGINAL_UNSEEN_COMBOS.get(dataset, set())
    if not unseen:
        return real_feats, real_meta

    N = len(real_feats)
    keep_list = []
    for i in range(N):
        cond = get_condition_key(real_meta, condition_keys, i)
        keep_list.append(cond not in unseen)

    keep = torch.tensor(keep_list, dtype=torch.bool)

    real_feats_f = real_feats[keep]
    real_meta_f = {}
    for k, v in real_meta.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == N:
            real_meta_f[k] = v[keep]
        elif isinstance(v, list) and len(v) == N:
            real_meta_f[k] = [vv for vv, kk in zip(v, keep_list) if kk]
        else:
            real_meta_f[k] = v

    return real_feats_f, real_meta_f


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    """L2 normalize features."""
    return F.normalize(x, p=2, dim=1)


def compute_mahalanobis(
    x: torch.Tensor, mu: torch.Tensor, precision: torch.Tensor
) -> torch.Tensor:
    """Compute Mahalanobis distance for a batch of samples."""
    centered = x - mu.unsqueeze(0)
    term1 = torch.matmul(centered, precision)
    return torch.sum(term1 * centered, dim=1)


def fit_global_stats(
    features: torch.Tensor,
    regularization: float = 1e-5,
    min_samples: int = 2,
) -> Dict[str, Any]:
    """Fit a single Gaussian (mu, precision) on all (filtered) real features."""
    features = normalize_features(features)
    N, D = features.shape

    mu = features.mean(dim=0)

    if N >= min_samples:
        feats_np = features.numpy()
        lw = LedoitWolf()
        try:
            cov_np = lw.fit(feats_np).covariance_
            cov = torch.from_numpy(cov_np).float()
            shrinkage = lw.shrinkage_
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
        "shrinkage": float(shrinkage),
    }


def zscore(x: np.ndarray, mean: float, std: float, eps: float = 1e-12) -> np.ndarray:
    return (x - mean) / (std + eps)


def compute_real_calibration_for_global_energy(
    real_feats: torch.Tensor,
    global_stats: Dict[str, Any],
    batch_size: int = 2000,
) -> Tuple[float, float]:
    """Compute mean/std of global Mahalanobis energy on real data (for z-scoring)."""
    real_feats = normalize_features(real_feats)
    N = len(real_feats)

    mu = global_stats["mu"]
    P = global_stats["precision"]

    vals = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        e = compute_mahalanobis(real_feats[start:end], mu, P)
        vals.append(e.numpy())
    e_all = np.concatenate(vals, axis=0)
    return float(np.mean(e_all)), float(
        np.std(e_all, ddof=1) if len(e_all) > 1 else 1.0
    )


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
    over real samples. Used for z-scoring margins on generated samples.
    """
    real_feats = normalize_features(real_feats)
    N = len(real_feats)

    calib = {}
    for attr_key in condition_keys:
        values = sorted(factorized_stats[attr_key].keys())
        V = len(values)
        mus = torch.stack(
            [factorized_stats[attr_key][v]["mu"] for v in values]
        )  # (V, D)
        Ps = torch.stack(
            [factorized_stats[attr_key][v]["precision"] for v in values]
        )  # (V, D, D)

        # true index for each real sample
        true_idx = np.zeros(N, dtype=int)
        for i in range(N):
            val = int(
                real_meta[attr_key][i].item()
                if isinstance(real_meta[attr_key][i], torch.Tensor)
                else real_meta[attr_key][i]
            )
            true_idx[i] = values.index(val) if val in values else -1

        margins = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            B = end - start
            x = real_feats[start:end]  # (B, D)

            centered = x.unsqueeze(1) - mus.unsqueeze(0)  # (B, V, D)
            term1 = torch.einsum("bvd,vde->bve", centered, Ps)  # (B, V, D)
            dists = torch.sum(term1 * centered, dim=2).numpy()  # (B, V)

            for bi, gi in enumerate(range(start, end)):
                ti = true_idx[gi]
                if ti < 0 or V <= 1:
                    continue
                true_d = dists[bi, ti]
                # min over other values
                mask = np.ones(V, dtype=bool)
                mask[ti] = False
                min_other = dists[bi, mask].min()
                margins.append(true_d - min_other)

        m = np.array(margins, dtype=np.float64)
        if len(m) == 0:
            calib[attr_key] = (0.0, 1.0)
        else:
            calib[attr_key] = (
                float(m.mean()),
                float(m.std(ddof=1) if len(m) > 1 else 1.0),
            )
    return calib


def compute_global_realism_z(
    gen_features: torch.Tensor,
    global_stats: Dict[str, Any],
    real_E_mean: float,
    real_E_std: float,
    batch_size: int = 2000,
) -> np.ndarray:
    """R(x) = z( E_global(x) ) using real calibration."""
    gen_features = normalize_features(gen_features)
    N = len(gen_features)

    mu = global_stats["mu"]
    P = global_stats["precision"]

    out = np.zeros(N, dtype=np.float64)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        e = compute_mahalanobis(gen_features[start:end], mu, P).numpy()
        out[start:end] = zscore(e, real_E_mean, real_E_std)
    return out


def compute_factorized_faithfulness_margin_z(
    gen_features: torch.Tensor,
    gen_metadata: Dict,
    factorized_stats: Dict[str, Dict[int, Dict[str, Any]]],
    condition_keys: List[str],
    margin_calib: Dict[str, Tuple[float, float]],
    batch_size: int = 1000,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    For each sample:
      margin_k(x) = E_k(x; true) - min_{v!=true} E_k(x; v)
      zmargin_k(x) = z( margin_k(x) ) using real calibration
    Aggregate faithfulness:
      F(x) = (1/K) * sum_k zmargin_k(x)
    Returns:
      F (N,), and per-attr margin arrays (raw margins).
    """
    gen_features = normalize_features(gen_features)
    N = len(gen_features)
    K = len(condition_keys)

    zsum = np.zeros(N, dtype=np.float64)
    per_attr_margin = {}

    for attr_key in condition_keys:
        values = sorted(factorized_stats[attr_key].keys())
        V = len(values)

        mus = torch.stack(
            [factorized_stats[attr_key][v]["mu"] for v in values]
        )  # (V, D)
        Ps = torch.stack(
            [factorized_stats[attr_key][v]["precision"] for v in values]
        )  # (V, D, D)

        # true index for each gen sample
        true_idx = np.zeros(N, dtype=int)
        for i in range(N):
            val = int(
                gen_metadata[attr_key][i].item()
                if isinstance(gen_metadata[attr_key][i], torch.Tensor)
                else gen_metadata[attr_key][i]
            )
            true_idx[i] = values.index(val) if val in values else -1

        margins = np.zeros(N, dtype=np.float64)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x = gen_features[start:end]  # (B, D)

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
                margins[gi] = true_d - min_other

        m_mean, m_std = margin_calib.get(attr_key, (0.0, 1.0))
        zsum += zscore(margins, m_mean, m_std)
        per_attr_margin[attr_key] = margins

    F = zsum / max(1, K)
    return F, per_attr_margin


# ============================================================================
# Factorized Trust Score Computation (per-attribute decomposition)
# ============================================================================


def fit_factorized_stats(
    features: torch.Tensor,
    metadata: Dict,
    condition_keys: List[str],
    regularization: float = 1e-5,
    min_samples: int = 2,
    use_shared_cov: bool = False,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    features = normalize_features(features)
    N, D = features.shape

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

        shared_precision = None
        if use_shared_cov and len(values) > 100:
            lw = LedoitWolf()
            try:
                cov_np = lw.fit(features.numpy()).covariance_
                cov = torch.from_numpy(cov_np).float()
                cov_reg = cov + regularization * torch.eye(D)
                L = torch.linalg.cholesky(cov_reg)
                shared_precision = torch.cholesky_inverse(L)
            except Exception:
                shared_precision = torch.eye(D) / (regularization + 1e-3)

        for val in tqdm(values, desc=f"Fitting {attr_key}", leave=False):
            idx = attr_groups[attr_key][val]
            feats = features[idx]
            n = len(idx)
            mu = feats.mean(dim=0)

            if shared_precision is not None:
                precision = shared_precision
            elif n >= min_samples:
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

            stats[attr_key][val] = {"mu": mu, "precision": precision, "n_samples": n}
    return stats


# ============================================================================
# Alaa et al. Metrics (α-precision, Authenticity)
# ============================================================================


def compute_alpha_precision_scores(
    gen_features: torch.Tensor,
    real_features: torch.Tensor,
    alphas: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """
    Compute α-precision curve in a fixed feature space (your pretrained encoder manifold).

    Let c_r be the real center and d_r(i)=||r_i - c_r||.
    For each alpha, define radius r_alpha as the alpha-quantile of d_r.
    Then P_alpha = mean_j 1{ ||g_j - c_r|| <= r_alpha }.

    Returns a dict containing:
      - alphas: (A,)
      - radii: (A,) radii r_alpha
      - P_alpha: (A,) alpha-precision curve values
      - gen_center_dist: (N_gen,) distances ||g - c_r||
      - real_center_dist: (N_real,) distances ||r - c_r||
    """
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 10, dtype=np.float64)
    else:
        alphas = np.asarray(alphas, dtype=np.float64)

    # Normalize consistently with the rest of your pipeline
    gen = normalize_features(gen_features).cpu().numpy().astype(np.float64)
    real = normalize_features(real_features).cpu().numpy().astype(np.float64)

    # Real center in feature space
    center = real.mean(axis=0)

    # Distances to real center
    real_d = np.linalg.norm(real - center[None, :], axis=1)
    gen_d = np.linalg.norm(gen - center[None, :], axis=1)

    # Radii r_alpha = quantiles of real distances
    # (np.quantile handles alpha endpoints; r_0=min, r_1=max)
    radii = np.quantile(real_d, alphas)

    # P_alpha curve: fraction of generated samples inside each radius
    # Vectorized: (N_gen, 1) <= (1, A) -> (N_gen, A)
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
    betas: np.ndarray | None = None,
    k: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Compute β-recall curve in a fixed feature space.

    Steps:
      1) Compute per-real local radius tau_i = distance to k-th NN in real set (excluding itself).
      2) For each beta:
         - Define typical generated subset G^beta by radius r_beta around generated center
           where r_beta is beta-quantile of ||g - c_g||.
         - For each real r_i, compute delta_i^beta = min_{g in G^beta} ||r_i - g||.
         - Mark covered if delta_i^beta <= tau_i.
         - R_beta = mean_i covered_i.

    Returns dict:
      - betas: (B,)
      - radii: (B,) radii r_beta for generated typical sets
      - R_beta: (B,) beta-recall curve values
      - real_knn_radius: (N_real,) tau_i
      - gen_center_dist: (N_gen,) ||g - c_g||
    """
    if betas is None:
        betas = np.linspace(0.0, 1.0, 10, dtype=np.float64)
    else:
        betas = np.asarray(betas, dtype=np.float64)

    # Normalize consistently with the rest of your pipeline
    gen = normalize_features(gen_features).cpu().numpy().astype(np.float64)
    real = normalize_features(real_features).cpu().numpy().astype(np.float64)

    n_real = real.shape[0]
    n_gen = gen.shape[0]
    if n_real < (k + 1):
        raise ValueError(
            f"Need at least k+1 real samples for kNN (got n_real={n_real}, k={k})"
        )
    if n_gen < 1:
        raise ValueError("Need at least 1 generated sample")

    # 1) Real local radii tau_i via kNN in real set (exclude self)
    # n_neighbors = k+1 because first neighbor is itself at distance 0
    nn_real = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn_real.fit(real)
    dists_real, _ = nn_real.kneighbors(real, return_distance=True)
    tau = dists_real[:, k].astype(
        np.float64
    )  # distance to k-th neighbor excluding self

    # 2) Generated typical sets by beta via center+quantile radius
    c_g = gen.mean(axis=0)
    gen_center_dist = np.linalg.norm(gen - c_g[None, :], axis=1)
    radii = np.quantile(gen_center_dist, betas).astype(np.float64)

    R_beta = np.zeros_like(betas, dtype=np.float64)

    # We will build NN indices on the typical subset per beta.
    # For small n_gen (CelebA) this is fine. For large n_gen, this is still OK for 101 betas but costs time.
    # If it becomes heavy, we can cache sorted indices by gen_center_dist and incrementally grow the subset.
    for t, r_beta in enumerate(radii):
        mask = gen_center_dist <= r_beta
        idx = np.where(mask)[0]

        # Edge cases: beta=0 can yield 1 point; still define coverage wrt that point.
        if idx.size == 0:
            R_beta[t] = 0.0
            continue

        gen_typ = gen[idx]

        nn_gen = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn_gen.fit(gen_typ)

        # Find nearest typical generated for each real point
        d_rg, _ = nn_gen.kneighbors(real, return_distance=True)
        delta = d_rg[:, 0].astype(np.float64)

        covered = delta <= tau
        R_beta[t] = covered.mean()

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
    """
    Compute Alaa et al. authenticity scores.

    For each generated sample x:
    - Find nearest real sample r
    - Find r's nearest neighbor among OTHER reals
    - If d(x,r) < d(r, nn(r)), sample is "unauthentic" (too close = memorized)

    Returns: binary authenticity (1 = authentic, 0 = potentially memorized)
    """
    gen_features = normalize_features(gen_features).numpy()
    real_features = normalize_features(real_features).numpy()

    # Build kNN index on real features
    nn_real = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn_real.fit(real_features)

    # For each real sample, find distance to its nearest neighbor
    distances_real, _ = nn_real.kneighbors(real_features)
    # distances_real[:, 0] is distance to self (0), [:, 1] is nearest neighbor
    real_nn_distances = distances_real[:, 1]

    # For each generated sample, find nearest real
    nn_gen = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn_gen.fit(real_features)
    distances_gen, indices_gen = nn_gen.kneighbors(gen_features)
    gen_to_real_distances = distances_gen[:, 0]
    nearest_real_indices = indices_gen[:, 0]

    # Authenticity: is d(gen, nearest_real) >= d(nearest_real, its_nn)?
    # If gen is farther from real than real's neighbors are, it's authentic
    authenticity = (
        gen_to_real_distances >= real_nn_distances[nearest_real_indices]
    ).astype(float)

    return authenticity


# ============================================================================
# Main
# ============================================================================


def process_single_config(
    dataset: str,
    model: str,
    encoder: str,
    condition_keys: List[str],
) -> Dict:
    """Process a single dataset/model/encoder configuration."""

    # Paths - using outputs/ directory structure
    real_path = Path(f"outputs/real_{dataset}_{encoder}/train_features.pt")

    # Map model name to generated directory
    gen_dir = GEN_PATH_MAP.get((dataset, model))
    if gen_dir is None:
        print(f"  Skip: no path mapping for ({dataset}, {model})")
        return None
    gen_path = Path(f"outputs/gen/{gen_dir}/{encoder}_features.pt")

    if not real_path.exists():
        print(f"  Skip: real features not found at {real_path}")
        return None

    if not gen_path.exists():
        print(f"  Skip: generated features not found at {gen_path}")
        return None

    print(f"\n=== {dataset} / {model} / {encoder} ===")

    # Load features
    print("  Loading features...")
    real_feats, real_meta = load_features(real_path)
    gen_feats, gen_meta = load_features(gen_path)
    print(f"  Real: {real_feats.shape}, Generated: {gen_feats.shape}")

    real_feats, real_meta = filter_real_by_model(
        dataset, model, real_feats, real_meta, condition_keys
    )
    print(f"  Real used for stats (after holdout filter): {len(real_feats)}")

    # Fit factorized (per-attribute) statistics
    # Use shared covariance for attributes with many values (e.g., sirna)
    print("  Fitting factorized (per-attribute) statistics...")
    use_shared_cov = dataset == "rxrx1"  # Enable for RxRx1 to handle 1138 sirnas
    factorized_stats = fit_factorized_stats(
        real_feats,
        real_meta,
        condition_keys,
        regularization=1e-5,
        use_shared_cov=use_shared_cov,
    )
    for k, v in factorized_stats.items():
        print(f"    {k}: {len(v)} values")

    print("  Fitting global (unconditional) Gaussian for realism...")
    global_stats = fit_global_stats(real_feats, regularization=1e-5)
    real_E_mean, real_E_std = compute_real_calibration_for_global_energy(
        real_feats, global_stats
    )

    print("  Calibrating factorized margins on real data (for z-scoring)...")
    margin_calib = compute_real_calibration_for_factorized_margins(
        real_feats, real_meta, factorized_stats, condition_keys
    )

    print("  Computing updated realism R_z (global) ...")
    realism_global_z = compute_global_realism_z(
        gen_feats, global_stats, real_E_mean, real_E_std
    )

    print("  Computing updated faithfulness F_z (factorized margin) ...")
    faithfulness_margin_z, per_attr_margins = compute_factorized_faithfulness_margin_z(
        gen_feats, gen_meta, factorized_stats, condition_keys, margin_calib
    )

    trust_updated = realism_global_z + faithfulness_margin_z

    print(
        f"  Updated realism R_z: mean={np.mean(realism_global_z):.4f}, std={np.std(realism_global_z):.4f}"
    )
    print(
        f"  Updated faithfulness F_z: mean={np.mean(faithfulness_margin_z):.4f}, std={np.std(faithfulness_margin_z):.4f}"
    )
    print(
        f"  Updated trust T=R+F: mean={np.mean(trust_updated):.4f}, std={np.std(trust_updated):.4f}"
    )

    # Compute Alaa et al. metrics
    print("  Computing α-precision scores...")
    alpha_res = compute_alpha_precision_scores(
        gen_feats, real_feats, alphas=np.linspace(0, 1, 11)
    )

    print("  Computing β-recall curve...")
    beta_res = compute_beta_recall_scores(
        gen_feats, real_feats, betas=np.linspace(0, 1, 11), k=5
    )

    print("  Computing authenticity scores...")
    authenticity = compute_authenticity_scores(gen_feats, real_feats)

    # Get true conditions for each sample
    true_conditions = [
        get_condition_key(gen_meta, condition_keys, i) for i in range(len(gen_feats))
    ]

    # Package results
    results = {
        "dataset": dataset,
        "model": model,
        "encoder": encoder,
        "n_samples": len(gen_feats),
        "n_real_used_for_stats": int(len(real_feats)),
        # Alaa et al. metrics
        # alpha precision
        "alpha_grid": alpha_res["alphas"],
        "P_alpha": alpha_res["P_alpha"],
        "gen_center_dist_real": alpha_res[
            "gen_center_dist"
        ],  # for per-alpha correlations later
        "real_center_dist_real": alpha_res["real_center_dist"],
        "alpha_radii": alpha_res["radii"],
        # beta recall
        "beta_grid": beta_res["betas"],
        "R_beta": beta_res["R_beta"],
        "beta_radii": beta_res["radii"],
        "real_knn_radius": beta_res["real_knn_radius"],
        "gen_center_dist_gen": beta_res["gen_center_dist"],
        # authenticity
        "authenticity": authenticity,
        # Metadata
        "true_conditions": true_conditions,
        # Updated baseline scores (your new definition)
        "realism_global_z": realism_global_z,  # R(x)
        "faithfulness_margin_z": faithfulness_margin_z,  # F(x)
        "trust_updated": trust_updated,  # T(x)=R+F
        "global_stats_summary": {
            "n_samples": global_stats["n_samples"],
            "shrinkage": global_stats["shrinkage"],
            "real_E_mean": real_E_mean,
            "real_E_std": real_E_std,
        },
        "margin_calib": margin_calib,  # per-attr (mean,std) for margins
    }

    return results


def run_analysis(dataset: str) -> List[Dict]:
    """Run trust score computation for all models/encoders."""

    condition_keys = CONDITION_ATTRS.get(dataset, [])
    if not condition_keys:
        print(f"No condition attributes defined for {dataset}")
        return []

    encoders = ENCODERS.get(dataset, [])
    models = ["fullmodel", "marginalmodel"]

    all_results = []

    for model in models:
        for encoder in encoders:
            result = process_single_config(dataset, model, encoder, condition_keys)
            if result is not None:
                all_results.append(result)

    return all_results


def save_results(results: List[Dict], dataset: str, output_dir: Path):
    """Save results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    output_path = output_dir / f"trust_scores_{dataset}.pt"
    torch.save(results, output_path)
    print(f"\nSaved results to {output_path}")

    # Create summary CSV
    summary_rows = []
    for r in results:
        row = {
            "dataset": r["dataset"],
            "model": r["model"],
            "encoder": r["encoder"],
            "n_samples": r["n_samples"],
            "realism_global_z_mean": np.nanmean(r["realism_global_z"]),
            "realism_global_z_std": np.nanstd(r["realism_global_z"]),
            "faithfulness_margin_z_mean": np.nanmean(r["faithfulness_margin_z"]),
            "faithfulness_margin_z_std": np.nanstd(r["faithfulness_margin_z"]),
            "trust_updated_mean": np.nanmean(r["trust_updated"]),
            "trust_updated_std": np.nanstd(r["trust_updated"]),
            # Alaa et al.
            "authenticity_mean": np.mean(r["authenticity"]),
            "alpha_precision_mean": float(np.mean(r["P_alpha"])),
            "beta_recall_mean": float(np.mean(r["R_beta"])),
        }
        summary_rows.append(row)

    import pandas as pd

    df = pd.DataFrame(summary_rows)
    csv_path = output_dir / f"trust_scores_summary_{dataset}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="celeba", choices=["celeba", "rxrx1"]
    )
    parser.add_argument("--output-dir", type=str, default="outputs/trust_scores")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("TRUST SCORE COMPUTATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Condition attributes: {CONDITION_ATTRS.get(args.dataset, [])}")

    results = run_analysis(args.dataset)

    if results:
        save_results(results, args.dataset, output_dir)

    print("\n" + "=" * 60)
    print("COMPUTATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
