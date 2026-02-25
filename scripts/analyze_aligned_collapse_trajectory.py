#!/usr/bin/env python3
"""
Timestep-wise Aligned Feature Distribution Analysis.

Generates samples while capturing aligned features at each timestep during denoising,
then analyzes distribution collapse metrics across the trajectory.

This extends h2_aligned_distribution.py by tracking HOW the distribution evolves:
- At t=1.0: aligned features start from noise projections
- At t=0.04 (t_cutoff): aligned features should match final DINO features

Metrics computed per timestep:
- Angular concentration (R = mean resultant length)
- Cosine to real mean direction (global shift)
- Pairwise cosine spread (within-distribution diversity)
- kNN radius to real (local support coverage)
- Mahalanobis energy statistics (typicality)

Usage:
    PYTHONPATH=src uv run python scripts/analyze_aligned_collapse_trajectory.py \
        --checkpoint-key celeba_repa_full_v1 \
        --output-dir outputs/collapse_trajectory \
        --samples-per-condition 50

    # For marginal models (separate seen/unseen analysis)
    PYTHONPATH=src uv run python scripts/analyze_aligned_collapse_trajectory.py \
        --checkpoint-key celeba_repa_marginal_v1 \
        --output-dir outputs/collapse_trajectory \
        --marginal \
        --samples-per-condition 50
"""

import argparse
import itertools
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from faithful_cond_gen.model.generator import GeneratorWrapper
from faithful_cond_gen.model.repa_encoder import REPAEncoder
from faithful_cond_gen.pl_modules.generator_pl import GeneratorPL
from faithful_cond_gen.utils.checkpoints import get_checkpoint_path
from hydra.utils import instantiate
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

CONDITION_KEYS = ["Blond_Hair", "Eyeglasses", "Male", "Smiling"]  # Sorted alphabetically

# Marginal seen combos (Hamming weight <= 1)
MARGINAL_SEEN_COMBOS = {
    (0, 0, 0, 0),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
}


# =============================================================================
# Model Loading
# =============================================================================


def load_repa_model(checkpoint_key: str, device: str = "cuda"):
    """Load REPA model from checkpoint using proper Hydra configs."""
    log.info(f"Loading model from checkpoint key: {checkpoint_key}")
    ckpt_path = get_checkpoint_path(checkpoint_key)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    log.info(f"Checkpoint path: {ckpt_path}")

    # Load config
    config_root = Path(__file__).parent.parent / "configs"

    if "celeba" in checkpoint_key.lower():
        model_cfg_path = config_root / "model" / "generator_celeba.yaml"
    elif "rxrx1" in checkpoint_key.lower():
        model_cfg_path = config_root / "model" / "generator_rxrx1.yaml"
    else:
        raise ValueError(f"Cannot determine dataset from checkpoint key: {checkpoint_key}")

    log.info(f"Loading generator config from: {model_cfg_path}")
    gen_cfg_dict = OmegaConf.load(model_cfg_path)

    # Enable REPA for REPA checkpoints
    if "repa" in checkpoint_key.lower():
        gen_cfg_dict.use_repa = True
        if gen_cfg_dict.repa_proj_coeff == 0.0:
            gen_cfg_dict.repa_proj_coeff = 0.5

    gen_cfg_obj = instantiate(gen_cfg_dict)
    generator = GeneratorWrapper(gen_cfg_obj)

    pl_module = GeneratorPL.load_from_checkpoint(
        ckpt_path,
        generator=generator,
        map_location=device,
        strict=False,
    )
    pl_module.to(device)
    pl_module.eval()

    if hasattr(pl_module, "ema"):
        log.info("Applying EMA weights...")
        pl_module.ema.apply()

    if not pl_module.generator.cfg.use_repa:
        raise ValueError("Model does not have REPA enabled!")

    return pl_module, gen_cfg_obj


def load_dino_encoder(encoder_name: str = "dinov3-vit-l", device: str = "cuda"):
    """Load DINO encoder for final feature extraction."""
    log.info(f"Loading DINO encoder: {encoder_name}")
    encoder = REPAEncoder(
        encoder_name=encoder_name,
        resolution=256,
        in_channels=3,
        target_grid=16,
        device=device,
    )
    encoder.to(device)
    encoder.eval()
    return encoder


def load_real_features(dataset: str = "celeba") -> Tuple[torch.Tensor, Dict]:
    """Load cached real features."""
    real_path = Path(f"outputs/real_{dataset}_dinov3_meanpatch/train_features.pt")
    if not real_path.exists():
        raise FileNotFoundError(f"Real features not found: {real_path}")

    data = torch.load(real_path, map_location="cpu", weights_only=False)
    features = data["features"]
    metadata = data.get("metadata", {})
    log.info(f"Loaded real features: {features.shape}")
    return features, metadata


# =============================================================================
# Condition Utilities
# =============================================================================


def is_seen_condition(cond_tuple: Tuple[int, ...]) -> bool:
    """Check if condition is 'seen' based on Hamming weight <= 1."""
    return sum(cond_tuple) <= 1


def get_celeba_conditions(is_marginal: bool = False) -> Tuple[List[Tuple], Dict]:
    """Get CelebA conditioning combinations."""
    attrs = sorted(CONDITION_KEYS)
    all_combos = list(itertools.product([0, 1], repeat=len(attrs)))

    metadata = {}
    for combo in all_combos:
        if is_marginal:
            metadata[combo] = "seen" if is_seen_condition(combo) else "unseen"
        else:
            metadata[combo] = "all"

    return all_combos, metadata


def format_condition(cond: Tuple[int, ...], keys: List[str] = CONDITION_KEYS) -> str:
    """Format condition tuple as string."""
    return "_".join(f"{k}{v}" for k, v in zip(keys, cond))


def get_condition_tuple(meta: Dict, idx: int, keys: List[str]) -> Tuple[int, ...]:
    """Extract condition tuple from metadata."""
    return tuple(
        int(meta[k][idx].item() if isinstance(meta[k][idx], torch.Tensor) else meta[k][idx])
        for k in keys
    )


def build_condition_index(meta: Dict, n_samples: int, keys: List[str]) -> Dict[Tuple, List[int]]:
    """Build index mapping condition -> sample indices."""
    by_cond = {}
    for i in range(n_samples):
        cond = get_condition_tuple(meta, i, keys)
        by_cond.setdefault(cond, []).append(i)
    return by_cond


# =============================================================================
# Feature Extraction
# =============================================================================


@torch.no_grad()
def extract_dino_features(images: torch.Tensor, dino_encoder) -> torch.Tensor:
    """Extract DINO features from images.

    Args:
        images: (B, 3, H, W) in [0, 1]
        dino_encoder: REPAEncoder instance

    Returns:
        features: (B, D) pooled DINO features
    """
    images = images.clamp(0.0, 1.0)
    patch_features = dino_encoder(images)  # (B, num_patches, D)
    pooled_features = patch_features.mean(dim=1)  # (B, D)
    return pooled_features


# =============================================================================
# Distribution Metrics (from h2_aligned_distribution.py)
# =============================================================================


def l2_normalize_np(X: np.ndarray) -> np.ndarray:
    """L2 normalize numpy array along axis=1."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-12)


def compute_mean_resultant_length(X_unit: np.ndarray) -> float:
    """Compute R = ||mean(x_i)||. Higher R = more concentrated."""
    mean_vec = X_unit.mean(axis=0)
    return float(np.linalg.norm(mean_vec))


def compute_cos_to_mean_stats(X_unit: np.ndarray, mean_dir: np.ndarray) -> Dict[str, float]:
    """Compute cosine similarity to a reference direction."""
    cos_vals = X_unit @ mean_dir
    return {
        "mean": float(np.mean(cos_vals)),
        "std": float(np.std(cos_vals)),
        "q05": float(np.percentile(cos_vals, 5)),
        "q50": float(np.percentile(cos_vals, 50)),
        "q95": float(np.percentile(cos_vals, 95)),
    }


def compute_pairwise_cosine_stats(
    X_unit: np.ndarray, n_pairs: int, rng: np.random.Generator
) -> Dict[str, float]:
    """Sample random pairs and compute pairwise cosine similarity distribution."""
    n = len(X_unit)
    if n < 2:
        return {"mean": np.nan, "std": np.nan, "q05": np.nan, "q50": np.nan, "q95": np.nan}

    idx1 = rng.integers(0, n, size=n_pairs)
    idx2 = rng.integers(0, n, size=n_pairs)
    same = idx1 == idx2
    idx2[same] = (idx2[same] + 1) % n

    cos_vals = np.sum(X_unit[idx1] * X_unit[idx2], axis=1)
    return {
        "mean": float(np.mean(cos_vals)),
        "std": float(np.std(cos_vals)),
        "q05": float(np.percentile(cos_vals, 5)),
        "q50": float(np.percentile(cos_vals, 50)),
        "q95": float(np.percentile(cos_vals, 95)),
    }


def compute_knn_radius_stats(
    query_unit: np.ndarray,
    ref_unit: np.ndarray,
    k: int,
    leave_one_out: bool = False,
    max_ref: int = 10000,
    max_query: int = 5000,
    rng: np.random.Generator = None,
) -> Dict[str, float]:
    """Compute kNN radius to reference set (cosine distance)."""
    if rng is None:
        rng = np.random.default_rng(42)

    if len(ref_unit) > max_ref and not leave_one_out:
        ref_idx = rng.choice(len(ref_unit), max_ref, replace=False)
        ref_unit = ref_unit[ref_idx]

    if len(query_unit) > max_query:
        if leave_one_out:
            query_idx = rng.choice(len(query_unit), max_query, replace=False)
            query_unit = query_unit[query_idx]
            ref_unit = query_unit
        else:
            query_idx = rng.choice(len(query_unit), max_query, replace=False)
            query_unit = query_unit[query_idx]

    if leave_one_out:
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="auto")
        nn.fit(ref_unit)
        distances, _ = nn.kneighbors(query_unit)
        kth_dist = distances[:, k]
    else:
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="auto")
        nn.fit(ref_unit)
        distances, _ = nn.kneighbors(query_unit)
        kth_dist = distances[:, k - 1]

    return {
        "mean": float(np.mean(kth_dist)),
        "q05": float(np.percentile(kth_dist, 5)),
        "q50": float(np.percentile(kth_dist, 50)),
        "q95": float(np.percentile(kth_dist, 95)),
    }


def normalize_features_torch(features: torch.Tensor) -> torch.Tensor:
    """L2 normalize features."""
    return features / (features.norm(dim=1, keepdim=True) + 1e-12)


def compute_mahalanobis(x: torch.Tensor, mu: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
    """Compute Mahalanobis distance."""
    centered = x - mu.unsqueeze(0)
    term = torch.einsum("bd,de->be", centered, precision)
    dist = torch.sum(term * centered, dim=1)
    return dist


def fit_global_stats(features: torch.Tensor, regularization: float = 1e-5) -> Dict:
    """Fit global Gaussian (mu, precision) on features."""
    features = normalize_features_torch(features)
    N, D = features.shape

    mu = features.mean(dim=0)

    if N >= 2:
        feats_np = features.numpy()
        lw = LedoitWolf()
        try:
            cov_np = lw.fit(feats_np).covariance_
            cov = torch.from_numpy(cov_np).float()
        except Exception:
            cov = torch.eye(D)
    else:
        cov = torch.eye(D)

    cov_reg = cov + regularization * torch.eye(D)
    try:
        L = torch.linalg.cholesky(cov_reg)
        precision = torch.cholesky_inverse(L)
    except RuntimeError:
        precision = torch.linalg.pinv(cov_reg)

    return {"mu": mu, "precision": precision, "n_samples": int(N)}


def compute_energy_stats(
    feats: torch.Tensor, global_stats: Dict
) -> Dict[str, float]:
    """Compute Mahalanobis energy statistics."""
    feats_norm = normalize_features_torch(feats)
    mu = global_stats["mu"]
    P = global_stats["precision"]
    energy = compute_mahalanobis(feats_norm, mu, P).numpy()

    return {
        "mean": float(np.mean(energy)),
        "std": float(np.std(energy)),
        "q05": float(np.percentile(energy, 5)),
        "q50": float(np.percentile(energy, 50)),
        "q95": float(np.percentile(energy, 95)),
    }


def compute_energy_aurocs(
    real_energy: np.ndarray,
    gen_energy: np.ndarray,
    real_E_mean: float,
    real_E_std: float,
) -> Dict[str, float]:
    """Compute AUROC for real vs gen using energy scores."""
    n_real = len(real_energy)
    n_gen = len(gen_energy)
    labels = np.concatenate([np.zeros(n_real), np.ones(n_gen)])

    real_z = (real_energy - real_E_mean) / (real_E_std + 1e-12)
    gen_z = (gen_energy - real_E_mean) / (real_E_std + 1e-12)

    all_z = np.concatenate([real_z, gen_z])
    all_absz = np.abs(all_z)
    all_sqz = all_z ** 2

    valid = np.isfinite(all_z)
    if valid.sum() < 10:
        return {"one_sided": np.nan, "two_sided_absz": np.nan, "two_sided_sqz": np.nan}

    return {
        "one_sided": float(roc_auc_score(labels[valid], all_z[valid])),
        "two_sided_absz": float(roc_auc_score(labels[valid], all_absz[valid])),
        "two_sided_sqz": float(roc_auc_score(labels[valid], all_sqz[valid])),
    }


# =============================================================================
# Timestep-wise Analysis
# =============================================================================


def analyze_timestep(
    aligned_np: np.ndarray,
    real_np: np.ndarray,
    gen_dino_np: np.ndarray,
    real_feats: torch.Tensor,
    aligned_feats: torch.Tensor,
    global_stats: Dict,
    real_E_mean: float,
    real_E_std: float,
    real_energy: np.ndarray,
    rng: np.random.Generator,
    k_idx: int,
    t_val: float,
    gen_dino_constant_stats: Dict,
    real_constant_stats: Dict,
    real_unit: np.ndarray,
    gen_dino_unit: np.ndarray,
    real_mean_dir: np.ndarray,
    n_pairwise_pairs: int = 20000,
) -> Dict[str, Any]:
    """Analyze distribution metrics for a single timestep.

    gen_dino and real stats that don't depend on aligned are passed as pre-computed constants.
    """
    n_aligned = len(aligned_np)

    # L2 normalize aligned (the only thing that changes per timestep)
    aligned_unit = l2_normalize_np(aligned_np)

    result = {
        "k_idx": k_idx,
        "t_val": t_val,
        "n_samples": n_aligned,
    }

    # 1) Angular concentration (R)
    result["R_aligned"] = compute_mean_resultant_length(aligned_unit)
    result["R_gen_dino"] = compute_mean_resultant_length(gen_dino_unit)
    result["R_real"] = compute_mean_resultant_length(real_unit)

    # 2) Cosine to real mean direction
    # Aligned varies per timestep
    stats = compute_cos_to_mean_stats(aligned_unit, real_mean_dir)
    for k, v in stats.items():
        result[f"cos_to_mean_aligned_{k}"] = v
    # Gen_dino and real are constant
    for name, X_unit in [("gen_dino", gen_dino_unit), ("real", real_unit)]:
        stats = compute_cos_to_mean_stats(X_unit, real_mean_dir)
        for k, v in stats.items():
            result[f"cos_to_mean_{name}_{k}"] = v

    # 3) Pairwise cosine spread
    n_pairs = min(n_pairwise_pairs, n_aligned * (n_aligned - 1) // 2)
    # Aligned varies per timestep
    stats = compute_pairwise_cosine_stats(aligned_unit, n_pairs, rng)
    for k, v in stats.items():
        result[f"pairwise_cos_aligned_{k}"] = v
    # Gen_dino and real are constant (but cheap to compute, so we still do it for consistency)
    for name, X_unit in [("gen_dino", gen_dino_unit), ("real", real_unit)]:
        stats = compute_pairwise_cosine_stats(X_unit, n_pairs, rng)
        for k, v in stats.items():
            result[f"pairwise_cos_{name}_{k}"] = v

    # 4) kNN radius to real
    if len(real_np) >= 20 and n_aligned >= 10:
        # Aligned -> Real (varies per timestep)
        stats = compute_knn_radius_stats(
            aligned_unit, real_unit, k=10, leave_one_out=False,
            max_ref=10000, max_query=5000, rng=rng
        )
        for stat_name, v in stats.items():
            result[f"knn10_aligned_{stat_name}"] = v

        # Gen_dino -> Real (CONSTANT - use pre-computed)
        for stat_name, v in gen_dino_constant_stats["knn10"].items():
            result[f"knn10_gen_dino_{stat_name}"] = v

        # Real -> Real baseline (CONSTANT - use pre-computed)
        for stat_name, v in real_constant_stats["knn10"].items():
            result[f"knn10_real_{stat_name}"] = v

    # 5) Mahalanobis energy statistics
    # Aligned varies per timestep
    aligned_energy_stats = compute_energy_stats(aligned_feats, global_stats)
    for k, v in aligned_energy_stats.items():
        result[f"energy_aligned_{k}"] = v

    # Gen_dino energy (CONSTANT - use pre-computed)
    for k, v in gen_dino_constant_stats["energy"].items():
        result[f"energy_gen_dino_{k}"] = v

    # 6) AUROC for real vs aligned/gen_dino
    # Aligned varies per timestep
    aligned_energy = compute_mahalanobis(
        normalize_features_torch(aligned_feats),
        global_stats["mu"],
        global_stats["precision"]
    ).numpy()

    aurocs_aligned = compute_energy_aurocs(real_energy, aligned_energy, real_E_mean, real_E_std)
    for k, v in aurocs_aligned.items():
        result[f"AUROC_{k}_aligned"] = v

    # Gen_dino AUROC (CONSTANT - use pre-computed)
    for k, v in gen_dino_constant_stats["aurocs"].items():
        result[f"AUROC_{k}_gen_dino"] = v

    # 7) Paired cosine: aligned vs gen_dino (same samples)
    paired_cos = np.sum(aligned_unit * gen_dino_unit, axis=1)
    result["paired_cos_aligned_gen_dino_mean"] = float(np.mean(paired_cos))
    result["paired_cos_aligned_gen_dino_std"] = float(np.std(paired_cos))

    return result


# =============================================================================
# Generation with Feature Capture
# =============================================================================


@torch.no_grad()
def generate_with_timestep_features(
    generator,
    dino_encoder,
    conditions: List[Tuple],
    samples_per_condition: int,
    batch_size: int,
    num_inference_steps: int,
    device: str,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, Dict]:
    """Generate samples while capturing aligned features at each timestep.

    Returns:
        aligned_by_timestep: Dict[k_idx] -> torch.Tensor (N_total, D)
        gen_dino_feats: (N_total, D) final DINO features
        gen_meta: metadata dict with condition keys
        timestep_values: Dict[k_idx] -> t_val
    """
    all_aligned_by_timestep = {}  # k_idx -> list of (B, D) tensors
    all_gen_dino = []
    timestep_values = {}  # k_idx -> t_val

    # Build metadata
    gen_meta = {k: [] for k in CONDITION_KEYS}

    for cond in tqdm(conditions, desc="Generating samples"):
        cond_tensor = torch.tensor(cond, device=device).long()

        num_batches = (samples_per_condition + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            current_bs = min(batch_size, samples_per_condition - batch_idx * batch_size)
            batch_cond = cond_tensor.unsqueeze(0).repeat(current_bs, 1)

            # Generate with feature capture at ALL steps (including t=0)
            images, aligned_by_idx = generator.sample(
                batch_cond,
                num_inference_steps=num_inference_steps,
                return_aligned_features=True,
                feature_capture_idx="all",
                capture_at_t0=True,
            )

            # Safety check
            images = images.clamp(0.0, 1.0)

            # Extract final DINO features
            final_dino = extract_dino_features(images, dino_encoder)
            all_gen_dino.append(final_dino.cpu())

            # Store aligned features by timestep
            for k_idx, t_val, pooled_feats in aligned_by_idx:
                if k_idx not in all_aligned_by_timestep:
                    all_aligned_by_timestep[k_idx] = []
                all_aligned_by_timestep[k_idx].append(pooled_feats)

                if k_idx not in timestep_values:
                    timestep_values[k_idx] = t_val

            # Store metadata
            for k, v in zip(CONDITION_KEYS, cond):
                gen_meta[k].extend([v] * current_bs)

    # Concatenate all
    gen_dino_feats = torch.cat(all_gen_dino, dim=0)

    aligned_by_timestep = {}
    for k_idx, feat_list in all_aligned_by_timestep.items():
        aligned_by_timestep[k_idx] = torch.cat(feat_list, dim=0)

    # Convert metadata to tensors
    for k in CONDITION_KEYS:
        gen_meta[k] = torch.tensor(gen_meta[k], dtype=torch.long)

    return aligned_by_timestep, gen_dino_feats, gen_meta, timestep_values


# =============================================================================
# Plotting
# =============================================================================


def plot_metric_trajectory(
    df: pd.DataFrame,
    metric_cols: List[str],
    ylabel: str,
    title: str,
    output_path: Path,
    include_real: bool = False,
    real_value: float = None,
):
    """Plot metric evolution across timesteps."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {"aligned": "C2", "gen_dino": "C1", "real": "C0"}
    labels = {"aligned": "Aligned (timestep)", "gen_dino": "Gen DINO (final)", "real": "Real"}

    timesteps = df["t_val"].values

    for col in metric_cols:
        # Parse source from column name
        if "_aligned_" in col or col.endswith("_aligned"):
            source = "aligned"
        elif "_gen_dino_" in col or col.endswith("_gen_dino"):
            source = "gen_dino"
        elif "_real_" in col or col.endswith("_real"):
            source = "real"
        else:
            continue

        values = df[col].values
        ax.plot(timesteps, values, color=colors.get(source, "gray"),
                label=labels.get(source, source), linewidth=2, marker="o", markersize=3)

    if include_real and real_value is not None:
        ax.axhline(y=real_value, color=colors["real"], linestyle="--",
                   label="Real (constant)", linewidth=1.5)

    ax.set_xlabel("Timestep t", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # t goes from 1.0 -> 0.04

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary_trajectories(df: pd.DataFrame, output_dir: Path):
    """Generate summary plots for key metrics."""

    # 1) Angular concentration R
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["t_val"], df["R_aligned"], "C2-o", label="Aligned", linewidth=2, markersize=4)
    ax.plot(df["t_val"], df["R_gen_dino"], "C1--", label="Gen DINO (const)", linewidth=2)
    ax.axhline(y=df["R_real"].iloc[0], color="C0", linestyle=":", label="Real", linewidth=2)
    ax.set_xlabel("Timestep t", fontsize=12)
    ax.set_ylabel("Mean Resultant Length R", fontsize=12)
    ax.set_title("Angular Concentration vs Timestep", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_R.png", dpi=150)
    plt.close()

    # 2) Cosine to real mean
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["t_val"], df["cos_to_mean_aligned_mean"], "C2-o", label="Aligned", linewidth=2, markersize=4)
    ax.plot(df["t_val"], df["cos_to_mean_gen_dino_mean"], "C1--", label="Gen DINO", linewidth=2)
    ax.axhline(y=df["cos_to_mean_real_mean"].iloc[0], color="C0", linestyle=":", label="Real", linewidth=2)
    ax.set_xlabel("Timestep t", fontsize=12)
    ax.set_ylabel("Mean Cosine to Real Mean", fontsize=12)
    ax.set_title("Global Shift (Cos to Real Mean) vs Timestep", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_cos_to_mean.png", dpi=150)
    plt.close()

    # 3) Pairwise cosine spread
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["t_val"], df["pairwise_cos_aligned_mean"], "C2-o", label="Aligned", linewidth=2, markersize=4)
    ax.plot(df["t_val"], df["pairwise_cos_gen_dino_mean"], "C1--", label="Gen DINO", linewidth=2)
    ax.axhline(y=df["pairwise_cos_real_mean"].iloc[0], color="C0", linestyle=":", label="Real", linewidth=2)
    ax.set_xlabel("Timestep t", fontsize=12)
    ax.set_ylabel("Mean Pairwise Cosine", fontsize=12)
    ax.set_title("Within-Distribution Diversity vs Timestep", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_pairwise_cos.png", dpi=150)
    plt.close()

    # 4) kNN radius to real
    if "knn10_aligned_mean" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["t_val"], df["knn10_aligned_mean"], "C2-o", label="Aligned→Real", linewidth=2, markersize=4)
        # Gen DINO and Real kNN are constant (pre-computed once) - use horizontal lines
        ax.axhline(y=df["knn10_gen_dino_mean"].iloc[0], color="C1", linestyle="--", label="Gen DINO→Real (const)", linewidth=2)
        ax.axhline(y=df["knn10_real_mean"].iloc[0], color="C0", linestyle=":", label="Real→Real (const)", linewidth=2)
        ax.set_xlabel("Timestep t", fontsize=12)
        ax.set_ylabel("Mean kNN(10) Cosine Distance", fontsize=12)
        ax.set_title("kNN Radius to Real vs Timestep", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        plt.tight_layout()
        plt.savefig(output_dir / "trajectory_knn10.png", dpi=150)
        plt.close()

    # 5) Mahalanobis energy
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["t_val"], df["energy_aligned_mean"], "C2-o", label="Aligned", linewidth=2, markersize=4)
    ax.axhline(y=df["energy_gen_dino_mean"].iloc[0], color="C1", linestyle="--", label="Gen DINO (const)", linewidth=2)
    ax.set_xlabel("Timestep t", fontsize=12)
    ax.set_ylabel("Mean Mahalanobis Energy", fontsize=12)
    ax.set_title("Typicality (Energy) vs Timestep", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_energy.png", dpi=150)
    plt.close()

    # 6) AUROC evolution
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["t_val"], df["AUROC_one_sided_aligned"], "C2-o", label="Aligned (one-sided)", linewidth=2, markersize=4)
    ax.plot(df["t_val"], df["AUROC_two_sided_sqz_aligned"], "C2--", label="Aligned (two-sided z²)", linewidth=2)
    ax.axhline(y=df["AUROC_one_sided_gen_dino"].iloc[0], color="C1", linestyle="-", label="Gen DINO (one-sided)", linewidth=1.5)
    ax.axhline(y=0.5, color="gray", linestyle=":", label="Random", linewidth=1)
    ax.set_xlabel("Timestep t", fontsize=12)
    ax.set_ylabel("AUROC (Real vs Gen)", fontsize=12)
    ax.set_title("OOD Detection AUROC vs Timestep", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_auroc.png", dpi=150)
    plt.close()

    # 7) Paired cosine (aligned vs gen_dino)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["t_val"], df["paired_cos_aligned_gen_dino_mean"], "C3-o", linewidth=2, markersize=4)
    ax.fill_between(
        df["t_val"],
        df["paired_cos_aligned_gen_dino_mean"] - df["paired_cos_aligned_gen_dino_std"],
        df["paired_cos_aligned_gen_dino_mean"] + df["paired_cos_aligned_gen_dino_std"],
        alpha=0.3, color="C3"
    )
    ax.set_xlabel("Timestep t", fontsize=12)
    ax.set_ylabel("Paired Cosine Similarity", fontsize=12)
    ax.set_title("Aligned vs Final DINO (Paired) vs Timestep", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_paired_cos.png", dpi=150)
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Analyze aligned feature distribution collapse across denoising timesteps"
    )
    parser.add_argument("--checkpoint-key", type=str, required=True, help="Checkpoint key")
    parser.add_argument("--output-dir", type=str, default="outputs/collapse_trajectory", help="Output directory")
    parser.add_argument("--marginal", action="store_true", help="Model is marginal (split seen/unseen)")
    parser.add_argument("--samples-per-condition", type=int, default=50, help="Samples per condition")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--num-inference-steps", type=int, default=250, help="Number of inference steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation, load from cache")

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Output directory
    output_dir = Path(args.output_dir) / args.checkpoint_key
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Aligned Feature Collapse Trajectory Analysis")
    log.info("=" * 70)
    log.info(f"Checkpoint: {args.checkpoint_key}")
    log.info(f"Output: {output_dir}")
    log.info(f"Samples/condition: {args.samples_per_condition}")
    log.info(f"Inference steps: {args.num_inference_steps}")

    # Load real features
    log.info("\nLoading real features...")
    real_feats, real_meta = load_real_features("celeba")
    real_np = real_feats.numpy()
    real_by_cond = build_condition_index(real_meta, len(real_feats), CONDITION_KEYS)

    # Cache paths
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    aligned_cache = cache_dir / "aligned_by_timestep.pt"
    gen_dino_cache = cache_dir / "gen_dino_feats.pt"
    gen_meta_cache = cache_dir / "gen_meta.pt"
    timestep_cache = cache_dir / "timestep_values.pt"

    if args.skip_generation and all(p.exists() for p in [aligned_cache, gen_dino_cache, gen_meta_cache, timestep_cache]):
        log.info("\nLoading cached features...")
        aligned_by_timestep = torch.load(aligned_cache, map_location="cpu")
        gen_dino_feats = torch.load(gen_dino_cache, map_location="cpu")
        gen_meta = torch.load(gen_meta_cache, map_location="cpu")
        timestep_values = torch.load(timestep_cache, map_location="cpu")
    else:
        # Load model
        log.info("\nLoading REPA model...")
        pl_module, gen_cfg = load_repa_model(args.checkpoint_key, args.device)
        generator = pl_module.generator

        # Load DINO encoder
        encoder_name = gen_cfg.repa_encoder
        log.info(f"Loading DINO encoder: {encoder_name}")
        dino_encoder = load_dino_encoder(encoder_name, args.device)

        # Get conditions
        all_conditions, cond_metadata = get_celeba_conditions(is_marginal=args.marginal)
        log.info(f"Generating for {len(all_conditions)} conditions")

        # Generate
        log.info("\nGenerating samples with timestep-wise feature capture...")
        aligned_by_timestep, gen_dino_feats, gen_meta, timestep_values = generate_with_timestep_features(
            generator,
            dino_encoder,
            all_conditions,
            args.samples_per_condition,
            args.batch_size,
            args.num_inference_steps,
            args.device,
        )

        # Cache
        log.info("Caching features...")
        torch.save(aligned_by_timestep, aligned_cache)
        torch.save(gen_dino_feats, gen_dino_cache)
        torch.save(gen_meta, gen_meta_cache)
        torch.save(timestep_values, timestep_cache)

    n_gen = len(gen_dino_feats)
    n_timesteps = len(aligned_by_timestep)
    log.info(f"Generated {n_gen} samples, {n_timesteps} timesteps captured")

    # Fit global stats on real
    log.info("\nFitting global Gaussian on real features...")
    global_stats = fit_global_stats(real_feats, regularization=1e-5)

    # Real energy calibration
    real_energy = compute_mahalanobis(
        normalize_features_torch(real_feats),
        global_stats["mu"],
        global_stats["precision"]
    ).numpy()
    real_E_mean = float(np.mean(real_energy))
    real_E_std = float(np.std(real_energy))

    # Pre-compute gen_dino stats ONCE (these are constant across timesteps)
    log.info("\nPre-computing gen_dino reference stats (constant across timesteps)...")
    gen_dino_np = gen_dino_feats.numpy()
    gen_dino_unit = l2_normalize_np(gen_dino_np)
    real_unit_full = l2_normalize_np(real_np)

    # Gen_dino kNN to real (computed once with fixed seed)
    gen_dino_knn_rng = np.random.default_rng(12345)  # Fixed seed for reproducibility
    gen_dino_knn10_stats = compute_knn_radius_stats(
        gen_dino_unit, real_unit_full, k=10, leave_one_out=False,
        max_ref=10000, max_query=5000, rng=gen_dino_knn_rng
    )

    # Real -> Real kNN baseline (computed once)
    real_knn_rng = np.random.default_rng(12346)
    real_knn10_stats = compute_knn_radius_stats(
        real_unit_full, real_unit_full, k=10, leave_one_out=True,
        max_ref=10000, max_query=5000, rng=real_knn_rng
    )

    # Gen_dino energy stats (computed once)
    gen_dino_energy_stats = compute_energy_stats(gen_dino_feats, global_stats)
    gen_dino_energy = compute_mahalanobis(
        normalize_features_torch(gen_dino_feats),
        global_stats["mu"],
        global_stats["precision"]
    ).numpy()
    gen_dino_aurocs = compute_energy_aurocs(real_energy, gen_dino_energy, real_E_mean, real_E_std)

    # Bundle constant stats
    gen_dino_constant_stats = {
        "knn10": gen_dino_knn10_stats,
        "energy": gen_dino_energy_stats,
        "aurocs": gen_dino_aurocs,
    }
    real_constant_stats = {
        "knn10": real_knn10_stats,
    }

    # Pre-compute real mean direction (constant)
    real_mean_dir = real_unit_full.mean(axis=0)
    real_mean_dir = real_mean_dir / (np.linalg.norm(real_mean_dir) + 1e-12)

    # Analyze each timestep
    log.info("\nAnalyzing timesteps...")
    results = []

    sorted_timesteps = sorted(aligned_by_timestep.keys())
    for k_idx in tqdm(sorted_timesteps, desc="Analyzing timesteps"):
        aligned_feats = aligned_by_timestep[k_idx]
        aligned_np = aligned_feats.numpy()
        t_val = timestep_values[k_idx]

        result = analyze_timestep(
            aligned_np=aligned_np,
            real_np=real_np,
            gen_dino_np=gen_dino_np,
            real_feats=real_feats,
            aligned_feats=aligned_feats,
            global_stats=global_stats,
            real_E_mean=real_E_mean,
            real_E_std=real_E_std,
            real_energy=real_energy,
            rng=rng,
            k_idx=k_idx,
            t_val=t_val,
            gen_dino_constant_stats=gen_dino_constant_stats,
            real_constant_stats=real_constant_stats,
            real_unit=real_unit_full,
            gen_dino_unit=gen_dino_unit,
            real_mean_dir=real_mean_dir,
        )
        results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values("k_idx").reset_index(drop=True)

    # Save CSV
    csv_path = output_dir / "collapse_trajectory.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    log.info(f"Saved: {csv_path}")

    # Generate plots
    log.info("\nGenerating plots...")
    plot_summary_trajectories(df, output_dir)

    # Print summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)

    # Key findings at early vs late timesteps
    early = df[df["t_val"] > 0.5].iloc[0] if len(df[df["t_val"] > 0.5]) > 0 else df.iloc[0]
    late = df[df["t_val"] < 0.1].iloc[-1] if len(df[df["t_val"] < 0.1]) > 0 else df.iloc[-1]
    final = df.iloc[-1]

    log.info(f"\nEarly timestep (t={early['t_val']:.3f}):")
    log.info(f"  R_aligned = {early['R_aligned']:.4f}")
    log.info(f"  cos_to_mean_aligned = {early['cos_to_mean_aligned_mean']:.4f}")
    log.info(f"  pairwise_cos_aligned = {early['pairwise_cos_aligned_mean']:.4f}")
    log.info(f"  paired_cos(aligned, gen_dino) = {early['paired_cos_aligned_gen_dino_mean']:.4f}")

    log.info(f"\nFinal timestep (t={final['t_val']:.3f}):")
    log.info(f"  R_aligned = {final['R_aligned']:.4f}")
    log.info(f"  cos_to_mean_aligned = {final['cos_to_mean_aligned_mean']:.4f}")
    log.info(f"  pairwise_cos_aligned = {final['pairwise_cos_aligned_mean']:.4f}")
    log.info(f"  paired_cos(aligned, gen_dino) = {final['paired_cos_aligned_gen_dino_mean']:.4f}")

    log.info(f"\nReference (Real):")
    log.info(f"  R_real = {final['R_real']:.4f}")
    log.info(f"  cos_to_mean_real = {final['cos_to_mean_real_mean']:.4f}")
    log.info(f"  pairwise_cos_real = {final['pairwise_cos_real_mean']:.4f}")

    log.info(f"\nReference (Gen DINO):")
    log.info(f"  R_gen_dino = {final['R_gen_dino']:.4f}")
    log.info(f"  cos_to_mean_gen_dino = {final['cos_to_mean_gen_dino_mean']:.4f}")
    log.info(f"  pairwise_cos_gen_dino = {final['pairwise_cos_gen_dino_mean']:.4f}")

    # AUROC trajectory interpretation
    auroc_early = early.get("AUROC_one_sided_aligned", np.nan)
    auroc_late = final.get("AUROC_one_sided_aligned", np.nan)
    log.info(f"\nAUROC evolution:")
    log.info(f"  Early (t={early['t_val']:.3f}): one-sided = {auroc_early:.4f}")
    log.info(f"  Final (t={final['t_val']:.3f}): one-sided = {auroc_late:.4f}")

    if auroc_late < 0.5:
        log.info("  -> One-sided AUROC inverted at final timestep (over-typical)")
    elif auroc_late < auroc_early:
        log.info("  -> AUROC decreased across timesteps (increasing typicality)")

    log.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
