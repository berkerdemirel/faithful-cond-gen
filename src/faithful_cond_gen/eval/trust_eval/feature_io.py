"""
Feature loading, normalization, and verification utilities.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from faithful_cond_gen.eval.trust_eval.config import (
    CONDITION_ATTRS,
    FEATURE_CONFIGS,
    REAL_FEATURE_PATHS,
    REAL_FEATURE_PATHS_BY_MODEL,
)

logger = logging.getLogger(__name__)


def l2_normalize_features(features: torch.Tensor) -> torch.Tensor:
    """
    Apply L2 normalization to feature vectors.

    Args:
        features: Feature tensor of shape (N, D)

    Returns:
        L2-normalized features with unit norm per row
    """
    norms = features.norm(dim=1, keepdim=True)
    return features / (norms + 1e-12)


def apply_normalization(
    features: torch.Tensor, normalize_mode: str, feature_name: str = "features"
) -> torch.Tensor:
    """
    Apply normalization to features based on mode.

    Args:
        features: Feature tensor of shape (N, D)
        normalize_mode: One of "none" or "l2"
        feature_name: Name for logging purposes

    Returns:
        Normalized (or unchanged) features
    """
    if normalize_mode == "l2":
        logger.info(f"  Applying L2 normalization to {feature_name} ({features.shape})")
        return l2_normalize_features(features)
    return features


def get_filenames_from_meta(meta: Dict) -> Optional[List[str]]:
    """
    Extract filename list from metadata dictionary.

    Checks common keys: filenames, file_names, paths, image_paths, img_paths.

    Returns:
        List of filename strings, or None if not found
    """
    for key in ["filenames", "file_names", "paths", "image_paths", "img_paths"]:
        if key in meta:
            v = meta[key]
            if isinstance(v, torch.Tensor):
                v = v.tolist()
            return [str(x) for x in v]
    return None


def verify_feature_ordering(meta1: Dict, meta2: Dict, name1: str, name2: str) -> bool:
    """
    Verify that two feature caches have matching sample ordering.

    Verification hierarchy:
    1. Filename metadata (strongest - exact sample matching)
    2. Condition metadata (fallback - verifies attribute values match)
    3. No metadata (fails - cannot verify)

    Args:
        meta1, meta2: Metadata dictionaries from feature caches
        name1, name2: Names for error messages

    Returns:
        True if verified matching, False if could not verify (no metadata)

    Raises:
        ValueError: If filenames/conditions exist but don't match
    """
    names1 = get_filenames_from_meta(meta1) if isinstance(meta1, dict) else None
    names2 = get_filenames_from_meta(meta2) if isinstance(meta2, dict) else None

    # Primary check: filename metadata
    if names1 is not None and names2 is not None:
        if len(names1) != len(names2):
            raise ValueError(
                f"Feature ordering mismatch: {name1} has {len(names1)} samples, "
                f"{name2} has {len(names2)} samples. Cannot safely compare."
            )
        # Check first 100 samples for efficiency
        mismatches = [
            (i, n1, n2)
            for i, (n1, n2) in enumerate(zip(names1[:100], names2[:100]))
            if n1 != n2
        ]
        if mismatches:
            first_mismatch = mismatches[0]
            raise ValueError(
                f"Feature ordering mismatch between {name1} and {name2}. "
                f"First mismatch at index {first_mismatch[0]}: "
                f"'{first_mismatch[1]}' vs '{first_mismatch[2]}'. "
                f"Regenerate caches with consistent ordering."
            )
        logger.info(f"  Verified (filenames): {name1} and {name2} have matching sample order")
        return True

    # Fallback check: condition metadata (same-model features should have identical conditions)
    if isinstance(meta1, dict) and isinstance(meta2, dict):
        # Find common condition keys (exclude 'filenames', 'paths', etc.)
        exclude_keys = {"filenames", "file_names", "paths", "image_paths", "img_paths"}
        cond_keys1 = set(meta1.keys()) - exclude_keys
        cond_keys2 = set(meta2.keys()) - exclude_keys
        common_keys = cond_keys1 & cond_keys2

        if common_keys:
            # Verify all common condition values match
            for key in common_keys:
                v1, v2 = meta1[key], meta2[key]
                # Convert to numpy for comparison
                if hasattr(v1, "numpy"):
                    v1 = v1.numpy()
                if hasattr(v2, "numpy"):
                    v2 = v2.numpy()
                if hasattr(v1, "__len__") and hasattr(v2, "__len__"):
                    if len(v1) != len(v2):
                        raise ValueError(
                            f"Condition metadata length mismatch for '{key}': "
                            f"{name1} has {len(v1)}, {name2} has {len(v2)}."
                        )
                    if not (v1 == v2).all():
                        raise ValueError(
                            f"Condition metadata mismatch for '{key}' between {name1} and {name2}. "
                            f"Features may have different ordering."
                        )
            logger.info(
                f"  Verified (conditions): {name1} and {name2} have matching condition metadata "
                f"({len(common_keys)} keys: {', '.join(sorted(common_keys))})"
            )
            return True

    # No verification possible
    missing = []
    if names1 is None:
        missing.append(name1)
    if names2 is None:
        missing.append(name2)
    warnings.warn(
        f"Cannot verify feature ordering: {', '.join(missing)} missing filename metadata "
        f"and no common condition metadata found.",
        UserWarning,
    )
    return False


def verify_consolidated_features(
    gen_feats: torch.Tensor,
    gen_meta: Dict,
    condition_keys: List[str],
    filenames: Optional[List[str]] = None,
    n_spot_checks: int = 10,
) -> bool:
    """
    Verify that consolidated features have proper metadata alignment.

    Checks:
    1. All condition keys exist in metadata
    2. Metadata arrays have same length as features
    3. For samples with filenames, condition values match filename encoding

    Args:
        gen_feats: Generated features tensor (N, D)
        gen_meta: Metadata dict with condition keys
        condition_keys: Expected condition attribute keys
        filenames: Optional list of filenames for cross-check
        n_spot_checks: Number of random samples to spot-check

    Returns:
        True if all checks pass

    Raises:
        ValueError if any check fails
    """
    N = gen_feats.shape[0]

    # Check 1: All condition keys exist
    missing_keys = [k for k in condition_keys if k not in gen_meta]
    if missing_keys:
        raise ValueError(f"Missing condition keys in metadata: {missing_keys}")

    # Check 2: Metadata arrays have correct length
    for key in condition_keys:
        meta_arr = gen_meta[key]
        meta_len = len(meta_arr) if hasattr(meta_arr, "__len__") else meta_arr.shape[0]
        if meta_len != N:
            raise ValueError(
                f"Metadata '{key}' length mismatch: {meta_len} vs {N} features"
            )

    # Check 3: Spot-check filename -> condition consistency
    if filenames and len(filenames) == N:
        rng = np.random.default_rng(42)
        check_indices = rng.choice(N, size=min(n_spot_checks, N), replace=False)

        for idx in check_indices:
            fname = filenames[idx]
            # Parse signature from filename (e.g., "Blond_Hair0_..._0.png")
            stem = Path(fname).stem
            sig, _ = stem.rsplit("_", 1)

            # Parse condition from signature
            expected_cond = {}
            parts = sig.split("_")
            buffer = []
            for p in parts:
                if p and p[-1] in ["0", "1"] and len(p) > 1:
                    attr_name = "_".join(buffer + [p[:-1]])
                    expected_cond[attr_name] = int(p[-1])
                    buffer = []
                else:
                    buffer.append(p)

            # Compare with metadata
            for key in condition_keys:
                if key in expected_cond:
                    meta_val = gen_meta[key][idx]
                    if isinstance(meta_val, torch.Tensor):
                        meta_val = meta_val.item()
                    if expected_cond[key] != meta_val:
                        raise ValueError(
                            f"Condition mismatch at index {idx}: "
                            f"filename '{fname}' implies {key}={expected_cond[key]}, "
                            f"but metadata has {key}={meta_val}"
                        )

        logger.info(
            f"  ✓ Spot-checked {len(check_indices)} samples: conditions match filenames"
        )

    logger.info(
        f"  ✓ Metadata integrity verified: {N} samples, {len(condition_keys)} condition keys"
    )
    return True


def load_features_for_dataset(
    dataset: str, model: str, feature_type: str, normalize_mode: str = "none"
) -> Tuple[
    Optional[torch.Tensor], Optional[Dict], Optional[torch.Tensor], Optional[Dict]
]:
    """
    Load real and generated features for a specific model/feature_type.

    Args:
        dataset: Dataset name (e.g., "celeba", "rxrx1")
        model: Model name (e.g., "vanilla_full", "repa_marginal")
        feature_type: Feature type (e.g., "dinov3", "aligned_mean")
        normalize_mode: Normalization to apply ("none" or "l2")

    Returns:
        Tuple of (real_feats, real_meta, gen_feats, gen_meta)
        Returns (None, None, None, None) if features not found
    """
    # Get feature config
    config_key = (dataset, model, feature_type)
    if config_key not in FEATURE_CONFIGS:
        logger.warning(f"No config for {config_key}")
        return None, None, None, None

    gen_dir, feature_file = FEATURE_CONFIGS[config_key]

    # Real features - check model-specific path first (for aligned features),
    # then dataset-level path (for dinov3), then fallback
    model_specific_key = (dataset, model, feature_type)
    real_key = (dataset, feature_type)

    if model_specific_key in REAL_FEATURE_PATHS_BY_MODEL:
        real_path = Path(REAL_FEATURE_PATHS_BY_MODEL[model_specific_key])
        logger.info(f"  Using model-specific real features for {model}/{feature_type}")
    elif real_key in REAL_FEATURE_PATHS:
        real_path = Path(REAL_FEATURE_PATHS[real_key])
    else:
        # Fallback to old path
        real_path = Path(f"outputs/real_{dataset}_dinov3_meanpatch/train_features.pt")

    gen_path = Path(f"outputs/gen/{gen_dir}/{feature_file}")

    if not real_path.exists():
        logger.warning(f"Real features not found at {real_path}")
        return None, None, None, None

    if not gen_path.exists():
        logger.warning(f"Generated features not found at {gen_path}")
        return None, None, None, None

    logger.info(f"  Loading real features from: {real_path}")
    logger.info(f"  Loading gen features from: {gen_path}")

    data = torch.load(real_path, map_location="cpu", weights_only=False)
    real_feats, real_meta = data["features"], data.get("metadata", {})

    data = torch.load(gen_path, map_location="cpu", weights_only=False)
    gen_feats, gen_meta = data["features"], data.get("metadata", {})
    gen_filenames = data.get("filenames", None)

    # Verify consolidated feature integrity
    condition_keys = CONDITION_ATTRS.get(dataset, [])
    if condition_keys and gen_meta:
        try:
            verify_consolidated_features(
                gen_feats, gen_meta, condition_keys, filenames=gen_filenames
            )
        except ValueError as e:
            logger.error(f"Consolidation integrity check FAILED: {e}")
            raise

    # Apply normalization if requested (once, at load time)
    real_feats = apply_normalization(real_feats, normalize_mode, f"real_{feature_type}")
    gen_feats = apply_normalization(gen_feats, normalize_mode, f"gen_{feature_type}")

    return real_feats, real_meta, gen_feats, gen_meta
