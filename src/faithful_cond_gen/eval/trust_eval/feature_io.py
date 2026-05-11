"""
Feature loading, normalization, and verification utilities.
"""

import glob
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from faithful_cond_gen.eval.trust_eval.config import (
    CONDITION_ATTRS,
    FEATURE_CONFIGS,
    POSTHOC_MODEL_KEYS,
    REAL_FEATURE_PATHS,
    REAL_FEATURE_PATHS_BY_MODEL,
)
from faithful_cond_gen.eval.trust_eval.subset_io import (
    filter_rxrx1_real_to_scoring_pool,
    filter_rxrx1_to_subset,
    load_rxrx1_subset,
    load_rxrx1_subset_sirnas,
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


def _load_posthoc_mapped_features(
    dataset: str,
    model: str,
    normalize_mode: str = "none",
    gen_override_path: Optional[Path] = None,
) -> Tuple[
    Optional[torch.Tensor], Optional[Dict], Optional[torch.Tensor], Optional[Dict]
]:
    """
    Load posthoc-mapped features: raw hidden states mapped through trained mapper.

    Real: raw_hidden at t=0.01 -> mapper -> 1152-dim features
    Gen: gen_cache cond_*.pt raw_hidden -> mapper -> 1152-dim features

    If gen_override_path is set, load gen raw hidden from a consolidated
    {features, metadata} shard (as written by generate_samples_repa.py with
    use_raw_hidden=true) instead of the diag/gen_cache/cond_*.pt pipeline.
    """
    from faithful_cond_gen.posthoc_alignment.mapper import ResidualAlignmentMapper

    model_key = POSTHOC_MODEL_KEYS.get((dataset, model))
    if model_key is None:
        logger.warning(f"No posthoc model key for ({dataset}, {model})")
        return None, None, None, None

    condition_keys = CONDITION_ATTRS.get(dataset, [])

    # Load mapper. Unified whitened-MSE rollout: every model lives under
    # mappers_whitened/. Mapper out_dim is read from its sibling
    # training_config.json so DINOv3 (1024) and SigLIP (1152) targets both work.
    import json
    import os

    # Per-bucket winning γ from the tempered-whitening sweep:
    #   rxrx1 full-support → γ=0.75 (mappers_whit075)
    #   rxrx1 marginal + all celeba → γ=1.0 (mappers_whitened)
    env_root = os.environ.get("POSTHOC_MAPPER_ROOT")
    if env_root is not None:
        mapper_root = env_root
    elif dataset == "rxrx1" and model_key.endswith("_full_v1"):
        mapper_root = "mappers_whit075"
    else:
        mapper_root = "mappers_whitened"
    mapper_dir = Path(f"outputs/posthoc_alignment/{mapper_root}/{model_key}")
    mapper_path = mapper_dir / "best_mapper.pt"
    if not mapper_path.exists():
        logger.warning(f"Mapper not found at {mapper_path}")
        return None, None, None, None

    training_cfg_path = mapper_dir / "training_config.json"
    if training_cfg_path.exists():
        with open(training_cfg_path) as f:
            training_cfg = json.load(f)
        mapper_cfg = training_cfg.get("mapper", {})
        in_dim = int(mapper_cfg.get("in_dim", 768))
        out_dim = int(mapper_cfg.get("out_dim", 1152))
        hidden_dim = int(mapper_cfg.get("hidden_dim", 2048))
    else:
        in_dim, out_dim, hidden_dim = 768, 1152, 2048

    mapper = ResidualAlignmentMapper(in_dim, out_dim, hidden_dim=hidden_dim)
    mapper.load_state_dict(torch.load(mapper_path, map_location="cpu", weights_only=True))
    mapper.eval()

    # Load optional centering stats (center + L2-norm preprocessing).
    stats_path = mapper_dir / "preprocessing_stats.pt"
    if stats_path.exists():
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        src_mean = stats["src_mean"]
        logger.info(f"  Loaded preprocessing stats (center+norm) from {stats_path}")
    else:
        src_mean = None

    # Load and map real features (from raw hidden states at t=0.01)
    raw_hidden_path = Path(f"outputs/posthoc_alignment/raw_hidden/{model_key}/t0.01_hidden.pt")
    if not raw_hidden_path.exists():
        logger.warning(f"Raw hidden not found at {raw_hidden_path}")
        return None, None, None, None

    raw_data = torch.load(raw_hidden_path, map_location="cpu", weights_only=False)
    real_hidden = raw_data["features"]
    real_meta = {k: raw_data["metadata"][k] for k in condition_keys}

    # For rxrx1, filter the raw hidden pool to the sirna-column scoring pool
    # so the mapper only runs on rows scoring will use. (Scoring composes
    # marginals over (ct, sirna); hence "subset sirnas across all 4 cts".)
    if dataset == "rxrx1":
        real_hidden, real_meta = filter_rxrx1_real_to_scoring_pool(real_hidden, real_meta)

    if src_mean is not None:
        real_hidden = l2_normalize_features(real_hidden - src_mean)

    with torch.no_grad():
        real_feats = mapper(real_hidden)
    logger.info(f"  Posthoc real: {real_feats.shape} from {raw_hidden_path}")

    # Load and map generated features.
    #   - CelebA: per-condition gen_cache/cond_*.pt files (legacy path).
    #   - RxRx1: feature_cache_rxrx1_subset/{model_key}_encoded.pt (built by
    #            filter + Step-5 partial regen). It stores gen_hidden directly
    #            for the canonical 50-condition subset.
    if gen_override_path is not None:
        if not gen_override_path.exists():
            logger.warning(f"Posthoc gen override not found at {gen_override_path}")
            return None, None, None, None
        cache = torch.load(gen_override_path, map_location="cpu", weights_only=False)
        gen_hidden = cache["features"]
        cache_meta = cache.get("metadata", {})
        gen_meta = {k: cache_meta[k] for k in condition_keys if k in cache_meta}
        if dataset == "rxrx1":
            gen_hidden, gen_meta = filter_rxrx1_to_subset(gen_hidden, gen_meta)
        if src_mean is not None:
            gen_hidden = l2_normalize_features(gen_hidden - src_mean)
        with torch.no_grad():
            gen_feats = mapper(gen_hidden)
        logger.info(
            f"  Posthoc gen (override): {gen_feats.shape} from {gen_override_path}"
        )
    elif dataset == "rxrx1":
        subset_cache = Path(
            f"outputs/posthoc_alignment/feature_cache_rxrx1_subset/{model_key}_encoded.pt"
        )
        if not subset_cache.exists():
            logger.warning(f"RxRx1 subset encoded cache not found at {subset_cache}")
            return None, None, None, None
        cache = torch.load(subset_cache, map_location="cpu", weights_only=False)
        gen_hidden = cache["gen_hidden"]
        gen_meta = {k: cache["gen_meta"][k] for k in condition_keys}
        if src_mean is not None:
            gen_hidden = l2_normalize_features(gen_hidden - src_mean)
        with torch.no_grad():
            gen_feats = mapper(gen_hidden)
        logger.info(
            f"  Posthoc gen (rxrx1 subset): {gen_feats.shape} from {subset_cache}"
        )
    else:
        cache_dir = Path(f"outputs/posthoc_alignment/diag/{model_key}/gen_cache")
        pts = sorted(glob.glob(str(cache_dir / "cond_*.pt")))
        if not pts:
            logger.warning(f"No gen_cache files in {cache_dir}")
            return None, None, None, None

        # Per-model stored-tuple order (verified visually from cond_*.pt samples):
        #   vanilla_marginal → CK order (Male, Smiling, Blond_Hair, Eyeglasses)
        #   all other celeba models → alphabetical (Blond_Hair, Eyeglasses, Male, Smiling)
        if model_key == "celeba_vanilla_marginal_v1":
            stored_order = list(condition_keys)
        else:
            stored_order = sorted(condition_keys)
        hids = []
        metas = {k: [] for k in condition_keys}
        for p in pts:
            d = torch.load(p, map_location="cpu", weights_only=False)
            hids.append(d["raw_hidden"])
            cond = d["condition"]
            n = d["raw_hidden"].shape[0]
            for ki, k in enumerate(stored_order):
                metas[k].append(torch.full((n,), cond[ki], dtype=torch.long))

        gen_hidden = torch.cat(hids)
        gen_meta = {k: torch.cat(metas[k]) for k in condition_keys}
        if src_mean is not None:
            gen_hidden = l2_normalize_features(gen_hidden - src_mean)
        with torch.no_grad():
            gen_feats = mapper(gen_hidden)
        logger.info(f"  Posthoc gen: {gen_feats.shape} from {len(pts)} conditions")

    del mapper

    # Apply normalization
    real_feats = apply_normalization(real_feats, normalize_mode, "real_posthoc_mapped")
    gen_feats = apply_normalization(gen_feats, normalize_mode, "gen_posthoc_mapped")

    return real_feats, real_meta, gen_feats, gen_meta


def load_posthoc_kid_features(
    dataset: str, model: str, normalize_mode: str = "none"
) -> Tuple[
    Optional[torch.Tensor], Optional[Dict], Optional[torch.Tensor], Optional[Dict]
]:
    """
    Load DINO features for posthoc_mapped gen samples (from pre-encoded caches).

    These caches were produced by the diagnostic scripts and contain gen_dinov3
    features for the same gen_cache samples used by posthoc_mapped scoring.

    Returns (real_dino_feats, real_dino_meta, gen_dino_feats, gen_dino_meta).
    """
    model_key = POSTHOC_MODEL_KEYS.get((dataset, model))
    if model_key is None:
        return None, None, None, None

    condition_keys = CONDITION_ATTRS.get(dataset, [])

    # Locate encoded cache (dataset-specific paths from diagnostic scripts).
    # RxRx1 now uses the canonical 50-cond subset cache.
    if dataset == "celeba":
        cache_path = Path(f"/mnt/pvc/posthoc_debug/feature_cache/{model_key}_encoded.pt")
    else:
        cache_path = Path(
            f"outputs/posthoc_alignment/feature_cache_rxrx1_subset/{model_key}_encoded.pt"
        )

    if not cache_path.exists():
        logger.warning(f"Posthoc encoded cache not found at {cache_path}")
        return None, None, None, None

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    gen_dino = cache["gen_dinov3"]
    gen_meta = cache["gen_meta"]
    logger.info(f"  Posthoc KID gen: {gen_dino.shape} from {cache_path}")

    # Real DINO features (standard path; rxrx1 prefers the scoring subset).
    real_path = Path(REAL_FEATURE_PATHS.get((dataset, "dinov3"), ""))
    if dataset == "rxrx1":
        scoring_sibling = real_path.with_name(real_path.stem + "_subset_scoring.pt")
        if scoring_sibling.exists():
            real_path = scoring_sibling
    if not real_path.exists():
        logger.warning(f"Real DINO features not found at {real_path}")
        return None, None, None, None

    real_data = torch.load(real_path, map_location="cpu", weights_only=False)
    real_dino = real_data["features"]
    real_meta = real_data.get("metadata", {})
    if dataset == "rxrx1":
        # Safety net if sibling was not the scoring-pool version.
        real_dino, real_meta = filter_rxrx1_real_to_scoring_pool(real_dino, real_meta)
    logger.info(f"  Posthoc KID real: {real_dino.shape} from {real_path}")

    # Apply normalization
    real_dino = apply_normalization(real_dino, normalize_mode, "kid_real_dinov3")
    gen_dino = apply_normalization(gen_dino, normalize_mode, "kid_gen_dinov3")

    return real_dino, real_meta, gen_dino, gen_meta


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
        feature_type: Feature type (e.g., "dinov3", "aligned_mean", "posthoc_mapped")
        normalize_mode: Normalization to apply ("none" or "l2")

    Returns:
        Tuple of (real_feats, real_meta, gen_feats, gen_meta)
        Returns (None, None, None, None) if features not found
    """
    # Posthoc mapped features use a completely different loading path
    if feature_type == "posthoc_mapped":
        return _load_posthoc_mapped_features(dataset, model, normalize_mode)

    # Posthoc mapped at a specific denoising step (timestep ablation).
    # Gen raw hidden comes from the consolidated aligned_mean_features_step{k}.pt
    # produced by generate_samples_repa.py with use_raw_hidden=true.
    if feature_type.startswith("posthoc_step"):
        config_key = (dataset, model, feature_type)
        if config_key not in FEATURE_CONFIGS:
            logger.warning(f"No config for {config_key}")
            return None, None, None, None
        gen_dir, feature_file = FEATURE_CONFIGS[config_key]
        gen_override = Path(f"outputs/gen/{gen_dir}/{feature_file}")
        return _load_posthoc_mapped_features(
            dataset, model, normalize_mode, gen_override_path=gen_override
        )

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

    # RxRx1: prefer the subset siblings.
    #   real -> *_subset_scoring.pt (sirna-column filter, used by scoring).
    #   gen  -> *_subset.pt          (strict 50-pair filter).
    if dataset == "rxrx1":
        real_scoring = real_path.with_name(real_path.stem + "_subset_scoring.pt")
        gen_subset = gen_path.with_name(gen_path.stem + "_subset.pt")
        if real_scoring.exists():
            real_path = real_scoring
        if gen_subset.exists():
            gen_path = gen_subset

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

    # Safety net: if a rxrx1 caller landed on a non-subset cache for any
    # reason (e.g. sibling missing), force-filter here.
    #   real -> sirna-column scoring pool (broader, for marginal modeling)
    #   gen  -> strict 50-pair filter
    if dataset == "rxrx1":
        real_feats, real_meta = filter_rxrx1_real_to_scoring_pool(real_feats, real_meta)
        subset = load_rxrx1_subset()
        if gen_filenames is not None and gen_filenames and len(gen_filenames) == gen_feats.shape[0]:
            ct = gen_meta["cell_type_id"].tolist()
            sr = gen_meta["sirna_id"].tolist()
            keep_idx = [
                i for i, (a, b) in enumerate(zip(ct, sr))
                if (int(a), int(b)) in subset
            ]
            gen_filenames = [gen_filenames[i] for i in keep_idx]
        gen_feats, gen_meta = filter_rxrx1_to_subset(gen_feats, gen_meta)

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
