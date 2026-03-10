"""
Consolidate REPA aligned features into a format matching extracted features.

The aligned features are per-patch (N, 256, 1024) saved per condition.
This script:
1. Builds a global index mapping from (signature, local_idx) -> global_idx
2. Loads each shard and assigns features to their global indices
3. Stacks features in sorted global index order
4. Adds integrity checks to ensure alignment

Usage:
    PYTHONPATH=src uv run python scripts/consolidate_aligned_features.py \
        --model celeba_repa_full --pooling mean
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


def parse_condition_from_signature(signature: str) -> Dict[str, int]:
    """
    Parse condition values from signature string.

    Supports two formats:
    - CelebA: "Blond_Hair0_Eyeglasses1_Male0_Smiling1" -> {Blond_Hair: 0, Eyeglasses: 1, ...}
    - RxRx1: "cell0_sirna123" -> {cell_type_id: 0, sirna_id: 123}
    """
    # Check if it's RxRx1 format (starts with "cell")
    if signature.startswith("cell"):
        # Parse RxRx1 format: cell{X}_sirna{Y}
        import re
        match = re.match(r"cell(\d+)_sirna(\d+)", signature)
        if match:
            return {
                "cell_type_id": int(match.group(1)),
                "sirna_id": int(match.group(2)),
            }

    # CelebA format: parse binary attributes
    parts = signature.split("_")
    cond_dict = {}
    buffer = []

    for p in parts:
        if not p:
            continue
        # Check if token ends with 0 or 1
        if p[-1] in ["0", "1"] and len(p) > 1:
            attr_name = "_".join(buffer + [p[:-1]])
            val = int(p[-1])
            cond_dict[attr_name] = val
            buffer = []
        else:
            buffer.append(p)

    return cond_dict


def parse_filename(fname: str) -> Tuple[str, int]:
    """
    Parse filename into (signature, local_idx).
    e.g., "Blond_Hair0_Eyeglasses0_Male0_Smiling0_123.png" -> ("Blond_Hair0_..._Smiling0", 123)
    """
    stem = Path(fname).stem
    sig, idx_str = stem.rsplit("_", 1)
    return sig, int(idx_str)


def pool_features(feats: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """
    Pool patch features (N, P, D) to per-sample features (N, D).

    Args:
        feats: (N, num_patches, feature_dim)
        method: 'mean', 'cls' (first patch), 'max'

    Returns:
        (N, D) pooled features
    """
    if method == "mean":
        return feats.mean(dim=1)
    elif method == "cls":
        return feats[:, 0, :]  # First patch as CLS equivalent
    elif method == "max":
        return feats.max(dim=1)[0]
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def build_global_index_mapping(images_dir: Path) -> Tuple[Dict[Tuple[str, int], int], List[str]]:
    """
    Build a mapping from (signature, local_idx) -> global_idx.

    Global indices are assigned based on alphabetically sorted filenames.
    This ensures deterministic ordering independent of file system enumeration order.

    Returns:
        - mapping: Dict[(signature, local_idx)] -> global_idx
        - sorted_filenames: List of filenames in sorted order
    """
    # Get all image files
    png_files = list(images_dir.glob("*.png"))
    pt_files = list(images_dir.glob("*.pt"))

    image_files = png_files if png_files else pt_files
    if not image_files:
        raise FileNotFoundError(f"No image files (*.png or *.pt) in {images_dir}")

    # Sort alphabetically for deterministic ordering
    sorted_files = sorted(image_files, key=lambda p: p.name)
    sorted_filenames = [f.name for f in sorted_files]

    # Build mapping
    mapping: Dict[Tuple[str, int], int] = {}
    for global_idx, fname in enumerate(sorted_filenames):
        sig, local_idx = parse_filename(fname)
        key = (sig, local_idx)
        if key in mapping:
            raise ValueError(f"Duplicate (signature, local_idx): {key}")
        mapping[key] = global_idx

    return mapping, sorted_filenames


def load_shard_with_indices(
    shard_path: Path,
    signature: str,
) -> Tuple[torch.Tensor, List[int], Dict[str, int]]:
    """
    Load a shard and extract features with their local indices.

    If shard contains explicit 'indices', use those.
    Otherwise, assume sequential local indices 0, 1, 2, ...

    Returns:
        - features: Tensor (n_samples, ...)
        - local_indices: List of local indices within this condition
        - condition: Dict of attribute values
    """
    data = torch.load(shard_path, map_location="cpu")

    # Handle both dict format (new) and raw tensor format (legacy)
    if isinstance(data, dict):
        features = data["aligned_features"] if "aligned_features" in data else data.get("features")
        if features is None:
            raise KeyError(f"Shard {shard_path} has dict but no 'aligned_features' or 'features' key")

        # Try to get explicit indices
        if "indices" in data:
            local_indices = data["indices"]
            if isinstance(local_indices, torch.Tensor):
                local_indices = local_indices.tolist()
        else:
            # Fall back to sequential
            local_indices = list(range(features.shape[0]))

        # Try to get condition from shard metadata
        condition = data.get("condition", None)
    else:
        # Legacy format: raw tensor
        features = data
        local_indices = list(range(features.shape[0]))
        condition = None

    # If no condition in shard, parse from signature
    if condition is None:
        condition = parse_condition_from_signature(signature)

    return features, local_indices, condition


def consolidate_aligned_features_by_index(
    model_dir: Path,
    pooling: str = "mean",
    condition_keys: Optional[List[str]] = None,
) -> Dict:
    """
    Consolidate aligned features using index-based joining.

    This ensures features[i] corresponds to the i-th image in sorted filename order,
    regardless of the order shards are loaded or concatenated.

    Args:
        model_dir: Path to model output directory
        pooling: Pooling method for patch features
        condition_keys: List of condition attribute keys (auto-detected if None)

    Returns:
        Dict with features, metadata, indices, and verification info
    """
    aligned_dir = model_dir / "aligned_features"
    images_dir = model_dir / "images"

    if not aligned_dir.exists():
        raise FileNotFoundError(f"No aligned_features directory in {model_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"No images directory in {model_dir}")

    # Step 1: Build global index mapping from images
    print("Building global index mapping from images...")
    idx_mapping, sorted_filenames = build_global_index_mapping(images_dir)
    N = len(sorted_filenames)
    print(f"  Found {N} images")
    print(f"  First: {sorted_filenames[0]}")
    print(f"  Last:  {sorted_filenames[-1]}")

    # Step 2: Find all shards
    shard_files = sorted(aligned_dir.glob("*_aligned_feats.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No aligned feature shards in {aligned_dir}")
    print(f"Found {len(shard_files)} aligned feature shards")

    # Step 3: Load shards and build index -> (feature, condition) mapping
    print("Loading shards and mapping to global indices...")

    feature_dim = None
    idx_to_feature: Dict[int, torch.Tensor] = {}
    idx_to_condition: Dict[int, Dict[str, int]] = {}
    all_condition_keys = set()

    for shard_path in tqdm(shard_files, desc="Loading shards"):
        # Extract signature from filename
        sig = shard_path.stem.replace("_aligned_feats", "")

        # Load shard
        features, local_indices, condition = load_shard_with_indices(shard_path, sig)
        all_condition_keys.update(condition.keys())

        # Pool if needed
        if features.ndim == 3:  # (N, patches, dim)
            features = pool_features(features, method=pooling)

        if feature_dim is None:
            feature_dim = features.shape[1]
        elif features.shape[1] != feature_dim:
            raise ValueError(f"Feature dim mismatch: {features.shape[1]} vs {feature_dim}")

        # Map to global indices
        for i, local_idx in enumerate(local_indices):
            key = (sig, local_idx)
            if key not in idx_mapping:
                warnings.warn(f"Shard sample {key} not found in image index mapping, skipping")
                continue

            global_idx = idx_mapping[key]

            if global_idx in idx_to_feature:
                raise ValueError(f"Duplicate global index {global_idx} from {key}")

            idx_to_feature[global_idx] = features[i]
            idx_to_condition[global_idx] = condition.copy()

    # Step 4: Verify completeness
    found_indices = set(idx_to_feature.keys())
    expected_indices = set(range(N))

    missing = expected_indices - found_indices
    extra = found_indices - expected_indices

    if missing:
        warnings.warn(f"Missing {len(missing)} indices (first 5: {sorted(missing)[:5]})")
    if extra:
        warnings.warn(f"Extra {len(extra)} indices (first 5: {sorted(extra)[:5]})")

    # Step 5: Build output tensors in sorted index order
    print("Building output tensors...")

    # Use condition_keys if provided, otherwise use detected keys
    if condition_keys is None:
        condition_keys = sorted(all_condition_keys)
    print(f"  Condition keys: {condition_keys}")

    # Sort indices for deterministic output
    sorted_indices = sorted(idx_to_feature.keys())
    actual_N = len(sorted_indices)

    # Stack features
    features_tensor = torch.stack([idx_to_feature[i] for i in sorted_indices], dim=0)

    # Build metadata arrays
    metadata = {}
    for key in condition_keys:
        vals = [idx_to_condition[i].get(key, 0) for i in sorted_indices]
        metadata[key] = torch.tensor(vals, dtype=torch.long)

    # Build indices array
    indices_tensor = torch.tensor(sorted_indices, dtype=torch.long)

    # Build filenames list for verification
    filenames = [sorted_filenames[i] for i in sorted_indices]

    print(f"Output shape: {features_tensor.shape}")

    # Step 6: Integrity checks
    print("\nRunning integrity checks...")

    # Check 1: No duplicate indices
    assert len(set(sorted_indices)) == len(sorted_indices), "Duplicate indices detected"
    print("  ✓ No duplicate indices")

    # Check 2: All metadata has correct length
    for key, vals in metadata.items():
        assert len(vals) == actual_N, f"Metadata {key} length mismatch: {len(vals)} vs {actual_N}"
    print("  ✓ All metadata arrays have correct length")

    # Check 3: Indices align with filenames
    for i, (idx, fname) in enumerate(zip(sorted_indices, filenames)):
        expected_sig, expected_local = parse_filename(fname)
        expected_cond = parse_condition_from_signature(expected_sig)
        actual_cond = idx_to_condition[idx]

        for key in condition_keys:
            if expected_cond.get(key, 0) != actual_cond.get(key, 0):
                raise ValueError(
                    f"Condition mismatch at index {idx}: "
                    f"filename says {key}={expected_cond.get(key)}, "
                    f"shard says {key}={actual_cond.get(key)}"
                )
    print("  ✓ Conditions match between filenames and shard metadata")

    # Check 4: Spot check a few random indices
    rng = np.random.default_rng(42)
    spot_check_indices = rng.choice(sorted_indices, size=min(5, len(sorted_indices)), replace=False)
    for idx in spot_check_indices:
        fname = sorted_filenames[idx]
        sig, local_idx = parse_filename(fname)
        expected_cond = parse_condition_from_signature(sig)

        for key in condition_keys:
            meta_val = metadata[key][sorted_indices.index(idx)].item()
            if expected_cond.get(key, 0) != meta_val:
                raise ValueError(f"Spot check failed at index {idx}")
    print(f"  ✓ Spot check passed for {len(spot_check_indices)} random samples")

    return {
        "features": features_tensor,
        "metadata": metadata,
        "indices": indices_tensor,
        "filenames": filenames,
        "encoder_name": "dinov3-vit-l_meanpatch_aligned",
        "feature_dim": feature_dim,
        "pooling_method": pooling,
        "n_samples": actual_N,
        "n_expected": N,
        "n_missing": len(missing),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate aligned features with index-based joining"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model directory name under outputs/gen/ (e.g., celeba_repa_full)",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "cls", "max"],
        help="Pooling method for patch features",
    )
    parser.add_argument(
        "--condition-keys",
        type=str,
        nargs="+",
        default=None,
        help="Condition attribute keys (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: aligned_{pooling}_features.pt)",
    )
    args = parser.parse_args()

    model_dir = Path("outputs/gen") / args.model
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Consolidating aligned features from {model_dir}")
    print(f"Pooling method: {args.pooling}")

    result = consolidate_aligned_features_by_index(
        model_dir,
        pooling=args.pooling,
        condition_keys=args.condition_keys,
    )

    # Save
    output_name = args.output or f"aligned_{args.pooling}_features.pt"
    output_path = model_dir / output_name
    torch.save(result, output_path)

    print(f"\nSaved to {output_path}")
    print(f"  Shape: {result['features'].shape}")
    print(f"  Samples: {result['n_samples']} / {result['n_expected']} expected")
    if result['n_missing'] > 0:
        print(f"  Warning: {result['n_missing']} samples missing")


if __name__ == "__main__":
    main()
