"""
Condition handling and binning utilities.

Functions for extracting, filtering, and binning samples by condition.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch


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


def filter_feats_and_meta_by_seen_combos(
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
    meta_f: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == N:
            meta_f[k] = v[keep]
        elif isinstance(v, list) and len(v) == N:
            meta_f[k] = [vv for vv, kk in zip(v, keep_list) if kk]
        else:
            meta_f[k] = v

    return feats_f, meta_f


# Backward-compatible alias (underscore-prefixed version used in original code)
_filter_feats_and_meta_by_seen_combos = filter_feats_and_meta_by_seen_combos


def build_condition_class_map(
    conditions: List[Tuple[int, ...]],
) -> Tuple[Dict[Tuple[int, ...], int], List[Tuple[int, ...]]]:
    """
    Build deterministic mapping from condition tuples to class IDs.

    Args:
        conditions: List of condition tuples (e.g., [(0,0,0,0), (1,0,0,0), ...])

    Returns:
        Tuple of:
        - cond_to_class: Dict mapping condition tuple -> class ID
        - class_to_cond: List where index is class ID, value is condition tuple
    """
    unique_conds = sorted(set(conditions))
    cond_to_class = {cond: i for i, cond in enumerate(unique_conds)}
    class_to_cond = unique_conds
    return cond_to_class, class_to_cond


def bin_samples_within_conditioning(
    conditions: List[Tuple[int, ...]],
    scores: np.ndarray,
    n_bins: int,
    ascending: bool = True,
) -> Dict[Tuple[int, ...], List[np.ndarray]]:
    """
    Bin samples by score within each conditioning group.

    Args:
        conditions: List of condition tuples per sample
        scores: Score array (N,) - lower is better if ascending=True
        n_bins: Number of bins per condition
        ascending: If True, bin 0 has lowest scores (best), bin n-1 has highest (worst)

    Returns:
        Dict[condition] -> List of n_bins np.ndarray index arrays
        Each inner array contains global indices into the original sample order.
    """
    # Group sample indices by condition
    cond_to_indices: Dict[Tuple[int, ...], List[int]] = {}
    for i, cond in enumerate(conditions):
        cond_to_indices.setdefault(cond, []).append(i)

    result: Dict[Tuple[int, ...], List[np.ndarray]] = {}

    for cond, indices in cond_to_indices.items():
        indices_arr = np.array(indices, dtype=int)
        cond_scores = scores[indices_arr]

        # Sort by score (ascending=True means lower scores first = best)
        sort_order = np.argsort(cond_scores)
        if not ascending:
            sort_order = sort_order[::-1]

        sorted_indices = indices_arr[sort_order]

        # Split into n_bins equal-sized bins
        bin_size = len(sorted_indices) // n_bins
        bins = []
        for b in range(n_bins):
            start = b * bin_size
            end = (b + 1) * bin_size if b < n_bins - 1 else len(sorted_indices)
            bins.append(sorted_indices[start:end])

        result[cond] = bins

    return result
