"""Canonical RxRx1 50-condition eval subset loader.

The subset JSON is built by scripts/posthoc_alignment/finalize_rxrx1_subset.py
and all rxrx1 eval paths should filter to it by calling load_rxrx1_subset().
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Set, Tuple

import torch

_REPO = Path(__file__).resolve().parents[4]
SUBSET_JSON = _REPO / "outputs/posthoc_alignment/rxrx1_eval_subset_final.json"


@lru_cache(maxsize=1)
def load_rxrx1_subset() -> Set[Tuple[int, int]]:
    """Return the canonical RxRx1 eval subset pairs as {(cell_type_id, sirna_id)}.

    Raises FileNotFoundError if the JSON is missing.
    """
    if not SUBSET_JSON.exists():
        raise FileNotFoundError(
            f"RxRx1 eval subset JSON missing at {SUBSET_JSON}. "
            "Run scripts/posthoc_alignment/finalize_rxrx1_subset.py first."
        )
    with open(SUBSET_JSON) as f:
        payload = json.load(f)
    rows = payload["seen"] + payload["unseen"]
    pairs = {(int(r["cell_type_id"]), int(r["sirna_id"])) for r in rows}
    if len(pairs) != 50:
        raise ValueError(f"RxRx1 subset JSON has {len(pairs)} unique pairs, expected 50")
    return pairs


@lru_cache(maxsize=1)
def load_rxrx1_subset_sirnas() -> frozenset[int]:
    """Return the set of unique sirna_ids used by the canonical eval subset.

    This is the filter for the "scoring pool" on the real side: scoring
    composes marginals over (cell_type, sirna), so every sirna in the subset
    needs its complete cross-cell-type column from real training data.
    """
    with open(SUBSET_JSON) as f:
        payload = json.load(f)
    rows = payload["seen"] + payload["unseen"]
    return frozenset(int(r["sirna_id"]) for r in rows)


@lru_cache(maxsize=1)
def load_rxrx1_subset_arms() -> dict:
    """Return {'seen': set[(ct, sirna)], 'unseen': set[(ct, sirna)]}."""
    with open(SUBSET_JSON) as f:
        payload = json.load(f)
    return {
        "seen": {(int(r["cell_type_id"]), int(r["sirna_id"])) for r in payload["seen"]},
        "unseen": {(int(r["cell_type_id"]), int(r["sirna_id"])) for r in payload["unseen"]},
    }


def _apply_mask(features: torch.Tensor, metadata: dict, mask: torch.Tensor):
    idx = mask.nonzero(as_tuple=True)[0].tolist()
    new_feats = features[mask] if isinstance(features, torch.Tensor) else [features[i] for i in idx]
    new_meta = {}
    for k, v in metadata.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == mask.shape[0]:
            new_meta[k] = v[mask]
        elif isinstance(v, list) and len(v) == mask.shape[0]:
            new_meta[k] = [v[i] for i in idx]
        else:
            new_meta[k] = v
    return new_feats, new_meta


def filter_rxrx1_to_subset(features: torch.Tensor, metadata: dict):
    """Strict 50-pair filter: keep only rows whose (ct, sirna) is in the subset.

    Use this for gen features and for any real path that must present exactly
    the 50 subset (ct, sirna) classes (e.g. matched-pair test sets).
    """
    subset = load_rxrx1_subset()
    ct = metadata["cell_type_id"]
    sr = metadata["sirna_id"]
    ct_list = ct.tolist() if isinstance(ct, torch.Tensor) else list(ct)
    sr_list = sr.tolist() if isinstance(sr, torch.Tensor) else list(sr)
    mask = torch.tensor(
        [(int(a), int(b)) in subset for a, b in zip(ct_list, sr_list)],
        dtype=torch.bool,
    )
    return _apply_mask(features, metadata, mask)


def filter_rxrx1_real_to_scoring_pool(features: torch.Tensor, metadata: dict):
    """Sirna-column filter: keep rows whose sirna_id is in the subset's sirna set.

    Use this for the REAL feature pool that drives scoring (faithfulness,
    realism). Scoring composes marginals over (ct, sirna), so every subset
    sirna needs its complete column of real data across all 4 cell types.
    """
    sirnas = load_rxrx1_subset_sirnas()
    sr = metadata["sirna_id"]
    sr_list = sr.tolist() if isinstance(sr, torch.Tensor) else list(sr)
    mask = torch.tensor([int(b) in sirnas for b in sr_list], dtype=torch.bool)
    return _apply_mask(features, metadata, mask)
