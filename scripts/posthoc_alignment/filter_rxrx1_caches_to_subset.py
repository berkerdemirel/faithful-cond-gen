"""Step 3 — filter rxrx1 caches to the 50-condition eval subset.

Produces two siblings for REAL feature caches (because scoring and
matched-pair lookups want different pools):

  train_features_subset_scoring.pt
    - filter: sirna_id in {union of subset sirna_ids} (45 sirnas)
    - purpose: marginal modeling (faithfulness composes (0, s), (1, s),
      (2, s), (3, s) to describe sirna s; this cache preserves those
      "sirna columns" across all 4 cell types).

  train_features_subset_pairs.pt
    - filter: (cell_type_id, sirna_id) in {50 subset pairs}
    - purpose: strict matched-pair lookups (downstream classifier test
      set, per-pair KID, etc.).

For GEN feature caches only the strict 50-pair filter is produced
(*_subset.pt sibling), because each gen sample already carries its target
(ct, sirna) and we evaluate exactly the 50 subset conds.

Also partially fills outputs/posthoc_alignment/feature_cache_rxrx1_subset/
with the overlap rows from feature_cache_rxrx1_balanced50/; the remaining
conds are generated in Step 5 and merged in on top.

Nothing is deleted; all outputs are additive.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

SUBSET_JSON = REPO / "outputs/posthoc_alignment/rxrx1_eval_subset_final.json"


def load_subset() -> tuple[set[tuple[int, int]], set[int]]:
    with open(SUBSET_JSON) as f:
        payload = json.load(f)
    rows = payload["seen"] + payload["unseen"]
    pairs = {(int(r["cell_type_id"]), int(r["sirna_id"])) for r in rows}
    sirnas = {int(r["sirna_id"]) for r in rows}
    return pairs, sirnas


def _mask_by_pairs(meta: dict, pairs: set[tuple[int, int]]) -> torch.Tensor:
    ct = meta["cell_type_id"].tolist()
    sr = meta["sirna_id"].tolist()
    return torch.tensor([(int(a), int(b)) in pairs for a, b in zip(ct, sr)], dtype=torch.bool)


def _mask_by_sirnas(meta: dict, sirnas: set[int]) -> torch.Tensor:
    sr = meta["sirna_id"].tolist()
    return torch.tensor([int(b) in sirnas for b in sr], dtype=torch.bool)


def _slice_by_mask(d: dict, mask: torch.Tensor) -> dict:
    n = int(mask.sum().item())
    idx = mask.nonzero(as_tuple=True)[0].tolist()
    out = {}
    for k, v in d.items():
        if k == "features" and isinstance(v, torch.Tensor):
            out[k] = v[mask]
        elif k == "metadata" and isinstance(v, dict):
            new_meta = {}
            for mk, mv in v.items():
                if isinstance(mv, torch.Tensor):
                    new_meta[mk] = mv[mask]
                elif isinstance(mv, list):
                    new_meta[mk] = [mv[i] for i in idx]
                else:
                    new_meta[mk] = mv
            out[k] = new_meta
        elif isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == mask.shape[0]:
            out[k] = v[mask]
        elif isinstance(v, list) and len(v) == mask.shape[0]:
            out[k] = [v[i] for i in idx]
        elif k == "n_samples":
            out[k] = n
        else:
            out[k] = v
    return out


def filter_real_cache(path: Path, pairs: set[tuple[int, int]], sirnas: set[int]) -> None:
    """Produce both _subset_scoring.pt and _subset_pairs.pt siblings."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    meta = d["metadata"]

    pair_mask = _mask_by_pairs(meta, pairs)
    sirna_mask = _mask_by_sirnas(meta, sirnas)
    before = int(d["features"].shape[0])

    pairs_out = _slice_by_mask(d, pair_mask)
    scoring_out = _slice_by_mask(d, sirna_mask)

    pairs_path = path.with_name(path.stem + "_subset_pairs.pt")
    scoring_path = path.with_name(path.stem + "_subset_scoring.pt")
    torch.save(pairs_out, pairs_path)
    torch.save(scoring_out, scoring_path)

    print(
        f"  [{before:>7d}] {path.relative_to(REPO)}"
        f"\n      -> _subset_pairs.pt   : {int(pair_mask.sum()):>6d} rows"
        f"\n      -> _subset_scoring.pt : {int(sirna_mask.sum()):>6d} rows"
    )


def filter_gen_cache(path: Path, pairs: set[tuple[int, int]]) -> None:
    d = torch.load(path, map_location="cpu", weights_only=False)
    mask = _mask_by_pairs(d["metadata"], pairs)
    out = _slice_by_mask(d, mask)
    out_path = path.with_name(path.stem + "_subset.pt")
    torch.save(out, out_path)
    print(
        f"  [{int(d['features'].shape[0]):>7d} -> {int(mask.sum()):>6d}] "
        f"{path.relative_to(REPO)} -> {out_path.name}"
    )


def filter_posthoc_encoded(path: Path, pairs: set[tuple[int, int]], out_path: Path) -> None:
    d = torch.load(path, map_location="cpu", weights_only=False)
    meta = d["gen_meta"]
    ct = meta["cell_type_id"]
    sr = meta["sirna_id"]
    mask = torch.tensor(
        [(int(a), int(b)) in pairs for a, b in zip(ct.tolist(), sr.tolist())],
        dtype=torch.bool,
    )
    kept = int(mask.sum().item())
    out = {
        "gen_siglip": d["gen_siglip"][mask],
        "gen_dinov3": d["gen_dinov3"][mask],
        "gen_hidden": d["gen_hidden"][mask],
        "gen_meta": {
            "cell_type_id": ct[mask],
            "sirna_id": sr[mask],
        },
    }
    n_conds = len({
        (int(a), int(b))
        for a, b in zip(out["gen_meta"]["cell_type_id"].tolist(), out["gen_meta"]["sirna_id"].tolist())
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(
        f"  [{int(ct.shape[0]):>6d} -> {kept:>6d}]  ({n_conds} conds)  "
        f"{path.name} -> {out_path.relative_to(REPO)}"
    )


def cleanup_obsolete_siblings() -> None:
    """Remove any pre-existing `*_subset.pt` REAL sibling (replaced by the two
    new sibling names). Gen and posthoc caches keep `_subset.pt`."""
    to_remove = []
    for p in (REPO / "outputs").glob("real_rxrx1_*/train_features_subset.pt"):
        to_remove.append(p)
    for p in (REPO / "outputs/real_rxrx1_aligned").glob("*/train_features_subset.pt"):
        to_remove.append(p)
    for p in sorted(set(to_remove)):
        p.unlink()
        print(f"  removed obsolete strict-pair real sibling: {p.relative_to(REPO)}")


def main() -> None:
    pairs, sirnas = load_subset()
    print(f"loaded subset: {len(pairs)} pairs, {len(sirnas)} unique sirnas")

    print("\n[cleanup]")
    cleanup_obsolete_siblings()

    print("\n[real feature caches — two siblings each]")
    real_paths = [
        REPO / "outputs/real_rxrx1_dinov3_meanpatch/train_features.pt",
    ]
    real_paths += sorted((REPO / "outputs/real_rxrx1_aligned").glob("*/train_features.pt"))
    # NOTE: real_rxrx1_siglip_meanpatch is intentionally skipped — it is not used
    # for scoring and has no heldout rows, so a subset sibling would be misleading.
    for p in real_paths:
        if p.exists():
            filter_real_cache(p, pairs, sirnas)
        else:
            print(f"  (missing) {p.relative_to(REPO)}")

    print("\n[gen feature caches — strict 50-pair filter]")
    gen_paths = []
    for dirp in sorted((REPO / "outputs/gen").glob("rxrx1_*")):
        for fname in ("dinov3_meanpatch_features.pt", "aligned_mean_features.pt"):
            p = dirp / fname
            if p.exists():
                gen_paths.append(p)
    for p in gen_paths:
        filter_gen_cache(p, pairs)

    print("\n[posthoc-alignment balanced50 -> subset partial]")
    bal_dir = REPO / "outputs/posthoc_alignment/feature_cache_rxrx1_balanced50"
    out_dir = REPO / "outputs/posthoc_alignment/feature_cache_rxrx1_subset"
    if bal_dir.exists():
        for p in sorted(bal_dir.glob("*_encoded.pt")):
            filter_posthoc_encoded(p, pairs, out_dir / p.name)
    else:
        print(f"  (missing) {bal_dir.relative_to(REPO)}")

    print("\ndone.")


if __name__ == "__main__":
    main()
