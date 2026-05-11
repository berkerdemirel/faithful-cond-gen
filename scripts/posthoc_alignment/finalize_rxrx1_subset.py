"""Finalize the principled RxRx1 50-condition eval subset.

Selection rules (see notes/rxrx1_subset_migration_prompt.md):
  Unseen (25): from RXRX1_HELDOUT_PAIRS
    - 8 ct=0 controls (sirna_id >= 1108)
    - 7 ct=1 non-control heldouts (sirna_id < 1108)
    - 2 ct=1 top-size controls
    - 8 ct=2 controls
    picked by (-n, cell_type_id, sirna_id)
  Seen (25):
    - 20 ct=1 non-controls (sirna_id < 1108, not heldout) by (-n, cell_type_id, sirna_id)
    - 5 ct=3 controls (sirna_id in [1108, 1138], not heldout) by (-n, sirna_id)

Writes outputs/posthoc_alignment/rxrx1_eval_subset_final.json.
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from faithful_cond_gen.eval.trust_eval.config import RXRX1_HELDOUT_PAIRS  # noqa: E402

REAL_FEATS = REPO / "outputs/real_rxrx1_dinov3_meanpatch/train_features.pt"
OUT_JSON = REPO / "outputs/posthoc_alignment/rxrx1_eval_subset_final.json"

MIN_N = 22  # floor per prompt


def load_pair_counts() -> dict[tuple[int, int], int]:
    d = torch.load(REAL_FEATS, map_location="cpu", weights_only=False)
    ct = d["metadata"]["cell_type_id"].tolist()
    sr = d["metadata"]["sirna_id"].tolist()
    return collections.Counter(zip(ct, sr))


def pick(pool, k):
    # deterministic tie-break: (-n, cell_type_id, sirna_id)
    pool = sorted(pool, key=lambda r: (-r["n"], r["cell_type_id"], r["sirna_id"]))
    return pool[:k]


def main() -> None:
    counts = load_pair_counts()

    seen_all = [
        {"cell_type_id": c, "sirna_id": s, "n": n}
        for (c, s), n in counts.items()
        if (c, s) not in RXRX1_HELDOUT_PAIRS
    ]
    unseen_all = [
        {"cell_type_id": c, "sirna_id": s, "n": n}
        for (c, s), n in counts.items()
        if (c, s) in RXRX1_HELDOUT_PAIRS
    ]

    # --- Unseen arm (25)
    unseen_ct0_ctrl = [r for r in unseen_all if r["cell_type_id"] == 0 and r["sirna_id"] >= 1108]
    unseen_ct1_nc = [r for r in unseen_all if r["cell_type_id"] == 1 and r["sirna_id"] < 1108]
    unseen_ct1_ctrl = [r for r in unseen_all if r["cell_type_id"] == 1 and r["sirna_id"] >= 1108]
    unseen_ct2_ctrl = [r for r in unseen_all if r["cell_type_id"] == 2 and r["sirna_id"] >= 1108]

    unseen = (
        pick(unseen_ct0_ctrl, 8)
        + pick(unseen_ct1_nc, 7)
        + pick(unseen_ct1_ctrl, 2)
        + pick(unseen_ct2_ctrl, 8)
    )
    assert len(unseen) == 25, f"unseen has {len(unseen)} rows"

    # --- Seen arm (25)
    seen_ct1_nc = [
        r for r in seen_all
        if r["cell_type_id"] == 1 and r["sirna_id"] < 1108
    ]
    seen_ct3_ctrl = [
        r for r in seen_all
        if r["cell_type_id"] == 3 and 1108 <= r["sirna_id"] <= 1138
    ]
    seen_rows = pick(seen_ct1_nc, 20) + pick(seen_ct3_ctrl, 5)
    assert len(seen_rows) == 25, f"seen has {len(seen_rows)} rows"

    # --- Validation
    picked_pairs = {(r["cell_type_id"], r["sirna_id"]) for r in unseen + seen_rows}
    for r in unseen + seen_rows:
        assert r["n"] >= MIN_N, f"pair {(r['cell_type_id'], r['sirna_id'])} has n={r['n']} < {MIN_N}"
    assert all((r["cell_type_id"], r["sirna_id"]) in RXRX1_HELDOUT_PAIRS for r in unseen)
    assert not any((r["cell_type_id"], r["sirna_id"]) in RXRX1_HELDOUT_PAIRS for r in seen_rows)
    assert len(picked_pairs) == 50, f"dup: {len(picked_pairs)}"
    cts_union = {r["cell_type_id"] for r in unseen + seen_rows}
    assert cts_union == {0, 1, 2, 3}, f"missing cts: {cts_union}"

    payload = {
        "meta": {
            "n_seen": 25,
            "n_unseen": 25,
            "built_at": dt.datetime.utcnow().isoformat() + "Z",
            "source_pair_counts": str(REAL_FEATS.relative_to(REPO)),
        },
        "seen": seen_rows,
        "unseen": unseen,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    # --- Sanity report
    def histo(rows, label):
        by_ct = collections.Counter(r["cell_type_id"] for r in rows)
        ns = sorted(r["n"] for r in rows)
        total = sum(ns)
        median = ns[len(ns) // 2]
        print(
            f"[{label}] n_cond={len(rows):2d}  "
            f"by_ct={dict(sorted(by_ct.items()))}  "
            f"median_n={median}  total_samples={total}  "
            f"min_n={ns[0]} max_n={ns[-1]}"
        )

    print(f"wrote {OUT_JSON}")
    histo(seen_rows, "SEEN ")
    histo(unseen, "UNSEEN")
    histo(seen_rows + unseen, "TOTAL")


if __name__ == "__main__":
    main()
