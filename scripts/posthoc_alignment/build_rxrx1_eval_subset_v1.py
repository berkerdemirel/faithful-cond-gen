"""Build RxRx1 restricted evaluation subset (Option B: 30 controls + 20 seen).

Selection rules:
  controls (sirna_id >= 1108) -- ranked by CV top5 on the full 3420-class probe:
    cell_type=1: top 20
    cell_type=2: top 6
    cell_type=0: top 4
    cell_type=3: 0 (no linearly separable controls)
    => 30 controls, stratified

  seen non-controls (sirna_id < 1108, not in RXRX1_HELDOUT_PAIRS):
    cell_type=1 only -- the only tier with n >= 25 per condition
    rank by (n desc, top5 desc), take top 20
    => 20 seen conditions with meaningful KID footprint

After selecting, rerun a 5-fold stratified CV linear probe restricted to just
these 50 classes and report per-condition top1 / top5. Saves:
  outputs/posthoc_alignment/rxrx1_eval_subset_v1.json
"""

import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from faithful_cond_gen.eval.trust_eval.config import RXRX1_HELDOUT_PAIRS

FEAT_PATH = "outputs/real_rxrx1_dinov3_meanpatch/train_features.pt"
PROBE_PATH = "outputs/posthoc_alignment/rxrx1_condition_cv_probe.json"
OUT_PATH = "outputs/posthoc_alignment/rxrx1_eval_subset_v1.json"
N_CONTROLS_PER_CT = {0: 4, 1: 20, 2: 6, 3: 0}
N_SEEN_CT1 = 20
SEEN_MIN_N = 25
K_FOLDS = 5
EPOCHS = 60
LR = 1e-3
WD = 1e-4
BATCH = 2048
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pick_subset(probe_conds):
    controls = [c for c in probe_conds if c["sirna_id"] >= 1108]
    seen_nc = [
        c for c in probe_conds
        if c["sirna_id"] < 1108
        and (c["cell_type_id"], c["sirna_id"]) not in RXRX1_HELDOUT_PAIRS
    ]
    by_ct_ctrl = defaultdict(list)
    for c in controls:
        by_ct_ctrl[c["cell_type_id"]].append(c)
    chosen_controls = []
    for ct, k in N_CONTROLS_PER_CT.items():
        cs = sorted(by_ct_ctrl.get(ct, []), key=lambda c: -c["top5"])
        chosen_controls.extend(cs[:k])

    ct1 = [c for c in seen_nc if c["cell_type_id"] == 1 and c["n"] >= SEEN_MIN_N]
    ct1_sorted = sorted(ct1, key=lambda c: (-c["n"], -c["top5"]))
    chosen_seen = ct1_sorted[:N_SEEN_CT1]

    return chosen_controls, chosen_seen


def run_restricted_cv(chosen_pairs):
    d = torch.load(FEAT_PATH, map_location="cpu", weights_only=False)
    feats = d["features"]
    ct = d["metadata"]["cell_type_id"].tolist()
    si = d["metadata"]["sirna_id"].tolist()
    pair_set = set(chosen_pairs)
    mask = np.array([(ct[i], si[i]) in pair_set for i in range(len(ct))])
    feats_f = feats[torch.from_numpy(mask)]
    pairs_f = [(ct[i], si[i]) for i in np.where(mask)[0]]
    uniq = sorted(set(pairs_f))
    pair2idx = {p: i for i, p in enumerate(uniq)}
    labels = np.array([pair2idx[p] for p in pairs_f], dtype=np.int64)
    X = F.normalize(feats_f, dim=-1).numpy().astype(np.float32)
    N, D = X.shape
    C = len(uniq)
    print(f"\nrestricted CV: {N} samples  {C} classes  chance_top1={1/C:.3f}  "
          f"chance_top5={5/C:.3f}")

    kf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    oof_top1 = np.zeros(N, dtype=bool)
    oof_top5 = np.zeros(N, dtype=bool)
    for fold, (tr, va) in enumerate(kf.split(X, labels)):
        Xtr = torch.from_numpy(X[tr]).to(DEVICE)
        ytr = torch.from_numpy(labels[tr]).long().to(DEVICE)
        Xva = torch.from_numpy(X[va]).to(DEVICE)
        yva = torch.from_numpy(labels[va]).long().to(DEVICE)
        probe = nn.Linear(D, C, bias=True).to(DEVICE)
        opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WD)
        for ep in range(EPOCHS):
            probe.train()
            idx = torch.randperm(len(Xtr), device=DEVICE)
            for i in range(0, len(Xtr), BATCH):
                b = idx[i:i + BATCH]
                loss = F.cross_entropy(probe(Xtr[b]), ytr[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        probe.eval()
        with torch.no_grad():
            logits = probe(Xva)
            topk = logits.topk(5, dim=-1).indices
            top1 = (topk[:, 0] == yva).cpu().numpy()
            top5 = (topk == yva[:, None]).any(dim=-1).cpu().numpy()
        oof_top1[va] = top1
        oof_top5[va] = top5
        print(f"  fold {fold}: top1={top1.mean():.4f}  top5={top5.mean():.4f}")

    per_top1 = np.zeros(C)
    per_top5 = np.zeros(C)
    per_n = np.zeros(C, dtype=int)
    for i, y in enumerate(labels):
        per_top1[y] += oof_top1[i]
        per_top5[y] += oof_top5[i]
        per_n[y] += 1
    per_top1 /= per_n
    per_top5 /= per_n

    records = []
    for i, p in enumerate(uniq):
        is_control = p[1] >= 1108
        is_heldout = p in RXRX1_HELDOUT_PAIRS
        records.append(dict(
            cell_type_id=int(p[0]),
            sirna_id=int(p[1]),
            n=int(per_n[i]),
            top1=float(per_top1[i]),
            top5=float(per_top5[i]),
            is_control=bool(is_control),
            is_heldout=bool(is_heldout),
        ))
    return dict(
        overall_top1=float(oof_top1.mean()),
        overall_top5=float(oof_top5.mean()),
        n_classes=int(C),
        records=records,
    )


def main():
    probe = json.load(open(PROBE_PATH))
    controls, seen = pick_subset(probe["conditions"])
    pairs = [(c["cell_type_id"], c["sirna_id"]) for c in controls + seen]
    print(f"picked {len(controls)} controls + {len(seen)} seen = {len(pairs)} conds")

    out = run_restricted_cv(pairs)

    ctrl_records = [r for r in out["records"] if r["is_control"]]
    seen_records = [r for r in out["records"] if not r["is_control"]]
    ctrl_records.sort(key=lambda r: (-r["top1"], -r["top5"]))
    seen_records.sort(key=lambda r: (-r["top1"], -r["top5"]))

    print(f"\nRESTRICTED CV over {out['n_classes']} classes")
    print(f"  OVERALL top1={out['overall_top1']:.4f}  top5={out['overall_top5']:.4f}")
    print(f"  controls top1 mean={np.mean([r['top1'] for r in ctrl_records]):.4f}  "
          f"top5 mean={np.mean([r['top5'] for r in ctrl_records]):.4f}")
    print(f"  seen     top1 mean={np.mean([r['top1'] for r in seen_records]):.4f}  "
          f"top5 mean={np.mean([r['top5'] for r in seen_records]):.4f}")

    print("\n--- 30 CONTROLS (stratified ct=0:4 / ct=1:20 / ct=2:6) ---")
    print(f"{'rk':>3} {'ct':>3} {'sirna':>5} {'n':>4}  top1  top5  heldout")
    for i, r in enumerate(ctrl_records):
        print(f"{i+1:>3} {r['cell_type_id']:>3} {r['sirna_id']:>5} {r['n']:>4}  "
              f"{r['top1']:.3f} {r['top5']:.3f}  {'Y' if r['is_heldout'] else 'N'}")

    print("\n--- 20 SEEN non-controls (ct=1, n>=25, ranked) ---")
    print(f"{'rk':>3} {'ct':>3} {'sirna':>5} {'n':>4}  top1  top5")
    for i, r in enumerate(seen_records):
        print(f"{i+1:>3} {r['cell_type_id']:>3} {r['sirna_id']:>5} {r['n']:>4}  "
              f"{r['top1']:.3f} {r['top5']:.3f}")

    with open(OUT_PATH, "w") as f:
        json.dump(dict(
            meta=dict(
                feat_path=FEAT_PATH,
                n_controls_per_ct=N_CONTROLS_PER_CT,
                n_seen_ct1=N_SEEN_CT1,
                seen_min_n=SEEN_MIN_N,
                k_folds=K_FOLDS,
                epochs=EPOCHS,
                overall_top1=out["overall_top1"],
                overall_top5=out["overall_top5"],
                n_classes=out["n_classes"],
            ),
            controls=ctrl_records,
            seen=seen_records,
        ), f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
