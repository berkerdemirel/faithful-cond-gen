"""Per-condition FPR95 on 50 random conditions with enough real samples.

Pick 50 conditions that have >=20 real samples (so KID is not pure noise).
For each: top 50% by score vs random 50%, KID in DINO space.
Compare aligned_mean vs dinov3 scoring.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from faithful_cond_gen.eval.trust_eval.config import CONDITION_ATTRS, RXRX1_HELDOUT_PAIRS
from faithful_cond_gen.eval.trust_eval.feature_io import load_features_for_dataset
from faithful_cond_gen.eval.trust_eval.condition_utils import filter_feats_and_meta_by_seen_combos
from faithful_cond_gen.eval.trust_eval.scoring_core import fit_trust_scoring_components, score_trust_from_components
from faithful_cond_gen.eval.trust_eval.metrics_kid import calculate_kid_same_m

ck = CONDITION_ATTRS["rxrx1"]

# Load DINO features (for KID)
print("Loading DINO features...", flush=True)
kid_rf, kid_rm, kid_gf, _ = load_features_for_dataset("rxrx1", "repa_marginal", "dinov3", normalize_mode="l2")
kid_gen_np = kid_gf.numpy()
kid_real_np = kid_rf.numpy()

# Build real_by_cond for DINO
kid_real_by_cond = {}
for i in range(len(kid_rf)):
    ct = int(kid_rm["cell_type_id"][i].item())
    sirna = int(kid_rm["sirna_id"][i].item())
    kid_real_by_cond.setdefault((ct, sirna), []).append(i)

# Pick 50 conditions with >=20 real samples
rng = np.random.default_rng(42)
eligible = [c for c, idx in kid_real_by_cond.items() if len(idx) >= 20]
chosen = [eligible[i] for i in rng.choice(len(eligible), size=min(50, len(eligible)), replace=False)]
print(f"Selected {len(chosen)} conditions with >=20 real samples (out of {len(eligible)} eligible)")
n_heldout = sum(1 for c in chosen if c in RXRX1_HELDOUT_PAIRS)
print(f"  {n_heldout} heldout, {len(chosen)-n_heldout} seen")

# Compute seen combos
real_f_tmp, real_m_tmp, _, _ = load_features_for_dataset("rxrx1", "repa_marginal", "aligned_mean", normalize_mode="l2")
all_combos = set()
for i in range(len(real_f_tmp)):
    c = tuple(int(real_m_tmp[k][i].item()) for k in ck)
    all_combos.add(c)
seen_combos = all_combos - RXRX1_HELDOUT_PAIRS
del real_f_tmp, real_m_tmp

# Gen conditions (same for both feature types — same generated images)
_, _, _, gen_m_tmp = load_features_for_dataset("rxrx1", "repa_marginal", "dinov3", normalize_mode="l2")
gen_conds = [tuple(int(gen_m_tmp[k][i].item()) for k in ck) for i in range(len(kid_gf))]
gen_by_cond = {}
for i, c in enumerate(gen_conds):
    gen_by_cond.setdefault(c, []).append(i)
del gen_m_tmp

for ft in ["dinov3", "aligned_mean"]:
    print(f"\n{'='*60}")
    print(f"repa_marginal/{ft} — Per-condition FPR95 (50 conditions, >=20 real)")
    print(f"{'='*60}", flush=True)

    real_f, real_m, gen_f, gen_m = load_features_for_dataset("rxrx1", "repa_marginal", ft, normalize_mode="l2")
    calib_f, calib_m = filter_feats_and_meta_by_seen_combos(real_f, real_m, ck, seen_combos)
    print(f"  Fitting scoring model ({calib_f.shape[0]} calib samples)...", flush=True)
    comp = fit_trust_scoring_components(calib_f, calib_m, ck)
    _, _, gen_scores = score_trust_from_components(gen_f, gen_m, comp)

    improvements = []
    results_per_cond = []

    for cond in chosen:
        gen_idx = np.array(gen_by_cond.get(cond, []))
        real_idx = np.array(kid_real_by_cond.get(cond, []))
        if len(gen_idx) < 10 or len(real_idx) < 10:
            continue

        cond_scores = gen_scores[gen_idx]
        valid = np.isfinite(cond_scores)
        if valid.sum() < 10:
            continue

        # Top 50% by score
        n_accept = valid.sum() // 2
        sorted_local = np.argsort(cond_scores[valid])
        accept_local = sorted_local[:n_accept]
        accept_global = gen_idx[np.where(valid)[0][accept_local]]

        k = min(len(accept_global), len(real_idx), 200)
        if k < 10:
            continue

        # KID: accepted vs real (5 repeats)
        kid_acc_vals = []
        kid_rand_vals = []
        for rep in range(5):
            rr = np.random.default_rng(42 + rep)
            pg = rr.permutation(len(accept_global))[:k]
            pr = rr.permutation(len(real_idx))[:k]
            kv = calculate_kid_same_m(kid_gen_np[accept_global[pg]], kid_real_np[real_idx[pr]], use_cosine=True)
            if np.isfinite(kv):
                kid_acc_vals.append(kv)

            # Random 50%
            rand_idx = rr.choice(gen_idx, size=len(accept_global), replace=False) if len(gen_idx) >= len(accept_global) else gen_idx
            pg2 = rr.permutation(len(rand_idx))[:k]
            pr2 = rr.permutation(len(real_idx))[:k]
            kv2 = calculate_kid_same_m(kid_gen_np[rand_idx[pg2]], kid_real_np[real_idx[pr2]], use_cosine=True)
            if np.isfinite(kv2):
                kid_rand_vals.append(kv2)

        if kid_acc_vals and kid_rand_vals:
            ka = np.mean(kid_acc_vals)
            kr = np.mean(kid_rand_vals)
            imp = (kr - ka) / kr * 100 if kr != 0 else 0
            improvements.append(imp)
            tag = "HELD" if cond in RXRX1_HELDOUT_PAIRS else "SEEN"
            results_per_cond.append((cond, len(accept_global), len(real_idx), ka, kr, imp, tag))

    # Sort by improvement
    results_per_cond.sort(key=lambda x: x[5], reverse=True)

    print(f"\n  Evaluated {len(improvements)} conditions")
    print(f"  Mean improvement: {np.mean(improvements):+.2f}%")
    print(f"  Median improvement: {np.median(improvements):+.2f}%")
    print(f"  Positive: {sum(1 for i in improvements if i > 0)}/{len(improvements)}")

    print(f"\n  {'Cond':<15} {'n_acc':>6} {'n_real':>6} {'KID_acc':>9} {'KID_rand':>9} {'Δ%':>8} {'Type':>5}")
    print(f"  {'-'*60}")
    for cond, na, nr, ka, kr, imp, tag in results_per_cond:
        print(f"  {str(cond):<15} {na:>6} {nr:>6} {ka:>9.4f} {kr:>9.4f} {imp:>+7.1f}% {tag:>5}")

print("\nDone.", flush=True)
