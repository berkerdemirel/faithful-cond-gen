"""
Global FPR95 with distribution analysis and corrected random baseline.

1. Fit scoring on seen-combo reals, score ALL real + gen
2. Set single FPR95 threshold from seen-combo real scores
3. Accept gen samples below threshold
4. Analyze: which conditions get accepted? How does the histogram shift?
5. KID comparison:
   - SELECTED: accepted gen vs condition-matched real (matched to accepted histogram)
   - RANDOM:   truly random gen (same N) vs condition-matched real (matched to random histogram)
"""
import glob
import numpy as np
import torch
from scipy.stats import spearmanr

from faithful_cond_gen.eval.trust_eval.config import (
    CONDITION_ATTRS, MARGINAL_SEEN_COMBOS,
)
from faithful_cond_gen.eval.trust_eval.condition_utils import (
    filter_feats_and_meta_by_seen_combos,
)
from faithful_cond_gen.eval.trust_eval.scoring_core import (
    fit_trust_scoring_components, score_trust_from_components,
)
from faithful_cond_gen.eval.trust_eval.metrics_kid import calculate_kid_same_m
from faithful_cond_gen.posthoc_alignment.mapper import ResidualAlignmentMapper

CK = CONDITION_ATTRS["celeba"]
SEED = 42
N_BOOT = 10
N_RAND = 10
KID_K = 500


def l2(x):
    if isinstance(x, torch.Tensor):
        return x / (x.norm(dim=1, keepdim=True) + 1e-12)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def get_cond(meta, i):
    return tuple(int(meta[k][i]) for k in CK)


def group_by_cond(meta, n):
    out = {}
    for i in range(n):
        c = get_cond(meta, i)
        out.setdefault(c, []).append(i)
    return out


def build_cond_matched_real(cond_hist, kid_real, kid_real_by_cond, seed):
    """Build condition-matched real pool from a condition histogram."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in sorted(cond_hist.keys()):
        rc = kid_real_by_cond.get(c, [])
        n_need = cond_hist[c]
        if rc:
            idx.extend(rng.choice(rc, n_need, replace=(len(rc) < n_need)))
    return np.array(idx, dtype=int)


def bootstrap_kid(gen_feats, real_feats, k=KID_K, n_boot=N_BOOT, seed=SEED):
    eff_k = min(k, len(gen_feats), len(real_feats))
    if eff_k < 10:
        return np.nan, np.nan
    vals = []
    for b in range(n_boot):
        rng = np.random.default_rng(seed + b)
        gi = rng.permutation(len(gen_feats))[:eff_k]
        ri = rng.permutation(len(real_feats))[:eff_k]
        kv = calculate_kid_same_m(gen_feats[gi], real_feats[ri], use_cosine=True)
        if np.isfinite(kv):
            vals.append(kv)
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, np.nan)


def cond_histogram(meta, indices):
    """Get condition histogram for a set of indices."""
    hist = {}
    for i in indices:
        c = get_cond(meta, i)
        hist[c] = hist.get(c, 0) + 1
    return hist


for model in ["celeba_vanilla_marginal_v1", "celeba_repa_siglip_marginal_v1"]:
    short = model.replace("celeba_", "").replace("_v1", "")
    print(f"\n{'='*70}")
    print(f"MODEL: {model} ({short})")
    print(f"{'='*70}")

    # --- Load posthoc_mapped features ---
    mapper = ResidualAlignmentMapper(768, 1152)
    mapper.load_state_dict(torch.load(
        f"outputs/posthoc_alignment/mappers/{model}/best_mapper.pt",
        map_location="cpu", weights_only=True))
    mapper.eval()

    raw = torch.load(f"outputs/posthoc_alignment/raw_hidden/{model}/t0.01_hidden.pt",
                     map_location="cpu", weights_only=False)
    with torch.no_grad():
        real_mapped = l2(mapper(raw["features"]))
    real_mapped_meta = {k: raw["metadata"][k] for k in CK}

    pts = sorted(glob.glob(f"outputs/posthoc_alignment/diag/{model}/gen_cache/cond_*.pt"))
    hids, metas = [], {k: [] for k in CK}
    for p in pts:
        d = torch.load(p, map_location="cpu", weights_only=False)
        hids.append(d["raw_hidden"])
        cond = d["condition"]
        n = d["raw_hidden"].shape[0]
        for ki, k in enumerate(CK):
            metas[k].append(torch.full((n,), cond[ki], dtype=torch.long))
    gen_hidden = torch.cat(hids)
    gen_meta = {k: torch.cat(metas[k]) for k in CK}
    with torch.no_grad():
        gen_mapped = l2(mapper(gen_hidden))
    del mapper

    # --- Load DINO for KID ---
    cache = torch.load(f"/mnt/pvc/posthoc_debug/feature_cache/{model}_encoded.pt",
                       map_location="cpu", weights_only=False)
    kid_gen = l2(cache["gen_dinov3"]).numpy()
    kid_gen_meta = cache["gen_meta"]

    real_dino = torch.load("outputs/real_celeba_dinov3_meanpatch/train_features.pt",
                           map_location="cpu", weights_only=False)
    kid_real = l2(real_dino["features"]).numpy()
    kid_real_meta = real_dino.get("metadata", {})

    n_gen = len(gen_mapped)
    n_real = len(real_mapped)
    print(f"Gen: {n_gen}, Real: {n_real}")

    # --- Fit on seen combos, score ALL real + gen ---
    calib_feats, calib_meta = filter_feats_and_meta_by_seen_combos(
        real_mapped, real_mapped_meta, CK, MARGINAL_SEEN_COMBOS
    )
    components = fit_trust_scoring_components(calib_feats, calib_meta, CK)

    gen_r, gen_f, gen_t = score_trust_from_components(gen_mapped, gen_meta, components)
    calib_r, calib_f, calib_t = score_trust_from_components(calib_feats, calib_meta, components)

    gen_scores = {"trust": gen_t, "realism": gen_r, "faithfulness": gen_f}
    calib_scores = {"trust": calib_t, "realism": calib_r, "faithfulness": calib_f}

    # Groupings
    gen_by_cond = group_by_cond(gen_meta, n_gen)
    kid_real_by_cond = group_by_cond(kid_real_meta, len(kid_real))

    # Original gen histogram (uniform: 1000 per condition)
    all_gen_idx = np.arange(n_gen)
    orig_hist = cond_histogram(gen_meta, all_gen_idx)

    # --- Global FPR95 with distribution analysis ---
    print(f"\n--- Global FPR95 (threshold from seen-combo real) ---")
    for comp in ["trust", "realism", "faithfulness"]:
        gs = gen_scores[comp]
        rs = calib_scores[comp]

        valid_r = np.isfinite(rs)
        threshold = float(np.percentile(rs[valid_r], 95))

        accept_mask = np.isfinite(gs) & (gs <= threshold)
        accept_idx = np.where(accept_mask)[0]
        n_acc = len(accept_idx)
        acc_rate = n_acc / n_gen

        # --- Distribution analysis ---
        acc_hist = cond_histogram(gen_meta, accept_idx)

        print(f"\n  {comp.upper()} (acc={acc_rate*100:.1f}%, {n_acc}/{n_gen}, threshold={threshold:.4f})")
        print(f"  {'Condition':<35s} {'Original':>8s} {'Accepted':>8s} {'Acc%':>6s} {'Seen?':>5s}")
        print(f"  {'-'*70}")

        seen_count = 0
        unseen_count = 0
        for c in sorted(set(list(orig_hist.keys()) + list(acc_hist.keys()))):
            n_orig = orig_hist.get(c, 0)
            n_sel = acc_hist.get(c, 0)
            pct = n_sel / n_orig * 100 if n_orig > 0 else 0
            is_seen = "Y" if c in MARGINAL_SEEN_COMBOS else ""
            if is_seen:
                seen_count += n_sel
            else:
                unseen_count += n_sel
            print(f"  {str(c):<35s} {n_orig:>8d} {n_sel:>8d} {pct:>5.1f}% {is_seen:>5s}")

        print(f"  {'':35s} {'':>8s} {'':>8s}")
        print(f"  Seen conditions:   {seen_count:>6d} ({seen_count/n_acc*100:.1f}% of accepted)")
        print(f"  Unseen conditions: {unseen_count:>6d} ({unseen_count/n_acc*100:.1f}% of accepted)")

        # --- KID: selected vs truly random ---
        # SELECTED: accepted gen vs condition-matched real (matched to accepted histogram)
        sel_real_idx = build_cond_matched_real(acc_hist, kid_real, kid_real_by_cond, SEED + 100)
        kid_sel_m, kid_sel_s = bootstrap_kid(
            kid_gen[accept_idx], kid_real[sel_real_idx],
            k=KID_K, n_boot=N_BOOT, seed=SEED + 200)

        # RANDOM: truly random gen (same N) vs condition-matched real (matched to RANDOM's histogram)
        rand_vals = []
        for rep in range(N_RAND):
            rng = np.random.default_rng(SEED + 300 + rep)
            rand_idx = rng.choice(n_gen, n_acc, replace=False)
            rand_hist = cond_histogram(gen_meta, rand_idx)
            rand_real_idx = build_cond_matched_real(rand_hist, kid_real, kid_real_by_cond, SEED + 400 + rep)

            km, _ = bootstrap_kid(
                kid_gen[rand_idx], kid_real[rand_real_idx],
                k=KID_K, n_boot=1, seed=SEED + 500 + rep)
            if np.isfinite(km):
                rand_vals.append(km)

        kid_rand_m = float(np.mean(rand_vals)) if rand_vals else np.nan
        kid_rand_s = float(np.std(rand_vals)) if rand_vals else np.nan
        delta = kid_sel_m - kid_rand_m
        delta_pct = delta / kid_rand_m * 100 if kid_rand_m != 0 else np.nan

        print(f"\n  KID_selected  = {kid_sel_m:.4f} +/- {kid_sel_s:.4f}  (accepted gen vs cond-matched real)")
        print(f"  KID_random    = {kid_rand_m:.4f} +/- {kid_rand_s:.4f}  (random gen vs its own cond-matched real)")
        print(f"  delta         = {delta:+.4f} ({delta_pct:+.1f}%)  (negative = selected is better)")

    # --- Per-condition FPR95 for comparison ---
    all_real_r, all_real_f, all_real_t = score_trust_from_components(
        real_mapped, real_mapped_meta, components
    )
    all_real_scores = {"trust": all_real_t, "realism": all_real_r, "faithfulness": all_real_f}
    real_score_by_cond = group_by_cond(real_mapped_meta, n_real)

    print(f"\n--- Per-condition FPR@95 (threshold from each condition's real) ---")
    for comp in ["trust", "realism", "faithfulness"]:
        gs = gen_scores[comp]
        rs_all = all_real_scores[comp]

        n_better = 0
        n_total = 0
        deltas = []

        for cond in sorted(gen_by_cond.keys()):
            gen_idx_c = gen_by_cond[cond]
            real_score_idx_c = real_score_by_cond.get(cond, [])
            real_kid_idx_c = kid_real_by_cond.get(cond, [])

            if len(real_score_idx_c) < 20 or len(gen_idx_c) < 20 or len(real_kid_idx_c) < 20:
                continue

            rs_c = rs_all[real_score_idx_c]
            valid = np.isfinite(rs_c)
            if np.sum(valid) < 20:
                continue
            threshold = float(np.percentile(rs_c[valid], 95))

            accept_idx = [i for i in gen_idx_c
                          if np.isfinite(gs[i]) and gs[i] <= threshold]
            if len(accept_idx) < 10:
                continue

            eff_k = min(KID_K, len(accept_idx), len(real_kid_idx_c))
            if eff_k < 10:
                continue

            acc_vals = []
            for b in range(N_BOOT):
                rng = np.random.default_rng(SEED + 4000 + hash(cond) % 10000 + b)
                gi = rng.choice(accept_idx, eff_k, replace=False)
                ri = rng.choice(real_kid_idx_c, eff_k, replace=False)
                kv = calculate_kid_same_m(kid_gen[gi], kid_real[ri], use_cosine=True)
                if np.isfinite(kv):
                    acc_vals.append(kv)

            rand_vals = []
            for rep in range(N_RAND):
                rng = np.random.default_rng(SEED + 5000 + hash(cond) % 10000 + rep)
                n_draw = min(len(accept_idx), len(gen_idx_c))
                ridx = rng.choice(gen_idx_c, n_draw, replace=False)
                eff_k_r = min(KID_K, len(ridx), len(real_kid_idx_c))
                if eff_k_r < 10:
                    continue
                gi = rng.choice(ridx, eff_k_r, replace=False)
                ri = rng.choice(real_kid_idx_c, eff_k_r, replace=False)
                kv = calculate_kid_same_m(kid_gen[gi], kid_real[ri], use_cosine=True)
                if np.isfinite(kv):
                    rand_vals.append(kv)

            kid_acc_m = float(np.mean(acc_vals)) if acc_vals else np.nan
            kid_rand_m = float(np.mean(rand_vals)) if rand_vals else np.nan
            d = kid_acc_m - kid_rand_m  # negative = selected is better

            n_total += 1
            if d < 0:
                n_better += 1
            deltas.append(d)

        mean_d = np.mean(deltas) if deltas else np.nan
        print(f"  {comp:15s}: {n_better}/{n_total} conds improved, mean delta={mean_d:+.5f}")

    # --- Ranking correlations ---
    print(f"\n--- Condition-level ranking ---")
    import hashlib
    delta_kids = {}
    dino_gen_by_cond = group_by_cond(kid_gen_meta, len(kid_gen))
    for cond in sorted(gen_by_cond.keys()):
        gi = dino_gen_by_cond.get(cond, [])
        ri = kid_real_by_cond.get(cond, [])
        if len(ri) < 20 or len(gi) < 5:
            delta_kids[cond] = np.nan
            continue
        k = min(len(ri) // 2, len(gi), 500)
        stable_hash = int(hashlib.md5(str(cond).encode()).hexdigest(), 16) % 1000
        rng = np.random.default_rng(SEED + stable_hash)
        deltas_k = []
        for _ in range(20):
            perm = rng.permutation(len(ri))
            ra, rb = kid_real[ri][perm[:k]], kid_real[ri][perm[k:2*k]]
            gs_samp = kid_gen[gi][rng.choice(len(gi), k, replace=False)]
            base = calculate_kid_same_m(ra, rb, use_cosine=True)
            gk = calculate_kid_same_m(ra, gs_samp, use_cosine=True)
            if np.isfinite(base) and np.isfinite(gk):
                deltas_k.append(gk - base)
        delta_kids[cond] = np.mean(deltas_k) if deltas_k else np.nan

    cond_trust = {}
    for cond in sorted(gen_by_cond.keys()):
        idx = gen_by_cond[cond]
        cond_trust[cond] = np.nanmean(gen_t[idx])

    common = sorted([c for c in cond_trust if np.isfinite(delta_kids.get(c, np.nan))])
    rho, _ = spearmanr([cond_trust[c] for c in common],
                       [delta_kids[c] for c in common])
    print(f"  Spearman rho(trust vs deltaKID): {rho:.4f}  (N={len(common)})")
