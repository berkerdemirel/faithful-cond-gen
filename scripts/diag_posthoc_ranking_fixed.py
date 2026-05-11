"""
Standalone diagnostic: posthoc_mapped condition-level ranking vs DINO ΔKID.
Uses FIXED gen_cache metadata (alphabetical condition ordering).
"""
import glob
import hashlib
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
N_BOOT = 20


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


def delta_kid_per_cond(gen_feats, gen_meta, real_feats, real_meta, gen_by_cond=None):
    gen_np = gen_feats if isinstance(gen_feats, np.ndarray) else gen_feats.numpy()
    real_np = real_feats if isinstance(real_feats, np.ndarray) else real_feats.numpy()
    if gen_by_cond is None:
        gen_by_cond = group_by_cond(gen_meta, len(gen_np))
    real_by_cond = group_by_cond(real_meta, len(real_np))

    delta_kids = {}
    for cond in sorted(gen_by_cond.keys()):
        gi = gen_by_cond.get(cond, [])
        ri = real_by_cond.get(cond, [])
        if len(ri) < 20 or len(gi) < 5:
            delta_kids[cond] = np.nan
            continue
        k = min(len(ri) // 2, len(gi), 500)
        stable_hash = int(hashlib.md5(str(cond).encode()).hexdigest(), 16) % 1000
        rng = np.random.default_rng(SEED + stable_hash)
        deltas = []
        for _ in range(N_BOOT):
            perm = rng.permutation(len(ri))
            ra, rb = real_np[ri][perm[:k]], real_np[ri][perm[k:2*k]]
            gs = gen_np[gi][rng.choice(len(gi), k, replace=False)]
            base = calculate_kid_same_m(ra, rb, use_cosine=True)
            gk = calculate_kid_same_m(ra, gs, use_cosine=True)
            if np.isfinite(base) and np.isfinite(gk):
                deltas.append(gk - base)
        delta_kids[cond] = np.mean(deltas) if deltas else np.nan
    return delta_kids


for model in ["celeba_vanilla_marginal_v1", "celeba_repa_siglip_marginal_v1"]:
    short = model.replace("celeba_", "").replace("_v1", "")
    print(f"\n{'='*70}")
    print(f"MODEL: {model} ({short})")
    print(f"{'='*70}")

    # --- Load posthoc_mapped features (fixed gen_cache) ---
    mapper_path = f"outputs/posthoc_alignment/mappers/{model}/best_mapper.pt"
    mapper = ResidualAlignmentMapper(768, 1152)
    mapper.load_state_dict(torch.load(mapper_path, map_location="cpu", weights_only=True))
    mapper.eval()

    # Real: raw hidden -> mapper
    raw = torch.load(f"outputs/posthoc_alignment/raw_hidden/{model}/t0.01_hidden.pt",
                     map_location="cpu", weights_only=False)
    with torch.no_grad():
        real_mapped = l2(mapper(raw["features"]))
    real_mapped_meta = {k: raw["metadata"][k] for k in CK}
    print(f"Real posthoc_mapped: {real_mapped.shape}")

    # Gen: fixed gen_cache -> mapper
    cache_dir = f"outputs/posthoc_alignment/diag/{model}/gen_cache"
    pts = sorted(glob.glob(f"{cache_dir}/cond_*.pt"))
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
    print(f"Gen posthoc_mapped: {gen_mapped.shape}")
    del mapper

    # --- Load DINO features (fixed metadata) ---
    cache_path = f"/mnt/pvc/posthoc_debug/feature_cache/{model}_encoded.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    gen_dino = l2(cache["gen_dinov3"])
    gen_dino_meta = cache["gen_meta"]
    print(f"Gen DINO: {gen_dino.shape}")

    real_dino_data = torch.load("outputs/real_celeba_dinov3_meanpatch/train_features.pt",
                                map_location="cpu", weights_only=False)
    real_dino = l2(real_dino_data["features"])
    real_dino_meta = real_dino_data.get("metadata", {})
    print(f"Real DINO: {real_dino.shape}")

    # Verify gen ordering matches
    for k in CK:
        assert (gen_meta[k] == gen_dino_meta[k]).all(), f"Gen meta mismatch on {k}"
    print("Gen meta match between posthoc_mapped and DINO: OK")

    # --- Fit scoring on seen combos, score gen ---
    calib_feats, calib_meta = filter_feats_and_meta_by_seen_combos(
        real_mapped, real_mapped_meta, CK, MARGINAL_SEEN_COMBOS
    )
    print(f"Calibration set (seen combos): {calib_feats.shape}")

    components = fit_trust_scoring_components(calib_feats, calib_meta, CK)
    gen_realism, gen_faith, gen_trust = score_trust_from_components(
        gen_mapped, gen_meta, components
    )

    # --- Per-condition mean scores ---
    gen_by_cond = group_by_cond(gen_meta, len(gen_trust))
    cond_trust, cond_realism, cond_faith = {}, {}, {}
    for cond in sorted(gen_by_cond.keys()):
        idx = gen_by_cond[cond]
        cond_trust[cond] = np.nanmean(gen_trust[idx])
        cond_realism[cond] = np.nanmean(gen_realism[idx])
        cond_faith[cond] = np.nanmean(gen_faith[idx])

    # --- Per-condition ΔKID in DINO space ---
    delta_kids = delta_kid_per_cond(gen_dino, gen_dino_meta, real_dino, real_dino_meta)

    # --- Spearman correlations ---
    common = sorted([c for c in cond_trust if np.isfinite(delta_kids.get(c, np.nan))])
    trust_arr = np.array([cond_trust[c] for c in common])
    realism_arr = np.array([cond_realism[c] for c in common])
    faith_arr = np.array([cond_faith[c] for c in common])
    kid_arr = np.array([delta_kids[c] for c in common])

    rho_t, _ = spearmanr(trust_arr, kid_arr)
    rho_r, _ = spearmanr(realism_arr, kid_arr)
    rho_f, _ = spearmanr(faith_arr, kid_arr)

    print(f"\n--- POSTHOC_MAPPED RANKING ({short}) ---")
    print(f"  N conditions: {len(common)}")
    print(f"  Spearman rho (trust vs deltaKID):       {rho_t:.4f}")
    print(f"  Spearman rho (realism vs deltaKID):     {rho_r:.4f}")
    print(f"  Spearman rho (faithfulness vs deltaKID): {rho_f:.4f}")

    # --- DINO trust comparison ---
    dino_calib, dino_calib_meta = filter_feats_and_meta_by_seen_combos(
        real_dino, real_dino_meta, CK, MARGINAL_SEEN_COMBOS
    )
    dino_components = fit_trust_scoring_components(dino_calib, dino_calib_meta, CK)
    dino_r, dino_f, dino_t = score_trust_from_components(gen_dino, gen_dino_meta, dino_components)

    dino_by_cond = group_by_cond(gen_dino_meta, len(dino_t))
    dino_cond_trust = {}
    for cond in sorted(dino_by_cond.keys()):
        idx = dino_by_cond[cond]
        dino_cond_trust[cond] = np.nanmean(dino_t[idx])

    dino_trust_arr = np.array([dino_cond_trust[c] for c in common])
    rho_dino, _ = spearmanr(dino_trust_arr, kid_arr)

    print(f"\n--- DINO TRUST COMPARISON (same gen_cache samples) ---")
    print(f"  Spearman rho (DINO trust vs deltaKID):  {rho_dino:.4f}")

    # --- Side by side ---
    print(f"\n{'Cond':<20s} {'Trust(PM)':>10s} {'Trust(DINO)':>12s} {'deltaKID':>10s}")
    for c in common:
        print(f"  {str(c):<18s} {cond_trust[c]:>10.4f} {dino_cond_trust[c]:>12.4f} {delta_kids[c]:>10.6f}")
