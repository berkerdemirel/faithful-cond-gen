"""Full decile + FPR95 evaluation: siglip_postgen vs posthoc_mapped,
scored by realism, faithfulness, and trust separately.

Both global and within-condition binning for deciles.
Saves results to outputs/posthoc_alignment/diag/full_component_eval.json

Usage:
    PYTHONPATH=src uv run python scripts/posthoc_alignment/full_component_eval.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from faithful_cond_gen.eval.trust_eval.condition_utils import (
    filter_feats_and_meta_by_seen_combos,
)
from faithful_cond_gen.eval.trust_eval.config import MARGINAL_SEEN_COMBOS
from faithful_cond_gen.eval.trust_eval.metrics_kid import calculate_kid_same_m
from faithful_cond_gen.eval.trust_eval.scoring_core import (
    fit_trust_scoring_components,
    score_trust_from_components,
)
from faithful_cond_gen.model.repa_encoder import REPAEncoder
from faithful_cond_gen.posthoc_alignment.mapper import ResidualAlignmentMapper

CONDITION_KEYS = ["Male", "Smiling", "Blond_Hair", "Eyeglasses"]
BATCH_SIZE = 128
N_BINS = 10
SEED = 42
KID_K = 500
N_RANDOM_REPEATS = 10
PERCENTILES = [50, 75, 90, 95]


def load_mapper(path):
    mapper = ResidualAlignmentMapper(in_dim=768, out_dim=1152)
    state = torch.load(path, map_location="cpu", weights_only=False)
    mapper.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    mapper.eval()
    return mapper


def encode_images(images, encoder_name, device):
    encoder = REPAEncoder(encoder_name=encoder_name, resolution=256,
                          in_channels=3, device=str(device))
    encoder.eval()
    feats = []
    for s in range(0, len(images), BATCH_SIZE):
        e = min(s + BATCH_SIZE, len(images))
        with torch.no_grad():
            feats.append(encoder(images[s:e].to(device)).mean(dim=1).cpu())
    del encoder
    torch.cuda.empty_cache()
    return torch.cat(feats, 0)


def get_cond(meta, idx):
    return tuple(int(meta[k][idx].item()) for k in CONDITION_KEYS)


def kid_with_repeats(gen_feats, real_feats, n_repeats, k, seed_base):
    vals = []
    for rep in range(n_repeats):
        rr = np.random.default_rng(seed_base + rep)
        pg = rr.permutation(len(gen_feats))[:k]
        pr = rr.permutation(len(real_feats))[:k]
        kv = calculate_kid_same_m(gen_feats[pg], real_feats[pr], use_cosine=True)
        if np.isfinite(kv):
            vals.append(kv)
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


# =========================================================================
# Decile
# =========================================================================
def decile_analysis(score_arr, kid_gen, kid_real, gen_meta, real_by_cond, mode="global"):
    valid = np.isfinite(score_arr)
    gen_by_cond = {}
    for i in range(len(score_arr)):
        c = get_cond(gen_meta, i)
        gen_by_cond.setdefault(c, []).append(i)
    unique_conds = sorted(gen_by_cond.keys())

    if mode == "global":
        valid_idx = np.where(valid)[0]
        sorted_idx = valid_idx[np.argsort(score_arr[valid_idx])]
        bin_size = len(sorted_idx) // N_BINS
        kids = []
        for b in range(N_BINS):
            s = b * bin_size
            e = (b + 1) * bin_size if b < N_BINS - 1 else len(sorted_idx)
            bidx = sorted_idx[s:e]
            bch = {}
            for i in bidx:
                c = get_cond(gen_meta, i)
                bch[c] = bch.get(c, 0) + 1
            kvals = []
            rng = np.random.default_rng(SEED + b)
            for cond, ng in bch.items():
                rc = real_by_cond.get(cond, [])
                gc = [i for i in bidx if get_cond(gen_meta, i) == cond]
                k = min(KID_K, len(gc), len(rc))
                if k < 5:
                    continue
                kv = calculate_kid_same_m(kid_gen[gc[:k]],
                                          kid_real[rng.choice(rc, k, replace=False)],
                                          use_cosine=True)
                if np.isfinite(kv):
                    kvals.append(kv)
            kids.append(float(np.mean(kvals)) if kvals else float("nan"))
        return kids
    else:
        kids = []
        for b in range(N_BINS):
            bg, br = [], []
            for cond in unique_conds:
                cg = [i for i in gen_by_cond[cond] if valid[i]]
                if len(cg) < N_BINS:
                    continue
                scores = score_arr[cg]
                order = np.argsort(scores)
                npb = len(order) // N_BINS
                s = b * npb
                e = s + npb if b < N_BINS - 1 else len(order)
                bg.extend([cg[j] for j in order[s:e]])
                rc = real_by_cond.get(cond, [])
                if rc:
                    rng = np.random.default_rng(SEED + b)
                    nt = min(len(rc), e - s)
                    br.extend(rng.choice(rc, nt, replace=len(rc) < nt))
            bg, br = np.array(bg), np.array(br)
            if len(bg) < 10 or len(br) < 10:
                kids.append(float("nan"))
                continue
            k = min(KID_K, len(bg), len(br))
            km, _ = kid_with_repeats(kid_gen[bg], kid_real[br], 5, k, SEED + 400 + b)
            kids.append(km)
        return kids


# =========================================================================
# FPR95
# =========================================================================
def fpr95_analysis(gen_score, real_score, real_meta_calib, gen_meta,
                   kid_gen, kid_real, gen_by_cond, real_by_cond,
                   selected_conditions):
    results = {}
    valid_real = np.isfinite(real_score)

    for pct in PERCENTILES:
        pct_results = {}
        for mode in ["global", "within_cond"]:
            if mode == "global":
                threshold = float(np.percentile(real_score[valid_real], pct))
                valid = np.isfinite(gen_score)
                accept_idx = np.where(valid & (gen_score <= threshold))[0]
            else:
                real_by_cond_scores = {}
                for i in range(len(real_score)):
                    if not valid_real[i]:
                        continue
                    c = get_cond(real_meta_calib, i)
                    real_by_cond_scores.setdefault(c, []).append(real_score[i])
                accept_list = []
                for cond in selected_conditions:
                    cs = real_by_cond_scores.get(cond, [])
                    if len(cs) < 5:
                        continue
                    tc = float(np.percentile(cs, pct))
                    for i in gen_by_cond.get(cond, []):
                        if np.isfinite(gen_score[i]) and gen_score[i] <= tc:
                            accept_list.append(i)
                accept_idx = np.array(accept_list, dtype=int) if accept_list else np.array([], dtype=int)

            n_acc = len(accept_idx)
            n_total = len(gen_score)
            if n_acc < 10:
                pct_results[mode] = {"n_accepted": n_acc, "status": "insufficient"}
                continue

            # Condition histogram of accepted
            cond_hist = {}
            for i in accept_idx:
                c = get_cond(gen_meta, i)
                cond_hist[c] = cond_hist.get(c, 0) + 1

            # Condition-matched random baseline + accepted KID
            rng = np.random.default_rng(SEED + pct)
            random_idx = []
            for cond, n_c in cond_hist.items():
                gc = gen_by_cond.get(cond, [])
                if gc:
                    random_idx.extend(rng.choice(gc, min(n_c, len(gc)), replace=False))
            random_idx = np.array(random_idx)

            k_acc = min(KID_K, n_acc)
            k_rand = min(KID_K, len(random_idx))

            # Per-condition KID
            wc_improvements = []
            for cond, n_c in cond_hist.items():
                rc = real_by_cond.get(cond, [])
                if len(rc) < 10 or n_c < 5:
                    continue
                gc_acc = [i for i in accept_idx if get_cond(gen_meta, i) == cond]
                gc_all = gen_by_cond.get(cond, [])
                k = min(KID_K, len(gc_acc), len(gc_all), len(rc))
                if k < 5:
                    continue
                rng2 = np.random.default_rng(SEED + pct + hash(cond) % 1000)
                rc_sub = rng2.choice(rc, k, replace=False)
                kid_a = calculate_kid_same_m(kid_gen[gc_acc[:k]], kid_real[rc_sub], use_cosine=True)
                kid_r = calculate_kid_same_m(kid_gen[rng2.choice(gc_all, k, replace=False)],
                                             kid_real[rc_sub], use_cosine=True)
                if np.isfinite(kid_a) and np.isfinite(kid_r) and kid_r > 1e-10:
                    wc_improvements.append((kid_r - kid_a) / kid_r * 100)

            # Pooled KID
            kid_acc, _ = kid_with_repeats(kid_gen[accept_idx], kid_real[np.concatenate(list(real_by_cond.values()))],
                                          N_RANDOM_REPEATS, k_acc, SEED + 800 + pct)
            kid_rand, _ = kid_with_repeats(kid_gen[random_idx], kid_real[np.concatenate(list(real_by_cond.values()))],
                                           N_RANDOM_REPEATS, k_rand, SEED + 900 + pct)

            pooled_imp = (kid_rand - kid_acc) / kid_rand * 100 if kid_rand > 1e-10 else 0

            pct_results[mode] = {
                "n_accepted": n_acc,
                "acceptance_rate": n_acc / n_total,
                "pooled_kid_accepted": kid_acc,
                "pooled_kid_random": kid_rand,
                "pooled_improvement_pct": pooled_imp,
                "within_cond_improvement_mean": float(np.mean(wc_improvements)) if wc_improvements else float("nan"),
                "within_cond_n_positive": sum(1 for x in wc_improvements if x > 0),
                "within_cond_n_total": len(wc_improvements),
            }
        results[f"P{pct}"] = pct_results
    return results


def run(model_key, mapper_path, device):
    print(f"\n{'#'*70}")
    print(f"  {model_key}")
    print(f"{'#'*70}")

    # Load gen cache
    gen_cache_dir = Path(f"outputs/posthoc_alignment/diag/{model_key}/gen_cache")
    gen_raw_list, gen_img_list, gen_conds = [], [], {k: [] for k in CONDITION_KEYS}
    for f in sorted(gen_cache_dir.glob("cond_*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        gen_raw_list.append(d["raw_hidden"])
        gen_img_list.append(d["images"])
        cond = d["condition"]
        n = len(d["raw_hidden"])
        for i, k in enumerate(CONDITION_KEYS):
            gen_conds[k].extend([cond[i]] * n)
    gen_raw = torch.cat(gen_raw_list, 0)
    gen_images = torch.cat(gen_img_list, 0)
    gen_meta = {k: torch.tensor(v, dtype=torch.long) for k, v in gen_conds.items()}
    print(f"  Gen: {len(gen_raw)} samples")

    # Gen features
    mapper = load_mapper(mapper_path)
    with torch.no_grad():
        gen_mapped = mapper(gen_raw)
    print("  SigLIP gen...", flush=True)
    gen_siglip = encode_images(gen_images, "siglip", device)
    print("  DINOv3 gen...", flush=True)
    gen_dinov3 = encode_images(gen_images, "dinov3-vit-l", device)
    del gen_images
    torch.cuda.empty_cache()

    # Real features
    real_siglip = torch.load("outputs/real_celeba_siglip_meanpatch/train_features.pt",
                             map_location="cpu", weights_only=False)
    real_dinov3 = torch.load("outputs/real_celeba_dinov3_meanpatch/train_features.pt",
                             map_location="cpu", weights_only=False)
    raw_train = torch.load(f"outputs/posthoc_alignment/raw_hidden/{model_key}/t0.01_hidden.pt",
                           map_location="cpu", weights_only=False)
    with torch.no_grad():
        real_mapped = mapper(raw_train["features"])
    real_mapped_meta = {k: raw_train["metadata"][k] for k in CONDITION_KEYS}

    seen = MARGINAL_SEEN_COMBOS

    # Score
    scoring_configs = {
        "siglip_postgen": (gen_siglip, real_siglip["features"],
                           {k: real_siglip["metadata"][k] for k in CONDITION_KEYS}),
        "posthoc_mapped": (gen_mapped, real_mapped, real_mapped_meta),
    }

    all_gen_scores = {}  # space -> {realism, faithfulness, trust}
    all_real_scores = {}
    all_calib_meta = {}
    for name, (gf, rf, rm) in scoring_configs.items():
        gn = gf / (gf.norm(dim=1, keepdim=True) + 1e-12)
        rn = rf / (rf.norm(dim=1, keepdim=True) + 1e-12)
        cf, cm = filter_feats_and_meta_by_seen_combos(rn, rm, CONDITION_KEYS, seen)
        comp = fit_trust_scoring_components(cf, cm, CONDITION_KEYS)
        r_g, f_g, t_g = score_trust_from_components(gn, gen_meta, comp)
        r_r, f_r, t_r = score_trust_from_components(cf, cm, comp)
        all_gen_scores[name] = {"realism": r_g, "faithfulness": f_g, "trust": t_g}
        all_real_scores[name] = {"realism": r_r, "faithfulness": f_r, "trust": t_r}
        all_calib_meta[name] = cm
        print(f"  {name}: R={np.nanmean(r_g):.3f}±{np.nanstd(r_g):.3f}  "
              f"F={np.nanmean(f_g):.3f}±{np.nanstd(f_g):.3f}  "
              f"T={np.nanmean(t_g):.3f}±{np.nanstd(t_g):.3f}")

    # KID features
    kid_gen = (gen_dinov3 / (gen_dinov3.norm(dim=1, keepdim=True) + 1e-12)).numpy()
    kid_real = (real_dinov3["features"] / (real_dinov3["features"].norm(dim=1, keepdim=True) + 1e-12)).numpy()

    real_dinov3_meta = {k: real_dinov3["metadata"][k] for k in CONDITION_KEYS}
    real_by_cond = {}
    for i in range(len(real_dinov3_meta[CONDITION_KEYS[0]])):
        c = get_cond(real_dinov3_meta, i)
        real_by_cond.setdefault(c, []).append(i)
    gen_by_cond = {}
    for i in range(len(gen_meta[CONDITION_KEYS[0]])):
        c = get_cond(gen_meta, i)
        gen_by_cond.setdefault(c, []).append(i)
    selected_conditions = sorted(gen_by_cond.keys())

    results = {"model": model_key, "decile": {}, "fpr95": {}}

    # =====================================================================
    # DECILE: for each (space, component, mode)
    # =====================================================================
    components = ["realism", "faithfulness", "trust"]
    modes = ["global", "within_cond"]

    for space in ["siglip_postgen", "posthoc_mapped"]:
        for comp_name in components:
            score_arr = all_gen_scores[space][comp_name]
            for mode in modes:
                key = f"{space}__{comp_name}__{mode}"
                kids = decile_analysis(score_arr, kid_gen, kid_real, gen_meta, real_by_cond, mode)
                vk = [k for k in kids if np.isfinite(k)]
                inc = sum(1 for i in range(len(vk) - 1) if vk[i + 1] > vk[i])
                direction = "CORRECT" if inc > len(vk) // 2 else "REVERSE"
                results["decile"][key] = {"kids": kids, "direction": direction, "increasing": inc}
                kid_str = " ".join(f"{k:.4f}" for k in kids)
                print(f"  DECILE {space:20s} {comp_name:15s} {mode:12s}: [{kid_str}] {direction} ({inc}/{len(vk)-1})")

    # =====================================================================
    # FPR95: for each (space, component)
    # =====================================================================
    print()
    for space in ["siglip_postgen", "posthoc_mapped"]:
        for comp_name in components:
            gen_score = all_gen_scores[space][comp_name]
            real_score = all_real_scores[space][comp_name]
            real_mc = all_calib_meta[space]

            fpr = fpr95_analysis(gen_score, real_score, real_mc, gen_meta,
                                 kid_gen, kid_real, gen_by_cond, real_by_cond,
                                 selected_conditions)
            key = f"{space}__{comp_name}"
            results["fpr95"][key] = fpr

            # Print P95 summary
            for mode in ["global", "within_cond"]:
                r = fpr.get("P95", {}).get(mode, {})
                if "status" in r:
                    print(f"  FPR95 {space:20s} {comp_name:15s} P95 {mode:12s}: insufficient ({r.get('n_accepted',0)})")
                else:
                    print(f"  FPR95 {space:20s} {comp_name:15s} P95 {mode:12s}: "
                          f"accept={r['n_accepted']} ({r['acceptance_rate']:.1%}) "
                          f"KID_acc={r['pooled_kid_accepted']:.4f} "
                          f"KID_rand={r['pooled_kid_random']:.4f} "
                          f"Δ={r['pooled_improvement_pct']:+.1f}% "
                          f"wc_mean={r['within_cond_improvement_mean']:+.1f}% "
                          f"pos={r['within_cond_n_positive']}/{r['within_cond_n_total']}")

    return results


def main():
    device = torch.device("cuda:0")

    all_results = {}
    for model_key, mapper_path in [
        ("celeba_vanilla_marginal_v1",
         "outputs/posthoc_alignment/mappers/celeba_vanilla_marginal_v1/best_mapper.pt"),
        ("celeba_repa_siglip_marginal_v1",
         "outputs/posthoc_alignment/mappers/celeba_repa_siglip_marginal_v1/best_mapper.pt"),
    ]:
        r = run(model_key, mapper_path, device)
        all_results[model_key] = r

    save_path = Path("outputs/posthoc_alignment/diag/full_component_eval.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {save_path}")

    # Copy to debug dir
    import shutil
    shutil.copy(save_path, "/mnt/pvc/posthoc_debug/full_component_eval.json")
    print("  Copied to /mnt/pvc/posthoc_debug/")


if __name__ == "__main__":
    main()
