"""
Comprehensive decile binning + FPR95 diagnostic for posthoc alignment (RxRx1).

Produces:
- Decile plots with bootstrap error bars (trust/realism/faithfulness)
  for global and within-condition modes
- FPR95: accepted vs condition-matched random at P50/P75/P90/P95
  for trust, realism-only, faithfulness-only
- Per-condition FPR95 at P95

All KID computed in DINOv3 space (L2-normed, cosine kernel).
All real references are condition-matched.

Usage:
    PYTHONPATH=src uv run python scripts/posthoc_alignment/eval_posthoc_alignment_rxrx1.py
"""

import glob
import json
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from faithful_cond_gen.data.rxrx1 import to_rgb
from faithful_cond_gen.eval.trust_eval.config import RXRX1_HELDOUT_PAIRS
from faithful_cond_gen.eval.trust_eval.metrics_kid import calculate_kid_same_m
from faithful_cond_gen.eval.trust_eval.scoring_core import (
    fit_trust_scoring_components,
    score_trust_from_components,
)
from faithful_cond_gen.model.repa_encoder import REPAEncoder
from faithful_cond_gen.posthoc_alignment.mapper import ResidualAlignmentMapper

# ---- Config ----
CK = ["cell_type_id", "sirna_id"]
N_BINS = 10
KID_K = 500
N_BOOT = 10
N_RAND = 10
SEED = 42
PCTS = [50, 75, 90, 95]
FIG_DIR = "outputs/posthoc_alignment/figures_rxrx1"
CACHE_DIR = "outputs/posthoc_alignment/feature_cache_rxrx1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPONENTS = ["trust", "realism", "faithfulness"]
MODELS = ["rxrx1_vanilla_marginal_v1", "rxrx1_repa_siglip_marginal_v1"]

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ---- Helpers ----

def get_cond(meta, i):
    return tuple(int(meta[k][i]) for k in CK)


def group_by_cond(meta, n):
    out = {}
    for i in range(n):
        c = get_cond(meta, i)
        out.setdefault(c, []).append(i)
    return out


def encode_batch(images, encoder_name):
    """Encode images with REPAEncoder. Handles 6ch -> RGB conversion."""
    enc = REPAEncoder(encoder_name=encoder_name, resolution=256,
                      in_channels=3, device=str(DEVICE))
    enc.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(images), 64):
            e = min(s + 64, len(images))
            batch = images[s:e].to(DEVICE)
            # Convert 6ch -> 3ch RGB for encoder
            if batch.shape[1] == 6:
                batch = to_rgb(batch)
            out.append(enc(batch).mean(dim=1).cpu())
    del enc
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(out)


def l2(x):
    if isinstance(x, torch.Tensor):
        return x / (x.norm(dim=1, keepdim=True) + 1e-12)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def filter_real_by_seen(features, metadata):
    """Filter real features to only include non-heldout (seen) pairs for calibration."""
    ct = metadata["cell_type_id"]
    sirna = metadata["sirna_id"]
    mask = torch.zeros(len(ct), dtype=torch.bool)
    for i in range(len(ct)):
        pair = (int(ct[i]), int(sirna[i]))
        if pair not in RXRX1_HELDOUT_PAIRS:
            mask[i] = True
    filtered_feats = features[mask]
    filtered_meta = {k: v[mask] for k, v in metadata.items() if isinstance(v, torch.Tensor)}
    return filtered_feats, filtered_meta


# ---- Data loading ----

def load_model_data(model_key):
    cache_path = os.path.join(CACHE_DIR, f"{model_key}_encoded.pt")

    if os.path.exists(cache_path):
        print(f"  Loading cached features...")
        c = torch.load(cache_path, map_location="cpu", weights_only=False)
        gen_siglip, gen_dinov3, gen_hidden, gen_meta = (
            c["gen_siglip"], c["gen_dinov3"], c["gen_hidden"], c["gen_meta"])
    else:
        cache_dir = f"outputs/posthoc_alignment/diag/{model_key}/gen_cache"
        pts = sorted(glob.glob(f"{cache_dir}/cond_*.pt"))
        if not pts:
            raise FileNotFoundError(f"No gen_cache files in {cache_dir}")

        imgs, hids, metas = [], [], {k: [] for k in CK}
        for p in pts:
            d = torch.load(p, map_location="cpu", weights_only=False)
            imgs.append(d["images"])
            hids.append(d["raw_hidden"])
            cond = d["condition"]
            n = d["images"].shape[0]
            # cond is (cell_type_id, sirna_id)
            metas["cell_type_id"].append(torch.full((n,), cond[0], dtype=torch.long))
            metas["sirna_id"].append(torch.full((n,), cond[1], dtype=torch.long))

        gen_images = torch.cat(imgs)
        gen_hidden = torch.cat(hids)
        gen_meta = {k: torch.cat(metas[k]) for k in CK}
        print(f"  {len(gen_images)} gen samples from {len(pts)} conditions")
        print(f"  Encoding SigLIP...")
        gen_siglip = encode_batch(gen_images, "siglip")
        print(f"  Encoding DINOv3...")
        gen_dinov3 = encode_batch(gen_images, "dinov3-vit-l")
        del gen_images
        torch.save({"gen_siglip": gen_siglip, "gen_dinov3": gen_dinov3,
                     "gen_hidden": gen_hidden, "gen_meta": gen_meta}, cache_path)
        print(f"  Cached -> {cache_path}")

    # Mapper
    mapper_path = f"outputs/posthoc_alignment/mappers/{model_key}/best_mapper.pt"
    mapper = ResidualAlignmentMapper(768, 1152)
    mapper.load_state_dict(torch.load(mapper_path, map_location="cpu", weights_only=True))
    mapper.eval()
    with torch.no_grad():
        gen_mapped = mapper(gen_hidden)

    # Real features
    real_siglip = torch.load(
        "outputs/real_rxrx1_siglip_meanpatch/train_features.pt",
        map_location="cpu", weights_only=False)
    real_dinov3 = torch.load(
        "outputs/real_rxrx1_dinov3_meanpatch/train_features.pt",
        map_location="cpu", weights_only=False)
    raw_train = torch.load(
        f"outputs/posthoc_alignment/raw_hidden/{model_key}/t0.01_hidden.pt",
        map_location="cpu", weights_only=False)
    with torch.no_grad():
        real_mapped = mapper(raw_train["features"])
    real_mapped_meta = {k: raw_train["metadata"][k] for k in CK}
    del mapper, raw_train

    # Score in each space - filter real to seen pairs for calibration
    spaces_cfg = {
        "siglip_postgen": (gen_siglip, real_siglip["features"],
                           {k: real_siglip["metadata"][k] for k in CK}),
        "posthoc_mapped": (gen_mapped, real_mapped, real_mapped_meta),
    }
    gen_scores, real_scores = {}, {}
    for name, (gf, rf, rm) in spaces_cfg.items():
        gn, rn = l2(gf), l2(rf)
        # Filter real to seen (non-heldout) pairs for calibration
        cf, cm = filter_real_by_seen(rn, rm)
        comp = fit_trust_scoring_components(cf, cm, CK)
        rg, fg, tg = score_trust_from_components(gn, gen_meta, comp)
        rr, fr, tr = score_trust_from_components(rn, rm, comp)
        gen_scores[name] = {"realism": rg, "faithfulness": fg, "trust": tg}
        real_scores[name] = {"realism": rr, "faithfulness": fr, "trust": tr, "meta": rm}
        print(f"    {name}: gen R={np.nanmean(rg):.3f}+/-{np.nanstd(rg):.3f}  "
              f"F={np.nanmean(fg):.3f}+/-{np.nanstd(fg):.3f}  "
              f"T={np.nanmean(tg):.3f}+/-{np.nanstd(tg):.3f}")

    # KID features (L2-normed DINOv3 -> numpy)
    kid_gen = l2(gen_dinov3).numpy()
    kid_real = l2(real_dinov3["features"]).numpy()
    kid_real_meta = {k: real_dinov3["metadata"][k] for k in CK}

    gen_by_cond = group_by_cond(gen_meta, len(gen_meta[CK[0]]))
    real_by_cond = group_by_cond(kid_real_meta, len(kid_real_meta[CK[0]]))

    return dict(gen_scores=gen_scores, real_scores=real_scores,
                kid_gen=kid_gen, kid_real=kid_real,
                gen_meta=gen_meta, gen_by_cond=gen_by_cond,
                real_by_cond=real_by_cond)


# ---- KID helpers ----

def build_condmatched_real(gen_indices, gen_meta, real_by_cond, seed):
    cond_hist = {}
    for i in gen_indices:
        c = get_cond(gen_meta, i)
        cond_hist[c] = cond_hist.get(c, 0) + 1

    rng = np.random.default_rng(seed)
    pool = []
    for c in sorted(cond_hist.keys()):
        rc = real_by_cond.get(c, [])
        if not rc:
            continue
        n = cond_hist[c]
        pool.extend(rng.choice(rc, n, replace=(len(rc) < n)))
    return np.array(pool, dtype=int), cond_hist


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


def kid_condmatched(gen_indices, kid_gen, kid_real, gen_meta, real_by_cond,
                    seed, k=KID_K, n_boot=N_BOOT):
    gen_indices = np.asarray(gen_indices)
    if len(gen_indices) < 10:
        return np.nan, np.nan
    rp, _ = build_condmatched_real(gen_indices, gen_meta, real_by_cond, seed)
    if len(rp) < 10:
        return np.nan, np.nan
    return bootstrap_kid(kid_gen[gen_indices], kid_real[rp], k, n_boot, seed + 50)


# ---- Decile binning ----

def decile_binning(score_arr, kid_gen, kid_real, gen_meta, real_by_cond,
                   gen_by_cond, mode="global"):
    valid = np.isfinite(score_arr)
    results = []

    if mode == "global":
        vi = np.where(valid)[0]
        si = vi[np.argsort(score_arr[vi])]
        bsz = len(si) // N_BINS
        for b in range(N_BINS):
            start = b * bsz
            end = (b + 1) * bsz if b < N_BINS - 1 else len(si)
            bidx = si[start:end]
            km, ks = kid_condmatched(
                bidx, kid_gen, kid_real, gen_meta, real_by_cond,
                seed=SEED + b * 100)
            results.append((km, ks))
    else:
        bins = {b: [] for b in range(N_BINS)}
        for cond in sorted(gen_by_cond.keys()):
            cg = np.array([i for i in gen_by_cond[cond] if valid[i]])
            if len(cg) < N_BINS:
                continue
            order = np.argsort(score_arr[cg])
            npb = len(order) // N_BINS
            for b in range(N_BINS):
                start = b * npb
                end = start + npb if b < N_BINS - 1 else len(order)
                bins[b].extend(cg[order[start:end]].tolist())

        for b in range(N_BINS):
            bidx = np.array(bins[b])
            if len(bidx) < 20:
                results.append((np.nan, np.nan))
                continue
            km, ks = kid_condmatched(
                bidx, kid_gen, kid_real, gen_meta, real_by_cond,
                seed=SEED + 5000 + b * 100)
            results.append((km, ks))

    return results


# ---- FPR95 global ----

def fpr95_global(gen_score, real_score, kid_gen, kid_real,
                 gen_meta, gen_by_cond, real_by_cond):
    valid_real = np.isfinite(real_score)
    out = {}

    for pct in PCTS:
        threshold = float(np.percentile(real_score[valid_real], pct))
        accept_idx = np.where(np.isfinite(gen_score) & (gen_score <= threshold))[0]
        if len(accept_idx) < 20:
            out[pct] = None
            continue

        rp_acc, cond_hist = build_condmatched_real(
            accept_idx, gen_meta, real_by_cond, seed=SEED + 1000 + pct)
        kid_acc_m, kid_acc_s = bootstrap_kid(
            kid_gen[accept_idx], kid_real[rp_acc],
            k=KID_K, n_boot=N_BOOT, seed=SEED + 1100 + pct)

        rand_vals = []
        for rep in range(N_RAND):
            rng = np.random.default_rng(SEED + 2000 + pct * 100 + rep)
            ridx = []
            for c, n_need in sorted(cond_hist.items()):
                gc = gen_by_cond.get(c, [])
                if gc:
                    ridx.extend(rng.choice(gc, min(n_need, len(gc)), replace=False))
            ridx = np.array(ridx, dtype=int)
            if len(ridx) < 20:
                continue
            km, _ = bootstrap_kid(
                kid_gen[ridx], kid_real[rp_acc],
                k=KID_K, n_boot=1, seed=SEED + 3000 + rep)
            if np.isfinite(km):
                rand_vals.append(km)

        kid_rand_m = float(np.mean(rand_vals)) if rand_vals else np.nan
        kid_rand_s = float(np.std(rand_vals)) if rand_vals else np.nan

        out[pct] = dict(
            n_accepted=int(len(accept_idx)),
            n_total=int(np.sum(np.isfinite(gen_score))),
            threshold=float(threshold),
            kid_acc=(kid_acc_m, kid_acc_s),
            kid_rand=(kid_rand_m, kid_rand_s),
            delta=kid_rand_m - kid_acc_m,
        )
    return out


# ---- FPR95 per-condition ----

def fpr95_per_cond(gen_score, real_score, real_meta,
                   kid_gen, kid_real, gen_by_cond, real_by_cond, pct=95):
    real_score_by_cond = {}
    n_real = len(real_score)
    for i in range(n_real):
        c = get_cond(real_meta, i)
        real_score_by_cond.setdefault(c, []).append(i)

    out = {}
    for cond in sorted(gen_by_cond.keys()):
        gen_idx_c = gen_by_cond[cond]
        real_score_idx_c = real_score_by_cond.get(cond, [])
        real_kid_idx_c = real_by_cond.get(cond, [])

        if len(real_score_idx_c) < 20 or len(gen_idx_c) < 20 or len(real_kid_idx_c) < 20:
            continue

        rs_c = real_score[real_score_idx_c]
        valid_r = np.isfinite(rs_c)
        if np.sum(valid_r) < 20:
            continue
        threshold = float(np.percentile(rs_c[valid_r], pct))

        accept_idx = [i for i in gen_idx_c
                      if np.isfinite(gen_score[i]) and gen_score[i] <= threshold]
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
        kid_acc_s = float(np.std(acc_vals)) if acc_vals else np.nan
        kid_rand_m = float(np.mean(rand_vals)) if rand_vals else np.nan
        kid_rand_s = float(np.std(rand_vals)) if rand_vals else np.nan

        out[str(cond)] = dict(
            n_accepted=len(accept_idx),
            n_total=len(gen_idx_c),
            threshold=float(threshold),
            kid_acc=(kid_acc_m, kid_acc_s),
            kid_rand=(kid_rand_m, kid_rand_s),
            delta=kid_rand_m - kid_acc_m,
        )
    return out


# ---- Plotting ----

COLORS = {"trust": "#1f77b4", "realism": "#ff7f0e", "faithfulness": "#2ca02c"}


def plot_decile(decile_by_comp, model_key, space, mode, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(N_BINS)
    for comp in COMPONENTS:
        means = [b[0] for b in decile_by_comp[comp]]
        stds = [b[1] for b in decile_by_comp[comp]]
        ax.errorbar(x, means, yerr=stds, marker='o', capsize=3,
                    label=comp.capitalize(), color=COLORS[comp], linewidth=1.5)
    ax.set_xlabel("Bin Index (0=best, 9=worst)")
    ax.set_ylabel("KID (lower = better)")
    short = model_key.replace("rxrx1_", "").replace("_marginal_v1", "")
    ax.set_title(f"Decile Binning: KID vs Ranking -- {short}/{space} [{mode}]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fpr95(fpr95_by_comp, model_key, space, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ci, comp in enumerate(COMPONENTS):
        ax = axes[ci]
        data = fpr95_by_comp[comp]
        pcts_valid, acc_m, acc_s, rand_m, rand_s = [], [], [], [], []
        for pct in PCTS:
            r = data.get(pct)
            if r is None:
                continue
            pcts_valid.append(pct)
            acc_m.append(r["kid_acc"][0]); acc_s.append(r["kid_acc"][1])
            rand_m.append(r["kid_rand"][0]); rand_s.append(r["kid_rand"][1])
        if not pcts_valid:
            ax.set_title(f"{comp.capitalize()} (no data)")
            continue
        x = np.arange(len(pcts_valid))
        w = 0.35
        ax.bar(x - w/2, acc_m, w, yerr=acc_s, label="Accepted",
               color=COLORS[comp], alpha=0.8, capsize=3)
        ax.bar(x + w/2, rand_m, w, yerr=rand_s, label="Random",
               color="gray", alpha=0.6, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"P{p}" for p in pcts_valid])
        ax.set_title(comp.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    axes[0].set_ylabel("KID (lower = better)")
    short = model_key.replace("rxrx1_", "").replace("_marginal_v1", "")
    fig.suptitle(f"FPR Selection: Accepted vs Random -- {short}/{space}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_percond_fpr95(percond_by_comp, model_key, space, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ci, comp in enumerate(COMPONENTS):
        ax = axes[ci]
        data = percond_by_comp[comp]
        acc, rand, labels = [], [], []
        for cond_str in sorted(data.keys()):
            r = data[cond_str]
            if np.isfinite(r["kid_acc"][0]) and np.isfinite(r["kid_rand"][0]):
                acc.append(r["kid_acc"][0])
                rand.append(r["kid_rand"][0])
                labels.append(cond_str)
        if not acc:
            ax.set_title(f"{comp.capitalize()} (no data)")
            continue
        ax.scatter(rand, acc, c=COLORS[comp], alpha=0.7, s=50, zorder=3)
        mn = min(min(acc), min(rand))
        mx = max(max(acc), max(rand))
        pad = (mx - mn) * 0.1
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad],
                'k--', alpha=0.3, linewidth=1)
        n_better = sum(1 for a, r in zip(acc, rand) if a < r)
        ax.set_xlabel("KID (random)")
        ax.set_ylabel("KID (accepted)")
        ax.set_title(f"{comp.capitalize()} ({n_better}/{len(acc)} conds improved)")
        ax.grid(True, alpha=0.3)
    short = model_key.replace("rxrx1_", "").replace("_marginal_v1", "")
    fig.suptitle(f"Per-Condition FPR@95: Accepted vs Random -- {short}/{space}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---- Main ----

def run_model(model_key):
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"MODEL: {model_key}")
    print(f"{'='*70}")

    data = load_model_data(model_key)
    kid_gen = data["kid_gen"]
    kid_real = data["kid_real"]
    gen_meta = data["gen_meta"]
    gen_by_cond = data["gen_by_cond"]
    real_by_cond = data["real_by_cond"]
    short = model_key.replace("rxrx1_", "").replace("_marginal_v1", "")

    all_results = {"model": model_key}

    for space in ["siglip_postgen", "posthoc_mapped"]:
        gen_sc = data["gen_scores"][space]
        real_sc = data["real_scores"][space]
        real_meta_sc = real_sc["meta"]

        print(f"\n  -- {space} {'--'*25}")

        # Decile
        for mode in ["global", "within_cond"]:
            print(f"\n  Decile [{mode}]:")
            dec = {}
            for comp in COMPONENTS:
                bins = decile_binning(gen_sc[comp], kid_gen, kid_real,
                                     gen_meta, real_by_cond, gen_by_cond, mode)
                dec[comp] = bins
                means = [b[0] for b in bins]
                valid = [m for m in means if np.isfinite(m)]
                inc = sum(1 for i in range(len(valid)-1) if valid[i+1] > valid[i])
                direction = "CORRECT" if inc > len(valid)//2 else "REVERSE"
                kid_str = " ".join(f"{m:.4f}" for m in means)
                print(f"    {comp:15s}: [{kid_str}]  {direction} ({inc}/{len(valid)-1})")

            fig_path = os.path.join(FIG_DIR, f"decile_{short}_{space}_{mode}.png")
            plot_decile(dec, model_key, space, mode, fig_path)
            print(f"    -> {fig_path}")
            all_results[f"decile_{space}_{mode}"] = {
                c: [(float(m), float(s)) for m, s in v] for c, v in dec.items()
            }

        # FPR95 global
        print(f"\n  FPR Selection (global):")
        fpr = {}
        for comp in COMPONENTS:
            fpr[comp] = fpr95_global(gen_sc[comp], real_sc[comp],
                                     kid_gen, kid_real, gen_meta,
                                     gen_by_cond, real_by_cond)
            print(f"    {comp:15s}:")
            for pct in PCTS:
                r = fpr[comp].get(pct)
                if r is None:
                    print(f"      P{pct:2d}: insufficient data")
                    continue
                marker = "+" if r['delta'] > 0 else "-"
                print(f"      P{pct:2d}: n_acc={r['n_accepted']:5d}/{r['n_total']:5d}  "
                      f"KID_acc={r['kid_acc'][0]:.5f}+/-{r['kid_acc'][1]:.5f}  "
                      f"KID_rand={r['kid_rand'][0]:.5f}+/-{r['kid_rand'][1]:.5f}  "
                      f"D={r['delta']:+.5f}  {marker}")

        fig_path = os.path.join(FIG_DIR, f"fpr95_{short}_{space}.png")
        plot_fpr95(fpr, model_key, space, fig_path)
        print(f"    -> {fig_path}")

        # FPR95 per-condition
        print(f"\n  Per-condition FPR@95:")
        pc = {}
        for comp in COMPONENTS:
            pc[comp] = fpr95_per_cond(gen_sc[comp], real_sc[comp], real_meta_sc,
                                      kid_gen, kid_real, gen_by_cond, real_by_cond)
            vals = [r["delta"] for r in pc[comp].values() if np.isfinite(r["delta"])]
            n_better = sum(1 for v in vals if v > 0)
            mean_d = np.mean(vals) if vals else np.nan
            print(f"    {comp:15s}: {n_better}/{len(vals)} conds improved, "
                  f"mean D={mean_d:+.5f}")
            for cond_str in sorted(pc[comp].keys()):
                r = pc[comp][cond_str]
                marker = "+" if r['delta'] > 0 else "-"
                print(f"      {cond_str:>20s}: n={r['n_accepted']:4d}/{r['n_total']:4d}  "
                      f"acc={r['kid_acc'][0]:.5f}  rand={r['kid_rand'][0]:.5f}  "
                      f"D={r['delta']:+.5f}  {marker}")

        fig_path = os.path.join(FIG_DIR, f"fpr95_percond_{short}_{space}.png")
        plot_percond_fpr95(pc, model_key, space, fig_path)
        print(f"    -> {fig_path}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s")
    return all_results


if __name__ == "__main__":
    all_results = {}
    for mk in MODELS:
        all_results[mk] = run_model(mk)

    def prep(obj):
        if isinstance(obj, dict):
            return {k: prep(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [prep(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    out_path = os.path.join(FIG_DIR, "comprehensive_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(prep(all_results), f, indent=2)
    print(f"\nResults saved -> {out_path}")
    print(f"Figures saved -> {FIG_DIR}/")
