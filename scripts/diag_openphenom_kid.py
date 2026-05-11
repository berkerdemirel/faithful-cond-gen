"""
Diagnostic: Per-condition KID comparison between openphenom and dinov2 aligned features.

Focus: Pick conditions with good sample sizes and compare:
1. KID values (are they meaningful or just noise?)
2. Within-condition variance (is there enough signal after L2 norm?)
3. Compare with dinov3 as ground truth

Key question: After L2 normalization, do openphenom features discriminate between conditions?
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

base = Path("/mnt/pvc/faithful-cond-gen/outputs")


def load_and_normalize(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    feats = data["features"]
    # L2 normalize
    feats = F.normalize(feats.float(), dim=-1)
    return feats, data


def compute_kid_cosine(gen_feats, real_feats):
    """Simple KID with cosine kernel k(x,y) = (x·y + 1)^3."""
    gen = gen_feats.numpy().astype(np.float64)
    real = real_feats.numpy().astype(np.float64)

    n, m = gen.shape[0], real.shape[0]
    if n < 2 or m < 2:
        return float('nan')

    # K(gen, gen)
    gg = (gen @ gen.T + 1) ** 3
    np.fill_diagonal(gg, 0)
    kgg = gg.sum() / (n * (n - 1))

    # K(real, real)
    rr = (real @ real.T + 1) ** 3
    np.fill_diagonal(rr, 0)
    krr = rr.sum() / (m * (m - 1))

    # K(gen, real)
    gr = (gen @ real.T + 1) ** 3
    kgr = gr.mean()

    return kgg + krr - 2 * kgr


def build_condition_index(data, feats):
    """Build condition -> indices mapping from filenames."""
    fnames = data.get("filenames", [])
    if not fnames:
        return {}

    cond_idx = defaultdict(list)
    for i, fn in enumerate(fnames):
        # Parse: cell{ct}_sirna{si}_{idx}.pt
        parts = fn.replace(".pt", "").replace(".png", "")
        # Extract condition from filename
        tokens = parts.split("_")
        ct = int(tokens[0].replace("cell", ""))
        si = int(tokens[1].replace("sirna", ""))
        cond_idx[(ct, si)].append(i)
    return cond_idx


def main():
    configs = {
        "repa_full (dinov2)": {
            "gen": base / "gen/rxrx1_repa_full/aligned_mean_features.pt",
            "real": base / "real_rxrx1_aligned/rxrx1_repa_full_v1/train_features.pt",
        },
        "repa_openphenom_full": {
            "gen": base / "gen/rxrx1_repa_openphenom_full/aligned_mean_features.pt",
            "real": base / "real_rxrx1_aligned/rxrx1_repa_openphenom_full_v1/train_features.pt",
        },
    }

    # Also load dinov3 for reference
    dinov3_gen_data = torch.load(
        base / "gen/rxrx1_repa_openphenom_full/dinov3_meanpatch_features.pt",
        map_location="cpu", weights_only=False
    )
    dinov3_real_data = torch.load(
        base / "real_rxrx1_dinov3_meanpatch/train_features.pt",
        map_location="cpu", weights_only=False
    )
    dinov3_gen_feats = F.normalize(dinov3_gen_data["features"].float(), dim=-1)
    dinov3_real_feats = F.normalize(dinov3_real_data["features"].float(), dim=-1)

    # Build condition indices from gen filenames (same for all models on same gen set)
    # dinov2 and openphenom may have different gen sets, so build separately

    for model_name, paths in configs.items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")

        gen_feats, gen_data = load_and_normalize(paths["gen"])
        real_feats, real_data = load_and_normalize(paths["real"])

        print(f"Gen: {gen_feats.shape}, Real: {real_feats.shape}")

        # Build condition index for generated
        gen_cond_idx = build_condition_index(gen_data, gen_feats)

        # For real: check if metadata has condition info
        real_meta = real_data.get("metadata", {})
        if "cell_type_id" in real_meta and "sirna_id" in real_meta:
            real_ct = real_meta["cell_type_id"]
            real_si = real_meta["sirna_id"]
            if isinstance(real_ct, torch.Tensor):
                real_ct = real_ct.tolist()
                real_si = real_si.tolist()
            real_cond_idx = defaultdict(list)
            for i, (c, s) in enumerate(zip(real_ct, real_si)):
                real_cond_idx[(int(c), int(s))].append(i)
        else:
            print("  No real condition metadata, skipping per-condition analysis")
            continue

        # Find conditions in both real and gen with good sample sizes
        common_conds = set(gen_cond_idx.keys()) & set(real_cond_idx.keys())
        good_conds = [(c, min(len(gen_cond_idx[c]), len(real_cond_idx[c])))
                      for c in common_conds
                      if len(gen_cond_idx[c]) >= 20 and len(real_cond_idx[c]) >= 20]
        good_conds.sort(key=lambda x: -x[1])

        print(f"  Common conditions: {len(common_conds)}, with >=20 samples: {len(good_conds)}")

        # Global stats
        pairwise_gen = gen_feats[:500] @ gen_feats[:500].T
        mask = ~torch.eye(500, dtype=torch.bool)
        print(f"  Global gen pairwise sim (L2-normed): mean={pairwise_gen[mask].mean():.4f} std={pairwise_gen[mask].std():.4f}")

        # Per-condition KID for top conditions
        print(f"\n  Per-condition KID (top 10 by sample count):")
        kids = []
        for cond, min_n in good_conds[:10]:
            gi = gen_cond_idx[cond]
            ri = real_cond_idx[cond]
            gf = gen_feats[gi]
            rf = real_feats[ri]

            kid = compute_kid_cosine(gf, rf)

            # Within-condition stats
            if len(gi) > 1:
                gs = gf @ gf.T
                gmask = ~torch.eye(len(gi), dtype=torch.bool)
                within_gen = gs[gmask].mean().item()
                within_gen_std = gs[gmask].std().item()
            else:
                within_gen = within_gen_std = float('nan')

            if len(ri) > 1:
                rs = rf @ rf.T
                rmask = ~torch.eye(len(ri), dtype=torch.bool)
                within_real = rs[rmask].mean().item()
            else:
                within_real = float('nan')

            cross = (gf @ rf.T).mean().item()

            print(f"    cond={cond} gen={len(gi):3d} real={len(ri):3d} | "
                  f"KID={kid:+.6f} within_gen={within_gen:.4f}±{within_gen_std:.4f} "
                  f"within_real={within_real:.4f} cross={cross:.4f}")
            kids.append(kid)

        print(f"\n  KID stats over top conditions: mean={np.nanmean(kids):.6f} std={np.nanstd(kids):.6f}")

    # Also compute dinov3 reference
    print(f"\n{'='*70}")
    print("REFERENCE: DINOv3 (repa_openphenom_full images)")
    print(f"{'='*70}")

    dinov3_gen_cond_idx = build_condition_index(dinov3_gen_data, dinov3_gen_feats)
    # dinov3 real has different structure
    dinov3_real_meta = dinov3_real_data.get("metadata", {})
    if "cell_type_id" in dinov3_real_meta:
        real_ct = dinov3_real_meta["cell_type_id"]
        real_si = dinov3_real_meta["sirna_id"]
        if isinstance(real_ct, torch.Tensor):
            real_ct = real_ct.tolist()
            real_si = real_si.tolist()
        dinov3_real_cond_idx = defaultdict(list)
        for i, (c, s) in enumerate(zip(real_ct, real_si)):
            dinov3_real_cond_idx[(int(c), int(s))].append(i)

        common = set(dinov3_gen_cond_idx.keys()) & set(dinov3_real_cond_idx.keys())
        good = [(c, min(len(dinov3_gen_cond_idx[c]), len(dinov3_real_cond_idx[c])))
                for c in common
                if len(dinov3_gen_cond_idx[c]) >= 20 and len(dinov3_real_cond_idx[c]) >= 20]
        good.sort(key=lambda x: -x[1])

        print(f"  Good conditions: {len(good)}")
        kids_d = []
        for cond, min_n in good[:10]:
            gi = dinov3_gen_cond_idx[cond]
            ri = dinov3_real_cond_idx[cond]
            gf = dinov3_gen_feats[gi]
            rf = dinov3_real_feats[ri]
            kid = compute_kid_cosine(gf, rf)
            within_gen = (gf @ gf.T)[~torch.eye(len(gi), dtype=torch.bool)].mean().item()
            cross = (gf @ rf.T).mean().item()
            print(f"    cond={cond} gen={len(gi):3d} real={len(ri):3d} | KID={kid:+.6f} within_gen={within_gen:.4f} cross={cross:.4f}")
            kids_d.append(kid)
        print(f"\n  KID stats: mean={np.nanmean(kids_d):.6f} std={np.nanstd(kids_d):.6f}")


if __name__ == "__main__":
    main()
