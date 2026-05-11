"""
Diagnostic: Compare aligned_mean features for openphenom vs dinov2 REPA models.

Checks:
1. Feature shapes, norms, variance (are features collapsed?)
2. Within-condition cosine similarity distributions (real vs gen)
3. Cross-condition similarity (should be lower than within-condition)
4. Per-condition KID-like metric to see if features discriminate at all
5. Direct comparison: pick a few conditions, compute real-gen similarity

Uses repa_full (dinov2 teacher) as the "known good" baseline.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

DEVICE = "cpu"  # keep on CPU, features are large but we only sample

def load_features(path):
    """Load features .pt file, return dict with features + metadata."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        return data
    return {"features": data}


def feature_stats(feats, name):
    """Print basic stats about a feature tensor."""
    print(f"\n--- {name} ---")
    print(f"  shape: {feats.shape}")
    norms = feats.norm(dim=-1)
    print(f"  L2 norm: mean={norms.mean():.4f} std={norms.std():.4f} min={norms.min():.4f} max={norms.max():.4f}")
    # Variance per dimension
    var_per_dim = feats.var(dim=0)
    print(f"  per-dim variance: mean={var_per_dim.mean():.6f} std={var_per_dim.std():.6f}")
    print(f"  # zero-variance dims: {(var_per_dim < 1e-8).sum().item()}")
    # Effective rank (via singular values)
    if feats.shape[0] > 50:
        subset = feats[:min(500, feats.shape[0])].float()
        subset = subset - subset.mean(dim=0)
        try:
            s = torch.linalg.svdvals(subset)
            s_norm = s / s.sum()
            entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
            eff_rank = torch.exp(entropy).item()
            print(f"  effective rank (500 samples): {eff_rank:.1f} / {min(500, feats.shape[1])}")
            print(f"  top-5 singular values: {s[:5].tolist()}")
            print(f"  s[0]/s[-1] ratio: {s[0]/s[-1]:.1f}")
        except Exception as e:
            print(f"  SVD failed: {e}")
    # Check for NaN/Inf
    print(f"  NaN: {feats.isnan().any().item()}, Inf: {feats.isinf().any().item()}")


def pairwise_cosine_stats(feats, name, n_samples=500):
    """Compute pairwise cosine similarity stats on a subset."""
    idx = torch.randperm(feats.shape[0])[:n_samples]
    subset = F.normalize(feats[idx].float(), dim=-1)
    sim = subset @ subset.T
    # Remove diagonal
    mask = ~torch.eye(n_samples, dtype=torch.bool)
    off_diag = sim[mask]
    print(f"\n  Pairwise cosine sim ({name}, {n_samples} samples):")
    print(f"    mean={off_diag.mean():.4f} std={off_diag.std():.4f}")
    print(f"    min={off_diag.min():.4f} max={off_diag.max():.4f}")
    print(f"    fraction > 0.9: {(off_diag > 0.9).float().mean():.4f}")
    print(f"    fraction > 0.95: {(off_diag > 0.95).float().mean():.4f}")


def condition_analysis(gen_feats, gen_data, real_feats, real_data, name, n_conditions=5):
    """Pick top conditions by sample count, compare real vs gen features."""
    # Build condition index for generated
    gen_conds = gen_data.get("conditions") or gen_data.get("condition_ids")
    real_conds = real_data.get("conditions") or real_data.get("condition_ids")

    if gen_conds is None or real_conds is None:
        print(f"\n  Cannot do condition analysis for {name}: missing condition metadata")
        # Try filenames approach
        if "filenames" in gen_data:
            print(f"  gen has filenames: {len(gen_data['filenames'])} entries")
            print(f"  sample filenames: {gen_data['filenames'][:3]}")
        return

    if isinstance(gen_conds, torch.Tensor):
        gen_conds = gen_conds.numpy()
    if isinstance(real_conds, torch.Tensor):
        real_conds = real_conds.numpy()

    # Group by condition
    gen_by_cond = defaultdict(list)
    for i, c in enumerate(gen_conds):
        key = tuple(c) if hasattr(c, '__iter__') else (c,)
        gen_by_cond[key].append(i)

    real_by_cond = defaultdict(list)
    for i, c in enumerate(real_conds):
        key = tuple(c) if hasattr(c, '__iter__') else (c,)
        real_by_cond[key].append(i)

    # Pick conditions that exist in both real and gen with good sample sizes
    common_conds = set(gen_by_cond.keys()) & set(real_by_cond.keys())
    print(f"\n  Condition analysis ({name}):")
    print(f"  # gen conditions: {len(gen_by_cond)}, # real conditions: {len(real_by_cond)}, # common: {len(common_conds)}")

    # Sort by min(gen, real) count
    sorted_conds = sorted(common_conds, key=lambda c: min(len(gen_by_cond[c]), len(real_by_cond[c])), reverse=True)

    for cond in sorted_conds[:n_conditions]:
        gi = gen_by_cond[cond]
        ri = real_by_cond[cond]
        gf = F.normalize(gen_feats[gi].float(), dim=-1)
        rf = F.normalize(real_feats[ri].float(), dim=-1)

        # Within-gen similarity
        if len(gi) > 1:
            gs = gf @ gf.T
            mask = ~torch.eye(len(gi), dtype=torch.bool)
            within_gen = gs[mask].mean().item()
        else:
            within_gen = float('nan')

        # Within-real similarity
        if len(ri) > 1:
            rs = rf @ rf.T
            mask = ~torch.eye(len(ri), dtype=torch.bool)
            within_real = rs[mask].mean().item()
        else:
            within_real = float('nan')

        # Cross real-gen similarity (mean of all pairwise)
        cross = (gf @ rf.T).mean().item()

        # Mean feature distance (for KID-like metric)
        gen_mean = gf.mean(dim=0)
        real_mean = rf.mean(dim=0)
        mean_cos = F.cosine_similarity(gen_mean.unsqueeze(0), real_mean.unsqueeze(0)).item()

        print(f"    cond={cond}: gen={len(gi)} real={len(ri)} | "
              f"within_gen={within_gen:.4f} within_real={within_real:.4f} "
              f"cross={cross:.4f} mean_cos={mean_cos:.4f}")


def main():
    base = Path("/mnt/pvc/faithful-cond-gen/outputs")

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

    for model_name, paths in configs.items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")

        gen_data = load_features(paths["gen"])
        real_data = load_features(paths["real"])

        gen_feats = gen_data["features"]
        real_feats = real_data["features"]

        print(f"\nKeys in gen data: {list(gen_data.keys())}")
        print(f"Keys in real data: {list(real_data.keys())}")

        feature_stats(gen_feats, f"{model_name} GEN")
        feature_stats(real_feats, f"{model_name} REAL")

        pairwise_cosine_stats(gen_feats, f"{model_name} GEN")
        pairwise_cosine_stats(real_feats, f"{model_name} REAL")

        # Cross real-gen pairwise
        n = 300
        gi = torch.randperm(gen_feats.shape[0])[:n]
        ri = torch.randperm(real_feats.shape[0])[:n]
        gf = F.normalize(gen_feats[gi].float(), dim=-1)
        rf = F.normalize(real_feats[ri].float(), dim=-1)
        cross_sim = (gf @ rf.T)
        print(f"\n  Cross real-gen cosine sim ({n} samples each):")
        print(f"    mean={cross_sim.mean():.4f} std={cross_sim.std():.4f}")

        condition_analysis(gen_feats, gen_data, real_feats, real_data, model_name)

    # Also check dinov3 features for reference
    print(f"\n{'='*70}")
    print("REFERENCE: DINOv3 meanpatch features (repa_openphenom_full)")
    print(f"{'='*70}")
    dinov3_gen = load_features(base / "gen/rxrx1_repa_openphenom_full/dinov3_meanpatch_features.pt")
    dinov3_real = load_features(base / "real_rxrx1_dinov3_meanpatch/train_features.pt")
    feature_stats(dinov3_gen["features"], "dinov3 GEN")
    feature_stats(dinov3_real["features"], "dinov3 REAL")
    pairwise_cosine_stats(dinov3_gen["features"], "dinov3 GEN")
    pairwise_cosine_stats(dinov3_real["features"], "dinov3 REAL")


if __name__ == "__main__":
    main()
