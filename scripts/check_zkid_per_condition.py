"""
Compute z-KID per condition for CelebA marginal model.

Outputs a table with:
- Condition
- n_real, n_gen
- KID(real, real) baseline (mu_c from null)
- KID(gen, real)
- z-KID = (KID(gen, real) - mu_c) / sigma_c

Usage:
    PYTHONPATH=src uv run python scripts/check_zkid_per_condition.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from faithful_cond_gen.eval.trust_eval.config import (
    CONDITION_ATTRS,
    FEATURE_CONFIGS,
    MARGINAL_SEEN_COMBOS,
    REAL_FEATURE_PATHS,
)
from faithful_cond_gen.eval.trust_eval.metrics_kid import (
    calculate_kid_same_m,
    estimate_kid_null_per_condition,
)


def load_features(dataset: str, model: str, feature_type: str, normalize: bool = True):
    """Load generated and real features."""
    # Generated features
    gen_cfg = FEATURE_CONFIGS.get((dataset, model, feature_type))
    if gen_cfg is None:
        raise ValueError(f"No config for {dataset}/{model}/{feature_type}")

    gen_dir, gen_file = gen_cfg
    gen_path = Path(f"outputs/gen/{gen_dir}/{gen_file}")
    if not gen_path.exists():
        raise FileNotFoundError(f"Generated features not found: {gen_path}")

    gen_data = torch.load(gen_path, weights_only=False)
    gen_feats = gen_data["features"]
    gen_meta = gen_data.get("metadata", gen_data.get("cond", {}))

    # Real features
    real_path = Path(REAL_FEATURE_PATHS[(dataset, feature_type)])
    if not real_path.exists():
        raise FileNotFoundError(f"Real features not found: {real_path}")

    real_data = torch.load(real_path, weights_only=False)
    real_feats = real_data["features"]
    real_meta = real_data.get("metadata", real_data.get("cond", {}))

    # L2 normalize
    if normalize:
        gen_feats = gen_feats / (gen_feats.norm(dim=1, keepdim=True) + 1e-12)
        real_feats = real_feats / (real_feats.norm(dim=1, keepdim=True) + 1e-12)

    return gen_feats, gen_meta, real_feats, real_meta


def main():
    dataset = "celeba"
    model = "repa_marginal"
    feature_type = "dinov3"
    condition_keys = CONDITION_ATTRS[dataset]

    print(f"Loading features for {dataset}/{model}/{feature_type}...")
    gen_feats, gen_meta, real_feats, real_meta = load_features(dataset, model, feature_type)

    print(f"  Generated: {gen_feats.shape}")
    print(f"  Real: {real_feats.shape}")

    # Group by condition
    gen_feats_np = gen_feats.numpy()
    real_feats_np = real_feats.numpy()

    # Group real by condition
    real_by_cond = {}
    for i in range(len(real_feats)):
        cond = tuple(
            int(real_meta[k][i].item() if isinstance(real_meta[k][i], torch.Tensor) else real_meta[k][i])
            for k in condition_keys
        )
        real_by_cond.setdefault(cond, []).append(i)

    # Group gen by condition
    gen_by_cond = {}
    for i in range(len(gen_feats)):
        cond = tuple(
            int(gen_meta[k][i].item() if isinstance(gen_meta[k][i], torch.Tensor) else gen_meta[k][i])
            for k in condition_keys
        )
        gen_by_cond.setdefault(cond, []).append(i)

    print(f"\nConditions in real: {len(real_by_cond)}")
    print(f"Conditions in gen: {len(gen_by_cond)}")

    # Estimate null distribution per condition
    print("\nEstimating KID null distribution per condition...")
    real_feats_by_cond = {cond: real_feats_np[idx] for cond, idx in real_by_cond.items()}
    kid_null_stats = estimate_kid_null_per_condition(
        real_feats_by_cond,
        n_resamples=100,
        use_cosine=True,  # L2-normalized features
        seed=42,
    )
    print(f"  Null stats for {len(kid_null_stats)} conditions")

    # Compute KID per condition - both real-real and gen-real
    print("\nComputing KID per condition (real-real and gen-real)...")

    results = []
    for cond in sorted(gen_by_cond.keys()):
        gen_idx = gen_by_cond[cond]
        real_idx = real_by_cond.get(cond, [])

        n_gen = len(gen_idx)
        n_real = len(real_idx)
        is_seen = cond in MARGINAL_SEEN_COMBOS

        row = {
            "condition": str(cond),
            "is_seen": is_seen,
            "n_gen": n_gen,
            "n_real": n_real,
        }

        if n_real < 20 or n_gen < 5:
            row["kid_real_real"] = np.nan
            row["kid_gen_real"] = np.nan
            row["z_real_real"] = np.nan
            row["z_gen_real"] = np.nan
            row["mu_c"] = np.nan
            row["sigma_c"] = np.nan
            results.append(row)
            continue

        # Use consistent sample size k for both
        k = min(n_real // 2, n_gen, 200)
        if k < 5:
            row["kid_real_real"] = np.nan
            row["kid_gen_real"] = np.nan
            row["z_real_real"] = np.nan
            row["z_gen_real"] = np.nan
            row["mu_c"] = np.nan
            row["sigma_c"] = np.nan
            results.append(row)
            continue

        gen_f = gen_feats_np[gen_idx]
        real_f = real_feats_np[real_idx]

        rng = np.random.default_rng(42)

        # Bootstrap KID(real, real) - split real into two halves
        kids_rr = []
        for _ in range(20):
            perm = rng.permutation(n_real)
            real_a = real_f[perm[:k]]
            real_b = real_f[perm[k:2*k]]
            kid = calculate_kid_same_m(real_a, real_b, use_cosine=True)
            if np.isfinite(kid):
                kids_rr.append(kid)

        kid_real_real = np.mean(kids_rr) if kids_rr else np.nan

        # Bootstrap KID(gen, real)
        kids_gr = []
        for _ in range(20):
            gen_samp = gen_f[rng.choice(n_gen, k, replace=False)]
            real_samp = real_f[rng.choice(n_real, k, replace=False)]
            kid = calculate_kid_same_m(real_samp, gen_samp, use_cosine=True)
            if np.isfinite(kid):
                kids_gr.append(kid)

        kid_gen_real = np.mean(kids_gr) if kids_gr else np.nan

        # z-scores using null stats
        if cond in kid_null_stats:
            mu_c, sigma_c = kid_null_stats[cond]
            z_real_real = (kid_real_real - mu_c) / sigma_c if np.isfinite(kid_real_real) else np.nan
            z_gen_real = (kid_gen_real - mu_c) / sigma_c if np.isfinite(kid_gen_real) else np.nan
        else:
            mu_c, sigma_c = np.nan, np.nan
            z_real_real = np.nan
            z_gen_real = np.nan

        row["mu_c"] = mu_c
        row["sigma_c"] = sigma_c
        row["kid_real_real"] = kid_real_real
        row["kid_gen_real"] = kid_gen_real
        row["z_real_real"] = z_real_real
        row["z_gen_real"] = z_gen_real
        results.append(row)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by is_seen (seen first), then by condition
    df = df.sort_values(["is_seen", "condition"], ascending=[False, True])

    # Print single unified table
    print("\n" + "=" * 120)
    print("z-KID per Condition (CelebA Marginal Model)")
    print("=" * 120)
    print("\nColumns: z_real_real should be ~0 (sanity), z_gen_real is what we care about")
    print("-" * 120)

    cols = ["condition", "is_seen", "n_gen", "n_real", "mu_c", "sigma_c", "kid_real_real", "kid_gen_real", "z_real_real", "z_gen_real"]

    # Format floats nicely
    def fmt(x):
        if pd.isna(x):
            return "NaN"
        elif isinstance(x, bool):
            return "SEEN" if x else "unseen"
        elif isinstance(x, float):
            if abs(x) < 0.01:
                return f"{x:.2e}"
            return f"{x:.4f}"
        return str(x)

    print(df[cols].to_string(index=False, formatters={c: fmt for c in cols}))

    # Summary statistics
    print("\n\nSUMMARY STATISTICS:")
    print("-" * 60)
    seen_df = df[df["is_seen"] == True]
    unseen_df = df[df["is_seen"] == False]

    # z_real_real should be ~0
    z_rr = df["z_real_real"].dropna()
    print(f"z_real_real (sanity): mean={z_rr.mean():.2f}, std={z_rr.std():.2f} (should be ~0, ~1)")

    # z_gen_real by seen/unseen
    seen_z = seen_df["z_gen_real"].dropna()
    unseen_z = unseen_df["z_gen_real"].dropna()
    print(f"\nz_gen_real (SEEN):    mean={seen_z.mean():.2f}, std={seen_z.std():.2f}")
    print(f"z_gen_real (unseen):  mean={unseen_z.mean():.2f}, std={unseen_z.std():.2f}")

    # Save to CSV
    output_path = Path("outputs/trust_evaluation/zkid_per_condition_marginal.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
