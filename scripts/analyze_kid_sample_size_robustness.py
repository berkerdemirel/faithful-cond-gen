"""
Analyze robustness of ΔKID rankings across different fixed sample sizes.

Currently ΔKID is computed using variable sample sizes per condition.
This script fixes sample size across all conditions to check if rankings are robust.

Sample sizes tested: 20, 40, 60, 80, 100, 200 reals per condition.
Conditions without enough samples are excluded.

Usage:
    PYTHONPATH=src uv run python scripts/analyze_kid_sample_size_robustness.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from faithful_cond_gen.eval.trust_eval.config import (
    CONDITION_ATTRS,
    FEATURE_CONFIGS,
    MARGINAL_SEEN_COMBOS,
    REAL_FEATURE_PATHS,
)
from faithful_cond_gen.eval.trust_eval.metrics_kid import calculate_kid_same_m
from faithful_cond_gen.eval.trust_eval.scoring_core import normalize_features


def load_features(dataset: str, model: str, feature_type: str):
    """Load generated and real features."""
    gen_cfg = FEATURE_CONFIGS.get((dataset, model, feature_type))
    if gen_cfg is None:
        raise ValueError(f"No config for {dataset}/{model}/{feature_type}")

    gen_dir, gen_file = gen_cfg
    gen_path = Path(f"outputs/gen/{gen_dir}/{gen_file}")
    gen_data = torch.load(gen_path, weights_only=False)
    gen_feats = gen_data["features"]
    gen_meta = gen_data.get("metadata", gen_data.get("cond", {}))

    real_path = Path(REAL_FEATURE_PATHS[(dataset, feature_type)])
    real_data = torch.load(real_path, weights_only=False)
    real_feats = real_data["features"]
    real_meta = real_data.get("metadata", real_data.get("cond", {}))

    return gen_feats, gen_meta, real_feats, real_meta


def compute_delta_kid_fixed_size(
    gen_feats_np, real_feats_np, gen_by_cond, real_by_cond, n_samples, n_bootstrap=20
):
    """
    Compute ΔKID per condition using fixed sample size.

    Args:
        n_samples: Fixed number of real samples to use per condition.
                   We use n_samples/2 for each split (real_a, real_b).
    """
    rng = np.random.default_rng(42)
    results = {}

    # We need 2*k samples for real splits, where k = n_samples/2
    k = n_samples // 2

    for cond in gen_by_cond:
        gen_idx = gen_by_cond[cond]
        real_idx = real_by_cond.get(cond, [])

        n_real = len(real_idx)
        n_gen = len(gen_idx)

        # Skip if not enough samples
        if n_real < n_samples or n_gen < k:
            results[cond] = np.nan
            continue

        gen_f = gen_feats_np[gen_idx]
        real_f = real_feats_np[real_idx]

        deltas = []
        for _ in range(n_bootstrap):
            # Sample exactly k reals for each split
            perm = rng.permutation(n_real)
            real_a = real_f[perm[:k]]
            real_b = real_f[perm[k:2*k]]

            # Sample k generated samples
            gen_samp = gen_f[rng.choice(n_gen, k, replace=n_gen < k)]

            # KID(real_a, real_b) - baseline
            kid_rr = calculate_kid_same_m(real_a, real_b, use_cosine=True)

            # KID(gen, real_a)
            kid_gr = calculate_kid_same_m(gen_samp, real_a, use_cosine=True)

            if np.isfinite(kid_rr) and np.isfinite(kid_gr):
                deltas.append(kid_gr - kid_rr)

        results[cond] = np.mean(deltas) if deltas else np.nan

    return results


def analyze_model(dataset, model, feature_type="dinov3"):
    """Analyze ΔKID robustness for a model."""
    condition_keys = CONDITION_ATTRS[dataset]
    sample_sizes = [20, 40, 60, 80, 100, 200]

    print(f"\n{'='*80}")
    print(f"Model: {dataset}/{model}")
    print(f"{'='*80}")

    # Load features
    gen_feats, gen_meta, real_feats, real_meta = load_features(dataset, model, feature_type)
    print(f"Generated: {gen_feats.shape}, Real: {real_feats.shape}")

    gen_feats_np = normalize_features(gen_feats).numpy()
    real_feats_np = normalize_features(real_feats).numpy()

    # Group by condition
    gen_by_cond = {}
    for i in range(len(gen_feats)):
        cond = tuple(
            int(gen_meta[k][i].item() if isinstance(gen_meta[k][i], torch.Tensor) else gen_meta[k][i])
            for k in condition_keys
        )
        gen_by_cond.setdefault(cond, []).append(i)

    real_by_cond = {}
    for i in range(len(real_feats)):
        cond = tuple(
            int(real_meta[k][i].item() if isinstance(real_meta[k][i], torch.Tensor) else real_meta[k][i])
            for k in condition_keys
        )
        real_by_cond.setdefault(cond, []).append(i)

    # Print sample counts per condition
    print("\nSamples per condition:")
    print("-" * 60)
    for cond in sorted(gen_by_cond.keys()):
        n_gen = len(gen_by_cond[cond])
        n_real = len(real_by_cond.get(cond, []))
        is_seen = cond in MARGINAL_SEEN_COMBOS if "marginal" in model else True
        seen_str = "SEEN" if is_seen else "UNSN"
        print(f"  {cond}: gen={n_gen}, real={n_real} [{seen_str}]")

    # Compute ΔKID for each sample size
    all_results = {}
    for n_samp in sample_sizes:
        print(f"\nComputing ΔKID with n_samples={n_samp}...")
        results = compute_delta_kid_fixed_size(
            gen_feats_np, real_feats_np, gen_by_cond, real_by_cond, n_samp
        )
        all_results[n_samp] = results

    # Build combined DataFrame
    rows = []
    for cond in sorted(gen_by_cond.keys()):
        is_seen = cond in MARGINAL_SEEN_COMBOS if "marginal" in model else True
        row = {
            "condition": str(cond),
            "is_seen": is_seen,
            "n_real": len(real_by_cond.get(cond, [])),
        }
        for n_samp in sample_sizes:
            row[f"dkid_{n_samp}"] = all_results[n_samp].get(cond, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Print table
    print(f"\nΔKID per condition at different sample sizes:")
    print("-" * 120)

    def fmt(x):
        if pd.isna(x):
            return "   -   "
        elif isinstance(x, bool):
            return "Y" if x else "N"
        elif isinstance(x, float):
            return f"{x:7.4f}"
        return str(x)

    cols = ["condition", "is_seen", "n_real"] + [f"dkid_{n}" for n in sample_sizes]
    print(df[cols].to_string(index=False, formatters={c: fmt for c in cols}))

    # Compute rank correlations between different sample sizes
    print(f"\n\nRANK CORRELATIONS (Spearman ρ) between sample sizes:")
    print("-" * 80)

    # Header
    header = "         " + "".join(f"  n={n:3d}" for n in sample_sizes)
    print(header)

    corr_matrix = np.zeros((len(sample_sizes), len(sample_sizes)))

    for i, n1 in enumerate(sample_sizes):
        row_str = f"n={n1:3d}    "
        for j, n2 in enumerate(sample_sizes):
            col1 = f"dkid_{n1}"
            col2 = f"dkid_{n2}"
            valid = df[df[col1].notna() & df[col2].notna()]

            if len(valid) >= 3:
                rho, _ = spearmanr(valid[col1], valid[col2])
                corr_matrix[i, j] = rho
                row_str += f"  {rho:5.3f}"
            else:
                corr_matrix[i, j] = np.nan
                row_str += "    -  "
        print(row_str)

    # Summary: correlation with largest sample size (200)
    print(f"\n\nCORRELATION WITH n=200 (most stable reference):")
    print("-" * 60)

    ref_col = "dkid_200"
    for n_samp in sample_sizes[:-1]:  # exclude 200 vs 200
        col = f"dkid_{n_samp}"
        valid = df[df[col].notna() & df[ref_col].notna()]

        if len(valid) >= 3:
            rho, p = spearmanr(valid[col], valid[ref_col])
            tau, _ = kendalltau(valid[col], valid[ref_col])
            print(f"  n={n_samp:3d} vs n=200: ρ = {rho:.4f} (p={p:.4f}), τ = {tau:.4f}, n_cond={len(valid)}")
        else:
            print(f"  n={n_samp:3d} vs n=200: insufficient conditions")

    # Check rank stability: which conditions are consistently top/bottom?
    print(f"\n\nRANK STABILITY ANALYSIS:")
    print("-" * 60)

    # Get ranks for each sample size (lower ΔKID = better rank)
    rank_cols = []
    for n_samp in sample_sizes:
        col = f"dkid_{n_samp}"
        rank_col = f"rank_{n_samp}"
        df[rank_col] = df[col].rank(ascending=True, na_option='keep')
        rank_cols.append(rank_col)

    # Compute mean rank and std across sample sizes
    df["mean_rank"] = df[rank_cols].mean(axis=1)
    df["std_rank"] = df[rank_cols].std(axis=1)

    # Conditions that have all sample sizes
    complete = df[df[rank_cols].notna().all(axis=1)].copy()
    if len(complete) > 0:
        complete = complete.sort_values("mean_rank")
        print(f"\nConditions with all sample sizes (n={len(complete)}):")
        print(f"{'condition':<20} {'seen':<5} " + " ".join(f"r{n:3d}" for n in sample_sizes) + " mean±std")
        print("-" * 80)
        for _, row in complete.iterrows():
            ranks = [row[f"rank_{n}"] for n in sample_sizes]
            rank_str = " ".join(f"{r:4.0f}" for r in ranks)
            seen_str = "Y" if row["is_seen"] else "N"
            print(f"{row['condition']:<20} {seen_str:<5} {rank_str} {row['mean_rank']:.1f}±{row['std_rank']:.1f}")

    return df, corr_matrix


def main():
    print("=" * 80)
    print("ΔKID SAMPLE SIZE ROBUSTNESS ANALYSIS")
    print("=" * 80)
    print("\nQuestion: Are ΔKID rankings robust to sample size?")
    print("Method: Compute ΔKID at fixed sample sizes (20,40,60,80,100,200) across conditions")
    print("        and check rank correlations.")

    # CelebA marginal
    df_marginal, corr_marginal = analyze_model("celeba", "repa_marginal")

    # CelebA full
    df_full, corr_full = analyze_model("celeba", "repa_full")

    # Save
    output_dir = Path("outputs/trust_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    df_marginal.to_csv(output_dir / "kid_sample_robustness_marginal.csv", index=False)
    df_full.to_csv(output_dir / "kid_sample_robustness_full.csv", index=False)
    print(f"\n\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
