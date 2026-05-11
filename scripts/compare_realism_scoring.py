"""
Compare Single Gaussian vs Mixture Gaussian realism scoring.

Computes per-condition ΔKID (ground truth) and correlates with:
1. Single Gaussian realism (global mean/cov)
2. Mixture Gaussian realism (per-component means, shared cov)

For both CelebA marginal and full models.

Usage:
    PYTHONPATH=src uv run python scripts/compare_realism_scoring.py
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
from faithful_cond_gen.eval.trust_eval.scoring_core import (
    normalize_features,
    fit_global_stats,
    fit_mixture_stats,
    compute_real_calibration_for_global_energy,
    compute_real_calibration_for_mixture_energy,
    compute_mahalanobis,
    zscore,
)


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


def compute_delta_kid_per_condition(
    gen_feats, gen_meta, real_feats, real_meta, condition_keys, n_bootstrap=20
):
    """Compute ΔKID per condition."""
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

    delta_kids = {}
    rng = np.random.default_rng(42)

    for cond in gen_by_cond:
        gen_idx = gen_by_cond[cond]
        real_idx = real_by_cond.get(cond, [])

        if len(real_idx) < 20 or len(gen_idx) < 5:
            delta_kids[cond] = np.nan
            continue

        k = min(len(real_idx) // 2, len(gen_idx), 200)
        if k < 5:
            delta_kids[cond] = np.nan
            continue

        gen_f = gen_feats_np[gen_idx]
        real_f = real_feats_np[real_idx]

        deltas = []
        for _ in range(n_bootstrap):
            perm = rng.permutation(len(real_idx))
            real_a, real_b = real_f[perm[:k]], real_f[perm[k:2*k]]
            gen_samp = gen_f[rng.choice(len(gen_f), k, replace=False)]

            base = calculate_kid_same_m(real_a, real_b, use_cosine=True)
            gen_kid = calculate_kid_same_m(real_a, gen_samp, use_cosine=True)
            if np.isfinite(base) and np.isfinite(gen_kid):
                deltas.append(gen_kid - base)

        delta_kids[cond] = np.mean(deltas) if deltas else np.nan

    return delta_kids, gen_by_cond, real_by_cond


def compute_single_gaussian_realism(gen_feats, real_feats, real_meta, condition_keys):
    """Compute single Gaussian realism scores."""
    # Fit global stats on real
    global_stats = fit_global_stats(real_feats)

    # Calibrate on real
    real_E_mean, real_E_std = compute_real_calibration_for_global_energy(real_feats, global_stats)

    # Score generated
    gen_normed = normalize_features(gen_feats)
    mu = global_stats["mu"]
    P = global_stats["precision"]

    energies = compute_mahalanobis(gen_normed, mu, P).numpy()
    z_scores = zscore(energies, real_E_mean, real_E_std)

    return z_scores


def compute_mixture_gaussian_realism(gen_feats, gen_meta, real_feats, real_meta, condition_keys, dataset, model):
    """Compute mixture Gaussian realism scores."""
    # Fit mixture stats on real
    mixture_stats = fit_mixture_stats(
        real_feats, real_meta, condition_keys, dataset, model
    )

    n_components = mixture_stats.get("n_components", 0)
    if n_components == 0:
        # Fallback to global
        return compute_single_gaussian_realism(gen_feats, real_feats, real_meta, condition_keys), 0

    # Calibrate on real
    real_E_mean, real_E_std = compute_real_calibration_for_mixture_energy(
        real_feats, real_meta, condition_keys, mixture_stats, dataset, model
    )

    # Score generated
    gen_normed = normalize_features(gen_feats)
    mus = mixture_stats["mus"]
    P = mixture_stats["precision"]
    component_keys = mixture_stats["component_keys"]

    # Stack means
    mus_stacked = torch.stack([mus[k] for k in component_keys], dim=0)

    # Compute min distance to any component
    N = len(gen_feats)
    energies = np.zeros(N)
    batch_size = 2000

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x = gen_normed[start:end]
        centered = x.unsqueeze(1) - mus_stacked.unsqueeze(0)
        term = torch.einsum("bkd,de->bke", centered, P)
        dists = torch.sum(term * centered, dim=2).numpy()
        energies[start:end] = dists.min(axis=1)

    z_scores = zscore(energies, real_E_mean, real_E_std)

    return z_scores, n_components


def analyze_model(dataset, model, feature_type="dinov3"):
    """Analyze single vs mixture realism for a model."""
    condition_keys = CONDITION_ATTRS[dataset]

    print(f"\n{'='*80}")
    print(f"Model: {dataset}/{model}")
    print(f"{'='*80}")

    # Load features
    gen_feats, gen_meta, real_feats, real_meta = load_features(dataset, model, feature_type)
    print(f"Generated: {gen_feats.shape}, Real: {real_feats.shape}")

    # Compute ΔKID per condition
    print("Computing ΔKID per condition...")
    delta_kids, gen_by_cond, real_by_cond = compute_delta_kid_per_condition(
        gen_feats, gen_meta, real_feats, real_meta, condition_keys
    )

    # Compute single Gaussian realism
    print("Computing single Gaussian realism...")
    single_z = compute_single_gaussian_realism(gen_feats, real_feats, real_meta, condition_keys)

    # Compute mixture Gaussian realism
    print("Computing mixture Gaussian realism...")
    mixture_z, n_components = compute_mixture_gaussian_realism(
        gen_feats, gen_meta, real_feats, real_meta, condition_keys, dataset, model
    )
    print(f"  Mixture components: {n_components}")

    # Aggregate per condition
    results = []
    for cond in sorted(gen_by_cond.keys()):
        idx = gen_by_cond[cond]
        is_seen = cond in MARGINAL_SEEN_COMBOS if "marginal" in model else True

        row = {
            "condition": str(cond),
            "is_seen": is_seen,
            "n_gen": len(idx),
            "n_real": len(real_by_cond.get(cond, [])),
            "delta_kid": delta_kids.get(cond, np.nan),
            "single_realism_mean": float(np.mean(single_z[idx])),
            "mixture_realism_mean": float(np.mean(mixture_z[idx])),
        }
        results.append(row)

    df = pd.DataFrame(results)

    # Print table
    print(f"\nPer-condition results (n_components={n_components}):")
    print("-" * 100)
    cols = ["condition", "is_seen", "n_gen", "n_real", "delta_kid", "single_realism_mean", "mixture_realism_mean"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "NaN"))

    # Compute correlations
    valid = df[df["delta_kid"].notna()]

    print(f"\n\nCORRELATIONS with ΔKID (n={len(valid)} conditions):")
    print("-" * 60)

    # Single Gaussian
    rho_single, p_single = spearmanr(valid["single_realism_mean"], valid["delta_kid"])
    tau_single, _ = kendalltau(valid["single_realism_mean"], valid["delta_kid"])

    # Mixture Gaussian
    rho_mixture, p_mixture = spearmanr(valid["mixture_realism_mean"], valid["delta_kid"])
    tau_mixture, _ = kendalltau(valid["mixture_realism_mean"], valid["delta_kid"])

    print(f"Single Gaussian:  Spearman ρ = {rho_single:.4f} (p={p_single:.4f}), Kendall τ = {tau_single:.4f}")
    print(f"Mixture Gaussian: Spearman ρ = {rho_mixture:.4f} (p={p_mixture:.4f}), Kendall τ = {tau_mixture:.4f}")

    # For marginal models, also compute correlations separately for seen/unseen
    if "marginal" in model:
        seen = valid[valid["is_seen"] == True]
        unseen = valid[valid["is_seen"] == False]

        if len(seen) >= 3:
            rho_s_seen, _ = spearmanr(seen["single_realism_mean"], seen["delta_kid"])
            rho_m_seen, _ = spearmanr(seen["mixture_realism_mean"], seen["delta_kid"])
            print(f"\n  SEEN only (n={len(seen)}):   Single ρ = {rho_s_seen:.4f}, Mixture ρ = {rho_m_seen:.4f}")

        if len(unseen) >= 3:
            rho_s_unseen, _ = spearmanr(unseen["single_realism_mean"], unseen["delta_kid"])
            rho_m_unseen, _ = spearmanr(unseen["mixture_realism_mean"], unseen["delta_kid"])
            print(f"  UNSEEN only (n={len(unseen)}): Single ρ = {rho_s_unseen:.4f}, Mixture ρ = {rho_m_unseen:.4f}")

    return df, {
        "model": model,
        "n_components": n_components,
        "rho_single": rho_single,
        "rho_mixture": rho_mixture,
        "tau_single": tau_single,
        "tau_mixture": tau_mixture,
    }


def main():
    print("=" * 80)
    print("COMPARING SINGLE vs MIXTURE GAUSSIAN REALISM SCORING")
    print("=" * 80)

    summaries = []

    # CelebA marginal
    df_marginal, summary_marginal = analyze_model("celeba", "repa_marginal")
    summaries.append(summary_marginal)

    # CelebA full
    df_full, summary_full = analyze_model("celeba", "repa_full")
    summaries.append(summary_full)

    # Final summary
    print("\n\n" + "=" * 80)
    print("SUMMARY: Correlation with ΔKID")
    print("=" * 80)
    summary_df = pd.DataFrame(summaries)
    print(summary_df.to_string(index=False))

    # Save
    output_dir = Path("outputs/trust_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    df_marginal.to_csv(output_dir / "realism_comparison_marginal.csv", index=False)
    df_full.to_csv(output_dir / "realism_comparison_full.csv", index=False)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
