"""
Analyze correlations of each score component with ΔKID.

Compares:
1. Supervised mixture (components = condition labels)
2. Unsupervised GMM (k components fitted via sklearn GMM)

Uses PCA for dimensionality reduction before GMM fitting.

Usage:
    PYTHONPATH=src uv run python scripts/analyze_score_component_correlations_with_gmm.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, kendalltau
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

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


def compute_realism_single_per_sample(gen_feats, real_feats):
    """Compute single Gaussian realism scores."""
    global_stats = fit_global_stats(real_feats)
    real_E_mean, real_E_std = compute_real_calibration_for_global_energy(real_feats, global_stats)

    gen_normed = normalize_features(gen_feats)
    mu = global_stats["mu"]
    P = global_stats["precision"]

    energies = compute_mahalanobis(gen_normed, mu, P).numpy()
    z_scores = zscore(energies, real_E_mean, real_E_std)

    return z_scores


def compute_realism_supervised_mixture(gen_feats, gen_meta, real_feats, real_meta, condition_keys, dataset, model):
    """
    Compute supervised mixture Gaussian realism scores.
    Components = condition labels.
    """
    mixture_stats = fit_mixture_stats(
        real_feats, real_meta, condition_keys, dataset, model
    )

    n_components = mixture_stats.get("n_components", 0)
    if n_components == 0:
        return compute_realism_single_per_sample(gen_feats, real_feats), 0

    real_E_mean, real_E_std = compute_real_calibration_for_mixture_energy(
        real_feats, real_meta, condition_keys, mixture_stats, dataset, model
    )

    gen_normed = normalize_features(gen_feats)
    mus = mixture_stats["mus"]
    P = mixture_stats["precision"]
    component_keys = mixture_stats["component_keys"]

    mus_stacked = torch.stack([mus[k] for k in component_keys], dim=0)

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


def fit_gmm_with_pca(real_feats, n_components, pca_dim=64, random_state=42):
    """
    Fit unsupervised GMM on real features with PCA dimensionality reduction.

    Args:
        real_feats: Real features (N, D)
        n_components: Number of mixture components
        pca_dim: Target dimensionality after PCA

    Returns:
        pca: Fitted PCA model
        gmm: Fitted GMM model
    """
    real_np = normalize_features(real_feats).numpy()

    # PCA
    print(f"    Fitting PCA ({real_np.shape[1]} -> {pca_dim})...")
    pca = PCA(n_components=pca_dim, random_state=random_state)
    real_pca = pca.fit_transform(real_np)
    print(f"    PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}")

    # GMM with tied covariance
    print(f"    Fitting GMM with k={n_components}...")
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="tied",
        random_state=random_state,
        n_init=3,
        max_iter=200,
    )
    gmm.fit(real_pca)
    print(f"    GMM converged: {gmm.converged_}, n_iter: {gmm.n_iter_}")

    return pca, gmm


def compute_realism_gmm(gen_feats, real_feats, pca, gmm):
    """
    Compute GMM-based realism using min Mahalanobis to any component.
    """
    gen_np = normalize_features(gen_feats).numpy()
    real_np = normalize_features(real_feats).numpy()

    # Project to PCA space
    gen_pca = pca.transform(gen_np)
    real_pca = pca.transform(real_np)

    # GMM means and precision (tied covariance)
    means = gmm.means_  # (k, pca_dim)
    precision = np.linalg.inv(gmm.covariances_ + 1e-6 * np.eye(gmm.covariances_.shape[0]))

    # Vectorized min Mahalanobis distance
    def min_mahalanobis_batch(X, means, precision):
        # X: (N, D), means: (K, D), precision: (D, D)
        N = X.shape[0]
        K = means.shape[0]

        # Compute (x - mu)^T P (x - mu) for all x and all mu
        # diff: (N, K, D)
        diff = X[:, np.newaxis, :] - means[np.newaxis, :, :]
        # term: (N, K, D)
        term = np.einsum("nkd,de->nke", diff, precision)
        # dists: (N, K)
        dists = np.sum(term * diff, axis=2)
        # min over K
        return dists.min(axis=1)

    # Compute energies
    energies_gen = min_mahalanobis_batch(gen_pca, means, precision)
    energies_real = min_mahalanobis_batch(real_pca, means, precision)

    # Calibrate
    real_E_mean = np.mean(energies_real)
    real_E_std = np.std(energies_real) + 1e-8

    z_scores = zscore(energies_gen, real_E_mean, real_E_std)

    return z_scores


def analyze_model(dataset, model, feature_type="dinov3", pca_dim=64):
    """Analyze score component correlations for a model."""
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

    # 1. Single Gaussian realism
    print("Computing single Gaussian realism...")
    realism_single = compute_realism_single_per_sample(gen_feats, real_feats)

    # 2. Supervised mixture (components = conditions)
    print("Computing supervised mixture realism...")
    realism_supervised, n_supervised = compute_realism_supervised_mixture(
        gen_feats, gen_meta, real_feats, real_meta, condition_keys, dataset, model
    )
    print(f"  Supervised components: {n_supervised}")

    # 3. Unsupervised GMM with k = n_conditions
    n_conditions = len(gen_by_cond)
    print(f"Fitting unsupervised GMM with k={n_conditions} (PCA dim={pca_dim})...")
    pca_k16, gmm_k16 = fit_gmm_with_pca(real_feats, n_components=n_conditions, pca_dim=pca_dim)
    realism_gmm_k16 = compute_realism_gmm(gen_feats, real_feats, pca_k16, gmm_k16)

    # Also try GMM with same k as supervised (for marginal, k=5 vs 16)
    if "marginal" in model and n_supervised != n_conditions:
        print(f"Fitting GMM with k={n_supervised} (same as supervised)...")
        pca_k5, gmm_k5 = fit_gmm_with_pca(real_feats, n_components=n_supervised, pca_dim=pca_dim)
        realism_gmm_k5 = compute_realism_gmm(gen_feats, real_feats, pca_k5, gmm_k5)
    else:
        realism_gmm_k5 = None

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
            "realism_single": float(np.mean(realism_single[idx])),
            "realism_supervised": float(np.mean(realism_supervised[idx])),
            "realism_gmm_k16": float(np.mean(realism_gmm_k16[idx])),
        }
        if realism_gmm_k5 is not None:
            row["realism_gmm_k5"] = float(np.mean(realism_gmm_k5[idx]))
        results.append(row)

    df = pd.DataFrame(results)

    # Print table
    print(f"\nPer-condition realism scores:")
    print("-" * 110)
    cols = ["condition", "is_seen", "delta_kid", "realism_single", "realism_supervised", "realism_gmm_k16"]
    if realism_gmm_k5 is not None:
        cols.append("realism_gmm_k5")

    def fmt(x):
        if pd.isna(x):
            return "   NaN"
        elif isinstance(x, bool):
            return "Y" if x else "N"
        elif isinstance(x, float):
            return f"{x:7.4f}"
        return str(x)

    print(df[cols].to_string(index=False, formatters={c: fmt for c in cols}))

    # Compute correlations with ΔKID
    valid = df[df["delta_kid"].notna()]

    print(f"\n\nCORRELATIONS WITH ΔKID (n={len(valid)} conditions):")
    print("-" * 70)
    print(f"{'Score':<25} {'Spearman ρ':>12} {'p-value':>10} {'Kendall τ':>12}")
    print("-" * 70)

    scores_to_check = [
        ("realism_single", "Single Gaussian"),
        ("realism_supervised", f"Supervised (k={n_supervised})"),
        ("realism_gmm_k16", f"GMM unsup (k={n_conditions})"),
    ]
    if realism_gmm_k5 is not None:
        scores_to_check.append(("realism_gmm_k5", f"GMM unsup (k={n_supervised})"))

    correlations = {}
    for col, label in scores_to_check:
        if col in valid.columns:
            rho, p = spearmanr(valid[col], valid["delta_kid"])
            tau, _ = kendalltau(valid[col], valid["delta_kid"])
            print(f"{label:<25} {rho:>12.4f} {p:>10.4f} {tau:>12.4f}")
            correlations[col] = {"rho": rho, "p": p, "tau": tau}

    return df, correlations, n_supervised, n_conditions


def main():
    print("=" * 80)
    print("SUPERVISED vs UNSUPERVISED MIXTURE REALISM")
    print("=" * 80)
    print("\nComparing:")
    print("  1. Single Gaussian (1 component)")
    print("  2. Supervised mixture (k = n_conditions, means from condition labels)")
    print("  3. Unsupervised GMM (k = n_conditions, means fitted via EM)")
    print("\nUsing PCA (dim=64) for GMM to make it tractable.")
    print()

    # CelebA marginal
    df_marginal, corr_marginal, k_sup_m, k_all_m = analyze_model("celeba", "repa_marginal")

    # CelebA full
    df_full, corr_full, k_sup_f, k_all_f = analyze_model("celeba", "repa_full")

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY: Spearman ρ(realism, ΔKID)")
    print("=" * 80)
    print(f"\n{'Method':<30} {'Marginal':>12} {'Full':>12}")
    print("-" * 55)
    print(f"{'Single Gaussian':<30} {corr_marginal['realism_single']['rho']:>12.4f} {corr_full['realism_single']['rho']:>12.4f}")
    print(f"{'Supervised (condition labels)':<30} {corr_marginal['realism_supervised']['rho']:>12.4f} {corr_full['realism_supervised']['rho']:>12.4f}")
    print(f"{'Unsupervised GMM (k=16)':<30} {corr_marginal['realism_gmm_k16']['rho']:>12.4f} {corr_full['realism_gmm_k16']['rho']:>12.4f}")
    if "realism_gmm_k5" in corr_marginal:
        print(f"{'Unsupervised GMM (k=5)':<30} {corr_marginal['realism_gmm_k5']['rho']:>12.4f} {'N/A':>12}")

    print("\n\nConclusion:")
    print("  If GMM unsupervised ≈ supervised, then condition labels are recoverable.")
    print("  If GMM unsupervised < single, then mixture doesn't help for realism.")

    # Save
    output_dir = Path("outputs/trust_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    df_marginal.to_csv(output_dir / "realism_gmm_comparison_marginal.csv", index=False)
    df_full.to_csv(output_dir / "realism_gmm_comparison_full.csv", index=False)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
