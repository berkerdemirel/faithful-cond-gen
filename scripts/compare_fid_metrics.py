"""
Compare FID-based metrics per condition.

Computes:
1. ΔKID = KID(gen, real) - KID(real, real)   [baseline-corrected, additive]
2. rFID = FID(gen, real) / FID(real, real)   [baseline-corrected, multiplicative]
3. FID(gen, real)

And checks correlations between ΔKID and rFID.

FID = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2*(Σ1*Σ2)^0.5)

Usage:
    PYTHONPATH=src uv run python scripts/compare_fid_metrics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import linalg
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


def compute_fid(feats1: np.ndarray, feats2: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute Fréchet Inception Distance between two sets of features.

    FID = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2*(Σ1*Σ2)^0.5)

    Args:
        feats1: Features from first distribution (N1, D)
        feats2: Features from second distribution (N2, D)
        eps: Small constant for numerical stability

    Returns:
        FID score (lower is better, 0 means identical distributions)
    """
    mu1 = np.mean(feats1, axis=0)
    mu2 = np.mean(feats2, axis=0)

    sigma1 = np.cov(feats1, rowvar=False)
    sigma2 = np.cov(feats2, rowvar=False)

    # Ensure covariance matrices are 2D
    if sigma1.ndim == 0:
        sigma1 = sigma1.reshape(1, 1)
    if sigma2.ndim == 0:
        sigma2 = sigma2.reshape(1, 1)

    # Mean difference term
    diff = mu1 - mu2
    mean_term = np.dot(diff, diff)

    # Matrix square root term
    # Compute sqrt(Σ1 @ Σ2) using eigendecomposition for stability
    try:
        # Use scipy's sqrtm for matrix square root
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

        # Handle numerical issues - sqrtm can return complex values
        if np.iscomplexobj(covmean):
            if not np.allclose(np.imag(covmean), 0, atol=1e-3):
                # If imaginary part is significant, something went wrong
                return np.nan
            covmean = np.real(covmean)

        trace_term = np.trace(sigma1 + sigma2 - 2 * covmean)

        # Handle numerical issues that can make trace negative
        if trace_term < 0:
            if trace_term > -1e-6:  # Small negative due to numerical error
                trace_term = 0
            else:
                return np.nan

    except (ValueError, np.linalg.LinAlgError):
        return np.nan

    fid = mean_term + trace_term
    return float(fid)


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


def compute_metrics_per_condition(
    gen_feats, gen_meta, real_feats, real_meta, condition_keys, n_bootstrap=10
):
    """Compute FID and KID metrics per condition."""
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

    results = {}
    rng = np.random.default_rng(42)

    for cond in gen_by_cond:
        gen_idx = gen_by_cond[cond]
        real_idx = real_by_cond.get(cond, [])

        n_gen = len(gen_idx)
        n_real = len(real_idx)

        # Need enough samples for covariance estimation
        min_samples = 50
        if n_real < min_samples or n_gen < min_samples:
            results[cond] = {
                "n_gen": n_gen,
                "n_real": n_real,
                "fid_gen_real": np.nan,
                "fid_real_real": np.nan,
                "r_fid": np.nan,
                "delta_kid": np.nan,
            }
            continue

        gen_f = gen_feats_np[gen_idx]
        real_f = real_feats_np[real_idx]

        # FID(gen, real) - use all samples
        fid_gen_real = compute_fid(gen_f, real_f)

        # FID(real_a, real_b) - split real into two halves, bootstrap
        fid_real_real_samples = []
        for _ in range(n_bootstrap):
            perm = rng.permutation(n_real)
            half = n_real // 2
            real_a = real_f[perm[:half]]
            real_b = real_f[perm[half:2*half]]
            fid_rr = compute_fid(real_a, real_b)
            if np.isfinite(fid_rr):
                fid_real_real_samples.append(fid_rr)

        fid_real_real = np.mean(fid_real_real_samples) if fid_real_real_samples else np.nan

        # rFID = FID(gen, real) / FID(real, real)
        if np.isfinite(fid_real_real) and np.isfinite(fid_gen_real) and fid_real_real > 1e-10:
            r_fid = fid_gen_real / fid_real_real
        else:
            r_fid = np.nan

        # Also compute ΔKID for correlation
        k = min(n_real // 2, n_gen, 200)
        kids_delta = []
        for _ in range(n_bootstrap):
            perm_real = rng.permutation(n_real)
            real_a = real_f[perm_real[:k]]
            real_b = real_f[perm_real[k:2*k]]
            gen_samp = gen_f[rng.choice(n_gen, k, replace=False)]

            kid_rr = calculate_kid_same_m(real_a, real_b, use_cosine=True)
            kid_gr = calculate_kid_same_m(gen_samp, real_a, use_cosine=True)
            if np.isfinite(kid_rr) and np.isfinite(kid_gr):
                kids_delta.append(kid_gr - kid_rr)

        delta_kid = np.mean(kids_delta) if kids_delta else np.nan

        results[cond] = {
            "n_gen": n_gen,
            "n_real": n_real,
            "fid_gen_real": fid_gen_real,
            "fid_real_real": fid_real_real,
            "r_fid": r_fid,
            "delta_kid": delta_kid,
        }

    return results, gen_by_cond, real_by_cond


def analyze_model(dataset, model, feature_type="dinov3"):
    """Analyze FID metrics for a model."""
    condition_keys = CONDITION_ATTRS[dataset]

    print(f"\n{'='*80}")
    print(f"Model: {dataset}/{model}")
    print(f"{'='*80}")

    # Load features
    gen_feats, gen_meta, real_feats, real_meta = load_features(dataset, model, feature_type)
    print(f"Generated: {gen_feats.shape}, Real: {real_feats.shape}")

    # Compute metrics
    print("Computing FID and KID metrics per condition...")
    metrics, gen_by_cond, real_by_cond = compute_metrics_per_condition(
        gen_feats, gen_meta, real_feats, real_meta, condition_keys
    )

    # Build DataFrame
    rows = []
    for cond in sorted(metrics.keys()):
        m = metrics[cond]
        is_seen = cond in MARGINAL_SEEN_COMBOS if "marginal" in model else True
        rows.append({
            "condition": str(cond),
            "is_seen": is_seen,
            **m,
        })

    df = pd.DataFrame(rows)

    # Print table
    print(f"\nPer-condition metrics:")
    print("-" * 110)
    cols = ["condition", "is_seen", "n_gen", "n_real", "fid_real_real", "fid_gen_real", "r_fid", "delta_kid"]

    def fmt(x):
        if pd.isna(x):
            return "NaN"
        elif isinstance(x, bool):
            return "Y" if x else "N"
        elif isinstance(x, float):
            if abs(x) < 0.001:
                return f"{x:.2e}"
            return f"{x:.4f}"
        return str(x)

    print(df[cols].to_string(index=False, formatters={c: fmt for c in cols}))

    # Compute correlations
    valid = df[df["delta_kid"].notna() & df["r_fid"].notna()]

    print(f"\n\nCORRELATIONS (n={len(valid)} conditions):")
    print("-" * 60)

    if len(valid) >= 3:
        # Correlation between ΔKID and rFID
        rho_delta_rfid, p_val = spearmanr(valid["delta_kid"], valid["r_fid"])
        tau_delta_rfid, _ = kendalltau(valid["delta_kid"], valid["r_fid"])
        print(f"ΔKID vs rFID:        Spearman ρ = {rho_delta_rfid:.4f} (p={p_val:.4f}), Kendall τ = {tau_delta_rfid:.4f}")

        # Correlation between ΔKID and FID(gen, real)
        rho_delta_fid, _ = spearmanr(valid["delta_kid"], valid["fid_gen_real"])
        print(f"ΔKID vs FID(g,r):    Spearman ρ = {rho_delta_fid:.4f}")

        # Correlation between rFID and FID(gen, real)
        rho_rfid_fid, _ = spearmanr(valid["r_fid"], valid["fid_gen_real"])
        print(f"rFID vs FID(g,r):    Spearman ρ = {rho_rfid_fid:.4f}")

    # Summary stats
    print(f"\nSummary stats:")
    print(f"  ΔKID:        mean={valid['delta_kid'].mean():.4f}, std={valid['delta_kid'].std():.4f}")
    print(f"  rFID:        mean={valid['r_fid'].mean():.4f}, std={valid['r_fid'].std():.4f}")
    print(f"  FID(g,r):    mean={valid['fid_gen_real'].mean():.4f}, std={valid['fid_gen_real'].std():.4f}")
    print(f"  FID(r,r):    mean={valid['fid_real_real'].mean():.4f}, std={valid['fid_real_real'].std():.4f}")

    # For marginal, split by seen/unseen
    if "marginal" in model:
        seen = valid[valid["is_seen"] == True]
        unseen = valid[valid["is_seen"] == False]

        if len(seen) > 0:
            print(f"\n  SEEN (n={len(seen)}):   ΔKID={seen['delta_kid'].mean():.4f}, rFID={seen['r_fid'].mean():.4f}, FID(g,r)={seen['fid_gen_real'].mean():.4f}")
        if len(unseen) > 0:
            print(f"  UNSEEN (n={len(unseen)}): ΔKID={unseen['delta_kid'].mean():.4f}, rFID={unseen['r_fid'].mean():.4f}, FID(g,r)={unseen['fid_gen_real'].mean():.4f}")

    return df


def main():
    print("=" * 80)
    print("COMPARING ΔKID vs rFID")
    print("=" * 80)
    print("\nDefinitions:")
    print("  ΔKID = KID(gen, real) - KID(real, real)")
    print("  rFID = FID(gen, gen) / FID(gen, real)")
    print("  FID  = ||μ1-μ2||² + Tr(Σ1 + Σ2 - 2*sqrt(Σ1*Σ2))")

    # CelebA marginal
    df_marginal = analyze_model("celeba", "repa_marginal")

    # CelebA full
    df_full = analyze_model("celeba", "repa_full")

    # Save
    output_dir = Path("outputs/trust_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    df_marginal.to_csv(output_dir / "fid_metrics_marginal.csv", index=False)
    df_full.to_csv(output_dir / "fid_metrics_full.csv", index=False)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
