"""
Compare different KID-based metrics per condition.

Computes:
1. ΔKID = KID(gen, real) - KID(real_a, real_b)  [baseline-corrected]
2. rKID = KID(gen_a, gen_b) / KID(gen, real)    [within-gen / gen-real ratio]
3. Raw KID(gen, real)

And checks correlations between them.

Usage:
    PYTHONPATH=src uv run python scripts/compare_kid_metrics.py
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


def compute_kid_metrics_per_condition(
    gen_feats, gen_meta, real_feats, real_meta, condition_keys, n_bootstrap=20
):
    """Compute various KID metrics per condition."""
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

        if n_real < 20 or n_gen < 10:
            results[cond] = {
                "n_gen": n_gen,
                "n_real": n_real,
                "kid_gen_real": np.nan,
                "kid_real_real": np.nan,
                "kid_gen_gen": np.nan,
                "delta_kid": np.nan,
                "r_kid": np.nan,
            }
            continue

        k = min(n_real // 2, n_gen // 2, 200)
        if k < 5:
            results[cond] = {
                "n_gen": n_gen,
                "n_real": n_real,
                "kid_gen_real": np.nan,
                "kid_real_real": np.nan,
                "kid_gen_gen": np.nan,
                "delta_kid": np.nan,
                "r_kid": np.nan,
            }
            continue

        gen_f = gen_feats_np[gen_idx]
        real_f = real_feats_np[real_idx]

        kids_gen_real = []
        kids_real_real = []
        kids_gen_gen = []

        for _ in range(n_bootstrap):
            # KID(gen, real)
            gen_samp = gen_f[rng.choice(n_gen, k, replace=False)]
            real_samp = real_f[rng.choice(n_real, k, replace=False)]
            kid_gr = calculate_kid_same_m(gen_samp, real_samp, use_cosine=True)
            if np.isfinite(kid_gr):
                kids_gen_real.append(kid_gr)

            # KID(real_a, real_b)
            perm_real = rng.permutation(n_real)
            real_a = real_f[perm_real[:k]]
            real_b = real_f[perm_real[k:2*k]]
            kid_rr = calculate_kid_same_m(real_a, real_b, use_cosine=True)
            if np.isfinite(kid_rr):
                kids_real_real.append(kid_rr)

            # KID(gen_a, gen_b)
            perm_gen = rng.permutation(n_gen)
            gen_a = gen_f[perm_gen[:k]]
            gen_b = gen_f[perm_gen[k:2*k]]
            kid_gg = calculate_kid_same_m(gen_a, gen_b, use_cosine=True)
            if np.isfinite(kid_gg):
                kids_gen_gen.append(kid_gg)

        kid_gen_real = np.mean(kids_gen_real) if kids_gen_real else np.nan
        kid_real_real = np.mean(kids_real_real) if kids_real_real else np.nan
        kid_gen_gen = np.mean(kids_gen_gen) if kids_gen_gen else np.nan

        # ΔKID = KID(gen, real) - KID(real, real)
        delta_kid = kid_gen_real - kid_real_real if np.isfinite(kid_gen_real) and np.isfinite(kid_real_real) else np.nan

        # rKID = KID(gen, gen) / KID(gen, real)
        if np.isfinite(kid_gen_gen) and np.isfinite(kid_gen_real) and kid_gen_real > 1e-10:
            r_kid = kid_gen_gen / kid_gen_real
        else:
            r_kid = np.nan

        results[cond] = {
            "n_gen": n_gen,
            "n_real": n_real,
            "kid_gen_real": kid_gen_real,
            "kid_real_real": kid_real_real,
            "kid_gen_gen": kid_gen_gen,
            "delta_kid": delta_kid,
            "r_kid": r_kid,
        }

    return results, gen_by_cond, real_by_cond


def analyze_model(dataset, model, feature_type="dinov3"):
    """Analyze KID metrics for a model."""
    condition_keys = CONDITION_ATTRS[dataset]

    print(f"\n{'='*80}")
    print(f"Model: {dataset}/{model}")
    print(f"{'='*80}")

    # Load features
    gen_feats, gen_meta, real_feats, real_meta = load_features(dataset, model, feature_type)
    print(f"Generated: {gen_feats.shape}, Real: {real_feats.shape}")

    # Compute KID metrics
    print("Computing KID metrics per condition...")
    metrics, gen_by_cond, real_by_cond = compute_kid_metrics_per_condition(
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
    print(f"\nPer-condition KID metrics:")
    print("-" * 120)
    cols = ["condition", "is_seen", "n_gen", "n_real", "kid_real_real", "kid_gen_gen", "kid_gen_real", "delta_kid", "r_kid"]

    def fmt(x):
        if pd.isna(x):
            return "NaN"
        elif isinstance(x, bool):
            return "Y" if x else "N"
        elif isinstance(x, float):
            if abs(x) < 0.0001:
                return f"{x:.2e}"
            return f"{x:.4f}"
        return str(x)

    print(df[cols].to_string(index=False, formatters={c: fmt for c in cols}))

    # Compute correlations
    valid = df[df["delta_kid"].notna() & df["r_kid"].notna()]

    print(f"\n\nCORRELATIONS (n={len(valid)} conditions):")
    print("-" * 60)

    # Correlation between ΔKID and rKID
    rho_delta_r, p_delta_r = spearmanr(valid["delta_kid"], valid["r_kid"])
    tau_delta_r, _ = kendalltau(valid["delta_kid"], valid["r_kid"])
    print(f"ΔKID vs rKID:        Spearman ρ = {rho_delta_r:.4f} (p={p_delta_r:.4f}), Kendall τ = {tau_delta_r:.4f}")

    # Correlation between raw KID(gen,real) and ΔKID
    rho_raw_delta, _ = spearmanr(valid["kid_gen_real"], valid["delta_kid"])
    print(f"KID(g,r) vs ΔKID:    Spearman ρ = {rho_raw_delta:.4f}")

    # Correlation between KID(gen,gen) and ΔKID
    rho_gg_delta, _ = spearmanr(valid["kid_gen_gen"], valid["delta_kid"])
    print(f"KID(g,g) vs ΔKID:    Spearman ρ = {rho_gg_delta:.4f}")

    # Summary stats
    print(f"\nSummary stats:")
    print(f"  ΔKID:  mean={valid['delta_kid'].mean():.4f}, std={valid['delta_kid'].std():.4f}")
    print(f"  rKID:  mean={valid['r_kid'].mean():.4f}, std={valid['r_kid'].std():.4f}")

    # For marginal, split by seen/unseen
    if "marginal" in model:
        seen = valid[valid["is_seen"] == True]
        unseen = valid[valid["is_seen"] == False]

        print(f"\n  SEEN (n={len(seen)}):   ΔKID mean={seen['delta_kid'].mean():.4f}, rKID mean={seen['r_kid'].mean():.4f}")
        print(f"  UNSEEN (n={len(unseen)}): ΔKID mean={unseen['delta_kid'].mean():.4f}, rKID mean={unseen['r_kid'].mean():.4f}")

    return df


def main():
    print("=" * 80)
    print("COMPARING KID METRICS: ΔKID vs rKID")
    print("=" * 80)
    print("\nDefinitions:")
    print("  ΔKID = KID(gen, real) - KID(real, real)   [baseline-corrected gap]")
    print("  rKID = KID(gen, gen) / KID(gen, real)     [within-gen diversity / gap ratio]")

    # CelebA marginal
    df_marginal = analyze_model("celeba", "repa_marginal")

    # CelebA full
    df_full = analyze_model("celeba", "repa_full")

    # Save
    output_dir = Path("outputs/trust_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    df_marginal.to_csv(output_dir / "kid_metrics_marginal.csv", index=False)
    df_full.to_csv(output_dir / "kid_metrics_full.csv", index=False)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
