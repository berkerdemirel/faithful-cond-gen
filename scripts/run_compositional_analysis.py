"""
Compositional Generalization Analysis for Faithfulness Scoring.

Evaluates scorers on their ability to:
1. Detect unseen (OOD) attribute combinations via AUROC/FPR95
2. Rank conditions by faithfulness (correlation with ground truth FID/KID)

Usage:
    uv run python scripts/run_compositional_analysis.py --dataset celeba
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from faithful_cond_gen.eval.scoring.mahalanobis import MahalanobisScore
from faithful_cond_gen.eval.scoring.marginal_mahalanobis import MarginalMahalanobisScore
from faithful_cond_gen.eval.scoring.relative_mahalanobis import RelativeMahalanobisScore
from faithful_cond_gen.eval.scoring.cosine import CosineScore
from faithful_cond_gen.eval.scoring.marginal_cosine import MarginalCosineScore
from faithful_cond_gen.eval.scoring.marginal_linear_probe import MarginalLinearProbeScore
from faithful_cond_gen.eval.scoring.knn import KNNScore
from faithful_cond_gen.eval.scoring.conditional_knn import ConditionalKNNScore

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================

FEATURE_CACHE_ROOT = Path("feature_cache")
OUTPUT_DIR = Path("outputs/compositional_analysis")

# Attributes used for compositional analysis (4 attrs for celeba)
COMP_ATTRS = {
    "celeba": ["Male", "Smiling", "Blond_Hair", "Eyeglasses"],
}

# Marginal model only saw single-attribute combos
# These are the SEEN combinations for marginal model
MARGINAL_SEEN_COMBOS = {
    (0, 0, 0, 0),  # all zeros
    (1, 0, 0, 0),  # Male only
    (0, 1, 0, 0),  # Smiling only
    (0, 0, 1, 0),  # Blond_Hair only
    (0, 0, 0, 1),  # Eyeglasses only
}

SCORERS = {
    "mahalanobis": lambda: MahalanobisScore(regularization=1e-5, use_shrinkage=True),
    "marginal_mahalanobis": lambda: MarginalMahalanobisScore(regularization=1e-5, use_shrinkage=True),
    "relative_mahalanobis": lambda: RelativeMahalanobisScore(regularization=1e-5, use_shrinkage=True),
    "cosine": lambda: CosineScore(use_softmax=False),
    "marginal_cosine": lambda: MarginalCosineScore(),
    "marginal_linear_probe": lambda: MarginalLinearProbeScore(soft_mode=True),
    "knn": lambda: KNNScore(k=5),
    "conditional_knn": lambda: ConditionalKNNScore(k=5),
}


# ============================================================================
# Helper functions
# ============================================================================

def load_features(path: Path) -> Tuple[torch.Tensor, Dict]:
    """Load features and metadata from a .pt file."""
    data = torch.load(path, map_location="cpu")
    features = data["features"]
    metadata = data.get("metadata", {})
    return features, metadata


def get_combo(metadata: Dict, keys: List[str], idx: int) -> tuple:
    """Get attribute combo tuple for a sample."""
    return tuple(int(metadata[k][idx].item() if isinstance(metadata[k][idx], torch.Tensor)
                     else metadata[k][idx]) for k in keys)


def assign_comp_labels(metadata: Dict, keys: List[str], model: str) -> np.ndarray:
    """Assign seen/unseen labels based on model type.

    - fullmodel: ALL combos are seen (trained on joint distribution)
    - marginalmodel: Only single-attribute combos are seen
    """
    n = len(metadata[keys[0]])
    labels = []

    for i in range(n):
        combo = get_combo(metadata, keys, i)

        if model == "marginalmodel":
            # Marginal model only saw single-attribute combos
            if combo in MARGINAL_SEEN_COMBOS:
                labels.append("seen")
            else:
                labels.append("unseen")
        else:
            # Full model saw all combos
            labels.append("seen")

    return np.array(labels)


def count_ones(combo: tuple) -> int:
    """Count number of active attributes in a combo."""
    return sum(combo)


def compute_fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, tpr_threshold: float = 0.95) -> float:
    """Compute FPR at given TPR threshold (e.g., FPR@95)."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    # Find FPR at TPR >= threshold
    idx = np.searchsorted(tpr, tpr_threshold)
    if idx >= len(fpr):
        return fpr[-1]
    return fpr[idx]


def calculate_kid_same_m(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Unbiased MMD^2 estimate with polynomial kernel: k(x,y)=(x·y/d + 1)^3
    Assumes X and Y have same number of samples m>=2.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"KID expects equal sample sizes, got {X.shape[0]} vs {Y.shape[0]}")
    m = X.shape[0]
    if m < 2:
        return np.nan

    dim = X.shape[1]

    Kxx = (X @ X.T / dim + 1.0) ** 3
    Kyy = (Y @ Y.T / dim + 1.0) ** 3
    Kxy = (X @ Y.T / dim + 1.0) ** 3

    kxx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    kyy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    kxy = Kxy.sum() / (m * m)

    return float(kxx + kyy - 2.0 * kxy)


def bootstrap_kid_metrics(
    feats_real_all: np.ndarray,
    feats_gen_all: np.ndarray,
    seed: int,
    n_bootstrap: int = 10,
    k_cap: int = 1000,
    min_real_pool: int = 20,
    min_k: int = 5,
) -> Dict[str, float]:
    """
    Bootstrap delta KID: kid_gen - kid_base.

    For each bootstrap:
      - split reals into Real_A (k) and Real_B (k)
      - sample Gen (k)
      - base_kid = KID(Real_A, Real_B)
      - gen_kid = KID(Real_A, Gen)
      - delta = gen_kid - base_kid
    """
    n_real = feats_real_all.shape[0]
    n_gen = feats_gen_all.shape[0]

    out = {"kid_delta_mean": np.nan, "kid_delta_std": np.nan, "k_used": np.nan}

    if n_real < min_real_pool:
        return out

    k = min(n_real // 2, n_gen, k_cap)
    if k < min_k:
        return out

    rng = np.random.default_rng(seed)
    deltas = []

    for _ in range(n_bootstrap):
        perm = rng.permutation(n_real)
        idx_a = perm[:k]
        idx_b = perm[k:2*k]

        real_a = feats_real_all[idx_a]
        real_b = feats_real_all[idx_b]

        idx_g = rng.choice(n_gen, size=k, replace=False)
        gen_samp = feats_gen_all[idx_g]

        base = calculate_kid_same_m(real_a, real_b)
        gen = calculate_kid_same_m(real_a, gen_samp)

        if np.isfinite(base) and np.isfinite(gen):
            deltas.append(gen - base)

    if len(deltas) == 0:
        return out

    delta_arr = np.asarray(deltas, dtype=np.float64)
    out["kid_delta_mean"] = float(delta_arr.mean())
    out["kid_delta_std"] = float(delta_arr.std(ddof=1)) if len(delta_arr) > 1 else 0.0
    out["k_used"] = float(k)

    return out


def compute_per_condition_delta_kid(
    gen_feats: torch.Tensor, gen_meta: Dict,
    real_feats: torch.Tensor, real_meta: Dict,
    keys: List[str],
    n_bootstrap: int = 10,
    seed: int = 42,
) -> Dict[tuple, float]:
    """Compute delta KID per condition using bootstrap."""
    # Group by condition
    gen_groups = {}
    for i in range(len(gen_feats)):
        combo = get_combo(gen_meta, keys, i)
        if combo not in gen_groups:
            gen_groups[combo] = []
        gen_groups[combo].append(i)

    real_groups = {}
    for i in range(len(real_feats)):
        combo = get_combo(real_meta, keys, i)
        if combo not in real_groups:
            real_groups[combo] = []
        real_groups[combo].append(i)

    delta_kids = {}
    for combo in gen_groups:
        gen_idx = gen_groups[combo]
        real_idx = real_groups.get(combo, [])

        if len(real_idx) < 20 or len(gen_idx) < 5:
            delta_kids[combo] = np.nan
            continue

        gen_f = gen_feats[gen_idx].numpy()
        real_f = real_feats[real_idx].numpy()

        # Use combo hash for reproducible seed
        combo_seed = (seed + hash(combo)) & 0xFFFFFFFF
        metrics = bootstrap_kid_metrics(real_f, gen_f, combo_seed, n_bootstrap)
        delta_kids[combo] = metrics["kid_delta_mean"]

    return delta_kids


# ============================================================================
# Main analysis
# ============================================================================

def analyze_single_config(
    dataset: str,
    model: str,
    encoder: str,
    scorer_name: str,
    train_feats: torch.Tensor,
    train_meta: Dict,
    gen_feats: torch.Tensor,
    gen_meta: Dict,
    comp_keys: List[str],
    per_cond_delta_kid: Dict[tuple, float],
) -> Dict:
    """Run analysis for a single scorer/model/encoder config."""

    result = {
        "dataset": dataset,
        "model": model,
        "encoder": encoder,
        "scorer": scorer_name,
    }

    try:
        # 1. Fit scorer and get scores
        scorer = SCORERS[scorer_name]()
        meta_filtered = {k: train_meta[k] for k in comp_keys if k in train_meta}
        scorer.fit(train_feats, meta_filtered)

        gen_meta_filtered = {k: gen_meta[k] for k in comp_keys if k in gen_meta}
        scores = scorer.score(gen_feats, gen_meta_filtered)
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        # 2. Assign seen/unseen labels (based on model type)
        labels = assign_comp_labels(gen_meta, comp_keys, model)
        is_unseen = (labels == "unseen").astype(int)

        n_seen = (labels == "seen").sum()
        n_unseen = (labels == "unseen").sum()

        result["n_seen"] = int(n_seen)
        result["n_unseen"] = int(n_unseen)

        # 3. AUROC and FPR@95 for OOD detection
        # Higher score = more likely OOD (unseen)
        if n_unseen > 0 and n_seen > 0:
            auroc = roc_auc_score(is_unseen, scores)
            fpr95 = compute_fpr_at_tpr(is_unseen, scores, 0.95)
            result["auroc"] = auroc
            result["fpr95"] = fpr95
        else:
            result["auroc"] = np.nan
            result["fpr95"] = np.nan

        # 4. Per-condition mean scores
        cond_scores = {}
        for i in range(len(scores)):
            combo = get_combo(gen_meta, comp_keys, i)
            if combo not in cond_scores:
                cond_scores[combo] = []
            cond_scores[combo].append(scores[i])

        mean_scores = {k: np.mean(v) for k, v in cond_scores.items()}
        result["cond_mean_scores"] = mean_scores

        # 5. Ranking correlation with per-condition delta KID
        # Only use conditions that exist in both
        common_conds = set(mean_scores.keys()) & set(per_cond_delta_kid.keys())
        if len(common_conds) >= 3:
            score_vals = [mean_scores[c] for c in common_conds]
            kid_vals = [per_cond_delta_kid[c] for c in common_conds]

            # Filter out NaN
            valid = [(s, k) for s, k in zip(score_vals, kid_vals) if not np.isnan(k)]
            if len(valid) >= 3:
                sv, kv = zip(*valid)
                spearman_rho, spearman_p = spearmanr(sv, kv)
                kendall_tau, kendall_p = kendalltau(sv, kv)
                result["spearman_rho"] = spearman_rho
                result["spearman_p"] = spearman_p
                result["kendall_tau"] = kendall_tau
                result["kendall_p"] = kendall_p
            else:
                result["spearman_rho"] = np.nan
                result["kendall_tau"] = np.nan
        else:
            result["spearman_rho"] = np.nan
            result["kendall_tau"] = np.nan

        # 6. Score statistics
        seen_scores = [scores[i] for i in range(len(scores)) if labels[i] == "seen"]
        unseen_scores = [scores[i] for i in range(len(scores)) if labels[i] == "unseen"]

        result["mean_score_seen"] = float(np.mean(seen_scores)) if seen_scores else np.nan
        result["mean_score_unseen"] = float(np.mean(unseen_scores)) if unseen_scores else np.nan
        result["delta_score"] = result["mean_score_unseen"] - result["mean_score_seen"] if unseen_scores and seen_scores else np.nan

        # 7. Per number-of-active-attrs breakdown (for compositional analysis)
        scores_by_n_attrs = {}
        for i in range(len(scores)):
            combo = get_combo(gen_meta, comp_keys, i)
            n = count_ones(combo)
            if n not in scores_by_n_attrs:
                scores_by_n_attrs[n] = []
            scores_by_n_attrs[n].append(scores[i])

        result["mean_by_n_attrs"] = {n: float(np.mean(v)) for n, v in scores_by_n_attrs.items()}

        result["status"] = "ok"

    except Exception as e:
        result["status"] = f"error: {str(e)[:80]}"
        result["auroc"] = np.nan
        result["fpr95"] = np.nan
        result["spearman_rho"] = np.nan
        result["kendall_tau"] = np.nan

    return result


def run_full_analysis(dataset: str) -> pd.DataFrame:
    """Run full compositional analysis for a dataset."""

    results = []
    comp_keys = COMP_ATTRS.get(dataset, [])

    if not comp_keys:
        print(f"No compositional attributes defined for {dataset}")
        return pd.DataFrame()

    # Discover available models and encoders
    gen_root = FEATURE_CACHE_ROOT / "generated_samples" / dataset
    if not gen_root.exists():
        print(f"No generated samples for {dataset}")
        return pd.DataFrame()

    for model_dir in gen_root.iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        for encoder_dir in model_dir.iterdir():
            if not encoder_dir.is_dir():
                continue
            encoder = encoder_dir.name

            # Load features
            gen_files = list(encoder_dir.glob("*features.pt"))
            if not gen_files:
                continue

            real_path = FEATURE_CACHE_ROOT / "real_samples" / dataset / encoder / "train_features.pt"
            if not real_path.exists():
                continue

            print(f"\n=== {model} / {encoder} ===")

            train_feats, train_meta = load_features(real_path)
            gen_feats, gen_meta = load_features(gen_files[0])

            # Compute per-condition delta KID (ground truth for ranking)
            print("  Computing per-condition delta KID...")
            per_cond_delta_kid = compute_per_condition_delta_kid(
                gen_feats, gen_meta, train_feats, train_meta, comp_keys,
                n_bootstrap=10, seed=42
            )

            # Run all scorers
            for scorer_name in tqdm(SCORERS, desc="Scorers"):
                result = analyze_single_config(
                    dataset, model, encoder, scorer_name,
                    train_feats, train_meta,
                    gen_feats, gen_meta,
                    comp_keys, per_cond_delta_kid
                )
                results.append(result)

    return pd.DataFrame(results)


def create_summary_report(df: pd.DataFrame, output_dir: Path):
    """Create summary markdown report."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results - remove dict columns for CSV
    df_csv = df.drop(columns=["cond_mean_scores", "mean_by_n_attrs"], errors="ignore")
    df_csv.to_csv(output_dir / "compositional_results.csv", index=False)

    # Create markdown report
    report = []
    report.append("# Compositional Generalization Analysis\n")

    report.append("## Experimental Setup\n")
    report.append("- **fullmodel**: Trained on ALL attribute combinations (joint conditioning)")
    report.append("- **marginalmodel**: Trained only on single-attribute combos (0000, 1000, 0100, 0010, 0001)")
    report.append("- For marginalmodel, any multi-attribute combo is **compositionally OOD**\n")

    report.append("## Metrics\n")
    report.append("- **AUROC**: OOD detection (seen vs unseen) - higher = better detection")
    report.append("- **FPR@95**: False positive rate at 95% TPR - lower = fewer false alarms")
    report.append("- **Delta Score**: mean(unseen) - mean(seen) - positive = unseen flagged correctly")
    report.append("- **Spearman ρ (vs ΔKID)**: Rank correlation with per-condition delta KID")
    report.append("  - Higher ρ = scorer rankings align with true quality (delta KID)\n")

    # Filter successful runs
    df_ok = df[df["status"] == "ok"].copy()

    if len(df_ok) == 0:
        report.append("\n**No successful runs.**\n")
    else:
        df_marginal = df_ok[df_ok["model"] == "marginalmodel"]
        df_full = df_ok[df_ok["model"] == "fullmodel"]

        # ===== OOD Detection (marginalmodel only) =====
        if len(df_marginal) > 0 and df_marginal["n_unseen"].sum() > 0:
            report.append("\n---\n## OOD Detection (marginalmodel)\n")
            report.append("*Detecting unseen multi-attribute combinations*\n")

            report.append("\n### AUROC by Scorer × Encoder\n")
            pivot = df_marginal.pivot_table(index="scorer", columns="encoder", values="auroc", aggfunc="mean")
            report.append(pivot.round(4).to_markdown())

            report.append("\n\n### FPR@95 by Scorer × Encoder\n")
            pivot = df_marginal.pivot_table(index="scorer", columns="encoder", values="fpr95", aggfunc="mean")
            report.append(pivot.round(4).to_markdown())

            report.append("\n\n### Score Delta (unseen - seen)\n")
            pivot = df_marginal.pivot_table(index="scorer", columns="encoder", values="delta_score", aggfunc="mean")
            report.append(pivot.round(4).to_markdown())

        # ===== Ranking Correlation (both models) =====
        report.append("\n\n---\n## Ranking Correlation with Δ KID\n")
        report.append("*Spearman ρ between per-condition mean score and delta KID*\n")

        for model in df_ok["model"].unique():
            df_model = df_ok[df_ok["model"] == model]
            report.append(f"\n### {model}\n")
            pivot = df_model.pivot_table(index="scorer", columns="encoder", values="spearman_rho", aggfunc="mean")
            report.append(pivot.round(4).to_markdown())
            report.append("\n")

        # ===== Best Scorers =====
        report.append("\n---\n## Best Scorers\n")

        if len(df_marginal) > 0:
            report.append("\n### Best for OOD Detection (by AUROC)\n")
            df_valid = df_marginal[df_marginal["auroc"].notna()]
            if len(df_valid) > 0:
                best = df_valid.loc[df_valid.groupby("encoder")["auroc"].idxmax()]
                report.append(best[["encoder", "scorer", "auroc", "fpr95", "delta_score"]].to_markdown(index=False))

        report.append("\n\n### Best for Ranking Correlation (by Spearman ρ)\n")
        for model in df_ok["model"].unique():
            df_model = df_ok[df_ok["model"] == model]
            df_valid = df_model[df_model["spearman_rho"].notna()]
            if len(df_valid) > 0:
                report.append(f"\n**{model}**\n")
                best = df_valid.loc[df_valid.groupby("encoder")["spearman_rho"].idxmax()]
                report.append(best[["encoder", "scorer", "spearman_rho", "kendall_tau"]].to_markdown(index=False))

        # ===== Summary by Scorer =====
        report.append("\n\n---\n## Summary: Average Metrics by Scorer\n")

        if len(df_marginal) > 0:
            report.append("\n### marginalmodel\n")
            cols = ["auroc", "fpr95", "delta_score", "spearman_rho"]
            cols = [c for c in cols if c in df_marginal.columns]
            scorer_avg = df_marginal.groupby("scorer")[cols].mean().sort_values("auroc", ascending=False)
            report.append(scorer_avg.round(4).to_markdown())

        if len(df_full) > 0:
            report.append("\n\n### fullmodel\n")
            cols = ["spearman_rho", "kendall_tau"]
            cols = [c for c in cols if c in df_full.columns]
            scorer_avg = df_full.groupby("scorer")[cols].mean().sort_values("spearman_rho", ascending=False)
            report.append(scorer_avg.round(4).to_markdown())

        # ===== Key Findings =====
        report.append("\n\n---\n## Key Findings\n")

        if len(df_marginal) > 0:
            # Best overall for OOD
            best_ood = df_marginal.groupby("scorer")["auroc"].mean().idxmax()
            best_ood_auroc = df_marginal.groupby("scorer")["auroc"].mean().max()
            report.append(f"1. **Best OOD detector**: {best_ood} (AUROC={best_ood_auroc:.4f})\n")

        # Best for ranking
        best_rank = df_ok.groupby("scorer")["spearman_rho"].mean().idxmax()
        best_rank_rho = df_ok.groupby("scorer")["spearman_rho"].mean().max()
        report.append(f"2. **Best for ranking**: {best_rank} (ρ={best_rank_rho:.4f})\n")

        # Encoder comparison
        report.append("\n### Encoder Comparison (avg across scorers)\n")
        enc_avg = df_ok.groupby("encoder")[["spearman_rho"]].mean().sort_values("spearman_rho", ascending=False)
        report.append(enc_avg.round(4).to_markdown())

    # Write report
    with open(output_dir / "COMPOSITIONAL_ANALYSIS.md", "w") as f:
        f.write("\n".join(report))

    print(f"\nReport saved to {output_dir / 'COMPOSITIONAL_ANALYSIS.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="celeba")
    parser.add_argument("--output-dir", type=str, default="outputs/compositional_analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("="*60)
    print("COMPOSITIONAL GENERALIZATION ANALYSIS")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Comp attributes: {COMP_ATTRS.get(args.dataset, [])}")

    df = run_full_analysis(args.dataset)

    if len(df) > 0:
        create_summary_report(df, output_dir)
        torch.save({"results": df.to_dict()}, output_dir / "results.pt")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
