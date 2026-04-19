"""
Task 3 / Layer 3: Sample-based OOD Detection (Real vs Generated).

Evaluates whether trust scores can distinguish real from generated samples.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

from faithful_cond_gen.eval.trust_eval.config import MARGINAL_SEEN_COMBOS
from faithful_cond_gen.eval.trust_eval.condition_utils import (
    filter_feats_and_meta_by_seen_combos,
)
from faithful_cond_gen.eval.trust_eval.scoring_core import (
    fit_trust_scoring_components,
    score_trust_from_components,
)


def stratified_subsample_real(
    real_feats: torch.Tensor,
    real_meta: Dict,
    condition_keys: List[str],
    max_samples: int,
    seed: int,
) -> np.ndarray:
    """
    Stratified subsample of real samples by condition.

    Returns indices into real_feats that maintain proportional representation
    of each condition.

    Args:
        real_feats: Real features tensor [N, D]
        real_meta: Metadata dict with condition keys
        condition_keys: List of condition attribute names
        max_samples: Maximum number of samples to return
        seed: Random seed for reproducibility

    Returns:
        np.ndarray of indices into real_feats
    """
    n_total = len(real_feats)
    if n_total <= max_samples:
        return np.arange(n_total)

    # Build condition tuples for each sample
    conditions = []
    for i in range(n_total):
        cond = tuple(
            int(
                real_meta[k][i].item()
                if isinstance(real_meta[k][i], torch.Tensor)
                else real_meta[k][i]
            )
            for k in condition_keys
        )
        conditions.append(cond)

    # Group indices by condition
    cond_to_indices = {}
    for i, cond in enumerate(conditions):
        if cond not in cond_to_indices:
            cond_to_indices[cond] = []
        cond_to_indices[cond].append(i)

    # Proportional allocation
    rng = np.random.default_rng(seed)
    selected_indices = []

    for cond, indices in cond_to_indices.items():
        # Allocate proportionally, at least 1 if non-empty
        n_cond = len(indices)
        n_alloc = max(1, int(np.round(n_cond / n_total * max_samples)))
        n_alloc = min(n_alloc, n_cond)  # Don't oversample

        # Sample from this stratum
        sampled = rng.choice(indices, size=n_alloc, replace=False)
        selected_indices.extend(sampled)

    # If we overshot due to rounding, trim
    selected_indices = np.array(selected_indices)
    if len(selected_indices) > max_samples:
        selected_indices = rng.choice(selected_indices, size=max_samples, replace=False)

    return selected_indices


def evaluate_sample_ood_detection(
    trust_results: Dict,
    real_feats: torch.Tensor,
    real_meta: Dict,
    gen_feats: torch.Tensor,
    gen_meta: Dict,
    condition_keys: List[str],
    dataset: str,
    model: str,
    output_dir: Path,
    config_key: str,
    max_real: int = 10000,
    n_resamples: int = 3,
    fit_fraction: float = 0.5,
) -> Dict:
    """
    Task 3: Sample-based OOD detection (real vs generated).

    Uses cross-fitting to avoid calibration bias:
    - Split real samples into fit_set and score_set
    - Fit scoring components on fit_set only
    - Score both score_set (held-out real) and gen samples using fit_set components

    This ensures fair comparison because neither real nor gen samples
    were used to fit the scoring Gaussians.

    Args:
        trust_results: Dict with trust scores and conditions (unused, kept for API compat)
        real_feats: Real features tensor
        real_meta: Real metadata dict
        gen_feats: Generated features tensor
        gen_meta: Generated metadata dict (required for re-scoring gen samples)
        condition_keys: List of condition attribute names
        dataset: Dataset name
        model: Model name
        output_dir: Output directory for plots
        config_key: Configuration key for naming files
        max_real: Maximum real samples to use (stratified subsample if exceeded)
        n_resamples: Number of resamples for CI computation
        fit_fraction: Fraction of real samples used for fitting (rest for scoring)
    """
    # Check if resampling is needed
    n_total_real = len(real_feats)
    do_resample = n_total_real > max_real

    results = {}

    filter_by_seen = "marginal" in model and dataset == "celeba"
    seen_combos = MARGINAL_SEEN_COMBOS if filter_by_seen else None

    # Define splits for gen samples (seen vs unseen for marginal models)
    splits = []
    if filter_by_seen:
        # Need to compute gen conditions for split
        n_gen = len(gen_feats)
        gen_conditions = []
        for i in range(n_gen):
            cond = tuple(
                int(
                    gen_meta[k][i].item()
                    if isinstance(gen_meta[k][i], torch.Tensor)
                    else gen_meta[k][i]
                )
                for k in condition_keys
            )
            gen_conditions.append(cond)
        gen_seen_mask = np.array([c in MARGINAL_SEEN_COMBOS for c in gen_conditions])
        splits.append(("seen", gen_seen_mask))
        splits.append(("unseen", ~gen_seen_mask))
    else:
        splits.append(("all", np.ones(len(gen_feats), dtype=bool)))

    for split_name, gen_mask in splits:
        gen_feats_split = gen_feats[gen_mask]
        # Build gen_meta_split
        gen_meta_split = {}
        for k in condition_keys:
            if isinstance(gen_meta[k], torch.Tensor):
                gen_meta_split[k] = gen_meta[k][gen_mask]
            elif isinstance(gen_meta[k], np.ndarray):
                gen_meta_split[k] = gen_meta[k][gen_mask]
            else:
                gen_meta_split[k] = [
                    gen_meta[k][i] for i, m in enumerate(gen_mask) if m
                ]

        if len(gen_feats_split) == 0:
            continue

        n_gen = len(gen_feats_split)

        # Storage for metrics across resamples
        resample_metrics = {
            "trust": {"auroc": [], "auprc": [], "fpr": [], "tpr": []},
            "realism": {"auroc": [], "auprc": [], "fpr": [], "tpr": []},
            "faithfulness": {"auroc": [], "auprc": [], "fpr": [], "tpr": []},
        }

        actual_resamples = n_resamples if do_resample else 1

        for resample_idx in range(actual_resamples):
            seed = 42 + resample_idx
            rng = np.random.default_rng(seed)

            # Get real sample indices (stratified subsample if needed)
            if do_resample:
                real_indices = stratified_subsample_real(
                    real_feats, real_meta, condition_keys, max_real, seed
                )
            else:
                real_indices = np.arange(n_total_real)

            # Split into fit_set and score_set (cross-fitting)
            n_subset = len(real_indices)
            n_fit = int(n_subset * fit_fraction)
            perm = rng.permutation(n_subset)
            fit_idx = real_indices[perm[:n_fit]]
            score_idx = real_indices[perm[n_fit:]]

            # Build fit metadata
            fit_meta = {}
            for k in condition_keys:
                if isinstance(real_meta[k], torch.Tensor):
                    fit_meta[k] = real_meta[k][fit_idx]
                elif isinstance(real_meta[k], np.ndarray):
                    fit_meta[k] = real_meta[k][fit_idx]
                else:
                    fit_meta[k] = [real_meta[k][i] for i in fit_idx]
            fit_feats = real_feats[fit_idx]

            # Build score metadata (held-out real samples)
            score_meta = {}
            for k in condition_keys:
                if isinstance(real_meta[k], torch.Tensor):
                    score_meta[k] = real_meta[k][score_idx]
                elif isinstance(real_meta[k], np.ndarray):
                    score_meta[k] = real_meta[k][score_idx]
                else:
                    score_meta[k] = [real_meta[k][i] for i in score_idx]
            score_feats = real_feats[score_idx]

            # Filter fit set by seen combos if needed (marginal models)
            if filter_by_seen and seen_combos:
                fit_feats_filtered, fit_meta_filtered = (
                    filter_feats_and_meta_by_seen_combos(
                        fit_feats, fit_meta, condition_keys, seen_combos
                    )
                )
            else:
                fit_feats_filtered, fit_meta_filtered = fit_feats, fit_meta

            # Fit scoring components on fit_set only (LDA-style shared cov for fair margin comparison)
            components = fit_trust_scoring_components(
                fit_feats_filtered,
                fit_meta_filtered,
                condition_keys,
                regularization=1e-5,
                use_shared_cov=True,
            )

            # Score held-out real samples using fitted components
            real_realism_z, real_faithfulness_z, real_trust_z = (
                score_trust_from_components(score_feats, score_meta, components)
            )

            # Score generated samples using the SAME fitted components
            gen_realism_z, gen_faithfulness_z, gen_trust_z = (
                score_trust_from_components(gen_feats_split, gen_meta_split, components)
            )

            n_real = len(real_trust_z)

            # Create binary labels: real=0 (ID), gen=1 (OOD)
            labels = np.concatenate([np.zeros(n_real), np.ones(n_gen)])

            # Score variants
            score_variants = {
                "trust": (
                    np.concatenate([real_trust_z, gen_trust_z]),
                    "Trust Score",
                ),
                "realism": (
                    np.concatenate([real_realism_z, gen_realism_z]),
                    "Realism Only",
                ),
                "faithfulness": (
                    np.concatenate([real_faithfulness_z, gen_faithfulness_z]),
                    "Faithfulness Only",
                ),
            }

            for score_name, (scores, label) in score_variants.items():
                # Filter valid scores
                valid = np.isfinite(scores)
                if valid.sum() < 10:
                    continue

                scores_v = scores[valid]
                labels_v = labels[valid]

                # AUROC
                auroc = roc_auc_score(labels_v, scores_v)

                # AUPRC
                precision, recall, _ = precision_recall_curve(labels_v, scores_v)
                auprc = auc(recall, precision)

                # ROC curve
                fpr, tpr, _ = roc_curve(labels_v, scores_v)

                resample_metrics[score_name]["auroc"].append(auroc)
                resample_metrics[score_name]["auprc"].append(auprc)
                resample_metrics[score_name]["fpr"].append(fpr)
                resample_metrics[score_name]["tpr"].append(tpr)

        # Aggregate results
        split_results = {
            "n_real_total": int(n_total_real),
            "n_real_used": int(min(n_total_real, max_real)),
            "n_gen": int(n_gen),
            "n_resamples": actual_resamples,
        }

        # Create ROC plot with all resamples
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

        for score_name in ["trust", "realism", "faithfulness"]:
            metrics = resample_metrics[score_name]
            if not metrics["auroc"]:
                continue

            aurocs = np.array(metrics["auroc"])
            auprcs = np.array(metrics["auprc"])

            # Mean and CI
            mean_auroc = np.mean(aurocs)
            mean_auprc = np.mean(auprcs)

            if len(aurocs) > 1:
                ci_low_auroc, ci_high_auroc = np.percentile(aurocs, [2.5, 97.5])
                ci_low_auprc, ci_high_auprc = np.percentile(auprcs, [2.5, 97.5])
            else:
                ci_low_auroc = ci_high_auroc = mean_auroc
                ci_low_auprc = ci_high_auprc = mean_auprc

            # Store aggregated metrics
            split_results[f"{score_name}_mean_auroc"] = float(mean_auroc)
            split_results[f"{score_name}_ci_low_auroc"] = float(ci_low_auroc)
            split_results[f"{score_name}_ci_high_auroc"] = float(ci_high_auroc)
            split_results[f"{score_name}_mean_auprc"] = float(mean_auprc)
            split_results[f"{score_name}_ci_low_auprc"] = float(ci_low_auprc)
            split_results[f"{score_name}_ci_high_auprc"] = float(ci_high_auprc)

            # Also store as simple auroc/auprc for backward compatibility
            split_results[f"{score_name}_auroc"] = float(mean_auroc)
            split_results[f"{score_name}_auprc"] = float(mean_auprc)

            # Plot each resample curve with alpha=0.3
            label_map = {
                "trust": "Trust Score",
                "realism": "Realism Only",
                "faithfulness": "Faithfulness Only",
            }
            for i, (fpr, tpr) in enumerate(zip(metrics["fpr"], metrics["tpr"])):
                if i == 0:
                    # First curve gets the label with mean±CI annotation
                    if len(aurocs) > 1:
                        label_str = f"{label_map[score_name]} (AUROC={mean_auroc:.4f} [{ci_low_auroc:.4f}, {ci_high_auroc:.4f}])"
                    else:
                        label_str = f"{label_map[score_name]} (AUROC={mean_auroc:.4f})"
                    ax_roc.plot(
                        fpr, tpr, label=label_str, alpha=0.3 if do_resample else 1.0
                    )
                else:
                    ax_roc.plot(
                        fpr, tpr, alpha=0.3, color=ax_roc.get_lines()[-1].get_color()
                    )

        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        title_suffix = f" ({actual_resamples} resamples)" if do_resample else ""
        ax_roc.set_title(
            f"ROC Curve: Real vs Gen({split_name}) - {config_key}{title_suffix}"
        )
        ax_roc.legend(loc="lower right", fontsize=8)
        ax_roc.grid(alpha=0.3)

        roc_path = (
            output_dir / f"{dataset}_ood_roc_{config_key.replace('/', '_')}_{split_name}.png"
        )
        fig_roc.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close(fig_roc)

        split_results["roc_path"] = str(roc_path)

        # Backward compatibility: n_real
        split_results["n_real"] = split_results["n_real_used"]

        results[split_name] = split_results

    return results
