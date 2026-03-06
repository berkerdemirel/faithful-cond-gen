"""
Layer 2: Failure Detection (OOD, Seen vs Unseen).

Evaluates whether trust scores can detect out-of-distribution conditions/samples.
"""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

from faithful_cond_gen.eval.trust_eval.config import MARGINAL_SEEN_COMBOS, RXRX1_HELDOUT_PAIRS


def evaluate_seen_vs_unseen_detection(
    trust_results: Dict,
    dataset: str,
    model: str,
    output_dir: Path,
    config_key: str,
) -> Dict:
    """
    Sample-level ROC/AUROC between seen vs unseen conditions (marginal models only).

    For CelebA: Labels based on MARGINAL_SEEN_COMBOS (condition tuples).
    For RxRx1: Labels based on RXRX1_HELDOUT_PAIRS (cell_type_id, sirna_id pairs).
    """
    if "marginal" not in model:
        return {"status": "not_applicable"}

    conditions = trust_results["true_conditions"]
    trust_scores = trust_results["trust_updated"]
    realism_scores = trust_results["realism_global_z"]
    faithfulness_scores = trust_results["faithfulness_margin_z"]

    # Build labels: seen=0, unseen=1
    if dataset == "celeba":
        labels = np.array([0 if c in MARGINAL_SEEN_COMBOS else 1 for c in conditions])
    elif dataset == "rxrx1":
        # For RxRx1: condition is (cell_type_id, sirna_id)
        labels = np.array([1 if c in RXRX1_HELDOUT_PAIRS else 0 for c in conditions])
    else:
        return {"status": "not_applicable"}

    n_seen = (labels == 0).sum()
    n_unseen = (labels == 1).sum()

    if n_seen == 0 or n_unseen == 0:
        return {
            "status": "insufficient_samples",
            "n_seen": n_seen,
            "n_unseen": n_unseen,
        }

    results = {
        "n_seen": int(n_seen),
        "n_unseen": int(n_unseen),
    }

    # Score variants
    score_variants = {
        "trust": (trust_scores, "Trust Score"),
        "realism": (realism_scores, "Realism Only"),
        "faithfulness": (faithfulness_scores, "Faithfulness Only"),
    }

    # Create ROC plot
    fig, ax = plt.subplots(figsize=(8, 6))

    for score_name, (scores, label) in score_variants.items():
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

        results[f"{score_name}_auroc"] = float(auroc)
        results[f"{score_name}_auprc"] = float(auprc)

        # ROC curve
        fpr, tpr, _ = roc_curve(labels_v, scores_v)
        ax.plot(fpr, tpr, label=f"{label} (AUROC={auroc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC: Seen vs Unseen Conditions - {config_key}")
    ax.legend()
    ax.grid(alpha=0.3)

    roc_path = output_dir / f"{dataset}_seen_unseen_roc_{config_key.replace('/', '_')}.png"
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    results["roc_path"] = str(roc_path)
    results["status"] = "success"

    return results


def evaluate_failure_detection(
    marginal_results: Dict,
    dataset: str,
) -> Dict:
    """
    Layer 2: Can trust predict which conditions fail?

    For marginal model, label conditions as "bad" if they are unseen (OOD).
    Evaluate if trust score can detect these at the CONDITION level (not sample level).

    For CelebA: Uses MARGINAL_SEEN_COMBOS.
    For RxRx1: Uses RXRX1_HELDOUT_PAIRS (cell_type_id, sirna_id).
    """
    # Aggregate trust scores per condition
    conditions = marginal_results["true_conditions"]
    trust_scores = marginal_results["trust_updated"]

    # Group by condition and compute mean trust per condition
    trust_by_cond = {}
    for i, cond in enumerate(conditions):
        if cond not in trust_by_cond:
            trust_by_cond[cond] = []
        trust_by_cond[cond].append(trust_scores[i])

    # Build condition-level labels and scores
    cond_labels = []
    cond_scores = []
    for cond, scores_list in trust_by_cond.items():
        if dataset == "celeba":
            is_seen = cond in MARGINAL_SEEN_COMBOS
        elif dataset == "rxrx1":
            is_seen = cond not in RXRX1_HELDOUT_PAIRS
        else:
            # Unsupported dataset
            return {"auroc": np.nan, "auprc": np.nan}

        cond_labels.append(0 if is_seen else 1)
        cond_scores.append(np.mean(scores_list))

    cond_labels = np.array(cond_labels)
    cond_scores = np.array(cond_scores)

    n_seen = (cond_labels == 0).sum()
    n_unseen = (cond_labels == 1).sum()

    if n_seen == 0 or n_unseen == 0:
        return {
            "auroc": np.nan,
            "auprc": np.nan,
            "n_seen_conds": n_seen,
            "n_unseen_conds": n_unseen,
        }

    # AUROC: higher score = more likely OOD
    auroc = roc_auc_score(cond_labels, cond_scores)

    # AUPRC
    precision, recall, _ = precision_recall_curve(cond_labels, cond_scores)
    auprc = auc(recall, precision)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "n_seen_conds": int(n_seen),
        "n_unseen_conds": int(n_unseen),
        "n_total_conds": len(trust_by_cond),
    }
