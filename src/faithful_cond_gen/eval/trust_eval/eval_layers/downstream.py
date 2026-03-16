"""
Task 5: Downstream Bin-Selection Evaluation.

Classification accuracy by trust-score bin.
Also includes RxRx1-specific downstream tasks:
- Cell type classification (4-way)
- Controlled perturbation classification
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from faithful_cond_gen.eval.trust_eval.condition_utils import (
    build_condition_class_map,
    bin_samples_within_conditioning,
)
from faithful_cond_gen.eval.trust_eval.config import RXRX1_HELDOUT_PAIRS


def evaluate_downstream_bin_selection_from_scores(
    trust_results: Dict,
    gen_feats_downstream: torch.Tensor,
    real_feats_downstream: torch.Tensor,
    real_meta: Dict,
    condition_keys: List[str],
    model_name: str,
    scoring_feature_type: str,
    downstream_feature_type: str,
    output_dir: Path,
    n_bins: int = 10,
    n_seeds: int = 5,
    seed: int = 0,
    dataset: str = "",
) -> pd.DataFrame:
    """
    Task 5: Downstream bin-selection evaluation.

    Tests whether trust-based scoring helps select better samples for downstream
    classification. Key insight: scoring feature space may differ from downstream
    feature space where classifiers are trained.

    Args:
        trust_results: Dict with scores (trust_updated, realism_global_z, faithfulness_margin_z)
                      and true_conditions
        gen_feats_downstream: Generated features in downstream feature space (N, D)
        real_feats_downstream: Real features in downstream feature space (M, D)
        real_meta: Real metadata dict with condition keys
        condition_keys: List of condition attribute names
        model_name: Model name for output naming
        scoring_feature_type: Feature type used for scoring (e.g., "aligned_mean")
        downstream_feature_type: Feature type for downstream task (e.g., "dinov3")
        output_dir: Output directory for CSV and plots
        n_bins: Number of bins (default 10 = deciles)
        n_seeds: Number of seeds for classifier training
        seed: Base random seed

    Returns:
        DataFrame with results per (ranking_mode, bin_idx, seed)
    """
    from sklearn.linear_model import LogisticRegression

    # Extract scores
    trust_scores = trust_results["trust_updated"]
    realism_scores = trust_results["realism_global_z"]
    faithfulness_scores = trust_results["faithfulness_margin_z"]
    true_conditions = trust_results["true_conditions"]

    n_gen = len(trust_scores)
    n_real = len(real_feats_downstream)

    # Sanity checks
    assert len(gen_feats_downstream) == n_gen, (
        f"Feature count mismatch: gen_feats_downstream has {len(gen_feats_downstream)} "
        f"but trust_results has {n_gen} samples"
    )

    # Build condition-to-class mapping (deterministic, sorted)
    cond_to_class, class_to_cond = build_condition_class_map(true_conditions)
    n_classes = len(class_to_cond)
    print(f"    {n_classes}-way classification task")

    # Check samples per condition
    cond_counts = {}
    for cond in true_conditions:
        cond_counts[cond] = cond_counts.get(cond, 0) + 1

    for cond, count in cond_counts.items():
        if count < n_bins:
            print(
                f"    WARNING: condition {cond} has only {count} samples (< n_bins={n_bins})"
            )

    # Build fixed stratified real test set (30% per condition, deterministic seed)
    rng = np.random.default_rng(seed)

    # Group real samples by condition
    real_by_cond: Dict[Tuple, List[int]] = {}
    for i in range(n_real):
        cond = tuple(
            int(
                real_meta[k][i].item()
                if isinstance(real_meta[k][i], torch.Tensor)
                else real_meta[k][i]
            )
            for k in condition_keys
        )
        real_by_cond.setdefault(cond, []).append(i)

    # Select 30% per condition for test set
    test_indices = []
    for cond in class_to_cond:
        cond_indices = real_by_cond.get(cond, [])
        if len(cond_indices) == 0:
            print(f"    WARNING: No real samples for condition {cond}")
            continue
        n_test = max(1, int(len(cond_indices) * 0.3))
        test_idx = rng.choice(cond_indices, size=n_test, replace=False)
        test_indices.extend(test_idx)

    test_indices = np.array(test_indices, dtype=int)
    X_test = real_feats_downstream[test_indices].numpy()
    y_test = np.array(
        [
            cond_to_class[
                tuple(
                    int(
                        real_meta[k][i].item()
                        if isinstance(real_meta[k][i], torch.Tensor)
                        else real_meta[k][i]
                    )
                    for k in condition_keys
                )
            ]
            for i in test_indices
        ],
        dtype=int,
    )

    print(
        f"    Fixed test set: {len(test_indices)} real samples, class balance: {np.bincount(y_test, minlength=n_classes)}"
    )

    # Ranking modes
    ranking_modes = {
        "trust": trust_scores,
        "realism": realism_scores,
        "faithfulness": faithfulness_scores,
        "random": rng.random(n_gen),  # Random baseline
    }

    all_results = []
    gen_feats_np = gen_feats_downstream.numpy()

    for mode_name, scores in ranking_modes.items():
        # Bin samples within each conditioning (ascending=True: lower score = better = bin 0)
        # For random mode, ascending doesn't matter
        ascending = True if mode_name != "random" else True
        binned = bin_samples_within_conditioning(
            true_conditions, np.asarray(scores), n_bins, ascending=ascending
        )

        # Sanity check: binning is per-conditioning
        total_binned = sum(sum(len(b) for b in bins) for bins in binned.values())
        assert total_binned == n_gen, f"Binning lost samples: {total_binned} vs {n_gen}"

        for bin_idx in range(n_bins):
            # Gather features from this bin across all conditions
            bin_indices = []
            bin_labels = []
            for cond in class_to_cond:
                if cond in binned and bin_idx < len(binned[cond]):
                    cond_bin_indices = binned[cond][bin_idx]
                    bin_indices.extend(cond_bin_indices)
                    bin_labels.extend([cond_to_class[cond]] * len(cond_bin_indices))

            if len(bin_indices) == 0:
                print(f"    WARNING: bin {bin_idx} for mode {mode_name} has no samples")
                continue

            X_train = gen_feats_np[bin_indices]
            y_train = np.array(bin_labels, dtype=int)

            bin_samples_per_cond = len(bin_indices) // n_classes if n_classes > 0 else 0

            # Train classifier with multiple seeds
            for s in range(n_seeds):
                clf_seed = seed + s * 1000 + bin_idx * 100
                try:
                    clf = LogisticRegression(
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=clf_seed,
                    )
                    clf.fit(X_train, y_train)
                    accuracy = clf.score(X_test, y_test)
                except Exception as e:
                    print(
                        f"    WARNING: Classifier failed for {mode_name} bin {bin_idx} seed {s}: {e}"
                    )
                    accuracy = np.nan

                all_results.append(
                    {
                        "model_name": model_name,
                        "scoring_feature_type": scoring_feature_type,
                        "downstream_feature_type": downstream_feature_type,
                        "ranking_mode": mode_name,
                        "bin_idx": bin_idx,
                        "seed": s,
                        "n_train": len(bin_indices),
                        "n_test": len(test_indices),
                        "accuracy": accuracy,
                        "bin_samples_per_condition": bin_samples_per_cond,
                        "n_conditions": n_classes,
                    }
                )

    df = pd.DataFrame(all_results)

    # Save detailed CSV
    dataset_prefix = f"{dataset}_" if dataset else ""
    config_key = f"{model_name}_{scoring_feature_type}_to_{downstream_feature_type}"
    csv_path = output_dir / f"{dataset_prefix}downstream_bin_selection_{config_key}.csv"
    df.to_csv(csv_path, index=False)

    # Create summary (mean/std by ranking_mode, bin_idx)
    summary = (
        df.groupby(["ranking_mode", "bin_idx"])
        .agg(
            {
                "accuracy": ["mean", "std", "count"],
                "n_train": "first",
            }
        )
        .reset_index()
    )
    summary.columns = [
        "ranking_mode",
        "bin_idx",
        "accuracy_mean",
        "accuracy_std",
        "n_seeds",
        "n_train",
    ]

    summary_path = output_dir / f"{dataset_prefix}downstream_bin_selection_summary_{config_key}.csv"
    summary.to_csv(summary_path, index=False)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"trust": "C0", "realism": "C1", "faithfulness": "C2", "random": "gray"}
    linestyles = {"trust": "-", "realism": "--", "faithfulness": ":", "random": "-."}

    for mode_name in ["trust", "realism", "faithfulness", "random"]:
        mode_summary = summary[summary["ranking_mode"] == mode_name]
        if len(mode_summary) == 0:
            continue

        x = mode_summary["bin_idx"]
        y = mode_summary["accuracy_mean"]
        yerr = mode_summary["accuracy_std"]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=mode_name.capitalize(),
            marker="o",
            capsize=4,
            alpha=0.8,
            color=colors.get(mode_name, "C0"),
            linestyle=linestyles.get(mode_name, "-"),
        )

    ax.set_xlabel("Bin Index (0=best scores, 9=worst scores)")
    ax.set_ylabel(f"Downstream {n_classes}-way Classification Accuracy")
    ax.set_title(
        f"Downstream Bin-Selection: {scoring_feature_type} → {downstream_feature_type}\n{model_name}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(n_bins))

    plot_path = output_dir / f"{dataset_prefix}downstream_bin_selection_{config_key}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"    Saved: {csv_path.name}, {summary_path.name}, {plot_path.name}")

    return df


def evaluate_rxrx1_downstream_bin_selection(
    trust_results: Dict,
    gen_feats: torch.Tensor,
    gen_meta: Dict,
    real_feats: torch.Tensor,
    real_meta: Dict,
    output_dir: Path,
    config_key: str,
    mode: str = "celltype",
    n_bins: int = 10,
    n_seeds: int = 5,
    seed: int = 0,
    dataset: str = "rxrx1",
) -> pd.DataFrame:
    """
    RxRx1 Task 5 analog: downstream bin-selection evaluation.

    Mirrors evaluate_downstream_bin_selection_from_scores for RxRx1.
    Two modes:
    - "celltype": 4 conditions (one per cell type), 4-way classifier, bin within cell type
    - "100pairs": 100 (cell_type_id, sirna_id) pairs (25 per cell type, include heldout), 100-way classifier

    Args:
        trust_results: Dict with scores and true_conditions (from compute_trust_results_from_features)
        gen_feats: Generated features (N, D) — used as training set for classifier
        gen_meta: Generated metadata with 'cell_type_id', 'sirna_id'
        real_feats: Real features (M, D) — used as test set
        real_meta: Real metadata with 'cell_type_id', 'sirna_id'
        output_dir: Output directory for CSV and plots
        config_key: Configuration key for output naming
        mode: "celltype" or "100pairs"
        n_bins: Number of bins (default 10)
        n_seeds: Number of seeds for classifier training
        seed: Base random seed
        dataset: Dataset name for output prefix
    """
    from sklearn.linear_model import LogisticRegression

    trust_scores = trust_results["trust_updated"]
    realism_scores = trust_results["realism_global_z"]
    faithfulness_scores = trust_results["faithfulness_margin_z"]
    true_conditions = trust_results["true_conditions"]  # list of (cell_type_id, sirna_id) tuples

    n_gen = len(trust_scores)
    assert len(gen_feats) == n_gen

    rng = np.random.default_rng(seed)

    # ---- Select conditions and build index mappings ----
    if mode == "celltype":
        # Collapse to cell_type_id only
        working_conditions = [tuple([cond[0]]) for cond in true_conditions]
        working_indices = list(range(n_gen))

        # Real: group by cell_type_id
        real_by_cond: Dict[Tuple, List[int]] = {}
        for i in range(len(real_feats)):
            ct = int(real_meta["cell_type_id"][i].item()
                     if isinstance(real_meta["cell_type_id"][i], torch.Tensor)
                     else real_meta["cell_type_id"][i])
            real_by_cond.setdefault((ct,), []).append(i)

    elif mode == "100pairs":
        # Select 100 (cell_type, sirna) pairs: 25/cell type, include heldout
        gen_by_cond: Dict[Tuple[int, int], List[int]] = {}
        for i in range(n_gen):
            cond = true_conditions[i]  # already (ct, sirna)
            gen_by_cond.setdefault(cond, []).append(i)

        real_by_cond_full: Dict[Tuple[int, int], List[int]] = {}
        for i in range(len(real_feats)):
            ct = int(real_meta["cell_type_id"][i].item()
                     if isinstance(real_meta["cell_type_id"][i], torch.Tensor)
                     else real_meta["cell_type_id"][i])
            sirna = int(real_meta["sirna_id"][i].item()
                        if isinstance(real_meta["sirna_id"][i], torch.Tensor)
                        else real_meta["sirna_id"][i])
            real_by_cond_full.setdefault((ct, sirna), []).append(i)

        cell_types = sorted(set(c[0] for c in gen_by_cond.keys()))
        n_per_cell = 100 // max(len(cell_types), 1)

        selected: Set[Tuple[int, int]] = set()
        for ct in cell_types:
            # Heldout pairs available for this cell type
            ct_heldout = [(ct, s) for (c, s) in RXRX1_HELDOUT_PAIRS
                          if c == ct and (ct, s) in gen_by_cond and (ct, s) in real_by_cond_full]
            selected.update(ct_heldout)
            # Fill remaining slots with seen pairs by support size
            ct_seen = [(ct, sirna, len(gen_by_cond[(ct, sirna)]))
                       for sirna in set(c[1] for c in gen_by_cond if c[0] == ct)
                       if (ct, sirna) not in selected and (ct, sirna) in real_by_cond_full]
            ct_seen.sort(key=lambda x: x[2], reverse=True)
            n_already = sum(1 for c in selected if c[0] == ct)
            for pair_ct, sirna, _ in ct_seen[:max(0, n_per_cell - n_already)]:
                selected.add((pair_ct, sirna))

        print(f"    Selected {len(selected)} pairs ({len([c for c in selected if c in RXRX1_HELDOUT_PAIRS])} heldout)")

        # Filter gen samples to selected pairs
        working_indices = [i for i, cond in enumerate(true_conditions) if cond in selected]
        working_conditions = [true_conditions[i] for i in working_indices]

        # Real: group by (ct, sirna) for selected pairs only
        real_by_cond = {cond: real_by_cond_full[cond]
                        for cond in selected if cond in real_by_cond_full}
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'celltype' or '100pairs'.")

    working_gen_feats = gen_feats[working_indices]
    n_working = len(working_indices)

    # Build class mapping
    cond_to_class, class_to_cond = build_condition_class_map(working_conditions)
    n_classes = len(class_to_cond)
    print(f"    Mode={mode!r}: {n_classes}-way classification, {n_working} gen samples")

    # Build real test set (30% per condition)
    test_indices = []
    for cond in class_to_cond:
        cond_indices = real_by_cond.get(cond, [])
        if not cond_indices:
            continue
        n_test = max(1, int(len(cond_indices) * 0.3))
        test_idx = rng.choice(cond_indices, size=n_test, replace=False)
        test_indices.extend(test_idx)

    test_indices = np.array(test_indices, dtype=int)
    X_test = real_feats[test_indices].numpy()

    if mode == "celltype":
        y_test = np.array([
            cond_to_class[tuple([int(real_meta["cell_type_id"][i].item()
                                     if isinstance(real_meta["cell_type_id"][i], torch.Tensor)
                                     else real_meta["cell_type_id"][i])])]
            for i in test_indices
        ], dtype=int)
    else:
        y_test = np.array([
            cond_to_class[tuple([
                int(real_meta["cell_type_id"][i].item()
                    if isinstance(real_meta["cell_type_id"][i], torch.Tensor)
                    else real_meta["cell_type_id"][i]),
                int(real_meta["sirna_id"][i].item()
                    if isinstance(real_meta["sirna_id"][i], torch.Tensor)
                    else real_meta["sirna_id"][i]),
            ])]
            for i in test_indices
        ], dtype=int)

    print(f"    Test set: {len(test_indices)} real samples")

    # Score arrays for working subset
    trust_w = np.asarray(trust_scores)[working_indices]
    realism_w = np.asarray(realism_scores)[working_indices]
    faithfulness_w = np.asarray(faithfulness_scores)[working_indices]
    random_w = rng.random(n_working)

    ranking_modes = {
        "trust": trust_w,
        "realism": realism_w,
        "faithfulness": faithfulness_w,
        "random": random_w,
    }

    all_results = []
    gen_feats_np = working_gen_feats.numpy() if isinstance(working_gen_feats, torch.Tensor) else working_gen_feats

    for mode_name, scores_w in ranking_modes.items():
        binned = bin_samples_within_conditioning(working_conditions, scores_w, n_bins, ascending=True)
        total_binned = sum(sum(len(b) for b in bins) for bins in binned.values())
        assert total_binned == n_working, f"Binning lost samples: {total_binned} vs {n_working}"

        for bin_idx in range(n_bins):
            bin_indices = []
            bin_labels = []
            for cond in class_to_cond:
                if cond in binned and bin_idx < len(binned[cond]):
                    idxs = binned[cond][bin_idx]
                    bin_indices.extend(idxs)
                    bin_labels.extend([cond_to_class[cond]] * len(idxs))

            if not bin_indices:
                continue

            X_train = gen_feats_np[bin_indices]
            y_train = np.array(bin_labels, dtype=int)

            for s in range(n_seeds):
                clf_seed = seed + s * 1000 + bin_idx * 100
                try:
                    clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=clf_seed)
                    clf.fit(X_train, y_train)
                    accuracy = clf.score(X_test, y_test)
                except Exception as e:
                    print(f"    WARNING: Classifier failed for {mode_name} bin {bin_idx} seed {s}: {e}")
                    accuracy = np.nan

                all_results.append({
                    "config_key": config_key,
                    "mode": mode,
                    "ranking_mode": mode_name,
                    "bin_idx": bin_idx,
                    "seed": s,
                    "n_train": len(bin_indices),
                    "n_test": len(test_indices),
                    "accuracy": accuracy,
                    "n_conditions": n_classes,
                })

    df = pd.DataFrame(all_results)

    # Save CSV
    dataset_prefix = f"{dataset}_" if dataset else ""
    csv_path = output_dir / f"{dataset_prefix}downstream_bin_{mode}_{config_key.replace('/', '_')}.csv"
    df.to_csv(csv_path, index=False)

    summary = (
        df.groupby(["ranking_mode", "bin_idx"])
        .agg(accuracy=("accuracy", "mean"), accuracy_std=("accuracy", "std"), n_train=("n_train", "first"))
        .reset_index()
    )
    summary_path = output_dir / f"{dataset_prefix}downstream_bin_{mode}_{config_key.replace('/', '_')}_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"trust": "C0", "realism": "C1", "faithfulness": "C2", "random": "gray"}
    linestyles = {"trust": "-", "realism": "--", "faithfulness": ":", "random": "-."}
    for mode_name in ["trust", "realism", "faithfulness", "random"]:
        row = summary[summary["ranking_mode"] == mode_name]
        if len(row) == 0:
            continue
        ax.errorbar(row["bin_idx"], row["accuracy"], yerr=row["accuracy_std"],
                    label=mode_name.capitalize(), marker="o", capsize=4, alpha=0.8,
                    color=colors.get(mode_name, "C0"), linestyle=linestyles.get(mode_name, "-"))
    ax.set_xlabel("Bin Index (0=best scores, 9=worst scores)")
    ax.set_ylabel(f"{n_classes}-way Classification Accuracy")
    ax.set_title(f"RxRx1 Downstream Bin-Selection ({mode})\n{config_key}")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(n_bins))
    plot_path = output_dir / f"{dataset_prefix}downstream_bin_{mode}_{config_key.replace('/', '_')}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"    Saved: {csv_path.name}, {summary_path.name}, {plot_path.name}")
    return df


def evaluate_celltype_classification(
    gen_feats: torch.Tensor,
    gen_meta: Dict,
    real_feats: torch.Tensor,
    real_meta: Dict,
    output_dir: Path,
    config_key: str,
    n_seeds: int = 5,
    seed: int = 42,
    dataset: str = "",
) -> Dict:
    """
    RxRx1 cell type classification (4-way).

    Train on generated samples (all perturbations), test on real samples.
    Classes: HEPG2=0, HUVEC=1, RPE=2, U2OS=3

    Args:
        gen_feats: Generated features (N_gen, D)
        gen_meta: Generated metadata with 'cell_type_id'
        real_feats: Real features (N_real, D)
        real_meta: Real metadata with 'cell_type_id'
        output_dir: Output directory for results
        config_key: Configuration key for naming
        n_seeds: Number of random seeds for classifier
        seed: Base random seed

    Returns:
        Dict with accuracy stats
    """
    from sklearn.linear_model import LogisticRegression

    # Extract cell type labels
    gen_labels = np.array([
        int(gen_meta["cell_type_id"][i].item()
            if isinstance(gen_meta["cell_type_id"][i], torch.Tensor)
            else gen_meta["cell_type_id"][i])
        for i in range(len(gen_feats))
    ])
    real_labels = np.array([
        int(real_meta["cell_type_id"][i].item()
            if isinstance(real_meta["cell_type_id"][i], torch.Tensor)
            else real_meta["cell_type_id"][i])
        for i in range(len(real_feats))
    ])

    X_train = gen_feats.numpy() if isinstance(gen_feats, torch.Tensor) else gen_feats
    X_test = real_feats.numpy() if isinstance(real_feats, torch.Tensor) else real_feats
    y_train = gen_labels
    y_test = real_labels

    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
    print(f"    Cell type classification: {n_classes}-way")
    print(f"    Train: {len(y_train)}, Test: {len(y_test)}")

    accuracies = []
    for s in range(n_seeds):
        clf_seed = seed + s * 1000
        clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=clf_seed)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        accuracies.append(acc)

    results = {
        "task": "celltype_classification",
        "n_classes": n_classes,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
    }

    # Save results
    dataset_prefix = f"{dataset}_" if dataset else ""
    csv_path = output_dir / f"{dataset_prefix}celltype_classification_{config_key.replace('/', '_')}.csv"
    pd.DataFrame([results]).to_csv(csv_path, index=False)
    print(f"    Cell type accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")

    return results


def evaluate_controlled_perturbation_classification(
    gen_feats: torch.Tensor,
    gen_meta: Dict,
    real_feats: torch.Tensor,
    real_meta: Dict,
    output_dir: Path,
    config_key: str,
    n_perturbations_per_cell: int = 25,
    n_heldout_per_experiment: int = 20,
    n_seeds: int = 5,
    seed: int = 42,
    dataset: str = "",
) -> Dict:
    """
    RxRx1 controlled perturbation classification.

    Select K perturbations per cell type, stratified by support size.
    Always include n_heldout_per_experiment unseen pairs (from RXRX1_HELDOUT_PAIRS).
    Train on generated, test on real.
    Report separately for seen vs heldout pairs.

    Args:
        gen_feats: Generated features (N_gen, D)
        gen_meta: Generated metadata with 'cell_type_id', 'sirna_id'
        real_feats: Real features (N_real, D)
        real_meta: Real metadata with 'cell_type_id', 'sirna_id'
        output_dir: Output directory
        config_key: Configuration key for naming
        n_perturbations_per_cell: K perturbations per cell type (default 25)
        n_heldout_per_experiment: Number of heldout pairs to always include (default 20 of 100)
        n_seeds: Number of random seeds
        seed: Base random seed

    Returns:
        Dict with accuracy stats for all/seen/unseen pairs
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)

    # Build condition to indices mapping for generated and real
    gen_by_cond: Dict[Tuple[int, int], List[int]] = {}
    for i in range(len(gen_feats)):
        cell_type = int(gen_meta["cell_type_id"][i].item()
                       if isinstance(gen_meta["cell_type_id"][i], torch.Tensor)
                       else gen_meta["cell_type_id"][i])
        sirna = int(gen_meta["sirna_id"][i].item()
                   if isinstance(gen_meta["sirna_id"][i], torch.Tensor)
                   else gen_meta["sirna_id"][i])
        cond = (cell_type, sirna)
        gen_by_cond.setdefault(cond, []).append(i)

    real_by_cond: Dict[Tuple[int, int], List[int]] = {}
    for i in range(len(real_feats)):
        cell_type = int(real_meta["cell_type_id"][i].item()
                       if isinstance(real_meta["cell_type_id"][i], torch.Tensor)
                       else real_meta["cell_type_id"][i])
        sirna = int(real_meta["sirna_id"][i].item()
                   if isinstance(real_meta["sirna_id"][i], torch.Tensor)
                   else real_meta["sirna_id"][i])
        cond = (cell_type, sirna)
        real_by_cond.setdefault(cond, []).append(i)

    # Get unique cell types
    cell_types = sorted(set(c[0] for c in gen_by_cond.keys()))
    print(f"    Found {len(cell_types)} cell types: {cell_types}")

    # Separate heldout pairs per cell type
    heldout_by_cell: Dict[int, Set[int]] = {ct: set() for ct in cell_types}
    for cell_type, sirna in RXRX1_HELDOUT_PAIRS:
        if cell_type in heldout_by_cell:
            heldout_by_cell[cell_type].add(sirna)

    # Select perturbations per cell type
    # Strategy: pick top K by support size, but reserve some slots for heldout
    selected_conditions: Set[Tuple[int, int]] = set()

    # First, select some heldout pairs (distributed across cell types)
    n_heldout_per_cell = n_heldout_per_experiment // len(cell_types)
    for cell_type in cell_types:
        cell_heldout = [(cell_type, s) for s in heldout_by_cell[cell_type]
                        if (cell_type, s) in gen_by_cond and (cell_type, s) in real_by_cond]
        n_to_select = min(n_heldout_per_cell, len(cell_heldout))
        if n_to_select > 0:
            selected_heldout = rng.choice(cell_heldout, size=n_to_select, replace=False)
            selected_conditions.update(tuple(c) for c in selected_heldout)

    # Then fill remaining slots with seen perturbations (by support size)
    for cell_type in cell_types:
        # Get perturbations for this cell type (not already selected)
        cell_perts = [(cell_type, sirna, len(gen_by_cond.get((cell_type, sirna), [])))
                      for sirna in set(c[1] for c in gen_by_cond.keys() if c[0] == cell_type)
                      if (cell_type, sirna) not in selected_conditions
                      and (cell_type, sirna) in real_by_cond]

        # Sort by support size (descending) and select top K - already_selected
        cell_perts.sort(key=lambda x: x[2], reverse=True)
        n_already = sum(1 for c in selected_conditions if c[0] == cell_type)
        n_to_add = max(0, n_perturbations_per_cell - n_already)
        for ct, sirna, _ in cell_perts[:n_to_add]:
            selected_conditions.add((ct, sirna))

    # Build class mapping
    selected_list = sorted(selected_conditions)
    cond_to_class = {c: i for i, c in enumerate(selected_list)}
    n_classes = len(selected_list)

    print(f"    Selected {n_classes} conditions ({len([c for c in selected_list if c in RXRX1_HELDOUT_PAIRS])} heldout)")

    # Gather training (gen) and test (real) data
    train_idx, train_labels = [], []
    test_idx, test_labels = [], []
    is_heldout = []

    for cond in selected_list:
        class_id = cond_to_class[cond]
        # Training: all gen samples for this condition
        train_idx.extend(gen_by_cond.get(cond, []))
        train_labels.extend([class_id] * len(gen_by_cond.get(cond, [])))
        # Test: all real samples for this condition
        real_indices = real_by_cond.get(cond, [])
        test_idx.extend(real_indices)
        test_labels.extend([class_id] * len(real_indices))
        is_heldout.extend([cond in RXRX1_HELDOUT_PAIRS] * len(real_indices))

    X_train = gen_feats[train_idx].numpy() if isinstance(gen_feats, torch.Tensor) else gen_feats[train_idx]
    X_test = real_feats[test_idx].numpy() if isinstance(real_feats, torch.Tensor) else real_feats[test_idx]
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    is_heldout = np.array(is_heldout)

    print(f"    Train: {len(y_train)}, Test: {len(y_test)} ({is_heldout.sum()} heldout)")

    # Train and evaluate
    results_all = []
    results_seen = []
    results_heldout = []

    for s in range(n_seeds):
        clf_seed = seed + s * 1000
        clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=clf_seed)
        clf.fit(X_train, y_train)

        # All samples
        y_pred = clf.predict(X_test)
        acc_all = (y_pred == y_test).mean()
        results_all.append(acc_all)

        # Seen only
        seen_mask = ~is_heldout
        if seen_mask.sum() > 0:
            acc_seen = (y_pred[seen_mask] == y_test[seen_mask]).mean()
            results_seen.append(acc_seen)

        # Heldout only
        heldout_mask = is_heldout
        if heldout_mask.sum() > 0:
            acc_heldout = (y_pred[heldout_mask] == y_test[heldout_mask]).mean()
            results_heldout.append(acc_heldout)

    results = {
        "task": "controlled_perturbation_classification",
        "n_classes": n_classes,
        "n_heldout_classes": len([c for c in selected_list if c in RXRX1_HELDOUT_PAIRS]),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_test_heldout": int(is_heldout.sum()),
        "accuracy_all_mean": float(np.mean(results_all)),
        "accuracy_all_std": float(np.std(results_all)),
        "accuracy_seen_mean": float(np.mean(results_seen)) if results_seen else np.nan,
        "accuracy_seen_std": float(np.std(results_seen)) if results_seen else np.nan,
        "accuracy_heldout_mean": float(np.mean(results_heldout)) if results_heldout else np.nan,
        "accuracy_heldout_std": float(np.std(results_heldout)) if results_heldout else np.nan,
    }

    # Save results
    dataset_prefix = f"{dataset}_" if dataset else ""
    csv_path = output_dir / f"{dataset_prefix}perturbation_classification_{config_key.replace('/', '_')}.csv"
    pd.DataFrame([results]).to_csv(csv_path, index=False)
    print(f"    Perturbation accuracy (all): {results['accuracy_all_mean']:.4f} ± {results['accuracy_all_std']:.4f}")
    print(f"    Perturbation accuracy (seen): {results['accuracy_seen_mean']:.4f} ± {results['accuracy_seen_std']:.4f}")
    print(f"    Perturbation accuracy (heldout): {results['accuracy_heldout_mean']:.4f} ± {results['accuracy_heldout_std']:.4f}")

    return results


def evaluate_rxrx1_decomposed_classification(
    gen_feats: torch.Tensor,
    gen_meta: Dict,
    real_feats: torch.Tensor,
    real_meta: Dict,
    output_dir: Path,
    config_key: str,
    n_heldout: int = 20,
    n_seen: int = 80,
    seed: int = 42,
    dataset: str = "",
) -> Dict:
    """
    RxRx1 decomposed 100-class pair classification.

    Trains a 100-class classifier on (cell_type, siRNA) pairs from generated data,
    tests on real data, and decomposes predictions into:
    - Cell type accuracy
    - siRNA accuracy
    - Joint (pair) accuracy

    Reports seen vs heldout breakdown for each metric.

    Args:
        gen_feats: Generated features (N_gen, D)
        gen_meta: Generated metadata with 'cell_type_id', 'sirna_id'
        real_feats: Real features (N_real, D)
        real_meta: Real metadata with 'cell_type_id', 'sirna_id'
        output_dir: Output directory for results
        config_key: Configuration key for naming
        n_heldout: Number of heldout pairs to include (default 20)
        n_seen: Number of seen pairs to include (default 80)
        seed: Random seed for pair selection

    Returns:
        Dict with accuracy stats for cell_type, siRNA, and joint (seen/heldout/overall)
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)

    # Extract labels
    def get_labels(meta, n):
        ct = np.array([
            int(meta["cell_type_id"][i].item()
                if isinstance(meta["cell_type_id"][i], torch.Tensor)
                else meta["cell_type_id"][i])
            for i in range(n)
        ])
        sirna = np.array([
            int(meta["sirna_id"][i].item()
                if isinstance(meta["sirna_id"][i], torch.Tensor)
                else meta["sirna_id"][i])
            for i in range(n)
        ])
        return ct, sirna

    gen_ct, gen_sirna = get_labels(gen_meta, len(gen_feats))
    real_ct, real_sirna = get_labels(real_meta, len(real_feats))

    # Find common pairs
    gen_pairs = set(zip(gen_ct, gen_sirna))
    real_pairs = set(zip(real_ct, real_sirna))
    common_pairs = gen_pairs & real_pairs

    # Separate heldout and seen
    heldout_available = [p for p in RXRX1_HELDOUT_PAIRS if p in common_pairs]
    seen_available = [p for p in common_pairs if p not in RXRX1_HELDOUT_PAIRS]

    if len(heldout_available) < n_heldout:
        print(f"    Warning: Only {len(heldout_available)} heldout pairs available (requested {n_heldout})")
    if len(seen_available) < n_seen:
        print(f"    Warning: Only {len(seen_available)} seen pairs available (requested {n_seen})")

    selected_heldout = [tuple(p) for p in rng.choice(
        heldout_available,
        size=min(n_heldout, len(heldout_available)),
        replace=False
    )]
    selected_seen = [tuple(p) for p in rng.choice(
        seen_available,
        size=min(n_seen, len(seen_available)),
        replace=False
    )]
    all_selected = selected_heldout + selected_seen
    n_classes = len(all_selected)

    # Build pair -> class mapping
    pair_to_class = {p: i for i, p in enumerate(all_selected)}
    class_to_ct = {i: p[0] for i, p in enumerate(all_selected)}
    class_to_sirna = {i: p[1] for i, p in enumerate(all_selected)}
    heldout_classes = set(range(len(selected_heldout)))

    print(f"    Selected {n_classes} pairs: {len(selected_heldout)} heldout, {len(selected_seen)} seen")

    # Build datasets
    gen_feats_np = gen_feats.numpy() if isinstance(gen_feats, torch.Tensor) else gen_feats
    real_feats_np = real_feats.numpy() if isinstance(real_feats, torch.Tensor) else real_feats

    gen_mask = np.array([tuple([gen_ct[i], gen_sirna[i]]) in pair_to_class for i in range(len(gen_feats))])
    real_mask = np.array([tuple([real_ct[i], real_sirna[i]]) in pair_to_class for i in range(len(real_feats))])

    X_gen = gen_feats_np[gen_mask]
    y_gen = np.array([pair_to_class[tuple([gen_ct[i], gen_sirna[i]])] for i in np.where(gen_mask)[0]])

    X_real = real_feats_np[real_mask]
    y_real = np.array([pair_to_class[tuple([real_ct[i], real_sirna[i]])] for i in np.where(real_mask)[0]])
    is_heldout_real = np.array([y in heldout_classes for y in y_real])

    print(f"    Train (gen): {len(X_gen)}, Test (real): {len(X_real)} ({is_heldout_real.sum()} heldout)")

    # Train 100-class pair classifier
    clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed)
    clf.fit(X_gen, y_gen)

    # Predict on real
    pred = clf.predict(X_real)

    # Decompose predictions
    pred_ct = np.array([class_to_ct[p] for p in pred])
    pred_sirna = np.array([class_to_sirna[p] for p in pred])
    true_ct = np.array([class_to_ct[y] for y in y_real])
    true_sirna = np.array([class_to_sirna[y] for y in y_real])

    ct_correct = (pred_ct == true_ct)
    sirna_correct = (pred_sirna == true_sirna)
    pair_correct = (pred == y_real)

    # Compute metrics
    def compute_metrics(correct_arr, is_heldout):
        return {
            "overall": float(correct_arr.mean()),
            "seen": float(correct_arr[~is_heldout].mean()) if (~is_heldout).sum() > 0 else np.nan,
            "heldout": float(correct_arr[is_heldout].mean()) if is_heldout.sum() > 0 else np.nan,
            "n_overall": len(correct_arr),
            "n_seen": int((~is_heldout).sum()),
            "n_heldout": int(is_heldout.sum()),
        }

    results = {
        "task": "rxrx1_decomposed_classification",
        "n_classes": n_classes,
        "n_heldout_classes": len(selected_heldout),
        "n_seen_classes": len(selected_seen),
        "random_baseline": 1.0 / n_classes,
        "celltype": compute_metrics(ct_correct, is_heldout_real),
        "sirna": compute_metrics(sirna_correct, is_heldout_real),
        "pair": compute_metrics(pair_correct, is_heldout_real),
    }

    # Print results
    print(f"    {n_classes}-class pair classification (random baseline: {results['random_baseline']:.4f}):")
    print(f"    {'Metric':<15} {'Overall':>10} {'Seen':>10} {'Heldout':>10}")
    print(f"    {'-'*45}")
    for metric_name in ["celltype", "sirna", "pair"]:
        m = results[metric_name]
        print(f"    {metric_name:<15} {m['overall']:>10.4f} {m['seen']:>10.4f} {m['heldout']:>10.4f}")

    # Save results
    dataset_prefix = f"{dataset}_" if dataset else ""
    csv_path = output_dir / f"{dataset_prefix}rxrx1_decomposed_{config_key.replace('/', '_')}.csv"

    # Flatten for CSV
    flat_results = {
        "config_key": config_key,
        "n_classes": n_classes,
        "random_baseline": results["random_baseline"],
    }
    for metric_name in ["celltype", "sirna", "pair"]:
        for stat in ["overall", "seen", "heldout"]:
            flat_results[f"{metric_name}_{stat}"] = results[metric_name][stat]

    pd.DataFrame([flat_results]).to_csv(csv_path, index=False)

    return results
