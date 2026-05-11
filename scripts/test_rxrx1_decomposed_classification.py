"""
RxRx1 Decomposed Classification Analysis.

Analyzes perturbation encoding by decomposing into:
1. Cell type classification (4-way)
2. siRNA classification (within selected set)
3. Joint (pair) accuracy = both correct

Tests transfer in multiple directions:
- Gen→Gen: Are generated features internally consistent?
- Gen→Real: Do generated features match real features?
- Real→Gen: Are generated features recognizable by real-trained classifier?
- Real→Real: Baseline

Breaks down by seen vs heldout (cell_type, siRNA) pairs.

Usage:
    PYTHONPATH=src uv run python scripts/test_rxrx1_decomposed_classification.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from faithful_cond_gen.eval.trust_eval.config import (
    FEATURE_CONFIGS,
    REAL_FEATURE_PATHS,
    RXRX1_HELDOUT_PAIRS,
)


def get_labels(meta, n):
    """Extract cell_type and sirna labels from metadata."""
    ct = np.array([
        int(meta["cell_type_id"][i].item()
            if hasattr(meta["cell_type_id"][i], "item")
            else meta["cell_type_id"][i])
        for i in range(n)
    ])
    sirna = np.array([
        int(meta["sirna_id"][i].item()
            if hasattr(meta["sirna_id"][i], "item")
            else meta["sirna_id"][i])
        for i in range(n)
    ])
    return ct, sirna


def load_features(model_name):
    """Load generated and real features for a model."""
    gen_cfg = FEATURE_CONFIGS[("rxrx1", model_name, "dinov3")]
    gen_path = Path(f"outputs/gen/{gen_cfg[0]}/{gen_cfg[1]}")
    gen_data = torch.load(gen_path, weights_only=False)
    gen_feats = gen_data["features"].numpy()
    gen_meta = gen_data.get("metadata", gen_data.get("cond", {}))
    gen_ct, gen_sirna = get_labels(gen_meta, len(gen_feats))

    real_path = Path(REAL_FEATURE_PATHS[("rxrx1", "dinov3")])
    real_data = torch.load(real_path, weights_only=False)
    real_feats = real_data["features"].numpy()
    real_meta = real_data.get("metadata", real_data.get("cond", {}))
    real_ct, real_sirna = get_labels(real_meta, len(real_feats))

    return {
        "gen_feats": gen_feats,
        "gen_ct": gen_ct,
        "gen_sirna": gen_sirna,
        "real_feats": real_feats,
        "real_ct": real_ct,
        "real_sirna": real_sirna,
    }


def select_conditions(gen_ct, gen_sirna, real_ct, real_sirna,
                      n_heldout=20, n_seen=80, seed=42):
    """Select conditions for evaluation: n_heldout + n_seen pairs."""
    rng = np.random.default_rng(seed)

    gen_pairs = set(zip(gen_ct, gen_sirna))
    real_pairs = set(zip(real_ct, real_sirna))
    common_pairs = gen_pairs & real_pairs

    # Separate heldout and seen
    heldout_available = [p for p in RXRX1_HELDOUT_PAIRS if p in common_pairs]
    seen_available = [p for p in common_pairs if p not in RXRX1_HELDOUT_PAIRS]

    selected_heldout = list(rng.choice(
        heldout_available,
        size=min(n_heldout, len(heldout_available)),
        replace=False
    ))
    selected_seen = list(rng.choice(
        seen_available,
        size=min(n_seen, len(seen_available)),
        replace=False
    ))

    return selected_heldout, selected_seen


def build_dataset(feats, ct, sirna, selected_pairs, sirna_to_class):
    """Build feature matrix and labels for selected pairs."""
    selected_set = set(tuple(p) for p in selected_pairs)
    heldout_set = set(tuple(p) for p in selected_pairs if tuple(p) in RXRX1_HELDOUT_PAIRS)

    mask = np.array([tuple([ct[i], sirna[i]]) in selected_set for i in range(len(feats))])

    X = feats[mask]
    y_ct = ct[mask]
    y_sirna = np.array([sirna_to_class[s] for s in sirna[mask]])
    is_heldout = np.array([tuple([ct[i], sirna[i]]) in heldout_set for i in np.where(mask)[0]])

    return X, y_ct, y_sirna, is_heldout


def evaluate_classifier(clf, X, y, is_heldout):
    """Evaluate classifier and return seen/heldout breakdown."""
    pred = clf.predict(X)
    correct = (pred == y)

    results = {
        "overall": correct.mean(),
        "n_overall": len(y),
    }

    if (~is_heldout).sum() > 0:
        results["seen"] = correct[~is_heldout].mean()
        results["n_seen"] = (~is_heldout).sum()
    else:
        results["seen"] = np.nan
        results["n_seen"] = 0

    if is_heldout.sum() > 0:
        results["heldout"] = correct[is_heldout].mean()
        results["n_heldout"] = is_heldout.sum()
    else:
        results["heldout"] = np.nan
        results["n_heldout"] = 0

    return results


def run_decomposed_analysis(data, selected_heldout, selected_seen, seed=42):
    """
    Run full decomposed analysis for one model.

    Returns dict with results for:
    - Cell type classifier (gen→real, real→gen, etc.)
    - siRNA classifier (gen→real, real→gen, etc.)
    - Joint accuracy
    """
    all_selected = selected_heldout + selected_seen
    selected_sirnas = sorted(set(p[1] for p in all_selected))
    sirna_to_class = {s: i for i, s in enumerate(selected_sirnas)}

    # Build datasets
    X_gen, y_gen_ct, y_gen_sirna, is_heldout_gen = build_dataset(
        data["gen_feats"], data["gen_ct"], data["gen_sirna"],
        all_selected, sirna_to_class
    )
    X_real, y_real_ct, y_real_sirna, is_heldout_real = build_dataset(
        data["real_feats"], data["real_ct"], data["real_sirna"],
        all_selected, sirna_to_class
    )

    n_sirna_classes = len(selected_sirnas)

    results = {
        "n_conditions": len(all_selected),
        "n_heldout_conditions": len(selected_heldout),
        "n_sirna_classes": n_sirna_classes,
        "n_gen": len(X_gen),
        "n_real": len(X_real),
    }

    # Split gen for gen→gen evaluation
    X_gen_tr, X_gen_te, y_gen_ct_tr, y_gen_ct_te, y_gen_sirna_tr, y_gen_sirna_te, h_gen_tr, h_gen_te = \
        train_test_split(X_gen, y_gen_ct, y_gen_sirna, is_heldout_gen,
                        test_size=0.3, random_state=seed, stratify=y_gen_sirna)

    # Split real for real→real evaluation
    X_real_tr, X_real_te, y_real_ct_tr, y_real_ct_te, y_real_sirna_tr, y_real_sirna_te, h_real_tr, h_real_te = \
        train_test_split(X_real, y_real_ct, y_real_sirna, is_heldout_real,
                        test_size=0.3, random_state=seed, stratify=y_real_sirna)

    # ===== CELL TYPE CLASSIFIERS =====

    # Gen→Gen (cell type)
    clf_ct_gen = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed)
    clf_ct_gen.fit(X_gen_tr, y_gen_ct_tr)
    results["ct_gen_gen"] = evaluate_classifier(clf_ct_gen, X_gen_te, y_gen_ct_te, h_gen_te)

    # Gen→Real (cell type) - train on all gen
    clf_ct_gen_full = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed)
    clf_ct_gen_full.fit(X_gen, y_gen_ct)
    results["ct_gen_real"] = evaluate_classifier(clf_ct_gen_full, X_real, y_real_ct, is_heldout_real)

    # Real→Real (cell type)
    clf_ct_real = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed)
    clf_ct_real.fit(X_real_tr, y_real_ct_tr)
    results["ct_real_real"] = evaluate_classifier(clf_ct_real, X_real_te, y_real_ct_te, h_real_te)

    # Real→Gen (cell type) - train on all real
    clf_ct_real_full = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed)
    clf_ct_real_full.fit(X_real, y_real_ct)
    results["ct_real_gen"] = evaluate_classifier(clf_ct_real_full, X_gen, y_gen_ct, is_heldout_gen)

    # ===== siRNA CLASSIFIERS =====

    # Gen→Gen (siRNA)
    clf_sirna_gen = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed)
    clf_sirna_gen.fit(X_gen_tr, y_gen_sirna_tr)
    results["sirna_gen_gen"] = evaluate_classifier(clf_sirna_gen, X_gen_te, y_gen_sirna_te, h_gen_te)

    # Gen→Real (siRNA)
    clf_sirna_gen_full = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed)
    clf_sirna_gen_full.fit(X_gen, y_gen_sirna)
    results["sirna_gen_real"] = evaluate_classifier(clf_sirna_gen_full, X_real, y_real_sirna, is_heldout_real)

    # Real→Real (siRNA)
    clf_sirna_real = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed)
    clf_sirna_real.fit(X_real_tr, y_real_sirna_tr)
    results["sirna_real_real"] = evaluate_classifier(clf_sirna_real, X_real_te, y_real_sirna_te, h_real_te)

    # Real→Gen (siRNA)
    clf_sirna_real_full = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed)
    clf_sirna_real_full.fit(X_real, y_real_sirna)
    results["sirna_real_gen"] = evaluate_classifier(clf_sirna_real_full, X_gen, y_gen_sirna, is_heldout_gen)

    # ===== JOINT (PAIR) ACCURACY =====
    # Using the full classifiers, compute joint accuracy

    # Gen→Real joint
    pred_ct = clf_ct_gen_full.predict(X_real)
    pred_sirna = clf_sirna_gen_full.predict(X_real)
    joint_correct = (pred_ct == y_real_ct) & (pred_sirna == y_real_sirna)
    results["joint_gen_real"] = {
        "overall": joint_correct.mean(),
        "seen": joint_correct[~is_heldout_real].mean() if (~is_heldout_real).sum() > 0 else np.nan,
        "heldout": joint_correct[is_heldout_real].mean() if is_heldout_real.sum() > 0 else np.nan,
    }

    # Real→Gen joint
    pred_ct = clf_ct_real_full.predict(X_gen)
    pred_sirna = clf_sirna_real_full.predict(X_gen)
    joint_correct = (pred_ct == y_gen_ct) & (pred_sirna == y_gen_sirna)
    results["joint_real_gen"] = {
        "overall": joint_correct.mean(),
        "seen": joint_correct[~is_heldout_gen].mean() if (~is_heldout_gen).sum() > 0 else np.nan,
        "heldout": joint_correct[is_heldout_gen].mean() if is_heldout_gen.sum() > 0 else np.nan,
    }

    return results


def print_results(model_name, results):
    """Print results in a nice table format."""
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")
    print(f"Conditions: {results['n_conditions']} ({results['n_heldout_conditions']} heldout)")
    print(f"siRNA classes: {results['n_sirna_classes']}")
    print(f"Samples: Gen={results['n_gen']}, Real={results['n_real']}")

    print(f"\n{'CELL TYPE CLASSIFICATION (4-way)':^70}")
    print("-" * 70)
    print(f"{'Direction':<20} {'Overall':>12} {'Seen':>12} {'Heldout':>12}")
    print("-" * 70)
    for key, label in [
        ("ct_gen_gen", "Gen → Gen"),
        ("ct_gen_real", "Gen → Real"),
        ("ct_real_real", "Real → Real"),
        ("ct_real_gen", "Real → Gen"),
    ]:
        r = results[key]
        print(f"{label:<20} {r['overall']:>12.4f} {r['seen']:>12.4f} {r['heldout']:>12.4f}")

    print(f"\n{'siRNA CLASSIFICATION (' + str(results['n_sirna_classes']) + '-way)':^70}")
    print("-" * 70)
    print(f"{'Direction':<20} {'Overall':>12} {'Seen':>12} {'Heldout':>12}")
    print("-" * 70)
    for key, label in [
        ("sirna_gen_gen", "Gen → Gen"),
        ("sirna_gen_real", "Gen → Real"),
        ("sirna_real_real", "Real → Real"),
        ("sirna_real_gen", "Real → Gen"),
    ]:
        r = results[key]
        print(f"{label:<20} {r['overall']:>12.4f} {r['seen']:>12.4f} {r['heldout']:>12.4f}")

    print(f"\n{'JOINT (PAIR) ACCURACY':^70}")
    print("-" * 70)
    print(f"{'Direction':<20} {'Overall':>12} {'Seen':>12} {'Heldout':>12}")
    print("-" * 70)
    for key, label in [
        ("joint_gen_real", "Gen → Real"),
        ("joint_real_gen", "Real → Gen"),
    ]:
        r = results[key]
        print(f"{label:<20} {r['overall']:>12.4f} {r['seen']:>12.4f} {r['heldout']:>12.4f}")


def main():
    print("=" * 70)
    print("RxRx1 DECOMPOSED CLASSIFICATION ANALYSIS")
    print("=" * 70)
    print("\nThis analysis decomposes the 100-class (cell_type, siRNA) task into:")
    print("  1. Cell type classification (4-way)")
    print("  2. siRNA classification (N-way)")
    print("  3. Joint accuracy (both correct)")
    print("\nTests transfer in multiple directions to understand feature quality.")

    models = ["repa_full", "repa_marginal", "vanilla_full", "vanilla_marginal"]
    seed = 42

    all_results = {}

    # Load real data once for condition selection
    real_path = Path(REAL_FEATURE_PATHS[("rxrx1", "dinov3")])
    real_data = torch.load(real_path, weights_only=False)
    real_ct, real_sirna = get_labels(
        real_data.get("metadata", real_data.get("cond", {})),
        len(real_data["features"])
    )

    # Use first model to get gen labels for condition selection
    first_data = load_features(models[0])

    # Select conditions (same for all models for fair comparison)
    selected_heldout, selected_seen = select_conditions(
        first_data["gen_ct"], first_data["gen_sirna"],
        real_ct, real_sirna,
        n_heldout=20, n_seen=80, seed=seed
    )

    print(f"\nSelected {len(selected_heldout)} heldout + {len(selected_seen)} seen conditions")

    for model_name in models:
        print(f"\nLoading {model_name}...")
        data = load_features(model_name)

        results = run_decomposed_analysis(data, selected_heldout, selected_seen, seed=seed)
        all_results[model_name] = results

        print_results(model_name, results)

    # Summary comparison
    print("\n\n" + "=" * 70)
    print("SUMMARY: siRNA Gen→Real Heldout Accuracy (key metric)")
    print("=" * 70)
    print(f"{'Model':<20} {'siRNA Heldout':>15} {'Joint Heldout':>15}")
    print("-" * 50)
    for model_name in models:
        r = all_results[model_name]
        sirna_held = r["sirna_gen_real"]["heldout"]
        joint_held = r["joint_gen_real"]["heldout"]
        print(f"{model_name:<20} {sirna_held:>15.4f} {joint_held:>15.4f}")

    # Save results
    output_dir = Path("outputs/trust_evaluation/rxrx1_decomposed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flatten results for CSV
    rows = []
    for model_name, results in all_results.items():
        for key in results:
            if isinstance(results[key], dict) and "overall" in results[key]:
                for metric in ["overall", "seen", "heldout"]:
                    rows.append({
                        "model": model_name,
                        "task": key,
                        "metric": metric,
                        "value": results[key][metric],
                    })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "decomposed_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
