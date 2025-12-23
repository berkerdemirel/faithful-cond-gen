"""Train linear classifiers on cached RxRx1 features to evaluate feature quality."""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def load_cached_features(cache_path: Path):
    """Load cached features and metadata."""
    train_path = cache_path / "train_features.pt"
    val_path = cache_path / "val_features.pt"

    print(f"Loading features from {cache_path}")
    train_data = torch.load(train_path, weights_only=False)
    val_data = torch.load(val_path, weights_only=False)

    return train_data, val_data


def load_generated_features(cache_path: Path):
    """Load generated features and metadata."""
    features_path = cache_path / "generated_unpacked_images_features.pt"

    if not features_path.exists():
        return None

    print(f"Loading generated features from {cache_path}")
    data = torch.load(features_path, weights_only=False)
    return data


def evaluate_multiclass_classifier(clf, X, y, n_classes):
    """Evaluate a trained multiclass classifier on a dataset."""
    # Convert to numpy if needed
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    # Predict
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    # Compute metrics
    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    f1_weighted = f1_score(y, y_pred, average="weighted")

    # Compute AUC-ROC (one-vs-rest)
    try:
        y_bin = label_binarize(y, classes=range(n_classes))
        if n_classes == 2:
            auc = roc_auc_score(y_bin, y_pred_proba[:, 1])
        else:
            auc = roc_auc_score(y_bin, y_pred_proba, average="macro", multi_class="ovr")
    except:
        auc = float("nan")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "auc_roc": auc,
    }


def train_multiclass_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    attribute_name: str,
    n_classes: int,
    X_generated_full=None,
    y_generated_full=None,
    X_generated_marginal=None,
    y_generated_marginal=None,
):
    """Train a multiclass classifier for a single attribute and evaluate on all datasets."""
    # Convert to numpy if needed
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.cpu().numpy()
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.cpu().numpy()

    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate on all datasets
    results = {}

    # Train set
    results["train"] = evaluate_multiclass_classifier(clf, X_train, y_train, n_classes)

    # Validation set
    results["val"] = evaluate_multiclass_classifier(clf, X_val, y_val, n_classes)

    # Generated full model
    if X_generated_full is not None:
        results["generated_full"] = evaluate_multiclass_classifier(
            clf, X_generated_full, y_generated_full, n_classes
        )

    # Generated marginal model
    if X_generated_marginal is not None:
        results["generated_marginal"] = evaluate_multiclass_classifier(
            clf, X_generated_marginal, y_generated_marginal, n_classes
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train linear probes on cached RxRx1 features"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Encoder name (e.g., dinov2, dinov3, mae, bioclip, openphenom)",
    )
    parser.add_argument(
        "--feature_cache_root",
        type=str,
        default="feature_cache",
        help="Root directory for cached features (default: feature_cache)",
    )
    parser.add_argument(
        "--skip_generated",
        action="store_true",
        help="Skip evaluation on generated samples",
    )
    args = parser.parse_args()

    # Construct paths
    dataset = "rxrx1"
    feature_cache_root = Path(args.feature_cache_root)
    cache_path = feature_cache_root / "real_samples" / dataset / args.encoder
    generated_full_path = (
        feature_cache_root / "generated_samples" / dataset / "fullmodel" / args.encoder
    )
    generated_marginal_path = (
        feature_cache_root
        / "generated_samples"
        / dataset
        / "marginalmodel"
        / args.encoder
    )

    print(f"\nDataset: {dataset}")
    print(f"Encoder: {args.encoder}")
    print(f"Real samples path: {cache_path}")

    # Load features
    train_data, val_data = load_cached_features(cache_path)

    X_train = train_data["features"]
    X_val = val_data["features"]

    print(f"\nFeature dimension: {train_data['feature_dim']}")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Val samples: {X_val.shape[0]}")

    # Load generated samples if provided
    generated_full_data = None
    generated_marginal_data = None

    if not args.skip_generated:
        generated_full_data = load_generated_features(generated_full_path)
        if generated_full_data:
            print(
                f"Generated (full) samples: {generated_full_data['features'].shape[0]}"
            )

        generated_marginal_data = load_generated_features(generated_marginal_path)
        if generated_marginal_data:
            print(
                f"Generated (marginal) samples: {generated_marginal_data['features'].shape[0]}"
            )

    # Determine which attributes to train on
    metadata_keys = list(train_data["metadata"].keys())
    # Exclude comp_category if present (it's an auxiliary key)
    attributes = [k for k in metadata_keys if k != "comp_category"]

    print(f"\nTraining linear classifiers for: {attributes}")
    print("=" * 80)

    # Train classifiers for each attribute
    all_results = {}

    for attr in attributes:
        if attr not in train_data["metadata"]:
            print(f"Warning: {attr} not found in metadata, skipping")
            continue

        # Get labels
        y_train = train_data["metadata"][attr]
        y_val = val_data["metadata"][attr]

        # Convert to tensor if needed
        if isinstance(y_train, list):
            y_train = torch.tensor(y_train)
        if isinstance(y_val, list):
            y_val = torch.tensor(y_val)

        # Determine number of classes
        n_classes = len(torch.unique(y_train))

        # Get generated labels if available
        y_generated_full = None
        X_generated_full = None
        if generated_full_data is not None and attr in generated_full_data["metadata"]:
            y_generated_full = generated_full_data["metadata"][attr]
            X_generated_full = generated_full_data["features"]
            if isinstance(y_generated_full, list):
                y_generated_full = torch.tensor(y_generated_full)

        y_generated_marginal = None
        X_generated_marginal = None
        if (
            generated_marginal_data is not None
            and attr in generated_marginal_data["metadata"]
        ):
            y_generated_marginal = generated_marginal_data["metadata"][attr]
            X_generated_marginal = generated_marginal_data["features"]
            if isinstance(y_generated_marginal, list):
                y_generated_marginal = torch.tensor(y_generated_marginal)

        print(f"\nTraining classifier for: {attr} ({n_classes} classes)")

        # For siRNA, we need to train per-class binary classifiers
        if attr == "sirna_id":
            sirna_results = {}
            unique_sirnas = torch.unique(y_train).tolist()

            print(f"  Training {len(unique_sirnas)} binary classifiers (one per siRNA)")
            for i, sirna in enumerate(unique_sirnas):
                # Create binary labels (one-vs-rest)
                y_train_binary = (y_train == sirna).long()
                y_val_binary = (y_val == sirna).long()

                y_gen_full_binary = None
                X_gen_full = None
                if y_generated_full is not None:
                    y_gen_full_binary = (y_generated_full == sirna).long()
                    X_gen_full = X_generated_full

                y_gen_marg_binary = None
                X_gen_marg = None
                if y_generated_marginal is not None:
                    y_gen_marg_binary = (y_generated_marginal == sirna).long()
                    X_gen_marg = X_generated_marginal

                # Train binary classifier using multiclass function with n_classes=2
                metrics = train_multiclass_classifier(
                    X_train,
                    y_train_binary,
                    X_val,
                    y_val_binary,
                    f"{attr}_{sirna}",
                    2,
                    X_gen_full,
                    y_gen_full_binary,
                    X_gen_marg,
                    y_gen_marg_binary,
                )
                sirna_results[sirna] = metrics

                if (i + 1) % 100 == 0:
                    print(
                        f"    Progress: {i + 1}/{len(unique_sirnas)} classifiers trained"
                    )

            all_results[attr] = sirna_results

            # Print top 5 and worst 5 for validation set
            sorted_by_acc = sorted(
                sirna_results.items(),
                key=lambda x: x[1]["val"]["accuracy"],
                reverse=True,
            )

            print(f"\n  Top 5 siRNAs (by validation accuracy):")
            for sirna, metrics in sorted_by_acc[:5]:
                print(
                    f"    siRNA {sirna}: Acc={metrics['val']['accuracy']:.4f}, F1={metrics['val']['f1_macro']:.4f}, AUC={metrics['val']['auc_roc']:.4f}"
                )

            print(f"\n  Worst 5 siRNAs (by validation accuracy):")
            for sirna, metrics in sorted_by_acc[-5:]:
                print(
                    f"    siRNA {sirna}: Acc={metrics['val']['accuracy']:.4f}, F1={metrics['val']['f1_macro']:.4f}, AUC={metrics['val']['auc_roc']:.4f}"
                )

            # Compute average metrics
            avg_val_acc = np.mean(
                [m["val"]["accuracy"] for m in sirna_results.values()]
            )
            avg_val_f1 = np.mean([m["val"]["f1_macro"] for m in sirna_results.values()])
            avg_val_auc = np.mean([m["val"]["auc_roc"] for m in sirna_results.values()])

            print(f"\n  Average siRNA metrics (validation):")
            print(
                f"    Acc={avg_val_acc:.4f}, F1={avg_val_f1:.4f}, AUC={avg_val_auc:.4f}"
            )

        else:
            # For cell_type_id, train a single multiclass classifier
            metrics = train_multiclass_classifier(
                X_train,
                y_train,
                X_val,
                y_val,
                attr,
                n_classes,
                X_generated_full,
                y_generated_full,
                X_generated_marginal,
                y_generated_marginal,
            )
            all_results[attr] = metrics

            # Print results
            print(
                f"  Train - Acc: {metrics['train']['accuracy']:.4f}, F1-macro: {metrics['train']['f1_macro']:.4f}, F1-weighted: {metrics['train']['f1_weighted']:.4f}, AUC: {metrics['train']['auc_roc']:.4f}"
            )
            print(
                f"  Val   - Acc: {metrics['val']['accuracy']:.4f}, F1-macro: {metrics['val']['f1_macro']:.4f}, F1-weighted: {metrics['val']['f1_weighted']:.4f}, AUC: {metrics['val']['auc_roc']:.4f}"
            )
            if "generated_full" in metrics:
                print(
                    f"  Gen-F - Acc: {metrics['generated_full']['accuracy']:.4f}, F1-macro: {metrics['generated_full']['f1_macro']:.4f}, F1-weighted: {metrics['generated_full']['f1_weighted']:.4f}, AUC: {metrics['generated_full']['auc_roc']:.4f}"
                )
            if "generated_marginal" in metrics:
                print(
                    f"  Gen-M - Acc: {metrics['generated_marginal']['accuracy']:.4f}, F1-macro: {metrics['generated_marginal']['f1_macro']:.4f}, F1-weighted: {metrics['generated_marginal']['f1_weighted']:.4f}, AUC: {metrics['generated_marginal']['auc_roc']:.4f}"
                )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Cell type summary
    if "cell_type_id" in all_results and not isinstance(
        all_results["cell_type_id"], dict
    ):
        print("\nCell Type Classification:")
        print("-" * 80)
        metrics = all_results["cell_type_id"]
        print(
            f"  Validation - Acc: {metrics['val']['accuracy']:.4f}, F1-macro: {metrics['val']['f1_macro']:.4f}, F1-weighted: {metrics['val']['f1_weighted']:.4f}, AUC: {metrics['val']['auc_roc']:.4f}"
        )

        if "generated_full" in metrics:
            print(
                f"  Gen (Full) - Acc: {metrics['generated_full']['accuracy']:.4f}, F1-macro: {metrics['generated_full']['f1_macro']:.4f}, F1-weighted: {metrics['generated_full']['f1_weighted']:.4f}, AUC: {metrics['generated_full']['auc_roc']:.4f}"
            )

        if "generated_marginal" in metrics:
            print(
                f"  Gen (Marg) - Acc: {metrics['generated_marginal']['accuracy']:.4f}, F1-macro: {metrics['generated_marginal']['f1_macro']:.4f}, F1-weighted: {metrics['generated_marginal']['f1_weighted']:.4f}, AUC: {metrics['generated_marginal']['auc_roc']:.4f}"
            )

    # siRNA summary
    if "sirna_id" in all_results and isinstance(all_results["sirna_id"], dict):
        sirna_results = all_results["sirna_id"]

        print("\nsiRNA Classification (Average over all siRNAs):")
        print("-" * 80)

        # Validation metrics
        avg_val_acc = np.mean([m["val"]["accuracy"] for m in sirna_results.values()])
        avg_val_f1 = np.mean([m["val"]["f1_macro"] for m in sirna_results.values()])
        avg_val_auc = np.mean([m["val"]["auc_roc"] for m in sirna_results.values()])
        print(
            f"  Validation - Acc: {avg_val_acc:.4f}, F1-macro: {avg_val_f1:.4f}, AUC: {avg_val_auc:.4f}"
        )

        # Generated full model metrics
        if any("generated_full" in m for m in sirna_results.values()):
            avg_gen_full_acc = np.mean(
                [
                    m["generated_full"]["accuracy"]
                    for m in sirna_results.values()
                    if "generated_full" in m
                ]
            )
            avg_gen_full_f1 = np.mean(
                [
                    m["generated_full"]["f1_macro"]
                    for m in sirna_results.values()
                    if "generated_full" in m
                ]
            )
            avg_gen_full_auc = np.mean(
                [
                    m["generated_full"]["auc_roc"]
                    for m in sirna_results.values()
                    if "generated_full" in m
                ]
            )
            print(
                f"  Gen (Full) - Acc: {avg_gen_full_acc:.4f}, F1-macro: {avg_gen_full_f1:.4f}, AUC: {avg_gen_full_auc:.4f}"
            )

        # Generated marginal model metrics
        if any("generated_marginal" in m for m in sirna_results.values()):
            avg_gen_marg_acc = np.mean(
                [
                    m["generated_marginal"]["accuracy"]
                    for m in sirna_results.values()
                    if "generated_marginal" in m
                ]
            )
            avg_gen_marg_f1 = np.mean(
                [
                    m["generated_marginal"]["f1_macro"]
                    for m in sirna_results.values()
                    if "generated_marginal" in m
                ]
            )
            avg_gen_marg_auc = np.mean(
                [
                    m["generated_marginal"]["auc_roc"]
                    for m in sirna_results.values()
                    if "generated_marginal" in m
                ]
            )
            print(
                f"  Gen (Marg) - Acc: {avg_gen_marg_acc:.4f}, F1-macro: {avg_gen_marg_f1:.4f}, AUC: {avg_gen_marg_auc:.4f}"
            )

    print("=" * 80)


if __name__ == "__main__":
    main()
