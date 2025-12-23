"""Train linear classifiers on cached features to evaluate feature quality."""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


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
    features_path = cache_path / "generated_images_features.pt"

    if not features_path.exists():
        return None

    print(f"Loading generated features from {cache_path}")
    data = torch.load(features_path, weights_only=False)
    return data


def evaluate_classifier(clf, X, y):
    """Evaluate a trained classifier on a dataset."""
    # Convert to numpy if needed
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    # Predict
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)[:, 1]

    # Compute metrics
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_proba)
    pos_ratio = y.mean()

    return {
        "accuracy": acc,
        "f1_score": f1,
        "auc_roc": auc,
        "pos_ratio": pos_ratio,
    }


def train_linear_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    attribute_name: str,
    X_generated_full=None,
    y_generated_full=None,
    X_generated_marginal=None,
    y_generated_marginal=None,
):
    """Train a linear classifier for a single attribute and evaluate on all datasets."""
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
    results["train"] = evaluate_classifier(clf, X_train, y_train)

    # Validation set
    results["val"] = evaluate_classifier(clf, X_val, y_val)

    # Generated full model
    if X_generated_full is not None:
        results["generated_full"] = evaluate_classifier(
            clf, X_generated_full, y_generated_full
        )

    # Generated marginal model
    if X_generated_marginal is not None:
        results["generated_marginal"] = evaluate_classifier(
            clf, X_generated_marginal, y_generated_marginal
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train linear probes on cached features"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="celeba",
        help="Dataset name (e.g., celeba, rxrx1)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Encoder name (e.g., dinov2, dinov3, mae, siglip, bioclip)",
    )
    parser.add_argument(
        "--attributes",
        type=str,
        nargs="+",
        default=None,
        help="List of attributes to train classifiers for (default: all in metadata except comp_category)",
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
    feature_cache_root = Path(args.feature_cache_root)
    cache_path = feature_cache_root / "real_samples" / args.dataset / args.encoder
    generated_full_path = (
        feature_cache_root
        / "generated_samples"
        / args.dataset
        / "fullmodel"
        / args.encoder
    )
    generated_marginal_path = (
        feature_cache_root
        / "generated_samples"
        / args.dataset
        / "marginalmodel"
        / args.encoder
    )

    print(f"\nDataset: {args.dataset}")
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
    if args.attributes is not None:
        attributes = args.attributes
    else:
        # Exclude comp_category if present (it's an auxiliary key)
        attributes = [k for k in metadata_keys if k != "comp_category"]

    print(f"\nTraining linear classifiers for: {attributes}")
    print("=" * 80)

    # Train a classifier for each attribute
    results = {}
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

        # Train classifier
        print(f"\nTraining classifier for: {attr}")
        metrics = train_linear_classifier(
            X_train,
            y_train,
            X_val,
            y_val,
            attr,
            X_generated_full,
            y_generated_full,
            X_generated_marginal,
            y_generated_marginal,
        )
        results[attr] = metrics

        # Print results
        print(
            f"  Train - Acc: {metrics['train']['accuracy']:.4f}, F1: {metrics['train']['f1_score']:.4f}, AUC: {metrics['train']['auc_roc']:.4f}, Pos: {metrics['train']['pos_ratio']:.4f}"
        )
        print(
            f"  Val   - Acc: {metrics['val']['accuracy']:.4f}, F1: {metrics['val']['f1_score']:.4f}, AUC: {metrics['val']['auc_roc']:.4f}, Pos: {metrics['val']['pos_ratio']:.4f}"
        )
        if "generated_full" in metrics:
            print(
                f"  Gen-F - Acc: {metrics['generated_full']['accuracy']:.4f}, F1: {metrics['generated_full']['f1_score']:.4f}, AUC: {metrics['generated_full']['auc_roc']:.4f}, Pos: {metrics['generated_full']['pos_ratio']:.4f}"
            )
        if "generated_marginal" in metrics:
            print(
                f"  Gen-M - Acc: {metrics['generated_marginal']['accuracy']:.4f}, F1: {metrics['generated_marginal']['f1_score']:.4f}, AUC: {metrics['generated_marginal']['auc_roc']:.4f}, Pos: {metrics['generated_marginal']['pos_ratio']:.4f}"
            )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY - VALIDATION SET")
    print("=" * 80)
    print(f"{'Attribute':<20} {'Accuracy':<12} {'F1 Score':<12} {'AUC-ROC':<12}")
    print("-" * 80)
    for attr, metrics in results.items():
        print(
            f"{attr:<20} {metrics['val']['accuracy']:<12.4f} {metrics['val']['f1_score']:<12.4f} {metrics['val']['auc_roc']:<12.4f}"
        )

    # Compute average metrics
    avg_val_acc = np.mean([m["val"]["accuracy"] for m in results.values()])
    avg_val_f1 = np.mean([m["val"]["f1_score"] for m in results.values()])
    avg_val_auc = np.mean([m["val"]["auc_roc"] for m in results.values()])

    print("-" * 80)
    print(
        f"{'AVERAGE':<20} {avg_val_acc:<12.4f} {avg_val_f1:<12.4f} {avg_val_auc:<12.4f}"
    )
    print("=" * 80)

    # Print generated samples summary if available
    if any("generated_full" in m for m in results.values()):
        print("\n" + "=" * 80)
        print("SUMMARY - GENERATED SAMPLES (FULL MODEL)")
        print("=" * 80)
        print(f"{'Attribute':<20} {'Accuracy':<12} {'F1 Score':<12} {'AUC-ROC':<12}")
        print("-" * 80)
        for attr, metrics in results.items():
            if "generated_full" in metrics:
                print(
                    f"{attr:<20} {metrics['generated_full']['accuracy']:<12.4f} {metrics['generated_full']['f1_score']:<12.4f} {metrics['generated_full']['auc_roc']:<12.4f}"
                )

        avg_gen_full_acc = np.mean(
            [
                m["generated_full"]["accuracy"]
                for m in results.values()
                if "generated_full" in m
            ]
        )
        avg_gen_full_f1 = np.mean(
            [
                m["generated_full"]["f1_score"]
                for m in results.values()
                if "generated_full" in m
            ]
        )
        avg_gen_full_auc = np.mean(
            [
                m["generated_full"]["auc_roc"]
                for m in results.values()
                if "generated_full" in m
            ]
        )

        print("-" * 80)
        print(
            f"{'AVERAGE':<20} {avg_gen_full_acc:<12.4f} {avg_gen_full_f1:<12.4f} {avg_gen_full_auc:<12.4f}"
        )
        print("=" * 80)

    if any("generated_marginal" in m for m in results.values()):
        print("\n" + "=" * 80)
        print("SUMMARY - GENERATED SAMPLES (MARGINAL MODEL)")
        print("=" * 80)
        print(f"{'Attribute':<20} {'Accuracy':<12} {'F1 Score':<12} {'AUC-ROC':<12}")
        print("-" * 80)
        for attr, metrics in results.items():
            if "generated_marginal" in metrics:
                print(
                    f"{attr:<20} {metrics['generated_marginal']['accuracy']:<12.4f} {metrics['generated_marginal']['f1_score']:<12.4f} {metrics['generated_marginal']['auc_roc']:<12.4f}"
                )

        avg_gen_marginal_acc = np.mean(
            [
                m["generated_marginal"]["accuracy"]
                for m in results.values()
                if "generated_marginal" in m
            ]
        )
        avg_gen_marginal_f1 = np.mean(
            [
                m["generated_marginal"]["f1_score"]
                for m in results.values()
                if "generated_marginal" in m
            ]
        )
        avg_gen_marginal_auc = np.mean(
            [
                m["generated_marginal"]["auc_roc"]
                for m in results.values()
                if "generated_marginal" in m
            ]
        )

        print("-" * 80)
        print(
            f"{'AVERAGE':<20} {avg_gen_marginal_acc:<12.4f} {avg_gen_marginal_f1:<12.4f} {avg_gen_marginal_auc:<12.4f}"
        )
        print("=" * 80)


if __name__ == "__main__":
    main()
