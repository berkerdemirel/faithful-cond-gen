#!/usr/bin/env python3
"""
Train linear probes on cached RxRx1 features (PyTorch single-layer softmax classifier).

- One probe per attribute (e.g., cell_type_id, sirna_id)
- Model: nn.Linear(D, C) with softmax CE loss
- Standardize features using train mean/std (important for stable optimization)
- Early stopping on val loss (patience)
- Evaluates on train/val + optionally generated_full / generated_marginal
"""

import argparse
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed: int):
    """Set seed for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------
# I/O
# --------------------------
def load_cached_features(cache_path: Path):
    train_path = cache_path / "train_features.pt"
    val_path = cache_path / "val_features.pt"

    print(f"Loading features from {cache_path}")
    train_data = torch.load(train_path, weights_only=False)
    val_data = torch.load(val_path, weights_only=False)
    return train_data, val_data


def load_generated_features(cache_path: Path):
    features_path = cache_path / "generated_unpacked_images_features.pt"
    if not features_path.exists():
        return None
    print(f"Loading generated features from {cache_path}")
    return torch.load(features_path, weights_only=False)


# --------------------------
# Utils
# --------------------------
@torch.no_grad()
def standardize_fit(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mean, std for standardization. std clamped for numerical stability."""
    mu = X.mean(dim=0)
    sigma = X.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mu, sigma


@torch.no_grad()
def standardize_apply(
    X: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    return (X - mu) / sigma


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, list):
        return torch.tensor(x)
    raise TypeError(f"Unsupported type: {type(x)}")


def filter_data_by_metadata(
    features: torch.Tensor,
    metadata: Dict[str, any],
    filter_cell_types: Optional[list] = None,
    filter_sirna_ids: Optional[list] = None,
) -> Tuple[torch.Tensor, Dict[str, any]]:
    """
    Filter features and metadata based on cell_type_id and/or sirna_id.
    Returns filtered features and metadata.
    """
    mask = torch.ones(features.shape[0], dtype=torch.bool)

    if filter_cell_types is not None and "cell_type_id" in metadata:
        cell_types = to_tensor(metadata["cell_type_id"])
        cell_mask = torch.zeros_like(mask)
        for ct in filter_cell_types:
            cell_mask |= cell_types == ct
        mask &= cell_mask
        print(f"  After cell_type filter: {mask.sum().item()} samples")

    if filter_sirna_ids is not None and "sirna_id" in metadata:
        sirna_ids = to_tensor(metadata["sirna_id"])
        sirna_mask = torch.zeros_like(mask)
        for sid in filter_sirna_ids:
            sirna_mask |= sirna_ids == sid
        mask &= sirna_mask
        print(f"  After sirna_id filter: {mask.sum().item()} samples")

    if mask.sum() == features.shape[0]:
        return features, metadata

    # Apply mask
    mask_np = mask.cpu().numpy()  # Convert to numpy for consistent indexing
    filtered_features = features[mask]
    filtered_metadata = {}
    for k, v in metadata.items():
        # Check if it's an array-like structure with the right length
        if isinstance(v, (torch.Tensor, np.ndarray, list)):
            try:
                if len(v) == features.shape[0]:
                    # Handle both torch tensors and numpy arrays
                    if isinstance(v, torch.Tensor):
                        filtered_metadata[k] = v[mask]
                    elif isinstance(v, np.ndarray):
                        filtered_metadata[k] = v[mask_np]
                    else:  # list
                        filtered_metadata[k] = [
                            v[i] for i in range(len(v)) if mask_np[i]
                        ]
                else:
                    # Keep as-is if length doesn't match
                    filtered_metadata[k] = v
            except (TypeError, IndexError):
                # If anything goes wrong, keep the original value
                filtered_metadata[k] = v
        else:
            # Not an array-like structure (e.g., scalar, string, dict)
            filtered_metadata[k] = v

    return filtered_features, filtered_metadata


def ensure_contiguous_labels(
    y_train: torch.Tensor,
    y_val: torch.Tensor,
    y_gen_full: Optional[torch.Tensor] = None,
    y_gen_marg: Optional[torch.Tensor] = None,
):
    """
    Map arbitrary integer labels to contiguous [0..C-1] using labels seen in TRAIN.
    If val/gen contain unseen labels, they will be marked as -1 and ignored in metrics.
    """
    y_train = y_train.long()
    y_val = y_val.long()
    train_classes = torch.unique(y_train).cpu().tolist()
    train_classes_sorted = sorted(train_classes)
    class_to_idx = {c: i for i, c in enumerate(train_classes_sorted)}

    def map_y(y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if y is None:
            return None
        y = y.long().cpu()
        mapped = torch.full_like(y, fill_value=-1)
        # vectorized-ish mapping
        for c, i in class_to_idx.items():
            mapped[y == c] = i
        return mapped

    y_train_m = map_y(y_train)
    y_val_m = map_y(y_val)
    y_gen_full_m = map_y(y_gen_full) if y_gen_full is not None else None
    y_gen_marg_m = map_y(y_gen_marg) if y_gen_marg is not None else None
    n_classes = len(train_classes_sorted)

    return y_train_m, y_val_m, y_gen_full_m, y_gen_marg_m, n_classes


@torch.no_grad()
def predict_logits_in_batches(
    model: nn.Module, X: torch.Tensor, batch_size: int, device: torch.device
):
    model.eval()
    logits_all = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i : i + batch_size].to(device, non_blocking=True)
        logits_all.append(model(xb).cpu())
    return torch.cat(logits_all, dim=0)


@torch.no_grad()
def compute_per_class_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> np.ndarray:
    """Compute per-class accuracy. Returns array of shape (n_classes,) with NaN for classes without samples."""
    per_class_acc = np.full(n_classes, np.nan)
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_acc[c] = (y_pred[mask] == c).mean()
    return per_class_acc


@torch.no_grad()
def evaluate_probe(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    topk: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """
    y expected in [0..C-1] with possible -1 entries (ignored).
    Returns acc, f1 macro/weighted, top-k acc, and per-class accuracies.
    """
    # filter ignored labels
    mask = y >= 0
    if mask.sum().item() == 0:
        return {
            "accuracy": float("nan"),
            "f1_macro": float("nan"),
            "f1_weighted": float("nan"),
            "top1": float("nan"),
            "top5": float("nan"),
            "per_class_acc": None,
        }

    Xf = X[mask]
    yf = y[mask].cpu().numpy()

    logits = predict_logits_in_batches(model, Xf, batch_size=batch_size, device=device)
    y_pred = logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(yf, y_pred)
    f1_macro = f1_score(yf, y_pred, average="macro")
    f1_weighted = f1_score(yf, y_pred, average="weighted")

    # per-class accuracy
    n_classes = logits.shape[1]
    per_class_acc = compute_per_class_accuracy(yf, y_pred, n_classes)

    # top-k
    topk_out = {}
    for k in topk:
        if k <= logits.shape[1]:
            topk_pred = torch.topk(logits, k=k, dim=1).indices.cpu().numpy()
            hit = (topk_pred == yf[:, None]).any(axis=1).mean()
            topk_out[f"top{k}"] = float(hit)
        else:
            topk_out[f"top{k}"] = float("nan")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_acc": per_class_acc,
        **topk_out,
    }


def train_linear_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_classes: int,
    device: torch.device,
    *,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    batch_size: int = 4096,
    max_epochs: int = 200,
    patience: int = 20,
    min_delta: float = 1e-4,
    use_class_weights: bool = False,
    eval_batch_size: int = 8192,
) -> nn.Module:
    """
    Single-layer softmax classifier (linear probe), trained with AdamW + early stopping on val loss.
    """
    D = X_train.shape[1]
    model = nn.Linear(D, n_classes, bias=True).to(device)

    # optional class weights (helps if sirna is imbalanced)
    class_weights = None
    if use_class_weights:
        counts = torch.bincount(y_train[y_train >= 0], minlength=n_classes).float()
        w = counts.sum() / counts.clamp_min(1.0)
        w = w / w.mean()
        class_weights = w.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Build index list for valid train labels
    train_mask = y_train >= 0
    Xtr = X_train[train_mask]
    ytr = y_train[train_mask]

    val_mask = y_val >= 0
    Xva = X_val[val_mask]
    yva = y_val[val_mask]

    best_state = None
    best_val = float("inf")
    bad = 0

    for epoch in range(1, max_epochs + 1):
        model.train()

        # shuffle indices
        idx = torch.randperm(Xtr.shape[0])
        total_loss = 0.0
        n_seen = 0

        for i in range(0, Xtr.shape[0], batch_size):
            j = idx[i : i + batch_size]
            xb = Xtr[j].to(device, non_blocking=True)
            yb = ytr[j].to(device, non_blocking=True)

            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=class_weights)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            total_loss += float(loss.item()) * bs
            n_seen += bs

        # val loss
        model.eval()
        with torch.no_grad():
            val_logits = predict_logits_in_batches(
                model, Xva, batch_size=eval_batch_size, device=device
            )
            val_loss = F.cross_entropy(
                val_logits.to(device), yva.to(device), weight=class_weights
            ).item()

        train_loss = total_loss / max(1, n_seen)
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
            )

        if val_loss < (best_val - min_delta):
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop (best val_loss={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train PyTorch linear probes on cached RxRx1 features"
    )
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--feature_cache_root", type=str, default="feature_cache")
    parser.add_argument("--skip_generated", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # training hyperparams
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--use_class_weights", action="store_true")

    # siRNA-specific hyperparams (overrides defaults for sirna_id)
    parser.add_argument(
        "--sirna_lr",
        type=float,
        default=None,
        help="Learning rate for siRNA (default: use --lr)",
    )
    parser.add_argument(
        "--sirna_weight_decay",
        type=float,
        default=None,
        help="Weight decay for siRNA (default: use --weight_decay)",
    )

    # filtering options
    parser.add_argument(
        "--filter_cell_types",
        type=int,
        nargs="+",
        default=None,
        help="Filter to specific cell type IDs (space-separated list, e.g., --filter_cell_types 0 1 2)",
    )
    parser.add_argument(
        "--filter_sirna_ids",
        type=int,
        nargs="+",
        default=None,
        help="Filter to specific siRNA IDs (space-separated list, e.g., --filter_sirna_ids 10 20 30)",
    )

    # eval
    parser.add_argument("--eval_batch_size", type=int, default=8192)

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

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

    train_data, val_data = load_cached_features(cache_path)
    X_train = to_tensor(train_data["features"]).float()
    X_val = to_tensor(val_data["features"]).float()

    print(f"\nFeature dimension: {X_train.shape[1]}")
    print(f"Train samples (before filtering): {X_train.shape[0]}")
    print(f"Val samples (before filtering): {X_val.shape[0]}")

    # Apply filtering if requested
    if args.filter_cell_types is not None or args.filter_sirna_ids is not None:
        print(f"\nApplying filters:")
        if args.filter_cell_types is not None:
            print(f"  Cell types: {args.filter_cell_types}")
        if args.filter_sirna_ids is not None:
            print(f"  siRNA IDs: {args.filter_sirna_ids}")

        print(f"\nFiltering train split:")
        X_train, train_data["metadata"] = filter_data_by_metadata(
            X_train,
            train_data["metadata"],
            args.filter_cell_types,
            args.filter_sirna_ids,
        )

        print(f"Filtering val split:")
        X_val, val_data["metadata"] = filter_data_by_metadata(
            X_val,
            val_data["metadata"],
            args.filter_cell_types,
            args.filter_sirna_ids,
        )

        print(f"\nAfter filtering:")
        print(f"  Train samples: {X_train.shape[0]}")
        print(f"  Val samples: {X_val.shape[0]}")

    generated_full_data = None
    generated_marginal_data = None
    if not args.skip_generated:
        generated_full_data = load_generated_features(generated_full_path)
        if generated_full_data:
            X_gen_full_orig = to_tensor(generated_full_data["features"]).float()
            print(
                f"Generated (full) samples (before filtering): {X_gen_full_orig.shape[0]}"
            )

            # Apply filtering to generated data
            if args.filter_cell_types is not None or args.filter_sirna_ids is not None:
                print(f"Filtering generated (full) split:")
                X_gen_full_filtered, generated_full_data["metadata"] = (
                    filter_data_by_metadata(
                        X_gen_full_orig,
                        generated_full_data["metadata"],
                        args.filter_cell_types,
                        args.filter_sirna_ids,
                    )
                )
                generated_full_data["features"] = X_gen_full_filtered
                print(f"  After filtering: {X_gen_full_filtered.shape[0]} samples")
            else:
                generated_full_data["features"] = X_gen_full_orig

        generated_marginal_data = load_generated_features(generated_marginal_path)
        if generated_marginal_data:
            X_gen_marg_orig = to_tensor(generated_marginal_data["features"]).float()
            print(
                f"Generated (marginal) samples (before filtering): {X_gen_marg_orig.shape[0]}"
            )

            # Apply filtering to generated data
            if args.filter_cell_types is not None or args.filter_sirna_ids is not None:
                print(f"Filtering generated (marginal) split:")
                X_gen_marg_filtered, generated_marginal_data["metadata"] = (
                    filter_data_by_metadata(
                        X_gen_marg_orig,
                        generated_marginal_data["metadata"],
                        args.filter_cell_types,
                        args.filter_sirna_ids,
                    )
                )
                generated_marginal_data["features"] = X_gen_marg_filtered
                print(f"  After filtering: {X_gen_marg_filtered.shape[0]} samples")
            else:
                generated_marginal_data["features"] = X_gen_marg_orig

    # Standardize features (fit on train, apply everywhere)
    mu, sigma = standardize_fit(X_train)
    X_train = standardize_apply(X_train, mu, sigma)
    X_val = standardize_apply(X_val, mu, sigma)

    X_gen_full = None
    X_gen_marg = None
    if generated_full_data is not None:
        X_gen_full = standardize_apply(
            to_tensor(generated_full_data["features"]).float(), mu, sigma
        )
    if generated_marginal_data is not None:
        X_gen_marg = standardize_apply(
            to_tensor(generated_marginal_data["features"]).float(), mu, sigma
        )

    # attributes
    metadata_keys = list(train_data["metadata"].keys())
    attributes = [k for k in metadata_keys if k != "comp_category"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Training linear probes for: {attributes}")
    print(f"\nHyperparameters:")
    print(f"  LR: {args.lr}, Weight Decay: {args.weight_decay}")
    if args.sirna_lr is not None or args.sirna_weight_decay is not None:
        print(f"  siRNA overrides: LR={args.sirna_lr}, WD={args.sirna_weight_decay}")
    print(
        f"  Batch Size: {args.batch_size}, Max Epochs: {args.max_epochs}, Patience: {args.patience}"
    )
    print("=" * 80)

    all_results = {}

    for attr in attributes:
        if attr not in train_data["metadata"]:
            print(f"Warning: {attr} not found in metadata, skipping")
            continue

        y_train_raw = to_tensor(train_data["metadata"][attr])
        y_val_raw = to_tensor(val_data["metadata"][attr])

        y_gen_full_raw = None
        y_gen_marg_raw = None
        if generated_full_data is not None and attr in generated_full_data["metadata"]:
            y_gen_full_raw = to_tensor(generated_full_data["metadata"][attr])
        if (
            generated_marginal_data is not None
            and attr in generated_marginal_data["metadata"]
        ):
            y_gen_marg_raw = to_tensor(generated_marginal_data["metadata"][attr])

        # map labels to contiguous indices based on TRAIN classes
        y_train, y_val, y_gen_full, y_gen_marg, n_classes = ensure_contiguous_labels(
            y_train_raw, y_val_raw, y_gen_full_raw, y_gen_marg_raw
        )

        print(f"\n[{attr}] classes (train): {n_classes}")

        # Use attribute-specific hyperparams if available
        current_lr = args.lr
        current_wd = args.weight_decay
        if attr == "sirna_id":
            if args.sirna_lr is not None:
                current_lr = args.sirna_lr
                print(f"  Using siRNA-specific lr: {current_lr}")
            if args.sirna_weight_decay is not None:
                current_wd = args.sirna_weight_decay
                print(f"  Using siRNA-specific weight_decay: {current_wd}")

        # train probe
        model = train_linear_probe(
            X_train,
            y_train,
            X_val,
            y_val,
            n_classes,
            device,
            lr=current_lr,
            weight_decay=current_wd,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            use_class_weights=args.use_class_weights,
            eval_batch_size=args.eval_batch_size,
        )

        # evaluate
        results = {}
        results["train"] = evaluate_probe(
            model, X_train, y_train, args.eval_batch_size, device
        )
        results["val"] = evaluate_probe(
            model, X_val, y_val, args.eval_batch_size, device
        )

        if X_gen_full is not None and y_gen_full is not None:
            results["generated_full"] = evaluate_probe(
                model, X_gen_full, y_gen_full, args.eval_batch_size, device
            )
        if X_gen_marg is not None and y_gen_marg is not None:
            results["generated_marginal"] = evaluate_probe(
                model, X_gen_marg, y_gen_marg, args.eval_batch_size, device
            )

        all_results[attr] = results

        # Pretty print results
        print(f"\n  Results for [{attr}]:")
        print(f"  {'-' * 78}")

        for split_name, split_key in [
            ("Train       ", "train"),
            ("Val         ", "val"),
            ("Gen-Full    ", "generated_full"),
            ("Gen-Marginal", "generated_marginal"),
        ]:
            if split_key in results:
                r = results[split_key]
                print(
                    f"  {split_name}: "
                    f"Acc={r['accuracy']:.4f}  "
                    f"F1-macro={r['f1_macro']:.4f}  "
                    f"F1-weighted={r['f1_weighted']:.4f}  "
                    f"Top1={r['top1']:.4f}  "
                    f"Top5={r['top5']:.4f}"
                )

        # Show per-class analysis for evaluation splits
        print(f"\n  Per-class analysis (top/bottom 5 classes by accuracy):")
        for split_name, split_key in [
            ("Val", "val"),
            ("Gen-Full", "generated_full"),
            ("Gen-Marginal", "generated_marginal"),
        ]:
            if split_key in results and results[split_key]["per_class_acc"] is not None:
                per_class = results[split_key]["per_class_acc"]
                valid_mask = ~np.isnan(per_class)

                if valid_mask.sum() > 0:
                    valid_classes = np.where(valid_mask)[0]
                    valid_accs = per_class[valid_mask]

                    # Sort by accuracy
                    sorted_idx = np.argsort(valid_accs)

                    print(f"\n    [{split_name}]")

                    # Top 5 classes
                    top_n = min(5, len(sorted_idx))
                    print(f"      Best classes:")
                    for i in range(-1, -top_n - 1, -1):
                        cls_id = valid_classes[sorted_idx[i]]
                        acc = valid_accs[sorted_idx[i]]
                        print(f"        Class {cls_id}: {acc:.4f}")

                    # Bottom 5 classes
                    bottom_n = min(5, len(sorted_idx))
                    print(f"      Worst classes:")
                    for i in range(bottom_n):
                        cls_id = valid_classes[sorted_idx[i]]
                        acc = valid_accs[sorted_idx[i]]
                        print(f"        Class {cls_id}: {acc:.4f}")

    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    # Cell Type Summary
    if "cell_type_id" in all_results:
        print("\n[CELL TYPE]")
        res = all_results["cell_type_id"]
        for split_name, split_key in [
            ("Train       ", "train"),
            ("Val         ", "val"),
            ("Gen-Full    ", "generated_full"),
            ("Gen-Marginal", "generated_marginal"),
        ]:
            if split_key in res:
                r = res[split_key]
                print(f"  {split_name}: Acc={r['accuracy']:.4f}")

    # siRNA Summary
    if "sirna_id" in all_results:
        print("\n[SIRNA]")
        res = all_results["sirna_id"]
        for split_name, split_key in [
            ("Train       ", "train"),
            ("Val         ", "val"),
            ("Gen-Full    ", "generated_full"),
            ("Gen-Marginal", "generated_marginal"),
        ]:
            if split_key in res:
                r = res[split_key]
                print(
                    f"  {split_name}: "
                    f"Acc={r['accuracy']:.4f}  "
                    f"Top5={r['top5']:.4f}"
                )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
