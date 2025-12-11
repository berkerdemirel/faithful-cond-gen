#!/usr/bin/env python
import argparse
import ast
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from faithful_cond_gen.data.rxrx1 import (
    CELL_TYPE_TO_LABEL,
    RxRx1DataConfig,
    RxRx1DataModule,
)
from PIL import Image
from torchvision.transforms import functional as TF


def _load_png_stack(image_paths: List[str]) -> np.ndarray:
    imgs = [np.array(Image.open(p), dtype=np.uint8) for p in image_paths]
    stacked = np.stack(imgs, axis=0)  # (C, H, W)
    return stacked


def _load_npy(path: str) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    return np.array(arr)


def _load_parquet(path: str, shape: Tuple[int, int, int]) -> np.ndarray:
    table = pq.read_table(path)
    flat = table["data"].to_numpy()
    arr = np.array(flat).reshape(shape)
    return arr


def test_io_consistency(
    data_dir: str,
    max_samples: int = 5,
) -> None:
    """Read same samples as PNG, NPY, Parquet and check they match."""
    md_path = os.path.join(data_dir, "metadata_extended.csv")
    if not os.path.exists(md_path):
        raise FileNotFoundError(
            f"metadata_extended.csv not found at {md_path}. "
            "Run prepare_rxrx1_metadata first."
        )

    md = pd.read_csv(md_path)

    if "image_paths" not in md.columns:
        raise ValueError("metadata_extended.csv missing 'image_paths' column.")
    if "numpy_path" not in md.columns:
        raise ValueError("metadata_extended.csv missing 'numpy_path' column.")
    if "parquet_path" not in md.columns:
        raise ValueError("metadata_extended.csv missing 'parquet_path' column.")

    # Filter rows that have both numpy and parquet paths
    mask = (
        md["numpy_path"].notna()
        & (md["numpy_path"] != "")
        & md["parquet_path"].notna()
        & (md["parquet_path"] != "")
    )
    subset = md[mask]

    if subset.empty:
        raise RuntimeError(
            "No rows have both numpy_path and parquet_path. "
            "Run prepare_rxrx1_metadata with --save-parquet."
        )

    subset = subset.head(max_samples)

    print(f"[test_io_consistency] Testing {len(subset)} samples...")

    for i, row in subset.iterrows():
        paths = row["image_paths"]
        if isinstance(paths, str):
            paths = ast.literal_eval(paths)
        png_arr = _load_png_stack(paths)

        npy_path = row["numpy_path"]
        parquet_path = row["parquet_path"]

        npy_arr = _load_npy(npy_path)
        parquet_arr = _load_parquet(parquet_path, png_arr.shape)

        assert png_arr.shape == npy_arr.shape == parquet_arr.shape, (
            f"Shape mismatch for index {i}: "
            f"png {png_arr.shape}, npy {npy_arr.shape}, parquet {parquet_arr.shape}"
        )

        if not np.array_equal(png_arr, npy_arr):
            max_diff = np.abs(png_arr.astype(int) - npy_arr.astype(int)).max()
            raise AssertionError(
                f"PNG vs NPY differ for index {i}, max diff={max_diff}"
            )

        if not np.array_equal(png_arr, parquet_arr):
            max_diff = np.abs(png_arr.astype(int) - parquet_arr.astype(int)).max()
            raise AssertionError(
                f"PNG vs Parquet differ for index {i}, max diff={max_diff}"
            )

    print("[test_io_consistency] PNG, NPY, Parquet all match for tested samples.")


def test_split_leakage(
    data_dir: str,
    val_size: float = 0.1,
) -> None:
    """Create DataModule and check that train/val/test sets do not overlap."""
    cfg = RxRx1DataConfig(
        data_dir=data_dir,
        val_size=val_size,
        use_numpy=True,
        use_parquet=False,
    )
    dm = RxRx1DataModule(cfg)
    md = dm.metadata

    train_idx = set(md.index[md["train_index"]].tolist())
    val_idx = set(md.index[md["val_index"]].tolist())
    test_idx = set(md.index[md["test_index"]].tolist())

    inter_train_val = train_idx & val_idx
    inter_train_test = train_idx & test_idx
    inter_val_test = val_idx & test_idx

    assert (
        len(inter_train_val) == 0
    ), f"Train/Val leakage: {len(inter_train_val)} overlapping indices."
    assert (
        len(inter_train_test) == 0
    ), f"Train/Test leakage: {len(inter_train_test)} overlapping indices."
    assert (
        len(inter_val_test) == 0
    ), f"Val/Test leakage: {len(inter_val_test)} overlapping indices."

    print(
        f"[test_split_leakage] No leakage for val_size={val_size}. "
        f"Counts: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )


def save_control_examples(
    data_dir: str,
    sirna_id: int = 1138,
    samples_per_cell_type: int = 4,
    resize: Tuple[int, int] = (512, 512),
) -> None:
    """Save a few RGB examples for each cell type with given sirna_id."""
    cfg = RxRx1DataConfig(
        data_dir=data_dir,
        img_size=(512, 512),
        resize=resize,
        reduce_channels=True,
        augment_train=False,
        normalize=False,
        use_numpy=True,
        use_parquet=False,
        batch_size=1,
        num_workers=0,
        val_size=0.1,
    )
    dm = RxRx1DataModule(cfg)

    ds = dm.get_dataset(
        split="train",
        cell_types=None,
        perturbations=[sirna_id],
        override_cfg={
            "reduce_channels": True,
            "augment": False,
            "normalize": False,
            "resize": resize,
        },
    )

    label_to_cell = {v: k for k, v in CELL_TYPE_TO_LABEL.items()}

    # Save into ./test_outputs/rxrx1_sirna_<id>
    out_root = os.path.join(os.getcwd(), "test_outputs")
    os.makedirs(out_root, exist_ok=True)
    out_dir = os.path.join(out_root, f"rxrx1_sirna_{sirna_id}")
    os.makedirs(out_dir, exist_ok=True)

    saved_counts: Dict[int, int] = {cid: 0 for cid in CELL_TYPE_TO_LABEL.values()}

    print(
        f"[save_control_examples] Saving up to {samples_per_cell_type} samples "
        f"per cell type for sirna_id={sirna_id} into {out_dir}"
    )

    for x, cond in ds:
        # x: (C, H, W) float in [0, 1] from dataset (after to_rgb + scaling)
        cell_type_id = int(cond["cell_type_id"].item())
        if cell_type_id not in saved_counts:
            saved_counts[cell_type_id] = 0

        if saved_counts[cell_type_id] >= samples_per_cell_type:
            # Already have enough for this cell type
            if all(c >= samples_per_cell_type for c in saved_counts.values()):
                break
            else:
                continue

        img = x.squeeze(0) if x.ndim == 4 else x
        img = img.detach().cpu()
        img = img.clamp(0.0, 1.0)
        img = (img * 255.0).to(torch.uint8)

        pil_img = TF.to_pil_image(img)

        cell_name = label_to_cell.get(cell_type_id, f"cell{cell_type_id}")
        idx = saved_counts[cell_type_id]
        fname = f"{cell_name}_sirna{sirna_id}_{idx}.png"
        pil_img.save(os.path.join(out_dir, fname))

        saved_counts[cell_type_id] += 1

    print("[save_control_examples] Saved counts per cell type:")
    for cid, count in saved_counts.items():
        cell_name = label_to_cell.get(cid, f"cell{cid}")
        print(f"  {cell_name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="RxRx1 dataset tests")
    parser.add_argument(
        "--data-dir",
        help="Root RxRx1 data directory",
        default="/mnt/pvc/AutoSync/data/rxrx1",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation fraction for split leakage test",
    )
    parser.add_argument(
        "--samples-per-cell-type",
        type=int,
        default=4,
        help="Number of control samples to save per cell type",
    )
    parser.add_argument(
        "--sirna-id",
        type=int,
        default=1138,
        help="Control sirna_id to visualize",
    )
    args = parser.parse_args()

    data_dir = args.data_dir

    # 1) IO consistency: PNG vs NPY vs Parquet
    test_io_consistency(data_dir=data_dir, max_samples=5)

    # 2) Split leakage
    test_split_leakage(data_dir=data_dir, val_size=args.val_size)

    # 3) Save control examples for visual inspection
    save_control_examples(
        data_dir=data_dir,
        sirna_id=args.sirna_id,
        samples_per_cell_type=args.samples_per_cell_type,
    )


if __name__ == "__main__":
    main()
