#!/usr/bin/env python
import argparse
import ast
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from faithful_cond_gen.data.rxrx1 import CELL_TYPE_TO_LABEL
from PIL import Image


def prepare_rxrx1_metadata(
    data_dir: str,
    save_numpy: bool = True,
    save_parquet: bool = False,
    img_size: Tuple[int, int] = (512, 512),
    overwrite: bool = False,
) -> None:
    """Prepare metadata_extended.csv for RxRx1.

    - Reads metadata.csv in `data_dir`.
    - Constructs image_paths for each row (6 PNG channels).
    - Optionally saves stacked images as .npy and/or Parquet.
    - Writes metadata_extended.csv with image_paths, numpy_path, parquet_path.

    Directory layout assumed (standard RxRx1):
      data_dir/
        ├─ metadata.csv
        ├─ images/
        │    └─ {experiment}/{PlateX}/{well}_s{site}_w{1..6}.png
        ├─ numpy_images/        (created if save_numpy=True)
        └─ parquet_data/        (created if save_parquet=True)
    """
    metadata_path = os.path.join(data_dir, "metadata.csv")
    extended_path = os.path.join(data_dir, "metadata_extended.csv")
    images_root = os.path.join(data_dir, "images")
    numpy_root = os.path.join(data_dir, "numpy_images")
    parquet_root = os.path.join(data_dir, "parquet_data")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")

    # If extended already exists and not overwriting, just bail out.
    if os.path.exists(extended_path) and not overwrite:
        print(f"[prepare_rxrx1_metadata] {extended_path} already exists. Skipping.")
        return

    md = pd.read_csv(metadata_path)

    required_cols = ["experiment", "plate", "well", "site", "sirna_id", "cell_type"]
    missing = [c for c in required_cols if c not in md.columns]
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {missing}")

    # Add cell_type_id if missing
    if "cell_type_id" not in md.columns:
        md["cell_type_id"] = md["cell_type"].map(CELL_TYPE_TO_LABEL)

    # Construct image_paths list for each row
    def construct_image_paths(row: pd.Series) -> List[str]:
        cell_type_batch = row["experiment"]
        plate = f"Plate{row['plate']}"
        paths = [
            os.path.join(
                images_root,
                cell_type_batch,
                plate,
                f"{row['well']}_s{row['site']}_w{w}.png",
            )
            for w in range(1, 7)
        ]
        return paths

    md["image_paths"] = md.apply(construct_image_paths, axis=1)

    # Ensure all image paths exist
    all_paths = [p for paths in md["image_paths"] for p in paths]
    missing_paths = [p for p in all_paths if not os.path.exists(p)]
    if missing_paths:
        # Keep the first few for debug; otherwise it's huge
        preview = "\n  ".join(missing_paths[:10])
        raise FileNotFoundError(
            f"Some image files are missing (showing up to 10):\n  {preview}\n"
            f"Total missing: {len(missing_paths)}"
        )

    # Optionally create numpy and parquet directories
    if save_numpy:
        os.makedirs(numpy_root, exist_ok=True)
    if save_parquet:
        os.makedirs(parquet_root, exist_ok=True)

    # ---- Save .npy and/or Parquet per sample ----
    numpy_paths: List[Optional[str]] = [None] * len(md)
    parquet_paths: List[Optional[str]] = [None] * len(md)

    for idx, row in md.iterrows():
        cell_type_batch = row["experiment"]
        plate = f"Plate{row['plate']}"
        well = row["well"]
        site = row["site"]

        # Load all 6 channel images once
        paths = row["image_paths"]
        # if coming from a CSV string (in case of re-run with overwrite=True)
        if isinstance(paths, str):
            paths = ast.literal_eval(paths)
        imgs = [np.array(Image.open(p), dtype=np.uint8) for p in paths]
        stacked = np.stack(imgs, axis=0)  # (6, H, W)

        if save_numpy:
            sample_dir = os.path.join(numpy_root, cell_type_batch, plate)
            os.makedirs(sample_dir, exist_ok=True)
            numpy_file_path = os.path.join(sample_dir, f"{well}_s{site}.npy")
            numpy_paths[idx] = numpy_file_path

            if overwrite or not os.path.exists(numpy_file_path):
                np.save(numpy_file_path, stacked)

        if save_parquet:
            sample_dir = os.path.join(parquet_root, cell_type_batch, plate)
            os.makedirs(sample_dir, exist_ok=True)
            filename = f"{cell_type_batch}_{plate}_{well}_s{site}.parquet"
            parquet_file_path = os.path.join(sample_dir, filename)
            parquet_paths[idx] = parquet_file_path

            if overwrite or not os.path.exists(parquet_file_path):
                table = pa.Table.from_arrays(
                    [pa.array(stacked.reshape(-1))], names=["data"]
                )
                pq.write_table(table, parquet_file_path)

    if save_numpy:
        md["numpy_path"] = numpy_paths
    if save_parquet:
        md["parquet_path"] = parquet_paths

    md.to_csv(extended_path, index=False)
    print(f"[prepare_rxrx1_metadata] Wrote {extended_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare RxRx1 metadata_extended.csv")
    parser.add_argument("--data-dir", required=True, help="Root RxRx1 data directory")
    parser.add_argument(
        "--no-numpy", action="store_true", help="Do NOT save .npy files"
    )
    parser.add_argument(
        "--save-parquet", action="store_true", help="Also save Parquet files"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing metadata_extended / numpy / parquet",
    )
    args = parser.parse_args()

    save_numpy = not args.no_numpy
    save_parquet = args.save_parquet

    prepare_rxrx1_metadata(
        data_dir=args.data_dir,
        save_numpy=save_numpy,
        save_parquet=save_parquet,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
