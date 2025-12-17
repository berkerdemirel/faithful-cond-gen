#!/usr/bin/env python3
"""
Analyze RxRx1 dataset and log the top 100 (cell type, perturbation) pairs by sample size.
"""
import os
import sys

import pandas as pd

# Add the src directory to the path
sys.path.insert(0, "/mnt/pvc/faithful-cond-gen/src")

from faithful_cond_gen.data.rxrx1 import RxRx1DataConfig, RxRx1DataModule


def main():
    # Use the proper data directory from config
    data_dir = "/mnt/pvc/AutoSync/data/rxrx1"

    # Create data config and module
    cfg = RxRx1DataConfig(
        data_dir=data_dir,
        batch_size=32,
        num_workers=4,
        val_size=0.1,
        seed=1337,
        rare_threshold=20,
    )

    # Initialize the data module
    print(f"Loading RxRx1 data from {data_dir}")
    data_module = RxRx1DataModule(cfg)

    # Access the metadata
    metadata = data_module.metadata

    print(f"Total samples in metadata: {len(metadata)}")

    # Filter to train + validation only (exclude test)
    train_val_metadata = metadata[
        metadata["train_index"] | metadata["val_index"]
    ].copy()

    print(f"Train + Val samples: {len(train_val_metadata)}")
    print(f"Test samples: {len(metadata[metadata['test_index']])}")
    print(f"Unique cell types: {train_val_metadata['cell_type'].nunique()}")
    print(f"Unique siRNA IDs: {train_val_metadata['sirna_id'].nunique()}")

    # Group by (cell_type, sirna_id) and count samples in train+val
    grouped = (
        train_val_metadata.groupby(["cell_type", "sirna_id"])
        .size()
        .reset_index(name="count")
    )

    # Sort by count in descending order
    grouped_sorted = grouped.sort_values("count", ascending=False)

    # Get top 100
    top_100 = grouped_sorted.head(100)

    # Filter metadata to only include the top 100 pairs
    top_100_pairs = set(zip(top_100["cell_type"], top_100["sirna_id"]))
    top_100_metadata = train_val_metadata[
        train_val_metadata.apply(
            lambda row: (row["cell_type"], row["sirna_id"]) in top_100_pairs, axis=1
        )
    ]

    # Calculate statistics for top 100 pairs
    top_100_total_samples = len(top_100_metadata)
    top_100_per_cell_type = (
        top_100_metadata.groupby("cell_type").size().sort_values(ascending=False)
    )
    top_100_per_perturbation = (
        top_100_metadata.groupby("sirna_id").size().sort_values(ascending=False)
    )

    print(f"\nTop 100 (cell type, perturbation) pairs:")
    print(f"Total unique pairs in train+val: {len(grouped)}")

    # Print statistics for TOP 100 pairs
    print(f"\n=== Statistics for TOP 100 pairs ===")
    print(f"Total samples in top 100 pairs: {top_100_total_samples}")
    print(f"\nPer cell type (in top 100):")
    for cell_type, count in top_100_per_cell_type.items():
        print(f"  {cell_type}: {count}")
    print(f"\nPer perturbation (in top 100, showing top 10):")
    for sirna_id, count in top_100_per_perturbation.head(10).items():
        print(f"  siRNA {sirna_id}: {count}")

    # Save to txt file
    output_file = "/mnt/pvc/faithful-cond-gen/top_100_cell_perturbation_pairs.txt"
    with open(output_file, "w") as f:
        f.write(
            "# Top 100 (cell type, perturbation) pairs by sample size (Train + Val only)\n"
        )
        f.write("# Format: <cell_type> <sirna_id> <nof_samples>\n\n")

        # Write statistics
        f.write("=== STATISTICS FOR TOP 100 PAIRS ===\n")
        f.write(f"Total samples in top 100 pairs: {top_100_total_samples}\n")
        f.write(f"Total unique pairs in train+val: {len(grouped)}\n")
        f.write(f"Number of pairs in top 100: {len(top_100)}\n\n")

        f.write("Per cell type (in top 100):\n")
        for cell_type, count in top_100_per_cell_type.items():
            f.write(f"  {cell_type}: {count}\n")
        f.write("\n")

        f.write("Per perturbation (in top 100, showing all):\n")
        for sirna_id, count in top_100_per_perturbation.items():
            f.write(f"  siRNA {sirna_id}: {count}\n")
        f.write("\n")
        f.write("=== TOP 100 PAIRS ===\n\n")

        for idx, row in top_100.iterrows():
            f.write(f"{row['cell_type']} {row['sirna_id']} {row['count']}\n")

    print(f"\nResults saved to: {output_file}")

    # Print preview
    print("\nPreview (first 10):")
    for idx, row in top_100.head(10).iterrows():
        print(f"{row['cell_type']} {row['sirna_id']} {row['count']}")

    # Generate YAML config for held_out_pairs
    # Need to convert cell_type to cell_type_id
    from faithful_cond_gen.data.rxrx1 import CELL_TYPE_TO_LABEL

    # Create held_out_pairs as (cell_type_id, sirna_id) tuples
    held_out_pairs = []
    for idx, row in top_100.iterrows():
        cell_type_id = CELL_TYPE_TO_LABEL[row["cell_type"]]
        sirna_id = int(row["sirna_id"])
        held_out_pairs.append((cell_type_id, sirna_id))

    # Save YAML config snippet
    yaml_output_file = "/mnt/pvc/faithful-cond-gen/top_100_held_out_pairs.yaml"
    with open(yaml_output_file, "w") as f:
        f.write("# Configuration snippet for held_out_pairs\n")
        f.write("# Copy this into your rxrx1 dataset config\n")
        f.write("# Format: [[cell_type_id, sirna_id], ...]\n")
        f.write("# cell_type_id mapping: HEPG2=0, HUVEC=1, RPE=2, U2OS=3\n\n")
        f.write("held_out_pairs:\n")
        for cell_type_id, sirna_id in held_out_pairs:
            f.write(f"  - [{cell_type_id}, {sirna_id}]\n")

    print(f"\nHeld-out pairs config saved to: {yaml_output_file}")
    print(f"Total held-out pairs: {len(held_out_pairs)}")
    print(
        f"\nTo use this, copy the held_out_pairs section to your rxrx1.yaml config file."
    )
    print("This will exclude these pairs from training and add them to validation.")


if __name__ == "__main__":
    main()
