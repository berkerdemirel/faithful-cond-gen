import argparse
import glob
import os

import numpy as np
import torch
from tqdm import tqdm


def unpack_batches(input_dir, output_dir, format="pt", delete_originals=False):
    """
    Reads batch .pt files and saves individual samples.

    Args:
        input_dir: Directory containing the _batchX.pt files.
        output_dir: Directory where individual files will be saved.
        format: "pt" for torch.save or "npy" for numpy.save.
        delete_originals: If True, deletes the batch file after successful unpacking.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Pattern to match your existing files: e.g., cell3_sirna1137_batch6.pt
    pattern = os.path.join(input_dir, "*_batch*.pt")
    batch_files = sorted(glob.glob(pattern))

    print(f"Found {len(batch_files)} batch files in {input_dir}")
    print(f"Unpacking to {output_dir} in format: {format}")

    # Track global count to ensure unique IDs across batches for the same condition
    # key: "cellX_sirnaY", value: current_sample_count
    counters = {}

    for file_path in tqdm(batch_files, desc="Unpacking"):
        try:
            # 1. Parse filename to get condition signature
            filename = os.path.basename(file_path)
            # Example: "cell3_sirna1137_batch6.pt"
            # Split by '_' -> ["cell3", "sirna1137", "batch6.pt"]
            parts = filename.split("_")

            # Reconstruct signature: "cell3_sirna1137"
            # We assume the first two parts are always the condition identifiers
            signature = f"{parts[0]}_{parts[1]}"

            if signature not in counters:
                counters[signature] = 0

            # 2. Load the batch
            # map_location='cpu' prevents OOM if running on a small machine
            batch_tensor = torch.load(file_path, map_location="cpu")

            # Ensure shape is (B, 6, H, W)
            if batch_tensor.dim() == 3:
                # Handle edge case where a batch of 1 was saved as (6,H,W)
                batch_tensor = batch_tensor.unsqueeze(0)

            # 3. Iterate and save individual samples
            batch_size = batch_tensor.shape[0]

            for i in range(batch_size):
                sample = batch_tensor[i]  # (6, H, W)

                # Create consistent name: cell3_sirna1137_0.pt, cell3_sirna1137_1.pt, etc.
                global_idx = counters[signature]

                if format == "npy":
                    out_name = f"{signature}_{global_idx}.npy"
                    out_path = os.path.join(output_dir, out_name)
                    # Convert to numpy and save
                    np.save(out_path, sample.numpy())
                else:
                    out_name = f"{signature}_{global_idx}.pt"
                    out_path = os.path.join(output_dir, out_name)
                    # Save as torch tensor (preserves gradients/device info if needed, usually cleaner for pytorch workflows)
                    torch.save(sample.clone(), out_path)

                counters[signature] += 1

            # Optional: Delete original batch file to save space
            if delete_originals:
                os.remove(file_path)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("Unpacking complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to folder with batch .pt files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save unpacked files"
    )
    parser.add_argument(
        "--format", type=str, default="pt", choices=["pt", "npy"], help="Output format"
    )
    parser.add_argument(
        "--delete_original",
        action="store_true",
        help="Delete batch files after unpacking (Use with caution!)",
    )

    args = parser.parse_args()

    unpack_batches(args.input_dir, args.output_dir, args.format, args.delete_original)
