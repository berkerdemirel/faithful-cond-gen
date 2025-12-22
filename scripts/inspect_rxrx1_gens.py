import argparse
import os

import numpy as np
import torch
from faithful_cond_gen.data.rxrx1 import to_rgb
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Convert RxRx1 .pt batch to PNGs")
    parser.add_argument(
        "file", type=str, help="Path to the .pt file (e.g., cell0_sirna669_batch4.pt)"
    )
    parser.add_argument(
        "--out", type=str, default="inspect_output", help="Directory to save PNGs"
    )
    parser.add_argument("--count", type=int, default=8, help="Number of images to save")
    args = parser.parse_args()

    # 1. Load File
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        return

    print(f"Loading {args.file}...")
    batch_tensor = torch.load(args.file, map_location="cpu")

    # Ensure it's (B, C, H, W)
    if batch_tensor.dim() == 3:
        batch_tensor = batch_tensor.unsqueeze(0)

    print(f"Tensor shape: {batch_tensor.shape}")

    # 2. Convert to RGB
    print("Converting to RGB...")
    rgb_batch = to_rgb(batch_tensor)  # Returns (B, 3, H, W) in [0, 1]

    # 3. Save Images
    os.makedirs(args.out, exist_ok=True)
    num_to_save = min(args.count, len(rgb_batch))

    # Get base filename without extension
    basename = os.path.splitext(os.path.basename(args.file))[0]

    for i in range(num_to_save):
        # Grab image i: (3, H, W) -> (H, W, 3)
        img_tensor = rgb_batch[i].permute(1, 2, 0)

        # Scale to 0-255 uint8
        img_np = (img_tensor.numpy() * 255).astype(np.uint8)

        pil_img = Image.fromarray(img_np)

        save_path = os.path.join(args.out, f"{basename}_sample_{i}.png")
        pil_img.save(save_path)
        print(f"Saved: {save_path}")

    print("Done.")


if __name__ == "__main__":
    main()
