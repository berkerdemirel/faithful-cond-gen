"""
Extract DINOv3 meanpatch features from generated images.

Usage:
    PYTHONPATH=src uv run python scripts/extract_gen_features.py \
        --model rxrx1_repa_full --batch-size 32
"""

import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from faithful_cond_gen.model.repa_encoder import REPAEncoder


def to_rgb(batch: torch.Tensor) -> torch.Tensor:
    """Convert 6-channel to 3-channel RGB for RxRx1."""
    if batch.shape[1] != 6:
        return batch
    # Use first 3 channels (RGB mapping)
    return batch[:, :3, :, :]


class GeneratedImageDataset(Dataset):
    """Dataset for loading generated .pt images."""

    def __init__(self, image_dir: Path):
        self.image_dir = image_dir
        self.image_files = sorted(image_dir.glob("*.pt"))
        if not self.image_files:
            self.image_files = sorted(image_dir.glob("*.png"))
        print(f"Found {len(self.image_files)} images in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        if img_path.suffix == ".pt":
            img = torch.load(img_path, map_location="cpu")
        else:
            from PIL import Image
            import torchvision.transforms as T
            img = Image.open(img_path)
            img = T.ToTensor()(img)

        # Ensure correct shape (C, H, W)
        if img.ndim == 2:
            img = img.unsqueeze(0)

        return img, str(img_path.name)


def extract_meanpatch_features(
    image_dir: Path,
    output_path: Path,
    batch_size: int = 32,
    encoder_name: str = "dinov3-vit-l",
    resolution: int = 256,
    device_id: int = 0,
):
    """Extract DINOv3 meanpatch features from generated images."""

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (GPU {device_id})")

    # Initialize REPAEncoder (returns patch tokens)
    print(f"Initializing {encoder_name}...")
    encoder = REPAEncoder(
        encoder_name=encoder_name,
        resolution=resolution,
        in_channels=3,
        target_grid=16,
        device=str(device),
    )
    encoder.eval()

    # Create dataset and loader
    dataset = GeneratedImageDataset(image_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Extract features
    all_features = []
    all_filenames = []

    print(f"Extracting features to {output_path}...")
    with torch.no_grad():
        for images, filenames in tqdm(loader, desc="Extracting meanpatch"):
            images = images.to(device)

            # Convert 6ch -> 3ch for RxRx1
            if images.shape[1] == 6:
                images = to_rgb(images)

            # Extract patch tokens: (B, 3, H, W) -> (B, num_patches, D)
            patch_tokens = encoder(images)

            # Mean pool over patches: (B, num_patches, D) -> (B, D)
            features = patch_tokens.mean(dim=1)

            all_features.append(features.cpu())
            all_filenames.extend(filenames)

    # Stack features
    features_tensor = torch.cat(all_features, dim=0)

    print(f"Extracted features shape: {features_tensor.shape}")

    # Parse metadata from filenames
    print("Parsing metadata from filenames...")
    metadata = {}

    # Detect format from first filename
    if all_filenames[0].startswith("cell"):
        # RxRx1 format: cell0_sirna123_0.pt
        import re
        cell_ids = []
        sirna_ids = []

        for fname in all_filenames:
            match = re.match(r"cell(\d+)_sirna(\d+)_\d+\.pt", fname)
            if match:
                cell_ids.append(int(match.group(1)))
                sirna_ids.append(int(match.group(2)))
            else:
                # Fallback
                cell_ids.append(0)
                sirna_ids.append(0)

        metadata["cell_type_id"] = torch.tensor(cell_ids, dtype=torch.long)
        metadata["sirna_id"] = torch.tensor(sirna_ids, dtype=torch.long)
    else:
        # CelebA format: Attr0_Attr1_..._idx.png
        # Parse binary attributes from filename
        print("Warning: CelebA format detection - implement if needed")

    print(f"Saving to {output_path}...")

    torch.save({
        "features": features_tensor,
        "metadata": metadata,
        "filenames": all_filenames,
        "encoder_name": f"{encoder_name}_meanpatch",
        "feature_dim": features_tensor.shape[1],
        "n_samples": features_tensor.shape[0],
    }, output_path)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Extract features from generated images")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model directory name under outputs/gen/ (e.g., rxrx1_repa_full)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--encoder-name",
        type=str,
        default="dinov3-vit-l",
        help="Encoder name",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Image resolution for encoder",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID (0-7)",
    )
    args = parser.parse_args()

    model_dir = Path("outputs/gen") / args.model
    image_dir = model_dir / "images"
    output_path = model_dir / "dinov3_meanpatch_features.pt"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if output_path.exists():
        print(f"Warning: {output_path} already exists, will overwrite")

    extract_meanpatch_features(
        image_dir=image_dir,
        output_path=output_path,
        batch_size=args.batch_size,
        encoder_name=args.encoder_name,
        resolution=args.resolution,
        device_id=args.device,
    )


if __name__ == "__main__":
    main()
