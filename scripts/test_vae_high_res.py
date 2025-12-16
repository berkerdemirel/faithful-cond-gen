#!/usr/bin/env python3
"""VAE High Resolution Sanity Check.

Tests the VAEBackbone at proper resolution (224x224 or higher) to demonstrate
that the "weird" reconstruction at 64x64 is due to extreme downsampling.

At 64x64 -> 8x8 latents (too small for face details!)
At 224x224 -> 28x28 latents (much better for preserving details)
At 256x256 -> 32x32 latents (even better)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from faithful_cond_gen.data.celeba import CelebaDataConfig, CelebaDataModule
from faithful_cond_gen.model.generator import VAEBackbone
from torchvision.utils import make_grid


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor):
    """Compute MSE and PSNR."""
    mse = F.mse_loss(reconstructed, original, reduction="mean").item()
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse))).item()
    return mse, psnr


def visualize_high_res_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    save_path: Path,
    n_samples: int = 4,
):
    """Create detailed visualization with original, reconstruction, and 10x amplified difference."""
    n_samples = min(n_samples, original.shape[0])

    orig_samples = original[:n_samples].cpu()
    recon_samples = reconstructed[:n_samples].cpu()

    # Compute per-sample MSE
    per_sample_mse = []
    for i in range(n_samples):
        mse = F.mse_loss(recon_samples[i], orig_samples[i], reduction="mean").item()
        per_sample_mse.append(mse)

    # Ensure values are in [0, 1]
    orig_samples = torch.clamp(orig_samples, 0, 1)
    recon_samples = torch.clamp(recon_samples, 0, 1)

    # Compute difference (amplified for visibility)
    diff_samples = torch.abs(orig_samples - recon_samples)
    diff_samples_viz = torch.clamp(diff_samples * 10, 0, 1)

    # Create side-by-side comparison for each sample
    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # Original
        axes[i, 0].imshow(orig_samples[i].permute(1, 2, 0).numpy())
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("Original", fontsize=14, fontweight="bold")

        # Reconstructed
        axes[i, 1].imshow(recon_samples[i].permute(1, 2, 0).numpy())
        axes[i, 1].axis("off")
        if i == 0:
            axes[i, 1].set_title("VAE Reconstruction", fontsize=14, fontweight="bold")

        # Difference (amplified)
        axes[i, 2].imshow(diff_samples_viz[i].permute(1, 2, 0).numpy())
        axes[i, 2].axis("off")
        if i == 0:
            axes[i, 2].set_title("Difference (10x)", fontsize=14, fontweight="bold")

        # Add MSE text on the right
        axes[i, 2].text(
            1.05,
            0.5,
            f"MSE: {per_sample_mse[i]:.5f}",
            transform=axes[i, 2].transAxes,
            fontsize=10,
            verticalalignment="center",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  ✓ High-res visualization saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="VAE High Resolution Sanity Check",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        choices=[64, 128, 192, 224, 256, 320, 384, 448, 512],
        help="Image size (higher = better quality, slower)",
    )
    parser.add_argument(
        "--vae-model",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="VAE model name",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/mnt/pvc/AutoSync/data/celeba_cache",
        help="Dataset cache directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_outputs/vae_high_res",
        help="Output directory",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VAE HIGH RESOLUTION RECONSTRUCTION TEST")
    print("=" * 80)
    print(f"\nImage size: {args.image_size}x{args.image_size}")

    # Calculate expected latent size
    latent_size = args.image_size // 8
    print(f"Expected latent size: {latent_size}x{latent_size}")
    print(f"Total latent pixels: {latent_size * latent_size}")

    if args.image_size <= 64:
        print("\n⚠️  WARNING: Using low resolution!")
        print(
            f"   At {args.image_size}x{args.image_size}, latents are only {latent_size}x{latent_size}"
        )
        print("   This is TOO SMALL to preserve facial details!")
        print("   Recommended: Use 224x224 or higher for face images.\n")
    elif args.image_size >= 224:
        print("\n✓ Good resolution choice!")
        print(
            f"  At {args.image_size}x{args.image_size}, latents are {latent_size}x{latent_size}"
        )
        print("  This should preserve facial features well.\n")

    # Load data
    print("[1/4] Loading CelebA data...")
    data_cfg = CelebaDataConfig(
        cache_dir=args.cache_dir,
        image_size=(args.image_size, args.image_size),
        augment_train=False,
        normalize=False,
        batch_size=args.num_samples,
        num_workers=0,
    )

    data_module = CelebaDataModule(cfg=data_cfg)
    test_dataset = data_module.get_dataset("test")
    test_loader = data_module.get_dataloader("test")

    print(f"✓ Loaded {len(test_dataset)} test samples")

    # Initialize VAE
    print(f"\n[2/4] Initializing VAE ({args.vae_model})...")
    vae = VAEBackbone(
        vae_model_name=args.vae_model,
        in_channels=3,
        freeze=True,
    ).to(device)
    vae.eval()

    print(f"✓ VAE loaded")
    print(f"  Downsampling factor: {vae.downsampling_factor}x")
    print(f"  Latent channels: {vae.out_channels}")

    # Run reconstruction
    print(f"\n[3/4] Running reconstruction test...")

    with torch.no_grad():
        batch = next(iter(test_loader))
        images, _ = batch
        images = images.to(device)

        actual_n = min(args.num_samples, images.shape[0])
        images = images[:actual_n]

        print(f"\nProcessing {actual_n} images...")
        print(f"  Input shape: {images.shape}")
        print(f"  Input range: [{images.min():.3f}, {images.max():.3f}]")

        # Encode
        latents = vae.encode(images)
        print(f"  Latent shape: {latents.shape}")
        print(f"  Latent range: [{latents.min():.3f}, {latents.max():.3f}]")

        # Decode
        reconstructed = vae.decode(latents)
        print(f"  Output shape: {reconstructed.shape}")
        print(f"  Output range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")

        # Compute metrics
        mse, psnr = compute_metrics(images, reconstructed)

        print(f"\n  Overall Metrics:")
        print(f"    MSE:  {mse:.6f}")
        print(f"    PSNR: {psnr:.2f} dB")

        # Save visualization
        print(f"\n[4/4] Saving visualization...")
        vis_path = (
            output_path / f"vae_reconstruction_{args.image_size}x{args.image_size}.png"
        )
        visualize_high_res_reconstruction(
            images, reconstructed, vis_path, n_samples=actual_n
        )

    # Save summary
    summary_path = output_path / f"summary_{args.image_size}x{args.image_size}.txt"
    with open(summary_path, "w") as f:
        f.write("VAE HIGH RESOLUTION RECONSTRUCTION TEST\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Image size: {args.image_size}x{args.image_size}\n")
        f.write(f"Latent size: {latent_size}x{latent_size}\n")
        f.write(f"Latent channels: {vae.out_channels}\n")
        f.write(
            f"Total latent dimensions: {vae.out_channels}x{latent_size}x{latent_size}\n"
        )
        f.write(f"Downsampling factor: {vae.downsampling_factor}x\n\n")
        f.write(f"Reconstruction Quality:\n")
        f.write(f"  MSE:  {mse:.6f}\n")
        f.write(f"  PSNR: {psnr:.2f} dB\n\n")

        if psnr > 30:
            f.write("Assessment: ✓ EXCELLENT reconstruction quality\n")
        elif psnr > 25:
            f.write("Assessment: ✓ GOOD reconstruction quality\n")
        elif psnr > 20:
            f.write("Assessment: ⚠ ACCEPTABLE reconstruction quality\n")
        else:
            f.write("Assessment: ✗ POOR reconstruction quality\n")

    print(f"✓ Summary saved to {summary_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("COMPARISON OF RESOLUTIONS")
    print("=" * 80)
    print("\nResolution Analysis:")
    for res in [64, 128, 224, 256, 512]:
        lat = res // 8
        total = lat * lat
        status = "❌ TOO LOW" if res < 128 else ("⚠️  LOW" if res < 224 else "✅ GOOD")
        print(f"  {res}x{res} -> {lat}x{lat} latents ({total:4d} pixels) {status}")

    print("\n" + "=" * 80)
    if args.image_size < 128:
        print("⚠️  CONCLUSION: Low resolution causes poor face reconstruction!")
        print(
            f"   Current: {args.image_size}x{args.image_size} -> {latent_size}x{latent_size} latents"
        )
        print("   Try running with: --image-size 224 or --image-size 256")
    else:
        print("✓ CONCLUSION: Resolution is adequate for face reconstruction")
        print(
            f"  At {args.image_size}x{args.image_size}, the VAE has {latent_size}x{latent_size} latents"
        )
        print("  This provides enough spatial resolution for facial details.")
    print("=" * 80)

    print(f"\nVisualization saved to: {vis_path}")
    print("Inspect the images to see the reconstruction quality!")


if __name__ == "__main__":
    main()
