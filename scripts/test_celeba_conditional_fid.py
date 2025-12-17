#!/usr/bin/env python3
"""
Test script for checking conditional FID sanity check on CelebA dataset.
Measures real vs real FID for different sample sizes with 0,0,0,0 conditioning.
"""
import sys

sys.path.insert(0, "/mnt/pvc/faithful-cond-gen/src")

import numpy as np
import torch
from faithful_cond_gen.data.celeba import CelebaDataConfig, CelebaDataModule
from faithful_cond_gen.utils.metrics import ConditionalFidelityMetrics


def main():
    print("=" * 80)
    print("CelebA Conditional FID Sanity Check")
    print("Testing real vs real FID for 0,0,0,0 conditioning")
    print("=" * 80)

    # Setup device
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Initialize CelebA DataModule
    print("\nInitializing CelebA DataModule...")
    cfg = CelebaDataConfig(
        cache_dir="/mnt/pvc/AutoSync/data/celeba_cache",
        image_size=(224, 224),
        augment_train=False,
        normalize=False,  # ConditionalFidelityMetrics handles normalization
        batch_size=32,
        num_workers=4,
        rare_threshold=50,
        selected_attrs=None,  # Will use default: Male, Smiling, Blond_Hair, Eyeglasses
    )

    data_module = CelebaDataModule(cfg)
    print(f"Selected attributes: {data_module.selected_attrs}")

    # Get validation samples for 0,0,0,0 conditioning
    print("\nFiltering validation samples for [0, 0, 0, 0] conditioning...")
    conditions = {attr: 0 for attr in data_module.selected_attrs}
    print(f"Conditions: {conditions}")

    val_dataset = data_module.get_matching_dataset(
        split="val",
        conditions=conditions,
        max_samples=None,  # Get all available samples
    )

    print(f"Found {len(val_dataset)} samples matching the conditions")

    if len(val_dataset) < 100:
        print(
            f"Warning: Only {len(val_dataset)} samples found. Need more for meaningful FID."
        )
        return

    # Load all images
    print("\nLoading all images into memory...")
    all_images = []
    for i in range(len(val_dataset)):
        img, _ = val_dataset[i]
        all_images.append(img)

    all_images = torch.stack(all_images)
    # Ensure images are in [0, 1] range and clamp any values outside
    all_images = torch.clamp(all_images, 0.0, 1.0)
    all_images = all_images.to(device)
    print(f"Loaded {len(all_images)} images with shape {all_images.shape}")
    print(
        f"Image value range: [{all_images.min().item():.3f}, {all_images.max().item():.3f}]"
    )

    # Initialize metrics
    metrics = ConditionalFidelityMetrics(device)

    # Test different sample sizes
    sample_sizes = [50, 100, 200, 500, 1000, 2000, 4000]
    # Filter out sample sizes that are too large
    sample_sizes = [s for s in sample_sizes if s * 2 <= len(all_images)]

    if not sample_sizes:
        print(
            f"\nError: Not enough samples. Need at least 100 samples, but only have {len(all_images)}"
        )
        return

    print(f"\nTesting sample sizes: {sample_sizes}")
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    print(f"{'Sample Size':<15} {'FID Baseline':<15} {'Notes':<30}")
    print("-" * 80)

    results = []

    for n_samples in sample_sizes:
        # Need 2*n_samples total for real vs real comparison
        required = 2 * n_samples

        if required > len(all_images):
            print(
                f"{n_samples:<15} {'Skipped':<15} Not enough samples (need {required})"
            )
            continue

        # Randomly sample without replacement
        indices = torch.randperm(len(all_images))[:required]
        sampled_images = all_images[indices]

        # For FID baseline, we use real_A and real_B
        # For generated case, we can use random noise
        gen_samples = torch.randn_like(sampled_images)

        # Compute rFID (which includes FID baseline)
        rfid, fid_gen, fid_baseline = metrics.compute_rfid(
            real_samples=sampled_images, gen_samples=gen_samples
        )

        results.append(
            {
                "n_samples": n_samples,
                "fid_baseline": fid_baseline,
                "fid_gen": fid_gen,
                "rfid": rfid,
            }
        )

        print(f"{n_samples:<15} {fid_baseline:<15.4f} {'Real vs Real':<30}")

    print("-" * 80)

    # Additional analysis
    print("\n" + "=" * 80)
    print("Detailed Analysis")
    print("=" * 80)

    for result in results:
        print(f"\nSample Size: {result['n_samples']}")
        print(f"  FID (Real A vs Real B): {result['fid_baseline']:.4f}")
        print(f"  FID (Gen vs Real A):    {result['fid_gen']:.4f}")
        print(f"  Relative FID (rFID):    {result['rfid']:.4f}")
        print(
            f"  Interpretation: Random noise is {result['rfid']:.2f}x worse than real baseline"
        )

    # Summary statistics
    if len(results) > 1:
        baselines = [r["fid_baseline"] for r in results]
        print(f"\n" + "=" * 80)
        print("Summary Statistics for FID Baseline")
        print("=" * 80)
        print(f"Mean FID Baseline:   {np.mean(baselines):.4f}")
        print(f"Std FID Baseline:    {np.std(baselines):.4f}")
        print(f"Min FID Baseline:    {np.min(baselines):.4f}")
        print(f"Max FID Baseline:    {np.max(baselines):.4f}")
        print("\nNote: FID baseline should generally decrease with more samples,")
        print("but may vary due to random sampling and distribution characteristics.")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)

    metrics.fid.reset()
    metrics.fid.update(all_images[:1000], real=True)
    metrics.fid.update(all_images[:1000], real=False)
    print(
        f"\nFinal FID (identity) check (1000 real vs 1000 real): {float(metrics.fid.compute()):.4f}"
    )


if __name__ == "__main__":
    main()
