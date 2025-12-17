#!/usr/bin/env python3
"""
Test CelebA DataModule with held-out combinations.
"""
import sys
sys.path.insert(0, "/mnt/pvc/faithful-cond-gen/src")

from faithful_cond_gen.data.celeba import CelebaDataModule, CelebaDataConfig

def main():
    print("Testing CelebA DataModule with held-out combinations...")
    
    # Test with held-out combos
    cfg = CelebaDataConfig(
        cache_dir="/mnt/pvc/AutoSync/data/celeba_cache",
        image_size=(224, 224),
        augment_train=False,
        normalize=True,
        batch_size=16,
        num_workers=4,
        rare_threshold=50,
        selected_attrs=None,  # Will use default
        held_out_combos=[
            (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0),
            (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 0, 0),
            (1, 0, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 0),
            (1, 1, 0, 1), (1, 1, 1, 0)
        ]
    )
    
    print("\nInitializing CelebA DataModule with 14 held-out combinations...")
    data_module = CelebaDataModule(cfg)
    
    print("\nDataModule initialized successfully!")
    print(f"Selected attributes: {data_module.selected_attrs}")
    print(f"Number of held-out combinations: {len(data_module.held_out_combos)}")

if __name__ == "__main__":
    main()
