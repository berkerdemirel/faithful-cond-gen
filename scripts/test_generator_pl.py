#!/usr/bin/env python
import os
import sys

import torch

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from faithful_cond_gen.model.generator import GeneratorConfig, GeneratorWrapper
from faithful_cond_gen.pl_modules.generator_pl import GeneratorPL, GeneratorPLConfig


def make_mock_batch(batch_size, channels, size, attr_names):
    """Creates a mock batch with nested 'cond' dictionary."""
    images = torch.randn(batch_size, channels, size, size)

    # Create ordered dictionary of attributes
    cond_dict = {}
    for name in attr_names:
        cond_dict[name] = torch.randint(0, 2, (batch_size,))

    conditioning = {"cond": cond_dict, "other_metadata": ["ignore_me"] * batch_size}

    return images, conditioning


def test_pl_pipeline():
    print(f"\n{'='*20} Testing GeneratorPL Pipeline {'='*20}")

    # Configs
    B, C, H, W = 4, 3, 32, 32
    # Mocking CelebA-like attributes
    attr_names = ["Male", "Smiling", "Young"]
    attr_counts = [2, 2, 2]  # Binary attrs

    gen_cfg = GeneratorConfig(
        image_size=H,
        in_channels=C,
        sit_arch="SiT-B/2",
        attr_num_classes=attr_counts,
        vae_freeze=True,
    )

    pl_cfg = GeneratorPLConfig(lr=1e-4)

    print("Initializing Model & PL Module...")
    generator = GeneratorWrapper(gen_cfg)
    pl_module = GeneratorPL(generator, pl_cfg)

    # Create Batch
    batch = make_mock_batch(B, C, H, attr_names)

    # 1. Test Unpacking
    print("Testing Adapter (Batch Unpacking)...")
    images, cond_ids = pl_module._unpack_batch(batch)

    if images.shape != (B, C, H, W):
        raise ValueError(f"❌ Image unpacking shape mismatch. Got {images.shape}")

    if cond_ids.shape != (B, len(attr_names)):
        raise ValueError(f"❌ Condition unpacking shape mismatch. Got {cond_ids.shape}")

    print(f"✅ Unpacking successful. Cond shape: {cond_ids.shape}")

    # 2. Test Training Step
    print("Testing Training Step (Forward + Loss)...")
    loss = pl_module.training_step(batch, batch_idx=0)

    if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
        raise ValueError(f"❌ Loss is not a scalar tensor: {loss}")

    print(f"✅ Training Step successful. Loss: {loss.item():.4f}")

    # 3. Test Validation Step
    print("Testing Validation Step...")
    val_loss = pl_module.validation_step(batch, batch_idx=0)
    print(f"✅ Validation Step successful. Val Loss: {val_loss.item():.4f}")

    # 4. Test Optimizer Config
    print("Testing Optimizer Configuration...")
    optim = pl_module.configure_optimizers()
    if optim is None:
        raise ValueError("❌ No optimizer returned.")
    print("✅ Optimizer configured.")


if __name__ == "__main__":
    try:
        test_pl_pipeline()
        print("\n🎉 All PL Pipeline Tests Passed!")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback

        traceback.print_exc()
