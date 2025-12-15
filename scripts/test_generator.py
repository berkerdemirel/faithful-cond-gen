#!/usr/bin/env python
import os
import sys

import torch

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from faithful_cond_gen.model.generator import GeneratorConfig, GeneratorWrapper


def test_rgb_generator():
    """Test standard 3-channel configuration (e.g. CelebA)."""
    print(f"\n{'='*20} Testing 3-Channel (RGB) Generator {'='*20}")

    # 1. Config: 3 input channels, 4 binary attributes
    B, C, H, W = 2, 3, 64, 64
    attr_counts = [2, 2, 2, 2]

    cfg = GeneratorConfig(
        image_size=H,
        in_channels=C,
        sit_arch="SiT-B/2",  # Small patch size for 64x64
        attr_num_classes=attr_counts,
        vae_freeze=True,
    )

    print(f"Initializing Generator (in_channels={C})...")
    model = GeneratorWrapper(cfg)
    model.eval()

    # 2. Mock Data
    images = torch.rand(B, C, H, W)
    # Mock conditioning: (B, 4)
    cond_ids = torch.randint(0, 2, (B, len(attr_counts)))

    # 3. Test Encode
    print("Testing Encode...")
    latents = model.encode(images)
    expected_latent_ch = 4
    expected_shape = (B, expected_latent_ch, H // 8, W // 8)

    if latents.shape != expected_shape:
        raise ValueError(
            f"❌ Encode shape mismatch. Expected {expected_shape}, got {latents.shape}"
        )
    print(f"✅ Encoded shape: {latents.shape}")

    # 4. Test Velocity Prediction
    print("Testing Velocity Prediction...")
    t = torch.rand(B)
    v_pred = model.velocity_prediction(latents, t, cond_ids)

    if v_pred.shape != latents.shape:
        raise ValueError(
            f"❌ Velocity shape mismatch. Expected {latents.shape}, got {v_pred.shape}"
        )
    print(f"✅ Velocity shape: {v_pred.shape}")

    # 5. Test Decode
    print("Testing Decode...")
    rec_images = model.decode(latents)

    if rec_images.shape != images.shape:
        raise ValueError(
            f"❌ Decode shape mismatch. Expected {images.shape}, got {rec_images.shape}"
        )

    if rec_images.min() < 0.0 or rec_images.max() > 1.0:
        print(
            f"⚠️ Warning: Decoded images out of [0,1] range: [{rec_images.min():.3f}, {rec_images.max():.3f}]"
        )
    else:
        print("✅ Decoded range is valid [0, 1].")

    # 6. Test Sampling
    print("Testing Sample...")
    samples = model.sample(cond_ids, num_inference_steps=2)
    if samples.shape != images.shape:
        raise ValueError(
            f"❌ Sample shape mismatch. Expected {images.shape}, got {samples.shape}"
        )
    print(f"✅ Sampled shape: {samples.shape}")


def test_rxrx1_generator():
    """Test 6-channel configuration (RxRx1)."""
    print(f"\n{'='*20} Testing 6-Channel (RxRx1) Generator {'='*20}")

    # 1. Config: 6 input channels, [Cell(4), siRNA(1138)]
    B, C, H, W = 2, 6, 64, 64
    attr_counts = [4, 1138]

    cfg = GeneratorConfig(
        image_size=H,
        in_channels=C,
        sit_arch="SiT-B/2",
        attr_num_classes=attr_counts,
        vae_freeze=True,
    )

    print(f"Initializing Generator (in_channels={C})...")
    model = GeneratorWrapper(cfg)
    model.eval()

    # 2. Mock Data
    images = torch.rand(B, C, H, W)
    # Mock conditioning: (B, 2)
    cond_ids = torch.stack(
        [torch.randint(0, 4, (B,)), torch.randint(0, 100, (B,))],  # Cell Type  # siRNA
        dim=1,
    )

    # 3. Test Encode (Expect Channel Folding)
    print("Testing Encode (Folding)...")
    latents = model.encode(images)

    # Expected: 6 channels * 4 latent dims = 24 channels
    expected_latent_ch = 24
    expected_shape = (B, expected_latent_ch, H // 8, W // 8)

    if latents.shape != expected_shape:
        raise ValueError(
            f"❌ Encode shape mismatch. Expected {expected_shape}, got {latents.shape}"
        )
    print(f"✅ Encoded shape (folded): {latents.shape}")

    # 4. Test Velocity Prediction
    print("Testing Velocity Prediction...")
    t = torch.rand(B)
    v_pred = model.velocity_prediction(latents, t, cond_ids)

    if v_pred.shape != latents.shape:
        raise ValueError(
            f"❌ Velocity shape mismatch. Expected {latents.shape}, got {v_pred.shape}"
        )
    print(f"✅ Velocity shape: {v_pred.shape}")

    # 5. Test Decode (Expect Unfolding)
    print("Testing Decode (Unfolding)...")
    rec_images = model.decode(latents)

    if rec_images.shape != images.shape:
        raise ValueError(
            f"❌ Decode shape mismatch. Expected {images.shape}, got {rec_images.shape}"
        )
    print(f"✅ Decoded shape: {rec_images.shape}")

    if (
        rec_images.min() < -0.1 or rec_images.max() > 1.1
    ):  # Allow slight buffer for float errors
        print(
            f"⚠️ Warning: Decoded images out of expected range: [{rec_images.min():.3f}, {rec_images.max():.3f}]"
        )

    # 6. Test Sampling
    print("Testing Sample...")
    # Initialize random cond_ids
    samples = model.sample(cond_ids, num_inference_steps=2)
    if samples.shape != images.shape:
        raise ValueError(
            f"❌ Sample shape mismatch. Expected {images.shape}, got {samples.shape}"
        )
    print(f"✅ Sampled shape: {samples.shape}")


if __name__ == "__main__":
    try:
        test_rgb_generator()
        test_rxrx1_generator()
        print("\n🎉 All Generator Tests Passed!")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback

        traceback.print_exc()
