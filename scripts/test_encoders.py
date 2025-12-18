# scripts/test_encoders.py
import traceback

import torch
from faithful_cond_gen.eval.configs.encoder_config import (
    BIOCLIP,
    DINOV2_L14,
    DINOV3_L16,
    MAE_LARGE,
    OPENPHENOM,
    SIGLIP_SO400M,
)
from faithful_cond_gen.eval.encoders.registry import load_encoder


def test_config(cfg):
    print(f"\n{'='*60}")
    print(f"Testing Config: {cfg.name}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    try:
        # 1. LOAD MODEL
        print(f"-> Loading encoder from: {cfg.hf_path} ...")
        model = load_encoder(cfg, device=device)
        print(f"   [OK] Model loaded. Feature Dim: {model.feature_dim}")

        # 2. CHECK TRANSFORMS
        print("-> Checking transforms...")
        tf = model.get_transform()
        print(f"   [OK] Transforms: {tf}")

        # 3. DUMMY INPUT
        # Create a random image batch matching input channels & size
        # (B, C, H, W)
        B = 2
        C = cfg.input_channels
        H, W = cfg.image_size
        x = torch.randn(B, C, H, W).to(device)
        print(f"-> Created dummy input: {x.shape} (Batch Size={B})")

        # 4. FORWARD PASS
        print("-> Running forward pass...")
        # Most models expect normalized inputs, but for this raw test we just pass random noise.
        # In a real pipeline, 'tf' would be applied to raw PIL images first.
        # Here we assume 'x' is already a tensor ready for the model's forward().
        with torch.no_grad():
            out = model(x)

        feats = out.get("features")
        if feats is None:
            raise ValueError("Output dictionary missing 'features' key.")

        print(f"   [OK] Forward pass successful.")
        print(f"   Output shape: {feats.shape}")

        # 5. SHAPE CHECK
        expected_shape = (B, model.feature_dim)
        if feats.shape != expected_shape:
            print(
                f"   [FAIL] Shape Mismatch! Expected {expected_shape}, got {feats.shape}"
            )
        else:
            print(f"   [OK] Shape matches expected dimension.")

    except Exception as e:
        print(f"   [ERROR] Failed to test {cfg.name}")
        print("-" * 20)
        traceback.print_exc()
        print("-" * 20)


if __name__ == "__main__":
    # List of all configs to test
    all_configs = [
        DINOV2_L14,
        DINOV3_L16,
        MAE_LARGE,
        SIGLIP_SO400M,
        BIOCLIP,
        OPENPHENOM,
    ]

    print(f"Starting generic encoder tests for {len(all_configs)} configurations...")

    for config in all_configs:
        test_config(config)

    print("\n\nAll tests completed.")
