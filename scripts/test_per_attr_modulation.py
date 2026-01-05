"""Quick sanity test for per-attribute modulation implementation.

Tests:
1. Shared modulation mode (baseline)
2. Per-attribute modulation mode (ablation)
3. Forward pass correctness
4. Gradient flow
"""

import torch
from faithful_cond_gen.model.generator import GeneratorConfig, GeneratorWrapper


def test_shared_modulation():
    """Test baseline shared modulation mode."""
    print("\n=== Testing Shared Modulation (Baseline) ===")

    cfg = GeneratorConfig(
        image_size=224,
        in_channels=3,
        sit_arch="SiT-B/2",
        attr_num_classes=[2, 2],  # Male, Smiling
        use_per_attr_modulation=False,  # Baseline
    )

    model = GeneratorWrapper(cfg).cuda()
    print(f"✓ Model created with shared modulation")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).cuda()
    cond_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).cuda()  # (B, 2)
    t = torch.rand(batch_size).cuda()

    # Encode
    x0 = model.encode(images)
    print(f"✓ Encoding: {images.shape} -> {x0.shape}")

    # Velocity prediction
    v_pred = model.velocity_prediction(x0, t, cond_ids)
    print(f"✓ Velocity prediction: {x0.shape} -> {v_pred.shape}")

    # Backward pass
    loss = v_pred.mean()
    loss.backward()
    print(f"✓ Backward pass successful")

    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"✓ Gradients computed for {has_grad} parameters")

    return model


def test_per_attr_modulation():
    """Test per-attribute modulation mode."""
    print("\n=== Testing Per-Attribute Modulation (Ablation) ===")

    cfg = GeneratorConfig(
        image_size=224,
        in_channels=3,
        sit_arch="SiT-B/2",
        attr_num_classes=[2, 2],  # Male, Smiling
        use_per_attr_modulation=True,  # ABLATION
    )

    model = GeneratorWrapper(cfg).cuda()
    print(f"✓ Model created with per-attribute modulation")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).cuda()
    cond_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).cuda()
    t = torch.rand(batch_size).cuda()

    # Encode
    x0 = model.encode(images)
    print(f"✓ Encoding: {images.shape} -> {x0.shape}")

    # Velocity prediction
    v_pred = model.velocity_prediction(x0, t, cond_ids)
    print(f"✓ Velocity prediction: {x0.shape} -> {v_pred.shape}")

    # Backward pass
    loss = v_pred.mean()
    loss.backward()
    print(f"✓ Backward pass successful")

    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"✓ Gradients computed for {has_grad} parameters")

    # Check that per-attr modulators exist
    backbone = model.diffusion_backbone
    block = backbone.blocks[0]
    assert hasattr(block, 't_modulation'), "Block missing t_modulation"
    assert hasattr(block, 'attr_modulators'), "Block missing attr_modulators"
    assert len(block.attr_modulators) == 2, f"Expected 2 attr modulators, got {len(block.attr_modulators)}"
    print(f"✓ Per-attribute modulators verified: t_modulation + {len(block.attr_modulators)} attr_modulators")

    return model


def compare_parameter_counts(model_shared, model_per_attr):
    """Compare parameter counts between modes."""
    print("\n=== Parameter Count Comparison ===")

    n_shared = sum(p.numel() for p in model_shared.parameters())
    n_per_attr = sum(p.numel() for p in model_per_attr.parameters())

    print(f"Shared modulation:     {n_shared:,} parameters")
    print(f"Per-attr modulation:   {n_per_attr:,} parameters")
    print(f"Difference:            {n_per_attr - n_shared:,} parameters (+{100*(n_per_attr/n_shared - 1):.2f}%)")

    # Expected increase: (n_attrs + 1) modulators vs 1 modulator
    # For SiT-B/2: 12 blocks × (2+1 vs 1) × 6 × hidden_size × hidden_size
    # = 12 × 2 × 6 × 768 × 768 ≈ 84M extra params


def test_sampling():
    """Test sampling with both modes."""
    print("\n=== Testing Sampling ===")

    # Shared modulation
    cfg_shared = GeneratorConfig(
        image_size=224,
        in_channels=3,
        sit_arch="SiT-B/2",
        attr_num_classes=[2, 2],
        use_per_attr_modulation=False,
    )
    model_shared = GeneratorWrapper(cfg_shared).cuda().eval()

    cond_ids = torch.tensor([[0, 1]], dtype=torch.long).cuda()

    with torch.no_grad():
        samples_shared = model_shared.sample(cond_ids, num_inference_steps=10)
    print(f"✓ Shared modulation sampling: {samples_shared.shape}")

    # Per-attr modulation
    cfg_per_attr = GeneratorConfig(
        image_size=224,
        in_channels=3,
        sit_arch="SiT-B/2",
        attr_num_classes=[2, 2],
        use_per_attr_modulation=True,
    )
    model_per_attr = GeneratorWrapper(cfg_per_attr).cuda().eval()

    with torch.no_grad():
        samples_per_attr = model_per_attr.sample(cond_ids, num_inference_steps=10)
    print(f"✓ Per-attr modulation sampling: {samples_per_attr.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("Per-Attribute Modulation Sanity Test")
    print("=" * 60)

    try:
        # Test both modes
        model_shared = test_shared_modulation()
        model_per_attr = test_per_attr_modulation()

        # Compare
        compare_parameter_counts(model_shared, model_per_attr)

        # Test sampling
        test_sampling()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
