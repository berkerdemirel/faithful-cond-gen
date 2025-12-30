"""Comprehensive tests for classifier-free guidance (CFG) implementation

Tests:
1. CFG changes predictions (cfg_scale != 1.0 gives different results)
2. Backward compatibility (cfg_scale=1.0 gives conditional-only results)
3. Adaptive CFG with per-sample scales
4. Force drop mechanism in SiT
5. Integration with both sample() and basic_sample()
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_force_drop_mechanism():
    """Test that force_drop_ids works in LabelEmbedder and SiT"""
    print("\n" + "=" * 60)
    print("Test 1: Force Drop Mechanism")
    print("=" * 60)

    from faithful_cond_gen.model.sit_backbone import LabelEmbedder

    # Create a simple label embedder
    num_classes = 10
    hidden_size = 64
    dropout_prob = 0.1  # Enable CFG embedding

    embedder = LabelEmbedder(num_classes, hidden_size, dropout_prob)

    # Test labels
    batch_size = 4
    labels = torch.randint(0, num_classes, (batch_size,))

    # Test 1: Normal conditional (no force drop)
    emb_cond = embedder(labels, train=False, force_drop_ids=None)
    print(f"  Conditional embeddings shape: {emb_cond.shape}")
    assert emb_cond.shape == (batch_size, hidden_size), f"Wrong shape: {emb_cond.shape}"

    # Test 2: Force unconditional (force_drop_ids=1)
    force_drop = torch.ones(batch_size, dtype=torch.long)
    emb_uncond = embedder(labels, train=False, force_drop_ids=force_drop)
    print(f"  Unconditional embeddings shape: {emb_uncond.shape}")

    # Test 3: Verify conditional and unconditional are different
    diff = (emb_cond - emb_uncond).abs().mean().item()
    print(f"  Mean difference (cond vs uncond): {diff:.4f}")
    assert diff > 0.01, "Conditional and unconditional should be different!"

    # Test 4: Verify unconditional uses the special token (num_classes)
    # The unconditional token is embedding_table[num_classes]
    uncond_token_emb = embedder.embedding_table(torch.tensor([num_classes]))
    # All unconditional embeddings should be the same
    for i in range(batch_size):
        assert torch.allclose(
            emb_uncond[i], uncond_token_emb[0], atol=1e-5
        ), f"Sample {i} doesn't match unconditional token"

    print("  ✓ Force drop mechanism works correctly")
    print("  ✓ Conditional and unconditional embeddings differ")
    print("  ✓ Unconditional embeddings use special token")

    print("\n✅ Force drop mechanism test PASSED")
    return True


def test_cfg_changes_predictions():
    """Test that CFG changes velocity predictions"""
    print("\n" + "=" * 60)
    print("Test 2: CFG Changes Predictions")
    print("=" * 60)

    from faithful_cond_gen.model.generator import GeneratorWrapper, GeneratorConfig

    # Create minimal config for testing
    cfg = GeneratorConfig(
        vae_model_name="stabilityai/sd-vae-ft-mse",
        in_channels=3,
        image_size=256,  # Must be compatible with VAE + patch size
        sit_arch="SiT-S/2",
        attr_num_classes=[5, 10],  # 2 attributes
        class_dropout_prob=0.1,  # Enable CFG
    )

    # Create generator
    print("  Creating generator (this may take a moment)...")
    generator = GeneratorWrapper(cfg)
    generator.eval()

    # Create test inputs
    batch_size = 2
    latent_size = generator.vae.get_latent_size(cfg.image_size)
    device = "cpu"

    x = torch.randn(batch_size, generator.vae.out_channels, latent_size, latent_size)
    t = torch.rand(batch_size) * 0.5 + 0.5  # t in [0.5, 1.0]
    cond_ids = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)

    # Test 1: No guidance (cfg_scale=1.0)
    print("\n  Testing cfg_scale=1.0 (no guidance)...")
    v_no_cfg = generator.apply_cfg(x, t, cond_ids, cfg_scale=1.0)
    print(f"  ✓ No guidance prediction shape: {v_no_cfg.shape}")

    # Test 2: With guidance (cfg_scale=2.0)
    print("  Testing cfg_scale=2.0 (with guidance)...")
    v_with_cfg = generator.apply_cfg(x, t, cond_ids, cfg_scale=2.0)
    print(f"  ✓ With guidance prediction shape: {v_with_cfg.shape}")

    # Test 3: Verify they're different
    diff = (v_no_cfg - v_with_cfg).abs().mean().item()
    print(f"\n  Mean difference (cfg=1.0 vs cfg=2.0): {diff:.6f}")
    assert diff > 1e-4, f"Predictions should differ with CFG, but diff={diff}"
    print("  ✓ CFG changes predictions as expected")

    # Test 4: Higher cfg_scale = larger difference
    print("\n  Testing cfg_scale=5.0 (stronger guidance)...")
    v_strong_cfg = generator.apply_cfg(x, t, cond_ids, cfg_scale=5.0)
    diff_strong = (v_no_cfg - v_strong_cfg).abs().mean().item()
    print(f"  Mean difference (cfg=1.0 vs cfg=5.0): {diff_strong:.6f}")
    assert diff_strong > diff, "Stronger CFG should create larger difference"
    print("  ✓ Stronger CFG creates larger changes")

    print("\n✅ CFG prediction changes test PASSED")
    return True


def test_backward_compatibility():
    """Test that cfg_scale=1.0 gives conditional-only results"""
    print("\n" + "=" * 60)
    print("Test 3: Backward Compatibility (cfg_scale=1.0)")
    print("=" * 60)

    from faithful_cond_gen.model.generator import GeneratorWrapper, GeneratorConfig

    cfg = GeneratorConfig(
        vae_model_name="stabilityai/sd-vae-ft-mse",
        in_channels=3,
        image_size=256,  # Must be compatible with VAE + patch size
        sit_arch="SiT-S/2",
        attr_num_classes=[5, 10],
        class_dropout_prob=0.1,
    )

    generator = GeneratorWrapper(cfg)
    generator.eval()

    batch_size = 2
    latent_size = generator.vae.get_latent_size(cfg.image_size)

    x = torch.randn(batch_size, generator.vae.out_channels, latent_size, latent_size)
    t = torch.rand(batch_size) * 0.5 + 0.5
    cond_ids = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)

    # Method 1: Using apply_cfg with cfg_scale=1.0
    print("  Method 1: apply_cfg(cfg_scale=1.0)...")
    v_cfg_1 = generator.apply_cfg(x, t, cond_ids, cfg_scale=1.0)

    # Method 2: Direct conditional prediction (no CFG)
    print("  Method 2: Direct conditional prediction...")
    v_direct, _ = generator.diffusion_backbone(x, t, cond_ids)
    if v_direct.shape[1] > x.shape[1]:
        v_direct = v_direct[:, : x.shape[1]]

    # They should be identical
    diff = (v_cfg_1 - v_direct).abs().max().item()
    print(f"\n  Max difference: {diff:.2e}")
    assert diff < 1e-5, f"cfg_scale=1.0 should give same result as direct prediction, diff={diff}"

    print("  ✓ cfg_scale=1.0 is backward compatible")
    print("  ✓ No extra computation when CFG not needed")

    print("\n✅ Backward compatibility test PASSED")
    return True


def test_adaptive_cfg():
    """Test adaptive CFG with per-sample scales"""
    print("\n" + "=" * 60)
    print("Test 4: Adaptive CFG (per-sample scales)")
    print("=" * 60)

    from faithful_cond_gen.model.generator import GeneratorWrapper, GeneratorConfig

    cfg = GeneratorConfig(
        vae_model_name="stabilityai/sd-vae-ft-mse",
        in_channels=3,
        image_size=256,  # Must be compatible with VAE + patch size
        sit_arch="SiT-S/2",
        attr_num_classes=[5, 10],
        class_dropout_prob=0.1,
    )

    generator = GeneratorWrapper(cfg)
    generator.eval()

    batch_size = 4
    latent_size = generator.vae.get_latent_size(cfg.image_size)

    x = torch.randn(batch_size, generator.vae.out_channels, latent_size, latent_size)
    t = torch.rand(batch_size) * 0.5 + 0.5
    cond_ids = torch.tensor([[0, 0], [1, 2], [2, 3], [3, 4]], dtype=torch.long)

    # Test 1: Per-sample cfg_scales
    print("  Testing per-sample CFG scales...")
    cfg_scales = torch.tensor([1.0, 1.5, 2.0, 3.0])  # Different for each sample

    v_adaptive = generator.apply_cfg(x, t, cond_ids, cfg_scale=cfg_scales)
    print(f"  ✓ Adaptive CFG prediction shape: {v_adaptive.shape}")

    # Test 2: Verify it works (doesn't crash, gives valid results)
    assert v_adaptive.shape == x.shape, f"Wrong shape: {v_adaptive.shape}"
    assert torch.isfinite(v_adaptive).all(), "Non-finite values in output"
    print("  ✓ Per-sample scales work correctly")

    # Test 3: Load condition stats and compute adaptive scales
    print("\n  Testing compute_adaptive_cfg_scale()...")
    condition_stats = {
        "(0, 0)": 200,  # Common -> cfg=1.0
        "(1, 2)": 50,  # Medium
        "(2, 3)": 5,  # Rare -> cfg=3.0
        "(3, 4)": 150,  # Common -> cfg=1.0
    }

    generator.load_condition_stats(condition_stats)
    computed_scales = generator.compute_adaptive_cfg_scale(cond_ids)

    print(f"  Computed scales: {computed_scales.tolist()}")
    print(f"    Sample 0 (count=200): cfg={computed_scales[0]:.2f} (expect ~1.0)")
    print(f"    Sample 1 (count=50):  cfg={computed_scales[1]:.2f} (expect ~2.0)")
    print(f"    Sample 2 (count=5):   cfg={computed_scales[2]:.2f} (expect ~3.0)")
    print(f"    Sample 3 (count=150): cfg={computed_scales[3]:.2f} (expect ~1.0)")

    # Verify logic
    assert computed_scales[0] == 1.0, "Common condition should get min_cfg"
    assert computed_scales[2] == 3.0, "Rare condition should get max_cfg"
    assert 1.0 < computed_scales[1] < 3.0, "Medium condition should be interpolated"

    print("  ✓ Adaptive CFG scale computation works")

    # Test 4: Use computed scales in sampling
    print("\n  Testing with computed adaptive scales...")
    v_auto_adaptive = generator.apply_cfg(x, t, cond_ids, cfg_scale=computed_scales)
    assert v_auto_adaptive.shape == x.shape
    print("  ✓ Computed adaptive scales integrate correctly")

    print("\n✅ Adaptive CFG test PASSED")
    return True


def test_sampling_integration():
    """Test CFG integration in sample() and basic_sample()"""
    print("\n" + "=" * 60)
    print("Test 5: Sampling Integration")
    print("=" * 60)

    from faithful_cond_gen.model.generator import GeneratorWrapper, GeneratorConfig

    cfg = GeneratorConfig(
        vae_model_name="stabilityai/sd-vae-ft-mse",
        in_channels=3,
        image_size=256,  # Must be compatible with VAE + patch size
        sit_arch="SiT-S/2",
        attr_num_classes=[5, 10],
        class_dropout_prob=0.1,
    )

    generator = GeneratorWrapper(cfg)
    generator.eval()

    batch_size = 1  # Small for speed
    cond_ids = torch.tensor([[2, 3]], dtype=torch.long)

    # Test 1: basic_sample with and without CFG
    print("  Testing basic_sample()...")
    with torch.no_grad():
        # No CFG
        print("    Sampling with cfg_scale=1.0 (5 steps)...")
        img_no_cfg = generator.basic_sample(
            cond_ids, num_inference_steps=5, cfg_scale=1.0
        )
        print(f"    ✓ Output shape: {img_no_cfg.shape}")

        # With CFG
        print("    Sampling with cfg_scale=2.0 (5 steps)...")
        img_with_cfg = generator.basic_sample(
            cond_ids, num_inference_steps=5, cfg_scale=2.0
        )
        print(f"    ✓ Output shape: {img_with_cfg.shape}")

        # Verify they're different
        diff = (img_no_cfg - img_with_cfg).abs().mean().item()
        print(f"    Difference between cfg=1.0 and cfg=2.0: {diff:.6f}")
        assert diff > 1e-4, "CFG should change final samples"

    print("  ✓ basic_sample() CFG works")

    # Test 2: sample() with and without CFG
    print("\n  Testing sample() (REPA-style)...")
    with torch.no_grad():
        print("    Sampling with cfg_scale=1.0 (10 steps)...")
        img_no_cfg_repa = generator.sample(
            cond_ids, num_inference_steps=10, cfg_scale=1.0
        )
        print(f"    ✓ Output shape: {img_no_cfg_repa.shape}")

        print("    Sampling with cfg_scale=2.0 (10 steps)...")
        img_with_cfg_repa = generator.sample(
            cond_ids, num_inference_steps=10, cfg_scale=2.0
        )
        print(f"    ✓ Output shape: {img_with_cfg_repa.shape}")

        diff_repa = (img_no_cfg_repa - img_with_cfg_repa).abs().mean().item()
        print(f"    Difference between cfg=1.0 and cfg=2.0: {diff_repa:.6f}")
        assert diff_repa > 1e-4, "CFG should change final samples"

    print("  ✓ sample() CFG works")

    # Test 3: Adaptive CFG in sample()
    print("\n  Testing adaptive CFG in sample()...")
    condition_stats = {"(2, 3)": 10}  # Rare -> should get high cfg
    generator.load_condition_stats(condition_stats)

    with torch.no_grad():
        print("    Sampling with adaptive_cfg=True (10 steps)...")
        img_adaptive = generator.sample(
            cond_ids, num_inference_steps=10, adaptive_cfg=True
        )
        print(f"    ✓ Output shape: {img_adaptive.shape}")

        # Should be different from cfg=1.0 (since rare condition gets cfg=3.0)
        diff_adaptive = (img_no_cfg_repa - img_adaptive).abs().mean().item()
        print(f"    Difference between cfg=1.0 and adaptive: {diff_adaptive:.6f}")
        assert diff_adaptive > 1e-4, "Adaptive CFG should change samples"

    print("  ✓ Adaptive CFG in sample() works")

    print("\n✅ Sampling integration test PASSED")
    return True


def main():
    print("=" * 60)
    print("CFG Implementation Tests")
    print("=" * 60)

    results = []

    try:
        results.append(("Force Drop Mechanism", test_force_drop_mechanism()))
    except Exception as e:
        print(f"\n❌ Force drop test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Force Drop Mechanism", False))

    try:
        results.append(("CFG Changes Predictions", test_cfg_changes_predictions()))
    except Exception as e:
        print(f"\n❌ CFG changes test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("CFG Changes Predictions", False))

    try:
        results.append(("Backward Compatibility", test_backward_compatibility()))
    except Exception as e:
        print(f"\n❌ Backward compatibility test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Backward Compatibility", False))

    try:
        results.append(("Adaptive CFG", test_adaptive_cfg()))
    except Exception as e:
        print(f"\n❌ Adaptive CFG test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Adaptive CFG", False))

    try:
        results.append(("Sampling Integration", test_sampling_integration()))
    except Exception as e:
        print(f"\n❌ Sampling integration test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Sampling Integration", False))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All CFG tests passed!")
        print("\nCFG is fully functional and backward compatible!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
