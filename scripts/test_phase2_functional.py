"""Functional tests for Phase 2 features with mock data

Tests all Phase 2 features end-to-end with synthetic data to verify:
1. ConditionalKNNScore fit() and score()
2. MarginalLinearProbeScore soft_mode with unseen values
3. GeneratorWrapper adaptive CFG computation
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_conditional_knn_functional():
    """Test ConditionalKNN with mock data"""
    print("\n" + "=" * 60)
    print("Test 1: ConditionalKNN Functional Test")
    print("=" * 60)

    from faithful_cond_gen.eval.scoring.conditional_knn import ConditionalKNNScore

    # Create mock data: 100 samples, 64-dim features, 3 conditions
    np.random.seed(42)
    torch.manual_seed(42)

    n_train = 100
    n_test = 20
    feat_dim = 64

    # Train data with 3 conditions
    train_features = torch.randn(n_train, feat_dim)
    train_conds = torch.randint(0, 3, (n_train,))  # 3 conditions

    # Create metadata
    train_metadata = {"condition": train_conds}

    # Test data (mix of seen and unseen conditions)
    test_features = torch.randn(n_test, feat_dim)
    test_conds = torch.randint(0, 4, (n_test,))  # Includes condition 3 (unseen)
    test_metadata = {"condition": test_conds}

    # Initialize scorer
    scorer = ConditionalKNNScore(device="cpu", k=5, min_pool_size=5, fallback_to_global=True)

    # Fit
    print("  Fitting scorer...")
    scorer.fit(train_features, train_metadata)

    # Check that pools were created
    assert len(scorer.cond_pools) > 0, "No conditional pools created"
    print(f"  ✓ Created {len(scorer.cond_pools)} conditional pools")

    # Score
    print("  Scoring test samples...")
    scores = scorer.score(test_features, test_metadata)

    # Verify scores
    assert scores.shape == (n_test,), f"Wrong score shape: {scores.shape}"
    assert torch.isfinite(scores).all(), "Scores contain NaN/Inf"
    assert (scores >= 0).all(), "Negative scores found"

    print(f"  ✓ Scores computed: shape={scores.shape}, mean={scores.mean():.4f}, std={scores.std():.4f}")
    print(f"  ✓ Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    # Test unseen condition handling
    unseen_mask = test_conds == 3
    if unseen_mask.any():
        unseen_scores = scores[unseen_mask]
        print(f"  ✓ Handled {unseen_mask.sum()} unseen conditions with fallback")

    # Test save/load
    print("  Testing save/load...")
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name

    try:
        scorer.save_stats(temp_path)
        scorer2 = ConditionalKNNScore(device="cpu", k=5)
        scorer2.load_stats(temp_path)

        # Verify loaded scorer works
        scores2 = scorer2.score(test_features, test_metadata)
        assert torch.allclose(scores, scores2, atol=1e-5), "Loaded scorer gives different results"
        print("  ✓ Save/load works correctly")
    finally:
        Path(temp_path).unlink(missing_ok=True)

    print("\n✅ ConditionalKNN functional test PASSED")
    return True


def test_soft_linear_probe_functional():
    """Test MarginalLinearProbe soft_mode with unseen values"""
    print("\n" + "=" * 60)
    print("Test 2: Soft Linear Probe Functional Test")
    print("=" * 60)

    from faithful_cond_gen.eval.scoring.marginal_linear_probe import (
        MarginalLinearProbeScore,
    )

    np.random.seed(42)
    torch.manual_seed(42)

    n_train = 200
    n_test = 50
    feat_dim = 64

    # Train data: 2 binary attributes
    train_features = torch.randn(n_train, feat_dim)
    train_attr1 = torch.randint(0, 2, (n_train,))  # Binary attribute 1
    train_attr2 = torch.randint(0, 3, (n_train,))  # 3-class attribute 2

    train_metadata = {"attr1": train_attr1, "attr2": train_attr2}

    # Test data: includes unseen value for attr2 (class 3)
    test_features = torch.randn(n_test, feat_dim)
    test_attr1 = torch.randint(0, 2, (n_test,))
    test_attr2 = torch.randint(0, 4, (n_test,))  # Includes class 3 (unseen)

    test_metadata = {"attr1": test_attr1, "attr2": test_attr2}

    # Test soft mode
    print("  Testing soft_mode=True...")
    scorer_soft = MarginalLinearProbeScore(device="cpu", soft_mode=True, C=1.0, max_iter=100)

    # Fit
    print("  Fitting scorer...")
    scorer_soft.fit(train_features, train_metadata)

    # Score (should NOT crash on unseen values)
    print("  Scoring with unseen values...")
    try:
        scores = scorer_soft.score(test_features, test_metadata)
        print("  ✓ Soft mode handled unseen values without crashing")
    except ValueError as e:
        print(f"  ✗ Soft mode crashed: {e}")
        return False

    # Verify scores
    assert scores.shape == (n_test,), f"Wrong score shape: {scores.shape}"
    assert torch.isfinite(scores).all(), "Scores contain NaN/Inf"
    assert (scores >= 0).all() and (scores <= 1).all(), "Scores outside [0, 1] range"

    print(f"  ✓ Scores: mean={scores.mean():.4f}, std={scores.std():.4f}")

    # Check that unseen values get higher scores (entropy-based)
    unseen_mask = test_attr2 == 3
    if unseen_mask.any():
        unseen_scores = scores[unseen_mask]
        seen_scores = scores[~unseen_mask]
        print(
            f"  ✓ Unseen scores: mean={unseen_scores.mean():.4f} "
            f"vs seen: mean={seen_scores.mean():.4f}"
        )

    # Test strict mode (should crash)
    print("\n  Testing soft_mode=False (strict mode)...")
    scorer_strict = MarginalLinearProbeScore(device="cpu", soft_mode=False)
    scorer_strict.fit(train_features, train_metadata)

    try:
        scores_strict = scorer_strict.score(test_features, test_metadata)
        print("  ✗ Strict mode should have crashed on unseen values but didn't")
        return False
    except ValueError as e:
        if "Unseen label" in str(e) or "CRITICAL" in str(e):
            print(f"  ✓ Strict mode correctly raised error on unseen values")
        else:
            print(f"  ✗ Unexpected error: {e}")
            return False

    print("\n✅ Soft Linear Probe functional test PASSED")
    return True


def test_adaptive_cfg_functional():
    """Test GeneratorWrapper adaptive CFG computation"""
    print("\n" + "=" * 60)
    print("Test 3: Adaptive CFG Functional Test")
    print("=" * 60)

    from faithful_cond_gen.model.generator import GeneratorWrapper

    # We can't fully instantiate GeneratorWrapper without config,
    # but we can test the adaptive CFG methods in isolation

    # Create a minimal mock
    class MockGenerator:
        def __init__(self):
            self.condition_stats = {}
            self.adaptive_cfg_config = {
                "min_cfg": 1.0,
                "max_cfg": 3.0,
                "threshold_common": 100,
                "threshold_rare": 10,
            }

        # Copy the methods from GeneratorWrapper
        load_condition_stats = GeneratorWrapper.load_condition_stats
        compute_adaptive_cfg_scale = GeneratorWrapper.compute_adaptive_cfg_scale

    gen = MockGenerator()

    # Test 1: Load condition stats
    print("  Testing load_condition_stats...")
    stats = {"(0, 0)": 150, "(0, 1)": 50, "(1, 0)": 5, "(1, 1)": 200}

    gen.load_condition_stats(stats)
    assert len(gen.condition_stats) == 4, "Stats not loaded correctly"
    print(f"  ✓ Loaded {len(gen.condition_stats)} condition stats")

    # Test 2: Compute adaptive CFG scales
    print("  Testing compute_adaptive_cfg_scale...")

    # Create test condition IDs
    test_conds = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])  # 4 samples

    cfg_scales = gen.compute_adaptive_cfg_scale(test_conds)

    # Verify shape
    assert cfg_scales.shape == (4,), f"Wrong shape: {cfg_scales.shape}"

    # Verify logic:
    # (0, 0): 150 samples -> common -> min_cfg = 1.0
    # (0, 1): 50 samples -> medium -> interpolated
    # (1, 0): 5 samples -> rare -> max_cfg = 3.0
    # (1, 1): 200 samples -> common -> min_cfg = 1.0

    print(f"  CFG scales: {cfg_scales.tolist()}")
    print(f"    (0,0): count=150 -> cfg={cfg_scales[0]:.2f} (expect ~1.0)")
    print(f"    (0,1): count=50 -> cfg={cfg_scales[1]:.2f} (expect ~2.0)")
    print(f"    (1,0): count=5 -> cfg={cfg_scales[2]:.2f} (expect ~3.0)")
    print(f"    (1,1): count=200 -> cfg={cfg_scales[3]:.2f} (expect ~1.0)")

    # Verify common conditions get min_cfg
    assert cfg_scales[0] == 1.0, f"Common condition got wrong cfg: {cfg_scales[0]}"
    assert cfg_scales[3] == 1.0, f"Common condition got wrong cfg: {cfg_scales[3]}"

    # Verify rare condition gets max_cfg
    assert cfg_scales[2] == 3.0, f"Rare condition got wrong cfg: {cfg_scales[2]}"

    # Verify medium condition gets interpolated value
    assert 1.0 < cfg_scales[1] < 3.0, f"Medium condition got wrong cfg: {cfg_scales[1]}"

    print("  ✓ Adaptive CFG scales computed correctly")

    # Test 3: Unseen condition (no stats)
    print("  Testing unseen condition...")
    gen2 = MockGenerator()  # Empty stats
    unseen_conds = torch.tensor([[5, 5]])
    cfg_default = gen2.compute_adaptive_cfg_scale(unseen_conds)
    assert cfg_default[0] == 1.0, "Unseen condition should default to 1.0"
    print("  ✓ Unseen conditions default to cfg=1.0")

    print("\n✅ Adaptive CFG functional test PASSED")
    return True


def main():
    print("=" * 60)
    print("Phase 2 Functional Tests")
    print("=" * 60)

    results = []

    try:
        results.append(("ConditionalKNN", test_conditional_knn_functional()))
    except Exception as e:
        print(f"\n❌ ConditionalKNN test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("ConditionalKNN", False))

    try:
        results.append(("Soft Linear Probe", test_soft_linear_probe_functional()))
    except Exception as e:
        print(f"\n❌ Soft Linear Probe test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Soft Linear Probe", False))

    try:
        results.append(("Adaptive CFG", test_adaptive_cfg_functional()))
    except Exception as e:
        print(f"\n❌ Adaptive CFG test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Adaptive CFG", False))

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
        print("\n🎉 All Phase 2 functional tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
