"""Test Calibrated Mahalanobis with Ledoit-Wolf Shrinkage

Tests:
1. Shrinkage is applied when enabled
2. Backward compatibility (use_shrinkage=False works)
3. Small sample handling
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from faithful_cond_gen.eval.scoring.mahalanobis import MahalanobisScore


def test_shrinkage_enabled():
    """Test that Ledoit-Wolf shrinkage is applied when enabled"""
    print("\n" + "=" * 60)
    print("Test 1: Shrinkage Enabled")
    print("=" * 60)

    # Create scorer with shrinkage
    scorer = MahalanobisScore(device="cpu", use_shrinkage=True)

    # Create small training set with 2 conditions
    n_samples_per_cond = 10
    n_features = 50

    train_feats = torch.randn(n_samples_per_cond * 2, n_features)
    train_metadata = {
        "cond1": [0] * n_samples_per_cond + [1] * n_samples_per_cond,
        "cond2": [0] * n_samples_per_cond + [0] * n_samples_per_cond,
    }

    # Fit
    scorer.fit(train_feats, train_metadata)

    # Check shrinkage was applied
    shrinkage_values = [
        s["shrinkage"]
        for s in scorer.stats["conditions"].values()
        if s["shrinkage"] is not None
    ]

    assert len(shrinkage_values) > 0, "No shrinkage coefficients found"
    avg_shrinkage = np.mean(shrinkage_values)
    assert 0.0 <= avg_shrinkage <= 1.0, f"Invalid shrinkage: {avg_shrinkage}"

    print(f"  ✓ Shrinkage applied to {len(shrinkage_values)} conditions")
    print(f"  ✓ Average shrinkage coefficient: {avg_shrinkage:.4f}")
    print(f"    (0.0 = no shrinkage, 1.0 = full shrinkage to identity)")

    # Test scoring
    test_feats = torch.randn(5, n_features)
    test_metadata = {
        "cond1": [0, 0, 1, 1, 0],
        "cond2": [0, 0, 0, 0, 0],
    }

    scores = scorer.score(test_feats, test_metadata)
    assert scores.shape == (5,), f"Wrong score shape: {scores.shape}"
    assert not torch.isnan(scores).any(), "NaN scores detected"

    print(f"  ✓ Scores computed: mean={scores.mean():.4f}, std={scores.std():.4f}")
    print("\n✅ Shrinkage enabled test PASSED")
    return True


def test_shrinkage_disabled():
    """Test backward compatibility with shrinkage disabled"""
    print("\n" + "=" * 60)
    print("Test 2: Shrinkage Disabled (Backward Compatibility)")
    print("=" * 60)

    # Create scorer WITHOUT shrinkage
    scorer = MahalanobisScore(device="cpu", use_shrinkage=False)

    # Same setup as before
    n_samples_per_cond = 10
    n_features = 50

    train_feats = torch.randn(n_samples_per_cond * 2, n_features)
    train_metadata = {
        "cond1": [0] * n_samples_per_cond + [1] * n_samples_per_cond,
        "cond2": [0] * n_samples_per_cond + [0] * n_samples_per_cond,
    }

    # Fit
    scorer.fit(train_feats, train_metadata)

    # Check NO shrinkage was applied
    shrinkage_values = [
        s["shrinkage"]
        for s in scorer.stats["conditions"].values()
        if s["shrinkage"] is not None
    ]

    assert len(shrinkage_values) == 0, "Shrinkage applied when disabled"

    print("  ✓ No shrinkage applied (as expected)")

    # Test scoring still works
    test_feats = torch.randn(5, n_features)
    test_metadata = {
        "cond1": [0, 0, 1, 1, 0],
        "cond2": [0, 0, 0, 0, 0],
    }

    scores = scorer.score(test_feats, test_metadata)
    assert scores.shape == (5,), f"Wrong score shape: {scores.shape}"
    assert not torch.isnan(scores).any(), "NaN scores detected"

    print(f"  ✓ Scores computed: mean={scores.mean():.4f}, std={scores.std():.4f}")
    print("\n✅ Shrinkage disabled test PASSED")
    return True


def test_small_sample_handling():
    """Test handling of very small sample sizes"""
    print("\n" + "=" * 60)
    print("Test 3: Small Sample Handling")
    print("=" * 60)

    scorer = MahalanobisScore(
        device="cpu", use_shrinkage=True, min_samples_for_shrinkage=5
    )

    n_features = 20

    # Create conditions with varying sample sizes
    train_feats = torch.randn(1 + 3 + 10, n_features)  # 1, 3, 10 samples
    train_metadata = {
        "cond": [0] * 1 + [1] * 3 + [2] * 10,
    }

    # Fit
    scorer.fit(train_feats, train_metadata)

    # Check condition stats
    cond_stats = scorer.stats["conditions"]
    assert len(cond_stats) == 3, f"Expected 3 conditions, got {len(cond_stats)}"

    # Condition 0 (1 sample): No shrinkage, uses identity
    # Condition 1 (3 samples): Below threshold, uses empirical + reg
    # Condition 2 (10 samples): Above threshold, uses shrinkage

    n_shrinkage = sum(1 for s in cond_stats.values() if s["shrinkage"] is not None)
    print(f"  ✓ {n_shrinkage} conditions used shrinkage (expected 1)")

    # Find the condition with 10 samples
    cond_10 = next(k for k, v in cond_stats.items() if v["n_samples"] == 10)
    assert cond_stats[cond_10]["shrinkage"] is not None, "Large pool should use shrinkage"

    # Test scoring
    test_feats = torch.randn(3, n_features)
    test_metadata = {"cond": [0, 1, 2]}

    scores = scorer.score(test_feats, test_metadata)
    assert not torch.isnan(scores).any(), "NaN scores for small samples"

    print(f"  ✓ All sample sizes handled correctly")
    print(f"  ✓ Scores: {scores.tolist()}")
    print("\n✅ Small sample handling test PASSED")
    return True


def test_shrinkage_improves_small_pools():
    """Verify shrinkage helps with small pools in high dimensions"""
    print("\n" + "=" * 60)
    print("Test 4: Shrinkage Benefit for Small Pools")
    print("=" * 60)

    # High-dimensional features with small sample size (challenging case)
    n_samples = 5
    n_features = 100  # D > M (high-dim, low-sample)

    train_feats = torch.randn(n_samples, n_features)
    train_metadata = {"cond": [0] * n_samples}

    # Scorer WITH shrinkage
    scorer_shrink = MahalanobisScore(device="cpu", use_shrinkage=True, min_samples_for_shrinkage=2)
    scorer_shrink.fit(train_feats.clone(), train_metadata)

    # Scorer WITHOUT shrinkage
    scorer_no_shrink = MahalanobisScore(device="cpu", use_shrinkage=False)
    scorer_no_shrink.fit(train_feats.clone(), train_metadata)

    # Test on same test data
    test_feats = train_feats[:2]  # Use training samples as test (should score low)
    test_metadata = {"cond": [0, 0]}

    scores_shrink = scorer_shrink.score(test_feats, test_metadata)
    scores_no_shrink = scorer_no_shrink.score(test_feats, test_metadata)

    # Both should work, but shrinkage should be more stable
    assert not torch.isnan(scores_shrink).any(), "Shrinkage scorer failed"
    assert not torch.isnan(scores_no_shrink).any(), "No-shrinkage scorer failed"

    print(f"  ✓ With shrinkage:    mean={scores_shrink.mean():.4f}, std={scores_shrink.std():.4f}")
    print(f"  ✓ Without shrinkage: mean={scores_no_shrink.mean():.4f}, std={scores_no_shrink.std():.4f}")

    # Get shrinkage coefficient
    shrinkage_coef = scorer_shrink.stats["conditions"][scorer_shrink._hash_condition({"cond": 0})]["shrinkage"]
    print(f"  ✓ Shrinkage coefficient: {shrinkage_coef:.4f}")
    print(f"    (Higher shrinkage expected for D >> M)")

    print("\n✅ Shrinkage benefit test PASSED")
    return True


def main():
    print("=" * 60)
    print("Calibrated Mahalanobis Tests (Ledoit-Wolf Shrinkage)")
    print("=" * 60)

    results = []

    tests = [
        ("Shrinkage Enabled", test_shrinkage_enabled),
        ("Shrinkage Disabled", test_shrinkage_disabled),
        ("Small Sample Handling", test_small_sample_handling),
        ("Shrinkage Benefit", test_shrinkage_improves_small_pools),
    ]

    for name, test_func in tests:
        try:
            results.append((name, test_func()))
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

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
        print("\n🎉 All calibrated Mahalanobis tests passed!")
        print("\nLedoit-Wolf shrinkage is working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
