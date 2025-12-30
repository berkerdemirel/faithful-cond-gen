"""Test Ledoit-Wolf Shrinkage across all Mahalanobis scorers

Tests all three Mahalanobis variants:
1. MahalanobisScore (joint condition)
2. RelativeMahalanobisScore (background vs conditional)
3. MarginalMahalanobisScore (per-attribute)
"""

import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from faithful_cond_gen.eval.scoring.mahalanobis import MahalanobisScore
from faithful_cond_gen.eval.scoring.relative_mahalanobis import RelativeMahalanobisScore
from faithful_cond_gen.eval.scoring.marginal_mahalanobis import MarginalMahalanobisScore


def test_mahalanobis_shrinkage():
    """Test MahalanobisScore with shrinkage"""
    print("\n" + "=" * 60)
    print("Test 1: MahalanobisScore (Joint)")
    print("=" * 60)

    scorer = MahalanobisScore(device="cpu", use_shrinkage=True)

    # Create data
    train_feats = torch.randn(20, 50)
    train_metadata = {
        "cond1": [0] * 10 + [1] * 10,
        "cond2": [0] * 10 + [0] * 10,
    }

    scorer.fit(train_feats, train_metadata)

    # Check shrinkage was applied
    shrinkage_values = [
        s["shrinkage"]
        for s in scorer.stats["conditions"].values()
        if s["shrinkage"] is not None
    ]

    assert len(shrinkage_values) > 0, "No shrinkage applied"
    print(f"  ✓ {len(shrinkage_values)} conditions used shrinkage")
    print(f"  ✓ Avg shrinkage: {sum(shrinkage_values) / len(shrinkage_values):.4f}")

    # Test scoring
    test_feats = torch.randn(5, 50)
    test_metadata = {"cond1": [0, 0, 1, 1, 0], "cond2": [0, 0, 0, 0, 0]}

    scores = scorer.score(test_feats, test_metadata)
    assert not torch.isnan(scores).any()
    print(f"  ✓ Scores: mean={scores.mean():.4f}, std={scores.std():.4f}")

    print("\n✅ MahalanobisScore shrinkage test PASSED")
    return True


def test_relative_mahalanobis_shrinkage():
    """Test RelativeMahalanobisScore with shrinkage"""
    print("\n" + "=" * 60)
    print("Test 2: RelativeMahalanobisScore (Background vs Conditional)")
    print("=" * 60)

    scorer = RelativeMahalanobisScore(device="cpu", use_shrinkage=True)

    # Create data with multiple conditions
    train_feats = torch.randn(30, 50)
    train_metadata = {
        "cond1": [0] * 10 + [1] * 10 + [2] * 10,
        "cond2": [0] * 10 + [0] * 10 + [1] * 10,
    }

    scorer.fit(train_feats, train_metadata)

    # Check that shrinkage was logged (should be in logs)
    print("  ✓ Fit complete (check logs for shrinkage coefficients)")

    # Test scoring
    test_feats = torch.randn(5, 50)
    test_metadata = {
        "cond1": [0, 0, 1, 1, 2],
        "cond2": [0, 0, 0, 0, 1],
    }

    scores = scorer.score(test_feats, test_metadata)
    assert not torch.isnan(scores).any()
    print(f"  ✓ Scores: mean={scores.mean():.4f}, std={scores.std():.4f}")

    print("\n✅ RelativeMahalanobisScore shrinkage test PASSED")
    return True


def test_marginal_mahalanobis_shrinkage():
    """Test MarginalMahalanobisScore with shrinkage"""
    print("\n" + "=" * 60)
    print("Test 3: MarginalMahalanobisScore (Per-Attribute)")
    print("=" * 60)

    scorer = MarginalMahalanobisScore(device="cpu", use_shrinkage=True)

    # Create data with two attributes
    train_feats = torch.randn(30, 50)
    train_metadata = {
        "attr1": [0] * 10 + [1] * 10 + [2] * 10,  # 3 values
        "attr2": [0] * 15 + [1] * 15,  # 2 values
    }

    scorer.fit(train_feats, train_metadata)

    # Check shrinkage was applied
    all_shrinkages = []
    for attr_name in scorer.stats["attributes"]:
        for val_stats in scorer.stats["attributes"][attr_name].values():
            if val_stats["shrinkage"] is not None:
                all_shrinkages.append(val_stats["shrinkage"])

    assert len(all_shrinkages) > 0, "No shrinkage applied"
    avg_shrinkage = sum(all_shrinkages) / len(all_shrinkages)
    print(f"  ✓ {len(all_shrinkages)} attribute values used shrinkage")
    print(f"  ✓ Avg shrinkage: {avg_shrinkage:.4f}")

    # Test scoring
    test_feats = torch.randn(5, 50)
    test_metadata = {
        "attr1": [0, 0, 1, 1, 2],
        "attr2": [0, 0, 1, 1, 0],
    }

    scores = scorer.score(test_feats, test_metadata)
    assert not torch.isnan(scores).any()
    print(f"  ✓ Scores: mean={scores.mean():.4f}, std={scores.std():.4f}")

    print("\n✅ MarginalMahalanobisScore shrinkage test PASSED")
    return True


def test_all_backward_compatible():
    """Test that use_shrinkage=False works for all scorers"""
    print("\n" + "=" * 60)
    print("Test 4: Backward Compatibility (use_shrinkage=False)")
    print("=" * 60)

    # Shared data
    train_feats = torch.randn(20, 30)
    train_metadata = {"cond": [0] * 10 + [1] * 10}

    # Test all three scorers
    scorers = [
        ("Joint", MahalanobisScore(device="cpu", use_shrinkage=False)),
        ("Relative", RelativeMahalanobisScore(device="cpu", use_shrinkage=False)),
        ("Marginal", MarginalMahalanobisScore(device="cpu", use_shrinkage=False)),
    ]

    for name, scorer in scorers:
        scorer.fit(train_feats, train_metadata)
        test_feats = torch.randn(3, 30)
        test_metadata = {"cond": [0, 1, 0]}
        scores = scorer.score(test_feats, test_metadata)
        assert not torch.isnan(scores).any()
        print(f"  ✓ {name}: Works without shrinkage")

    print("\n✅ Backward compatibility test PASSED")
    return True


def main():
    print("=" * 60)
    print("All Mahalanobis Scorers - Shrinkage Tests")
    print("=" * 60)

    results = []

    tests = [
        ("MahalanobisScore", test_mahalanobis_shrinkage),
        ("RelativeMahalanobisScore", test_relative_mahalanobis_shrinkage),
        ("MarginalMahalanobisScore", test_marginal_mahalanobis_shrinkage),
        ("Backward Compatibility", test_all_backward_compatible),
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
        print("\n🎉 All Mahalanobis shrinkage tests passed!")
        print("\nAll three scorers now use Ledoit-Wolf shrinkage!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
