"""Quick integration tests for Phase 2 features

Tests:
1. ConditionalKNNScore can be instantiated and has correct methods
2. MarginalLinearProbeScore has soft_mode parameter
3. GeneratorWrapper has adaptive CFG methods
4. run_ablation.py has correct structure
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_conditional_knn():
    """Test ConditionalKNNScore class structure"""
    print("Testing ConditionalKNNScore...")

    try:
        from faithful_cond_gen.eval.scoring.conditional_knn import ConditionalKNNScore

        # Check class can be instantiated
        scorer = ConditionalKNNScore(device="cpu", k=5)

        # Check key attributes
        assert hasattr(scorer, "cond_pools"), "Missing cond_pools attribute"
        assert hasattr(scorer, "global_pool"), "Missing global_pool attribute"
        assert hasattr(scorer, "fit"), "Missing fit method"
        assert hasattr(scorer, "score"), "Missing score method"
        assert hasattr(scorer, "save_stats"), "Missing save_stats method"
        assert hasattr(scorer, "load_stats"), "Missing load_stats method"

        # Check parameters
        assert scorer.k == 5, "k parameter not set correctly"
        assert scorer.min_pool_size == 10, "min_pool_size has wrong default"
        assert scorer.fallback_to_global == True, "fallback_to_global has wrong default"

        print("  ✓ ConditionalKNNScore structure OK")
        return True

    except Exception as e:
        print(f"  ✗ ConditionalKNNScore test failed: {e}")
        return False


def test_marginal_linear_probe_soft_mode():
    """Test MarginalLinearProbeScore soft_mode feature"""
    print("Testing MarginalLinearProbeScore soft_mode...")

    try:
        from faithful_cond_gen.eval.scoring.marginal_linear_probe import (
            MarginalLinearProbeScore,
        )

        # Check class can be instantiated with soft_mode
        scorer_soft = MarginalLinearProbeScore(device="cpu", soft_mode=True)
        scorer_strict = MarginalLinearProbeScore(device="cpu", soft_mode=False)

        # Check soft_mode attribute
        assert hasattr(scorer_soft, "soft_mode"), "Missing soft_mode attribute"
        assert scorer_soft.soft_mode == True, "soft_mode not set correctly"
        assert scorer_strict.soft_mode == False, "soft_mode not set correctly"

        # Check it has fit and score methods
        assert hasattr(scorer_soft, "fit"), "Missing fit method"
        assert hasattr(scorer_soft, "score"), "Missing score method"

        print("  ✓ MarginalLinearProbeScore soft_mode OK")
        return True

    except Exception as e:
        print(f"  ✗ MarginalLinearProbeScore test failed: {e}")
        return False


def test_generator_adaptive_cfg():
    """Test GeneratorWrapper adaptive CFG methods"""
    print("Testing GeneratorWrapper adaptive CFG...")

    try:
        from faithful_cond_gen.model.generator import GeneratorWrapper

        # Check class has adaptive CFG attributes and methods
        # We can't instantiate without full config, so just check the class
        assert hasattr(GeneratorWrapper, "load_condition_stats"), "Missing load_condition_stats method"
        assert hasattr(
            GeneratorWrapper, "compute_adaptive_cfg_scale"
        ), "Missing compute_adaptive_cfg_scale method"

        # Check sample method signature includes adaptive_cfg
        import inspect

        sample_sig = inspect.signature(GeneratorWrapper.sample)
        assert "adaptive_cfg" in sample_sig.parameters, "sample() missing adaptive_cfg parameter"

        print("  ✓ GeneratorWrapper adaptive CFG methods OK")
        return True

    except Exception as e:
        print(f"  ✗ GeneratorWrapper test failed: {e}")
        return False


def test_ablation_script():
    """Test run_ablation.py structure"""
    print("Testing run_ablation.py...")

    try:
        # Import the module
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "run_ablation", Path(__file__).parent / "run_ablation.py"
        )
        ablation_module = importlib.util.module_from_spec(spec)

        # Check key functions exist
        assert hasattr(ablation_module, "run_ablation"), "Missing run_ablation function"
        assert hasattr(ablation_module, "load_scoring_result"), "Missing load_scoring_result function"
        assert hasattr(ablation_module, "extract_metrics"), "Missing extract_metrics function"
        assert hasattr(ablation_module, "find_score_files"), "Missing find_score_files function"

        print("  ✓ run_ablation.py structure OK")
        return True

    except Exception as e:
        print(f"  ✗ run_ablation.py test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Phase 2 Integration Tests")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("ConditionalKNN", test_conditional_knn()))
    results.append(("MarginalLinearProbe soft_mode", test_marginal_linear_probe_soft_mode()))
    results.append(("GeneratorWrapper adaptive CFG", test_generator_adaptive_cfg()))
    results.append(("run_ablation.py", test_ablation_script()))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All Phase 2 tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
