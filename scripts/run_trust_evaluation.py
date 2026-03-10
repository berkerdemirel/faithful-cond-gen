"""
Trust Score Evaluation: Extended Analysis with Decile Binning.

Implements the evaluation layers from the research plan:
1. Condition-level ranking validity (T̄ vs KID correlation)
2. Failure detection (PR/AUROC for predicting bad conditions)
   - 2A: Condition-level OOD (seen vs unseen conditions)
   - 2B: Sample-level OOD (seen vs unseen samples, marginal models)
3. Real vs Generated OOD detection (sample-level)
4. Decile binning analysis
5. Correlation with Alaa et al. metrics
6. Multi-backbone aggregation

Usage:
    uv run python scripts/run_trust_evaluation.py --dataset celeba
    uv run python scripts/run_trust_evaluation.py --dataset celeba --normalize-features l2

Flags:
    --normalize-features {none,l2}  Apply L2 normalization to all features after loading.
                                    Default: none (backward-compatible).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import from refactored package
from faithful_cond_gen.eval.trust_eval.config import (
    CONDITION_ATTRS,
    FEATURE_CONFIGS,
    MARGINAL_SEEN_COMBOS,
)
from faithful_cond_gen.eval.trust_eval.diagnostics import (
    print_cosine_kid_transitivity_checks,
)
from faithful_cond_gen.eval.trust_eval.eval_layers import (
    create_realism_faithfulness_grids,
    evaluate_alaa_correlation,
    evaluate_celltype_classification,
    evaluate_controlled_perturbation_classification,
    evaluate_decile_binning,
    evaluate_downstream_bin_selection_from_scores,
    evaluate_failure_detection,
    evaluate_fpr95_selection,
    evaluate_fpr95_selection,
    evaluate_full_condition_ranking,
    evaluate_multi_backbone,
    evaluate_ranking_validity,
    evaluate_rxrx1_decomposed_classification,
    evaluate_sample_ood_detection,
    evaluate_seen_vs_unseen_detection,
    get_effective_kid_mode,
)
from faithful_cond_gen.eval.trust_eval.feature_io import (
    load_features_for_dataset,
    verify_feature_ordering,
)
from faithful_cond_gen.eval.trust_eval.reporting import create_report
from faithful_cond_gen.eval.trust_eval.scoring_core import (
    compute_trust_results_from_features,
)


def main():
    """
    Main entry point for trust score evaluation.

    Runs all evaluation layers and generates a comprehensive report.
    """
    parser = argparse.ArgumentParser(
        description="Trust Score Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default run (no normalization)
  python scripts/run_trust_evaluation.py --dataset celeba

  # With L2 normalization on all features
  python scripts/run_trust_evaluation.py --dataset celeba --normalize-features l2
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="celeba",
        choices=["celeba", "rxrx1"],
        help="Dataset to evaluate (default: celeba)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/trust_evaluation",
        help="Output directory for reports and artifacts (default: outputs/trust_evaluation)",
    )
    parser.add_argument(
        "--enable-grids",
        action="store_true",
        help="Enable 2x2 grid generation (Task 2, slower)",
    )
    parser.add_argument(
        "--normalize-features",
        type=str,
        choices=["none", "l2"],
        default="l2",
        help="Feature normalization mode. 'none': no normalization (default, backward-compatible). "
        "'l2': L2-normalize all feature vectors immediately after loading.",
    )
    parser.add_argument(
        "--scoring-method",
        type=str,
        choices=["mahalanobis", "knn"],
        default="mahalanobis",
        help="Scoring method for realism/faithfulness. 'mahalanobis': Mahalanobis distance (default). "
        "'knn': kNN radius-based scoring (sanity check alternative).",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=10,
        help="Number of neighbors for kNN scoring (only used if --scoring-method=knn). Default: 10.",
    )
    parser.add_argument(
        "--use-zkid",
        action="store_true",
        help="Use z-normalized KID instead of ΔKID for ranking evaluation (experimental).",
    )
    parser.add_argument(
        "--global-binning",
        action="store_true",
        help="Use legacy global binning instead of within-condition binning.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_keys = CONDITION_ATTRS.get(args.dataset, [])
    normalize_mode = args.normalize_features
    scoring_method = args.scoring_method
    knn_k = args.knn_k
    use_kid_z = args.use_zkid
    within_condition_binning = not args.global_binning

    print("=" * 60)
    print("TRUST SCORE EVALUATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Normalize features: {normalize_mode}")
    print(f"Scoring method: {scoring_method}")
    print(f"KID metric: {'z-normalized (experimental)' if use_kid_z else 'ΔKID'}")
    print(f"Binning mode: {'within-condition' if within_condition_binning else 'global'}")
    if scoring_method == "knn":
        print(f"  kNN k: {knn_k}")
    if normalize_mode == "l2":
        logger.info(
            "L2 normalization enabled: ALL feature vectors will be normalized after loading"
        )

    # Compute trust scores on the fly (from cached features)
    print("\nComputing trust scores on the fly (from cached features)...")
    all_results: List[Dict] = []
    feature_cache: Dict[
        Tuple[str, str], Tuple[torch.Tensor, Dict, torch.Tensor, Dict]
    ] = {}
    # Separate cache for KID features (always DINO for ground truth)
    kid_feature_cache: Dict[
        str, Tuple[torch.Tensor, Dict, torch.Tensor, Dict]
    ] = {}

    for (cfg_dataset, model, feature_type), _cfg in FEATURE_CONFIGS.items():
        if cfg_dataset != args.dataset:
            continue

        config_key = f"{model}/{feature_type}"
        print(f"  Computing trust scores for {config_key} ...")

        # Load features once per config (with optional normalization)
        real_feats, real_meta, gen_feats, gen_meta = load_features_for_dataset(
            args.dataset, model, feature_type, normalize_mode=normalize_mode
        )
        if real_feats is None or gen_feats is None:
            continue

        feature_cache[(model, feature_type)] = (
            real_feats,
            real_meta,
            gen_feats,
            gen_meta,
        )

        # For KID ground truth, always use DINO features (not aligned)
        # This ensures ΔKID is computed in a consistent, meaningful space
        if feature_type != "dinov3" and model not in kid_feature_cache:
            print(f"    Loading DINO features for KID ground truth ({model})...")
            dino_real, dino_real_meta, dino_gen, dino_gen_meta = load_features_for_dataset(
                args.dataset, model, "dinov3", normalize_mode=normalize_mode
            )
            if dino_real is not None and dino_gen is not None:
                kid_feature_cache[model] = (dino_real, dino_real_meta, dino_gen, dino_gen_meta)
                print(f"    ✓ DINO features loaded for KID: real={dino_real.shape}, gen={dino_gen.shape}")

        # For marginal models, restrict real calibration to seen combos
        filter_by_seen = "marginal" in model and args.dataset == "celeba"
        seen_combos = MARGINAL_SEEN_COMBOS if filter_by_seen else None

        trust_res = compute_trust_results_from_features(
            dataset=args.dataset,
            model=model,
            feature_type=feature_type,
            real_feats=real_feats,
            real_meta=real_meta,
            gen_feats=gen_feats,
            gen_meta=gen_meta,
            condition_keys=condition_keys,
            filter_by_seen=filter_by_seen,
            seen_combos=seen_combos,
            scoring_method=scoring_method,
            knn_k=knn_k,
        )
        all_results.append(trust_res)

    print(f"Computed trust results for {len(all_results)} configurations")

    # Group by (model, feature_type)
    by_config = {}
    for r in all_results:
        model = r["model"]
        feature_type = r.get("feature_type", r.get("encoder", "unknown"))
        config_key = f"{model}/{feature_type}"
        if config_key not in by_config:
            by_config[config_key] = []
        by_config[config_key].append(r)

    # Run evaluations
    ranking_results = {}
    failure_results = {}
    seen_unseen_results = {}
    alaa_results = {}
    multi_backbone = {}
    task1_results = {}
    task2_results = {}
    task3_results = {}
    task4_results = {}
    task5_results = {}
    fpr95_results = {}
    rxrx1_decomposed_results = {}

    for config_key in by_config:
        config_results = by_config[config_key]
        first_result = config_results[0]
        model = first_result["model"]
        feature_type = first_result.get(
            "feature_type", first_result.get("encoder", "unknown")
        )

        print(f"\n--- Evaluating {config_key} ---")

        # Determine effective KID mode based on normalization and feature type
        effective_kid_mode = get_effective_kid_mode(normalize_mode, feature_type)
        print(
            f"  KID mode: {effective_kid_mode} (normalize={normalize_mode}, feature_type={feature_type})"
        )

        # Use cached features for this configuration (already loaded above)
        cache_key = (model, feature_type)
        if cache_key not in feature_cache:
            logger.warning(f"No cached features for {cache_key}; skipping.")
            continue
        real_feats, real_meta, gen_feats, gen_meta = feature_cache[cache_key]

        # For KID computation, always use DINO features (ground truth metric)
        # Trust scores are computed in the feature space (aligned or dino)
        # but KID should always be in DINO space for valid quality measurement
        if feature_type != "dinov3" and model in kid_feature_cache:
            kid_real_feats, kid_real_meta, kid_gen_feats, kid_gen_meta = kid_feature_cache[model]
            kid_feature_type = "dinov3"
            print(f"  Using DINO features for KID ground truth (scoring in {feature_type} space)")
            # Verify ordering between scoring and KID features (abort if unverifiable)
            try:
                verified = verify_feature_ordering(
                    gen_meta, kid_gen_meta, f"scoring_{feature_type}", "kid_dinov3"
                )
                if not verified:
                    logger.error(f"FATAL: Cannot verify KID feature ordering (missing metadata)")
                    raise ValueError("KID feature ordering unverifiable - missing filename metadata")
                print(f"    ✓ Verified: scoring and KID features have matching sample order")
            except ValueError as e:
                logger.error(f"FATAL: KID feature ordering mismatch - {e}")
                raise
        else:
            kid_real_feats, kid_real_meta, kid_gen_feats = real_feats, real_meta, gen_feats
            kid_feature_type = feature_type

        # KID mode for DINO features
        kid_effective_mode = get_effective_kid_mode(normalize_mode, kid_feature_type)

        # Layer 1: Ranking validity
        if real_feats is not None and gen_feats is not None:
            print("  Layer 1: Ranking validity...")
            # Use DINO features for KID computation (ground truth)
            ranking_results[config_key] = evaluate_ranking_validity(
                first_result,
                kid_real_feats,
                kid_real_meta,
                kid_gen_feats,
                condition_keys,
                kid_feature_type,
                kid_mode=kid_effective_mode,
                use_kid_z=use_kid_z,
            )
            # Note: cosine diagnostic removed since we always use DINO for KID now

        # Layer 2: Failure detection (marginal models only)
        if "marginal" in model:
            print("  Layer 2: Failure detection...")
            failure_results[config_key] = evaluate_failure_detection(
                first_result, args.dataset
            )

        # Seen vs Unseen sample detection (marginal models only)
        if "marginal" in model:
            print("  Seen vs Unseen sample detection...")
            seen_unseen_results[config_key] = evaluate_seen_vs_unseen_detection(
                first_result, args.dataset, model, output_dir, config_key
            )

        # Layer 4: Alaa correlation
        print("  Layer 4: Alaa et al. correlation...")
        alaa_results[config_key] = evaluate_alaa_correlation(first_result)

        # Layer 5: Multi-backbone (per model, aggregating feature types)
        print("  Layer 5: Multi-backbone aggregation...")
        multi_backbone[config_key] = evaluate_multi_backbone(all_results, model)

        # Extension Tasks
        # Task 1: Full condition ranking + gap analysis (after Layer 1)
        if real_feats is not None and ranking_results.get(config_key):
            print("  Task 1: Full condition ranking + gap analysis...")
            task1_results[config_key] = evaluate_full_condition_ranking(
                first_result,
                ranking_results[config_key],
                args.dataset,
                condition_keys,
                output_dir,
                config_key,
            )

        # Task 2: 2×2 grids (optional, after Layer 1)
        if args.enable_grids and real_feats is not None:
            print("  Task 2: Creating realism/faithfulness grids...")
            model_dir = FEATURE_CONFIGS.get(
                (args.dataset, model, feature_type), [None]
            )[0]
            if model_dir:
                task2_results[config_key] = create_realism_faithfulness_grids(
                    first_result,
                    model_dir,
                    args.dataset,
                    condition_keys,
                    output_dir,
                    config_key,
                )

        # Task 3: Sample-based OOD detection (after Layer 1)
        if real_feats is not None and gen_feats is not None:
            print("  Task 3: Sample-based OOD detection (cross-fit)...")
            task3_results[config_key] = evaluate_sample_ood_detection(
                first_result,
                real_feats,
                real_meta,
                gen_feats,
                gen_meta,
                condition_keys,
                args.dataset,
                model,
                output_dir,
                config_key,
                scoring_method=scoring_method,
                knn_k=knn_k,
            )

        # Task 4: Decile binning with ablations (after Layer 3)
        if real_feats is not None and gen_feats is not None:
            print("  Task 4: Decile binning with ablations...")
            # Use DINO features for KID computation (ground truth metric)
            task4_results[config_key] = evaluate_decile_binning(
                first_result,
                kid_real_feats,
                kid_real_meta,
                kid_gen_feats,
                condition_keys,
                kid_feature_type,
                output_dir,
                config_key,
                kid_mode=kid_effective_mode,  # Use KID mode for DINO features
                within_condition=within_condition_binning,
                dataset=args.dataset,
            )

        # Task 5: Downstream bin-selection evaluation (celeba only)
        # For rxrx1, use decomposed classification and celltype classification instead
        if real_feats is not None and gen_feats is not None and args.dataset != "rxrx1":
            print("  Task 5: Downstream bin-selection evaluation...")

            # Downstream features always use dinov3 for fair comparison
            downstream_feature_type = "dinov3"

            if feature_type != downstream_feature_type:
                # Load dinov3 features for same generated samples
                cache_key_downstream = (model, downstream_feature_type)
                if cache_key_downstream in feature_cache:
                    real_feats_downstream, real_meta_downstream, gen_feats_downstream, gen_meta_downstream = feature_cache[
                        cache_key_downstream
                    ]
                else:
                    # Try to load dinov3 features for both real and gen
                    real_feats_downstream, real_meta_downstream, gen_feats_downstream, gen_meta_downstream = (
                        load_features_for_dataset(
                            args.dataset, model, downstream_feature_type, normalize_mode
                        )
                    )
            else:
                # Same feature space for scoring and downstream - no change needed
                gen_feats_downstream = gen_feats
                gen_meta_downstream = gen_meta
                real_feats_downstream = real_feats
                real_meta_downstream = real_meta

            if gen_feats_downstream is not None:
                # Verify feature ordering between scoring and downstream spaces (abort if unverifiable)
                try:
                    verified = verify_feature_ordering(
                        gen_meta,
                        gen_meta_downstream,
                        f"scoring_{feature_type}",
                        f"downstream_{downstream_feature_type}",
                    )
                    if not verified:
                        print(f"    ✗ SKIPPING Task 5: Cannot verify feature ordering (missing metadata)")
                        gen_feats_downstream = None
                    else:
                        print(
                            f"    ✓ Verified: scoring and downstream features have matching sample order"
                        )
                except ValueError as e:
                    print(f"    ✗ SKIPPING Task 5: Feature ordering mismatch - {e}")
                    gen_feats_downstream = None

            if gen_feats_downstream is not None:
                task5_results[config_key] = (
                    evaluate_downstream_bin_selection_from_scores(
                        trust_results=first_result,
                        gen_feats_downstream=gen_feats_downstream,
                        real_feats_downstream=real_feats_downstream,  # Now correctly in DINO space
                        real_meta=real_meta_downstream,  # Matching metadata
                        condition_keys=condition_keys,
                        model_name=model,
                        scoring_feature_type=feature_type,
                        downstream_feature_type=downstream_feature_type,
                        output_dir=output_dir,
                        dataset=args.dataset,
                    )
                )
            else:
                print(
                    f"    Skipping Task 5: could not load {downstream_feature_type} features for downstream task"
                )

        # FPR@95 selection evaluation
        if real_feats is not None and gen_feats is not None:
            print("  FPR@95 selection evaluation...")
            # Use DINO features for KID computation (ground truth metric)
            fpr95_results[config_key] = evaluate_fpr95_selection(
                trust_results=first_result,
                real_feats=kid_real_feats,
                real_meta=kid_real_meta,
                gen_feats=kid_gen_feats,
                gen_meta=gen_meta,
                condition_keys=condition_keys,
                dataset=args.dataset,
                model=model,
                output_dir=output_dir,
                config_key=config_key,
                kid_mode=kid_effective_mode,
                feature_type=kid_feature_type,
                scoring_method=scoring_method,
                use_kid_z=use_kid_z,
            )

        # RxRx1 decomposed classification (rxrx1 only)
        if args.dataset == "rxrx1" and gen_feats is not None:
            print("  RxRx1 decomposed classification...")
            rxrx1_decomposed_results[config_key] = evaluate_rxrx1_decomposed_classification(
                gen_feats=gen_feats,
                gen_meta=gen_meta,
                real_feats=real_feats,
                real_meta=real_meta,
                output_dir=output_dir,
                config_key=config_key,
                dataset=args.dataset,
            )

    # Create report
    print("\nGenerating report...")
    create_report(
        args.dataset,
        all_results,
        ranking_results,
        failure_results,
        seen_unseen_results,
        alaa_results,
        multi_backbone,
        task1_results,
        task2_results,
        task3_results,
        task4_results,
        output_dir,
        normalize_mode=normalize_mode,
        task5_results=task5_results,
    )

    # Save detailed results
    torch.save(
        {
            "ranking_results": ranking_results,
            "failure_results": failure_results,
            "seen_unseen_results": seen_unseen_results,
            "alaa_results": alaa_results,
            "multi_backbone": multi_backbone,
            "task1_results": task1_results,
            "task2_results": task2_results,
            "task3_results": task3_results,
            "task4_results": task4_results,
            "task5_results": task5_results,
            "fpr95_results": fpr95_results,
            "rxrx1_decomposed_results": rxrx1_decomposed_results,
        },
        output_dir / f"detailed_results_{args.dataset}.pt",
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    # Console-only transitivity sanity checks for aligned features
    if args.dataset == "celeba":
        # Pick a reference trust_results for each model (prefer dinov3 if available)
        for m in ["repa_full", "repa_marginal"]:
            ref = next(
                (
                    r
                    for r in all_results
                    if r["model"] == m
                    and (r.get("feature_type", r.get("encoder", "")) == "dinov3")
                ),
                None,
            )
            if ref is None:
                # fall back to any entry for that model
                ref = next((r for r in all_results if r["model"] == m), None)
            if ref is None:
                continue

            print_cosine_kid_transitivity_checks(
                dataset=args.dataset,
                model=m,
                condition_keys=condition_keys,
                ref_trust_results=ref,
                n_conditions=3,
                k=256,
                seed=0,
            )


if __name__ == "__main__":
    main()
