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
    RXRX1_HELDOUT_PAIRS,
)
from faithful_cond_gen.eval.trust_eval.diagnostics import (
    print_cosine_kid_transitivity_checks,
)
from faithful_cond_gen.eval.trust_eval.eval_layers import (
    create_realism_faithfulness_grids,
    evaluate_alaa_correlation,
    evaluate_decile_binning,
    evaluate_downstream_bin_selection_from_scores,
    evaluate_failure_detection,
    evaluate_fpr95_selection,
    evaluate_fpr95_selection,
    evaluate_full_condition_ranking,
    evaluate_multi_backbone,
    evaluate_ranking_validity,
    evaluate_rxrx1_decomposed_classification,
    evaluate_rxrx1_downstream_bin_selection,
    evaluate_sample_ood_detection,
    evaluate_seen_vs_unseen_detection,
    get_effective_kid_mode,
)
from faithful_cond_gen.eval.trust_eval.feature_io import (
    load_features_for_dataset,
    load_posthoc_kid_features,
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
        choices=["mahalanobis", "linear_probe", "clip", "knn_per_attr"],
        default="mahalanobis",
        help="Scoring method. 'mahalanobis' (default): global energy + per-attribute "
        "margin in feature space. 'linear_probe': per-attribute logistic-regression "
        "energy summed over attributes. 'clip': CLIP image-text alignment vs. joint "
        "combo prompts (CelebA only). 'knn_per_attr': cosine distance to k-th NN "
        "within target-class real subset, summed across attributes. All: lower = better.",
    )
    parser.add_argument(
        "--clip-cache-dir",
        type=str,
        default="outputs/real_celeba_clip-vitb16",
        help="Directory holding cached real CLIP features (train_features.pt). "
        "Per-model gen CLIP features are loaded from outputs/gen/{model_dir}/clip-vitb16_features.pt.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=5,
        help="k for --scoring-method knn_per_attr (k-th nearest neighbour distance). Default: 5.",
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
    use_kid_z = args.use_zkid
    within_condition_binning = not args.global_binning

    print("=" * 60)
    print("TRUST SCORE EVALUATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    if args.dataset == "rxrx1":
        from faithful_cond_gen.eval.trust_eval.subset_io import (
            load_rxrx1_subset,
            load_rxrx1_subset_arms,
        )
        rxrx1_subset = load_rxrx1_subset()
        rxrx1_arms = load_rxrx1_subset_arms()
        print(
            f"RxRx1 eval subset loaded: N={len(rxrx1_subset)} "
            f"(seen={len(rxrx1_arms['seen'])}, unseen={len(rxrx1_arms['unseen'])})"
        )
        print(f"  seen:   {sorted(rxrx1_arms['seen'])}")
        print(f"  unseen: {sorted(rxrx1_arms['unseen'])}")
    print(f"Normalize features: {normalize_mode}")
    print(f"Scoring method: {scoring_method}")
    print(f"KID metric: {'z-normalized (experimental)' if use_kid_z else 'ΔKID'}")
    print(f"Binning mode: {'within-condition' if within_condition_binning else 'global'}")
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

    if scoring_method == "clip" and args.dataset != "celeba":
        raise SystemExit(
            "--scoring-method clip is CelebA-only (requires per-condition text prompts)."
        )

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

        # CLIP scorer: swap the scoring features to cached CLIP image embeddings.
        # gen_dir for posthoc configs is suffixed with "_v1" — strip it because
        # CLIP features were extracted from the model's standard generation dir.
        if scoring_method == "clip":
            gen_dir, _filename = _cfg
            base_dir = gen_dir[:-3] if gen_dir.endswith("_v1") else gen_dir
            gen_clip_path = Path(f"outputs/gen/{base_dir}/clip-vitb16_features.pt")
            real_clip_path = Path(args.clip_cache_dir) / "train_features.pt"
            if not gen_clip_path.exists() or not real_clip_path.exists():
                logger.warning(
                    f"CLIP features missing (gen={gen_clip_path.exists()}, "
                    f"real={real_clip_path.exists()}); skipping {config_key}"
                )
                continue
            clip_real = torch.load(real_clip_path, map_location="cpu", weights_only=False)
            clip_gen = torch.load(gen_clip_path, map_location="cpu", weights_only=False)
            real_feats = clip_real["features"]
            real_meta = clip_real.get("metadata", real_meta)
            gen_feats = clip_gen["features"]
            gen_meta = clip_gen.get("metadata", gen_meta)
            print(f"    CLIP features loaded: real={real_feats.shape}, gen={gen_feats.shape}")

        feature_cache[(model, feature_type)] = (
            real_feats,
            real_meta,
            gen_feats,
            gen_meta,
        )

        # For KID ground truth, always use DINO features (not aligned)
        # This ensures ΔKID is computed in a consistent, meaningful space
        kid_cache_key = (model, feature_type) if feature_type == "posthoc_mapped" else (model, "dinov3")
        if feature_type != "dinov3" and kid_cache_key not in kid_feature_cache:
            print(f"    Loading DINO features for KID ground truth ({model}/{feature_type})...")
            if feature_type == "posthoc_mapped":
                # posthoc_mapped gen samples come from gen_cache, need matching DINO features
                dino_real, dino_real_meta, dino_gen, dino_gen_meta = load_posthoc_kid_features(
                    args.dataset, model, normalize_mode=normalize_mode
                )
            else:
                dino_real, dino_real_meta, dino_gen, dino_gen_meta = load_features_for_dataset(
                    args.dataset, model, "dinov3", normalize_mode=normalize_mode
                )
            if dino_real is not None and dino_gen is not None:
                kid_feature_cache[kid_cache_key] = (dino_real, dino_real_meta, dino_gen, dino_gen_meta)
                print(f"    DINO features loaded for KID: real={dino_real.shape}, gen={dino_gen.shape}")

        # For marginal models, restrict real calibration to seen combos
        if "marginal" in model and args.dataset == "celeba":
            seen_combos = MARGINAL_SEEN_COMBOS
        elif "marginal" in model and args.dataset == "rxrx1":
            # Seen = all real combos minus heldout pairs
            all_combos = set()
            for i in range(len(real_feats)):
                c = tuple(
                    int(real_meta[k][i].item() if isinstance(real_meta[k][i], torch.Tensor) else real_meta[k][i])
                    for k in condition_keys
                )
                all_combos.add(c)
            seen_combos = all_combos - RXRX1_HELDOUT_PAIRS
            logger.info(f"  RxRx1 marginal: {len(all_combos)} total combos, {len(RXRX1_HELDOUT_PAIRS)} heldout, {len(seen_combos)} seen")
        else:
            seen_combos = None
        filter_by_seen = seen_combos is not None

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
            knn_k=args.knn_k,
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
        kid_cache_key = (model, feature_type) if feature_type == "posthoc_mapped" else (model, "dinov3")
        if feature_type != "dinov3" and kid_cache_key in kid_feature_cache:
            kid_real_feats, kid_real_meta, kid_gen_feats, kid_gen_meta = kid_feature_cache[kid_cache_key]
            kid_feature_type = "dinov3"
            print(f"  Using DINO features for KID ground truth (scoring in {feature_type} space)")
            # Verify ordering between scoring and KID features (abort if unverifiable)
            try:
                verified_gen = verify_feature_ordering(
                    gen_meta, kid_gen_meta, f"scoring_{feature_type}", "kid_dinov3"
                )
                if not verified_gen:
                    logger.error(f"FATAL: Cannot verify KID gen feature ordering (missing metadata)")
                    raise ValueError("KID gen feature ordering unverifiable - missing filename metadata")
                print(f"    ✓ Verified: scoring and KID gen features have matching sample order")
                # For posthoc_mapped (and per-step posthoc_step{k}), real features
                # come from different sources (raw_hidden vs standard DINO) so size
                # mismatch is expected.
                if feature_type == "posthoc_mapped" or feature_type.startswith("posthoc_step"):
                    logger.info(f"  Skipping real verification for {feature_type} (different real sources)")
                else:
                    verified_real = verify_feature_ordering(
                        real_meta, kid_real_meta, f"real_{feature_type}", "real_kid_dinov3"
                    )
                    if not verified_real:
                        logger.warning(f"Cannot verify real feature ordering between {feature_type} and dinov3 (missing metadata)")
                    else:
                        print(f"    ✓ Verified: scoring and KID real features have matching sample order")
            except ValueError as e:
                logger.error(f"FATAL: Feature ordering mismatch - {e}")
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
            # For marginal models, use seen-combo filtering (matching gen_scores calibration)
            if "marginal" in model and args.dataset == "celeba":
                fpr95_seen_combos = MARGINAL_SEEN_COMBOS
            elif "marginal" in model and args.dataset == "rxrx1":
                all_combos = set()
                for i in range(len(real_feats)):
                    c = tuple(
                        int(real_meta[k][i].item() if isinstance(real_meta[k][i], torch.Tensor) else real_meta[k][i])
                        for k in condition_keys
                    )
                    all_combos.add(c)
                fpr95_seen_combos = all_combos - RXRX1_HELDOUT_PAIRS
            else:
                fpr95_seen_combos = None
            # Scoring-space features for threshold, DINO features for KID
            fpr95_results[config_key] = evaluate_fpr95_selection(
                trust_results=first_result,
                real_feats=real_feats,
                real_meta=real_meta,
                gen_feats=gen_feats,
                gen_meta=gen_meta,
                condition_keys=condition_keys,
                dataset=args.dataset,
                model=model,
                output_dir=output_dir,
                config_key=config_key,
                kid_real_feats=kid_real_feats,
                kid_gen_feats=kid_gen_feats,
                kid_real_meta=kid_real_meta if feature_type != "dinov3" else None,
                kid_mode=kid_effective_mode,
                feature_type=kid_feature_type,
                use_kid_z=use_kid_z,
                seen_combos=fpr95_seen_combos,
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

            # RxRx1 downstream bin-selection (Task 5 analog, celltype and subset modes)
            # Always use dinov3 features for classifier (scoring determines binning only)
            if first_result is not None:
                if feature_type == "dinov3":
                    ds_real, ds_real_meta, ds_gen, ds_gen_meta = real_feats, real_meta, gen_feats, gen_meta
                elif feature_type == "posthoc_mapped" and kid_cache_key in kid_feature_cache:
                    # posthoc_mapped gen rows come from the subset cache; the matching
                    # dinov3 lives in kid_feature_cache (load_posthoc_kid_features).
                    ds_real, ds_real_meta, ds_gen, ds_gen_meta = kid_feature_cache[kid_cache_key]
                else:
                    cache_key_downstream = (model, "dinov3")
                    if cache_key_downstream in feature_cache:
                        ds_real, ds_real_meta, ds_gen, ds_gen_meta = feature_cache[cache_key_downstream]
                    else:
                        ds_real, ds_real_meta, ds_gen, ds_gen_meta = load_features_for_dataset(
                            args.dataset, model, "dinov3", normalize_mode=normalize_mode
                        )

                n_trust = len(first_result["trust_updated"])
                if ds_gen is not None and len(ds_gen) != n_trust:
                    print(f"  Skipping RxRx1 downstream bin-selection: gen count mismatch "
                          f"(trust_results={n_trust}, dinov3={len(ds_gen)})")
                elif ds_gen is not None:
                    for rxrx1_bin_mode in ["celltype", "subset"]:
                        print(f"  RxRx1 downstream bin-selection ({rxrx1_bin_mode})...")
                        evaluate_rxrx1_downstream_bin_selection(
                            trust_results=first_result,
                            gen_feats=ds_gen,
                            gen_meta=ds_gen_meta,
                            real_feats=ds_real,
                            real_meta=ds_real_meta,
                            output_dir=output_dir,
                            config_key=config_key,
                            mode=rxrx1_bin_mode,
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
