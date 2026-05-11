"""
Trust evaluation package.

This package provides modular trust score computation and evaluation
for conditional generative models.

Public API:
- condition_to_signature
- get_image_path
- create_image_grid
- compute_trust_results_from_features
- fit_trust_scoring_components
- score_trust_from_components
- compute_real_sample_scores
- calculate_kid_same_m
- bootstrap_kid_for_bin
- build_condition_class_map
- bin_samples_within_conditioning
- dedupe_generated
"""

# Public API exports - lazy import to avoid import-time side effects
from faithful_cond_gen.eval.trust_eval.image_utils import (
    condition_to_signature,
    get_image_path,
    create_image_grid,
)
from faithful_cond_gen.eval.trust_eval.metrics_kid import (
    calculate_kid_same_m,
    bootstrap_kid_for_bin,
)
from faithful_cond_gen.eval.trust_eval.condition_utils import (
    get_condition_key,
    build_condition_class_map,
    bin_samples_within_conditioning,
    filter_feats_and_meta_by_seen_combos,
)
from faithful_cond_gen.eval.trust_eval.scoring_core import (
    normalize_features,
    compute_mahalanobis,
    fit_global_stats,
    fit_factorized_stats,
    zscore,
    compute_real_calibration_for_global_energy,
    compute_real_calibration_for_factorized_margins,
    compute_global_realism_z,
    compute_factorized_faithfulness_margin_z,
    compute_alpha_precision_scores,
    compute_beta_recall_scores,
    compute_authenticity_scores,
    compute_trust_results_from_features,
    fit_trust_scoring_components,
    score_trust_from_components,
    compute_real_sample_scores,
    dedupe_generated,
)

__all__ = [
    # Image utilities
    "condition_to_signature",
    "get_image_path",
    "create_image_grid",
    # KID metrics
    "calculate_kid_same_m",
    "bootstrap_kid_for_bin",
    # Condition utilities
    "get_condition_key",
    "build_condition_class_map",
    "bin_samples_within_conditioning",
    "filter_feats_and_meta_by_seen_combos",
    # Scoring core
    "normalize_features",
    "compute_mahalanobis",
    "fit_global_stats",
    "fit_factorized_stats",
    "zscore",
    "compute_real_calibration_for_global_energy",
    "compute_real_calibration_for_factorized_margins",
    "compute_global_realism_z",
    "compute_factorized_faithfulness_margin_z",
    "compute_alpha_precision_scores",
    "compute_beta_recall_scores",
    "compute_authenticity_scores",
    "compute_trust_results_from_features",
    "fit_trust_scoring_components",
    "score_trust_from_components",
    "compute_real_sample_scores",
    "dedupe_generated",
]
