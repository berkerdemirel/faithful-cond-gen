"""
Evaluation layers for trust scoring.

Each module implements one evaluation layer or task from the research plan.
"""

from faithful_cond_gen.eval.trust_eval.eval_layers.ranking import (
    evaluate_ranking_validity,
    get_effective_kid_mode,
    topk_overlap,
    stratified_correlation,
)
from faithful_cond_gen.eval.trust_eval.eval_layers.failure_detection import (
    evaluate_seen_vs_unseen_detection,
    evaluate_failure_detection,
)
from faithful_cond_gen.eval.trust_eval.eval_layers.alaa_corr import (
    evaluate_alaa_correlation,
)
from faithful_cond_gen.eval.trust_eval.eval_layers.multi_backbone import (
    evaluate_multi_backbone,
)
from faithful_cond_gen.eval.trust_eval.eval_layers.full_ranking import (
    evaluate_full_condition_ranking,
)
from faithful_cond_gen.eval.trust_eval.eval_layers.grids import (
    create_realism_faithfulness_grids,
)
from faithful_cond_gen.eval.trust_eval.eval_layers.ood_real_vs_gen import (
    evaluate_sample_ood_detection,
    stratified_subsample_real,
)
from faithful_cond_gen.eval.trust_eval.eval_layers.binning import (
    evaluate_decile_binning,
)
from faithful_cond_gen.eval.trust_eval.eval_layers.downstream import (
    evaluate_downstream_bin_selection_from_scores,
    evaluate_celltype_classification,
    evaluate_controlled_perturbation_classification,
    evaluate_rxrx1_decomposed_classification,
)
from faithful_cond_gen.eval.trust_eval.eval_layers.fpr95_selection import (
    evaluate_fpr95_selection,
)

__all__ = [
    # Ranking
    "evaluate_ranking_validity",
    "get_effective_kid_mode",
    "topk_overlap",
    "stratified_correlation",
    # Failure detection
    "evaluate_seen_vs_unseen_detection",
    "evaluate_failure_detection",
    # Alaa correlation
    "evaluate_alaa_correlation",
    # Multi-backbone
    "evaluate_multi_backbone",
    # Full ranking
    "evaluate_full_condition_ranking",
    # Grids
    "create_realism_faithfulness_grids",
    # OOD detection
    "evaluate_sample_ood_detection",
    "stratified_subsample_real",
    # Binning
    "evaluate_decile_binning",
    # Downstream
    "evaluate_downstream_bin_selection_from_scores",
    "evaluate_celltype_classification",
    "evaluate_controlled_perturbation_classification",
    "evaluate_rxrx1_decomposed_classification",
    # FPR@95 Selection
    "evaluate_fpr95_selection",
]
