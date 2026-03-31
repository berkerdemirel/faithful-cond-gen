"""
Configuration constants for trust evaluation.

All constants are module-level definitions only - no import-time side effects.
"""

from pathlib import Path
from typing import Dict, Set, Tuple

import yaml

# Output directory for evaluation artifacts
OUTPUT_DIR = Path("outputs/trust_evaluation")

# Feature configurations - use meanpatch features for consistent KID comparison
# dinov3 uses REPAEncoder meanpatch to match aligned_mean feature space
FEATURE_CONFIGS: Dict[Tuple[str, str, str], Tuple[str, str]] = {
    # DINOv3 meanpatch features (consistent with REPA training)
    ("celeba", "vanilla_full", "dinov3"): (
        "celeba_vanilla_full",
        "dinov3_meanpatch_features.pt",
    ),
    ("celeba", "vanilla_marginal", "dinov3"): (
        "celeba_vanilla_marginal",
        "dinov3_meanpatch_features.pt",
    ),
    ("celeba", "repa_full", "dinov3"): (
        "celeba_repa_full",
        "dinov3_meanpatch_features.pt",
    ),
    ("celeba", "repa_marginal", "dinov3"): (
        "celeba_repa_marginal",
        "dinov3_meanpatch_features.pt",
    ),
    # Aligned mean features (from REPA training, now with correct ordering)
    ("celeba", "repa_full", "aligned_mean"): (
        "celeba_repa_full",
        "aligned_mean_features.pt",
    ),
    ("celeba", "repa_marginal", "aligned_mean"): (
        "celeba_repa_marginal",
        "aligned_mean_features.pt",
    ),
    # Pre-MLP hidden state ablation (raw SiT hidden at encoder_depth, before projector MLP)
    ("celeba", "repa_full", "pre_mlp"): (
        "celeba_repa_full_preMLP",
        "aligned_mean_features.pt",
    ),
    ("celeba", "repa_marginal", "pre_mlp"): (
        "celeba_repa_marginal_preMLP",
        "aligned_mean_features.pt",
    ),
    # CelebA REPA timestep ablation (aligned features at different denoising steps)
    # Uses separate model keys (_ts) so DINOv3 KID features come from same generated images
    ("celeba", "repa_full_ts", "dinov3"): (
        "celeba_repa_full_timesteps",
        "dinov3_meanpatch_features.pt",
    ),
    ("celeba", "repa_full_ts", "aligned_step0"): (
        "celeba_repa_full_timesteps",
        "aligned_mean_features_step0.pt",
    ),
    ("celeba", "repa_full_ts", "aligned_step83"): (
        "celeba_repa_full_timesteps",
        "aligned_mean_features_step83.pt",
    ),
    ("celeba", "repa_full_ts", "aligned_step166"): (
        "celeba_repa_full_timesteps",
        "aligned_mean_features_step166.pt",
    ),
    ("celeba", "repa_marginal_ts", "dinov3"): (
        "celeba_repa_marginal_timesteps",
        "dinov3_meanpatch_features.pt",
    ),
    ("celeba", "repa_marginal_ts", "aligned_step0"): (
        "celeba_repa_marginal_timesteps",
        "aligned_mean_features_step0.pt",
    ),
    ("celeba", "repa_marginal_ts", "aligned_step83"): (
        "celeba_repa_marginal_timesteps",
        "aligned_mean_features_step83.pt",
    ),
    ("celeba", "repa_marginal_ts", "aligned_step166"): (
        "celeba_repa_marginal_timesteps",
        "aligned_mean_features_step166.pt",
    ),
    # CelebA SigLIP teacher (alternative teacher ablation)
    (("celeba", "repa_siglip_full", "dinov3")): (
        "celeba_repa_siglip_full",
        "dinov3_meanpatch_features.pt",
    ),
    (("celeba", "repa_siglip_marginal", "dinov3")): (
        "celeba_repa_siglip_marginal",
        "dinov3_meanpatch_features.pt",
    ),
    (("celeba", "repa_siglip_full", "aligned_mean")): (
        "celeba_repa_siglip_full",
        "aligned_mean_features.pt",
    ),
    (("celeba", "repa_siglip_marginal", "aligned_mean")): (
        "celeba_repa_siglip_marginal",
        "aligned_mean_features.pt",
    ),
    # RxRx1 DINOv3 meanpatch features
    ("rxrx1", "vanilla_full", "dinov3"): (
        "rxrx1_vanilla_full",
        "dinov3_meanpatch_features.pt",
    ),
    ("rxrx1", "vanilla_marginal", "dinov3"): (
        "rxrx1_vanilla_marginal",
        "dinov3_meanpatch_features.pt",
    ),
    ("rxrx1", "repa_full", "dinov3"): (
        "rxrx1_repa_full",
        "dinov3_meanpatch_features.pt",
    ),
    ("rxrx1", "repa_marginal", "dinov3"): (
        "rxrx1_repa_marginal",
        "dinov3_meanpatch_features.pt",
    ),
    # RxRx1 Aligned mean features (from REPA training)
    ("rxrx1", "repa_full", "aligned_mean"): (
        "rxrx1_repa_full",
        "aligned_mean_features.pt",
    ),
    ("rxrx1", "repa_marginal", "aligned_mean"): (
        "rxrx1_repa_marginal",
        "aligned_mean_features.pt",
    ),
    # RxRx1 OpenPhenom teacher (alternative teacher ablation)
    ("rxrx1", "repa_openphenom_full", "dinov3"): (
        "rxrx1_repa_openphenom_full",
        "dinov3_meanpatch_features.pt",
    ),
    ("rxrx1", "repa_openphenom_marginal", "dinov3"): (
        "rxrx1_repa_openphenom_marginal",
        "dinov3_meanpatch_features.pt",
    ),
    ("rxrx1", "repa_openphenom_full", "aligned_mean"): (
        "rxrx1_repa_openphenom_full",
        "aligned_mean_features.pt",
    ),
    ("rxrx1", "repa_openphenom_marginal", "aligned_mean"): (
        "rxrx1_repa_openphenom_marginal",
        "aligned_mean_features.pt",
    ),
    # RxRx1 SigLIP teacher (alternative teacher ablation)
    ("rxrx1", "repa_siglip_full", "dinov3"): (
        "rxrx1_repa_siglip_full",
        "dinov3_meanpatch_features.pt",
    ),
    ("rxrx1", "repa_siglip_marginal", "dinov3"): (
        "rxrx1_repa_siglip_marginal",
        "dinov3_meanpatch_features.pt",
    ),
    ("rxrx1", "repa_siglip_full", "aligned_mean"): (
        "rxrx1_repa_siglip_full",
        "aligned_mean_features.pt",
    ),
    ("rxrx1", "repa_siglip_marginal", "aligned_mean"): (
        "rxrx1_repa_siglip_marginal",
        "aligned_mean_features.pt",
    ),
}

# Real feature paths - use meanpatch for dinov3 comparisons
REAL_FEATURE_PATHS: Dict[Tuple[str, str], str] = {
    ("celeba", "dinov3"): "outputs/real_celeba_dinov3_meanpatch/train_features.pt",
    ("rxrx1", "dinov3"): "outputs/real_rxrx1_dinov3_meanpatch/train_features.pt",
}

# Model-specific real feature paths (for aligned features where projector differs per model)
# Key: (dataset, model, feature_type) -> path
REAL_FEATURE_PATHS_BY_MODEL: Dict[Tuple[str, str, str], str] = {
    # CelebA aligned features - use model-specific projectors for real samples
    ("celeba", "repa_full", "aligned_mean"): "outputs/real_celeba_aligned/celeba_repa_full_v1/train_features.pt",
    ("celeba", "repa_marginal", "aligned_mean"): "outputs/real_celeba_aligned/celeba_repa_marginal_v1/train_features.pt",
    # RxRx1 aligned features
    ("rxrx1", "repa_full", "aligned_mean"): "outputs/real_rxrx1_aligned/rxrx1_repa_full_v1/train_features.pt",
    ("rxrx1", "repa_marginal", "aligned_mean"): "outputs/real_rxrx1_aligned/rxrx1_repa_marginal_v1/train_features.pt",
    # CelebA SigLIP real aligned features
    ("celeba", "repa_siglip_full", "aligned_mean"): "outputs/real_celeba_aligned/celeba_repa_siglip_full_v1/train_features.pt",
    ("celeba", "repa_siglip_marginal", "aligned_mean"): "outputs/real_celeba_aligned/celeba_repa_siglip_marginal_v1/train_features.pt",
    # CelebA pre-MLP hidden state ablation
    ("celeba", "repa_full", "pre_mlp"): "outputs/real_celeba_preMLP/celeba_repa_full_v1/train_features.pt",
    ("celeba", "repa_marginal", "pre_mlp"): "outputs/real_celeba_preMLP/celeba_repa_marginal_v1/train_features.pt",
    # RxRx1 OpenPhenom real aligned features
    ("rxrx1", "repa_openphenom_full", "aligned_mean"): "outputs/real_rxrx1_aligned/rxrx1_repa_openphenom_full_v1/train_features.pt",
    ("rxrx1", "repa_openphenom_marginal", "aligned_mean"): "outputs/real_rxrx1_aligned/rxrx1_repa_openphenom_marginal_v1/train_features.pt",
    # RxRx1 SigLIP real aligned features
    ("rxrx1", "repa_siglip_full", "aligned_mean"): "outputs/real_rxrx1_aligned/rxrx1_repa_siglip_full_v1/train_features.pt",
    ("rxrx1", "repa_siglip_marginal", "aligned_mean"): "outputs/real_rxrx1_aligned/rxrx1_repa_siglip_marginal_v1/train_features.pt",
    # CelebA timestep ablation — reuse same real features (same projector, different capture step)
    ("celeba", "repa_full_ts", "aligned_step0"): "outputs/real_celeba_aligned/celeba_repa_full_v1/train_features.pt",
    ("celeba", "repa_full_ts", "aligned_step83"): "outputs/real_celeba_aligned/celeba_repa_full_v1/train_features.pt",
    ("celeba", "repa_full_ts", "aligned_step166"): "outputs/real_celeba_aligned/celeba_repa_full_v1/train_features.pt",
    ("celeba", "repa_marginal_ts", "aligned_step0"): "outputs/real_celeba_aligned/celeba_repa_marginal_v1/train_features.pt",
    ("celeba", "repa_marginal_ts", "aligned_step83"): "outputs/real_celeba_aligned/celeba_repa_marginal_v1/train_features.pt",
    ("celeba", "repa_marginal_ts", "aligned_step166"): "outputs/real_celeba_aligned/celeba_repa_marginal_v1/train_features.pt",
}

# NOTE: For consistent KID computation, all features should use the same extraction method:
# - dinov3: REPAEncoder(dinov3-vit-l) mean-pooled patch tokens (not eval CLS/pooler)
# - aligned_mean: REPA aligned features mean-pooled (same encoder, extracted during generation)
# This ensures both use cosine-similar representations for fair comparison.

# Condition attributes per dataset
CONDITION_ATTRS: Dict[str, list] = {
    "celeba": ["Male", "Smiling", "Blond_Hair", "Eyeglasses"],
    "rxrx1": ["cell_type_id", "sirna_id"],
}

# Marginal model seen combos (for CelebA)
MARGINAL_SEEN_COMBOS: Set[Tuple[int, ...]] = {
    (0, 0, 0, 0),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
}


def load_rxrx1_heldout_pairs() -> Set[Tuple[int, int]]:
    """
    Load RxRx1 heldout pairs from the marginal config yaml.

    Returns:
        Set of (cell_type_id, sirna_id) tuples that are held out.
    """
    yaml_path = Path(__file__).parent.parent.parent.parent.parent / "configs/dataset/rxrx1_marginal.yaml"
    if not yaml_path.exists():
        return set()
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    pairs = cfg.get("held_out_pairs", [])
    return {(int(p[0]), int(p[1])) for p in pairs}


# RxRx1 heldout pairs (cell_type_id, sirna_id) - loaded from yaml
RXRX1_HELDOUT_PAIRS: Set[Tuple[int, int]] = load_rxrx1_heldout_pairs()

# Mixture realism component configurations
# Maps (dataset, model_type) to component grouping strategy
MIXTURE_COMPONENTS: Dict[Tuple[str, str], str] = {
    ("celeba", "full"): "condition",  # 16 condition combos
    ("celeba", "marginal"): "seen_condition",  # 5 seen combos
    ("rxrx1", "full"): "cell_type",  # 4 cell types
    ("rxrx1", "marginal"): "cell_type",  # 4 cell types
}
