"""Analyze DINO Alignment Performance Across Timesteps.

For REPA models, this script generates samples while extracting aligned features at
EACH timestep, then compares with final DINO features to measure alignment quality.

KEY BEHAVIORS:
- Uses the REAL generator.sample() method (no custom reimplementation)
- Captures aligned features at every timestep during Stage 1 (SDE) plus t_cutoff
- Timestep range: t=1.0 → t=0.04 (t_cutoff), NOT including true t=0.0
  (Model is never called at t=0.0 in REPA sampling, so no features to extract there)
- Seen/unseen definition: Hamming weight <= 1 (sum of bits <= 1) for seen

Usage:
    # Full models (single plot for all conditions)
    PYTHONPATH=src uv run python scripts/analyze_dino_alignment.py \\
        --checkpoint-key celeba_repa_full_v1 \\
        --output-dir outputs/alignment_analysis \\
        --samples-per-condition 50

    # Marginal models (separate plots for seen vs unseen based on Hamming weight)
    PYTHONPATH=src uv run python scripts/analyze_dino_alignment.py \\
        --checkpoint-key celeba_repa_marginal_v1 \\
        --output-dir outputs/alignment_analysis \\
        --marginal \\
        --samples-per-condition 50

Safety checks:
- Verifies decoded images are in [0, 1] range
- Validates REPA is enabled in checkpoint
- Checks seen/unseen split is order-invariant (Hamming weight based)

Output artifacts:
- PNG plots: mean ± std cosine similarity vs timestep
- CSV: timestep, mean_similarity, std_similarity (optionally split by seen/unseen)
- PT: raw per-condition per-timestep similarities for further analysis
"""

import argparse
import itertools
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from faithful_cond_gen.data.celeba import CelebaDataConfig, CelebaDataModule
from faithful_cond_gen.model.generator import GeneratorWrapper
from faithful_cond_gen.model.repa_encoder import REPAEncoder
from faithful_cond_gen.pl_modules.generator_pl import GeneratorPL
from faithful_cond_gen.utils.checkpoints import get_checkpoint_path
from hydra.utils import instantiate
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_repa_model(checkpoint_key: str, device: str = "cuda"):
    """Load REPA model from checkpoint using proper Hydra configs."""
    log.info(f"Loading model from checkpoint key: {checkpoint_key}")
    ckpt_path = get_checkpoint_path(checkpoint_key)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    log.info(f"Checkpoint path: {ckpt_path}")

    # Load config from Hydra config file (like generate_samples.py does)
    config_root = Path(__file__).parent.parent / "configs"

    if "celeba" in checkpoint_key.lower():
        model_cfg_path = config_root / "model" / "generator_celeba.yaml"
    elif "rxrx1" in checkpoint_key.lower():
        model_cfg_path = config_root / "model" / "generator_rxrx1.yaml"
    else:
        raise ValueError(
            f"Cannot determine dataset from checkpoint key: {checkpoint_key}"
        )

    log.info(f"Loading generator config from: {model_cfg_path}")
    gen_cfg_dict = OmegaConf.load(model_cfg_path)

    # Enable REPA for REPA checkpoints
    if "repa" in checkpoint_key.lower():
        gen_cfg_dict.use_repa = True
        if gen_cfg_dict.repa_proj_coeff == 0.0:
            gen_cfg_dict.repa_proj_coeff = 0.5  # Match training setting

    # Instantiate config
    gen_cfg_obj = instantiate(gen_cfg_dict)

    # Create generator
    generator = GeneratorWrapper(gen_cfg_obj)

    # Load full PL module with strict=False (like generate_samples.py)
    pl_module = GeneratorPL.load_from_checkpoint(
        ckpt_path,
        generator=generator,
        map_location=device,
        strict=False,
    )
    pl_module.to(device)
    pl_module.eval()

    # Apply EMA if available
    if hasattr(pl_module, "ema"):
        log.info("Applying EMA weights...")
        pl_module.ema.apply()

    # Verify REPA is enabled
    if not pl_module.generator.cfg.use_repa:
        raise ValueError("Model does not have REPA enabled!")

    return pl_module, gen_cfg_obj


def load_dino_encoder(encoder_name: str = "dinov3-vit-l", device: str = "cuda"):
    """Load DINO encoder for final feature extraction.

    IMPORTANT: Must use the SAME encoder as was used during REPA training.
    """
    log.info(f"Loading DINO encoder for final feature extraction: {encoder_name}")
    encoder = REPAEncoder(
        encoder_name=encoder_name,
        resolution=256,
        in_channels=3,
        target_grid=16,
        device=device,
    )
    encoder.to(device)
    encoder.eval()
    return encoder


def is_seen_condition(cond_tuple: Tuple[int, ...]) -> bool:
    """Check if condition is 'seen' based on Hamming weight.

    Seen iff Hamming weight <= 1 (i.e., at most one attribute is 1).
    This is order-invariant and matches the marginal training support.

    Args:
        cond_tuple: Tuple of binary attribute values (0 or 1)

    Returns:
        True if seen (Hamming weight <= 1), False if unseen
    """
    hamming_weight = sum(cond_tuple)
    return hamming_weight <= 1


def get_celeba_conditions(is_marginal: bool = False) -> Tuple[List[Tuple], Dict]:
    """Get CelebA conditioning combinations.

    Returns:
        conditions: List of (Male, Smiling, Blond_Hair, Eyeglasses) tuples (sorted order)
        metadata: Dict mapping condition -> "seen"/"unseen" label
    """
    # IMPORTANT: Use sorted order for canonical display
    attrs = sorted(["Male", "Smiling", "Blond_Hair", "Eyeglasses"])
    all_combos = list(itertools.product([0, 1], repeat=len(attrs)))

    metadata = {}
    if is_marginal:
        for combo in all_combos:
            # Use Hamming weight to determine seen/unseen (order-invariant)
            if is_seen_condition(combo):
                metadata[combo] = "seen"
            else:
                metadata[combo] = "unseen"
    else:
        for combo in all_combos:
            metadata[combo] = "all"

    log.info(f"Canonical attribute order: {attrs}")
    return all_combos, metadata


@torch.no_grad()
def extract_dino_features(images: torch.Tensor, dino_encoder):
    """Extract DINO features from final images.

    Args:
        images: (B, 3, H, W) in [0, 1]
        dino_encoder: REPAEncoder instance

    Returns:
        features: (B, D) pooled DINO features
    """
    # Safety check: verify images are in [0, 1]
    if images.min() < -0.01 or images.max() > 1.01:
        log.warning(
            f"⚠️  Images out of [0,1] range: min={images.min():.3f}, max={images.max():.3f}. "
            f"Clamping to [0,1]."
        )
        images = images.clamp(0.0, 1.0)

    # REPAEncoder expects images in [0, 1] and handles normalization internally
    patch_features = dino_encoder(images)  # (B, num_patches, D)
    # Average pool over patches
    pooled_features = patch_features.mean(dim=1)  # (B, D)
    return pooled_features


def compute_cosine_similarity(
    feat_a: torch.Tensor, feat_b: torch.Tensor
) -> torch.Tensor:
    """Compute cosine similarity between two feature sets.

    Args:
        feat_a: (B, D)
        feat_b: (B, D)

    Returns:
        similarity: (B,) cosine similarities
    """
    return F.cosine_similarity(feat_a, feat_b, dim=1)


def analyze_alignment_for_conditions(
    generator,
    dino_encoder,
    conditions: List[Tuple],
    samples_per_condition: int = 50,
    batch_size: int = 8,
    num_inference_steps: int = 250,
    device: str = "cuda",
) -> Dict[Tuple, Dict]:
    """Analyze alignment for a set of conditions using REAL generator.sample().

    Returns:
        results: Dict mapping condition -> {
            'step_indices': List[int],
            'timestep_values': List[float],
            'similarities': List[List[float]],  # [step_idx][sample_idx]
        }
    """
    results = {}

    for cond in tqdm(conditions, desc="Analyzing conditions"):
        cond_tensor = torch.tensor(cond, device=device).long()

        all_similarities_by_step = {}  # step_idx -> List[similarities]
        step_indices = None
        timestep_values = None

        # Generate samples in batches
        num_batches = (samples_per_condition + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            current_bs = min(batch_size, samples_per_condition - batch_idx * batch_size)
            batch_cond = cond_tensor.unsqueeze(0).repeat(current_bs, 1)

            # Generate with feature capture at ALL steps using REAL sampler with integer indices
            images, aligned_by_idx = generator.sample(
                batch_cond,
                num_inference_steps=num_inference_steps,
                return_aligned_features=True,
                feature_capture_idx="all",  # Capture at every step index
            )

            # Safety check: verify images are in [0, 1]
            if images.min() < -0.01 or images.max() > 1.01:
                log.warning(
                    f"⚠️  Generated images out of [0,1] range for condition {cond}: "
                    f"min={images.min():.3f}, max={images.max():.3f}"
                )
                images = images.clamp(0.0, 1.0)

            # Extract final DINO features
            final_dino_feats = extract_dino_features(images, dino_encoder)  # (B, D)

            # Compute cosine similarity for each step
            if step_indices is None:
                step_indices = [k_idx for k_idx, _, _ in aligned_by_idx]
                timestep_values = [t_val for _, t_val, _ in aligned_by_idx]

            for k_idx, t_val, aligned_feats_pooled in aligned_by_idx:
                # aligned_feats_pooled is already (B, D) and on CPU from generator
                aligned_feats_pooled = aligned_feats_pooled.to(device)
                sims = compute_cosine_similarity(
                    aligned_feats_pooled, final_dino_feats
                )  # (B,)

                if k_idx not in all_similarities_by_step:
                    all_similarities_by_step[k_idx] = []
                all_similarities_by_step[k_idx].extend(sims.cpu().tolist())

        # Store results for this condition
        results[cond] = {
            "step_indices": step_indices,
            "timestep_values": timestep_values,
            "similarities": [all_similarities_by_step[k] for k in step_indices],
        }

    return results


def aggregate_results(
    results: Dict[Tuple, Dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate results across conditions.

    Returns:
        step_indices: (T,) array of step indices
        timestep_values: (T,) array of timestep values at each step
        mean_similarities: (T,) mean cosine similarity per step
        std_similarities: (T,) std dev per step
    """
    # Get step info from first condition
    first_cond = next(iter(results.values()))
    step_indices = np.array(first_cond["step_indices"])
    timestep_values = np.array(first_cond["timestep_values"])
    num_steps = len(step_indices)

    # Collect all similarities per step
    all_sims_by_step = [[] for _ in range(num_steps)]

    for cond_result in results.values():
        for step_idx, sims in enumerate(cond_result["similarities"]):
            all_sims_by_step[step_idx].extend(sims)

    # Compute mean and std
    mean_sims = np.array([np.mean(sims) for sims in all_sims_by_step])
    std_sims = np.array([np.std(sims) for sims in all_sims_by_step])

    return step_indices, timestep_values, mean_sims, std_sims


def plot_alignment_curve(
    timesteps: np.ndarray,
    mean_sims: np.ndarray,
    std_sims: np.ndarray,
    title: str,
    output_path: str,
):
    """Plot alignment curve with confidence bands."""
    plt.figure(figsize=(10, 6))

    # Plot mean with shaded std
    plt.plot(timesteps, mean_sims, "b-", linewidth=2, label="Mean Cosine Similarity")
    plt.fill_between(
        timesteps,
        mean_sims - std_sims,
        mean_sims + std_sims,
        alpha=0.3,
        color="b",
        label="±1 Std Dev",
    )

    plt.xlabel("Timestep (t)", fontsize=12)
    plt.ylabel("Cosine Similarity (Aligned vs Final DINO)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    log.info(f"Saved plot to {output_path}")


def save_csv(
    timesteps: np.ndarray,
    mean_sims: np.ndarray,
    std_sims: np.ndarray,
    output_path: str,
    group_label: str = "all",
):
    """Save alignment data to CSV."""
    df = pd.DataFrame(
        {
            "timestep": timesteps,
            "mean_similarity": mean_sims,
            "std_similarity": std_sims,
            "group": group_label,
        }
    )
    df.to_csv(output_path, index=False)
    log.info(f"Saved CSV to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze DINO alignment across timesteps"
    )
    parser.add_argument(
        "--checkpoint-key", type=str, required=True, help="Checkpoint key for model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/alignment_analysis",
        help="Output directory",
    )
    parser.add_argument(
        "--marginal", action="store_true", help="Model is marginal (split seen/unseen)"
    )
    parser.add_argument(
        "--samples-per-condition", type=int, default=50, help="Samples per condition"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for generation"
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=250, help="Number of inference steps"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    pl_module, gen_cfg = load_repa_model(args.checkpoint_key, args.device)
    generator = pl_module.generator

    # Load DINO encoder (must match training encoder)
    encoder_name = gen_cfg.repa_encoder
    log.info(f"Model was trained with REPA encoder: {encoder_name}")
    log.info(f"Using same encoder for final feature extraction to ensure consistency")
    dino_encoder = load_dino_encoder(encoder_name, args.device)

    # Get conditions
    all_conditions, cond_metadata = get_celeba_conditions(is_marginal=args.marginal)

    if args.marginal:
        # Separate seen and unseen using Hamming weight
        seen_conditions = [c for c in all_conditions if cond_metadata[c] == "seen"]
        unseen_conditions = [c for c in all_conditions if cond_metadata[c] == "unseen"]

        log.info(
            f"Marginal model: {len(seen_conditions)} seen, {len(unseen_conditions)} unseen conditions"
        )
        log.info(f"Seen conditions (Hamming weight <= 1): {seen_conditions}")

        # Analyze seen
        log.info("\n=== Analyzing SEEN conditions ===")
        seen_results = analyze_alignment_for_conditions(
            generator,
            dino_encoder,
            seen_conditions,
            args.samples_per_condition,
            args.batch_size,
            args.num_inference_steps,
            args.device,
        )
        seen_idx, seen_t, seen_mean, seen_std = aggregate_results(seen_results)

        # Analyze unseen
        log.info("\n=== Analyzing UNSEEN conditions ===")
        unseen_results = analyze_alignment_for_conditions(
            generator,
            dino_encoder,
            unseen_conditions,
            args.samples_per_condition,
            args.batch_size,
            args.num_inference_steps,
            args.device,
        )
        unseen_idx, unseen_t, unseen_mean, unseen_std = aggregate_results(
            unseen_results
        )

        # Plot both (use timestep values for x-axis)
        plot_alignment_curve(
            seen_t,
            seen_mean,
            seen_std,
            f"DINO Alignment - SEEN Conditions ({args.checkpoint_key})",
            output_dir / f"{args.checkpoint_key}_seen_alignment.png",
        )

        plot_alignment_curve(
            unseen_t,
            unseen_mean,
            unseen_std,
            f"DINO Alignment - UNSEEN Conditions ({args.checkpoint_key})",
            output_dir / f"{args.checkpoint_key}_unseen_alignment.png",
        )

        # Save CSVs (include both step indices and timestep values)
        save_csv(
            seen_t,
            seen_mean,
            seen_std,
            output_dir / f"{args.checkpoint_key}_seen_alignment.csv",
            "seen",
        )
        save_csv(
            unseen_t,
            unseen_mean,
            unseen_std,
            output_dir / f"{args.checkpoint_key}_unseen_alignment.csv",
            "unseen",
        )

        # Save data artifacts
        torch.save(
            {
                "seen": {
                    "step_indices": seen_idx,
                    "timesteps": seen_t,
                    "mean": seen_mean,
                    "std": seen_std,
                    "results": seen_results,
                },
                "unseen": {
                    "step_indices": unseen_idx,
                    "timesteps": unseen_t,
                    "mean": unseen_mean,
                    "std": unseen_std,
                    "results": unseen_results,
                },
            },
            output_dir / f"{args.checkpoint_key}_alignment_data.pt",
        )

        log.info(f"\n=== Analysis complete! Results saved to {output_dir} ===")
        log.info(
            f"\nNote: Captured {len(seen_idx)} steps from index 0 to {seen_idx[-1]}"
        )
        log.info(f"Timestep range: t=1.0 → t=0.04 (t_cutoff), NOT including t=0.0")
        log.info(
            "The model is never called at t=0.0 in REPA-style sampling (Stage 2 is deterministic ODE)"
        )

    else:
        # Full model: all conditions together
        log.info(f"\n=== Analyzing all {len(all_conditions)} conditions ===")
        all_results = analyze_alignment_for_conditions(
            generator,
            dino_encoder,
            all_conditions,
            args.samples_per_condition,
            args.batch_size,
            args.num_inference_steps,
            args.device,
        )
        step_indices, timesteps, mean_sims, std_sims = aggregate_results(all_results)

        # Plot (use timestep values for x-axis)
        plot_alignment_curve(
            timesteps,
            mean_sims,
            std_sims,
            f"DINO Alignment - All Conditions ({args.checkpoint_key})",
            output_dir / f"{args.checkpoint_key}_alignment.png",
        )

        # Save CSV
        save_csv(
            timesteps,
            mean_sims,
            std_sims,
            output_dir / f"{args.checkpoint_key}_alignment.csv",
            "all",
        )

        # Save data artifact
        torch.save(
            {
                "step_indices": step_indices,
                "timesteps": timesteps,
                "mean": mean_sims,
                "std": std_sims,
                "results": all_results,
            },
            output_dir / f"{args.checkpoint_key}_alignment_data.pt",
        )

        log.info(f"\n=== Analysis complete! Results saved to {output_dir} ===")
        log.info(
            f"\nNote: Captured {len(step_indices)} steps from index 0 to {step_indices[-1]}"
        )
        log.info(f"Timestep range: t=1.0 → t=0.04 (t_cutoff), NOT including t=0.0")
        log.info(
            "The model is never called at t=0.0 in REPA-style sampling (Stage 2 is deterministic ODE)"
        )

    breakpoint()


if __name__ == "__main__":
    main()
