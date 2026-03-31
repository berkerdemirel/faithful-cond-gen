"""
Extract aligned-mean features from REPA model for real samples.

Protocol: Add minimal Gaussian noise scaled to timestep t=0.01, run one denoiser
forward pass, and extract aligned layer activations from REPA projectors.
This ensures feature distribution matches generated samples.

Usage:
    PYTHONPATH=src uv run python scripts/extract_aligned_features_real.py \
        dataset=celeba checkpoint_key=repa_full

Output:
    outputs/real_{dataset}_aligned/{checkpoint_key}/train_features.pt
"""

import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from faithful_cond_gen.data.celeba import CelebaDataModule
from faithful_cond_gen.data.rxrx1 import RxRx1DataModule, to_rgb
from faithful_cond_gen.model.generator import GeneratorWrapper
from faithful_cond_gen.pl_modules.generator_pl import GeneratorPL
from faithful_cond_gen.utils.checkpoints import get_checkpoint_path

log = logging.getLogger(__name__)


def get_data_module(cfg: DictConfig):
    """Instantiate data module based on dataset config."""
    dm_conf = instantiate(cfg.dataset)
    if "RxRx1" in cfg.dataset._target_:
        dm = RxRx1DataModule(dm_conf)
    else:
        dm = CelebaDataModule(dm_conf)
    dm.setup(stage="fit")
    return dm


def add_noise_at_timestep(
    x: torch.Tensor,
    t: float,
    noise_schedule: str = "linear",
) -> torch.Tensor:
    """
    Add noise to images at a specific timestep in [0, 1].

    For t=0.01 (very small), this adds minimal noise.
    Uses linear schedule: alpha_bar = 1 - t, so noise is sqrt(t) * eps.
    """
    # Linear schedule: alpha_bar(t) = 1 - t
    alpha_bar = 1 - t
    noise = torch.randn_like(x)
    noisy_x = np.sqrt(alpha_bar) * x + np.sqrt(1 - alpha_bar) * noise
    return noisy_x


def extract_aligned_features_from_model(
    model: GeneratorPL,
    dataloader: DataLoader,
    timestep: float = 0.01,
    device: torch.device = None,
    condition_keys: List[str] = None,
    return_raw_hidden: bool = False,
) -> Dict:
    """
    Extract aligned features from REPA model for real samples.

    Args:
        model: REPA model with repa_projectors
        dataloader: DataLoader for real samples
        timestep: Noise level (default 0.01)
        device: torch device
        condition_keys: List of condition attribute names

    Returns:
        Dict with features, metadata, etc.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Check for REPA projectors in diffusion_backbone
    backbone = model.generator.diffusion_backbone
    if not hasattr(backbone, "projectors") or backbone.projectors is None:
        raise ValueError("Model does not have REPA projectors - is this a REPA model?")

    all_features = []
    all_metadata = {k: [] for k in condition_keys} if condition_keys else {}

    with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting aligned features"):
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                    meta = batch[1] if len(batch) > 1 else {}
                else:
                    images = batch
                    meta = {}

                images = images.to(device)
                B = images.shape[0]

                # NOTE: Do NOT convert 6ch to 3ch here - VAE handles 6ch internally for RxRx1

                # Scale images to [0, 1] if needed (VAE expects [0, 1] and does its own scaling)
                # Ensure images are in [0, 1] range for VAE
                if images.min() < 0:
                    images = (images + 1) / 2  # [-1, 1] -> [0, 1]

                # Ensure contiguous tensor for VAE's view operations
                images = images.contiguous()

                # Encode images to latent space
                latents = model.generator.encode(images)

                # Add noise to latents at timestep (linear schedule)
                alpha_bar = 1 - timestep
                noise = torch.randn_like(latents)
                noisy_latents = np.sqrt(alpha_bar) * latents + np.sqrt(1 - alpha_bar) * noise

                # Create timestep tensor
                t_tensor = torch.full((B,), timestep, device=device, dtype=torch.float32)

                # Extract condition IDs from metadata
                cond_dict = meta.get("cond", meta)
                cond_ids = []
                for k in condition_keys:
                    if k in cond_dict:
                        v = cond_dict[k]
                        if isinstance(v, torch.Tensor):
                            cond_ids.append(v.to(device))
                        else:
                            cond_ids.append(torch.tensor([v] * B, device=device))
                if cond_ids:
                    cond_ids = torch.stack(cond_ids, dim=1)  # (B, K)
                else:
                    cond_ids = torch.zeros((B, 1), device=device, dtype=torch.long)

                # Forward pass through diffusion backbone to get projected features (or raw hidden)
                _, zs_tilde = model.generator.velocity_prediction(
                    noisy_latents, t_tensor, cond_ids,
                    return_projected=True,
                    return_raw_hidden=return_raw_hidden,
                )

                # zs_tilde is a list of projected features from each projector
                # Take the first projector's output
                if zs_tilde is not None and len(zs_tilde) > 0:
                    aligned = zs_tilde[0]  # (B, T, D) or (B, D) depending on use_global_alignment
                    # Mean pool over patches if needed
                    if aligned.dim() == 3:
                        aligned_mean = aligned.mean(dim=1)  # (B, D)
                    else:
                        aligned_mean = aligned  # Already (B, D)
                    all_features.append(aligned_mean.cpu())

                # Collect metadata
                if condition_keys and meta:
                    cond_dict = meta.get("cond", meta)
                    for k in condition_keys:
                        if k in cond_dict:
                            v = cond_dict[k]
                            if isinstance(v, torch.Tensor):
                                all_metadata[k].extend(v.cpu().tolist())
                            else:
                                all_metadata[k].extend([v] * B)

    # Stack features
    features = torch.cat(all_features, dim=0)

    # Convert metadata to tensors
    metadata = {}
    for k, v in all_metadata.items():
        if v:
            metadata[k] = torch.tensor(v, dtype=torch.long)

    encoder_name = "sit_raw_hidden_real" if return_raw_hidden else "dinov3-vit-l_meanpatch_aligned_real"
    return {
        "features": features,
        "metadata": metadata,
        "encoder_name": encoder_name,
        "timestep": timestep,
        "n_samples": features.shape[0],
        "feature_dim": features.shape[1],
    }


@hydra.main(
    config_path="../configs",
    config_name="extract_aligned_real",
    version_base="1.3",
)
def main(cfg: DictConfig):
    """Main entry point."""
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Setup
    checkpoint_key = cfg.checkpoint_key
    batch_size = cfg.get("batch_size", 32)
    timestep = cfg.get("noise_timestep", 0.01)
    split = cfg.get("split", "train")
    output_dir = Path(cfg.output_dir)

    # Detect dataset type from config target
    is_rxrx1 = "RxRx1" in cfg.dataset._target_

    # Get condition keys
    if is_rxrx1:
        condition_keys = ["cell_type_id", "sirna_id"]
    else:
        condition_keys = ["Male", "Smiling", "Blond_Hair", "Eyeglasses"]

    # Load checkpoint
    ckpt_path = get_checkpoint_path(checkpoint_key)
    log.info(f"Loading checkpoint: {ckpt_path}")

    # Instantiate generator backbone (same pattern as generate_samples_repa.py)
    gen_cfg = instantiate(cfg.model)
    generator_backbone = GeneratorWrapper(gen_cfg)

    # Load model with generator
    model = GeneratorPL.load_from_checkpoint(
        ckpt_path,
        generator=generator_backbone,
        strict=False,
        map_location="cpu",
    )
    # Apply EMA weights to match generation (generate_samples_repa.py applies EMA)
    if hasattr(model, "ema"):
        log.info("Applying EMA weights (matching generation pipeline)")
        model.ema.apply()
    model.eval()

    # Setup data module
    dm = get_data_module(cfg)

    # Get dataloader
    if split == "train":
        dataloader = dm.train_dataloader()
    else:
        dataloader = dm.val_dataloader()

    # Wrap in new dataloader with desired batch size
    dataloader = DataLoader(
        dataloader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # Extract features
    device = torch.device(f"cuda:{cfg.get('device', 0)}" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    return_raw_hidden = cfg.get("return_raw_hidden", False)
    result = extract_aligned_features_from_model(
        model=model,
        dataloader=dataloader,
        timestep=timestep,
        device=device,
        condition_keys=condition_keys,
        return_raw_hidden=return_raw_hidden,
    )

    # Save
    output_dir = output_dir / checkpoint_key
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}_features.pt"

    log.info(f"Saving to {output_path}")
    log.info(f"  Features shape: {result['features'].shape}")
    log.info(f"  Metadata keys: {list(result['metadata'].keys())}")

    torch.save(result, output_path)
    log.info("Done!")


if __name__ == "__main__":
    main()
