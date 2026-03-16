"""Generate samples from REPA-aligned models with aligned feature extraction.

This script extends generate_samples.py to also extract and save the aligned
layer features during generation, avoiding the need to re-run DINOv3.

Features are automatically consolidated into aligned_mean_features.pt at the end.
"""
import glob
import itertools
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
import torch.distributed as dist
from faithful_cond_gen.data.celeba import CelebaDataConfig, CelebaDataModule
from faithful_cond_gen.data.rxrx1 import RxRx1DataConfig, RxRx1DataModule
from faithful_cond_gen.pl_modules.generator_pl import GeneratorPL
from faithful_cond_gen.utils.checkpoints import get_checkpoint_path
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)


# =============================================================================
# Consolidation utilities
# =============================================================================


def parse_condition_from_signature(signature: str) -> Dict[str, int]:
    """Parse condition values from signature string."""
    parts = signature.split("_")
    cond_dict = {}
    buffer = []
    for p in parts:
        if not p:
            continue
        if p[-1] in ["0", "1"] and len(p) > 1:
            attr_name = "_".join(buffer + [p[:-1]])
            cond_dict[attr_name] = int(p[-1])
            buffer = []
        else:
            buffer.append(p)
    return cond_dict


def parse_filename(fname: str) -> Tuple[str, int]:
    """Parse filename into (signature, local_idx)."""
    stem = Path(fname).stem
    sig, idx_str = stem.rsplit("_", 1)
    return sig, int(idx_str)


def consolidate_aligned_features(
    out_dir: str,
    pooling: str = "mean",
    condition_keys: List[str] = None,
) -> Dict:
    """
    Consolidate aligned feature shards into a single file using index-based joining.

    Uses image filenames as the source of truth for global ordering.
    Ensures features[i] corresponds to sorted_images[i].

    Args:
        out_dir: Output directory containing images/ and aligned_features/
        pooling: Pooling method for patch features
        condition_keys: List of condition keys (auto-detected if None)

    Returns:
        Dict with features, metadata, indices
    """
    images_dir = Path(out_dir) / "images"
    aligned_dir = Path(out_dir) / "aligned_features"

    # Step 1: Build global index mapping from images
    log.info("Consolidating: Building global index mapping...")
    png_files = list(images_dir.glob("*.png"))
    pt_files = list(images_dir.glob("*.pt"))
    image_files = png_files if png_files else pt_files

    if not image_files:
        raise FileNotFoundError(f"No image files in {images_dir}")

    sorted_files = sorted(image_files, key=lambda p: p.name)
    sorted_filenames = [f.name for f in sorted_files]
    N = len(sorted_filenames)
    log.info(f"  Found {N} images")

    # Build (signature, local_idx) -> global_idx mapping
    idx_mapping: Dict[Tuple[str, int], int] = {}
    for global_idx, fname in enumerate(sorted_filenames):
        sig, local_idx = parse_filename(fname)
        key = (sig, local_idx)
        if key in idx_mapping:
            raise ValueError(f"Duplicate (signature, local_idx): {key}")
        idx_mapping[key] = global_idx

    # Step 2: Load shards
    shard_files = sorted(aligned_dir.glob("*_aligned_feats.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No aligned feature shards in {aligned_dir}")
    log.info(f"  Loading {len(shard_files)} shards...")

    feature_dim = None
    idx_to_feature: Dict[int, torch.Tensor] = {}
    idx_to_condition: Dict[int, Dict[str, int]] = {}
    all_condition_keys = set()

    for shard_path in tqdm(shard_files, desc="Loading shards", disable=True):
        sig = shard_path.stem.replace("_aligned_feats", "")
        data = torch.load(shard_path, map_location="cpu")

        # Handle dict format (new) or raw tensor (legacy)
        if isinstance(data, dict):
            features = data.get("aligned_features", data.get("features"))
            local_indices = data.get("indices", list(range(features.shape[0])))
            if isinstance(local_indices, torch.Tensor):
                local_indices = local_indices.tolist()
            condition = data.get("condition", parse_condition_from_signature(sig))
        else:
            features = data
            local_indices = list(range(features.shape[0]))
            condition = parse_condition_from_signature(sig)

        all_condition_keys.update(condition.keys())

        # Pool if needed (N, patches, dim) -> (N, dim)
        if features.ndim == 3:
            if pooling == "mean":
                features = features.mean(dim=1)
            elif pooling == "cls":
                features = features[:, 0, :]
            elif pooling == "max":
                features = features.max(dim=1)[0]

        if feature_dim is None:
            feature_dim = features.shape[1]

        # Map to global indices
        for i, local_idx in enumerate(local_indices):
            key = (sig, local_idx)
            if key not in idx_mapping:
                log.warning(f"Sample {key} not in image index mapping, skipping")
                continue
            global_idx = idx_mapping[key]
            if global_idx in idx_to_feature:
                raise ValueError(f"Duplicate global index {global_idx}")
            idx_to_feature[global_idx] = features[i]
            idx_to_condition[global_idx] = condition.copy()

    # Step 3: Build output tensors in sorted index order
    if condition_keys is None:
        condition_keys = sorted(all_condition_keys)

    sorted_indices = sorted(idx_to_feature.keys())
    actual_N = len(sorted_indices)

    features_tensor = torch.stack([idx_to_feature[i] for i in sorted_indices], dim=0)
    indices_tensor = torch.tensor(sorted_indices, dtype=torch.long)

    metadata = {}
    for key in condition_keys:
        vals = [idx_to_condition[i].get(key, 0) for i in sorted_indices]
        metadata[key] = torch.tensor(vals, dtype=torch.long)

    filenames = [sorted_filenames[i] for i in sorted_indices]

    # Step 4: Integrity checks
    log.info("  Running integrity checks...")
    assert len(set(sorted_indices)) == len(sorted_indices), "Duplicate indices"

    for key, vals in metadata.items():
        assert len(vals) == actual_N, f"Metadata {key} length mismatch"

    # Spot check
    rng = np.random.default_rng(42)
    check_indices = rng.choice(sorted_indices, size=min(5, len(sorted_indices)), replace=False)
    for idx in check_indices:
        fname = sorted_filenames[idx]
        sig, _ = parse_filename(fname)
        expected_cond = parse_condition_from_signature(sig)
        for key in condition_keys:
            pos = sorted_indices.index(idx)
            if expected_cond.get(key, 0) != metadata[key][pos].item():
                raise ValueError(f"Spot check failed at index {idx}")

    log.info(f"  ✓ Consolidated {actual_N}/{N} samples, shape {features_tensor.shape}")

    return {
        "features": features_tensor,
        "metadata": metadata,
        "indices": indices_tensor,
        "filenames": filenames,
        "encoder_name": "dinov3-vit-l_meanpatch_aligned",
        "feature_dim": feature_dim,
        "pooling_method": pooling,
        "n_samples": actual_N,
    }


# =============================================================================
# DDP and generation utilities
# =============================================================================


def setup_ddp():
    """Initializes DDP if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return rank, world_size, True
    else:
        return 0, 1, False


def get_conditions_list(cfg, dm):
    """Returns list of {'cond_ids': [...], 'signature': str}"""
    conditions = []

    if "Celeba" in cfg.dataset._target_:
        attrs = (
            dm.selected_attrs
            if dm.selected_attrs
            else ["Male", "Smiling", "Blond_Hair", "Eyeglasses"]
        )
        # CRITICAL: Sort alphabetically to match model's internal ordering
        # Model uses sorted(cond_dict.keys()) during training
        attrs = sorted(attrs)
        combos = list(itertools.product([0, 1], repeat=len(attrs)))

        for c in combos:
            sig = "_".join([f"{k}{v}" for k, v in zip(attrs, c)])
            conditions.append({"cond_ids": list(c), "signature": sig, "type": "celeba"})

    elif "RxRx1" in cfg.dataset._target_:
        metadata = dm.metadata
        unique_sirnas = sorted(metadata["sirna_id"].unique())
        unique_cells = sorted(metadata["cell_type_id"].unique())

        log.info(
            f"RxRx1: Found {len(unique_cells)} cells and {len(unique_sirnas)} sirnas."
        )

        for c in unique_cells:
            for s in unique_sirnas:
                sig = f"cell{c}_sirna{s}"
                conditions.append(
                    {"cond_ids": [int(c), int(s)], "signature": sig, "type": "rxrx1"}
                )

    log.info(f"Total unique conditions to generate: {len(conditions)}")
    return conditions


@hydra.main(
    config_path="../configs", config_name="generate_samples_celeba", version_base=None
)
def main(cfg: DictConfig):
    # 1. DDP Setup
    rank, world_size, is_ddp = setup_ddp()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # 2. Output Dirs
    out_dir = cfg.output_dir
    img_dir = os.path.join(out_dir, "images")
    feat_dir = os.path.join(out_dir, "aligned_features")

    if rank == 0:
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(feat_dir, exist_ok=True)
        from omegaconf import OmegaConf

        OmegaConf.save(cfg, os.path.join(out_dir, "gen_config.yaml"))

    if is_ddp:
        dist.barrier()

    # 3. Load Generator
    log.info(f"[Rank {rank}] Loading REPA Generator...")
    gen_cfg = instantiate(cfg.model)
    from faithful_cond_gen.model.generator import GeneratorWrapper

    generator_backbone = GeneratorWrapper(gen_cfg)

    # Resolve checkpoint path
    if cfg.get("checkpoint_key"):
        ckpt_path = get_checkpoint_path(cfg.checkpoint_key)
        log.info(f"[Rank {rank}] Using checkpoint key '{cfg.checkpoint_key}' -> {ckpt_path}")
    else:
        ckpt_path = cfg.ckpt_path

    pl_module = GeneratorPL.load_from_checkpoint(
        ckpt_path, generator=generator_backbone, map_location=device, strict=False
    )
    pl_module.to(device)
    pl_module.eval()

    if hasattr(pl_module, "ema"):
        log.info(f"[Rank {rank}] Swapping to EMA weights...")
        pl_module.ema.apply()
    else:
        log.warning(f"[Rank {rank}] No EMA weights found in checkpoint module!")

    # Check REPA is enabled
    if not pl_module.generator.cfg.use_repa:
        log.warning(
            "[Rank {rank}] Model does not have REPA enabled! "
            "Aligned features will be None. Consider using generate_samples.py instead."
        )

    # 4. Get Workload
    dm_conf = instantiate(cfg.dataset)
    if "RxRx1" in cfg.dataset._target_:
        dm = RxRx1DataModule(dm_conf)
    else:
        dm = CelebaDataModule(dm_conf)
    all_conditions = get_conditions_list(cfg, dm)
    my_conditions = all_conditions[rank::world_size]
    log.info(f"[Rank {rank}] Assigned {len(my_conditions)} conditions.")

    # 5. Generation Loop with Feature Extraction
    samples_per_cond = cfg.samples_per_condition
    batch_size = cfg.batch_size
    feature_capture_idx = cfg.get("feature_capture_idx", None)  # None = final step (index num_inference_steps-1)
    use_raw_hidden = cfg.get("use_raw_hidden", False)  # If True, capture pre-MLP diffusion hidden state

    # Resume support: check for existing aligned features
    resume = cfg.get("resume", False)
    skipped = 0

    for cond_data in tqdm(my_conditions, desc=f"Rank {rank}"):
        cond_ids_list = cond_data["cond_ids"]
        signature = cond_data["signature"]
        data_type = cond_data["type"]

        # Skip if aligned features already exist (resume mode)
        if resume:
            feat_fname = f"{signature}_aligned_feats.pt"
            if os.path.exists(os.path.join(feat_dir, feat_fname)):
                skipped += 1
                continue

        generated_count = 0
        batch_idx = 0
        all_features_for_cond = []

        while generated_count < samples_per_cond:
            current_bs = min(batch_size, samples_per_cond - generated_count)

            # Prepare Batch Conditioning
            cond_tensor = torch.tensor(cond_ids_list, device=device).long()
            batch_cond_ids = cond_tensor.unsqueeze(0).repeat(current_bs, 1)

            # Generate with aligned feature extraction
            with torch.no_grad():
                images, aligned_features = pl_module.generator.sample(
                    cond_ids=batch_cond_ids,
                    num_inference_steps=cfg.get("num_inference_steps", 250),
                    t_cutoff=cfg.get("t_cutoff", 0.04),
                    cfg_scale=cfg.get("cfg_scale", 1.0),
                    adaptive_cfg=cfg.get("adaptive_cfg", False),
                    return_aligned_features=True,
                    feature_capture_idx=feature_capture_idx,
                    return_raw_hidden=use_raw_hidden,
                )
                images = torch.clamp(images, 0, 1)

            # Collect aligned features (if available)
            if aligned_features is not None and len(aligned_features) > 0:
                # aligned_features is List[Tensor], take first projector
                # Shape: (B, num_patches, embed_dim)
                feats = aligned_features[0].cpu()
                all_features_for_cond.append(feats)

            # Save images
            if data_type == "celeba":
                img_np = images.cpu().numpy()
                img_np = np.transpose(img_np, (0, 2, 3, 1))
                img_np = (img_np * 255).astype(np.uint8)

                for i in range(current_bs):
                    pil_img = Image.fromarray(img_np[i])
                    fname = f"{signature}_{generated_count + i}.png"
                    pil_img.save(os.path.join(img_dir, fname))

            elif data_type == "rxrx1":
                for i in range(current_bs):
                    img_tensor = images[i]
                    fname = f"{signature}_{generated_count + i}.pt"
                    torch.save(img_tensor.cpu(), os.path.join(img_dir, fname))

            generated_count += current_bs
            batch_idx += 1

        # Save aligned features for this condition with metadata
        if all_features_for_cond:
            cond_features = torch.cat(all_features_for_cond, dim=0)
            feat_fname = f"{signature}_aligned_feats.pt"

            # Parse condition from signature for metadata
            cond_dict = parse_condition_from_signature(signature)

            # Save with metadata for robust consolidation
            shard_data = {
                "aligned_features": cond_features,
                "indices": list(range(cond_features.shape[0])),  # Local indices within condition
                "condition": cond_dict,
                "signature": signature,
            }
            torch.save(shard_data, os.path.join(feat_dir, feat_fname))

    if resume and skipped > 0:
        log.info(f"[Rank {rank}] Skipped {skipped} conditions (already completed).")
    log.info(f"[Rank {rank}] Finished generation with feature extraction.")

    # Consolidate aligned features (rank 0 only, after all ranks finish)
    if is_ddp:
        dist.barrier()  # Wait for all ranks to finish

    if rank == 0:
        log.info("Consolidating aligned features...")
        try:
            result = consolidate_aligned_features(out_dir, pooling="mean")
            output_path = os.path.join(out_dir, "aligned_mean_features.pt")
            torch.save(result, output_path)
            log.info(f"Saved consolidated features to {output_path}")
        except Exception as e:
            log.error(f"Consolidation failed: {e}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
