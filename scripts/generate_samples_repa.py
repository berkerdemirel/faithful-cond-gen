"""Generate samples from REPA-aligned models with aligned feature extraction.

This script extends generate_samples.py to also extract and save the aligned
layer features during generation, avoiding the need to re-run DINOv3.
"""
import itertools
import logging
import os

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
    feature_capture_t = cfg.get("feature_capture_t", None)  # None = final step

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
                    feature_capture_t=feature_capture_t,
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

        # Save aligned features for this condition
        if all_features_for_cond:
            cond_features = torch.cat(all_features_for_cond, dim=0)
            feat_fname = f"{signature}_aligned_feats.pt"
            torch.save(cond_features, os.path.join(feat_dir, feat_fname))

    if resume and skipped > 0:
        log.info(f"[Rank {rank}] Skipped {skipped} conditions (already completed).")
    log.info(f"[Rank {rank}] Finished generation with feature extraction.")
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
