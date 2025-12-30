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
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

# We don't need Encoders here anymore!

log = logging.getLogger(__name__)

# --- UTILS ---


def setup_ddp():
    """Initializes DDP if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # Use LOCAL_RANK for device selection (multi-node compatible)
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return rank, world_size, True
    else:
        return 0, 1, False


def get_conditions_list(cfg, dm):
    """
    Returns list of {'cond_ids': [...], 'signature': str}
    """
    conditions = []

    # CASE A: CelebA
    if "Celeba" in cfg.dataset._target_:
        attrs = (
            dm.selected_attrs
            if dm.selected_attrs
            else ["Male", "Smiling", "Blond_Hair", "Eyeglasses"]
        )
        combos = list(itertools.product([0, 1], repeat=len(attrs)))

        for c in combos:
            sig = "_".join([f"{k}{v}" for k, v in zip(attrs, c)])
            conditions.append({"cond_ids": list(c), "signature": sig, "type": "celeba"})

    # CASE B: RxRx1
    elif "RxRx1" in cfg.dataset._target_:
        # We need metadata to know valid/interesting conditions
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


# --- MAIN ---


@hydra.main(
    config_path="../configs", config_name="generate_samples_celeba", version_base=None
)
def main(cfg: DictConfig):
    # 1. DDP Setup
    rank, world_size, is_ddp = setup_ddp()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # 2. Output Dirs (Rank 0 create)
    out_dir = cfg.output_dir
    img_dir = os.path.join(out_dir, "images")

    if rank == 0:
        os.makedirs(img_dir, exist_ok=True)
        # Save config for reproducibility
        from omegaconf import OmegaConf

        OmegaConf.save(cfg, os.path.join(out_dir, "gen_config.yaml"))

    if is_ddp:
        dist.barrier()

    # 3. Load Generator
    log.info(f"[Rank {rank}] Loading Generator...")
    gen_cfg = instantiate(cfg.model)
    from faithful_cond_gen.model.generator import GeneratorWrapper

    generator_backbone = GeneratorWrapper(gen_cfg)

    pl_module = GeneratorPL.load_from_checkpoint(
        cfg.ckpt_path, generator=generator_backbone, map_location=device, strict=False
    )
    pl_module.to(device)
    pl_module.eval()

    if hasattr(pl_module, "ema"):
        log.info(f"[Rank {rank}] Swapping to EMA weights...")
        pl_module.ema.apply()
    else:
        log.warning(f"[Rank {rank}] No EMA weights found in checkpoint module!")
    # 4. Get Workload
    dm_conf = instantiate(cfg.dataset)
    if "RxRx1" in cfg.dataset._target_:
        dm = RxRx1DataModule(dm_conf)
    else:
        dm = CelebaDataModule(dm_conf)
    all_conditions = get_conditions_list(cfg, dm)
    my_conditions = all_conditions[rank::world_size]
    log.info(f"[Rank {rank}] Assigned {len(my_conditions)} conditions.")

    # 5. Generation Loop
    samples_per_cond = cfg.samples_per_condition
    batch_size = cfg.batch_size

    for cond_data in tqdm(my_conditions, desc=f"Rank {rank}"):
        cond_ids_list = cond_data["cond_ids"]
        signature = cond_data["signature"]
        data_type = cond_data["type"]

        generated_count = 0
        batch_idx = 0

        while generated_count < samples_per_cond:
            current_bs = min(batch_size, samples_per_cond - generated_count)

            # Prepare Batch Conditioning
            cond_tensor = torch.tensor(cond_ids_list, device=device).long()
            batch_cond_ids = cond_tensor.unsqueeze(0).repeat(current_bs, 1)

            # Generate
            with torch.no_grad():
                images = pl_module.generator.sample(
                    cond_ids=batch_cond_ids,
                    num_inference_steps=250,  # REPA-style: higher quality
                    t_cutoff=0.04,
                )
                images = torch.clamp(images, 0, 1)

            # SAVE LOGIC
            if data_type == "celeba":
                # Save individual PNGs (Standard format for vision metrics)
                img_np = images.cpu().numpy()  # (B, 3, H, W)
                # Channel last
                img_np = np.transpose(img_np, (0, 2, 3, 1))
                img_np = (img_np * 255).astype(np.uint8)

                for i in range(current_bs):
                    pil_img = Image.fromarray(img_np[i])
                    fname = f"{signature}_{generated_count + i}.png"
                    pil_img.save(os.path.join(img_dir, fname))

            elif data_type == "rxrx1":
                # Save Batch Tensors (Preserve 6 channels, faster IO)
                # Filename: cell0_sirna10_batch0.pt
                # fname = f"{signature}_batch{batch_idx}.pt"
                # torch.save(images.cpu(), os.path.join(img_dir, fname))
                for i in range(current_bs):
                    pil_img = images[i]
                    fname = f"{signature}_{generated_count + i}.pt"
                    torch.save(pil_img.cpu(), os.path.join(img_dir, fname))

            generated_count += current_bs
            batch_idx += 1

    log.info(f"[Rank {rank}] Finished generation.")
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
