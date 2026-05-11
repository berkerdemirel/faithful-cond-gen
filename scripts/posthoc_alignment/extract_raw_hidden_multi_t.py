"""Extract mean-pooled raw hidden features at multiple timesteps for mapper training.

For each real CelebA training image and each timestep t:
  1. VAE encode to latents
  2. Add noise at level t
  3. Forward pass with return_raw_hidden=True
  4. Mean-pool (B, 256, 768) -> (B, 768)
  5. Save per-timestep shard

Multi-GPU: each timestep runs on a separate GPU in parallel.

Usage:
    # Vanilla
    PYTHONPATH=src uv run python scripts/posthoc_alignment/extract_raw_hidden_multi_t.py \
        checkpoint_key=celeba_vanilla_marginal_v1

    # REPA SigLIP
    PYTHONPATH=src uv run python scripts/posthoc_alignment/extract_raw_hidden_multi_t.py \
        checkpoint_key=celeba_repa_siglip_marginal_v1 \
        model.use_repa=true model.repa_encoder=siglip model.repa_proj_coeff=0.5
"""

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from faithful_cond_gen.data.celeba import CelebaDataModule
from faithful_cond_gen.model.generator import GeneratorWrapper
from faithful_cond_gen.pl_modules.generator_pl import GeneratorPL
from faithful_cond_gen.utils.checkpoints import get_checkpoint_path

log = logging.getLogger(__name__)

CONDITION_KEYS = ["Male", "Smiling", "Blond_Hair", "Eyeglasses"]


def extract_hidden_at_timestep(
    model: GeneratorPL,
    dataloader: DataLoader,
    timestep: float,
    device: torch.device,
) -> dict:
    """Extract mean-pooled raw hidden features at a single timestep."""
    all_features = []
    all_metadata = {k: [] for k in CONDITION_KEYS}

    for batch in tqdm(dataloader, desc=f"t={timestep:.2f}", leave=False):
        images, meta = batch[0], batch[1] if len(batch) > 1 else {}
        images = images.to(device)
        B = images.shape[0]

        if images.min() < 0:
            images = (images + 1) / 2
        images = images.contiguous()

        with torch.no_grad():
            latents = model.generator.encode(images)

            alpha_bar = 1 - timestep
            noise = torch.randn_like(latents)
            noisy_latents = (
                np.sqrt(alpha_bar) * latents + np.sqrt(1 - alpha_bar) * noise
            )

            t_tensor = torch.full(
                (B,), timestep, device=device, dtype=torch.float32
            )

            cond_dict = meta.get("cond", meta)
            cond_ids = torch.stack(
                [cond_dict[k].to(device) for k in CONDITION_KEYS], dim=1
            )

            _, zs = model.generator.velocity_prediction(
                noisy_latents,
                t_tensor,
                cond_ids,
                return_projected=True,
                return_raw_hidden=True,
            )

            # zs[0] is (B, T, D) = (B, 256, 768) for SiT-B/2
            raw_hidden = zs[0]
            pooled = raw_hidden.mean(dim=1)  # (B, 768)
            all_features.append(pooled.cpu())

        for k in CONDITION_KEYS:
            v = cond_dict[k]
            if isinstance(v, torch.Tensor):
                all_metadata[k].extend(v.cpu().tolist())
            else:
                all_metadata[k].extend([v] * B)

    features = torch.cat(all_features, dim=0)
    metadata = {k: torch.tensor(v, dtype=torch.long) for k, v in all_metadata.items()}

    return {
        "features": features,
        "metadata": metadata,
        "timestep": timestep,
        "n_samples": features.shape[0],
        "feature_dim": features.shape[1],
    }


def worker_fn(gpu_id, timestep, cfg_dict, output_dir, siglip_meta_dict):
    """Worker: load model on gpu_id, extract one timestep, save shard."""
    cfg = OmegaConf.create(cfg_dict)
    device = torch.device(f"cuda:{gpu_id}")

    shard_path = Path(output_dir) / f"t{timestep}_hidden.pt"
    if shard_path.exists():
        print(f"[GPU {gpu_id}] Shard t={timestep} already exists, skipping")
        return

    print(f"[GPU {gpu_id}] Loading model for t={timestep}...")
    ckpt_path = get_checkpoint_path(cfg.checkpoint_key)
    gen_cfg = instantiate(cfg.model)
    generator_backbone = GeneratorWrapper(gen_cfg)
    model = GeneratorPL.load_from_checkpoint(
        ckpt_path, generator=generator_backbone, strict=False, map_location="cpu",
    )
    if hasattr(model, "ema"):
        model.ema.apply()
    model.to(device)
    model.eval()

    dm_conf = instantiate(cfg.dataset)
    dm = CelebaDataModule(dm_conf)
    dm.setup(stage="fit")
    dataloader = DataLoader(
        dm.get_dataset("train"),
        batch_size=cfg.get("batch_size", 64),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    print(f"[GPU {gpu_id}] Extracting t={timestep}...")
    result = extract_hidden_at_timestep(model, dataloader, timestep, device)
    print(f"[GPU {gpu_id}] Features: {result['features'].shape}")

    # Verify metadata
    if siglip_meta_dict is not None:
        for k in CONDITION_KEYS:
            if not torch.equal(result["metadata"][k], siglip_meta_dict[k]):
                n_mm = (result["metadata"][k] != siglip_meta_dict[k]).sum().item()
                print(f"[GPU {gpu_id}] WARNING: Metadata mismatch for '{k}': {n_mm} samples differ!")
                break
        else:
            print(f"[GPU {gpu_id}] Metadata matches SigLIP cache")

    torch.save(result, shard_path)
    print(f"[GPU {gpu_id}] Saved t={timestep} to {shard_path}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()


@hydra.main(
    config_path="../../configs",
    config_name="extract_raw_hidden",
    version_base="1.3",
)
def main(cfg: DictConfig):
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    checkpoint_key = cfg.checkpoint_key
    timesteps = list(cfg.timesteps)
    output_dir = Path(cfg.output_dir) / checkpoint_key
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = torch.cuda.device_count()
    log.info(f"Available GPUs: {n_gpus}, Timesteps: {timesteps}")

    # Filter out already-done timesteps
    todo = [(t, output_dir / f"t{t}_hidden.pt") for t in timesteps]
    todo = [(t, p) for t, p in todo if not p.exists()]
    if not todo:
        log.info("All shards already exist!")
        return
    log.info(f"Timesteps to extract: {[t for t, _ in todo]}")

    # Load SigLIP metadata for verification
    siglip_meta = None
    siglip_path = cfg.get("siglip_cache", None)
    if siglip_path and Path(siglip_path).exists():
        siglip_data = torch.load(siglip_path, map_location="cpu", weights_only=False)
        siglip_meta = {k: v for k, v in siglip_data["metadata"].items() if k in CONDITION_KEYS}
        log.info(f"Loaded SigLIP metadata for verification ({siglip_data['features'].shape[0]} samples)")
        del siglip_data

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Launch workers in parallel, one per GPU
    mp.set_start_method("spawn", force=True)
    processes = []
    for i, (t, _) in enumerate(todo):
        gpu_id = i % n_gpus
        p = mp.Process(target=worker_fn, args=(gpu_id, t, cfg_dict, str(output_dir), siglip_meta))
        p.start()
        processes.append((t, p))

    for t, p in processes:
        p.join()
        if p.exitcode != 0:
            log.error(f"Worker for t={t} failed with exit code {p.exitcode}")
        else:
            log.info(f"Worker for t={t} completed successfully")

    log.info("Done!")


if __name__ == "__main__":
    main()
