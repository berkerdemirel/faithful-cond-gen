import logging
import os

import hydra
import torch
from faithful_cond_gen.data.celeba import CelebaDataModule

# Import DataModules explicitly (to wrap the config)
from faithful_cond_gen.data.rxrx1 import RxRx1DataModule, to_rgb

# Import Encoders
from faithful_cond_gen.eval.configs.encoder_config import (
    BIOCLIP,
    DINOV2_L14,
    DINOV3_L16,
    MAE_LARGE,
    OPENPHENOM,
    SIGLIP_SO400M,
)
from faithful_cond_gen.eval.encoders.registry import load_encoder
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

log = logging.getLogger(__name__)

CONFIG_MAP = {
    "dinov2": DINOV2_L14,
    "dinov3": DINOV3_L16,
    "mae": MAE_LARGE,
    "siglip": SIGLIP_SO400M,
    "bioclip": BIOCLIP,
    "openphenom": OPENPHENOM,
}


def adapt_batch(images: torch.Tensor, target_channels: int) -> torch.Tensor:
    """Converts batch to match encoder's expected channels."""
    current_channels = images.shape[1]

    # Case A: RxRx1 (6ch) -> DINO/SigLIP (3ch)
    if current_channels == 6 and target_channels == 3:
        device = images.device
        return torch.stack([to_rgb(img.cpu()[None]).squeeze(0) for img in images]).to(
            device
        )

    # Case B: Grayscale (1ch) -> RGB (3ch)
    elif current_channels == 1 and target_channels == 3:
        return images.repeat(1, 3, 1, 1)

    # Case C: Pass-through (6->6 for OpenPhenom, 3->3 for CelebA)
    return images


@torch.no_grad()
def extract_and_save(loader, encoder, save_path, device):
    """
    Iterates through a dataloader, extracts features, and saves them to disk.
    """
    if os.path.exists(save_path):
        log.warning(f"Cache file already exists at {save_path}. Skipping.")
        return

    log.info(f"Starting extraction -> {save_path}")

    all_features = []
    all_metadata = []

    for batch in tqdm(loader, desc="Extracting"):
        images = None
        meta_batch = {}

        # 1. Unpack Batch Logic (Matching your Dataset structure)
        if (
            isinstance(batch, (list, tuple))
            and len(batch) == 2
            and isinstance(batch[1], dict)
        ):
            images = batch[0]
            cond_wrapper = batch[1]  # e.g. {'cond': {...}, 'comp_category': ...}

            # Flatten 'cond' dictionary
            if "cond" in cond_wrapper and isinstance(cond_wrapper["cond"], dict):
                for attr_name, attr_val in cond_wrapper["cond"].items():
                    meta_batch[attr_name] = (
                        attr_val.cpu()
                        if isinstance(attr_val, torch.Tensor)
                        else attr_val
                    )

            # Extract other keys
            for k, v in cond_wrapper.items():
                if k == "cond":
                    continue
                meta_batch[k] = v.cpu() if isinstance(v, torch.Tensor) else v

        elif isinstance(batch, (list, tuple)):
            images = batch[0]
            meta_batch["labels"] = (
                batch[1].cpu() if isinstance(batch[1], torch.Tensor) else batch[1]
            )

        elif isinstance(batch, dict) and "image" in batch:
            images = batch["image"]
            for k, v in batch.items():
                if k != "image":
                    meta_batch[k] = v.cpu() if isinstance(v, torch.Tensor) else v
        else:
            raise ValueError(f"Unknown batch format: {type(batch)}")

        images = images.to(device)
        images = adapt_batch(images, encoder.cfg.input_channels)

        # 2. Forward Pass
        out = encoder(images)
        feats = out["features"].cpu()

        all_features.append(feats)
        all_metadata.append(meta_batch)

    if not all_features:
        log.error("No features extracted!")
        return

    # 3. Collate
    features_cat = torch.cat(all_features, dim=0)

    collated_meta = {}
    if all_metadata:
        keys = all_metadata[0].keys()
        for k in keys:
            vals = [m[k] for m in all_metadata]
            if isinstance(vals[0], torch.Tensor):
                if vals[0].ndim == 0:
                    collated_meta[k] = torch.stack(vals, dim=0)
                else:
                    collated_meta[k] = torch.cat(vals, dim=0)
            elif isinstance(vals[0], (list, tuple)):
                flat_list = []
                for v in vals:
                    flat_list.extend(list(v))
                collated_meta[k] = flat_list
            else:
                collated_meta[k] = vals

    # 4. Save
    payload = {
        "features": features_cat,
        "metadata": collated_meta,
        "encoder_name": encoder.cfg.name,
        "feature_dim": encoder.feature_dim,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(payload, save_path)
    log.info(f"Saved {features_cat.shape[0]} samples to {save_path}")


@hydra.main(config_path="../configs", config_name="cache_celeba", version_base=None)
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Select Encoder
    enc_name = cfg.get("encoder_name", "dinov2")
    if enc_name not in CONFIG_MAP:
        raise ValueError(
            f"Unknown encoder '{enc_name}'. Available: {list(CONFIG_MAP.keys())}"
        )

    enc_config = CONFIG_MAP[enc_name]
    log.info(f"Selected Encoder: {enc_name} ({enc_config.name})")

    # 2. Instantiate DataModule (Correctly wrapping the Config)
    log.info(f"Instantiating Config: {cfg.dataset._target_}")
    dm_conf = instantiate(cfg.dataset)

    if "RxRx1" in cfg.dataset._target_:
        log.info("Initializing RxRx1DataModule...")
        dm = RxRx1DataModule(dm_conf)
    elif "Celeba" in cfg.dataset._target_:
        log.info("Initializing CelebaDataModule...")
        dm = CelebaDataModule(dm_conf)
    else:
        raise ValueError(f"Unknown dataset target: {cfg.dataset._target_}")

    # 3. Load Encoder Model
    encoder = load_encoder(enc_config, device=device)

    # 4. Inject Encoder Transform
    # dm.setup()
    enc_transform = encoder.get_transform()
    log.info(f"Injecting Encoder Transform: {enc_transform}")

    for split_name in ["train_ds", "val_ds", "test_ds"]:
        if hasattr(dm, split_name):
            ds = getattr(dm, split_name)
            if ds is not None and hasattr(ds, "transform"):
                ds.transform = enc_transform

    # 5. Output Path
    base_cache_dir = cfg.dataset.get("cache_dir", ".") or "."
    feature_cache_dir = os.path.join(base_cache_dir, "feature_cache", enc_name)
    os.makedirs(feature_cache_dir, exist_ok=True)
    log.info(f"Output Directory: {feature_cache_dir}")

    # 6. Run
    loaders = {
        "train": dm.train_dataloader(),
        "val": dm.val_dataloader(),
    }

    for split_name, loader in loaders.items():
        if loader is None:
            continue
        save_path = os.path.join(feature_cache_dir, f"{split_name}_features.pt")
        log.info(f"Processing split: {split_name}...")
        extract_and_save(loader, encoder, save_path, device)


if __name__ == "__main__":
    main()
