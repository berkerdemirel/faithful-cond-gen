import glob
import logging
import os
import re
from typing import Dict, List, Optional

import hydra
import torch
from faithful_cond_gen.data.celeba import CelebaDataModule
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
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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

# --- Utils for Generated Data ---


class GeneratedDataset(Dataset):
    """
    Reads generated samples from a flat directory.
    Parses filenames to reconstruct conditioning metadata.
    """

    def __init__(
        self, root_dir: str, dataset_type: str, transform=None, input_channels=3
    ):
        self.root_dir = root_dir
        self.dataset_type = dataset_type.lower()
        self.transform = transform
        self.input_channels = input_channels

        # Gather files
        if "rxrx1" in self.dataset_type:
            self.files = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        else:
            self.files = sorted(glob.glob(os.path.join(root_dir, "*.png")))

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No files found in {root_dir} for type {dataset_type}"
            )

        log.info(f"Found {len(self.files)} generated files in {root_dir}")

    def __len__(self):
        return len(self.files)

    def _parse_rxrx1_filename(self, fname: str) -> Dict[str, int]:
        """
        Expects: cell{c}_sirna{s}_{idx}.pt
        Example: cell0_sirna0_12.pt
        """
        basename = os.path.basename(fname)
        # Regex to find cell ID and sirna ID
        m = re.match(r"cell(\d+)_sirna(\d+)_", basename)
        if m:
            return {
                "cell_type_id": torch.tensor(int(m.group(1)), dtype=torch.long),
                "sirna_id": torch.tensor(int(m.group(2)), dtype=torch.long),
            }
        return {}

    def _parse_celeba_filename(self, fname: str) -> Dict[str, int]:
        """
        Expects: Attr0_Attr1_..._{idx}.png
        Example: Male0_Smiling0_Blond_Hair0_Eyeglasses0_101.png
        """
        basename = os.path.basename(fname)
        # Remove extension and the sample index (last part after underscore)
        name_no_ext = os.path.splitext(basename)[0]
        parts = name_no_ext.split("_")

        # The last part is the sample index (e.g. "101"), pop it
        # However, we must be careful if an attribute name ends with digits (unlikely here)
        # Standard format from generate_samples.py is "AttrVal_AttrVal_..._Index"
        parts = parts[:-1]

        cond_dict = {}

        # Reconstruct attributes.
        # Tricky case: "Blond_Hair0" splits into "Blond", "Hair0".
        # Logic: Accumulate tokens until one ends with a digit.
        buffer = []
        for p in parts:
            if not p:
                continue
            # Check if the token ends with 0 or 1 (binary attributes)
            # Note: This assumes attributes are binary 0/1.
            if p[-1] in ["0", "1"] and p[:-1].isalnum():
                # This is a value token. Combine buffer + this token base
                attr_name = "_".join(buffer + [p[:-1]])
                val = int(p[-1])
                cond_dict[attr_name] = torch.tensor(val, dtype=torch.long)
                buffer = []
            else:
                # Part of a multi-word attribute name (like "Blond")
                buffer.append(p)

        return cond_dict

    def __getitem__(self, idx):
        fpath = self.files[idx]

        # 1. Load Image
        if "rxrx1" in self.dataset_type:
            # Load Tensor (6, H, W)
            # map_location='cpu' is safer for dataloaders
            img = torch.load(fpath, map_location="cpu")

            # Convert 6ch -> 3ch RGB BEFORE applying encoder transforms
            # Encoder expects 3ch RGB for normalization
            if img.shape[0] == 6 and self.input_channels == 3:
                img = to_rgb(img.unsqueeze(0))[
                    0
                ]  # (6,H,W) -> (1,6,H,W) -> (1,3,H,W) -> (3,H,W)
        else:
            # Load PNG (H, W, 3) -> Convert to Tensor
            img_pil = Image.open(fpath).convert("RGB")
            # We delay transform to the end, but if transform expects PIL, perfect.
            # If transform expects Tensor (which base encoders usually do via ToTensor in compose),
            # we need to be careful.
            # Most transforms in `registry.py` start with Resize/CenterCrop.
            # Let's assume the transform handles PIL or we convert here.
            img = img_pil

        # 2. Parse Metadata
        if "rxrx1" in self.dataset_type:
            cond = self._parse_rxrx1_filename(fpath)
        else:
            cond = self._parse_celeba_filename(fpath)

        # 3. Apply Transform
        if self.transform:
            img = self.transform(img)

        # 4. Return in format compatible with extract_and_save
        # extract_and_save expects batch[1] to be a dict wrapper usually
        # We return: image, {"cond": cond}
        return img, {"cond": cond}


# --- Core Logic ---


def adapt_batch(images: torch.Tensor, target_channels: int) -> torch.Tensor:
    """Converts batch to match encoder's expected channels."""
    current_channels = images.shape[1]

    # Case A: RxRx1 (6ch) -> DINO/SigLIP (3ch) - GPU-friendly batch conversion
    if current_channels == 6 and target_channels == 3:
        return to_rgb(images)

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

        # --- Batch Unpacking Logic ---
        # 1. Standard DataModule tuple: (images, conditioning_dict)
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

            # Extract other keys (e.g. 'comp_category' if present)
            for k, v in cond_wrapper.items():
                if k == "cond":
                    continue
                meta_batch[k] = v.cpu() if isinstance(v, torch.Tensor) else v

        # 2. HuggingFace style or simple tuple: (images, labels)
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
            meta_batch["labels"] = (
                batch[1].cpu() if isinstance(batch[1], torch.Tensor) else batch[1]
            )

        # 3. Dict style
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
            # Helper to stack tensors or lists
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
                # Fallback for strings/ints
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
    device = "cuda:4" if torch.cuda.is_available() else "cpu"

    # 1. Select Encoder
    enc_name = cfg.get("encoder_name", "dinov2")
    if enc_name not in CONFIG_MAP:
        raise ValueError(
            f"Unknown encoder '{enc_name}'. Available: {list(CONFIG_MAP.keys())}"
        )

    enc_config = CONFIG_MAP[enc_name]
    log.info(f"Selected Encoder: {enc_name} ({enc_config.name})")

    # 2. Load Encoder Model
    encoder = load_encoder(enc_config, device=device)
    enc_transform = encoder.get_transform()
    log.info(f"Injecting Encoder Transform: {enc_transform}")

    # 3. Determine Mode: Real DataModule OR Generated Folder
    generated_path = cfg.get("generated_path", None)

    if generated_path:
        # --- PATH A: Generated Features ---
        if not os.path.exists(generated_path):
            raise FileNotFoundError(f"generated_path not found: {generated_path}")

        # Infer dataset type from config target or explicitly
        dataset_type = "rxrx1" if "RxRx1" in cfg.dataset._target_ else "celeba"

        log.info(
            f"Mode: GENERATED DATA | Type: {dataset_type} | Path: {generated_path}"
        )

        gen_dataset = GeneratedDataset(
            root_dir=generated_path,
            dataset_type=dataset_type,
            transform=enc_transform,
            input_channels=enc_config.input_channels,
        )
        bsize = cfg.dataset.get("batch_size", 32)
        loader = DataLoader(
            gen_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=32,  # cfg.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        # Save to feature_cache/{encoder}/generated_{foldername}.pt
        folder_name = os.path.basename(os.path.normpath(generated_path))
        base_cache_dir = "."
        feature_cache_dir = os.path.join(base_cache_dir, "feature_cache", enc_name)
        save_path = os.path.join(
            feature_cache_dir, f"generated_{folder_name}_features.pt"
        )

        extract_and_save(loader, encoder, save_path, device)

    else:
        # --- PATH B: Real DataModule (Original Logic) ---
        log.info(f"Mode: REAL DATAMODULE | Config: {cfg.dataset._target_}")

        dm_conf = instantiate(cfg.dataset)
        if "RxRx1" in cfg.dataset._target_:
            log.info("Initializing RxRx1DataModule...")
            dm = RxRx1DataModule(dm_conf)
        elif "Celeba" in cfg.dataset._target_:
            log.info("Initializing CelebaDataModule...")
            dm = CelebaDataModule(dm_conf)
        else:
            raise ValueError(f"Unknown dataset target: {cfg.dataset._target_}")

        # Inject Transform
        # Encoder transforms now include ToTensor/ConvertImageDtype and are robust
        # to both PIL images and Tensors, so we can safely replace dataset transforms

        # Patch datasets inside DM if they exist
        for split_name in [
            "train_ds",
            "val_ds",
            "test_ds",
            "ds_train",
            "ds_val",
            "ds_test",
        ]:
            if hasattr(dm, split_name):
                ds = getattr(dm, split_name)
                # CelebaDataset / RxRx1Dataset have self.transform
                if ds is not None and hasattr(ds, "transform"):
                    # Encoder transform now handles ToTensor + type conversion + normalization
                    ds.transform = enc_transform

        base_cache_dir = "."
        feature_cache_dir = os.path.join(base_cache_dir, "feature_cache", enc_name)
        os.makedirs(feature_cache_dir, exist_ok=True)
        log.info(f"Output Directory: {feature_cache_dir}")

        loaders = {
            "train": dm.train_dataloader(),
            "val": dm.val_dataloader(),
            # "test": dm.test_dataloader() # Optional
        }

        for split_name, loader in loaders.items():
            if loader is None:
                continue
            save_path = os.path.join(feature_cache_dir, f"{split_name}_features.pt")
            log.info(f"Processing split: {split_name}...")
            extract_and_save(loader, encoder, save_path, device)


if __name__ == "__main__":
    main()
