"""Dataset for posthoc alignment mapper training."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


def _verify_metadata_match(
    hidden_meta: Dict[str, torch.Tensor],
    siglip_meta: Dict[str, torch.Tensor],
    condition_keys: List[str],
) -> None:
    """Assert that metadata tensors match between hidden and SigLIP caches."""
    for key in condition_keys:
        h = hidden_meta.get(key)
        s = siglip_meta.get(key)
        if h is None or s is None:
            raise ValueError(f"Missing metadata key '{key}' in one of the caches")
        if not torch.equal(h, s):
            n_mismatch = (h != s).sum().item()
            raise ValueError(
                f"Metadata mismatch for '{key}': {n_mismatch}/{len(h)} samples differ. "
                "Ensure both caches use the same DataModule with shuffle=False."
            )


class RawHiddenSigLIPDataset(Dataset):
    """Loads pre-extracted raw_hidden + SigLIP target pairs for mapper training.

    Concatenates all timestep shards; hidden[t][i] maps to siglip[i].
    Train/val split is by image index (not by sample), so the same image's
    different timestep variants stay in the same split.
    """

    def __init__(
        self,
        hidden_dir: str,
        siglip_path: str,
        timesteps: Optional[List[float]] = None,
        val_fraction: float = 0.05,
        split: str = "train",
        seed: int = 42,
        condition_keys: Optional[List[str]] = None,
        seen_combos: Optional[List[Tuple[int, ...]]] = None,
    ):
        hidden_dir = Path(hidden_dir)
        if condition_keys is None:
            condition_keys = ["Male", "Smiling", "Blond_Hair", "Eyeglasses"]

        # Load SigLIP targets
        log.info(f"Loading SigLIP targets from {siglip_path}")
        siglip_data = torch.load(siglip_path, map_location="cpu", weights_only=False)
        self.siglip_targets = siglip_data["features"]  # (N_images, 1152)
        siglip_meta = siglip_data["metadata"]
        self.n_images = len(self.siglip_targets)
        log.info(f"  SigLIP: {self.siglip_targets.shape}")

        # Discover timestep shards
        if timesteps is None:
            shard_files = sorted(hidden_dir.glob("t*_hidden.pt"))
            timesteps = []
            for f in shard_files:
                t_str = f.stem.replace("_hidden", "").replace("t", "")
                timesteps.append(float(t_str))
        timesteps = sorted(timesteps)
        log.info(f"  Timesteps: {timesteps}")

        # Load all timestep shards
        self.hiddens = []  # List of (N_images, 768) tensors
        self.timestep_values = timesteps
        for t in timesteps:
            shard_path = hidden_dir / f"t{t}_hidden.pt"
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing shard: {shard_path}")
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            feats = shard["features"]
            assert feats.shape[0] == self.n_images, (
                f"Shard t={t} has {feats.shape[0]} samples, expected {self.n_images}"
            )
            # Verify metadata correspondence with SigLIP cache
            _verify_metadata_match(shard["metadata"], siglip_meta, condition_keys)
            self.hiddens.append(feats)
            log.info(f"  t={t}: {feats.shape}")

        self.n_timesteps = len(timesteps)

        # Optional filter to a whitelist of (cond_1, cond_2, ...) combos.
        # Only images whose metadata tuple is in `seen_combos` participate in
        # train/val (used for CelebA marginal mappers to avoid data leakage).
        if seen_combos is not None:
            seen_set = {tuple(int(x) for x in c) for c in seen_combos}
            cond_stack = torch.stack(
                [siglip_meta[k].to(torch.long) for k in condition_keys], dim=1
            )  # (N, K)
            mask = torch.zeros(self.n_images, dtype=torch.bool)
            for i in range(self.n_images):
                if tuple(cond_stack[i].tolist()) in seen_set:
                    mask[i] = True
            kept = int(mask.sum().item())
            log.info(
                f"  seen_combos filter: kept {kept}/{self.n_images} images "
                f"({len(seen_set)} allowed combos)"
            )
            valid_image_indices = mask.nonzero(as_tuple=False).squeeze(1).numpy()
        else:
            valid_image_indices = np.arange(self.n_images)

        # Train/val split by image index
        rng = np.random.default_rng(seed)
        all_indices = valid_image_indices.copy()
        rng.shuffle(all_indices)
        n_val = max(1, int(len(all_indices) * val_fraction))
        if split == "val":
            self.image_indices = all_indices[:n_val]
        else:
            self.image_indices = all_indices[n_val:]

        log.info(
            f"  Split={split}: {len(self.image_indices)} images × {self.n_timesteps} "
            f"timesteps = {len(self)} samples"
        )

    def __len__(self) -> int:
        return len(self.image_indices) * self.n_timesteps

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
        """Return (hidden_768, siglip_target_1152, timestep, image_index)."""
        t_idx = idx // len(self.image_indices)
        img_pos = idx % len(self.image_indices)
        img_idx = self.image_indices[img_pos]

        hidden = self.hiddens[t_idx][img_idx]
        target = self.siglip_targets[img_idx]
        timestep = self.timestep_values[t_idx]

        return hidden, target, timestep, img_idx
