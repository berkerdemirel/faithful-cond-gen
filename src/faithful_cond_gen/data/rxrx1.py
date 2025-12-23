import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

# ---- Utilities: intensity, RGB mapping, augmentations ----


def rescale_intensity(
    arr: torch.Tensor,
    bounds: Tuple[float, float] = (0.5, 99.5),
    out_range: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Percentile-based contrast stretching per image (GPU-friendly, batch-compatible).

    - If arr is (B,C,H,W): percentiles computed independently for each item in B,
      over all channels/pixels (same as flatten() in old code).
    - If arr is (C,H,W) or (H,W): treated as a single image.
    - Uses the same subsampling scheme: flatten()[::100].
    """
    # Normalize shape to (B, C, H, W)
    squeeze_b = False
    if arr.dim() == 2:
        arr = arr.unsqueeze(0).unsqueeze(0)
        squeeze_b = True
    elif arr.dim() == 3:
        arr = arr.unsqueeze(0)
        squeeze_b = True
    elif arr.dim() != 4:
        raise ValueError(f"Expected 2D/3D/4D tensor, got {tuple(arr.shape)}")

    B = arr.shape[0]

    # Match original scaling condition, but PER IMAGE
    arr = arr.float()
    arr_min = arr.amin(dim=(1, 2, 3))
    arr_max = arr.amax(dim=(1, 2, 3))
    need_div = (arr_min >= 0) & (arr_max > 1.0 + 1e-3)
    if need_div.any():
        # broadcast (B,1,1,1)
        div = torch.where(
            need_div, torch.full_like(arr_min, 255.0), torch.ones_like(arr_min)
        ).view(B, 1, 1, 1)
        arr = arr / div

    # Per-image sampling: flatten()[::100]
    flat = arr.reshape(B, -1)
    sample = flat[:, ::100]

    q = torch.tensor(
        [bounds[0] / 100.0, bounds[1] / 100.0],
        device=arr.device,
        dtype=arr.dtype,
    )
    # Per-image quantiles along the flattened dimension
    p = torch.quantile(sample, q, dim=1)  # (2, B)
    lo = p[0].view(B, 1, 1, 1)
    hi = p[1].view(B, 1, 1, 1)

    arr = torch.clamp(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-6)
    arr = arr * (out_range[1] - out_range[0]) + out_range[0]

    if squeeze_b:
        arr = arr.squeeze(0)
        if arr.shape[0] == 1:
            arr = arr.squeeze(0)
    return arr


def to_rgb(img: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """Convert 6-channel Cell Painting image to RGB (GPU-friendly, batch-compatible).

    Expects input of shape (B, C, H, W) with C <= 6.

    Reference: https://github.com/recursionpharma/rxrx1-utils/blob/d34b2b0db0af1cb4fe357573bb8de76bd042b34f/rxrx/io.py#L61
    """
    if img.dim() != 4:
        raise ValueError(f"Expected (B,C,H,W), got {tuple(img.shape)}")

    b, c, h, w = img.shape
    num_channels_required = 6

    # Pad/truncate to 6 channels (same behavior as original)
    prepped = torch.zeros(
        (b, num_channels_required, h, w), dtype=img.dtype, device=img.device
    )
    if c < num_channels_required:
        prepped[:, :c].copy_(img)
    else:
        prepped.copy_(img[:, :num_channels_required])

    # Fixed color map
    rgb_map = torch.tensor(
        [
            [0, 0, 1],  # blue
            [0, 1, 0],  # green
            [1, 0, 0],  # red
            [0, 1, 1],  # cyan
            [1, 0, 1],  # magenta
            [1, 1, 0],  # yellow
        ],
        dtype=dtype,
        device=prepped.device,
    )

    # (B,6,H,W) x (6,3) -> (B,3,H,W)
    rgb_img = torch.einsum("nchw,ct->nthw", prepped.to(dtype=dtype), rgb_map) / 3.0
    return rescale_intensity(rgb_img, bounds=(0.1, 99.9))


class RandomExactRotation:
    """Rotate by given angles with probability p."""

    def __init__(self, angles: Sequence[int], p: float = 0.5):
        self.angles = list(angles)
        self.p = float(p)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if np.random.rand() < self.p:
            angle = int(np.random.choice(self.angles))
            return TF.rotate(img, angle)
        return img


class CustomTransform:
    """Resize + optional flips/rotations + optional per-channel standardization."""

    def __init__(
        self,
        augment: bool = False,
        normalize: bool = False,
        img_size: Tuple[int, int] = (512, 512),
        reduce_channels: bool = False,
    ):
        self.augment = augment
        self.normalize = normalize
        self.resize_shape = img_size
        self.reduce_channels = reduce_channels

    def self_standardize(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (C, H, W)
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True) + 1e-6
        return (x - mean) / std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (C, H, W)
        t = [T.Resize(self.resize_shape, interpolation=Image.BICUBIC)]

        if self.augment:
            # choose p so (1-p)^3 ~ 0.5
            t.append(T.RandomHorizontalFlip(p=0.2))
            t.append(T.RandomVerticalFlip(p=0.2))
            t.append(RandomExactRotation(angles=[90, 180, 270], p=0.2))

        transform = T.Compose(t)

        # if not reducing channels, assume raw [0,255], rescale to [0,1]
        if not self.reduce_channels:
            x = x / 255.0  # else assume already in [0,1] bc of to_rgb rescaling

        x = transform(x)

        if self.normalize:
            x = self.self_standardize(x)

        return x


# ---- Dataset ----


CELL_TYPE_TO_LABEL: Dict[str, int] = {
    "HEPG2": 0,
    "HUVEC": 1,
    "RPE": 2,
    "U2OS": 3,
}
LABEL_TO_CELL_TYPE: Dict[int, str] = {v: k for k, v in CELL_TYPE_TO_LABEL.items()}

# Canonical ordering of conditioning keys (used for stacking into tensors)
RXRX1_COND_KEYS: List[str] = ["cell_type_id", "sirna_id"]


@dataclass
class RxRx1DatasetConfig:
    img_size: Tuple[int, int] = (512, 512)
    resize: Tuple[int, int] = (512, 512)
    reduce_channels: bool = False
    augment: bool = False
    normalize: bool = False
    use_numpy: bool = True
    use_parquet: bool = False
    split: str = "train"


class RxRx1Dataset(Dataset):
    """RxRx1 dataset.

    Expects metadata with at least:
      - 'sirna_id' (perturbation id, int)
      - 'cell_type' (string: HEPG2/HUVEC/RPE/U2OS)
      - one of:
          'parquet_path', 'numpy_path', or 'image_paths' (list of 6 PNGs or string repr)

    __getitem__ returns:
      image: torch.Tensor (C,H,W)
      conditioning: dict with keys:
        - 'sirna_id' (int)
        - 'cell_type_id' (int)
        - 'cell_type' (str)
    """

    def __init__(self, metadata: pd.DataFrame, cfg: RxRx1DatasetConfig):
        self.metadata = metadata.reset_index(drop=True)
        self.cfg = cfg

        self.use_numpy = cfg.use_numpy
        self.use_parquet = cfg.use_parquet

        # Transform
        if cfg.split == "train":
            self.transform = CustomTransform(
                augment=cfg.augment,
                normalize=cfg.normalize,
                img_size=cfg.resize,
                reduce_channels=cfg.reduce_channels,
            )
        else:
            self.transform = CustomTransform(
                augment=False,
                normalize=cfg.normalize,
                img_size=cfg.resize,
                reduce_channels=cfg.reduce_channels,
            )

        # Canonicalize cell_type_id column if not present
        if "cell_type_id" not in self.metadata.columns:
            self.metadata["cell_type_id"] = self.metadata["cell_type"].map(
                CELL_TYPE_TO_LABEL
            )

        # For convenience
        self.sirna_ids = self.metadata["sirna_id"].to_numpy()
        self.cell_type_ids = self.metadata["cell_type_id"].to_numpy()

        # Optional composition category (seen / rare / unseen)
        if "comp_category" in self.metadata.columns:
            self.comp_categories = self.metadata["comp_category"].astype(str).to_numpy()
        else:
            self.comp_categories = None

    def __len__(self) -> int:
        return len(self.metadata)

    def _load_sample_array(self, idx: int) -> np.ndarray:
        row = self.metadata.iloc[idx]

        if (
            self.use_parquet
            and "parquet_path" in row
            and isinstance(row["parquet_path"], str)
        ):
            with pa.memory_map(row["parquet_path"], "r") as source:
                table = pq.read_table(source)
                sample = np.array(table.column("data")).reshape(
                    6, *self.cfg.img_size
                )  # (C,H,W)
            return sample

        if (
            self.use_numpy
            and "numpy_path" in row
            and isinstance(row["numpy_path"], str)
        ):
            sample = np.load(row["numpy_path"], mmap_mode="r")
            return np.array(sample)

        # Fallback: stack PNGs from image_paths
        image_paths = row["image_paths"]
        if isinstance(image_paths, str):
            # if stored as string representation of list
            image_paths = eval(image_paths)
        images = [np.array(Image.open(p)) for p in image_paths]
        sample = np.stack(images, axis=0)  # (C,H,W)
        return sample

    def __getitem__(self, idx: int):
        sample_np = self._load_sample_array(idx)  # (C,H,W)
        x = torch.from_numpy(sample_np).float()

        # If we need to reduce to RGB
        if self.cfg.reduce_channels:
            x = to_rgb(x.unsqueeze(0))[0]  # (3,H,W) after RGB mapping

        x = self.transform(x)  # (C,H,W)

        sirna_id = torch.tensor(self.sirna_ids[idx], dtype=torch.long)
        cell_type_id = torch.tensor(self.cell_type_ids[idx], dtype=torch.long)

        # Use canonical key order defined in RXRX1_COND_KEYS
        conditioning = {
            "cond": {
                "cell_type_id": cell_type_id,
                "sirna_id": sirna_id,
            }
        }

        if self.comp_categories is not None:
            conditioning["comp_category"] = self.comp_categories[idx]
        return x, conditioning


# ---- DataModule-like helper ----


@dataclass
class RxRx1DataConfig:
    data_dir: str
    img_size: Tuple[int, int] = (512, 512)
    resize: Tuple[int, int] = (512, 512)
    reduce_channels: bool = False
    augment_train: bool = False
    normalize: bool = False
    use_numpy: bool = True
    use_parquet: bool = False
    batch_size: int = 32
    num_workers: int = 4
    val_size: float = 0.1
    seed: int = 1337
    rare_threshold: int = 50
    held_out_pairs: Optional[Sequence[Tuple[int, int]]] = None


class RxRx1DataModule(pl.LightningDataModule):
    """Lightweight data module for RxRx1.

    Responsibilities:
      - load metadata.csv or metadata_extended.csv
      - add cell_type_id
      - define train/val/test splits
      - provide get_dataset / get_dataloader with optional filtering by condition
    """

    def __init__(self, cfg: RxRx1DataConfig):
        super().__init__()
        self.cfg = cfg
        self.data_dir = cfg.data_dir

        metadata_path_extended = os.path.join(self.data_dir, "metadata_extended.csv")
        metadata_path = os.path.join(self.data_dir, "metadata.csv")

        if os.path.exists(metadata_path_extended):
            metadata = pd.read_csv(metadata_path_extended)
        elif os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
        else:
            raise FileNotFoundError(
                f"Could not find metadata.csv or metadata_extended.csv in {self.data_dir}"
            )

        # Add cell_type_id
        if "cell_type_id" not in metadata.columns:
            metadata["cell_type_id"] = metadata["cell_type"].map(CELL_TYPE_TO_LABEL)

        self.metadata = metadata

        # self.held_out_pairs = (
        #     set(cfg.held_out_pairs) if cfg.held_out_pairs is not None else None
        # )
        if cfg.held_out_pairs is not None:
            self.held_out_pairs = {(int(a), int(b)) for a, b in cfg.held_out_pairs}
        else:
            self.held_out_pairs = None
        # Define splits
        self._make_splits()

        # Add composition categories (seen / rare / unseen)
        self._add_composition_categories()

        # Print split sizes
        self._print_split_info()

    # ---- LIGHTNING HOOKS ----

    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("val")

    def test_dataloader(self):
        return self.get_dataloader("test")

    def _make_splits(self):
        md = self.metadata

        if "dataset" in md.columns:
            train_md = md[md["dataset"] == "train"]
            test_md = md[md["dataset"] == "test"]
        else:
            # no explicit dataset column, treat all as train+val, no test
            train_md = md
            test_md = md.iloc[0:0]  # empty

        if self.cfg.val_size > 0.0 and len(train_md) > 0:
            train_idx, val_idx = train_test_split(
                train_md.index,
                test_size=self.cfg.val_size,
                random_state=self.cfg.seed,
                # could stratify by sirna_id if you want
            )
            md["train_index"] = md.index.isin(train_idx)
            md["val_index"] = md.index.isin(val_idx)
        else:
            md["train_index"] = (
                md["dataset"] == "train" if "dataset" in md.columns else True
            )
            md["val_index"] = False

        if "dataset" in md.columns:
            md["test_index"] = md["dataset"] == "test"
        else:
            md["test_index"] = False

        if self.held_out_pairs is not None:
            mask_held_out = md[["cell_type_id", "sirna_id"]].apply(
                lambda row: (row["cell_type_id"], row["sirna_id"])
                in self.held_out_pairs,
                axis=1,
            )
            # ensure these never appear in train
            md.loc[mask_held_out, "train_index"] = False
            not_eval = mask_held_out & ~(md["val_index"] | md["test_index"])
            md.loc[not_eval, "val_index"] = True

        self.metadata = md

    def _add_composition_categories(self) -> None:
        """Tag each (cell_type_id, sirna_id) pair as seen / rare / unseen."""
        md = self.metadata  # <— this line is required

        if "train_index" not in md.columns:
            md["comp_category"] = "seen"
            self.metadata = md
            return

        train_md = md[md["train_index"]]
        if len(train_md) == 0:
            md["comp_category"] = "seen"
            self.metadata = md
            return

        counts = train_md.groupby(["cell_type_id", "sirna_id"]).size().to_dict()

        rare_threshold = self.cfg.rare_threshold

        def categorize(row: pd.Series) -> str:
            key = (row["cell_type_id"], row["sirna_id"])
            c = counts.get(key, None)
            if c is None:
                return "unseen"
            if c < rare_threshold:
                return "rare"
            return "seen"

        md["comp_category"] = md.apply(categorize, axis=1)
        self.metadata = md

    def _print_split_info(self) -> None:
        """Print information about train/val/test splits."""
        md = self.metadata

        train_count = md["train_index"].sum()
        val_count = md["val_index"].sum()
        test_count = md["test_index"].sum()
        total = len(md)

        print(f"\n{'='*60}")
        print(f"RxRx1 Dataset Split Information")
        print(f"{'='*60}")
        print(f"Total samples:      {total:>8}")
        print(f"Train samples:      {train_count:>8} ({100*train_count/total:>5.1f}%)")
        print(f"Validation samples: {val_count:>8} ({100*val_count/total:>5.1f}%)")
        print(f"Test samples:       {test_count:>8} ({100*test_count/total:>5.1f}%)")

        if self.held_out_pairs is not None:
            # Count samples in held_out_pairs
            mask_held_out = md[["cell_type_id", "sirna_id"]].apply(
                lambda row: (row["cell_type_id"], row["sirna_id"])
                in self.held_out_pairs,
                axis=1,
            )
            held_out_count = mask_held_out.sum()
            held_out_in_val = (mask_held_out & md["val_index"]).sum()
            held_out_in_test = (mask_held_out & md["test_index"]).sum()

            print(f"\nHeld-out pairs info:")
            print(f"  Number of held-out pairs:     {len(self.held_out_pairs):>8}")
            print(f"  Samples in held-out pairs:    {held_out_count:>8}")
            print(f"    - Moved to validation:      {held_out_in_val:>8}")
            print(f"    - Already in test:          {held_out_in_test:>8}")

        print(f"{'='*60}\n")

    def _filtered_metadata(
        self,
        split: str,
        cell_types: Optional[Sequence[int]] = None,
        perturbations: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        md = self.metadata

        if split == "train":
            md = md[md["train_index"]]
        elif split == "val":
            md = md[md["val_index"]]
        elif split == "test":
            md = md[md["test_index"]]
        else:
            raise ValueError(f"Unknown split: {split}")

        if cell_types is not None:
            md = md[md["cell_type_id"].isin(cell_types)]

        if perturbations is not None:
            md = md[md["sirna_id"].isin(perturbations)]

        return md

    def get_dataset(
        self,
        split: str,
        cell_types: Optional[Sequence[int]] = None,
        perturbations: Optional[Sequence[int]] = None,
        override_cfg: Optional[Dict] = None,
    ) -> RxRx1Dataset:
        md = self._filtered_metadata(split, cell_types, perturbations)
        if len(md) == 0:
            raise ValueError(f"No samples found for split={split} with given filters.")

        ds_cfg = RxRx1DatasetConfig(
            img_size=self.cfg.img_size,
            resize=self.cfg.resize,
            reduce_channels=self.cfg.reduce_channels,
            augment=(self.cfg.augment_train and split == "train"),
            normalize=self.cfg.normalize,
            use_numpy=self.cfg.use_numpy,
            use_parquet=self.cfg.use_parquet,
            split=split,
        )
        if override_cfg is not None:
            for k, v in override_cfg.items():
                setattr(ds_cfg, k, v)

        return RxRx1Dataset(md, ds_cfg)

    def get_dataloader(
        self,
        split: str,
        cell_types: Optional[Sequence[int]] = None,
        perturbations: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        override_cfg: Optional[Dict] = None,
    ) -> DataLoader:
        dataset = self.get_dataset(
            split=split,
            cell_types=cell_types,
            perturbations=perturbations,
            override_cfg=override_cfg,
        )

        if batch_size is None:
            batch_size = self.cfg.batch_size

        if shuffle is None:
            shuffle = split == "train"

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )

    def available_conditions(self, split: str = "train") -> pd.DataFrame:
        """Return a small table of (cell_type_id, sirna_id, comp_category, count) for the given split."""

        md = self._filtered_metadata(split, None, None)

        group_cols = ["cell_type_id", "sirna_id"]
        if "comp_category" in md.columns:
            group_cols.append("comp_category")

        grouped = md.groupby(group_cols).size().reset_index(name="count")
        return grouped

    def get_matching_dataset(
        self, split: str, conditions: Dict[str, int], max_samples: Optional[int] = None
    ) -> RxRx1Dataset:
        """
        Return a dataset containing only samples matching the specific conditions.
        Args:
            split: 'train', 'val', or 'test'
            conditions: Dict containing 'cell_type_id' and/or 'sirna_id'.
                        e.g. {'cell_type_id': 1, 'sirna_id': 1138}
        """
        # 1. Map generic dict to specific filter arguments
        cell_types = None
        if "cell_type_id" in conditions:
            cell_types = [conditions["cell_type_id"]]

        perturbations = None
        if "sirna_id" in conditions:
            perturbations = [conditions["sirna_id"]]

        # 2. Get the filtered dataset using existing logic
        ds = self.get_dataset(
            split=split,
            cell_types=cell_types,
            perturbations=perturbations,
            override_cfg=None,
        )

        # 3. Handle max_samples if requested
        if max_samples is not None and len(ds) > max_samples:
            # RxRx1Dataset wraps a DataFrame, so we can just slice it
            ds.metadata = ds.metadata.iloc[:max_samples].reset_index(drop=True)
            # Re-cache numpy arrays for convenience
            ds.sirna_ids = ds.metadata["sirna_id"].to_numpy()
            ds.cell_type_ids = ds.metadata["cell_type_id"].to_numpy()
            if ds.comp_categories is not None:
                ds.comp_categories = ds.comp_categories[:max_samples]

        return ds
