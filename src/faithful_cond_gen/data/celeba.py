import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# Small default subset of attributes for compositional analysis
COMPOSITION_ATTRS_DEFAULT: List[str] = [
    "Male",
    "Smiling",
    "Blond_Hair",
    "Eyeglasses",
]


@dataclass
class CelebaDataConfig:
    """Config for CelebA data access (HuggingFace flwrlabs/celeba)."""

    cache_dir: Optional[str] = None
    image_size: Tuple[int, int] = (224, 224)
    augment_train: bool = False
    normalize: bool = True
    batch_size: int = 64
    num_workers: int = 4
    # threshold for rare vs seen combinations
    rare_threshold: int = 50
    held_out_combos: Optional[Sequence[Tuple[int, ...]]] = None
    selected_attrs: Optional[Sequence[str]] = None


class CelebaDataset(Dataset):
    """CelebA dataset for this project.

    __getitem__ returns:
      image: torch.Tensor (3, H, W)
      conditioning: dict with keys:
        - 'attrs': (K,) tensor of selected attributes (0/1 floats)
        - 'comp_category': str in {'seen', 'rare', 'unseen', 'contradictory'}
    """

    def __init__(
        self,
        hf_dataset,
        attr_names: List[str],
        selected_attrs: List[str],
        transform: T.Compose,
        comp_categories: Optional[Sequence[str]] = None,
    ):
        self.dataset = hf_dataset
        self.attr_names = attr_names
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.comp_categories = (
            list(comp_categories) if comp_categories is not None else None
        )

        # map attr -> index in full attr vector for fast lookup
        self.attr_index: Dict[str, int] = {
            name: i for i, name in enumerate(self.attr_names)
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = self.transform(image)

        cond_dict = {}
        for name in self.selected_attrs:
            # Convert float 0.0/1.0 -> long 0/1
            val = int(sample[name])
            cond_dict[name] = torch.tensor(val, dtype=torch.long)

        conditioning = {
            "cond": cond_dict,
        }
        if self.comp_categories is not None:
            conditioning["comp_category"] = self.comp_categories[idx]

        return image, conditioning


class CelebaDataModule(pl.LightningDataModule):
    """Lightweight DataModule-style wrapper around flwrlabs/celeba.

    Responsibilities:
      - load HF splits (train/validation/test)
      - define selected attributes for compositions
      - tag each sample with 'seen'/'rare'/'unseen'/'contradictory'
      - provide get_dataset / get_dataloader / available_conditions
    """

    def __init__(
        self,
        cfg: CelebaDataConfig,
    ):
        super().__init__()
        self.cfg = cfg
        selected_attrs = cfg.selected_attrs
        # load HF splits once; caching handled by datasets
        self.ds_train = load_dataset(
            "flwrlabs/celeba",
            split="train",
            cache_dir=cfg.cache_dir,
        )
        self.ds_val = load_dataset(
            "flwrlabs/celeba",
            split="valid",
            cache_dir=cfg.cache_dir,
        )
        self.ds_test = load_dataset(
            "flwrlabs/celeba",
            split="test",
            cache_dir=cfg.cache_dir,
        )

        # infer attribute names (all non-image, non-id keys)
        sample = self.ds_train[0]
        self.attr_names: List[str] = [
            k
            for k in sample.keys()
            if k not in ["image", "label", "image_id", "img_id", "celeb_id"]
        ]

        if selected_attrs is None:
            # use default subset intersected with what actually exists
            sel = [a for a in COMPOSITION_ATTRS_DEFAULT if a in self.attr_names]
            if not sel:
                raise ValueError(
                    "None of COMPOSITION_ATTRS_DEFAULT found in CelebA attributes."
                )
            self.selected_attrs = sel
        else:
            for a in selected_attrs:
                if a not in self.attr_names:
                    raise ValueError(f"Selected attribute '{a}' not in CelebA attrs.")
            self.selected_attrs = list(selected_attrs)

        self.held_out_combos = (
            set(cfg.held_out_combos) if cfg.held_out_combos is not None else None
        )
        # Apply held-out combos to the *actual* training HF dataset so lengths stay aligned
        if self.held_out_combos is not None:
            md_train_full = self._hf_to_dataframe(self.ds_train)
            md_train_full["combo"] = md_train_full.apply(self._combo_key, axis=1)
            keep_idx = md_train_full.index[
                ~md_train_full["combo"].isin(self.held_out_combos)
            ].tolist()

            # Replace ds_train with the filtered view
            self.ds_train = self.ds_train.select(keep_idx)
        # composition categories per split
        (
            self.train_comp_categories,
            self.val_comp_categories,
            self.test_comp_categories,
            self._md_train,
            self._md_val,
            self._md_test,
        ) = self._compute_composition_categories()

    # ---- LIGHTNING HOOKS ----

    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("val")

    def test_dataloader(self):
        return self.get_dataloader("test")

    # ---- internal: transforms ----

    def _build_transform(self, train: bool) -> T.Compose:
        t: List[T.transforms] = []

        # basic resize / center crop like DINOv2-ish
        t.append(T.Resize(self.cfg.image_size))
        t.append(T.CenterCrop(self.cfg.image_size))

        if train and self.cfg.augment_train:
            t.append(T.RandomHorizontalFlip(p=0.5))

        t.append(T.ToTensor())

        # if self.cfg.normalize:
        #     # ImageNet normalization
        #     t.append(
        #         T.Normalize(
        #             mean=[0.485, 0.456, 0.406],
        #             std=[0.229, 0.224, 0.225],
        #         )
        #     )

        return T.Compose(t)

    # ---- internal: composition logic ----

    def _hf_to_dataframe(self, hf_ds) -> pd.DataFrame:
        """Convert a HF Dataset into a pandas DataFrame of selected attrs."""
        cols = {a: hf_ds[a] for a in self.selected_attrs}
        # HF gives lists; they become columns
        md = pd.DataFrame(cols)
        # ensure ints
        for a in self.selected_attrs:
            md[a] = md[a].astype(int)
        return md

    def _combo_key(self, row: pd.Series) -> Tuple[int, ...]:
        return tuple(int(row[a]) for a in self.selected_attrs)

    def _is_contradictory(self, row: pd.Series) -> bool:
        """Placeholder contradictory rule.

        For now we keep everything non-contradictory; you can plug in custom
        rules here, e.g. impossible or semantically weird combinations.
        """
        # Example (commented, optional):
        # if (row["Male"] == 1) and (row["Wearing_Lipstick"] == 1) and (row["Mustache"] == 1):
        #     return True
        return False

    def _compute_composition_categories(self):
        """Compute 'seen'/'rare'/'unseen'/'contradictory' labels for each split.

        - 'seen': combination appears in train with count >= rare_threshold
        - 'rare': combination appears in train with 1 <= count < rare_threshold
        - 'unseen': combination never appears in train
        - 'contradictory': optional rule-based override
        """
        md_train = self._hf_to_dataframe(self.ds_train)
        md_val = self._hf_to_dataframe(self.ds_val)
        md_test = self._hf_to_dataframe(self.ds_test)

        md_train["combo"] = md_train.apply(self._combo_key, axis=1)

        # counts per combination in train
        counts = md_train["combo"].value_counts()
        counts_dict: Dict[Tuple[int, ...], int] = counts.to_dict()

        def categorize(row: pd.Series, split: str) -> str:
            combo = self._combo_key(row)

            if self._is_contradictory(row):
                return "contradictory"

            if combo not in counts_dict:
                # Appears only in val/test, never in train
                return "unseen"

            c = counts_dict[combo]
            if c < self.cfg.rare_threshold:
                return "rare"
            return "seen"

        md_train["comp_category"] = md_train.apply(
            lambda r: categorize(r, "train"), axis=1
        )
        md_val["comp_category"] = md_val.apply(lambda r: categorize(r, "val"), axis=1)
        md_test["comp_category"] = md_test.apply(
            lambda r: categorize(r, "test"), axis=1
        )

        train_comp_categories = md_train["comp_category"].tolist()
        val_comp_categories = md_val["comp_category"].tolist()
        test_comp_categories = md_test["comp_category"].tolist()

        return (
            train_comp_categories,
            val_comp_categories,
            test_comp_categories,
            md_train,
            md_val,
            md_test,
        )

    # ---- public API ----

    def get_dataset(
        self,
        split: str,
        override_cfg: Optional[Dict] = None,
    ) -> CelebaDataset:
        """Return a CelebaDataset for the given split ('train', 'val', 'test')."""
        split = split.lower()
        if split == "validation":
            split = "val"

        cfg = self.cfg
        if override_cfg is not None:
            # shallow override for things like image_size, augment_train, normalize
            for k, v in override_cfg.items():
                setattr(cfg, k, v)

        if split == "train":
            hf_ds = self.ds_train
            comp_cats = self.train_comp_categories
            transform = self._build_transform(train=True)
        elif split == "val":
            hf_ds = self.ds_val
            comp_cats = self.val_comp_categories
            transform = self._build_transform(train=False)
        elif split == "test":
            hf_ds = self.ds_test
            comp_cats = self.test_comp_categories
            transform = self._build_transform(train=False)
        else:
            raise ValueError(f"Unknown split: {split}")

        return CelebaDataset(
            hf_dataset=hf_ds,
            attr_names=self.attr_names,
            selected_attrs=self.selected_attrs,
            transform=transform,
            comp_categories=comp_cats,
        )

    def get_dataloader(
        self,
        split: str,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        override_cfg: Optional[Dict] = None,
    ) -> DataLoader:
        ds = self.get_dataset(split=split, override_cfg=override_cfg)

        if batch_size is None:
            batch_size = self.cfg.batch_size

        if shuffle is None:
            shuffle = split.lower() == "train"

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=(split.lower() == "train"),
        )

    def available_conditions(self, split: str = "train") -> pd.DataFrame:
        """Return a small table of (attrs..., comp_category, count) for the split."""
        split = split.lower()
        if split == "validation":
            split = "val"

        if split == "train":
            md = self._md_train
        elif split == "val":
            md = self._md_val
        elif split == "test":
            md = self._md_test
        else:
            raise ValueError(f"Unknown split: {split}")

        grouped = (
            md.groupby(self.selected_attrs + ["comp_category"])
            .size()
            .reset_index(name="count")
        )
        return grouped

    def get_matching_dataset(
        self, split: str, conditions: Dict[str, int], max_samples: Optional[int] = None
    ) -> CelebaDataset:
        """
        Return a dataset containing only samples matching the specific conditions.
        Args:
            split: 'train', 'val', or 'test'
            conditions: Dict {attribute_name: value (0/1)}, e.g. {'Male': 1, 'Smiling': 0}
            max_samples: If provided, limit result to this many samples.
        """
        split = split.lower()
        if split == "validation":
            split = "val"

        # 1. Get appropriate metadata and source objects
        if split == "train":
            md = self._md_train
            hf_ds_full = self.ds_train
            comp_cats_full = self.train_comp_categories
            transform = self._build_transform(train=True)
        elif split == "val":
            md = self._md_val
            hf_ds_full = self.ds_val
            comp_cats_full = self.val_comp_categories
            transform = self._build_transform(train=False)
        elif split == "test":
            md = self._md_test
            hf_ds_full = self.ds_test
            comp_cats_full = self.test_comp_categories
            transform = self._build_transform(train=False)
        else:
            raise ValueError(f"Unknown split: {split}")

        # 2. Build Filter Mask
        # Start with all True
        mask = pd.Series(True, index=md.index)

        for attr, val in conditions.items():
            if attr not in md.columns:
                raise ValueError(f"Attribute '{attr}' not found in metadata.")
            mask &= md[attr] == val

        # 3. Get Indices
        indices = md.index[mask].tolist()

        if len(indices) == 0:
            print(
                f"Warning: No samples found for conditions {conditions} in split {split}"
            )
            # Return empty dataset logic could be complex; for now let HF handle empty select

        if max_samples is not None and len(indices) > max_samples:
            indices = indices[:max_samples]

        # 4. Subset Data
        # HF Datasets .select() creates a cheap view/copy
        hf_subset = hf_ds_full.select(indices)

        # Subset composition categories if they exist
        comp_cats_subset = None
        if comp_cats_full is not None:
            comp_cats_subset = [comp_cats_full[i] for i in indices]

        # 5. Return new Dataset instance
        return CelebaDataset(
            hf_dataset=hf_subset,
            attr_names=self.attr_names,
            selected_attrs=self.selected_attrs,
            transform=transform,
            comp_categories=comp_cats_subset,
        )
