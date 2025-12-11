from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CelebAConfig:
    root: str
    split: str = "train"
    attributes: Optional[Dict[str, int]] = None
    """attributes example:
    {
        "Smiling": 1,
        "Male": 0,
        "Wavy_Hair": 1
    }
    """


class CelebADataset:
    """Minimal CelebA dataset stub.

    Later this will:
      - load attribute annotations
      - filter by user-specified attribute values
      - load image paths
    """

    def __init__(self, cfg: CelebAConfig):
        self.cfg = cfg
        self.samples = []  # placeholder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError("CelebADataset.__getitem__ is not implemented yet.")
