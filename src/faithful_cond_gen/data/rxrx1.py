from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RxRx1Config:
    root: str
    split: str = "train"
    cell_types: Optional[List[str]] = None
    perturbations: Optional[List[str]] = None


class RxRx1Dataset:
    """Minimal stub for RxRx1 dataset.

    For now this just stores config and an empty list of samples.
    We'll later load real metadata and implement filtering.
    """

    def __init__(self, cfg: RxRx1Config):
        self.cfg = cfg
        self.samples = []  # placeholder; will hold (path, labels, etc.)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        raise NotImplementedError("RxRx1Dataset.__getitem__ is not implemented yet.")
