import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class ScoreFunction(ABC):
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.stats = {}  # Dictionary to store fitted params (mu, sigma, etc.)

    @abstractmethod
    def fit(self, features: torch.Tensor, metadata: Dict[str, Any]) -> None:
        """
        Fits the scoring model on real data.

        Args:
            features: (N, D) tensor of real features.
            metadata: Dictionary containing conditioning info (e.g. {'cell_type': [...], ...})
                      Length of arrays in metadata must match N.
        """
        pass

    @abstractmethod
    def score(self, features: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Calculates per-sample scores for new data.

        Args:
            features: (M, D) tensor of generated/test features.
            metadata: Conditioning info for these M samples.

        Returns:
            (M,) tensor of scores. Lower/Higher meaning depends on metric,
            but for Mahalanobis, higher = less faithful (further away).
        """
        pass

    def save_stats(self, path: str):
        """Saves fitted statistics to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.stats, path)
        print(f"Stats saved to {path}")

    def load_stats(self, path: str):
        """Loads fitted statistics from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stats file not found: {path}")
        self.stats = torch.load(path, map_location=self.device)
        print(f"Stats loaded from {path}")

    def _hash_condition(self, cond_dict: Dict[str, Any]) -> tuple:
        """
        Helper to convert a dictionary of condition scalar tensors/ints
        into a hashable tuple key.

        Example: {'cell': 1, 'sirna': 5} -> (('cell', 1), ('sirna', 5))
        """
        # Sort keys to ensure consistent ordering
        items = []
        for k in sorted(cond_dict.keys()):
            val = cond_dict[k]
            if isinstance(val, torch.Tensor):
                val = val.item()
            items.append((k, val))
        return tuple(items)
