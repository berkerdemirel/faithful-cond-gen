from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn
import torchvision.transforms as T
from faithful_cond_gen.eval.configs.encoder_config import EncoderConfig


class BaseEncoder(nn.Module, ABC):
    def __init__(self, config: EncoderConfig, device: str = "cuda"):
        super().__init__()
        self.cfg = config
        self.device = device

    @abstractmethod
    def get_transform(self) -> T.Compose:
        """Returns the specific transform for this model (resize, norm)."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: (B, C, H, W) tensor.
        Returns:
            Dict containing 'features' (B, D) normalized.
        """
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        pass
