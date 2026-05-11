"""Residual MLP mapper from SiT hidden states to SigLIP space."""

import torch
import torch.nn as nn


class ResidualAlignmentMapper(nn.Module):
    """Residual MLP: in_dim -> out_dim with dimension-adapting skip.

    Architecture:
        skip: Linear(in_dim -> out_dim)
        mlp:  Linear(in_dim -> hidden_dim) + SiLU + Linear(hidden_dim -> hidden_dim) + SiLU + Linear(hidden_dim -> out_dim)
        out = skip(x) + mlp(x)
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 1152, hidden_dim: int = 2048):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.skip = nn.Linear(in_dim, out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map hidden states to target space.

        Args:
            x: (B, in_dim) mean-pooled SiT hidden states

        Returns:
            (B, out_dim) mapped features in SigLIP-like space
        """
        return self.skip(x) + self.mlp(x)
