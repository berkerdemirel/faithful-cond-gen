from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from faithful_cond_gen.model.generator import GeneratorWrapper
from torch.optim import AdamW


@dataclass
class GeneratorPLConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0


class GeneratorPL(pl.LightningModule):
    def __init__(
        self, generator: GeneratorWrapper, cfg: Optional[GeneratorPLConfig] = None
    ):
        super().__init__()
        self.generator = generator
        self.cfg = cfg or GeneratorPLConfig()

        self.save_hyperparameters(
            {"lr": self.cfg.lr, "weight_decay": self.cfg.weight_decay},
            ignore=["generator", "cfg"],
        )

    def configure_optimizers(self):
        params = [p for p in self.generator.parameters() if p.requires_grad]
        return AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    @staticmethod
    def linear_interpolant(
        x0: torch.Tensor,  # (B,C,h,w)
        t: torch.Tensor,  # (B,) float
        eps: torch.Tensor,  # (B,C,h,w)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_t = (1-t)x0 + t eps; v_tgt = d/dt x_t = -x0 + eps."""
        b = x0.shape[0]
        t_b = t.view(b, 1, 1, 1)
        x_t = (1.0 - t_b) * x0 + t_b * eps
        v_tgt = -x0 + eps
        return x_t, v_tgt

    def _unpack_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        images, conditioning = batch

        # Agnostic Unpacking:
        # We assume the dataset __getitem__ inserted keys in the correct order.
        # e.g. RxRx1: {'cell_type': ..., 'sirna': ...} -> [cell_type, sirna]
        # e.g. CelebA: {'Male': ..., 'Smiling': ...} -> [Male, Smiling]
        cond_dict = conditioning.get("cond")
        if cond_dict is None:
            raise ValueError("Batch missing 'cond' dict in conditioning")

        # dict.values() preserves insertion order in Python 3.7+
        cond_tensors = list(cond_dict.values())

        # Stack (B,) tensors into (B, K)
        cond_ids = torch.stack(cond_tensors, dim=1)

        return images, cond_ids

    def training_step(self, batch, batch_idx: int):
        images, cond_ids = self._unpack_batch(batch)

        # --- encode to latents ---
        with torch.no_grad():
            x0 = self.generator.encode(images)  # (B,4,h,w) if VAE frozen

        b = x0.shape[0]

        # --- forward/noising process ---
        t = torch.rand(b, device=x0.device, dtype=x0.dtype)
        eps = torch.randn_like(x0)
        x_t, v_tgt = self.linear_interpolant(x0, t, eps)

        # --- velocity prediction ---
        v_hat = self.generator.velocity_prediction(x_t=x_t, t=t, cond_ids=cond_ids)

        # --- loss ---
        loss = F.mse_loss(v_hat, v_tgt)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int):
        images, cond_ids = self._unpack_batch(batch)

        x0 = self.generator.encode(images)
        b = x0.shape[0]

        t = torch.rand(b, device=x0.device, dtype=x0.dtype)
        eps = torch.randn_like(x0)
        x_t, v_tgt = self.linear_interpolant(x0, t, eps)

        v_hat = self.generator.velocity_prediction(x_t=x_t, t=t, cond_ids=cond_ids)
        loss = F.mse_loss(v_hat, v_tgt)

        self.log("val/loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx: int):
        images, cond_ids = self._unpack_batch(batch)

        x0 = self.generator.encode(images)
        b = x0.shape[0]

        t = torch.rand(b, device=x0.device, dtype=x0.dtype)
        eps = torch.randn_like(x0)
        x_t, v_tgt = self.linear_interpolant(x0, t, eps)

        v_hat = self.generator.velocity_prediction(x_t=x_t, t=t, cond_ids=cond_ids)
        loss = F.mse_loss(v_hat, v_tgt)

        self.log("test/loss", loss, prog_bar=True)
        return loss
