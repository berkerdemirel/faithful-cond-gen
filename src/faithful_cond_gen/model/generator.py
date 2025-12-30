# src/faithful_cond_gen/model/generator.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from faithful_cond_gen.model.sit_backbone import SiT_models  # you’ll put SiT code here

# -----------------------------------------------------------------------------
# VAE backbone
# -----------------------------------------------------------------------------


class VAEBackbone(nn.Module):
    """VAE backbone for encoding/decoding images.

    Thin wrapper around a Stable-Diffusion-style AutoencoderKL.
    """

    def __init__(
        self,
        vae_model_name: str = "stabilityai/sd-vae-ft-mse",
        in_channels: int = 3,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.freeze = freeze
        self.in_channels = in_channels

        # Load pre-trained VAE
        self.vae = AutoencoderKL.from_pretrained(vae_model_name)

        if freeze:
            for p in self.vae.parameters():
                p.requires_grad = False
            self.vae.eval()

        # SD VAE conventions
        self.base_latent_channels = 4
        self.downsampling_factor = 8  # spatial H,W -> H/8, W/8

        # Determine effective latent channels for the diffusion model
        if self.in_channels == 3:
            self.out_channels = 4
        elif self.in_channels == 6:
            self.out_channels = 24  # 6 channels * 4 latent dims each

        self.register_buffer(
            "latents_scale",
            torch.tensor([0.18215, 0.18215, 0.18215, 0.18215]).view(1, 4, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_bias",
            torch.zeros(1, 4, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def sample_posterior(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from posterior and apply latent scaling/bias."""
        z = mean + std * torch.randn_like(mean)
        z = z * self.latents_scale + self.latents_bias
        return z

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space using sample_posterior."""
        # Handle Channel Folding for RxRx1 (6 channels)
        if self.in_channels == 6:
            b, c, h, w = images.shape
            # Fold channels into batch: (B*6, 1, H, W)
            images = images.view(b * c, 1, h, w)
            # Expand to RGB for VAE: (B*6, 3, H, W)
            images = images.repeat(1, 3, 1, 1)

        images = 2.0 * images - 1.0  # map [0, 1] -> [-1, 1]
        posterior = self.vae.encode(images).latent_dist
        latents = self.sample_posterior(
            posterior.mean,
            posterior.std,
        )
        # Handle Unfolding for RxRx1
        if self.in_channels == 6:
            # (B*6, 4, h, w) -> (B, 6, 4, h, w) -> (B, 24, h, w)
            _, lc, lh, lw = latents.shape
            latents = latents.reshape(b, c, lc, lh, lw)
            latents = latents.reshape(b, c * lc, lh, lw)
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images in [0,1]."""
        # 1. Handle Folding for RxRx1
        if self.in_channels == 6:
            b, total_c, lh, lw = latents.shape
            # (B, 24, h, w) -> (B, 6, 4, h, w) -> (B*6, 4, h, w)
            latents = latents.reshape(b, 6, 4, lh, lw)
            latents = latents.reshape(b * 6, 4, lh, lw)
        latents = (latents - self.latents_bias) / self.latents_scale
        images = self.vae.decode(latents).sample
        images = (images + 1.0) / 2.0
        images = images.clamp(0.0, 1.0)

        # 5. Handle Unfolding and RGB->Gray reduction for RxRx1
        if self.in_channels == 6:
            # VAE output is (B*6, 3, H, W), we need (B, 6, H, W)
            # Mean pool RGB channels: (B*6, 3, H, W) -> (B*6, 1, H, W)
            images = images.mean(dim=1, keepdim=True)
            # Reshape: (B*6, 1, H, W) -> (B, 6, H, W)
            images = images.view(b, 6, images.shape[-2], images.shape[-1])
        return images

    def get_latent_size(self, image_size: int) -> int:
        return image_size // self.downsampling_factor


# -----------------------------------------------------------------------------
# Generator config
# -----------------------------------------------------------------------------


@dataclass
class GeneratorConfig:
    """Config for LatentGenerator (VAE + SiT)."""

    image_size: int = 224
    in_channels: int = 3
    # VAE
    vae_model_name: str = "stabilityai/sd-vae-ft-mse"
    vae_freeze: bool = True

    # SiT backbone
    sit_arch: str = "SiT-B/4"  # must be a key in SiT_models

    # List of class counts per factor (e.g. RxRx1=[4, 1138], CelebA=[2,2,2,2])
    attr_num_classes: List[int] = field(
        default_factory=lambda: [1000]
    )  # number of discrete condition ids (dataset specific)

    # Block specific kwargs for SiT
    qk_norm: bool = False
    fused_attn: bool = True

    class_dropout_prob: float = 0.1

    # diffusion / training settings (interpolant etc.)
    path_type: str = "linear"  # "linear" or "cosine" later, but currently linear [0,1]
    # you can extend with EDM-style sigma schedule later

    # optional checkpoint
    checkpoint: Optional[str] = None


# -----------------------------------------------------------------------------
# Generator wrapper: VAE + SiT (velocity net in latent space)
# -----------------------------------------------------------------------------


class GeneratorWrapper(nn.Module):
    """Latent-space conditional generator (VAE + SiT).

    - Images are encoded to latents via VAEBackbone.
    - Diffusion backbone (SiT) predicts *velocity* in latent space.
    - Conditioning is via integer class ids (B,) -- dataset defines the mapping
      from its own multi-attribute representation → class ids.

    This keeps the generator agnostic to:
      * how many attributes each dataset has, and
      * how those attributes are represented.
    """

    def __init__(self, cfg: GeneratorConfig):
        super().__init__()
        self.cfg = cfg

        # --- VAE ---
        self.vae = VAEBackbone(
            vae_model_name=cfg.vae_model_name,
            in_channels=cfg.in_channels,
            freeze=cfg.vae_freeze,
        )

        latent_size = self.vae.get_latent_size(cfg.image_size)

        # --- SiT diffusion backbone (velocity net in latent space) ---
        if cfg.sit_arch not in SiT_models:
            raise ValueError(
                f"Unknown SiT arch '{cfg.sit_arch}'. "
                f"Available: {list(SiT_models.keys())}"
            )
        sit_in_channels = self.vae.out_channels
        self.diffusion_backbone = SiT_models[cfg.sit_arch](
            path_type="linear",  # not too important for our wrapper
            input_size=latent_size,  # H_latent == W_latent
            in_channels=sit_in_channels,
            attr_num_classes=cfg.attr_num_classes,
            class_dropout_prob=cfg.class_dropout_prob,
            qk_norm=cfg.qk_norm,
            fused_attn=cfg.fused_attn,
        )

        # if you later add alignment / projector loss, you’ll have extra flags here

        if cfg.checkpoint is not None:
            self._load_checkpoint(cfg.checkpoint)

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _load_checkpoint(self, ckpt_path: str) -> None:
        state = torch.load(ckpt_path, map_location="cpu")
        sd = state.get("state_dict", state)
        sd = {k.replace("generator.", ""): v for k, v in sd.items()}
        self.load_state_dict(sd, strict=False)

    # Public encode/decode helpers -------------------------------------

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(images)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents)

    def velocity_prediction(
        self,
        x_t: torch.Tensor,  # (B, C, h, w) noisy latents
        t: torch.Tensor,  # (B,) float in [0,1]
        cond_ids: torch.Tensor,  # (B,K) long OR (B,) long
    ) -> torch.Tensor:
        """Predict latent velocity v(x_t, t, cond). No noising, no loss, no VAE."""
        if cond_ids.dim() == 1:
            cond_ids = cond_ids.unsqueeze(1)  # (B,) -> (B,1)

        v_hat, _ = self.diffusion_backbone(
            x_t,
            t,
            attr_ids=cond_ids.to(device=x_t.device, dtype=torch.long),
        )

        # If your SiT ever returns extra channels, keep the first C
        if v_hat.shape[1] > x_t.shape[1]:
            v_hat = v_hat[:, : x_t.shape[1]]

        return v_hat

    @staticmethod
    def get_score_from_velocity(
        v: torch.Tensor,  # Predicted velocity (B, C, H, W)
        x: torch.Tensor,  # Current state (B, C, H, W)
        t: torch.Tensor,  # Timestep (B,)
    ) -> torch.Tensor:
        """Convert velocity to score for linear interpolant.

        For linear path x_t = (1-t)·x_0 + t·ε:
        - alpha(t) = 1 - t, sigma(t) = t
        - d_alpha/dt = -1, d_sigma/dt = 1

        Score ∇_x log p(x_t|c) = (-(1-t)·v - x) / t

        Args:
            v: Predicted velocity from model
            x: Current noisy latent state
            t: Timestep in [0, 1]

        Returns:
            Score (gradient of log probability)
        """
        b = t.shape[0]
        t_b = t.view(b, 1, 1, 1)

        # Linear interpolant coefficients
        alpha_t = 1.0 - t_b
        sigma_t = t_b
        d_alpha_t = -1.0
        d_sigma_t = 1.0

        # REPA formula: score = (reverse_alpha_ratio * v - x) / var
        # where reverse_alpha_ratio = alpha_t / d_alpha_t = -(1-t)
        # and var = t² - (-(1-t)) * t = t
        reverse_alpha_ratio = alpha_t / d_alpha_t  # -(1-t)
        var = sigma_t.square() - reverse_alpha_ratio * d_sigma_t * sigma_t  # t
        score = (reverse_alpha_ratio * v - x) / (var + 1e-8)

        return score

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def basic_sample(
        self,
        cond_ids: torch.Tensor,  # (B,) long condition indices
        num_inference_steps: int = 50,
        eta: float = 0.0,  # 0: deterministic (ODE-like), >0: Euler-Maruyama
    ) -> torch.Tensor:
        """Basic Euler / Euler-Maruyama sampler in latent space.

        Uses same linear interpolant as training:
            x_t = (1-t) x0 + t eps, t in [0,1].
        At t=1: x_1 ~ N(0,I). Integrates backward to t=0.
        Returns decoded images in [0,1].

        Note: This is the old simple sampler. Use sample() for REPA-style sampling.
        """
        device = cond_ids.device
        model_dtype = next(self.parameters()).dtype
        b = cond_ids.shape[0]

        def alpha_sigma(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            a = 1.0 - t
            s = t
            return a, s

        # time grid from 1 → 0
        t_grid = torch.linspace(
            1.0,
            0.0,
            num_inference_steps,
            device=device,
            dtype=model_dtype,
        )
        _, sigmas = alpha_sigma(t_grid)
        sigma2 = sigmas.square()

        latent_size = self.vae.get_latent_size(self.cfg.image_size)

        # CHANGED: Initialize x with self.vae.out_channels (4 or 24)
        x = torch.randn(
            b,
            self.vae.out_channels,
            latent_size,
            latent_size,
            device=device,
            dtype=model_dtype,
        )

        cond_ids = cond_ids.to(device=device, dtype=torch.long)

        for k in range(num_inference_steps - 1):
            t_cur = t_grid[k]
            t_nxt = t_grid[k + 1]
            dt = t_nxt - t_cur  # negative

            t_in = t_cur.expand(b)

            # predict velocity
            v, _ = self.diffusion_backbone(x, t_in, cond_ids)
            if v.shape[1] > x.shape[1]:
                v = v[:, : x.shape[1]]

            # Euler ODE step
            x = x + dt * v

            # Euler-Maruyama noise
            if eta > 0.0:
                var_inc = sigma2[k + 1] - sigma2[k]
                noise_std = eta * torch.sqrt(var_inc.abs().to(model_dtype))
                if float(noise_std) > 0:
                    x = x + noise_std * torch.randn_like(x)

        return self.decode(x)

    @torch.no_grad()
    def sample(
        self,
        cond_ids: torch.Tensor,
        num_inference_steps: int = 250,
        t_cutoff: float = 0.04,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """REPA-style two-stage sampling (default sampler).

        Stage 1 (SDE): t=1.0 → t=0.04 with score-based drift correction
        Stage 2 (ODE): t=0.04 → t=0.0 deterministic refinement

        Args:
            cond_ids: (B,) or (B, K) conditioning indices
            num_inference_steps: Total timesteps (REPA default: 250)
            t_cutoff: Transition point between SDE and ODE (default: 0.04)
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance)

        Returns:
            Decoded images in [0, 1]
        """
        device = cond_ids.device
        model_dtype = next(self.parameters()).dtype
        b = cond_ids.shape[0]

        latent_size = self.vae.get_latent_size(self.cfg.image_size)

        # Initialize from noise at t=1
        x = torch.randn(
            b,
            self.vae.out_channels,
            latent_size,
            latent_size,
            device=device,
            dtype=model_dtype,
        )

        cond_ids = cond_ids.to(device=device, dtype=torch.long)

        # --- Timestep Grid (REPA-style) ---
        # Create grid from 1.0 → t_cutoff, then append 0.0
        t_steps = torch.linspace(
            1.0, t_cutoff, num_inference_steps, device=device, dtype=model_dtype
        )
        t_steps = torch.cat(
            [t_steps, torch.tensor([0.0], device=device, dtype=model_dtype)]
        )

        # --- STAGE 1: SDE (t=1.0 → t_cutoff) ---
        # Iterate over all but the last transition (which is t_cutoff → 0.0)
        for k in range(len(t_steps) - 2):
            t_cur = t_steps[k]
            t_next = t_steps[k + 1]
            dt = t_next - t_cur  # Negative

            t_in = t_cur.expand(b)

            # Predict velocity
            v, _ = self.diffusion_backbone(x, t_in, cond_ids)
            if v.shape[1] > x.shape[1]:
                v = v[:, : x.shape[1]]

            # Convert velocity to score
            score = self.get_score_from_velocity(v, x, t_in)

            # Diffusion coefficient for linear interpolant: g²(t) = 2t
            diffusion_coeff = 2.0 * t_cur

            # SDE drift with score correction
            drift = v - 0.5 * diffusion_coeff * score

            # Euler-Maruyama step
            x = x + drift * dt

            # Stochastic noise term: sqrt(g²(t)) * eps * sqrt(|dt|)
            noise_std = torch.sqrt(diffusion_coeff * (-dt).abs())
            if noise_std > 1e-8:
                x = x + noise_std * torch.randn_like(x)

        # --- STAGE 2: ODE (t_cutoff → 0.0) ---
        # REPA: Single deterministic Euler step with score-corrected drift (NO noise)
        t_final = torch.full((b,), t_cutoff, device=device, dtype=model_dtype)

        # Predict velocity
        v_final, _ = self.diffusion_backbone(x, t_final, cond_ids)
        if v_final.shape[1] > x.shape[1]:
            v_final = v_final[:, : x.shape[1]]

        # Convert to score
        score_final = self.get_score_from_velocity(v_final, x, t_final)

        # Score-corrected drift (same as SDE, but no noise)
        diffusion_coeff_final = 2.0 * t_cutoff
        drift_final = v_final - 0.5 * diffusion_coeff_final * score_final

        # Deterministic Euler step: x_0 = x_t + (-t) * drift
        dt_final = 0.0 - t_cutoff  # = -0.04
        x = x + dt_final * drift_final

        return self.decode(x)
