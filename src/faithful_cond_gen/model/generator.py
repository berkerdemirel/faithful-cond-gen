# src/faithful_cond_gen/model/generator.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from faithful_cond_gen.model.sit_backbone import SiT_models  # you'll put SiT code here


def hash_condition(cond_ids: torch.Tensor, cond_idx: int = None) -> str:
    """Create canonical hash from condition IDs.

    Args:
        cond_ids: (B, K) or (K,) tensor of condition IDs
        cond_idx: If cond_ids is 2D, which sample to hash. If None, hash the whole thing.

    Returns:
        String hash suitable for dictionary lookup
    """
    if cond_ids.dim() == 1:
        # Single sample
        return str(tuple(cond_ids.tolist()))
    elif cond_idx is not None:
        # Extract specific sample
        return str(tuple(cond_ids[cond_idx].tolist()))
    else:
        # Hash all samples (rare, but handle it)
        return str([tuple(row.tolist()) for row in cond_ids])


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

    # Conditioning mode
    use_per_attr_modulation: bool = False  # If True, use per-attribute adaLN (ablation)

    # diffusion / training settings (interpolant etc.)
    path_type: str = "linear"  # "linear" or "cosine" later, but currently linear [0,1]
    # you can extend with EDM-style sigma schedule later

    # REPA (Representation Alignment) loss settings
    use_repa: bool = False  # Enable REPA projection loss
    repa_encoder: str = "dinov2-vit-b"  # Encoder for REPA (dinov2-vit-{s,b,l})
    repa_proj_coeff: float = 0.5  # Weight for projection loss
    repa_encoder_depth: int = 8  # SiT layer to extract features from
    repa_projector_dim: int = 2048  # Intermediate dimension for projector MLP

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

        # Determine encoder embed dim for REPA projectors
        repa_z_dims = []
        if cfg.use_repa:
            # Map encoder name to embed dim
            # Must match encoders supported in model/repa_encoder.py
            encoder_dim_map = {
                # HuggingFace encoders
                "dinov3": 1024,  # ViT-L/16
                "dinov2": 1024,  # ViT-L/14
                # Specialized
                "openphenom": 384 * 6,  # 384 per channel, 6 channels
                "clip": 1024,  # ViT-B/16
                "siglip": 1152,  # ViT-L/14
            }
            enc_name = cfg.repa_encoder.lower()
            # Find matching encoder (prefix match for dinov2 variants)
            embed_dim = None
            for key, dim in encoder_dim_map.items():
                if enc_name == key or enc_name.startswith(key):
                    embed_dim = dim
                    break
            if embed_dim is None:
                raise ValueError(
                    f"Unknown REPA encoder: {cfg.repa_encoder}. "
                    f"Supported: {list(encoder_dim_map.keys())}"
                )
            repa_z_dims = [embed_dim]

        self.diffusion_backbone = SiT_models[cfg.sit_arch](
            path_type="linear",  # not too important for our wrapper
            input_size=latent_size,  # H_latent == W_latent
            in_channels=sit_in_channels,
            attr_num_classes=cfg.attr_num_classes,
            class_dropout_prob=cfg.class_dropout_prob,
            use_per_attr_modulation=cfg.use_per_attr_modulation,
            qk_norm=cfg.qk_norm,
            fused_attn=cfg.fused_attn,
            # REPA parameters
            use_repa=cfg.use_repa,
            encoder_depth=cfg.repa_encoder_depth,
            z_dims=repa_z_dims,
            projector_dim=cfg.repa_projector_dim,
        )

        # if you later add alignment / projector loss, you'll have extra flags here

        # Adaptive CFG: store condition frequency statistics
        self.condition_stats = {}  # condition_hash -> count
        self.adaptive_cfg_config = {
            "min_cfg": 1.0,  # CFG scale for common conditions
            "max_cfg": 3.0,  # CFG scale for rare conditions
            "threshold_common": 100,  # Conditions with >threshold_common samples
            "threshold_rare": 10,  # Conditions with <threshold_rare samples
        }

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
        force_drop_ids: torch.Tensor = None,  # (B,) 0/1 for CFG
        return_projected: bool = False,  # Return projected features for REPA
    ) -> torch.Tensor:
        """Predict latent velocity v(x_t, t, cond). No noising, no loss, no VAE.

        Args:
            x_t: Noisy latent state
            t: Timestep
            cond_ids: Conditioning IDs
            force_drop_ids: Optional force dropout for CFG (0=conditional, 1=unconditional)
            return_projected: If True, also return projected features (for REPA loss)

        Returns:
            v_hat: (B, C, h, w) predicted velocity
            zs_tilde: List of projected features (only if return_projected=True and REPA enabled)
        """
        if cond_ids.dim() == 1:
            cond_ids = cond_ids.unsqueeze(1)  # (B,) -> (B,1)

        v_hat, zs_tilde = self.diffusion_backbone(
            x_t,
            t,
            attr_ids=cond_ids.to(device=x_t.device, dtype=torch.long),
            force_drop_ids=force_drop_ids,
        )

        # If your SiT ever returns extra channels, keep the first C
        if v_hat.shape[1] > x_t.shape[1]:
            v_hat = v_hat[:, : x_t.shape[1]]

        if return_projected:
            return v_hat, zs_tilde
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
    # Adaptive Classifier-Free Guidance
    # ------------------------------------------------------------------

    def load_condition_stats(self, stats_dict: Dict[str, int]):
        """Load condition frequency statistics for adaptive CFG.

        Args:
            stats_dict: Dictionary mapping condition hash to count
                       Keys should be string hashes from hash_condition()
                       Example: {"(0, 0)": 150, "(0, 1)": 50, ...}
        """
        self.condition_stats = stats_dict

        if len(self.condition_stats) > 0:
            print(
                f"Loaded condition stats: {len(self.condition_stats)} unique conditions, "
                f"range [{min(self.condition_stats.values())}, {max(self.condition_stats.values())}]"
            )
        else:
            print("⚠️ Warning: Loaded empty condition stats")

    def compute_condition_stats_from_metadata(self, metadata):
        """Helper to compute condition stats from dataset metadata.

        Args:
            metadata: Can be:
                     - pandas DataFrame with condition columns
                     - dict with condition arrays
                     - list of condition dicts

        Returns:
            Dictionary suitable for load_condition_stats()
        """
        from collections import Counter

        import pandas as pd

        if isinstance(metadata, pd.DataFrame):
            # Get condition columns (exclude metadata columns)
            exclude_cols = [
                "count",
                "comp_category",
                "condition_hash",
                "numpy_path",
                "parquet_path",
            ]
            cond_cols = [c for c in metadata.columns if c not in exclude_cols]

            # Create hashes for each row
            hashes = []
            for _, row in metadata.iterrows():
                # Create condition tuple
                cond_tuple = tuple(row[c] for c in sorted(cond_cols))
                hashes.append(str(cond_tuple))

            return dict(Counter(hashes))

        elif isinstance(metadata, dict):
            # Dict of arrays - zip them together
            keys = sorted(metadata.keys())
            n_samples = len(metadata[keys[0]])

            hashes = []
            for i in range(n_samples):
                cond_tuple = tuple(metadata[k][i] for k in keys)
                hashes.append(str(cond_tuple))

            return dict(Counter(hashes))

        elif isinstance(metadata, list):
            # List of dicts
            hashes = []
            for cond_dict in metadata:
                keys = sorted(cond_dict.keys())
                cond_tuple = tuple(cond_dict[k] for k in keys)
                hashes.append(str(cond_tuple))

            return dict(Counter(hashes))

        else:
            raise ValueError(f"Unsupported metadata type: {type(metadata)}")

    def compute_adaptive_cfg_scale(self, cond_ids: torch.Tensor) -> torch.Tensor:
        """Compute per-sample CFG scale based on condition rarity.

        Heuristic: Rare conditions get higher guidance to improve faithfulness.

        Args:
            cond_ids: (B,) or (B, K) conditioning indices

        Returns:
            (B,) tensor of cfg_scale values
        """
        if not self.condition_stats:
            # No stats loaded, return default
            return torch.ones(cond_ids.shape[0], device=cond_ids.device)

        cfg_scales = []

        for i in range(cond_ids.shape[0]):
            # Use canonical hash function
            cond_hash = hash_condition(cond_ids, cond_idx=i)
            count = self.condition_stats.get(cond_hash, 0)

            # Adaptive heuristic:
            # - Common conditions (>threshold_common): min_cfg
            # - Rare conditions (<threshold_rare): max_cfg
            # - In between: linear interpolation
            min_cfg = self.adaptive_cfg_config["min_cfg"]
            max_cfg = self.adaptive_cfg_config["max_cfg"]
            thresh_common = self.adaptive_cfg_config["threshold_common"]
            thresh_rare = self.adaptive_cfg_config["threshold_rare"]

            if count >= thresh_common:
                cfg = min_cfg
            elif count <= thresh_rare:
                cfg = max_cfg
            else:
                # Linear interpolation
                alpha = (count - thresh_rare) / (thresh_common - thresh_rare)
                cfg = max_cfg + alpha * (min_cfg - max_cfg)

            cfg_scales.append(cfg)

        return torch.tensor(cfg_scales, device=cond_ids.device)

    # ------------------------------------------------------------------
    # Classifier-Free Guidance Helper
    # ------------------------------------------------------------------

    def apply_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_ids: torch.Tensor,
        cfg_scale: torch.Tensor,
        return_features: bool = False,
    ):
        """Apply classifier-free guidance to velocity prediction.

        Args:
            x: Current latent state (B, C, H, W)
            t: Timestep (B,)
            cond_ids: Conditioning IDs (B, K)
            cfg_scale: Guidance scale (B,) or scalar
            return_features: If True, also return projected features from conditional pass

        Returns:
            Guided velocity (B, C, H, W)
            If return_features=True: (velocity, zs_tilde) tuple
        """
        b = x.shape[0]

        # Convert scalar to tensor if needed
        if isinstance(cfg_scale, (int, float)):
            cfg_scale = torch.full((b,), cfg_scale, device=x.device, dtype=x.dtype)
        else:
            cfg_scale = cfg_scale.to(device=x.device, dtype=x.dtype)

        # Check if we need CFG
        needs_cfg = (cfg_scale != 1.0).any()

        if not needs_cfg:
            # No guidance needed, just predict conditional
            v_cond, zs_tilde = self.diffusion_backbone(x, t, cond_ids)
            if v_cond.shape[1] > x.shape[1]:
                v_cond = v_cond[:, : x.shape[1]]
            if return_features:
                return v_cond, zs_tilde
            return v_cond

        # Predict conditional velocity
        v_cond, zs_tilde = self.diffusion_backbone(x, t, cond_ids)
        if v_cond.shape[1] > x.shape[1]:
            v_cond = v_cond[:, : x.shape[1]]

        # Predict unconditional velocity (force all conditions to be dropped)
        force_drop = torch.ones(b, device=x.device, dtype=torch.long)
        v_uncond, _ = self.diffusion_backbone(x, t, cond_ids, force_drop_ids=force_drop)
        if v_uncond.shape[1] > x.shape[1]:
            v_uncond = v_uncond[:, : x.shape[1]]

        # Apply classifier-free guidance
        # v = v_uncond + cfg_scale * (v_cond - v_uncond)
        cfg_scale_expanded = cfg_scale.view(b, 1, 1, 1)  # (B, 1, 1, 1) for broadcasting
        v_guided = v_uncond + cfg_scale_expanded * (v_cond - v_uncond)

        if return_features:
            return v_guided, zs_tilde
        return v_guided

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def basic_sample(
        self,
        cond_ids: torch.Tensor,  # (B,) long condition indices
        num_inference_steps: int = 50,
        eta: float = 0.0,  # 0: deterministic (ODE-like), >0: Euler-Maruyama
        cfg_scale: float = 1.0,  # Classifier-free guidance scale
    ) -> torch.Tensor:
        """Basic Euler / Euler-Maruyama sampler in latent space.

        Uses same linear interpolant as training:
            x_t = (1-t) x0 + t eps, t in [0,1].
        At t=1: x_1 ~ N(0,I). Integrates backward to t=0.
        Returns decoded images in [0,1].

        Args:
            cond_ids: Conditioning IDs (B,) or (B, K)
            num_inference_steps: Number of sampling steps
            eta: Noise scale (0=deterministic ODE, >0=SDE)
            cfg_scale: Classifier-free guidance scale (1.0=no guidance)

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

            # Predict velocity with classifier-free guidance
            v = self.apply_cfg(x, t_in, cond_ids, cfg_scale)

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
        adaptive_cfg: bool = False,
        return_aligned_features: bool = False,
        feature_capture_idx: Optional[Union[int, List[int], str]] = None,
    ):
        """REPA-style two-stage sampling (default sampler).

        Stage 1 (SDE): t=1.0 → t=0.04 with score-based drift correction
        Stage 2 (ODE): t=0.04 → t=0.0 deterministic refinement

        Args:
            cond_ids: (B,) or (B, K) conditioning indices
            num_inference_steps: Total timesteps (REPA default: 250)
            t_cutoff: Transition point between SDE and ODE (default: 0.04)
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
                      If adaptive_cfg=True, this is ignored and per-sample scales are computed
            adaptive_cfg: If True, use condition-adaptive CFG based on rarity
            return_aligned_features: If True, also return aligned features from REPA projectors
            feature_capture_idx: Step index/indices at which to capture features (only used if return_aligned_features=True).
                              None (default): capture at final Stage 2 call (index num_inference_steps-1)
                              int: capture at that Stage 1 index k (0 <= k <= num_inference_steps-2), or -1 for final Stage 2
                              List[int]: capture at each listed index; -1 allowed for final Stage 2
                              "all": capture at every Stage 1 step (0 to num_inference_steps-2) + final Stage 2

        Returns:
            Decoded images in [0, 1]
            If return_aligned_features=True: (images, aligned_features) tuple where:
                - Single capture (int/None): aligned_features is List[Tensor] or None (unpooled patch features)
                - Multiple captures (list/"all"): aligned_features is List[(k_idx, t_val, Tensor)] where:
                  - k_idx (int): step index (0 to num_inference_steps-2 for Stage 1, num_inference_steps-1 for Stage 2)
                  - t_val (float): actual timestep value at that index
                  - Tensor: pooled features (B, D) on CPU for memory efficiency
        """
        # Compute adaptive CFG scales if requested
        if adaptive_cfg:
            cfg_scale_tensor = self.compute_adaptive_cfg_scale(cond_ids)
        else:
            cfg_scale_tensor = torch.full(
                (cond_ids.shape[0],), cfg_scale, device=cond_ids.device
            )
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

        # Feature capture setup
        capture_mode = None  # "single", "multiple", or "all"
        capture_list = []  # List of (k_idx, t_val, features) tuples for multiple capture
        captured_features = None  # For single capture mode
        capture_indices_set = None
        final_idx = num_inference_steps - 1  # Index for Stage 2 capture

        if return_aligned_features:
            if feature_capture_idx is None:
                # Default: capture at final Stage 2 call (single mode)
                capture_mode = "single"
                capture_indices_set = {final_idx}
            elif isinstance(feature_capture_idx, str) and feature_capture_idx.lower() == "all":
                # Capture at every Stage 1 step + final Stage 2
                capture_mode = "all"
                # Will capture at indices 0 to num_inference_steps-2 (Stage 1) + final_idx (Stage 2)
            elif isinstance(feature_capture_idx, (list, tuple)):
                # Capture at specific indices
                capture_mode = "multiple"
                # Validate indices
                for idx in feature_capture_idx:
                    if idx == -1:
                        continue  # -1 is allowed (means final Stage 2)
                    if not isinstance(idx, int) or idx < 0 or idx > num_inference_steps - 2:
                        raise ValueError(
                            f"Invalid capture index {idx}. Must be in [0, {num_inference_steps-2}] "
                            f"for Stage 1 steps, or -1 for final Stage 2 call."
                        )
                # Replace -1 with actual final index
                capture_indices_set = {final_idx if idx == -1 else idx for idx in feature_capture_idx}
            elif isinstance(feature_capture_idx, int):
                # Capture once at specified index (single mode)
                capture_mode = "single"
                if feature_capture_idx == -1:
                    capture_indices_set = {final_idx}
                elif 0 <= feature_capture_idx <= num_inference_steps - 2:
                    capture_indices_set = {feature_capture_idx}
                else:
                    raise ValueError(
                        f"Invalid capture index {feature_capture_idx}. Must be in [0, {num_inference_steps-2}] "
                        f"for Stage 1 steps, or -1 for final Stage 2 call."
                    )
            else:
                raise ValueError(
                    f"Invalid feature_capture_idx: {feature_capture_idx}. "
                    f"Must be None, int, list of ints, or 'all'"
                )

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

            # Determine if we should capture features at this step index
            should_capture = False
            if return_aligned_features:
                if capture_mode == "all":
                    should_capture = True
                elif capture_mode == "multiple" and k in capture_indices_set:
                    should_capture = True
                elif capture_mode == "single" and k in capture_indices_set and captured_features is None:
                    should_capture = True

            if should_capture:
                v, zs_tilde = self.apply_cfg(
                    x, t_in, cond_ids, cfg_scale_tensor, return_features=True
                )

                # Validate REPA features are available
                if zs_tilde is None or len(zs_tilde) == 0:
                    raise ValueError(
                        f"REPA features not available at step {k}. "
                        f"Ensure model has use_repa=True and REPA projectors are enabled."
                    )

                # Store features based on capture mode
                if capture_mode in ["all", "multiple"]:
                    # Pool over patches and move to CPU for memory efficiency
                    # Take first projector: (B, num_patches, D) -> (B, D)
                    pooled = zs_tilde[0].mean(dim=1).cpu()
                    capture_list.append((k, t_cur.item(), pooled))
                else:  # single mode
                    captured_features = zs_tilde
            else:
                v = self.apply_cfg(x, t_in, cond_ids, cfg_scale_tensor)

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
            if noise_std.item() > 1e-8:
                x = x + noise_std * torch.randn_like(x)

        # --- STAGE 2: ODE (t_cutoff → 0.0) ---
        # REPA: Single deterministic Euler step with score-corrected drift (NO noise)
        t_final = torch.full((b,), t_cutoff, device=device, dtype=model_dtype)

        # Capture features at final Stage 2 call based on mode
        should_capture_final = False
        if return_aligned_features:
            if capture_mode == "all":
                should_capture_final = True
            elif capture_mode in ["multiple", "single"] and final_idx in capture_indices_set:
                should_capture_final = True

        if should_capture_final:
            v_final, zs_tilde = self.apply_cfg(
                x, t_final, cond_ids, cfg_scale_tensor, return_features=True
            )

            # Validate REPA features are available
            if zs_tilde is None or len(zs_tilde) == 0:
                raise ValueError(
                    f"REPA features not available at final step (index {final_idx}). "
                    f"Ensure model has use_repa=True and REPA projectors are enabled."
                )

            # Store based on mode
            if capture_mode in ["all", "multiple"]:
                pooled = zs_tilde[0].mean(dim=1).cpu()
                capture_list.append((final_idx, t_cutoff, pooled))
            else:  # single mode
                captured_features = zs_tilde
        else:
            v_final = self.apply_cfg(x, t_final, cond_ids, cfg_scale_tensor)

        # Convert to score
        score_final = self.get_score_from_velocity(v_final, x, t_final)

        # Score-corrected drift (same as SDE, but no noise)
        diffusion_coeff_final = 2.0 * t_cutoff
        drift_final = v_final - 0.5 * diffusion_coeff_final * score_final

        # Deterministic Euler step: x_0 = x_t + (-t) * drift
        dt_final = 0.0 - t_cutoff  # = -0.04
        x = x + dt_final * drift_final

        images = self.decode(x)

        if return_aligned_features:
            # Return format depends on capture mode
            if capture_mode in ["all", "multiple"]:
                # Validation for "all" mode
                if capture_mode == "all" and len(capture_list) != num_inference_steps:
                    raise ValueError(
                        f"Expected {num_inference_steps} captures in 'all' mode, "
                        f"got {len(capture_list)}. This indicates a bug in feature capture logic."
                    )
                return images, capture_list if capture_list else None
            else:
                return images, captured_features
        return images
