# src/faithful_cond_gen/model/repa_encoder.py
"""
REPA Encoder for training-time representation alignment.

Unlike eval encoders (which return pooled features), REPA encoders return
patch tokens for per-patch alignment with SiT intermediate representations.

Output shape: (B, 256, D)   # always 16x16 tokens

Supports:
- DINOv2 (torch.hub)
- DINOv3, MAE, SigLIP, CLIP (HF)
- BioCLIP (open_clip)
- OpenPhenom (RxRx1 6-channel teacher)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from faithful_cond_gen.data.rxrx1 import to_rgb

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _hf_auth_kwargs() -> dict:
    import os

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    return {"token": token} if token else {}


class REPAEncoder(nn.Module):
    """
    Frozen patch-token encoder used for REPA alignment.

    Returns patch tokens (not pooled):
        (B, 256, embed_dim)  where 256=16*16
    """

    def __init__(
        self,
        encoder_name: str = "dinov2-vit-b",
        resolution: int = 256,
        in_channels: int = 3,
        encoder_input_size: Optional[int] = None,
        target_grid: int = 16,  # 16x16 = 256 tokens
        device: str = "cuda",
    ):
        super().__init__()
        self.encoder_name = encoder_name.lower()
        self.encoder_type = self._parse_encoder_type(self.encoder_name)

        self.resolution = int(resolution)
        self.in_channels = int(in_channels)
        self._device = device

        self.target_grid = int(target_grid)

        # If not explicitly set, pick teacher input size that naturally yields 16x16 tokens.
        self.encoder_input_size = (
            int(encoder_input_size)
            if encoder_input_size is not None
            else int(self._default_teacher_input_size())
        )

        if self.encoder_type == "openphenom":
            self._load_openphenom_encoder(device)
        else:
            self._load_hf_encoder(device)

    # -------------------------------------------------------------------------
    # Setup helpers
    # -------------------------------------------------------------------------
    def _parse_encoder_type(self, name: str) -> str:
        if "dinov2" in name:
            return "dinov2"
        if "dinov3" in name:
            return "dinov3"
        if "mae" in name:
            return "mae"
        if "siglip" in name:
            return "siglip"
        if "bioclip" in name:
            return "bioclip"
        if "openphenom" in name:
            return "openphenom"
        if "clip" in name:
            return "clip"
        raise ValueError(f"Unknown encoder type: {name}")

    def _default_teacher_input_size(self) -> int:
        """
        Choose an input size that yields exactly 16x16 patch tokens without needing interpolation:
          - patch14 encoders -> 224 (224/14 = 16)
          - patch16 encoders -> 256 (256/16 = 16)
          - OpenPhenom fixed
        """
        if self.encoder_type in {"dinov2", "siglip"}:
            return 224
        if self.encoder_type in {"dinov3", "mae", "clip", "bioclip"}:
            return 256
        if self.encoder_type == "openphenom":
            return 512
        return 256

    # -------------------------------------------------------------------------
    # HF encoders (dinov2/dinov3/mae/siglip/clip) + BioCLIP (open_clip)
    # -------------------------------------------------------------------------
    def _load_hf_encoder(self, device: str):
        from faithful_cond_gen.eval.configs.encoder_config import (
            BIOCLIP,
            CLIP_VITB16,
            CLIP_VITL14,
            DINOV2_L14,
            DINOV3_L16,
            MAE_LARGE,
            SIGLIP_SO400M_224,
        )
        from transformers import AutoModel, CLIPModel, SiglipVisionModel

        config_map = {
            "dinov3": DINOV3_L16,
            "dinov2": DINOV2_L14,
            "mae": MAE_LARGE,
            "siglip": SIGLIP_SO400M_224,
            "bioclip": BIOCLIP,
            "clip": CLIP_VITL14,
        }
        config = config_map[self.encoder_type]

        # ---------------- BioCLIP (open_clip) ----------------
        if self.encoder_type == "bioclip":
            import open_clip

            model, _, _ = open_clip.create_model_and_transforms(
                f"hf-hub:{config.hf_path}"
            )
            self.encoder = model.visual
            self.encoder.eval().to(device)
            for p in self.encoder.parameters():
                p.requires_grad = False

            self.target_size = int(self.encoder_input_size)
            self.transform = T.Compose(
                [
                    T.Resize(
                        (self.target_size, self.target_size),
                        interpolation=T.InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    T.Normalize(mean=config.mean, std=config.std),
                ]
            )
            # best-effort embed dim
            self.embed_dim = getattr(self.encoder, "width", 768)
            return

        # ---------------- CLIP (HF) ----------------
        if self.encoder_type == "clip":
            clip = CLIPModel.from_pretrained(config.hf_path, **_hf_auth_kwargs())
            self.encoder = clip.vision_model
            self.encoder.eval().to(device)
            for p in self.encoder.parameters():
                p.requires_grad = False

            # CLIP ViT-L/14 requires 224x224 -> gives 16x16 tokens (256)
            self.target_size = int(self.encoder.config.image_size)  # 224
            self.embed_dim = int(self.encoder.config.hidden_size)

            self.transform = T.Compose(
                [
                    T.Resize(
                        (self.target_size, self.target_size),
                        interpolation=T.InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    T.Normalize(mean=config.mean, std=config.std),
                ]
            )
            return

        # ---------------- SigLIP (HF) ----------------
        auth = _hf_auth_kwargs()
        if self.encoder_type == "siglip":
            self.encoder = SiglipVisionModel.from_pretrained(config.hf_path, **auth)
        else:
            # dinov3 / mae
            self.encoder = AutoModel.from_pretrained(
                config.hf_path,
                trust_remote_code=getattr(config, "trust_remote_code", False),
                **getattr(config, "model_kwargs", {}),
                **auth,
            )

        self.encoder.eval().to(device)
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Pick patch size from config if possible (fallback 16).
        patch = int(getattr(self.encoder.config, "patch_size", 16))
        self.target_size = patch * self.target_grid

        # set preprocessing
        self.transform = T.Compose(
            [
                T.Resize(
                    (self.target_size, self.target_size),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=config.mean, std=config.std),
            ]
        )

        # probe embed dim
        self.embed_dim = self._probe_embed_dim(device)

    def _probe_embed_dim(self, device: str) -> int:
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.target_size, self.target_size, device=device)
            out = self.encoder(pixel_values=dummy)
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                return 768
            return int(hidden.shape[-1])

    # -------------------------------------------------------------------------
    # OpenPhenom (RxRx1 teacher)
    # -------------------------------------------------------------------------
    def _load_openphenom_encoder(self, device: str):
        from faithful_cond_gen.eval.configs.encoder_config import OPENPHENOM
        from transformers import AutoModel

        print(f"[REPA] Loading OpenPhenom from {OPENPHENOM.hf_path}...")
        self.encoder = AutoModel.from_pretrained(
            OPENPHENOM.hf_path,
            trust_remote_code=True,
        )
        self.encoder.eval().to(device)
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.op_channels = 6
        self.op_crops = 4
        self.op_img_dim = 512
        self.op_crop_size = 256
        self.op_patch_dim = 16

        self.op_patches_per_crop = (self.op_crop_size // self.op_patch_dim) ** 2  # 256
        self.op_patch_grid = int(math.sqrt(self.op_patches_per_crop))  # 16

        self.op_token_dim = 384
        self.embed_dim = self.op_channels * self.op_token_dim  # 2304

        self.instance_norm = nn.InstanceNorm2d(
            self.op_channels, affine=False, eps=1e-6
        ).to(device)
        self.transform = None

    def _cropify_openphenom(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if C != self.op_channels:
            raise ValueError(f"OpenPhenom expects {self.op_channels} channels, got {C}")
        if H != self.op_img_dim or W != self.op_img_dim:
            raise ValueError(
                f"OpenPhenom expects {self.op_img_dim}x{self.op_img_dim}, got {H}x{W}"
            )

        img = x.reshape(
            B,
            self.op_channels,
            self.op_crops // 2,
            self.op_crop_size,
            self.op_crops // 2,
            self.op_crop_size,
        )
        img = img.permute(0, 2, 4, 1, 3, 5)
        return img.reshape(
            B * (self.op_crops // 2) * (self.op_crops // 2),
            self.op_channels,
            self.op_crop_size,
            self.op_crop_size,
        )

    def _encode_latent_openphenom(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns latent stitched into a 32x32 grid per channel:
            (B, 6, 32, 32, 384)
        """
        B = x.shape[0]
        normed_x = self.instance_norm(x)
        cropped_x = self._cropify_openphenom(normed_x)  # (B*4, 6, 256, 256)

        latent, _, _ = self.encoder.encoder.forward_masked(cropped_x, 0.0)
        latent = latent.reshape(
            B,
            self.op_crops,
            self.op_channels * self.op_patches_per_crop + 1,
            self.op_token_dim,
        )

        latent = latent[:, :, 1:, :].reshape(
            B,
            self.op_crops,
            self.op_channels,
            self.op_patches_per_crop,
            self.op_token_dim,
        )

        # stitch crops 2x2 -> 32x32
        latent = latent.permute(0, 2, 1, 3, 4)  # (B,6,4,256,384)
        p = self.op_patch_grid  # 16
        latent = latent.reshape(B, self.op_channels, 2, 2, p, p, self.op_token_dim)
        latent = latent.permute(0, 1, 2, 4, 3, 5, 6).reshape(
            B, self.op_channels, 2 * p, 2 * p, self.op_token_dim
        )
        return latent  # (B, 6, 32, 32, 384)

    def _forward_openphenom(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, 256, 2304):
          - build 32x32 grid tokens
          - concat channels into embedding dim
          - avg pool 32x32 -> 16x16
        """
        B, C, H, W = images.shape
        if C != self.op_channels:
            raise ValueError(f"OpenPhenom expects {self.op_channels} channels, got {C}")

        if H != self.op_img_dim or W != self.op_img_dim:
            images = F.interpolate(
                images,
                size=(self.op_img_dim, self.op_img_dim),
                mode="bilinear",
                align_corners=False,
            )

        latent = self._encode_latent_openphenom(images)  # (B,6,32,32,384)

        tokens_32 = latent.permute(0, 2, 3, 1, 4).reshape(
            B, 32 * 32, self.op_token_dim * self.op_channels
        )  # (B,1024,2304)

        D = tokens_32.shape[-1]
        grid = tokens_32.transpose(1, 2).reshape(B, D, 32, 32)
        grid = F.avg_pool2d(grid, kernel_size=2, stride=2)  # (B,D,16,16)
        tokens_16 = grid.reshape(B, D, 16 * 16).transpose(1, 2)
        return tokens_16  # (B,256,2304)

    # -------------------------------------------------------------------------
    # Common helpers
    # -------------------------------------------------------------------------
    def _strip_special_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Many ViTs return: [CLS] + [register tokens] + [patch tokens].
        Drop the smallest prefix that makes the remainder a square.
        """
        L = hidden.shape[1]
        if L <= 1:
            return hidden

        for prefix in (0, 1, 2, 3, 4, 5, 8, 16):
            n = L - prefix
            if n <= 0:
                continue
            s = int(math.isqrt(n))
            if s * s == n:
                return hidden[:, prefix:]
        return hidden

    def _ensure_token_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, N, D) where N is square. Resize to (target_grid x target_grid).
        """
        B, N, D = tokens.shape
        s = int(math.isqrt(N))
        if s * s != N:
            raise ValueError(f"N={N} is not square (tokens.shape={tokens.shape})")

        if s == self.target_grid:
            return tokens
        else:
            raise NotImplementedError(
                f"Token grid resizing not implemented: {s} -> {self.target_grid}"
            )

        # grid = tokens.transpose(1, 2).reshape(B, D, s, s)
        # grid = F.interpolate(
        #     grid,
        #     size=(self.target_grid, self.target_grid),
        #     mode="bilinear",
        #     align_corners=False,
        # )
        # return grid.reshape(B, D, self.target_grid * self.target_grid).transpose(1, 2)

    def _convert_to_rgb(self, images: torch.Tensor) -> torch.Tensor:
        if self.in_channels == 6:
            return to_rgb(images)
        if self.in_channels == 3:
            return images
        raise ValueError(f"Unsupported in_channels: {self.in_channels}")

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "openphenom":
            return self._forward_openphenom(images)

        rgb_images = self._convert_to_rgb(images)
        x = self.transform(rgb_images)

        if self.encoder_type == "bioclip":
            tokens = self._forward_bioclip(x)
            return self._ensure_token_grid(tokens)

        tokens = self._forward_hf(x)
        return self._ensure_token_grid(tokens)

    def _forward_dinov2(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder.forward_features(x)
        if isinstance(out, dict):
            return out["x_norm_patchtokens"]  # already patch tokens
        return out[:, 1:]

    def _forward_bioclip(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x, return_all_tokens=True)
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        tokens = self._strip_special_tokens(tokens)
        return tokens

    def _forward_hf(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(pixel_values=x)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            raise ValueError(f"Encoder output has no last_hidden_state: {type(out)}")

        hidden = self._strip_special_tokens(hidden)
        return hidden


def load_repa_encoder(
    encoder_name: str, resolution: int, in_channels: int, device: str
) -> REPAEncoder:
    # Let REPAEncoder pick the correct teacher input size automatically.
    return REPAEncoder(
        encoder_name=encoder_name,
        resolution=resolution,
        in_channels=in_channels,
        encoder_input_size=None,
        target_grid=16,
        device=device,
    )
