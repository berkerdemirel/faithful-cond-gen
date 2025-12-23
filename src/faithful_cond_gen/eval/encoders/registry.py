import inspect
import math
import os
from typing import Dict, Optional

import open_clip
import torch
import torch.nn as nn
import torchvision.transforms as T
from faithful_cond_gen.eval.configs.encoder_config import EncoderConfig
from PIL import Image
from transformers import AutoModel, CLIPModel, SiglipVisionModel

from .base import BaseEncoder


# -----------------------------------------------------------------------------
# Smart ToTensor that handles both PIL and Tensor inputs
# -----------------------------------------------------------------------------
class SmartToTensor:
    """Converts PIL/ndarray to Tensor, but passes through if already a Tensor."""

    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            # Already a tensor - ensure it's float and in [0,1] range
            if pic.dtype == torch.uint8:
                return pic.float().div(255.0)
            return pic.float()
        # PIL or ndarray - use standard ToTensor
        return T.functional.to_tensor(pic)


def _hf_auth_kwargs():
    tok = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not tok:
        return {}
    # transformers versions differ: some accept token=, older use use_auth_token=
    try:
        sig = inspect.signature(AutoModel.from_pretrained)
        if "token" in sig.parameters:
            return {"token": tok}
        if "use_auth_token" in sig.parameters:
            return {"use_auth_token": tok}
    except Exception:
        pass
    # safe fallback: try token first in caller with try/except if needed
    return {"token": tok}


# -----------------------------------------------------------------------------
# 1. Standard HF Encoder (DINOv2, generic ViTs)
# -----------------------------------------------------------------------------
class HFEncoder(BaseEncoder):
    def __init__(self, config: EncoderConfig, device="cuda"):
        super().__init__(config, device)
        print(f"Loading {config.name} from {config.hf_path}...")

        auth = _hf_auth_kwargs()

        if "siglip" in config.name.lower():
            self.backbone = SiglipVisionModel.from_pretrained(config.hf_path, **auth)
        elif "clip" in config.name.lower() and "bio" not in config.name.lower():
            self.backbone = CLIPModel.from_pretrained(
                config.hf_path, **auth
            ).vision_model
        else:
            self.backbone = AutoModel.from_pretrained(
                config.hf_path,
                trust_remote_code=config.trust_remote_code,
                **config.model_kwargs,
                **auth,
            )

        self.backbone.to(device)
        self.backbone.eval()

        # Determine Feature Dim
        with torch.no_grad():
            dummy = torch.randn(1, config.input_channels, *config.image_size).to(device)
            out = self._forward_impl(dummy)
            self._feat_dim = out.shape[-1]

    def _forward_impl(self, x):
        outputs = self.backbone(pixel_values=x)

        # 1. Handle Pooling Strategies
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output

        last_hidden = outputs.last_hidden_state

        if self.cfg.pooling_type == "cls":
            return last_hidden[:, 0]
        elif self.cfg.pooling_type == "mean":
            return last_hidden.mean(dim=1)
        elif self.cfg.pooling_type == "last_hidden_state":
            return last_hidden.mean(dim=1)

        # Default fallback
        return last_hidden[:, 0]

    def forward(self, x):
        features = self._forward_impl(x)
        # RAW FEATURES (No Normalization)
        return {"features": features}

    def get_transform(self):
        """Transform that handles both PIL images and Tensors."""
        return T.Compose(
            [
                SmartToTensor(),  # Smart converter: PIL/ndarray -> Tensor, or pass-through
                T.ConvertImageDtype(torch.float32),  # Ensure float32
                T.Resize(
                    self.cfg.image_size, interpolation=T.InterpolationMode.BICUBIC
                ),
                T.CenterCrop(self.cfg.image_size),
                T.Normalize(mean=self.cfg.mean, std=self.cfg.std),
            ]
        )

    @property
    def feature_dim(self):
        return self._feat_dim


# -----------------------------------------------------------------------------
# 2. OpenCLIP Encoder (For BioCLIP)
# -----------------------------------------------------------------------------
class OpenCLIPEncoder(BaseEncoder):
    def __init__(self, config: EncoderConfig, device="cuda"):
        super().__init__(config, device)
        if open_clip is None:
            raise ImportError("Please install open_clip: `pip install open_clip_torch`")

        print(f"Loading OpenCLIP model: {config.hf_path}...")
        model_name = (
            config.hf_path
            if "hf-hub:" in config.hf_path
            else f"hf-hub:{config.hf_path}"
        )

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name, device=device
        )
        self.model.eval()
        self._feat_dim = self.model.visual.output_dim

    def forward(self, x):
        features = self.model.encode_image(x)
        # RAW FEATURES (No Normalization)
        return {"features": features}

    def get_transform(self):
        """Transform that handles both PIL images and Tensors."""
        return T.Compose(
            [
                SmartToTensor(),  # PIL -> Tensor [0,1], no-op if already Tensor
                T.ConvertImageDtype(torch.float32),  # Ensure float32
                T.Resize(
                    self.cfg.image_size, interpolation=T.InterpolationMode.BICUBIC
                ),
                T.CenterCrop(self.cfg.image_size),
                T.Normalize(mean=self.cfg.mean, std=self.cfg.std),
            ]
        )

    @property
    def feature_dim(self):
        return self._feat_dim


# -----------------------------------------------------------------------------
# 3. OpenPhenom Encoder (Custom logic)
# -----------------------------------------------------------------------------
class OpenPhenomWrapper(BaseEncoder):
    def __init__(self, config: EncoderConfig, device="cuda"):
        super().__init__(config, device)
        print(f"Loading OpenPhenom logic from {config.hf_path}...")

        # --- Copied/Adapted Logic Start ---
        auth = _hf_auth_kwargs()
        self.encoder = AutoModel.from_pretrained(
            config.hf_path,
            trust_remote_code=config.trust_remote_code,
            **auth,
        ).to(device)
        self.encoder.eval()

        self.patch_size = 256
        self.feature_dim_raw = 384
        self.channels = config.input_channels  # 6
        self.crops = 4
        self.img_dim = config.image_size[0]  # 512

        # Instance Norm is part of the model definition
        self.instance_norm = nn.InstanceNorm2d(self.channels, affine=False, eps=1e-6)
        # --- Copied Logic End ---

        self._feat_dim = 384

    # --- Methods from your script ---

    def iter_border_patches(self, width: int, height: int, patch_size: int):
        x_start, x_end, y_start, y_end = (0, width, 0, height)
        for x in range(x_start, x_end - patch_size + 1, patch_size):
            for y in range(y_start, y_end - patch_size + 1, patch_size):
                yield x, y

    def patch_image(self, image_array: torch.Tensor) -> torch.Tensor:
        _, width, height = image_array.shape
        output_patches = []
        for x, y in self.iter_border_patches(width, height, self.patch_size):
            patch = image_array[
                :, y : y + self.patch_size, x : x + self.patch_size
            ].clone()
            output_patches.append(patch)
        return torch.stack(output_patches)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Adapted forward pass.
        Input: (B, 6, 512, 512)
        Output: {'features': (B, 384)}
        """
        B, C, H, W = x.shape
        # Ensure input is on device
        x = x.to(self.device)

        # 1. Instance Norm
        x = self.instance_norm(x)

        # 2. Patching (B, 6, 512, 512) -> (B*4, 6, 256, 256)
        # Note: Your script had a list comprehension over batch. Vectorizing is tricky
        # with the loop logic, so we stick to your stacked approach for correctness.
        x_patched = torch.vstack([self.patch_image(img) for img in x])

        # 3. Predict via HF backbone
        # OpenPhenom `predict` expects inputs on device
        out = self.encoder.predict(x_patched)  # (B*4, 384)

        # 4. Aggregate Crops
        n_crops = (H // self.patch_size) * (W // self.patch_size)  # Should be 4
        out = out.view(B, n_crops, out.shape[1]).mean(dim=1)  # (B, 384)

        return {"features": out}

    def get_transform(self) -> T.Compose:
        """Transform that handles both PIL images and Tensors.

        OpenPhenom performs InstanceNorm internally.
        We need to ensure size is 512x512 and convert to float tensor.
        """
        return T.Compose(
            [
                SmartToTensor(),  # Smart converter: PIL/ndarray -> Tensor, or pass-through
                T.ConvertImageDtype(torch.float32),  # Ensure float32
                T.Resize(
                    self.cfg.image_size, interpolation=T.InterpolationMode.NEAREST
                ),
                # No Normalize here! InstanceNorm is in forward().
            ]
        )

    @property
    def feature_dim(self) -> int:
        return self._feat_dim


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
def load_encoder(config: EncoderConfig, device="cuda") -> BaseEncoder:
    if "openphenom" in config.name.lower():
        return OpenPhenomWrapper(config, device=device)
    elif "bioclip" in config.name.lower():
        return OpenCLIPEncoder(config, device=device)
    else:
        return HFEncoder(config, device=device)
