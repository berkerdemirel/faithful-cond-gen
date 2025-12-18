# src/faithful_cond_gen/eval/configs/encoder_config.py
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class EncoderConfig:
    name: str
    hf_path: str
    image_size: Tuple[int, int]
    input_channels: int = 3
    model_kwargs: dict = field(default_factory=dict)

    # Preprocessing stats
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    trust_remote_code: bool = False

    # Optional: For models like SigLIP that behave slightly differently in getting embeddings
    pooling_type: str = "cls"  # 'cls', 'mean', 'last_hidden_state'


# --- PRESETS ---

# 1. DINOv2 (Existing)
DINOV2_L14 = EncoderConfig(
    name="dinov2-vitl14",
    hf_path="facebook/dinov2-large",
    image_size=(224, 224),
)

# 2. DINOv3 (New - Meta's latest)
# Using the standard ViT-L/16 backbone
DINOV3_L16 = EncoderConfig(
    name="dinov3-vitl16",
    hf_path="facebook/dinov3-vitl16-pretrain-lvd1689m",
    image_size=(224, 224),
    pooling_type="cls",
)

# 3. MAE (Masked Autoencoder) - My Choice: ViT-Large
# MAE learns very different features (texture/pixel-level) compared to contrastive models
MAE_LARGE = EncoderConfig(
    name="mae-vit-large",
    hf_path="facebook/vit-mae-large",
    image_size=(224, 224),
    pooling_type="cls",  # MAE encoder output [0] is the CLS token
)

# 4. OpenPhenom (Specialized for Multi-Channel Scientific Images)
OPENPHENOM = EncoderConfig(
    name="openphenom",
    hf_path="recursionpharma/OpenPhenom",
    image_size=(512, 512),
    input_channels=6,
    trust_remote_code=True,
    pooling_type="mean",  # Our wrapper does mean pooling
)

# 5. SigLIP-SO400M (Specialized CLIP for Scientific Images)
SIGLIP_SO400M = EncoderConfig(
    name="siglip-so400m",
    hf_path="google/siglip-so400m-patch14-384",
    image_size=(384, 384),
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    pooling_type="mean",  # SigLIP vision model output is usually spatial map, so we mean pool
)

# 6. BioCLIP (Specialized CLIP for Biological Images)
BIOCLIP = EncoderConfig(
    name="bioclip",
    hf_path="imageomics/bioclip",  # Will be treated as hf-hub:imageomics/bioclip
    image_size=(224, 224),
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
)
