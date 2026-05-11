"""
OpenPhenom encoder→decoder reconstruction sanity check.

For a few real RxRx1 images:
  1. real → to_rgb → save
  2. real → OpenPhenom encoder → decoder → unpatchify → to_rgb → save
  3. Compute per-image MSE and correlation

Also checks: does REPA's encode path (6ch, 512x512 crop) match the native
model path (11ch padded, 256x256)?

Model expects 11 channels x 256x256. RxRx1 has 6 channels x 256x256.
We zero-pad channels 6-10 for the native forward path.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict
from transformers import AutoModel
from torchvision.utils import save_image
from faithful_cond_gen.data.rxrx1 import RxRx1DataModule, RxRx1DataConfig
from faithful_cond_gen.model.repa_encoder import load_repa_encoder

DEVICE = "cuda:0"
OUT_DIR = Path("outputs/openphenom_reconstruction")
N_IMAGES = 8
BS = 4
N_CHANNELS_NATIVE = 11
N_CHANNELS_DATA = 6
PATCH_SIZE = 16
IMG_SIZE = 256
PATCHES_PER_SIDE = IMG_SIZE // PATCH_SIZE  # 16
TOKENS_PER_CHANNEL = PATCHES_PER_SIDE ** 2  # 256


def rxrx1_to_rgb(img):
    """R=Mito(4), G=AGP(3), B=DNA(0). Per-image normalize to [0,1]."""
    if img.ndim == 4:
        rgb = img[:, [4, 3, 0]]
    else:
        rgb = img[[4, 3, 0]]
    rgb = rgb - rgb.min()
    if rgb.max() > 0:
        rgb = rgb / rgb.max()
    return rgb


def unpatchify(recon_tokens, n_channels):
    """Convert (B, n_channels*256, 256) patch tokens back to (B, C, 256, 256) image."""
    B = recon_tokens.shape[0]
    # (B, C*tokens_per_ch, patch_h*patch_w)
    recon = recon_tokens.reshape(
        B, n_channels, TOKENS_PER_CHANNEL, PATCH_SIZE, PATCH_SIZE
    )
    # tokens are in row-major 16x16 grid
    recon = recon.reshape(
        B, n_channels, PATCHES_PER_SIDE, PATCHES_PER_SIDE, PATCH_SIZE, PATCH_SIZE
    )
    # (B, C, grid_h, patch_h, grid_w, patch_w) -> (B, C, H, W)
    recon = recon.permute(0, 1, 2, 4, 3, 5).reshape(
        B, n_channels, IMG_SIZE, IMG_SIZE
    )
    return recon


@torch.no_grad()
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    dm = RxRx1DataModule(RxRx1DataConfig(
        data_dir="/mnt/pvc/AutoSync/data/rxrx1",
        img_size=[512, 512], resize=[256, 256],
        reduce_channels=False, augment_train=False, normalize=False,
        use_numpy=True, use_parquet=False,
        batch_size=BS, num_workers=4, val_size=0.1,
        seed=1337, rare_threshold=20, held_out_pairs=None,
    ))
    ds = dm.train_dataloader().dataset

    # Pick one image per top condition
    counts = defaultdict(int)
    for ct, si in zip(ds.cell_type_ids, ds.sirna_ids):
        counts[(int(ct), int(si))] += 1
    top_conds = [c for c, _ in sorted(counts.items(), key=lambda x: -x[1])[:N_IMAGES]]
    cond_to_idx = defaultdict(list)
    for i, (ct, si) in enumerate(zip(ds.cell_type_ids, ds.sirna_ids)):
        key = (int(ct), int(si))
        if key in set(top_conds):
            cond_to_idx[key].append(i)

    indices = [cond_to_idx[c][0] for c in top_conds]
    images = torch.stack([ds[i][0] for i in indices]).to(DEVICE)
    if images.min() < 0:
        images = (images + 1) / 2
    B = len(images)
    print(f"Images: {images.shape}, range=[{images.min():.3f}, {images.max():.3f}]")

    # ── Load full OpenPhenom model ───────────────────────────────────────
    print("\nLoading OpenPhenom (full model with decoder)...")
    model = AutoModel.from_pretrained(
        "recursionpharma/OpenPhenom", trust_remote_code=True
    ).to(DEVICE).eval()

    # Pad 6ch → 11ch with zeros
    images_11ch = torch.zeros(B, N_CHANNELS_NATIVE, IMG_SIZE, IMG_SIZE, device=DEVICE)
    images_11ch[:, :N_CHANNELS_DATA] = images
    print(f"Padded to {images_11ch.shape}")

    # Normalize (what the model sees)
    normed = model.input_norm(images_11ch)
    print(f"After input_norm: range=[{normed.min():.3f}, {normed.max():.3f}]")

    # Encode (11ch → 2817 tokens = CLS + 11*256)
    latent, mask, ind_restore = model.encoder.forward_masked(normed, 0.0)
    print(f"Latent: {latent.shape}, mask sum: {mask.sum().item()}/{mask.numel()}")

    # Decoder expects 6 channels (1537 tokens = CLS + 6*256).
    # Select CLS + first 6 channels from encoder output, then decode.
    n_tokens_6ch = N_CHANNELS_DATA * TOKENS_PER_CHANNEL  # 1536
    proj = model.encoder_decoder_proj(latent)  # (B, 2817, 512)
    proj_6ch = torch.cat([proj[:, :1], proj[:, 1:1+n_tokens_6ch]], dim=1)  # (B, 1537, 512)
    ind_restore_6ch = torch.arange(n_tokens_6ch, device=DEVICE).unsqueeze(0).expand(B, -1)

    decoder_tokens = model.decoder.forward_masked(proj_6ch, ind_restore_6ch)
    reconstruction = model.decoder_pred(decoder_tokens)[:, 1:]  # drop CLS
    print(f"Reconstruction tokens: {reconstruction.shape}")

    # Unpatchify (6 channels)
    recon_img = unpatchify(reconstruction, N_CHANNELS_DATA)
    print(f"Reconstructed image: {recon_img.shape}")

    recon_6ch = recon_img
    normed_6ch = normed[:, :N_CHANNELS_DATA]

    # ── MSE and correlation in normalized space ──────────────────────────
    print(f"\n{'='*60}")
    print("Per-image reconstruction quality (normalized pixel space)")
    print(f"{'='*60}")
    for i in range(B):
        mse = F.mse_loss(recon_6ch[i], normed_6ch[i]).item()
        x = normed_6ch[i].flatten().cpu().float().numpy()
        y = recon_6ch[i].flatten().cpu().float().numpy()
        corr = np.corrcoef(x, y)[0, 1]
        per_ch_mse = [F.mse_loss(recon_6ch[i, c], normed_6ch[i, c]).item() for c in range(6)]
        print(f"  Img {i} (cond={top_conds[i]}): MSE={mse:.6f}  r={corr:.4f}  "
              f"ch_mse={[f'{v:.4f}' for v in per_ch_mse]}")

    # Check: maybe the token ordering is wrong. The encoder processes channels
    # interleaved or in a different order than assumed. Let's check by also trying
    # 6ch directly (no padding) with interpolated pos_embed.
    print(f"\n{'='*60}")
    print("Sanity check: try 6ch directly (skip 11ch padding)")
    print(f"{'='*60}")
    normed_6only = model.input_norm(images)  # 6ch input
    # Manually patch embed + handle pos_embed
    patched = model.encoder.vit_backbone.patch_embed(normed_6only)  # (B, 1536, 384)
    print(f"  6ch patch_embed: {patched.shape}")
    # Interpolate pos_embed from 2817 to 1537
    pos = model.encoder.vit_backbone.pos_embed  # (1, 2817, 384)
    cls_pos = pos[:, :1]
    patch_pos = pos[:, 1:]  # (1, 2816, 384)
    # Reshape to spatial: (1, 384, 2816) -> need to figure out spatial layout
    # Actually for channel-agnostic, pos_embed is 1D over all channel*spatial tokens
    # Just take first 1536 positions (for 6 channels)
    patch_pos_6 = patch_pos[:, :1536]
    pos_6 = torch.cat([cls_pos, patch_pos_6], dim=1)
    x = patched + pos_6[:, 1:]  # no CLS yet
    # Add CLS
    cls_token = model.encoder.vit_backbone.cls_token.expand(B, -1, -1) + cls_pos
    x = torch.cat([cls_token, x], dim=1)
    # Run through transformer blocks
    x = model.encoder.vit_backbone.norm_pre(x)
    x = model.encoder.vit_backbone.blocks(x)
    x = model.encoder.vit_backbone.norm(x)
    latent_6direct = x
    print(f"  6ch-direct latent: {latent_6direct.shape}")

    # Decode
    proj_6direct = model.encoder_decoder_proj(latent_6direct)
    ind_restore_id = torch.arange(1536, device=DEVICE).unsqueeze(0).expand(B, -1)
    dec_6direct = model.decoder.forward_masked(proj_6direct, ind_restore_id)
    recon_6direct = model.decoder_pred(dec_6direct)[:, 1:]
    recon_6direct_img = unpatchify(recon_6direct, N_CHANNELS_DATA)

    for i in range(min(B, 4)):
        mse = F.mse_loss(recon_6direct_img[i], normed_6only[i]).item()
        x_np = normed_6only[i].flatten().cpu().numpy()
        y_np = recon_6direct_img[i].flatten().cpu().numpy()
        corr = np.corrcoef(x_np, y_np)[0, 1]
        print(f"  Img {i} (6ch-direct): MSE={mse:.6f}  r={corr:.4f}")

    # ── Compare REPA encoder output vs native encoder output ─────────────
    print(f"\n{'='*60}")
    print("REPA encoder vs native encoder feature comparison")
    print(f"{'='*60}")

    # Native: take encoder latent, extract 6-channel tokens, pool
    # latent is (B, 1 + 11*256, 384), drop CLS
    native_tokens = latent[:, 1:, :]  # (B, 11*256, 384)
    # Reshape to (B, 11, 256, 384), take first 6 channels
    native_tokens = native_tokens.reshape(B, N_CHANNELS_NATIVE, TOKENS_PER_CHANNEL, 384)
    native_6ch = native_tokens[:, :N_CHANNELS_DATA]  # (B, 6, 256, 384)
    # Concatenate channels: (B, 256, 6*384=2304)
    native_2304 = native_6ch.permute(0, 2, 1, 3).reshape(B, TOKENS_PER_CHANNEL, N_CHANNELS_DATA * 384)
    # Mean pool over patches
    native_pooled = native_2304.mean(dim=1)  # (B, 2304)
    native_pooled = native_pooled / (native_pooled.norm(dim=-1, keepdim=True) + 1e-12)

    # REPA encoder: uses 512x512 crop approach
    del model
    torch.cuda.empty_cache()
    repa_enc = load_repa_encoder("openphenom", 256, 6, device=DEVICE).eval()

    # REPA needs 512x512 input
    images_512 = F.interpolate(images, size=(512, 512), mode="bilinear", align_corners=False)
    repa_tokens = repa_enc(images_512)  # (B, 256, 2304)
    repa_pooled = repa_tokens.mean(dim=1)  # (B, 2304)
    repa_pooled = repa_pooled / (repa_pooled.norm(dim=-1, keepdim=True) + 1e-12)

    # Compare
    for i in range(B):
        cos = (native_pooled[i] @ repa_pooled[i]).item()
        print(f"  Img {i}: cosine(native_11ch_padded, repa_512crop) = {cos:.4f}")

    # Also compare native 256 (no upsample) vs REPA 512 (with upsample)
    print(f"\n  Note: REPA upsamples 256→512 then crops 2x2, pools 32x32→16x16")
    print(f"  Native: uses 256x256 directly with 11ch padding")

    del repa_enc
    torch.cuda.empty_cache()

    # ── Save visualizations ──────────────────────────────────────────────
    print(f"\nSaving visualizations to {OUT_DIR}/...")
    for i in range(B):
        orig_rgb = rxrx1_to_rgb(images[i].cpu())
        recon_rgb = rxrx1_to_rgb(recon_6ch[i].cpu())
        recon_rgb = recon_rgb - recon_rgb.min()
        if recon_rgb.max() > 0:
            recon_rgb = recon_rgb / recon_rgb.max()
        pair = torch.cat([orig_rgb, recon_rgb], dim=2)
        save_image(pair, OUT_DIR / f"img{i}_cond{top_conds[i][0]}_{top_conds[i][1]}.png")

    originals = torch.stack([rxrx1_to_rgb(images[i].cpu()) for i in range(B)])
    recons = []
    for i in range(B):
        r = rxrx1_to_rgb(recon_6ch[i].cpu())
        r = r - r.min()
        if r.max() > 0:
            r = r / r.max()
        recons.append(r)
    recons = torch.stack(recons)
    grid = torch.cat([originals, recons], dim=0)
    save_image(grid, OUT_DIR / "grid_orig_vs_recon.png", nrow=B)
    print("Done.")


if __name__ == "__main__":
    main()
