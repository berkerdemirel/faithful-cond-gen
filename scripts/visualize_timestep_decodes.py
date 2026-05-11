"""Save a grid of x0_hat decodes across the 10 capture timesteps.

Rows = samples, cols = capture step. Each cell is the VAE-decoded x0_hat at
that step — i.e. "if we bailed here, this is what the model thinks the clean
image would be". Pairs visually with consecutive_image_distance.csv.

Usage:
    PYTHONPATH=src uv run python scripts/visualize_timestep_decodes.py \\
        --run-dir outputs/gen/celeba_vanilla_marginal_timesteps \\
        --checkpoint-key celeba_vanilla_marginal_v1 \\
        --sample-indices 0 2000 4000 6000 8000 10000 12000 14000
"""
import argparse
import logging
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from PIL import Image

from faithful_cond_gen.model.generator import GeneratorWrapper
from faithful_cond_gen.pl_modules.generator_pl import GeneratorPL
from faithful_cond_gen.utils.checkpoints import get_checkpoint_path

log = logging.getLogger(__name__)


def load_vae(checkpoint_key: str, device: torch.device):
    config_dir = str(Path(__file__).parent.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="generate_samples_celeba")
    gen_cfg = instantiate(cfg.model)
    backbone = GeneratorWrapper(gen_cfg)
    ckpt_path = get_checkpoint_path(checkpoint_key)
    module = GeneratorPL.load_from_checkpoint(
        ckpt_path, generator=backbone, map_location="cpu", strict=False,
    )
    if hasattr(module, "ema"):
        module.ema.apply()
    module.to(device).eval()
    return module.generator


def decode(generator, latents, device):
    dtype = next(generator.parameters()).dtype
    with torch.no_grad():
        out = generator.decode(latents.to(device=device, dtype=dtype))
    return torch.clamp(out.float().cpu(), 0.0, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint-key", required=True)
    p.add_argument("--sample-indices", type=int, nargs="+",
                   default=[0, 2000, 4000, 6000, 8000, 10000, 12000, 14000])
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--final-image-dir", type=Path, default=None,
                   help="If set, append the actual decoded final image next to each row "
                        "(pulled from images/ directory, keyed by consolidated index).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    ks = sorted(
        int(p.stem.replace("x0hat_step", ""))
        for p in args.run_dir.glob("x0hat_step*.pt")
    )
    log.info(f"Steps: {ks}")

    generator = load_vae(args.checkpoint_key, device)

    # Collect rows: for each sample idx, decode x0hat at each k
    rows = {idx: [] for idx in args.sample_indices}
    filenames_sample = None
    for k in ks:
        d = torch.load(args.run_dir / f"x0hat_step{k}.pt", map_location="cpu", weights_only=False)
        latents = d["x0hat"]
        if filenames_sample is None:
            filenames_sample = d.get("filenames")
        batch = torch.stack([latents[i] for i in args.sample_indices], dim=0)
        decoded = decode(generator, batch, device)  # (len(indices), 3, H, W)
        for pos, idx in enumerate(args.sample_indices):
            rows[idx].append(decoded[pos])
        log.info(f"  step{k}: decoded rows")

    # Optionally append the final VAE-decoded image from disk for reference
    if args.final_image_dir is None:
        args.final_image_dir = args.run_dir / "images"
    final_imgs = {}
    if args.final_image_dir.exists() and filenames_sample is not None:
        for idx in args.sample_indices:
            if idx < len(filenames_sample):
                p_img = args.final_image_dir / filenames_sample[idx]
                if p_img.exists():
                    final_imgs[idx] = Image.open(p_img).convert("RGB")

    # Build grid
    n_rows = len(args.sample_indices)
    n_cols = len(ks) + (1 if final_imgs else 0)
    # All decoded tensors are (3, H, W) float in [0,1]; H=W=256
    H, W = rows[args.sample_indices[0]][0].shape[1:]
    padding = 4
    grid_w = n_cols * W + (n_cols + 1) * padding
    grid_h = n_rows * H + (n_rows + 1) * padding

    grid = Image.new("RGB", (grid_w, grid_h), color=(32, 32, 32))
    for r, idx in enumerate(args.sample_indices):
        for c, img_t in enumerate(rows[idx]):
            arr = (img_t.permute(1, 2, 0).numpy() * 255).astype("uint8")
            tile = Image.fromarray(arr)
            x = padding + c * (W + padding)
            y = padding + r * (H + padding)
            grid.paste(tile, (x, y))
        if final_imgs and idx in final_imgs:
            tile = final_imgs[idx].resize((W, H))
            x = padding + len(ks) * (W + padding)
            y = padding + r * (H + padding)
            grid.paste(tile, (x, y))

    out = args.output or (args.run_dir / "timestep_decode_grid.png")
    grid.save(out)
    log.info(f"Saved {out}  ({n_rows} rows x {n_cols} cols; cols=k in order + final)")


if __name__ == "__main__":
    main()
