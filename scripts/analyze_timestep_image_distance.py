"""Consecutive-step image distance for a timestep ablation run.

For each x0hat_step{k}.pt shard produced by generate_samples_repa.py with
return_x0hat=true, VAE-decode the predicted clean latent x0_hat and compute
the average image distance between consecutive capture steps. Writes a CSV
suitable to pair with the trust-scoring-vs-k curve.

Usage:
    PYTHONPATH=src uv run python scripts/analyze_timestep_image_distance.py \\
        --run-dir outputs/gen/celeba_vanilla_marginal_timesteps \\
        --checkpoint-key celeba_vanilla_marginal_v1 \\
        --max-samples 500
"""
import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List

import hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from faithful_cond_gen.model.generator import GeneratorWrapper
from faithful_cond_gen.pl_modules.generator_pl import GeneratorPL
from faithful_cond_gen.utils.checkpoints import get_checkpoint_path

log = logging.getLogger(__name__)


def discover_step_files(run_dir: Path) -> List[int]:
    """Return sorted list of k indices for which x0hat_step{k}.pt exists."""
    ks = []
    for p in run_dir.glob("x0hat_step*.pt"):
        try:
            ks.append(int(p.stem.replace("x0hat_step", "")))
        except ValueError:
            continue
    return sorted(ks)


def load_vae(checkpoint_key: str, device: torch.device, config_name: str = "generate_samples_celeba"):
    """Load generator, return the VAE (its .decode matches the training VAE)."""
    config_dir = str(Path(__file__).parent.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
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


def decode_latents(
    generator: GeneratorWrapper, latents: torch.Tensor, batch_size: int, device
) -> torch.Tensor:
    """Decode (N, C, H, W) latents to (N, 3, H', W') images in [0, 1]."""
    imgs = []
    dtype = next(generator.parameters()).dtype
    for i in range(0, latents.shape[0], batch_size):
        chunk = latents[i : i + batch_size].to(device=device, dtype=dtype)
        with torch.no_grad():
            out = generator.decode(chunk)
        out = torch.clamp(out.float().cpu(), 0.0, 1.0)
        imgs.append(out)
    return torch.cat(imgs, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint-key", required=True)
    p.add_argument("--max-samples", type=int, default=500,
                   help="Subsample this many samples (deterministic) for the distance average.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--config-name", default=None,
                   help="Hydra config to load model from. Default: infer from checkpoint key.")
    args = p.parse_args()
    if args.config_name is None:
        args.config_name = (
            "generate_samples_rxrx1" if "rxrx1" in args.checkpoint_key
            else "generate_samples_celeba"
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    ks = discover_step_files(args.run_dir)
    if len(ks) < 2:
        log.error(f"Need >=2 x0hat_step*.pt files; found {len(ks)} in {args.run_dir}")
        sys.exit(1)
    log.info(f"Found steps: {ks}")

    first = torch.load(args.run_dir / f"x0hat_step{ks[0]}.pt", map_location="cpu", weights_only=False)
    n_total = first["x0hat"].shape[0]
    idx = torch.arange(n_total)
    if args.max_samples is not None and args.max_samples < n_total:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(n_total, generator=g)[: args.max_samples]
    log.info(f"Using {idx.shape[0]}/{n_total} samples for distance average")

    generator = load_vae(args.checkpoint_key, device, config_name=args.config_name)

    decoded = {}
    for k in ks:
        d = torch.load(args.run_dir / f"x0hat_step{k}.pt", map_location="cpu", weights_only=False)
        x0h = d["x0hat"][idx]
        imgs = decode_latents(generator, x0h, args.batch_size, device)
        decoded[k] = imgs
        log.info(f"  step{k}: decoded {imgs.shape}")

    out_csv = args.output_csv or (args.run_dir / "consecutive_image_distance.csv")
    rows = []
    for k_from, k_to in zip(ks[:-1], ks[1:]):
        diff = (decoded[k_to] - decoded[k_from]).flatten(1)
        l2 = diff.norm(dim=1).mean().item()
        l1 = diff.abs().mean().item()
        rows.append({"k_from": k_from, "k_to": k_to, "mean_l2": l2, "mean_l1": l1})
        log.info(f"  k={k_from}->{k_to}: l2={l2:.4f}, l1={l1:.4f}")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k_from", "k_to", "mean_l2", "mean_l1"])
        w.writeheader()
        w.writerows(rows)
    log.info(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
