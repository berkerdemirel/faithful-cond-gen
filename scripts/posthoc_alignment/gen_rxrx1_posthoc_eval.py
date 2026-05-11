"""Generate samples for 50 RxRx1 perturbation conditions with raw_hidden capture.

Selects 5 heldout + 45 seen (cell_type_id, sirna_id) pairs,
generates ~200 samples per condition, and saves images + raw_hidden per condition.

Multi-GPU: conditions distributed across available GPUs.

Usage:
    # Vanilla
    PYTHONPATH=src uv run python scripts/posthoc_alignment/gen_rxrx1_posthoc_eval.py \
        --checkpoint-key rxrx1_vanilla_marginal_v1

    # REPA SigLIP
    PYTHONPATH=src uv run python scripts/posthoc_alignment/gen_rxrx1_posthoc_eval.py \
        --checkpoint-key rxrx1_repa_siglip_marginal_v1 \
        --use-repa --repa-encoder siglip
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from faithful_cond_gen.eval.trust_eval.config import RXRX1_HELDOUT_PAIRS
from faithful_cond_gen.model.generator import GeneratorConfig, GeneratorWrapper
from faithful_cond_gen.pl_modules.generator_pl import GeneratorPL
from faithful_cond_gen.utils.checkpoints import get_checkpoint_path

CONDITION_KEYS = ["cell_type_id", "sirna_id"]
NUM_INFERENCE_STEPS = 250
T_CUTOFF = 0.04
CFG_SCALE = 1.0
BATCH_SIZE = 24  # 6ch images use VAE channel-folding (6x effective batch)
SEED = 42


def select_50_conditions(real_features_path):
    """Select 5 heldout + 45 seen conditions with the most real samples."""
    real = torch.load(real_features_path, map_location="cpu", weights_only=False)
    ct = real["metadata"]["cell_type_id"].numpy()
    sirna = real["metadata"]["sirna_id"].numpy()

    real_pairs = set(zip(ct.tolist(), sirna.tolist()))
    pair_counts = Counter(zip(ct.tolist(), sirna.tolist()))

    # 5 heldout pairs with the most real samples (for robust KID)
    heldout_with_real = [(p, pair_counts[p]) for p in sorted(RXRX1_HELDOUT_PAIRS) if p in real_pairs]
    heldout_with_real.sort(key=lambda x: -x[1])
    selected_heldout = [p for p, _ in heldout_with_real[:5]]

    # 45 seen pairs with the most real samples
    seen = sorted(real_pairs - RXRX1_HELDOUT_PAIRS)
    seen_with_counts = [(p, pair_counts[p]) for p in seen]
    seen_with_counts.sort(key=lambda x: -x[1])
    selected_seen = [p for p, _ in seen_with_counts[:45]]

    print(f"Selected {len(selected_heldout)} heldout (top counts: {[c for _, c in heldout_with_real[:5]]})")
    print(f"Selected {len(selected_seen)} seen (count range: {seen_with_counts[0][1]}-{seen_with_counts[44][1]})")
    return selected_heldout, selected_seen


def build_generator_config(args):
    return GeneratorConfig(
        image_size=256,
        in_channels=6,
        vae_model_name="stabilityai/sd-vae-ft-mse",
        vae_freeze=True,
        sit_arch="SiT-B/2",
        attr_num_classes=[4, 1138],
        class_dropout_prob=0.1,
        use_repa=args.use_repa,
        repa_encoder=args.repa_encoder if args.use_repa else "dinov3-vit-l",
        repa_proj_coeff=0.5 if args.use_repa else 0.0,
        repa_encoder_depth=8,
        repa_projector_dim=2048,
    )


def load_model(args, device):
    gen_cfg = build_generator_config(args)
    backbone = GeneratorWrapper(gen_cfg)
    ckpt_path = get_checkpoint_path(args.checkpoint_key)
    print(f"  Loading checkpoint: {ckpt_path}")
    model = GeneratorPL.load_from_checkpoint(
        ckpt_path, generator=backbone, map_location="cpu", strict=False,
    )
    if hasattr(model, "ema"):
        print("  Applying EMA weights")
        model.ema.apply()
    model.to(device).eval()
    return model


def generate_for_conditions(gpu_id, conditions, args, spc, bs, save_dir):
    """Worker: generate samples for assigned conditions on one GPU."""
    device = torch.device(f"cuda:{gpu_id}")
    model = load_model(args, device)

    for cond in conditions:
        cell_type_id, sirna_id = cond
        cond_str = f"{cell_type_id}_{sirna_id}"
        save_path = save_dir / f"cond_{cond_str}.pt"
        if save_path.exists():
            print(f"  [GPU {gpu_id}] {cond_str} already exists, skipping", flush=True)
            continue

        cond_tensor = torch.tensor([cell_type_id, sirna_id], device=device, dtype=torch.long)
        cond_images, cond_raw_hidden = [], []

        remaining = spc
        while remaining > 0:
            b = min(bs, remaining)
            batch_cond = cond_tensor.unsqueeze(0).expand(b, -1)

            with torch.no_grad():
                images, raw_hidden_features = model.generator.sample(
                    cond_ids=batch_cond,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    t_cutoff=T_CUTOFF,
                    cfg_scale=CFG_SCALE,
                    return_aligned_features=True,
                    return_raw_hidden=True,
                )
                images = torch.clamp(images, 0, 1)

                if raw_hidden_features is not None and len(raw_hidden_features) > 0:
                    rh = raw_hidden_features[0]
                    if rh.dim() == 3:
                        rh = rh.mean(dim=1)  # (B, 256, 768) -> (B, 768)
                    cond_raw_hidden.append(rh.cpu())

            cond_images.append(images.cpu())
            remaining -= b

        result = {
            "images": torch.cat(cond_images, dim=0),
            "raw_hidden": torch.cat(cond_raw_hidden, dim=0),
            "condition": (cell_type_id, sirna_id),
        }
        torch.save(result, save_path)
        print(f"  [GPU {gpu_id}] Generated {spc} for cell={cell_type_id} sirna={sirna_id} "
              f"(imgs={result['images'].shape}, hidden={result['raw_hidden'].shape})", flush=True)

    del model
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-key", required=True)
    parser.add_argument("--use-repa", action="store_true")
    parser.add_argument("--repa-encoder", default="siglip")
    parser.add_argument("--output-dir", default="outputs/posthoc_alignment/diag")
    parser.add_argument("--samples-per-cond", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--real-features", default="outputs/real_rxrx1_dinov3_meanpatch/train_features.pt")
    return parser.parse_args()


def main():
    args = parse_args()

    gen_cache_dir = Path(args.output_dir) / args.checkpoint_key / "gen_cache"
    gen_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {args.checkpoint_key}")
    print(f"REPA: {args.use_repa}, Samples/cond: {args.samples_per_cond}")
    print(f"Output: {gen_cache_dir}")

    # Select 50 conditions
    selected_heldout, selected_seen = select_50_conditions(args.real_features)
    all_conditions = sorted(selected_heldout + selected_seen)
    print(f"Total conditions: {len(all_conditions)}")

    # Save condition list for reproducibility
    torch.save({
        "heldout": selected_heldout,
        "seen": selected_seen,
        "all": all_conditions,
    }, gen_cache_dir.parent / "selected_conditions.pt")

    # Check which conditions still need generation
    todo_conds = []
    for cond in all_conditions:
        cond_str = f"{cond[0]}_{cond[1]}"
        if not (gen_cache_dir / f"cond_{cond_str}.pt").exists():
            todo_conds.append(cond)

    if not todo_conds:
        print(f"All {len(all_conditions)} conditions already cached!")
        return

    print(f"Generating {len(todo_conds)} conditions...")

    # Multi-GPU dispatch
    n_gpus = torch.cuda.device_count()
    print(f"Using {n_gpus} GPUs")

    chunks = [[] for _ in range(n_gpus)]
    for i, cond in enumerate(todo_conds):
        chunks[i % n_gpus].append(cond)

    mp.set_start_method("spawn", force=True)
    processes = []
    for gpu_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        p = mp.Process(
            target=generate_for_conditions,
            args=(gpu_id, chunk, args, args.samples_per_cond, args.batch_size, gen_cache_dir),
        )
        p.start()
        processes.append((gpu_id, p))

    for gpu_id, p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"ERROR: GPU {gpu_id} failed with exit code {p.exitcode}")
        else:
            print(f"GPU {gpu_id} done")

    # Verify
    n_cached = sum(1 for c in all_conditions
                   if (gen_cache_dir / f"cond_{c[0]}_{c[1]}.pt").exists())
    print(f"\nCached: {n_cached}/{len(all_conditions)} conditions")


if __name__ == "__main__":
    main()
