"""Generate per-cond shards for the canonical rxrx1 50-cond eval subset.

Sibling of ``gen_rxrx1_posthoc_eval_balanced50.py``. Conditions are loaded from
``outputs/posthoc_alignment/rxrx1_eval_subset_final.json`` and already-cached
per-cond shards under ``diag_subset/{key}/gen_cache/`` are skipped, so this
script is safe to rerun and only fills the gap.

Generation protocol is identical to the balanced50 sibling (t_cutoff=0.04,
CFG=1.0, BS=24, 250 steps, 200 samples/cond, multi-GPU spawn dispatch) so the
same trained mapper and downstream KID comparisons still apply.

Usage:
    PYTHONPATH=src uv run python \\
        scripts/posthoc_alignment/gen_rxrx1_posthoc_eval_subset.py \\
        --checkpoint-key rxrx1_vanilla_marginal_v1

    PYTHONPATH=src uv run python \\
        scripts/posthoc_alignment/gen_rxrx1_posthoc_eval_subset.py \\
        --checkpoint-key rxrx1_repa_siglip_marginal_v1 \\
        --use-repa --repa-encoder siglip
"""

import argparse
import json
from pathlib import Path

import torch
import torch.multiprocessing as mp

from faithful_cond_gen.model.generator import GeneratorConfig, GeneratorWrapper
from faithful_cond_gen.pl_modules.generator_pl import GeneratorPL
from faithful_cond_gen.utils.checkpoints import get_checkpoint_path

NUM_INFERENCE_STEPS = 250
T_CUTOFF = 0.04
CFG_SCALE = 1.0
BATCH_SIZE = 24
SEED = 42

REPO = Path(__file__).resolve().parents[2]
SUBSET_JSON = REPO / "outputs/posthoc_alignment/rxrx1_eval_subset_final.json"


def load_subset_conditions():
    with open(SUBSET_JSON) as f:
        payload = json.load(f)
    rows = payload["seen"] + payload["unseen"]
    return sorted({(int(r["cell_type_id"]), int(r["sirna_id"])) for r in rows})


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
                        rh = rh.mean(dim=1)
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
    parser.add_argument("--output-dir", default="outputs/posthoc_alignment/diag_subset")
    parser.add_argument("--samples-per-cond", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()

    gen_cache_dir = Path(args.output_dir) / args.checkpoint_key / "gen_cache"
    gen_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {args.checkpoint_key}")
    print(f"REPA: {args.use_repa}, Samples/cond: {args.samples_per_cond}")
    print(f"Output: {gen_cache_dir}")

    all_conditions = load_subset_conditions()
    print(f"Total subset conditions: {len(all_conditions)}")

    torch.save({
        "all": all_conditions,
        "source": str(SUBSET_JSON.relative_to(REPO)),
    }, gen_cache_dir.parent / "selected_conditions.pt")

    todo_conds = []
    for cond in all_conditions:
        cond_str = f"{cond[0]}_{cond[1]}"
        if not (gen_cache_dir / f"cond_{cond_str}.pt").exists():
            todo_conds.append(cond)

    if not todo_conds:
        print(f"All {len(all_conditions)} conditions already cached!")
        return

    print(f"Generating {len(todo_conds)} conditions (skipping {len(all_conditions) - len(todo_conds)} already cached)")

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

    n_cached = sum(1 for c in all_conditions
                   if (gen_cache_dir / f"cond_{c[0]}_{c[1]}.pt").exists())
    print(f"\nCached: {n_cached}/{len(all_conditions)} conditions")


if __name__ == "__main__":
    main()
