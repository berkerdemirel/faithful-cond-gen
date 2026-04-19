#!/usr/bin/env python
"""Profile FLOPs + runtime for REPA vs vanilla scoring.

This script supports two profiling modes:

1) FLOPs mode (--mode flops, default): Component-level FLOPs via fvcore
   - Generation total (denoise + decode)
   - Score-compute breakdown (SiT steps, projector, post-hoc encoder)

2) E2E mode (--mode e2e): Wall-clock end-to-end timing
   - Generation: actual sample() call timing
   - REPA score: single SiT+projector forward at t=0 (scoring overhead)
   - Vanilla score: VAE decode + post-hoc encoder

Examples:

  # FLOPs profiling (component breakdown)
  PYTHONPATH=src uv run python scripts/profile_flops.py \
    --checkpoint_key celeba_repa_full_v1 \
    --output outputs/flops/repa.yaml \
    --num_inference_steps 250

  # E2E wall-clock profiling
  PYTHONPATH=src uv run python scripts/profile_flops.py \
    --checkpoint_key celeba_repa_full_v1 \
    --mode e2e \
    --output outputs/flops/repa_e2e.yaml \
    --num_inference_steps 250

  # Compare vanilla with REPA (needs posthoc encoder for vanilla)
  PYTHONPATH=src uv run python scripts/profile_flops.py \
    --checkpoint_key celeba_vanilla_full_v1 \
    --posthoc_encoder_spec repa:dinov3 \
    --mode e2e \
    --output outputs/flops/vanilla_e2e.yaml

  # Compare two FLOPs YAMLs
  PYTHONPATH=src uv run python scripts/profile_flops.py --compare \
    outputs/flops/repa.yaml outputs/flops/vanilla.yaml

  # Run both modes
  PYTHONPATH=src uv run python scripts/profile_flops.py \
    --checkpoint_key celeba_repa_full_v1 \
    --mode both \
    --output outputs/flops/repa.yaml

Notes:
- For RxRx1, use repa:openphenom as the post-hoc encoder.
- Ensure HF models are cached locally for accurate runtime measurements.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml


def load_generator(checkpoint_key: str, device: torch.device):
    """Load your diffusion generator from a checkpoint key."""
    from faithful_cond_gen.utils.checkpoints import get_checkpoint, get_checkpoint_path

    config = get_checkpoint(checkpoint_key)
    ckpt_path = get_checkpoint_path(checkpoint_key)

    print(f"Loading checkpoint: {checkpoint_key}")
    print(f"  Path: {ckpt_path}")
    print(f"  Dataset: {config.get('dataset', 'unknown')}")
    print(f"  Model: {config.get('model', 'unknown')}")

    dataset = config.get("dataset", "celeba")

    from faithful_cond_gen.model.generator import GeneratorConfig, GeneratorWrapper

    if dataset == "celeba":
        gen_cfg = GeneratorConfig(
            image_size=256,
            in_channels=3,
            sit_arch="SiT-B/2",
            attr_num_classes=[2, 2, 2, 2],
            use_repa="repa" in config.get("model", ""),
            repa_encoder="dinov3" if "repa" in config.get("model", "") else "",
            repa_encoder_depth=8,
        )
        num_conditions = 4
        image_shape = None  # filled below
    elif dataset == "rxrx1":
        gen_cfg = GeneratorConfig(
            image_size=256,
            in_channels=6,
            sit_arch="SiT-B/2",
            attr_num_classes=[4, 1139],
            use_repa="repa" in config.get("model", ""),
            repa_encoder="dinov2" if "repa" in config.get("model", "") else "",
            repa_encoder_depth=8,
        )
        num_conditions = 2
        image_shape = None
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    generator = GeneratorWrapper(gen_cfg)

    from faithful_cond_gen.pl_modules.generator_pl import GeneratorPL

    pl_module = GeneratorPL.load_from_checkpoint(
        ckpt_path, generator=generator, map_location=device, strict=False
    )
    pl_module.to(device)
    pl_module.eval()

    if hasattr(pl_module, "ema"):
        pl_module.ema.apply()
        print("  Applied EMA weights")

    # Return generator, config, and dataset-derived shapes
    in_channels = int(gen_cfg.in_channels)
    image_shape = (1, in_channels, int(gen_cfg.image_size), int(gen_cfg.image_size))
    return pl_module.generator, config, dataset, image_shape, num_conditions


def load_encoder_from_spec(
    spec: str,
    device: torch.device,
    *,
    in_channels: int,
    resolution: int,
) -> torch.nn.Module:
    """Load a post-hoc encoder for profiling.

    Supported specs:
      - repa:<encoder_name>
          Uses faithful_cond_gen.model.repa_encoder.load_repa_encoder.
          This is the recommended path for profiling DINOv3/SigLIP/OpenPhenom
          with the same preprocessing you use in training.

      - timm:<model_name>
          Uses timm.create_model(pretrained=False, num_classes=0). (FLOPs ok; runtime not representative.)

      - module.path:callable_name
          Your callable returns nn.Module.

    """
    spec = spec.strip()

    if spec.startswith("repa:"):
        enc_name = spec.split(":", 1)[1]
        from faithful_cond_gen.model.repa_encoder import load_repa_encoder

        # REPAEncoder can accept dataset-native channels (e.g. 6ch RxRx1) and converts to RGB internally.
        model = load_repa_encoder(
            enc_name, resolution=resolution, in_channels=in_channels, device=str(device)
        )
        model.eval().to(device)
        return model

    if spec.startswith("timm:"):
        name = spec.split(":", 1)[1]
        import timm

        model = timm.create_model(name, pretrained=False, num_classes=0)
        model.eval().to(device)
        return model

    if ":" not in spec:
        raise ValueError(
            "Encoder spec must be repa:<name>, timm:<name>, or module.path:callable"
        )

    module_path, fn_name = spec.split(":", 1)
    mod = importlib.import_module(module_path)
    fn = getattr(mod, fn_name)
    model = fn()
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"{spec} did not return an nn.Module")
    model.eval().to(device)
    return model


def profile_e2e_checkpoint(
    *,
    checkpoint_key: str,
    output_path: Optional[str],
    num_inference_steps: int,
    cfg_scale: float,
    batch_size: int,
    warmup: int,
    runs: int,
    posthoc_encoder_spec: Optional[str],
    include_vae_decode_in_vanilla_score: bool,
) -> Dict[str, Any]:
    """End-to-end profiling with actual wall-clock measurements."""
    from faithful_cond_gen.utils.flops_profiler import profile_e2e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator, config, dataset, base_image_shape, num_conditions = load_generator(
        checkpoint_key, device
    )

    resolution = int(base_image_shape[2])
    in_channels = int(base_image_shape[1])
    is_repa = "repa" in str(config.get("model", ""))

    posthoc_encoder = None
    if posthoc_encoder_spec is not None:
        enc_in = in_channels if posthoc_encoder_spec.strip().startswith("repa:") else 3
        posthoc_encoder = load_encoder_from_spec(
            posthoc_encoder_spec,
            device,
            in_channels=enc_in,
            resolution=resolution,
        )

    print("\nE2E Profiling checkpoint...")
    print(f"  checkpoint_key: {checkpoint_key}")
    print(f"  dataset:        {dataset}")
    print(f"  model_type:     {'REPA' if is_repa else 'Vanilla'}")
    print(f"  batch_size:     {batch_size}")
    print(f"  steps:          {num_inference_steps}")
    print(f"  cfg_scale:      {cfg_scale}")
    print(f"  warmup/runs:    {warmup}/{runs}")

    prof = profile_e2e(
        generator,
        batch_size=batch_size,
        num_conditions=num_conditions,
        device=device,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        warmup=warmup,
        runs=runs,
        posthoc_encoder=posthoc_encoder,
        include_vae_decode_in_vanilla=include_vae_decode_in_vanilla_score,
    )

    results: Dict[str, Any] = {
        "checkpoint_key": checkpoint_key,
        "config": config,
        "mode": "e2e",
        "profiling_config": {
            "num_inference_steps": int(num_inference_steps),
            "cfg_scale": float(cfg_scale),
            "batch_size": int(batch_size),
            "device": str(device),
            "posthoc_encoder_spec": posthoc_encoder_spec,
            "include_vae_decode_in_vanilla_score": bool(
                include_vae_decode_in_vanilla_score
            ),
        },
        "e2e_profile": prof.to_dict(),
    }

    # Console summary with CIs
    def fmt_ci(mean: float, ci: tuple) -> str:
        margin = (ci[1] - ci[0]) / 2
        return f"{mean:.2f} ± {margin:.2f}"

    print("\n" + "=" * 80)
    print("E2E SUMMARY (95% CI)")
    print("=" * 80)
    print(
        f"Generation:       {fmt_ci(prof.generation_time_ms, prof.generation_ci)} ms/batch, {prof.generation_images_per_sec:.1f} img/s"
    )
    if is_repa:
        print(
            f"REPA score:       {fmt_ci(prof.repa_score_time_ms, prof.repa_score_ci)} ms (+{prof.repa_score_overhead_pct:.1f}% overhead)"
        )
    if posthoc_encoder is not None:
        print(
            f"Vanilla score:    {fmt_ci(prof.vanilla_score_time_ms, prof.vanilla_score_ci)} ms (encoder: {prof.vanilla_posthoc_encoder_time_ms:.2f} ms)"
        )
    print("=" * 80)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            yaml.safe_dump(results, f, sort_keys=False)
        print(f"Saved YAML: {out}")

    return results


def profile_checkpoint(
    *,
    checkpoint_key: str,
    output_path: Optional[str],
    num_inference_steps: int,
    cfg_scale: float,
    batch_size: int,
    posthoc_encoder_spec: Optional[str],
    projector_calls_for_scoring: int,
    allow_partial_flops: bool,
    include_vae_encode_in_score: bool,
    include_vae_decode_in_repa_score: bool,
    include_vae_decode_in_vanilla_score: bool,
) -> Dict[str, Any]:
    """FLOPs-based profiling with component breakdown."""
    from faithful_cond_gen.utils.flops_profiler import (
        compute_score_compute_totals,
        profile_sampling_pipeline,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator, config, dataset, base_image_shape, num_conditions = load_generator(
        checkpoint_key, device
    )

    # Override batch size for profiling
    image_shape = (
        batch_size,
        base_image_shape[1],
        base_image_shape[2],
        base_image_shape[3],
    )
    resolution = int(base_image_shape[2])

    is_repa = "repa" in str(config.get("model", ""))

    posthoc_encoder = None
    posthoc_image_shape: Optional[Tuple[int, int, int, int]] = None
    if posthoc_encoder_spec is not None:
        enc_in = (
            int(image_shape[1])
            if posthoc_encoder_spec.strip().startswith("repa:")
            else 3
        )
        posthoc_image_shape = (batch_size, enc_in, resolution, resolution)
        posthoc_encoder = load_encoder_from_spec(
            posthoc_encoder_spec,
            device,
            in_channels=enc_in,
            resolution=resolution,
        )

    print("\nFLOPs Profiling checkpoint...")
    print(f"  checkpoint_key: {checkpoint_key}")
    print(f"  dataset:        {dataset}")
    print(f"  model_type:     {'REPA' if is_repa else 'Vanilla'}")
    print(f"  image_shape:    {image_shape}")
    print(f"  steps:          {num_inference_steps}")
    print(f"  cfg_scale:      {cfg_scale}")

    prof = profile_sampling_pipeline(
        generator=generator,
        image_shape=image_shape,
        num_conditions=num_conditions,
        device=device,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        posthoc_encoder=posthoc_encoder,
        profile_posthoc=(posthoc_encoder is not None),
        posthoc_image_shape=posthoc_image_shape,
        allow_partial_flops=allow_partial_flops,
    )

    # Compute score-compute totals for the intended accounting
    if is_repa:
        score = compute_score_compute_totals(
            prof,
            mode="repa",
            include_vae_encode=include_vae_encode_in_score,
            include_vae_decode=include_vae_decode_in_repa_score,
            include_posthoc_encoder=False,
            projector_calls_for_scoring=projector_calls_for_scoring,
        )
        score["mode"] = "repa"
    else:
        if posthoc_encoder is None:
            score = {
                "mode": "vanilla_posthoc",
                "error": "Vanilla scoring requires --posthoc_encoder_spec to profile post-hoc encoder FLOPs.",
            }
        else:
            score = compute_score_compute_totals(
                prof,
                mode="vanilla_posthoc",
                include_vae_encode=include_vae_encode_in_score,
                include_vae_decode=include_vae_decode_in_vanilla_score,
                include_posthoc_encoder=True,
            )
            score["mode"] = "vanilla_posthoc"

    results: Dict[str, Any] = {
        "checkpoint_key": checkpoint_key,
        "config": config,
        "profiling_config": {
            "num_inference_steps": int(num_inference_steps),
            "cfg_scale": float(cfg_scale),
            "batch_size": int(batch_size),
            "device": str(device),
            "projector_calls_for_scoring": int(projector_calls_for_scoring),
            "include_vae_encode_in_score": bool(include_vae_encode_in_score),
            "include_vae_decode_in_repa_score": bool(include_vae_decode_in_repa_score),
            "include_vae_decode_in_vanilla_score": bool(
                include_vae_decode_in_vanilla_score
            ),
            "posthoc_encoder_spec": posthoc_encoder_spec,
            "allow_partial_flops": bool(allow_partial_flops),
        },
        "profile": prof.to_dict(),
        "score_compute": score,
    }

    # Console summary
    print("\n" + "=" * 80)
    print("FLOPS SUMMARY")
    print("=" * 80)
    print(f"Generation total: {prof.total_generation_flops/1e9:.2f} GFLOPs")
    if "error" not in score:
        print(f"Score-compute:    {score['gflops']:.2f} GFLOPs")
    else:
        print(f"Score-compute:    ERROR: {score['error']}")
    print("=" * 80)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            yaml.safe_dump(results, f, sort_keys=False)
        print(f"Saved YAML: {out}")

    return results


def compare_profiles(p1_path: str, p2_path: str):
    """Compare two FLOPs profiles (expects one REPA and one vanilla)."""
    with open(p1_path) as f:
        p1 = yaml.safe_load(f)
    with open(p2_path) as f:
        p2 = yaml.safe_load(f)

    def is_repa(p: Dict[str, Any]) -> bool:
        return "repa" in str(p.get("config", {}).get("model", ""))

    repa = p1 if is_repa(p1) else p2
    vanilla = p2 if repa is p1 else p1

    repa_score = repa.get("score_compute", {})
    van_score = vanilla.get("score_compute", {})

    if "error" in repa_score or "error" in van_score:
        raise ValueError(
            f"Cannot compare: repa_error={repa_score.get('error')} vanilla_error={van_score.get('error')}"
        )

    flops_savings = float(van_score["flops"]) - float(repa_score["flops"])
    flops_savings_pct = (
        (flops_savings / float(van_score["flops"])) * 100.0
        if float(van_score["flops"]) > 0
        else 0.0
    )

    print("=" * 70)
    print("SCORE-COMPUTE FLOPS COMPARISON")
    print("=" * 70)
    print(f"REPA:    {repa.get('checkpoint_key', p1_path)}")
    print(f"Vanilla: {vanilla.get('checkpoint_key', p2_path)}")
    print()
    print(f"REPA score-compute:    {float(repa_score['flops'])/1e9:10.2f} GFLOPs")
    print(f"Vanilla+posthoc score: {float(van_score['flops'])/1e9:10.2f} GFLOPs")
    print(f"Savings:               {flops_savings/1e9:10.2f} GFLOPs ({flops_savings_pct:.1f}%)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Profile FLOPs/runtime for REPA vs vanilla scoring"
    )

    parser.add_argument("--checkpoint_key", type=str, help="Checkpoint key to profile")
    parser.add_argument(
        "--output", type=str, default=None, help="Write YAML results to this path"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["flops", "e2e", "both"],
        default="flops",
        help="Profiling mode: 'flops' (component FLOPs), 'e2e' (wall-clock), or 'both'",
    )

    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("PROFILE1", "PROFILE2"),
        help="Compare two FLOPs YAML profiles (expects one REPA and one vanilla)",
    )

    parser.add_argument("--num_inference_steps", type=int, default=250)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations (e2e mode only)")
    parser.add_argument("--runs", type=int, default=100, help="Timing runs (e2e mode only)")

    parser.add_argument(
        "--posthoc_encoder_spec",
        type=str,
        default=None,
        help="Post-hoc encoder spec (repa:<name>, timm:<name>, or module:callable). Required for vanilla score totals.",
    )

    parser.add_argument(
        "--include_vae_encode_in_score",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--include_vae_decode_in_repa_score",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--include_vae_decode_in_vanilla_score",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--allow_partial_flops",
        action="store_true",
        help="Allow partial fvcore FLOP counting when unsupported ops exist (not recommended for final numbers).",
    )

    parser.add_argument(
        "--projector_calls_for_scoring",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    if args.compare:
        compare_profiles(args.compare[0], args.compare[1])
        return

    if not args.checkpoint_key:
        parser.error(
            "Provide --checkpoint_key to profile, or --compare to compare two YAMLs"
        )

    if args.mode in ("flops", "both"):
        profile_checkpoint(
            checkpoint_key=args.checkpoint_key,
            output_path=args.output if args.mode == "flops" else None,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            batch_size=args.batch_size,
            posthoc_encoder_spec=args.posthoc_encoder_spec,
            projector_calls_for_scoring=args.projector_calls_for_scoring,
            allow_partial_flops=args.allow_partial_flops,
            include_vae_encode_in_score=args.include_vae_encode_in_score,
            include_vae_decode_in_repa_score=args.include_vae_decode_in_repa_score,
            include_vae_decode_in_vanilla_score=args.include_vae_decode_in_vanilla_score,
        )

    if args.mode in ("e2e", "both"):
        e2e_output = args.output
        if args.mode == "both" and args.output:
            # Use different output path for e2e when running both
            base = Path(args.output)
            e2e_output = str(base.parent / f"{base.stem}_e2e{base.suffix}")

        profile_e2e_checkpoint(
            checkpoint_key=args.checkpoint_key,
            output_path=e2e_output,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            batch_size=args.batch_size,
            warmup=args.warmup,
            runs=args.runs,
            posthoc_encoder_spec=args.posthoc_encoder_spec,
            include_vae_decode_in_vanilla_score=args.include_vae_decode_in_vanilla_score,
        )


if __name__ == "__main__":
    main()
