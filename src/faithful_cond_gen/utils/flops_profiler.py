"""FLOPs + runtime profiling utilities for REPA vs vanilla scoring.

This module is designed to support paper-quality accounting without hand-written
FLOP *estimates*.

- FLOPs are computed via fvcore.nn.FlopCountAnalysis.
- Runtime is measured with CUDA events when running on GPU.

It profiles the following components:
  - VAE encode
  - VAE decode
  - SiT backbone forward (one denoising step)
  - REPA projector MLP (if present)
  - Optional post-hoc encoder on pixels (e.g., DINOv3, SigLIP, OpenPhenom)

The key distinction is between:
  (A) Full generation cost (denoise + decode)
  (B) Score-compute cost for rejection/filtering
      - REPA score: (optional VAE encode) + denoise + projector (+ optional decode)
      - Vanilla post-hoc score: (optional VAE encode) + denoise + decode + post-hoc encoder

All totals are computed from per-component measurements.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from fvcore.nn.jit_handles import get_shape


@dataclass
class ProfileResult:
    name: str
    flops: int = 0
    params: int = 0
    calls: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def gflops(self) -> float:
        return float(self.flops) / 1e9

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "flops": int(self.flops),
            "gflops": float(self.gflops),
            "params": int(self.params),
            "calls": int(self.calls),
            "extra": dict(self.extra),
        }


@dataclass
class SamplingProfile:
    num_inference_steps: int
    cfg_scale: float
    batch_size: int

    vae_encode: Optional[ProfileResult] = None
    vae_decode: Optional[ProfileResult] = None
    sit_forward_per_step_no_zs: Optional[ProfileResult] = None
    sit_forward_per_step_with_zs: Optional[ProfileResult] = None
    repa_projector: Optional[ProfileResult] = None  # optional diagnostic only
    posthoc_encoder: Optional[ProfileResult] = None

    total_generation_flops: int = 0

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "config": {
                "num_inference_steps": int(self.num_inference_steps),
                "cfg_scale": float(self.cfg_scale),
                "batch_size": int(self.batch_size),
            },
            "results": {},
            "totals": {
                "generation_flops": int(self.total_generation_flops),
                "generation_gflops": float(self.total_generation_flops) / 1e9,
                "flops_convention": "fvcore",
            },
        }
        if self.vae_encode:
            out["results"]["vae_encode"] = self.vae_encode.to_dict()
        if self.vae_decode:
            out["results"]["vae_decode"] = self.vae_decode.to_dict()
        if self.sit_forward_per_step_no_zs:
            out["results"][
                "sit_forward_per_step_no_zs"
            ] = self.sit_forward_per_step_no_zs.to_dict()
        if self.sit_forward_per_step_with_zs:
            out["results"][
                "sit_forward_per_step_with_zs"
            ] = self.sit_forward_per_step_with_zs.to_dict()
        if self.repa_projector:
            out["results"]["repa_projector"] = self.repa_projector.to_dict()
        if self.posthoc_encoder:
            out["results"]["posthoc_encoder"] = self.posthoc_encoder.to_dict()
        return out


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _op_numel(outputs: Any) -> int:
    """Return total number of elements in (possibly nested) outputs."""
    if outputs is None:
        return 0
    if isinstance(outputs, torch.Tensor):
        return int(outputs.numel())
    if isinstance(outputs, (list, tuple)):
        return sum(_op_numel(o) for o in outputs)
    if isinstance(outputs, dict):
        return sum(_op_numel(v) for v in outputs.values())
    return 0


# Treat elementwise trig as 1 flop per output element (reasonable convention)
def _value_numel(v) -> int:
    # v is a torch._C.Value
    shape = get_shape(v)
    if shape is None:
        return 0
    n = 1
    for s in shape:
        if s is None:
            return 0
        n *= int(s)
    return int(n)


def _elemwise_1_flop(inputs, outputs):
    # outputs is usually a list/tuple of torch._C.Value
    v = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    return _value_numel(v)


def _zero_flop(*args, **kwargs):
    return 0


# Embedding: gather is memory-bound; count 0 FLOPs (or 1 per output element if you prefer)
def _embedding_flops(inputs, outputs):
    out = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    return 0  # safest: do not pretend this is compute


def _elementwise_flops(inputs: Tuple[Any, ...], outputs: Any) -> int:
    # Count 1 FLOP per output element for common pointwise ops.
    return _op_numel(outputs)


def _zero_flops(inputs: Tuple[Any, ...], outputs: Any) -> int:
    return 0


def _sdpa_flops(inputs: Tuple[Any, ...], outputs: Any) -> int:
    """FLOPs for scaled dot-product attention derived from tensor shapes.

    Counts only the two GEMMs:
      QK^T and (softmax(QK^T))V
    Ignores softmax/masking/dropout FLOPs (small relative to GEMMs).
    """
    if len(inputs) < 3:
        return 0
    q, k, v = inputs[0], inputs[1], inputs[2]
    if not (
        isinstance(q, torch.Tensor)
        and isinstance(k, torch.Tensor)
        and isinstance(v, torch.Tensor)
    ):
        return 0

    def parse(tq: torch.Tensor, tk: torch.Tensor):
        # Common shapes:
        #  - (B, H, L, D)
        #  - (B, L, H, D)
        #  - (B, L, D)
        if tq.dim() == 4 and tk.dim() == 4:
            if tq.shape[1] == tk.shape[1] and tq.shape[-1] == tk.shape[-1]:
                B = int(tq.shape[0])
                H = int(tq.shape[1])
                Lq = int(tq.shape[2])
                D = int(tq.shape[3])
                Lk = int(tk.shape[2])
                return B, H, Lq, Lk, D
            if tq.shape[2] == tk.shape[2] and tq.shape[-1] == tk.shape[-1]:
                B = int(tq.shape[0])
                H = int(tq.shape[2])
                Lq = int(tq.shape[1])
                D = int(tq.shape[3])
                Lk = int(tk.shape[1])
                return B, H, Lq, Lk, D
        if tq.dim() == 3 and tk.dim() == 3:
            B = int(tq.shape[0])
            H = 1
            Lq = int(tq.shape[1])
            D = int(tq.shape[2])
            Lk = int(tk.shape[1])
            return B, H, Lq, Lk, D
        return 0, 0, 0, 0, 0

    B, H, Lq, Lk, D = parse(q, k)
    if B == 0:
        return 0

    # FLOPs for matmul: 2 * M * N * K
    qk = 2 * Lq * Lk * D
    av = 2 * Lq * Lk * D
    return int(B * H * (qk + av))


def _default_op_handles() -> Dict[str, Callable[[Tuple[Any, ...], Any], int]]:
    # Elementwise ops commonly flagged as unsupported by fvcore
    elementwise = {
        "aten::add": _elementwise_flops,
        "aten::sub": _elementwise_flops,
        "aten::mul": _elementwise_flops,
        "aten::div": _elementwise_flops,
        "aten::exp": _elementwise_flops,
        "aten::silu": _elementwise_flops,
        "aten::gelu": _elementwise_flops,
        "aten::relu": _elementwise_flops,
        "aten::tanh": _elementwise_flops,
        "aten::sqrt": _elementwise_flops,
        "aten::rsqrt": _elementwise_flops,
        "aten::sigmoid": _elementwise_flops,
        "aten::softmax": _elementwise_flops,
        "aten::_softmax": _elementwise_flops,
        "aten::native_layer_norm": _elementwise_flops,
        "aten::layer_norm": _elementwise_flops,
        "aten::sin": _elemwise_1_flop,
        "aten::cos": _elemwise_1_flop,
        "aten::embedding": _embedding_flops,
        "aten::clone": _zero_flop,
        "aten::meshgrid": _zero_flop,
        "aten::tile": _zero_flop,
        # in-place ops: count like elementwise
        "aten::sub_": _elemwise_1_flop,
        "aten::div_": _elemwise_1_flop,
        "aten::neg": _elemwise_1_flop,
    }

    sdpa = {
        "aten::scaled_dot_product_attention": _sdpa_flops,
        "aten::_scaled_dot_product_attention": _sdpa_flops,
        "aten::_scaled_dot_product_attention_math": _sdpa_flops,
        "aten::_scaled_dot_product_efficient_attention": _sdpa_flops,
        "aten::_scaled_dot_product_flash_attention": _sdpa_flops,
    }

    zero = {
        "aten::pad": _zero_flops,
        "aten::randn_like": _zero_flops,  # random number generation is not counted as FLOPs
        "aten::dropout": _zero_flops,
        "aten::native_dropout": _zero_flops,
    }

    out: Dict[str, Callable[[Tuple[Any, ...], Any], int]] = {}
    out.update(elementwise)
    out.update(sdpa)
    out.update(zero)
    return out


def _fvcore_flops(
    module: nn.Module, inputs: Tuple[torch.Tensor, ...]
) -> Tuple[int, Dict[str, int], int]:
    import logging

    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as e:
        raise ImportError(
            "fvcore is required for FLOP counting. Install with: pip install fvcore"
        ) from e

    # Suppress "submodules never called during trace" warnings from fvcore
    # fvcore uses logging, not warnings module
    fvcore_logger = logging.getLogger("fvcore.nn.jit_analysis")
    old_level = fvcore_logger.level
    fvcore_logger.setLevel(logging.ERROR)

    try:
        analyzer = FlopCountAnalysis(module, inputs)
        analyzer.set_op_handle(**_default_op_handles())

        total = int(analyzer.total())
        unsupported = analyzer.unsupported_ops()
        uncalled = analyzer.uncalled_modules()
    finally:
        fvcore_logger.setLevel(old_level)

    return total, unsupported, len(uncalled)


def profile_module_flops(
    module: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    *,
    allow_partial: bool = False,
) -> ProfileResult:
    params = _count_params(module)
    try:
        flops, unsupported, uncalled_n = _fvcore_flops(module, inputs)
        if unsupported and not allow_partial:
            raise RuntimeError(
                f"fvcore could not fully analyze {module.__class__.__name__}. "
                f"unsupported_ops={list(unsupported.keys())[:8]} uncalled_modules={uncalled_n}."
            )
    except Exception as e:
        if not allow_partial:
            raise
        warnings.warn(
            f"Proceeding with partial FLOPs for {module.__class__.__name__}: {e}",
            UserWarning,
        )
        flops, unsupported, uncalled_n = 0, {}, 0

    return ProfileResult(
        name=module.__class__.__name__,
        flops=int(flops),
        params=int(params),
        extra={
            "flops_source": "fvcore",
            "unsupported_ops": list(unsupported.keys()) if unsupported else [],
            "uncalled_modules": int(uncalled_n),
        },
    )


def profile_time(
    fn: Callable[[], Any],
    *,
    warmup: int,
    runs: int,
    device: torch.device,
    measure_memory: bool = True,
    ci_level: float = 0.95,
) -> Tuple[float, Dict[str, Any]]:
    """Profile timing with confidence intervals.

    Returns:
        (mean_ms, extra) where extra contains std_ms, ci_lower_ms, ci_upper_ms
    """
    import math

    for _ in range(max(0, int(warmup))):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    runs = max(1, int(runs))
    times_ms = []

    extra: Dict[str, Any] = {
        "timing_method": "cuda_event" if device.type == "cuda" else "perf_counter",
        "timing_warmup": int(warmup),
        "timing_runs": runs,
    }

    if device.type == "cuda":
        if measure_memory:
            torch.cuda.reset_peak_memory_stats(device)

        # Measure each run individually for CI computation
        for _ in range(runs):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            fn()
            end_evt.record()
            torch.cuda.synchronize(device)
            times_ms.append(float(start_evt.elapsed_time(end_evt)))

        if measure_memory:
            extra["peak_mem_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
    else:
        for _ in range(runs):
            start = time.perf_counter()
            fn()
            elapsed = (time.perf_counter() - start) * 1000.0
            times_ms.append(elapsed)

    # Compute statistics
    mean_ms = sum(times_ms) / len(times_ms)
    if len(times_ms) > 1:
        variance = sum((t - mean_ms) ** 2 for t in times_ms) / (len(times_ms) - 1)
        std_ms = math.sqrt(variance)
        # t-distribution critical value approximation for 95% CI
        # For large n, use 1.96; for smaller n, use approximation
        if runs >= 30:
            t_crit = 1.96
        else:
            # Approximate t-critical for common CI levels
            t_crit = 2.0 + 3.0 / runs  # rough approximation
        ci_margin = t_crit * std_ms / math.sqrt(runs)
        ci_lower = mean_ms - ci_margin
        ci_upper = mean_ms + ci_margin
    else:
        std_ms = 0.0
        ci_lower = mean_ms
        ci_upper = mean_ms

    extra["std_ms"] = float(std_ms)
    extra["ci_lower_ms"] = float(ci_lower)
    extra["ci_upper_ms"] = float(ci_upper)
    extra["ci_level"] = float(ci_level)

    return mean_ms, extra


def _unwrap_generator(module: nn.Module) -> nn.Module:
    return module.generator if hasattr(module, "generator") else module


def _get_backbone(module: nn.Module) -> nn.Module:
    # GeneratorWrapper has diffusion_backbone; PL module wraps a generator.
    if hasattr(module, "diffusion_backbone"):
        return module.diffusion_backbone
    if hasattr(module, "generator") and hasattr(module.generator, "diffusion_backbone"):
        return module.generator.diffusion_backbone
    return module


def profile_sit_forward_per_step(
    generator: nn.Module,
    latent_shape: Tuple[int, int, int, int],
    num_conditions: int,
    device: torch.device,
    *,
    allow_partial_flops: bool,
    return_zs: bool,
) -> ProfileResult:
    B, _, _, _ = latent_shape
    x = torch.randn(latent_shape, device=device)
    t = torch.rand(B, device=device)
    cond_ids = torch.zeros(B, num_conditions, dtype=torch.long, device=device)

    backbone = _get_backbone(generator).eval().to(device)

    class _WrappedStep(nn.Module):
        def __init__(self, backbone, return_zs):
            super().__init__()
            self.backbone = backbone
            self.return_zs = return_zs

        def forward(self, x, t, cond_ids):
            eps, zs = self.backbone(x, t, cond_ids, return_zs=self.return_zs)
            if self.return_zs and zs is None:
                raise RuntimeError("return_zs=True but zs is None")
            return eps

    wrapped = _WrappedStep(backbone, return_zs=return_zs).eval().to(device)

    fl = profile_module_flops(
        wrapped, (x, t, cond_ids), allow_partial=allow_partial_flops
    )
    fl.name = f"SiT Forward (per step, return_zs={return_zs})"
    return fl


def profile_vae_encode_decode(
    vae: nn.Module,
    image_shape: Tuple[int, int, int, int],
    device: torch.device,
    *,
    allow_partial_flops: bool,
) -> Tuple[ProfileResult, ProfileResult]:
    """Profile VAE encode + decode FLOPs using tensor-returning wrappers.

    We avoid calling `latent_dist.sample()` to prevent `aten::randn_like` from
    appearing in the trace, and we wrap only the encoder/decoder submodules to
    avoid thousands of "uncalled" modules from the unused half of the VAE.
    """
    B, C, H, W = image_shape
    vae = vae.eval().to(device)

    images = torch.rand(image_shape, device=device)

    # ---------------- Encode-only wrapper ----------------
    class _EncodeOnly(nn.Module):
        def __init__(self, inner: nn.Module):
            super().__init__()
            # Diffusers AutoencoderKL style
            if hasattr(inner, "encoder") and hasattr(inner, "quant_conv"):
                self.encoder = inner.encoder
                self.quant_conv = inner.quant_conv
                self._mode = "direct"
            else:
                self.inner = inner
                self._mode = "encode"

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self._mode == "direct":
                h = self.encoder(x)
                # Some encoders return a tuple
                if isinstance(h, (tuple, list)):
                    h = h[0]
                moments = self.quant_conv(h)
                return moments
            # Fallback: return mean (not sample) if possible
            out = self.inner.encode(x)
            # Preferred: latent_dist.mean
            if hasattr(out, "latent_dist"):
                dist = out.latent_dist
                if hasattr(dist, "mean") and dist.mean is not None:
                    return dist.mean
                # If mean missing, mode() is deterministic but MUST be called
                if hasattr(dist, "mode") and callable(dist.mode):
                    return dist.mode()  # NOTE parentheses
                # As last resort, sample() is callable; but introduces randn_like
                if hasattr(dist, "sample") and callable(dist.sample):
                    return dist.sample()

            # Some VAEs return an object with .mean directly
            if hasattr(out, "mean") and torch.is_tensor(out.mean):
                return out.mean

            # Some return a tensor directly
            if torch.is_tensor(out):
                return out

            raise TypeError(f"VAE encode returned unsupported type: {type(out)}")

    enc = _EncodeOnly(vae).eval().to(device)
    enc_fl = profile_module_flops(enc, (images,), allow_partial=allow_partial_flops)
    enc_fl.name = "VAE Encode"
    enc_fl.params = _count_params(vae)

    # ---------------- Decode-only wrapper ----------------
    latent_size = H // 8
    latent_channels = 24 if C == 6 else 4
    latents = torch.randn(B, latent_channels, latent_size, latent_size, device=device)

    class _DecodeOnly(nn.Module):
        def __init__(self, inner: nn.Module):
            super().__init__()
            if hasattr(inner, "decoder") and hasattr(inner, "post_quant_conv"):
                self.post_quant_conv = inner.post_quant_conv
                self.decoder = inner.decoder
                self._mode = "direct"
            else:
                self.inner = inner
                self._mode = "decode"

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            if self._mode == "direct":
                z = self.post_quant_conv(z)
                out = self.decoder(z)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                return out
            out = self.inner.decode(z)
            if hasattr(out, "sample") and torch.is_tensor(out.sample):
                return out.sample
            if torch.is_tensor(out):
                return out
            if (
                isinstance(out, (tuple, list))
                and len(out) > 0
                and torch.is_tensor(out[0])
            ):
                return out[0]
            raise TypeError(f"VAE decode returned unsupported type: {type(out)}")

    dec = _DecodeOnly(vae).eval().to(device)
    dec_fl = profile_module_flops(dec, (latents,), allow_partial=allow_partial_flops)
    dec_fl.name = "VAE Decode"
    dec_fl.params = _count_params(vae)

    return enc_fl, dec_fl


def profile_repa_projector(
    generator: nn.Module,
    latent_shape: Tuple[int, int, int, int],
    device: torch.device,
    *,
    allow_partial_flops: bool,
) -> Optional[ProfileResult]:
    backbone = _get_backbone(generator)
    if (
        not hasattr(backbone, "projectors")
        or backbone.projectors is None
        or len(backbone.projectors) == 0
    ):
        return None

    projector = backbone.projectors[0].eval().to(device)

    # Approximate real call shape: projector consumes (B * num_patches, hidden_size)
    B, _, H, _ = latent_shape
    patch = int(getattr(backbone, "patch_size", 2))
    hidden = (
        int(backbone.pos_embed.shape[-1]) if hasattr(backbone, "pos_embed") else 768
    )
    num_patches = (int(H) // patch) ** 2
    x2d = torch.randn(B * num_patches, hidden, device=device)

    fl = profile_module_flops(projector, (x2d,), allow_partial=allow_partial_flops)
    fl.name = "REPA Projector"
    return fl


def profile_posthoc_encoder(
    encoder: nn.Module,
    image_shape: Tuple[int, int, int, int],
    device: torch.device,
    *,
    allow_partial_flops: bool,
) -> ProfileResult:
    x = torch.randn(image_shape, device=device)
    encoder = encoder.eval().to(device)

    fl = profile_module_flops(encoder, (x,), allow_partial=allow_partial_flops)
    fl.name = "Post-hoc Encoder"
    return fl


def profile_sampling_pipeline(
    generator: nn.Module,
    *,
    image_shape: Tuple[int, int, int, int],
    num_conditions: int,
    device: torch.device,
    num_inference_steps: int,
    cfg_scale: float,
    posthoc_encoder: Optional[nn.Module],
    profile_posthoc: bool,
    posthoc_image_shape: Optional[Tuple[int, int, int, int]] = None,
    allow_partial_flops: bool,
) -> SamplingProfile:
    gen = _unwrap_generator(generator).eval().to(device)

    B, C, H, W = image_shape
    latent_size = H // 8
    latent_channels = 24 if C == 6 else 4
    latent_shape = (B, latent_channels, latent_size, latent_size)

    vae_enc, vae_dec = profile_vae_encode_decode(
        gen.vae,
        image_shape,
        device,
        allow_partial_flops=allow_partial_flops,
    )

    sit_no_zs = profile_sit_forward_per_step(
        gen,
        latent_shape,
        num_conditions,
        device,
        allow_partial_flops=allow_partial_flops,
        return_zs=False,
    )

    backbone = _get_backbone(gen)
    has_repa = bool(getattr(backbone, "use_repa", False)) and (
        getattr(backbone, "projectors", None) is not None
    )

    sit_with_zs = None
    if has_repa:
        sit_with_zs = profile_sit_forward_per_step(
            gen,
            latent_shape,
            num_conditions,
            device,
            allow_partial_flops=allow_partial_flops,
            return_zs=True,
        )

    proj = profile_repa_projector(
        gen,
        latent_shape,
        device,
        allow_partial_flops=allow_partial_flops,
    )

    enc_prof: Optional[ProfileResult] = None
    if profile_posthoc:
        if posthoc_encoder is None:
            raise ValueError("profile_posthoc=True but posthoc_encoder is None")
        enc_shape = (
            posthoc_image_shape if posthoc_image_shape is not None else image_shape
        )
        enc_prof = profile_posthoc_encoder(
            posthoc_encoder,
            enc_shape,
            device,
            allow_partial_flops=allow_partial_flops,
        )
        if posthoc_image_shape is not None:
            enc_prof.extra["profiled_image_shape"] = tuple(int(x) for x in enc_shape)

    cfg_mult = 2 if float(cfg_scale) != 1.0 else 1

    # Full generation: denoise + decode (projector not part of standard sampling)
    total_gen_flops = (
        sit_no_zs.flops * int(num_inference_steps) * cfg_mult + vae_dec.flops
    )

    return SamplingProfile(
        num_inference_steps=int(num_inference_steps),
        cfg_scale=float(cfg_scale),
        batch_size=int(B),
        vae_encode=vae_enc,
        vae_decode=vae_dec,
        sit_forward_per_step_no_zs=sit_no_zs,
        sit_forward_per_step_with_zs=sit_with_zs,
        repa_projector=proj,
        posthoc_encoder=enc_prof,
        total_generation_flops=int(total_gen_flops),
    )


def compute_score_compute_totals(
    profile: SamplingProfile,
    *,
    mode: str,  # "repa" or "vanilla_posthoc"
    include_vae_encode: bool,
    include_vae_decode: bool,
    include_posthoc_encoder: bool,
    projector_calls_for_scoring: int = 1,
) -> Dict[str, Any]:
    """Compute total FLOPs for score computation (REPA or vanilla+posthoc)."""
    cfg_mult = 2 if float(profile.cfg_scale) != 1.0 else 1

    if profile.sit_forward_per_step_no_zs is None:
        raise ValueError("Missing sit_forward_per_step_no_zs")

    total_flops = 0
    parts: Dict[str, Any] = {
        "mode": mode,
        "cfg_multiplier": cfg_mult,
        "num_inference_steps": int(profile.num_inference_steps),
    }

    if mode == "repa":
        if profile.sit_forward_per_step_with_zs is None:
            raise ValueError("REPA mode requires sit_forward_per_step_with_zs")

        n = int(profile.num_inference_steps)
        k = int(projector_calls_for_scoring)
        if not (0 <= k <= n):
            raise ValueError(
                f"projector_calls_for_scoring must be in [0, {n}], got {k}"
            )

        no_zs_calls = n - k
        with_zs_calls = k

        sit_flops = (
            profile.sit_forward_per_step_no_zs.flops * no_zs_calls * cfg_mult
            + profile.sit_forward_per_step_with_zs.flops * with_zs_calls * cfg_mult
        )

        parts["projector_calls_for_scoring"] = int(k)
        parts["sit_no_zs_calls"] = int(no_zs_calls)
        parts["sit_with_zs_calls"] = int(with_zs_calls)
        parts["sit_no_zs_total_flops"] = int(
            profile.sit_forward_per_step_no_zs.flops * no_zs_calls * cfg_mult
        )
        parts["sit_with_zs_total_flops"] = int(
            profile.sit_forward_per_step_with_zs.flops * with_zs_calls * cfg_mult
        )

    elif mode == "vanilla_posthoc":
        sit_flops = (
            profile.sit_forward_per_step_no_zs.flops
            * profile.num_inference_steps
            * cfg_mult
        )
        parts["sit_no_zs_total_flops"] = int(sit_flops)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    total_flops += sit_flops

    if include_vae_encode and profile.vae_encode is not None:
        total_flops += profile.vae_encode.flops
        parts["vae_encode_flops"] = int(profile.vae_encode.flops)
    else:
        parts["vae_encode_flops"] = 0

    if include_vae_decode and profile.vae_decode is not None:
        total_flops += profile.vae_decode.flops
        parts["vae_decode_flops"] = int(profile.vae_decode.flops)
    else:
        parts["vae_decode_flops"] = 0

    if include_posthoc_encoder:
        if profile.posthoc_encoder is None:
            raise ValueError("include_posthoc_encoder=True but posthoc_encoder missing")
        total_flops += profile.posthoc_encoder.flops
        parts["posthoc_encoder_flops"] = int(profile.posthoc_encoder.flops)
        parts["posthoc_encoder_extra"] = dict(profile.posthoc_encoder.extra)
    else:
        parts["posthoc_encoder_flops"] = 0

    return {
        "flops": int(total_flops),
        "gflops": float(total_flops) / 1e9,
        "detail": parts,
    }


# =============================================================================
# End-to-end profiling (actual wall clock, no estimation)
# =============================================================================


@dataclass
class E2EProfile:
    """End-to-end profiling results with actual wall-clock measurements and CIs."""

    batch_size: int
    num_inference_steps: int
    cfg_scale: float

    # Generation: full sample() call without feature extraction
    generation_time_ms: float = 0.0
    generation_ci: Tuple[float, float] = (0.0, 0.0)  # (lower, upper)
    generation_images_per_sec: float = 0.0

    # REPA score: single SiT forward + projector at t≈0
    repa_score_time_ms: float = 0.0
    repa_score_ci: Tuple[float, float] = (0.0, 0.0)
    repa_score_overhead_pct: float = 0.0  # relative to generation

    # Vanilla score: VAE decode + posthoc encoder
    vanilla_score_time_ms: float = 0.0
    vanilla_score_ci: Tuple[float, float] = (0.0, 0.0)
    vanilla_posthoc_encoder_time_ms: float = 0.0

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "batch_size": int(self.batch_size),
                "num_inference_steps": int(self.num_inference_steps),
                "cfg_scale": float(self.cfg_scale),
            },
            "generation": {
                "time_ms": float(self.generation_time_ms),
                "ci_lower_ms": float(self.generation_ci[0]),
                "ci_upper_ms": float(self.generation_ci[1]),
                "images_per_sec": float(self.generation_images_per_sec),
            },
            "repa_score": {
                "time_ms": float(self.repa_score_time_ms),
                "ci_lower_ms": float(self.repa_score_ci[0]),
                "ci_upper_ms": float(self.repa_score_ci[1]),
                "overhead_pct": float(self.repa_score_overhead_pct),
            },
            "vanilla_score": {
                "time_ms": float(self.vanilla_score_time_ms),
                "ci_lower_ms": float(self.vanilla_score_ci[0]),
                "ci_upper_ms": float(self.vanilla_score_ci[1]),
                "posthoc_encoder_time_ms": float(self.vanilla_posthoc_encoder_time_ms),
            },
            "extra": dict(self.extra),
        }


def profile_e2e_generation(
    generator: nn.Module,
    *,
    batch_size: int,
    num_conditions: int,
    device: torch.device,
    num_inference_steps: int,
    cfg_scale: float,
    warmup: int,
    runs: int,
) -> Tuple[float, Dict[str, Any]]:
    """Profile end-to-end generation time (no feature extraction).

    Returns:
        (ms_per_batch, extra_info)
    """
    gen = _unwrap_generator(generator).eval().to(device)

    cond_ids = torch.zeros(batch_size, num_conditions, dtype=torch.long, device=device)

    def fn():
        with torch.inference_mode():
            gen.sample(
                cond_ids=cond_ids,
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
                return_aligned_features=False,
            )

    ms, extra = profile_time(fn, warmup=warmup, runs=runs, device=device)
    return ms, extra


def profile_e2e_repa_score(
    generator: nn.Module,
    *,
    batch_size: int,
    num_conditions: int,
    device: torch.device,
    warmup: int,
    runs: int,
) -> Tuple[float, Dict[str, Any]]:
    """Profile REPA score extraction: single SiT forward + projector at t≈0.

    This simulates scoring a batch of already-generated images by running
    one denoising step at t=0 with return_zs=True.

    Returns:
        (ms_per_batch, extra_info)
    """
    gen = _unwrap_generator(generator).eval().to(device)
    backbone = _get_backbone(gen)

    if not getattr(backbone, "use_repa", False):
        return 0.0, {"error": "Model does not have REPA enabled"}

    # Create latent at t=0 (clean latent, as if generation just finished)
    latent_size = gen.vae.get_latent_size(gen.cfg.image_size)
    latent_channels = gen.vae.out_channels
    x = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device)
    t = torch.zeros(batch_size, device=device)  # t=0
    cond_ids = torch.zeros(batch_size, num_conditions, dtype=torch.long, device=device)

    def fn():
        with torch.inference_mode():
            # Single forward with return_zs=True to get projector output
            _, zs = backbone(x, t, cond_ids, return_zs=True)
            if zs is None or len(zs) == 0:
                raise RuntimeError("REPA projector returned None")

    ms, extra = profile_time(fn, warmup=warmup, runs=runs, device=device)
    return ms, extra


def profile_e2e_vanilla_score(
    generator: nn.Module,
    posthoc_encoder: nn.Module,
    *,
    batch_size: int,
    device: torch.device,
    warmup: int,
    runs: int,
    include_vae_decode: bool = True,
) -> Tuple[float, float, Dict[str, Any]]:
    """Profile vanilla score extraction: (optional VAE decode) + posthoc encoder.

    Returns:
        (total_ms, posthoc_encoder_ms, extra_info)
    """
    import math

    gen = _unwrap_generator(generator).eval().to(device)
    posthoc_encoder = posthoc_encoder.eval().to(device)

    image_size = gen.cfg.image_size
    in_channels = gen.cfg.in_channels

    # For timing decode, create a latent
    latent_size = gen.vae.get_latent_size(image_size)
    latent_channels = gen.vae.out_channels
    latents = torch.randn(
        batch_size, latent_channels, latent_size, latent_size, device=device
    )

    # For timing encoder, create pixel images matching encoder's expected input
    # REPAEncoder handles channel conversion internally
    images = torch.randn(batch_size, in_channels, image_size, image_size, device=device)

    # Profile VAE decode
    decode_ms = 0.0
    decode_std = 0.0
    decode_extra = {}
    if include_vae_decode:

        def decode_fn():
            with torch.inference_mode():
                gen.vae.decode(latents)

        decode_ms, decode_extra = profile_time(
            decode_fn, warmup=warmup, runs=runs, device=device
        )
        decode_std = decode_extra.get("std_ms", 0.0)

    # Profile posthoc encoder
    def encoder_fn():
        with torch.inference_mode():
            posthoc_encoder(images)

    encoder_ms, encoder_extra = profile_time(
        encoder_fn, warmup=warmup, runs=runs, device=device
    )
    encoder_std = encoder_extra.get("std_ms", 0.0)

    # Compute total and combined CI
    # For sum of independent measurements: Var(X+Y) = Var(X) + Var(Y)
    total_ms = decode_ms + encoder_ms
    total_std = math.sqrt(decode_std**2 + encoder_std**2)

    # Compute CI for total (same approach as profile_time)
    if runs >= 30:
        t_crit = 1.96
    else:
        t_crit = 2.0 + 3.0 / runs
    ci_margin = t_crit * total_std / math.sqrt(runs)

    extra = {
        "timing_method": encoder_extra.get("timing_method", "cuda_event"),
        "timing_warmup": encoder_extra.get("timing_warmup", warmup),
        "timing_runs": encoder_extra.get("timing_runs", runs),
        "peak_mem_allocated_bytes": encoder_extra.get("peak_mem_allocated_bytes", 0),
        "vae_decode_ms": float(decode_ms),
        "vae_decode_std_ms": float(decode_std),
        "posthoc_encoder_ms": float(encoder_ms),
        "posthoc_encoder_std_ms": float(encoder_std),
        "include_vae_decode": include_vae_decode,
        "std_ms": float(total_std),
        "ci_lower_ms": float(total_ms - ci_margin),
        "ci_upper_ms": float(total_ms + ci_margin),
    }

    return total_ms, encoder_ms, extra


def profile_e2e(
    generator: nn.Module,
    *,
    batch_size: int,
    num_conditions: int,
    device: torch.device,
    num_inference_steps: int,
    cfg_scale: float,
    warmup: int,
    runs: int,
    posthoc_encoder: Optional[nn.Module] = None,
    include_vae_decode_in_vanilla: bool = True,
) -> E2EProfile:
    """Full end-to-end profiling with actual wall-clock measurements.

    Profiles:
    1. Generation: full sample() without feature extraction
    2. REPA score: single SiT+projector forward at t=0
    3. Vanilla score: VAE decode + posthoc encoder (if provided)
    """
    gen = _unwrap_generator(generator).eval().to(device)
    backbone = _get_backbone(gen)
    has_repa = getattr(backbone, "use_repa", False) and backbone.projectors is not None

    # 1. Profile generation
    gen_ms, gen_extra = profile_e2e_generation(
        generator,
        batch_size=batch_size,
        num_conditions=num_conditions,
        device=device,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        warmup=warmup,
        runs=runs,
    )
    images_per_sec = (batch_size / gen_ms) * 1000.0 if gen_ms > 0 else 0.0
    gen_ci = (gen_extra.get("ci_lower_ms", gen_ms), gen_extra.get("ci_upper_ms", gen_ms))

    # 2. Profile REPA score (if available)
    repa_ms = 0.0
    repa_ci = (0.0, 0.0)
    repa_extra = {}
    if has_repa:
        repa_ms, repa_extra = profile_e2e_repa_score(
            generator,
            batch_size=batch_size,
            num_conditions=num_conditions,
            device=device,
            warmup=warmup,
            runs=runs,
        )
        repa_ci = (repa_extra.get("ci_lower_ms", repa_ms), repa_extra.get("ci_upper_ms", repa_ms))
    repa_overhead_pct = (repa_ms / gen_ms) * 100.0 if gen_ms > 0 else 0.0

    # 3. Profile vanilla score (if posthoc encoder provided)
    vanilla_ms = 0.0
    vanilla_ci = (0.0, 0.0)
    posthoc_ms = 0.0
    vanilla_extra = {}
    if posthoc_encoder is not None:
        vanilla_ms, posthoc_ms, vanilla_extra = profile_e2e_vanilla_score(
            generator,
            posthoc_encoder,
            batch_size=batch_size,
            device=device,
            warmup=warmup,
            runs=runs,
            include_vae_decode=include_vae_decode_in_vanilla,
        )
        vanilla_ci = (vanilla_extra.get("ci_lower_ms", vanilla_ms), vanilla_extra.get("ci_upper_ms", vanilla_ms))

    return E2EProfile(
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        generation_time_ms=gen_ms,
        generation_ci=gen_ci,
        generation_images_per_sec=images_per_sec,
        repa_score_time_ms=repa_ms,
        repa_score_ci=repa_ci,
        repa_score_overhead_pct=repa_overhead_pct,
        vanilla_score_time_ms=vanilla_ms,
        vanilla_score_ci=vanilla_ci,
        vanilla_posthoc_encoder_time_ms=posthoc_ms,
        extra={
            "generation": gen_extra,
            "repa_score": repa_extra,
            "vanilla_score": vanilla_extra,
            "has_repa": has_repa,
            "warmup": warmup,
            "runs": runs,
        },
    )
