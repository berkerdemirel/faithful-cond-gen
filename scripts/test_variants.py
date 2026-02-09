#!/usr/bin/env python
"""Test script to verify each variant activates the correct losses."""

import sys
from dataclasses import dataclass, field
from typing import Dict, Set

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from faithful_cond_gen.model.generator import GeneratorWrapper
from faithful_cond_gen.pl_modules.generator_pl import GeneratorPL
from faithful_cond_gen.utils.config import auto_run_name
from hydra.utils import instantiate

CONFIG_DIR = "/mnt/pvc/faithful-cond-gen/configs"


@dataclass
class VariantSpec:
    name: str
    expected_run_name: str
    must_be_nonzero: Set[str]  # losses that MUST be computed AND non-zero
    should_skip: Set[str]      # losses that SHOULD be skipped (weight=0)
    allow_zero: Set[str] = field(default_factory=set)  # computed but may be 0 (e.g. attr_delta with no matches)


VARIANT_SPECS = [
    VariantSpec(
        name="vanilla",
        expected_run_name="celeba_vanilla_full",
        must_be_nonzero={"denoising"},
        should_skip={"add", "proj", "relational", "attr_delta"},
    ),
    VariantSpec(
        name="compositional",
        expected_run_name="celeba_compositional_full",
        must_be_nonzero={"denoising"},
        should_skip={"proj", "relational", "attr_delta"},
        allow_zero={"add"},  # architecturally ~0 due to additive conditioning
    ),
    VariantSpec(
        name="repa",
        expected_run_name="celeba_repa_full",
        must_be_nonzero={"denoising", "proj"},
        should_skip={"relational", "attr_delta"},
        allow_zero={"add"},
    ),
    VariantSpec(
        name="relational",
        expected_run_name="celeba_relational_full",
        must_be_nonzero={"denoising", "proj", "relational"},
        should_skip={"attr_delta"},
        allow_zero={"add"},
    ),
    VariantSpec(
        name="attr_delta",
        expected_run_name="celeba_attr_delta_full",
        must_be_nonzero={"denoising"},
        should_skip={"proj", "relational"},
        allow_zero={"add", "attr_delta"},  # add: arch ~0; attr_delta: may be 0 if no matches
    ),
]


def load_config(variant: str):
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=CONFIG_DIR, version_base=None)
    return compose(config_name="config_celeba", overrides=[f"variant={variant}"])


def create_fake_batch(batch_size: int = 64, num_attrs: int = 4, device: str = "cuda"):
    images = torch.randn(batch_size, 3, 256, 256, device=device)
    cond = {f"attr_{i}": torch.randint(0, 2, (batch_size,), device=device) for i in range(num_attrs)}
    return images, {"cond": cond}


def test_variant(spec: VariantSpec, device: str = "cuda") -> Dict:
    print(f"\n{'='*60}")
    print(f"Testing variant: {spec.name}")
    print(f"{'='*60}")

    cfg = load_config(spec.name)

    # Check run name
    run_name = auto_run_name(cfg)
    name_ok = run_name == spec.expected_run_name
    print(f"  Run name: {run_name} {'OK' if name_ok else 'FAIL'}")

    # Print config
    print(f"  Config:")
    print(f"    use_repa={cfg.model.use_repa}, proj_coeff={cfg.model.repa_proj_coeff}")
    print(f"    add_w={cfg.pl_module.additivity_loss_weight}, rel_w={cfg.pl_module.relational_loss_weight}, attr_w={cfg.pl_module.attr_delta_loss_weight}")

    # Build model
    cfg.model.attr_num_classes = [2, 2, 2, 2]
    cfg.model.in_channels = 3
    gen_cfg = instantiate(cfg.model)
    generator = GeneratorWrapper(gen_cfg)
    pl_cfg = instantiate(cfg.pl_module)
    model = GeneratorPL(generator, cfg=pl_cfg).to(device)
    model.train()

    # Load REPA encoder if needed
    if cfg.model.use_repa:
        from faithful_cond_gen.model.repa_encoder import load_repa_encoder
        model.repa_encoder = load_repa_encoder(
            encoder_name=cfg.model.repa_encoder,
            resolution=cfg.model.image_size,
            in_channels=cfg.model.in_channels,
            device=device,
        )
        print(f"  REPA encoder: {cfg.model.repa_encoder}")
    else:
        model.repa_encoder = None

    # Warmup: run training steps so model produces non-zero outputs
    warmup_model(model, device, num_steps=20)

    # Create batch for evaluation
    batch = create_fake_batch(batch_size=64, num_attrs=4, device=device)

    # Force probability gates to 1.0
    model.cfg.additivity_loss_prob = 1.0
    model.cfg.attr_delta_loss_prob = 1.0

    # Run forward and track which losses were computed
    computed, values = forward_with_tracking(model, batch)

    # Check results
    print(f"\n  Loss computation (must_nonzero={spec.must_be_nonzero}):")
    all_ok = name_ok

    for loss_name in ["denoising", "add", "proj", "relational", "attr_delta"]:
        was_computed = computed.get(loss_name, False)
        value = values.get(loss_name, 0.0)
        is_valid = not (torch.isnan(torch.tensor(value)) or torch.isinf(torch.tensor(value)))
        is_nonzero = abs(value) > 1e-12  # Very low threshold - add_loss is architecturally ~0

        if loss_name in spec.must_be_nonzero:
            expected = "nonzero"
            ok = was_computed and is_valid and is_nonzero
        elif loss_name in spec.should_skip:
            expected = "skipped"
            ok = not was_computed
        elif loss_name in spec.allow_zero:
            expected = "computed (can be 0)"
            ok = was_computed and is_valid
        else:
            expected = "?"
            ok = True

        status = "OK" if ok else "FAIL"
        val_str = f"={value:.8f}" if was_computed else ""
        print(f"    {loss_name}: {expected}{val_str} {status}")
        all_ok = all_ok and ok

    print(f"\n  Overall: {'PASS' if all_ok else 'FAIL'}")

    del model, generator
    torch.cuda.empty_cache()

    return {"variant": spec.name, "passed": all_ok}


def warmup_model(model: GeneratorPL, device: str, num_steps: int = 5):
    """Run a few training steps to get non-zero model outputs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(num_steps):
        batch = create_fake_batch(batch_size=32, num_attrs=4, device=device)
        images, cond_ids_raw = model._unpack_batch(batch)
        cond_ids = model.apply_condition_dropout(cond_ids_raw)

        x0 = model.generator.encode(images)
        b = x0.shape[0]
        t = torch.rand(b, device=x0.device)
        eps = torch.randn_like(x0)
        x_t, v_tgt = model.linear_interpolant(x0, t, eps)

        v_hat = model.generator.velocity_prediction(x_t=x_t, t=t, cond_ids=cond_ids)
        loss = F.mse_loss(v_hat, v_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"  Warmup: {num_steps} steps done")


def forward_with_tracking(model: GeneratorPL, batch):
    """Run forward and track which losses were computed."""
    images, cond_ids_raw = model._unpack_batch(batch)
    cond_ids = model.apply_condition_dropout(cond_ids_raw)

    with torch.no_grad():
        x0 = model.generator.encode(images)

    b = x0.shape[0]
    t = torch.rand(b, device=x0.device) * 0.2 + 0.8  # Force high t
    eps = torch.randn_like(x0)
    x_t, v_tgt = model.linear_interpolant(x0, t, eps)

    # Knobs
    use_repa = model.repa_encoder is not None
    add_w = float(model.cfg.additivity_loss_weight)
    proj_w = float(model.generator.cfg.repa_proj_coeff) if use_repa else 0.0
    rel_w = float(model.cfg.relational_loss_weight) if use_repa else 0.0
    attr_w = float(model.cfg.attr_delta_loss_weight) if use_repa else 0.0

    # Teacher embeddings
    zs = None
    if use_repa and (proj_w > 0 or rel_w > 0 or attr_w > 0):
        with torch.no_grad():
            zs = model.repa_encoder(images)

    # Velocity prediction
    zs_tilde = None
    if use_repa:
        v_hat, zs_tilde = model.generator.velocity_prediction(
            x_t=x_t, t=t, cond_ids=cond_ids, return_projected=True
        )
    else:
        v_hat = model.generator.velocity_prediction(x_t=x_t, t=t, cond_ids=cond_ids)

    computed = {}
    values = {}

    # Denoising (always computed)
    computed["denoising"] = True
    values["denoising"] = F.mse_loss(v_hat, v_tgt).item()

    # Additivity
    if add_w > 0:
        computed["add"] = True
        values["add"] = model.compute_additivity_loss(x_t, t, cond_ids_raw).item()

    # Projection (REPA)
    if proj_w > 0 and zs is not None and zs_tilde is not None:
        computed["proj"] = True
        values["proj"] = model._compute_repa_loss(zs, zs_tilde).item()

    # Relational
    if rel_w > 0 and zs is not None and zs_tilde is not None:
        computed["relational"] = True
        values["relational"] = model._compute_repa_relational(zs, zs_tilde).item()

    # Attr delta
    if attr_w > 0 and zs is not None:
        computed["attr_delta"] = True
        values["attr_delta"] = model.compute_attr_delta_loss(x_t, t, cond_ids_raw, zs).item()

    return computed, values


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = []
    for spec in VARIANT_SPECS:
        try:
            result = test_variant(spec, device)
            results.append(result)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({"variant": spec.name, "passed": False})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r.get("passed", False))
    for r in results:
        print(f"  {r['variant']}: {'PASS' if r.get('passed') else 'FAIL'}")
    print(f"\n  Total: {passed}/{len(results)}")

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
