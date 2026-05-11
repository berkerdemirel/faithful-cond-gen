"""
Rewrite gen_meta in posthoc encoded caches with per-model attribute order.

Visual spot-check (single-attribute-on probes for Male/Smiling/Blond/Eye)
revealed that the stored `condition` tuple inside each `cond_*.pt` is NOT in
a single canonical order across all celeba models:

  * celeba_vanilla_marginal_v1  → CK order (Male, Smiling, Blond_Hair, Eyeglasses)
  * all other celeba models     → alphabetical (Blond_Hair, Eyeglasses, Male, Smiling)

This script rewrites `gen_meta` in
`/mnt/pvc/posthoc_debug/feature_cache/{model_key}_encoded.pt` accordingly
and saves 5 single-attribute-on sample grids per model so the labels can
be visually confirmed.

Usage
-----
PYTHONPATH=src uv run python scripts/posthoc_alignment/relabel_posthoc_caches.py
"""

import glob
from pathlib import Path

import torch
import torchvision.utils as vutils

CK = ["Male", "Smiling", "Blond_Hair", "Eyeglasses"]

CACHE_DIR = Path("/mnt/pvc/posthoc_debug/feature_cache")
DIAG_ROOT = Path("/mnt/pvc/faithful-cond-gen/outputs/posthoc_alignment/diag")
SAMPLE_DIR = Path("/mnt/pvc/faithful-cond-gen/big_debug/relabel_samples_final")
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

# model_key -> stored order in cond_*.pt files
MODEL_ORDER = {
    "celeba_vanilla_full_v1": sorted(CK),           # alphabetical
    "celeba_vanilla_marginal_v1": list(CK),         # CK
    "celeba_repa_full_v1": sorted(CK),
    "celeba_repa_marginal_v1": sorted(CK),
    "celeba_repa_siglip_full_v1": sorted(CK),
    "celeba_repa_siglip_marginal_v1": sorted(CK),
}

# Single-attribute-on probes. Filename digits are in the model's stored order.
PROBES = [
    (0, 0, 0, 0),
    (0, 0, 0, 1),
    (0, 0, 1, 0),
    (0, 1, 0, 0),
    (1, 0, 0, 0),
]
N_PER_COND = 6


def _label(cond_tuple, order):
    return ", ".join(f"{k}={v}" for k, v in zip(order, cond_tuple))


def relabel_cache(model_key: str, order):
    cache_path = CACHE_DIR / f"{model_key}_encoded.pt"
    if not cache_path.exists():
        print(f"  [skip] {model_key}: no encoded cache")
        return
    gen_cache_dir = DIAG_ROOT / model_key / "gen_cache"
    pts = sorted(glob.glob(str(gen_cache_dir / "cond_*.pt")))
    if not pts:
        print(f"  [skip] {model_key}: no cond_*.pt files")
        return

    c = torch.load(cache_path, map_location="cpu", weights_only=False)

    new_meta = {k: [] for k in CK}
    total = 0
    for p in pts:
        d = torch.load(p, map_location="cpu", weights_only=False)
        cond = d["condition"]
        n = d["images"].shape[0]
        total += n
        for ki, k in enumerate(order):
            new_meta[k].append(torch.full((n,), int(cond[ki]), dtype=torch.long))
    new_meta = {k: torch.cat(new_meta[k]) for k in CK}

    cached_n = c["gen_hidden"].shape[0]
    if total != cached_n:
        print(f"  [ERROR] {model_key}: cond files have {total}, cache has {cached_n}")
        return

    c["gen_meta"] = new_meta
    c["relabel_fix"] = f"stored_order={order}"
    torch.save(c, cache_path)
    print(f"  [ok]   {model_key}: relabeled {total} samples (order={order}) → {cache_path.name}")


def save_samples(model_key: str, order):
    gen_cache_dir = DIAG_ROOT / model_key / "gen_cache"
    out_dir = SAMPLE_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    for cond in PROBES:
        fname = "cond_" + "_".join(str(c) for c in cond) + ".pt"
        p = gen_cache_dir / fname
        if not p.exists():
            print(f"  [miss] {model_key}/{fname}")
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        imgs = d["images"][:N_PER_COND].clamp(0, 1)
        stored = tuple(int(x) for x in d["condition"])
        label = _label(stored, order)

        tag = "_".join(f"{k}{v}" for k, v in zip(order, cond))
        out_path = out_dir / f"{tag}.png"
        vutils.save_image(imgs, str(out_path), nrow=N_PER_COND, padding=2)
        print(f"  [img]  {model_key}/{tag}.png  label=({label})  file={fname}")


def main():
    print("─── Relabel encoded caches (per-model order) ──────────")
    for m, order in MODEL_ORDER.items():
        relabel_cache(m, order)
    print()
    print("─── Save sample grids ─────────────────────────────────")
    for m, order in MODEL_ORDER.items():
        save_samples(m, order)
    print()
    print(f"Visual samples → {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
