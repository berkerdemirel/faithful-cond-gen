#!/usr/bin/env python
"""Export RGB-rendered RxRx1 images for the 50 eval-subset pairs.

Writes PNGs to /mnt/pvc/rgb_imgs_rxrx1/<model>/ for each of the 6 gen model
output dirs under outputs/gen, plus /mnt/pvc/rgb_imgs_rxrx1/real_imgs/ for
the real dataset. Filenames preserve the cell{c}_sirna{s}_{idx} convention.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from faithful_cond_gen.data.rxrx1 import (
    RxRx1DataConfig,
    RxRx1DataModule,
    to_rgb,
)

EVAL_SUBSET_JSON = (
    "/mnt/pvc/faithful-cond-gen/outputs/posthoc_alignment/"
    "rxrx1_eval_subset_final.json"
)
GEN_ROOT = "/mnt/pvc/faithful-cond-gen/outputs/gen"
OUT_ROOT = "/mnt/pvc/rgb_imgs_rxrx1"
REAL_DATA_DIR = "/mnt/pvc/AutoSync/data/rxrx1"

GEN_MODELS = [
    "rxrx1_repa_full",
    "rxrx1_repa_marginal",
    "rxrx1_repa_siglip_full",
    "rxrx1_repa_siglip_marginal",
    "rxrx1_vanilla_full",
    "rxrx1_vanilla_marginal",
]

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_pairs(json_path):
    with open(json_path) as f:
        data = json.load(f)
    pairs = []
    for entry in data["seen"] + data["unseen"]:
        pairs.append((int(entry["cell_type_id"]), int(entry["sirna_id"])))
    return pairs


def tensor_to_png(rgb: torch.Tensor) -> Image.Image:
    arr = (rgb.clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)
    return Image.fromarray(arr, mode="RGB")


def flush_batch_gen(batch, names, out_dir: Path):
    x = torch.stack(batch, dim=0).to(DEVICE)
    with torch.no_grad():
        rgb = to_rgb(x)
    for i, nm in enumerate(names):
        tensor_to_png(rgb[i]).save(out_dir / f"{nm}.png")


def export_gen_model(model_name: str, pairs):
    images_dir = Path(GEN_ROOT) / model_name / "images"
    out_dir = Path(OUT_ROOT) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for (c, s) in pairs:
        pattern = str(images_dir / f"cell{c}_sirna{s}_*.pt")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  [{model_name}] no files for cell={c} sirna={s}")
            continue

        batch, batch_names = [], []
        for fp in files:
            x = torch.load(fp, map_location="cpu", weights_only=False)
            batch.append(x)
            batch_names.append(Path(fp).stem)
            if len(batch) == BATCH_SIZE:
                flush_batch_gen(batch, batch_names, out_dir)
                batch, batch_names = [], []
        if batch:
            flush_batch_gen(batch, batch_names, out_dir)
        total_written += len(files)
    print(f"[{model_name}] wrote {total_written} PNGs -> {out_dir}")


def export_real(pairs):
    out_dir = Path(OUT_ROOT) / "real_imgs"
    out_dir.mkdir(parents=True, exist_ok=True)

    dm_cfg = RxRx1DataConfig(
        data_dir=REAL_DATA_DIR,
        img_size=(512, 512),
        resize=(256, 256),
        reduce_channels=True,
        augment_train=False,
        normalize=False,
        use_numpy=True,
        use_parquet=False,
        batch_size=32,
        num_workers=0,
        val_size=0.1,
        seed=1337,
        rare_threshold=20,
        held_out_pairs=None,
    )
    dm = RxRx1DataModule(dm_cfg)

    total = 0
    for (c, s) in pairs:
        try:
            ds = dm.get_matching_dataset(
                "train",
                {"cell_type_id": c, "sirna_id": s},
                max_samples=None,
            )
        except ValueError:
            print(f"  [real] no samples for cell={c} sirna={s}")
            continue

        for i in range(len(ds)):
            x, _ = ds[i]
            tensor_to_png(x).save(out_dir / f"cell{c}_sirna{s}_{i}.png")
        total += len(ds)
    print(f"[real] wrote {total} PNGs -> {out_dir}")


def main():
    pairs = load_pairs(EVAL_SUBSET_JSON)
    print(f"Loaded {len(pairs)} pairs from eval subset; device={DEVICE}")
    os.makedirs(OUT_ROOT, exist_ok=True)

    for mn in GEN_MODELS:
        export_gen_model(mn, pairs)

    export_real(pairs)


if __name__ == "__main__":
    main()
