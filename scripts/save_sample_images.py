"""Save one generated image per condition (top-50) as RGB PNGs for visual inspection."""

import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from torchvision.utils import save_image
from faithful_cond_gen.data.rxrx1 import to_rgb, RxRx1DataModule, RxRx1DataConfig

GEN_DIR = Path("outputs/gen/rxrx1_repa_openphenom_full/images")
OUT_DIR = Path("outputs/sample_images_openphenom_full")
TOP_K = 50

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pick top-50 conditions (same logic as e2e diagnostic)
    dm = RxRx1DataModule(RxRx1DataConfig(
        data_dir="/mnt/pvc/AutoSync/data/rxrx1",
        img_size=[512, 512], resize=[256, 256],
        reduce_channels=False, augment_train=False, normalize=False,
        use_numpy=True, use_parquet=False,
        batch_size=16, num_workers=4, val_size=0.1,
        seed=1337, rare_threshold=20, held_out_pairs=None,
    ))
    ds = dm.train_dataloader().dataset
    counts = defaultdict(int)
    for ct, si in zip(ds.cell_type_ids, ds.sirna_ids):
        counts[(int(ct), int(si))] += 1
    top = sorted(counts.items(), key=lambda x: -x[1])[:TOP_K]
    conditions = [c for c, _ in top]

    # Map conditions to their string form used in filenames
    # Filenames are like: cell_0_sirna_123_sample_0.pt
    print(f"Scanning {GEN_DIR} for generated images...")
    saved = 0
    for ct, si in conditions:
        # Filename format: cell{ct}_sirna{si}_{idx}.pt
        fpath = GEN_DIR / f"cell{ct}_sirna{si}_0.pt"
        if not fpath.exists():
            matches = list(GEN_DIR.glob(f"cell{ct}_sirna{si}_*.pt"))
            if not matches:
                print(f"  SKIP: cell={ct} sirna={si} — no files found")
                continue
            fpath = matches[0]

        data = torch.load(fpath, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            img = data.get("image", data.get("images", None))
            if img is None:
                img = next(iter(data.values()))
        else:
            img = data

        if img.dim() == 3:
            img = img.unsqueeze(0)  # (1, C, H, W)

        # Clamp to [0,1]
        img = torch.clamp(img.float(), 0, 1)

        # Convert 6ch -> RGB
        rgb = to_rgb(img)  # (1, 3, H, W)

        out_path = OUT_DIR / f"cell{ct}_sirna{si}.png"
        save_image(rgb[0], str(out_path))
        saved += 1

    print(f"\nSaved {saved}/{TOP_K} images to {OUT_DIR}")

    # Also save a grid
    if saved > 0:
        from torchvision.utils import make_grid
        all_imgs = []
        for ct, si in conditions:
            fpath = GEN_DIR / f"cell{ct}_sirna{si}_0.pt"
            if not fpath.exists():
                matches = list(GEN_DIR.glob(f"cell{ct}_sirna{si}_*.pt"))
                if not matches:
                    continue
                fpath = matches[0]
            data = torch.load(fpath, map_location="cpu", weights_only=False)
            img = data if isinstance(data, torch.Tensor) else next(v for v in data.values() if isinstance(v, torch.Tensor))
            if img.dim() == 3:
                img = img.unsqueeze(0)
            rgb = to_rgb(torch.clamp(img.float(), 0, 1))
            all_imgs.append(rgb[0])
            if len(all_imgs) >= 50:
                break
        grid = make_grid(all_imgs, nrow=10, padding=2)
        save_image(grid, str(OUT_DIR / "grid_50.png"))
        print(f"Saved grid ({len(all_imgs)} images) to {OUT_DIR / 'grid_50.png'}")

if __name__ == "__main__":
    main()
