"""Extract DINOv3 meanpatch features on the rxrx1_marginal real train pool.

Produces a 66,181-row cache aligned row-for-row with
outputs/posthoc_alignment/raw_hidden/rxrx1_*/t*.pt. Used as the mapper
target for RxRx1 DINOv3 posthoc mappers (768 -> 1024).

Usage:
    PYTHONPATH=src uv run python scripts/posthoc_alignment/cache_rxrx1_marginal_dinov3.py
"""

import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from cache_features import extract_and_save_meanpatch  # noqa: E402

from faithful_cond_gen.data.rxrx1 import RxRx1DataModule  # noqa: E402


def main():
    repo_root = Path(__file__).parent.parent.parent
    ds_cfg = OmegaConf.load(repo_root / "configs/dataset/rxrx1_marginal.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm_conf = instantiate(ds_cfg)
    dm = RxRx1DataModule(dm_conf)
    dm.setup(stage="fit")

    save_dir = repo_root / "outputs/real_rxrx1_dinov3_meanpatch_marginal"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(save_dir / "train_features.pt")

    loader = dm.get_dataloader("train", shuffle=False, drop_last=False)
    extract_and_save_meanpatch(
        loader, device, save_path, encoder_name="dinov3-vit-l", image_size=256
    )


if __name__ == "__main__":
    main()
