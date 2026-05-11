"""Extract SigLIP meanpatch features on the rxrx1 FULL train pool (73,101 rows).

Aligned row-for-row with extract_raw_hidden output under dataset=rxrx1. Used
as the mapper target for rxrx1 full-model SigLIP mappers (768 -> 1152).

Output: outputs/real_rxrx1_siglip_meanpatch_full/train_features.pt

Usage:
    PYTHONPATH=src uv run python scripts/posthoc_alignment/cache_rxrx1_full_siglip.py
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
    ds_cfg = OmegaConf.load(repo_root / "configs/dataset/rxrx1.yaml")
    ds_cfg.batch_size = 32
    ds_cfg.num_workers = 8
    ds_cfg.augment_train = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm_conf = instantiate(ds_cfg)
    dm = RxRx1DataModule(dm_conf)
    dm.setup(stage="fit")

    save_dir = repo_root / "outputs/real_rxrx1_siglip_meanpatch_full"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(save_dir / "train_features.pt")

    loader = dm.get_dataloader("train", shuffle=False, drop_last=False)
    extract_and_save_meanpatch(
        loader, device, save_path, encoder_name="siglip", image_size=256
    )


if __name__ == "__main__":
    main()
