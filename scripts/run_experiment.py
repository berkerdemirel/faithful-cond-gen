#!/usr/bin/env python
import hydra
from faithful_cond_gen.data.celeba import CelebaDataConfig, CelebaDataModule
from faithful_cond_gen.data.rxrx1 import RxRx1DataConfig, RxRx1DataModule
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print("Config loaded:")
    print(cfg)

    # instantiate dataset module
    dm_cfg = instantiate(cfg.dataset)
    if isinstance(dm_cfg, RxRx1DataConfig):
        dm = RxRx1DataModule(dm_cfg)
    elif isinstance(dm_cfg, CelebaDataConfig):
        dm = CelebaDataModule(dm_cfg)
    print(dm)

    # example: train dataloader
    dl = dm.get_dataloader("train")
    x, cond = next(iter(dl))
    print("Sample batch:", x.shape)

    # # instantiate models when implemented
    # gen = instantiate(cfg.model)
    # print(gen)

    # # instantiate eval config
    print(cfg.eval)


if __name__ == "__main__":
    main()
