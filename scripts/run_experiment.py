#!/usr/bin/env python
import hydra
import pytorch_lightning as pl
import torch

# Import your modules
from faithful_cond_gen.data.celeba import CelebaDataConfig, CelebaDataModule
from faithful_cond_gen.data.rxrx1 import RxRx1DataConfig, RxRx1DataModule
from faithful_cond_gen.model.generator import GeneratorConfig, GeneratorWrapper
from faithful_cond_gen.pl_modules.generator_pl import GeneratorPL, GeneratorPLConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../configs", config_name="config_rxrx1", version_base=None)
def main(cfg: DictConfig):
    # 1. Set Seed & Device logic (handled by PL Trainer usually, but good for setup)
    pl.seed_everything(cfg.get("seed", 1337), workers=True)
    torch.set_float32_matmul_precision("high")  # or "medium"

    print(f"[{cfg.dataset._target_}] Initializing DataModule...")

    # 2. Instantiate DataModule
    # We instantiate the Config first, then the Module, to ensure types are correct
    dm_conf = instantiate(cfg.dataset)
    if "RxRx1" in cfg.dataset._target_:
        dm = RxRx1DataModule(dm_conf)
    else:
        dm = CelebaDataModule(dm_conf)

    # 3. Prepare Model Config
    # CelebA default attributes are binary (2 classes).
    # If using default 4 attrs: [2, 2, 2, 2]
    if hasattr(dm, "selected_attrs"):
        num_attrs = len(dm.selected_attrs)
        # Assuming binary attributes for CelebA;
        # For RxRx1 you might need logic like [4, 1138]
        if "Celeba" in cfg.dataset._target_:
            # Override config to match data
            print(
                f"Detected {num_attrs} attributes for CelebA. Setting attr_num_classes to [2]*{num_attrs}"
            )
            cfg.model.attr_num_classes = [2] * num_attrs
            cfg.model.in_channels = 3

    print("Initializing Generator...")
    gen_cfg = instantiate(cfg.model)
    generator = GeneratorWrapper(gen_cfg)

    # 4. Instantiate Lightning Module
    # We pass the constructed generator and the PL-specific config
    pl_cfg = instantiate(cfg.pl_module)
    model = GeneratorPL(generator, cfg=pl_cfg)

    # 5. Instantiate Logger
    print("Initializing WandB Logger...")
    # This uses the logger config group
    logger = instantiate(cfg.logger)

    # 6. Instantiate Callbacks (NEW)
    callbacks = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                print(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(instantiate(cb_conf))

    # 7. Instantiate Trainer
    # We use the 'train' section of the config for trainer args
    print("Initializing Trainer...")
    trainer = instantiate(cfg.train, logger=logger, callbacks=callbacks)

    # 8. Run Training
    print("Starting Training...")
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
