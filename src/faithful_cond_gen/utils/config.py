"""Config utilities for experiment management."""

from omegaconf import DictConfig


def auto_run_name(cfg: DictConfig) -> str:
    """Generate run name from config based on dataset, variant, and setting."""
    # Detect dataset
    dataset = "celeba" if "Celeba" in cfg.dataset._target_ else "rxrx1"

    # Detect variant from enabled flags
    use_repa = cfg.model.get("use_repa", False)
    proj_coeff = cfg.model.get("repa_proj_coeff", 0.0)
    add_w = cfg.pl_module.get("additivity_loss_weight", 0.0)
    rel_w = cfg.pl_module.get("relational_loss_weight", 0.0)
    attr_w = cfg.pl_module.get("attr_delta_loss_weight", 0.0)

    if attr_w > 0:
        variant = "attr_delta"
    elif rel_w > 0:
        variant = "relational"
    elif use_repa and proj_coeff > 0:
        variant = "repa"
    elif add_w > 0:
        variant = "compositional"
    else:
        variant = "vanilla"

    # Detect setting (full vs marginal)
    setting = "marginal" if "marginal" in cfg.dataset._target_.lower() else "full"

    return f"{dataset}_{variant}_{setting}"


def maybe_set_run_name(cfg: DictConfig) -> None:
    """Set logger.name if not explicitly provided or using default."""
    current = cfg.logger.get("name")
    if current is None or current.startswith("run_"):
        cfg.logger.name = auto_run_name(cfg)
        print(f"[config] Auto-generated run name: {cfg.logger.name}")
