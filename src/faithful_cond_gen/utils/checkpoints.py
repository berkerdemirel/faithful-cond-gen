"""Checkpoint registry loader for faithful-cond-gen."""
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_REGISTRY_PATH = Path(__file__).parents[3] / "configs" / "checkpoints.yaml"
_CACHE: Optional[Dict[str, Any]] = None


def load_registry() -> Dict[str, Any]:
    """Load checkpoint registry from configs/checkpoints.yaml."""
    global _CACHE
    if _CACHE is None:
        with open(_REGISTRY_PATH) as f:
            _CACHE = yaml.safe_load(f)
    return _CACHE


def get_checkpoint(key: str) -> Dict[str, Any]:
    """Get checkpoint metadata by key."""
    reg = load_registry()
    if key not in reg["checkpoints"]:
        available = list(reg["checkpoints"].keys())
        raise KeyError(f"Checkpoint '{key}' not found. Available: {available}")
    return reg["checkpoints"][key]


def get_checkpoint_path(key: str, absolute: bool = True) -> str:
    """Get checkpoint path by key. Returns absolute path by default."""
    ckpt = get_checkpoint(key)
    path = ckpt["ckpt_path"]
    if absolute:
        base = Path(__file__).parents[3]
        path = str(base / path)
    return path


def list_checkpoints(
    dataset: Optional[str] = None, paper_ready: Optional[bool] = None
) -> Dict[str, Dict[str, Any]]:
    """List checkpoints, optionally filtered by dataset or paper_ready status."""
    reg = load_registry()
    result = {}
    for key, meta in reg["checkpoints"].items():
        if dataset and meta.get("dataset") != dataset:
            continue
        if paper_ready is not None and meta.get("paper_ready") != paper_ready:
            continue
        result[key] = meta
    return result
