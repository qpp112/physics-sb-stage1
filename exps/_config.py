"""Config loading with simple _base: inheritance."""

from __future__ import annotations
from pathlib import Path
import yaml


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("_base", None)
    if base is not None:
        base_path = path.parent / base
        base_cfg = load_config(str(base_path))
        cfg = _deep_update(base_cfg, cfg)
    return cfg
