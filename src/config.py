from copy import deepcopy
from pathlib import Path

import yaml


def _deep_update(base, override):
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    base_name = cfg.pop("base", None)
    if base_name is not None:
        base_cfg = load_config(path.parent / base_name)
        cfg = _deep_update(base_cfg, cfg)
    return cfg
