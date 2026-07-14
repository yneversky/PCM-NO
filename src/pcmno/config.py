from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    cfg = deepcopy(cfg)
    cfg.setdefault("_meta", {})
    cfg["_meta"]["config_path"] = str(path)
    cfg["_meta"]["config_dir"] = str(path.parent)
    return cfg


def parse_scalar(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return cfg
    out = deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item}")
        dotted, raw = item.split("=", 1)
        keys = dotted.split(".")
        node: dict[str, Any] = out
        for key in keys[:-1]:
            child = node.setdefault(key, {})
            if not isinstance(child, dict):
                raise TypeError(f"Cannot descend into non-mapping key: {dotted}")
            node = child
        node[keys[-1]] = parse_scalar(raw)
    return out


def resolve_path(value: str | Path, repo_root: str | Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    base = Path(repo_root).expanduser().resolve() if repo_root else Path.cwd()
    return (base / path).resolve()


def require(cfg: dict[str, Any], dotted_key: str) -> Any:
    node: Any = cfg
    for key in dotted_key.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Missing configuration key: {dotted_key}")
        node = node[key]
    return node
