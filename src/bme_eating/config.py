from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    parent_name = payload.pop("extends", None)
    if parent_name:
        parent_path = (config_path.parent / parent_name).resolve()
        parent = load_config(parent_path)
        payload = _deep_merge(parent, payload)
    payload["_config_path"] = str(config_path)
    return payload


def _load_project_dotenv(config: dict[str, Any]) -> None:
    config_path = Path(config["_config_path"])
    dotenv_path = config_path.parent.parent / ".env"
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if name:
            os.environ.setdefault(name, value)


def require_environment_path(config: dict[str, Any], key: str) -> Path:
    env_name = config["project"][key]
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"Environment variable {env_name} is required")
    return Path(value).expanduser().resolve()


def resolve_roots(config: dict[str, Any]) -> tuple[Path, Path]:
    _load_project_dotenv(config)
    data_root = require_environment_path(config, "data_root_env")
    output_root = require_environment_path(config, "output_root_env")
    output_root.mkdir(parents=True, exist_ok=True)
    return data_root, output_root
