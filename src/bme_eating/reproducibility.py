from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from bme_eating.config import feature_artifact_name


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(project_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def write_run_manifest(
    output_dir: Path,
    config: dict[str, Any],
    output_root: Path,
    *,
    command: list[str] | None = None,
) -> Path:
    """Write a path-free, credential-free record of a completed experiment run."""

    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(str(config["_config_path"]))
    feature_name = feature_artifact_name(config)
    tracked_inputs = {
        "config": config_path,
        "quality_report": output_root / "indices" / "quality_report.json",
        "quality_expectations": output_root / "indices" / "quality_expectations.json",
        "subject_folds": output_root / "indices" / "subject_folds.json",
        "subject_folds_manifest": output_root / "indices" / "subject_folds.manifest.json",
        "events": output_root / "indices" / "events.parquet",
        "features": output_root / "features" / f"{feature_name}.parquet",
    }
    model_files = {
        path.name: _sha256(path)
        for path in sorted(output_dir.glob("*model.json"))
        if path.is_file()
    }
    try:
        import xgboost

        xgboost_version = xgboost.__version__
    except ImportError:  # pragma: no cover - training environment always has XGBoost.
        xgboost_version = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        cuda_version = torch.version.cuda
    except ImportError:  # pragma: no cover - project environment includes torch.
        cuda_available = False
        gpu_name = None
        cuda_version = None
    dirty_output = _git_value(project_root, "status", "--porcelain")
    public_config = {
        key: value
        for key, value in config.items()
        if not key.startswith("_") and key not in {"credentials", "secrets"}
    }
    resolved_config_hash = hashlib.sha256(
        json.dumps(public_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    raw_command = list(command if command is not None else sys.argv)
    safe_command = [
        Path(value).name if Path(value).is_absolute() else value for value in raw_command
    ]
    payload = {
        "version": 1,
        "git": {
            "commit": _git_value(project_root, "rev-parse", "HEAD"),
            "dirty": bool(dirty_output),
        },
        "command": safe_command,
        "random_seeds": {
            "project": int(config.get("project", {}).get("seed", 0)),
            "split": int(config.get("data", {}).get("split_seed", 0)),
            "xgboost": int(config.get("xgboost", {}).get("random_seed", 0)),
        },
        "hashes": {
            name: value
            for name, path in tracked_inputs.items()
            if (value := _sha256(path)) is not None
        },
        "resolved_config_sha256": resolved_config_hash,
        "model_hashes": model_files,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "operating_system": platform.platform(),
            "xgboost": xgboost_version,
            "cuda_available": cuda_available,
            "cuda_runtime": cuda_version,
            "gpu": gpu_name,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        },
    }
    path = output_dir / "run_manifest.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
