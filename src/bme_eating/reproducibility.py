from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

from bme_eating.config import feature_artifact_name


def epoch_random_seed(base_seed: int, outer_fold: int, epoch: int) -> int:
    if outer_fold < 0 or epoch < 0:
        raise ValueError("outer_fold and epoch must be non-negative")
    return int(
        (int(base_seed) + 1_000_003 * int(outer_fold) + 10_007 * (int(epoch) + 1))
        % (2**32 - 1)
    )


def should_validate_epoch(epoch: int, interval: int) -> bool:
    if epoch < 0 or interval <= 0:
        raise ValueError("epoch must be non-negative and validation interval must be positive")
    return epoch == 0 or (epoch + 1) % interval == 0


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
            encoding="utf-8",
            errors="strict",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    if result.stdout is None:
        return None
    return result.stdout.strip()


def git_worktree_identity(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or Path(__file__).resolve().parents[2]
    commit = _git_value(root, "rev-parse", "HEAD")
    dirty_output = _git_value(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    diff_output = _git_value(root, "diff", "--binary", "HEAD", "--")
    untracked_output = _git_value(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    if (
        not commit
        or dirty_output is None
        or diff_output is None
        or untracked_output is None
    ):
        raise RuntimeError("Cannot verify the Git identity for experiment tracking")

    digest = hashlib.sha256()
    digest.update(commit.encode("utf-8"))
    digest.update(b"\0status\0")
    digest.update(dirty_output.encode("utf-8"))
    digest.update(b"\0diff\0")
    digest.update(diff_output.encode("utf-8"))
    for relative in sorted(value for value in untracked_output.split("\0") if value):
        path = root / relative
        digest.update(b"\0untracked\0")
        digest.update(relative.encode("utf-8"))
        file_hash = _sha256(path)
        if file_hash is not None:
            digest.update(file_hash.encode("ascii"))
    return {
        "commit": commit,
        "dirty": bool(dirty_output),
        "worktree_sha256": digest.hexdigest(),
    }


def require_git_worktree(project_root: Path | None = None) -> str:
    identity = git_worktree_identity(project_root)
    if identity["dirty"]:
        warnings.warn(
            "Git worktree has uncommitted changes; execution is allowed and the exact "
            "worktree fingerprint will be recorded in the run manifest",
            RuntimeWarning,
            stacklevel=2,
        )
    return str(identity["commit"])


def require_clean_git_worktree(project_root: Path | None = None) -> str:
    identity = git_worktree_identity(project_root)
    if identity["dirty"]:
        raise RuntimeError(
            "Formal training requires a clean Git worktree; review, commit, and synchronize "
            "the current changes first"
        )
    return str(identity["commit"])


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
    tracked_inputs = {
        "config": config_path,
        "quality_report": output_root / "indices" / "quality_report.json",
        "quality_expectations": output_root / "indices" / "quality_expectations.json",
        "subject_folds": output_root / "indices" / "subject_folds.json",
        "subject_folds_manifest": output_root / "indices" / "subject_folds.manifest.json",
        "events": output_root / "indices" / "events.parquet",
        "anchors": output_root / "indices" / "anchors.parquet",
        "segments": output_root / "indices" / "segments.parquet",
    }
    if "features" in config:
        feature_name = feature_artifact_name(config)
        tracked_inputs["features"] = output_root / "features" / f"{feature_name}.parquet"
    model_files = {
        path.relative_to(output_dir).as_posix(): _sha256(path)
        for path in sorted(output_dir.rglob("*model.json"))
        if path.is_file()
    }
    artifact_names = (
        "model.json",
        "metadata.json",
        "best_validation_selection.json",
        "best_validation_predictions.parquet",
        "validation_predictions.parquet",
        "dtp_oof_predictions.parquet",
        "dtp_test_predictions.parquet",
        "dtp_source.json",
        "test_predictions.parquet",
        "meta_oof_events.csv",
        "meta_oof_metrics.json",
        "selected_postprocess.json",
        "selected_fusion.json",
        "fusion_trials.csv",
        "test_events.csv",
        "test_failure_cases.csv",
        "test_metrics.json",
        "best.pt",
    )
    artifact_hashes = {
        name: digest
        for name in artifact_names
        if (digest := _sha256(output_dir / name)) is not None
    }
    for partition_dir in sorted(output_dir.glob("crossfit_*")):
        for name in (
            "best.pt",
            "metadata.json",
            "normalization.json",
            "best_validation_predictions.parquet",
            "best_validation_predictions.parquet.manifest.json",
            "dtp_test_predictions.parquet",
            "dtp_test_predictions.parquet.manifest.json",
        ):
            path = partition_dir / name
            if (digest := _sha256(path)) is not None:
                artifact_hashes[path.relative_to(output_dir).as_posix()] = digest
    postprocess_trials = output_dir / "postprocess_trials"
    if postprocess_trials.is_dir():
        for path in sorted(postprocess_trials.rglob("*")):
            if path.is_file() and (digest := _sha256(path)) is not None:
                artifact_hashes[path.relative_to(output_dir).as_posix()] = digest
    try:
        import xgboost

        xgboost_version = xgboost.__version__
    except ImportError:  # pragma: no cover - training environment always has XGBoost.
        xgboost_version = None
    torch_version = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        cuda_version = torch.version.cuda
    except ImportError:  # pragma: no cover - project environment includes torch.
        cuda_available = False
        gpu_name = None
        cuda_version = None
    git_identity = git_worktree_identity(project_root)
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
    fold_match = re.fullmatch(r"fold_(\d+)", output_dir.name)
    payload = {
        "version": 2,
        "experiment": {
            "name": output_dir.parent.name,
            "fold": int(fold_match.group(1)) if fold_match else None,
        },
        "git": git_identity,
        "command": safe_command,
        "random_seeds": {
            "project": int(config.get("project", {}).get("seed", 0)),
            "split": int(config.get("data", {}).get("split_seed", 0)),
            "xgboost": int(config.get("xgboost", {}).get("random_seed", 0)),
            "training": int(config.get("training", {}).get("random_seed", 0)),
        },
        "hashes": {
            name: value
            for name, path in tracked_inputs.items()
            if (value := _sha256(path)) is not None
        },
        "resolved_config_sha256": resolved_config_hash,
        "model_hashes": model_files,
        "artifact_hashes": artifact_hashes,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "operating_system": platform.platform(),
            "xgboost": xgboost_version,
            "torch": torch_version,
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
