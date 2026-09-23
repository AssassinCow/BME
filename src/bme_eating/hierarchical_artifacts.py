from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from bme_eating.config import feature_artifact_name

RUN_STAGES = (
    "CREATED",
    "STATE_COMPLETE",
    "PROPOSALS_COMPLETE",
    "VERIFIER_COMPLETE",
    "BOUNDARY_COMPLETE",
    "SELECTED",
    "EVALUATED",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    return value


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_yaml_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_value(project_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def validate_run_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,79}", name):
        raise ValueError("Run name must be 3-80 safe filename characters")
    return name


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if not key.startswith("_") and key not in {"credentials", "secrets"}
    }


def _tracked_inputs(config: dict[str, Any], input_root: Path) -> dict[str, Path]:
    feature_name = feature_artifact_name(config)
    return {
        "anchors": input_root / "indices" / "anchors.parquet",
        "events": input_root / "indices" / "events.parquet",
        "segments": input_root / "indices" / "segments.parquet",
        "subject_folds": input_root / "indices" / "subject_folds.json",
        "subject_folds_manifest": input_root / "indices" / "subject_folds.manifest.json",
        "quality_report": input_root / "indices" / "quality_report.json",
        "features": input_root / "features" / f"{feature_name}.parquet",
    }


def _verify_freeze_manifest(
    output_root: Path,
    run_name: str,
    fold: int,
    config_hash: str,
    git_commit: str | None,
) -> None:
    experiment_root = output_root / "experiments" / run_name
    fold_zero_manifest = experiment_root / "fold_0" / "run_manifest.json"
    if fold > 0 and fold_zero_manifest.is_file():
        reference = json.loads(fold_zero_manifest.read_text(encoding="utf-8"))
        if reference.get("resolved_config_sha256") != config_hash:
            raise RuntimeError("Fold configuration differs from the registered fold 0 protocol")
    if fold < 2:
        return
    freeze_path = experiment_root / "freeze_manifest.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("Folds 2-4 require a locked freeze_manifest.json")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("resolved_config_sha256") != config_hash:
        raise RuntimeError("Frozen stress fold configuration differs from the locked protocol")
    if freeze.get("git_commit") != git_commit:
        raise RuntimeError("Frozen stress folds must use the Git commit recorded at freeze time")


def assert_disjoint_subjects(**groups: set[str]) -> None:
    names = list(groups)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = {str(value) for value in groups[left_name]} & {
                str(value) for value in groups[right_name]
            }
            if overlap:
                raise RuntimeError(
                    f"Subject leakage between {left_name} and {right_name}: {sorted(overlap)}"
                )


def assert_oof_provenance(
    frame: pd.DataFrame,
    subject_partition: dict[str, int],
    partition_column: str = "calibration_fold",
) -> None:
    required = {"subject_key", partition_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OOF artifact is missing columns: {sorted(missing)}")
    actual = frame[["subject_key", partition_column]].drop_duplicates()
    if actual["subject_key"].duplicated().any():
        raise RuntimeError("An OOF subject appears in more than one prediction partition")
    for row in actual.itertuples(index=False):
        expected = subject_partition.get(str(row.subject_key))
        if expected is None or int(getattr(row, partition_column)) != int(expected):
            raise RuntimeError(f"Invalid OOF provenance for subject {row.subject_key}")


@dataclass(frozen=True)
class OuterLabelGuard:
    outer_subjects: frozenset[str]
    allow_outer_labels: bool = False

    def select_labels(self, events: pd.DataFrame, subjects: set[str]) -> pd.DataFrame:
        requested = {str(value) for value in subjects}
        forbidden = requested & set(self.outer_subjects)
        if forbidden and not self.allow_outer_labels:
            raise RuntimeError(
                "Outer-fold labels may only be read by the sealed evaluation command"
            )
        return events[events["subject_key"].astype(str).isin(requested)].copy()


@dataclass
class HierarchicalRun:
    root: Path
    manifest_path: Path
    payload: dict[str, Any]

    @property
    def stage(self) -> str:
        return str(self.payload["stage"])

    def require_stage(self, expected: str) -> None:
        if self.stage != expected:
            raise RuntimeError(f"Expected run stage {expected}, found {self.stage}")
        self.verify_artifacts()

    def verify_artifacts(self) -> None:
        for relative, expected in self.payload.get("artifact_hashes", {}).items():
            path = self.root / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"Manifested artifact changed or is missing: {relative}")

    def transition(
        self,
        next_stage: str,
        artifacts: list[Path],
        updates: dict[str, Any] | None = None,
    ) -> None:
        if next_stage not in RUN_STAGES:
            raise ValueError(f"Unknown hierarchical run stage: {next_stage}")
        current_index = RUN_STAGES.index(self.stage)
        next_index = RUN_STAGES.index(next_stage)
        if next_index != current_index + 1:
            raise RuntimeError(f"Invalid run transition: {self.stage} -> {next_stage}")
        hashes = dict(self.payload.get("artifact_hashes", {}))
        for path in artifacts:
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(self.root.resolve()).as_posix()
            except ValueError as error:
                raise ValueError(f"Artifact is outside the run directory: {path}") from error
            if not resolved.is_file():
                raise FileNotFoundError(f"Stage artifact is missing: {resolved}")
            hashes[relative] = sha256_file(resolved)
        self.payload["artifact_hashes"] = hashes
        self.payload["stage"] = next_stage
        if updates:
            self.payload.update(updates)
        write_json_atomic(self.manifest_path, self.payload)


def initialize_hierarchical_run(
    config: dict[str, Any],
    input_root: Path,
    output_root: Path,
    run_name: str,
    fold: int,
    *,
    fresh: bool,
) -> HierarchicalRun:
    validate_run_name(run_name)
    if fold not in range(int(config["data"]["subject_folds"])):
        raise ValueError("Outer fold is outside the configured fold range")
    run_root = output_root / "experiments" / run_name / f"fold_{fold}"
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.is_file():
        if fresh:
            raise FileExistsError(f"Run already exists: {run_root}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("run_name") != run_name or int(payload.get("outer_fold", -1)) != fold:
            raise RuntimeError("Existing run manifest identity is inconsistent")
        public_config = _public_config(config)
        config_hash = _canonical_hash(public_config)
        if payload.get("resolved_config_sha256") != config_hash:
            raise RuntimeError("Active configuration differs from the existing run manifest")
        project_root = Path(__file__).resolve().parents[2]
        git_commit = _git_value(project_root, "rev-parse", "HEAD")
        if payload.get("git", {}).get("commit") != git_commit:
            raise RuntimeError("Active Git commit differs from the existing run manifest")
        _verify_freeze_manifest(output_root, run_name, fold, config_hash, git_commit)
        if fold >= 2:
            freeze_path = output_root / "experiments" / run_name / "freeze_manifest.json"
            if payload.get("freeze_manifest_sha256") != sha256_file(freeze_path):
                raise RuntimeError("Frozen stress protocol manifest changed after run creation")
        tracked = _tracked_inputs(config, input_root)
        current_hashes = {
            name: sha256_file(path) for name, path in tracked.items() if path.is_file()
        }
        if payload.get("input_hashes") != current_hashes:
            raise RuntimeError("v2 inputs changed after the v3 run was initialized")
        return HierarchicalRun(run_root, manifest_path, payload)
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError("Run directory exists without a valid manifest")
    run_root.mkdir(parents=True, exist_ok=True)

    public_config = _public_config(config)
    config_hash = _canonical_hash(public_config)
    project_root = Path(__file__).resolve().parents[2]
    git_commit = _git_value(project_root, "rev-parse", "HEAD")
    _verify_freeze_manifest(output_root, run_name, fold, config_hash, git_commit)
    write_yaml_atomic(run_root / "resolved_config.yaml", public_config)
    tracked = _tracked_inputs(config, input_root)
    missing = [name for name, path in tracked.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required v2 inputs are missing: {missing}")
    input_hashes = {name: sha256_file(path) for name, path in tracked.items()}
    input_snapshot = {
        "version": 3,
        "input_artifact_schema_version": config["project"]["input_artifact_schema_version"],
        "hashes": input_hashes,
    }
    snapshot_path = output_root / "input_snapshot.json"
    if snapshot_path.is_file():
        existing = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if existing != input_snapshot:
            raise RuntimeError("The v3 input snapshot conflicts with existing v2 input hashes")
    else:
        write_json_atomic(snapshot_path, input_snapshot)

    dirty = bool(_git_value(project_root, "status", "--porcelain"))
    from bme_eating.models.factory import build_state_model

    state_model = build_state_model(config["model"])
    parameter_count = sum(parameter.numel() for parameter in state_model.parameters())
    del state_model
    payload = {
        "version": 3,
        "protocol_version": 3,
        "run_name": run_name,
        "outer_fold": fold,
        "stage": "CREATED",
        "git": {
            "commit": git_commit,
            "dirty": dirty,
        },
        "command": [Path(value).name if Path(value).is_absolute() else value for value in sys.argv],
        "resolved_config_sha256": config_hash,
        "input_hashes": input_hashes,
        "random_seeds": {
            "state": int(config["training"]["random_seed"]),
            "verifier": [int(value) for value in config["verifier"]["seeds"]],
            "boundary": [int(value) for value in config["boundary"]["seeds"]],
        },
        "state_model_parameter_count": parameter_count,
        "artifact_hashes": {
            "resolved_config.yaml": sha256_file(run_root / "resolved_config.yaml")
        },
        "maximum_future_context_seconds": int(
            config["hierarchical"]["maximum_event_latency_seconds"]
        ),
        "metric_contract": {
            "event_iou_operator": ">",
            "event_iou_threshold": float(config["postprocess"]["iou_threshold"]),
            "matching_method": str(config["postprocess"]["matching_method"]),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    if fold >= 2:
        payload["freeze_manifest_sha256"] = sha256_file(
            output_root / "experiments" / run_name / "freeze_manifest.json"
        )
    write_json_atomic(manifest_path, payload)
    return HierarchicalRun(run_root, manifest_path, payload)
