from __future__ import annotations

import json
import platform
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml

from bme_eating.config import feature_artifact_name
from bme_eating.data.stats_fusion_inputs import (
    canonical_input_paths,
    verify_canonical_statsfusion_inputs,
)
from bme_eating.hierarchical_artifacts import (
    HierarchicalRun,
    _canonical_hash,
    sha256_file,
    validate_run_name,
    write_json_atomic,
    write_yaml_atomic,
)
from bme_eating.hierarchical_v4_gates import evaluate_ppg_promotion, verify_gate_evidence
from bme_eating.reproducibility import git_worktree_identity
from bme_eating.stats_features import audit_feature_provenance

RESUME_RUNTIME_CONFIG_PATHS = frozenset(
    {
        "training.num_workers",
        "training.inference_num_workers",
        "training.inference_batch_size",
        "training.inference_resume_chunk_rows",
    }
)


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if not key.startswith("_") and key not in {"credentials", "secrets"}
    }


def _resume_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(_public_config(config))
    training = normalized.get("training")
    if isinstance(training, dict):
        for path in RESUME_RUNTIME_CONFIG_PATHS:
            section, key = path.split(".", 1)
            if section == "training":
                training.pop(key, None)
    return normalized


def resume_config_hash(config: dict[str, Any]) -> str:
    """Hash only settings that can change model outputs during a resume."""
    return _canonical_hash(_resume_config(config))


def _saved_resume_config_hash(run_root: Path, payload: dict[str, Any]) -> str:
    resolved_path = run_root / "resolved_config.yaml"
    if resolved_path.is_file():
        expected = payload.get("artifact_hashes", {}).get("resolved_config.yaml")
        if expected and sha256_file(resolved_path) != expected:
            raise RuntimeError("V4 saved configuration changed after run initialization")
        saved_config = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
        original_hash = payload.get("resolved_config_sha256")
        if original_hash and _canonical_hash(saved_config) != original_hash:
            raise RuntimeError("V4 saved configuration does not match the run manifest")
        return resume_config_hash(saved_config)
    saved_hash = payload.get("resolved_config_sha256")
    if not saved_hash:
        raise RuntimeError("V4 run has no resumable configuration identity")
    return str(saved_hash)


def _runtime_config_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training", {})
    if not isinstance(training, dict):
        return {}
    return {
        path: training.get(path.split(".", 1)[1])
        for path in sorted(RESUME_RUNTIME_CONFIG_PATHS)
        if path.split(".", 1)[1] in training
    }


def _has_only_pretraining_artifacts(run_root: Path) -> bool:
    allowed = {"resolved_config.yaml", "run_manifest.json"}
    return all(
        path.relative_to(run_root).as_posix() in allowed
        or (
            path.name in {"statistics_scaler.json", "sensor_normalization.json"}
            and len(path.relative_to(run_root).parts) == 4
            and path.relative_to(run_root).parts[0] == "crossfit"
            and path.relative_to(run_root).parts[1].startswith("partition_")
            and path.relative_to(run_root).parts[2] == "state"
        )
        for path in run_root.rglob("*")
        if path.is_file()
    )


def _tracked_inputs(config: dict[str, Any], input_root: Path) -> dict[str, Path]:
    feature_name = feature_artifact_name(config)
    output_root = input_root.parent / str(config["project"]["artifact_schema_version"])
    canonical = canonical_input_paths(output_root)
    verify_canonical_statsfusion_inputs(input_root, output_root)
    return {
        "anchors": input_root / "indices" / "anchors.parquet",
        "events": input_root / "indices" / "events.parquet",
        "segments": input_root / "indices" / "segments.parquet",
        "subject_folds": input_root / "indices" / "subject_folds.json",
        "subject_folds_manifest": input_root / "indices" / "subject_folds.manifest.json",
        "quality_report": input_root / "indices" / "quality_report.json",
        "features": input_root / "features" / f"{feature_name}.parquet",
        "canonical_anchors": canonical["anchors"],
        "canonical_statistics": canonical["statistics"],
        "canonical_preparation_identity": canonical["preparation_identity"],
        "canonical_anchors_identity": canonical["anchors_identity"],
        "canonical_manifest": canonical["manifest"],
    }


def _validate_s4_promotion(config: dict[str, Any], output_root: Path) -> None:
    experiment = config.get("experiment", {})
    if str(experiment.get("ablation_id", "")) != "S4":
        return
    decision = experiment.get("ppg_promotion")
    if decision is None:
        raise RuntimeError(
            "S4 requires hash-locked S2/S3 PPG promotion evidence; "
            "run prepare_hierarchical_v4_s4.py first"
        )
    if not isinstance(decision, dict):
        raise TypeError("S4 PPG promotion evidence must be a mapping")
    if decision.get("protocol_version") != "statsfusion-r2":
        raise RuntimeError("S4 PPG promotion evidence uses a different protocol")
    if set(decision.get("source_runs", {})) != {"S2", "S3"}:
        raise RuntimeError("S4 PPG promotion evidence must identify S2 and S3 source runs")
    if "resolved_use_ppg" not in decision:
        raise RuntimeError("S4 PPG promotion evidence has no resolved PPG decision")
    verify_gate_evidence(output_root.parent, decision)
    source_runs = decision["source_runs"]
    recomputed = evaluate_ppg_promotion(
        output_root,
        s2_run=str(source_runs["S2"]),
        s3_run=str(source_runs["S3"]),
        gate=config["promotion_gate"],
    )
    if decision != recomputed:
        raise RuntimeError("S4 PPG promotion decision does not match its locked evidence")
    configured_use_ppg = bool(config.get("model", {}).get("use_ppg", True))
    if configured_use_ppg != bool(decision["resolved_use_ppg"]):
        raise RuntimeError("S4 model.use_ppg differs from its locked PPG promotion decision")


def current_v4_identity(config: dict[str, Any], input_root: Path) -> dict[str, Any]:
    if config["project"].get("artifact_schema_version") != "v4":
        raise ValueError("StatsFusion runs must write to artifact schema v4")
    if config.get("experiment", {}).get("protocol_version") != "statsfusion-r2":
        raise ValueError("Formal v4 runs require protocol_version: statsfusion-r2")
    if not bool(config["project"].get("strict_resume_identity", False)):
        raise ValueError("StatsFusion v4 requires strict_resume_identity: true")
    output_root = input_root.parent / str(config["project"]["artifact_schema_version"])
    _validate_s4_promotion(config, output_root)
    tracked = _tracked_inputs(config, input_root)
    missing = [name for name, path in tracked.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required StatsFusion inputs are missing: {missing}")
    return {
        "protocol_version": "statsfusion-r2",
        "resolved_config_sha256": resume_config_hash(config),
        "git": git_worktree_identity(Path(__file__).resolve().parents[2]),
        "input_hashes": {name: sha256_file(path) for name, path in tracked.items()},
    }


def initialize_v4_run(
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
    if config["project"].get("artifact_schema_version") != "v4":
        raise ValueError("StatsFusion runs must write to artifact schema v4")
    if config.get("experiment", {}).get("protocol_version", "statsfusion-r2") != "statsfusion-r2":
        raise ValueError("Formal v4 runs require protocol_version: statsfusion-r2")
    if not bool(config["project"].get("strict_resume_identity", False)):
        raise ValueError("StatsFusion v4 requires strict_resume_identity: true")
    _validate_s4_promotion(config, output_root)
    run_root = output_root / "experiments" / run_name / f"fold_{fold}"
    manifest_path = run_root / "run_manifest.json"
    public_config = _public_config(config)
    config_hash = resume_config_hash(config)
    project_root = Path(__file__).resolve().parents[2]
    git_identity = git_worktree_identity(project_root)
    experiment_root = output_root / "experiments" / run_name
    fold_zero_manifest = experiment_root / "fold_0" / "run_manifest.json"
    reference = None
    if fold > 0 and fold_zero_manifest.is_file():
        reference = json.loads(fold_zero_manifest.read_text(encoding="utf-8"))
        reference_hash = _saved_resume_config_hash(
            fold_zero_manifest.parent, reference
        )
        if reference_hash != config_hash:
            raise RuntimeError("V4 fold configuration differs from fold 0")
    freeze_path = experiment_root / "freeze_manifest.json"
    if fold >= 2:
        if not freeze_path.is_file():
            raise FileNotFoundError("V4 folds 2-4 require freeze_manifest.json")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        freeze_hash = freeze.get("resume_config_sha256")
        if freeze_hash is None and reference is not None:
            if freeze.get("resolved_config_sha256") != reference.get("resolved_config_sha256"):
                raise RuntimeError("V4 frozen configuration differs from fold 0")
            freeze_hash = _saved_resume_config_hash(fold_zero_manifest.parent, reference)
        if freeze_hash != config_hash:
            raise RuntimeError("V4 frozen configuration differs from the locked protocol")
        if freeze.get("git") != git_identity:
            raise RuntimeError("V4 frozen worktree differs from the locked protocol")
    tracked = _tracked_inputs(config, input_root)
    missing = [name for name, path in tracked.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required StatsFusion inputs are missing: {missing}")
    input_hashes = {name: sha256_file(path) for name, path in tracked.items()}
    if manifest_path.is_file():
        if fresh:
            raise FileExistsError(f"V4 run already exists: {run_root}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        saved_config_hash = _saved_resume_config_hash(run_root, payload)
        if saved_config_hash != config_hash:
            raise RuntimeError("Active v4 configuration differs from the run manifest")
        if payload.get("input_hashes") != input_hashes:
            raise RuntimeError("V2 inputs changed after the v4 run was initialized")
        if fold >= 2 and payload.get("freeze_manifest_sha256") != sha256_file(freeze_path):
            raise RuntimeError("V4 freeze manifest changed after run initialization")
        run = HierarchicalRun(run_root, manifest_path, payload)
        run.verify_artifacts()
        previous_git = payload.get("git")
        if previous_git != git_identity:
            if (
                fold >= 2
                or payload.get("stage") != "CREATED"
                or not _has_only_pretraining_artifacts(run_root)
                or not isinstance(previous_git, dict)
                or previous_git.get("commit") != git_identity.get("commit")
            ):
                raise RuntimeError("Active v4 worktree differs from the run manifest")
            payload.setdefault("pretraining_worktree_history", []).append(
                {"previous": previous_git, "active": git_identity}
            )
            payload["git"] = git_identity
            payload["recompute_pretraining_scalers"] = True
        if "runtime_config" in payload:
            previous_runtime = payload["runtime_config"]
        else:
            saved_config = yaml.safe_load(
                (run_root / "resolved_config.yaml").read_text(encoding="utf-8")
            )
            previous_runtime = _runtime_config_snapshot(saved_config)
        current_runtime = _runtime_config_snapshot(config)
        if previous_runtime != current_runtime:
            payload.setdefault("runtime_config_history", []).append(
                {
                    "previous": previous_runtime,
                    "active": current_runtime,
                }
            )
        if payload.get("runtime_config") != current_runtime or previous_git != git_identity:
            payload["runtime_config"] = current_runtime
            payload["resume_config_sha256"] = config_hash
            write_json_atomic(manifest_path, payload)
        return run
    if not fresh:
        raise FileNotFoundError("V4 run does not exist; initialize it with --fresh")
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError("V4 run directory exists without a valid manifest")
    run_root.mkdir(parents=True, exist_ok=True)
    write_yaml_atomic(run_root / "resolved_config.yaml", public_config)
    snapshot = {
        "version": 4,
        "protocol_version": "statsfusion-r2",
        "input_artifact_schema_version": "v2",
        "hashes": input_hashes,
    }
    snapshot_path = output_root / "input_snapshot.json"
    if snapshot_path.is_file():
        if json.loads(snapshot_path.read_text(encoding="utf-8")) != snapshot:
            raise RuntimeError("V4 input snapshot conflicts with current v2 artifacts")
    else:
        write_json_atomic(snapshot_path, snapshot)
    provenance_config = config["feature_provenance"]
    provenance = audit_feature_provenance(
        project_root=project_root,
        input_root=input_root,
        source_paths=provenance_config["source_paths"],
        selection_note_path=provenance_config.get("selection_note_path"),
        assumed_used_all_outer_folds=bool(
            provenance_config.get("assumed_used_all_outer_folds", True)
        ),
    )
    provenance_path = output_root / "feature_provenance.json"
    if provenance_path.is_file():
        if json.loads(provenance_path.read_text(encoding="utf-8")) != provenance:
            raise RuntimeError("V4 feature provenance changed after initialization")
    else:
        write_json_atomic(provenance_path, provenance)
    from bme_eating.models.factory import build_state_model

    state_model = build_state_model(config["model"])
    parameter_count = sum(parameter.numel() for parameter in state_model.parameters())
    payload = {
        "version": 4,
        "protocol_version": "statsfusion-r2",
        "blocked_predecessors": ["statsfusion-r0-blocked", "statsfusion-r1-blocked"],
        "run_name": run_name,
        "outer_fold": int(fold),
        "stage": "CREATED",
        "git": git_identity,
        "command": [Path(value).name if Path(value).is_absolute() else value for value in sys.argv],
        "resolved_config_sha256": _canonical_hash(public_config),
        "resume_config_sha256": config_hash,
        "runtime_config": _runtime_config_snapshot(config),
        "input_hashes": input_hashes,
        "feature_provenance_sha256": sha256_file(provenance_path),
        "random_seeds": {
            "state": [
                int(value)
                for value in config.get("final_training", {}).get(
                    "state_seeds", [config["training"]["random_seed"]]
                )
            ],
            "verifier": [int(value) for value in config["verifier"]["seeds"]],
            "boundary": [int(value) for value in config["boundary"]["seeds"]],
        },
        "state_model_parameter_count": parameter_count,
        "artifact_hashes": {"resolved_config.yaml": sha256_file(run_root / "resolved_config.yaml")},
        "maximum_future_context_seconds": int(
            config["hierarchical"]["maximum_event_latency_seconds"]
        ),
        "metric_contract": {
            "event_iou_operator": ">",
            "event_iou_threshold": 0.25,
            "matching_methods_reported": ["max_cardinality_iou", "greedy"],
        },
        "xgboost_policy": {
            "models": "forbidden",
            "predictions": "forbidden",
            "events": "forbidden",
            "candidates": "forbidden",
            "distillation": "forbidden",
            "retained_statistics_only": True,
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
        payload["freeze_manifest_sha256"] = sha256_file(freeze_path)
    write_json_atomic(manifest_path, payload)
    return HierarchicalRun(run_root, manifest_path, payload)
