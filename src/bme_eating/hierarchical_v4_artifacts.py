from __future__ import annotations

import json
import platform
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml

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
from bme_eating.v4_protocol import (
    BLOCKED_PREDECESSORS,
    CALIBRATION_PROTOCOL,
    CODE_VERSION,
    DECODER_PROTOCOL,
    INPUT_SNAPSHOT_FILENAME,
    POOLED_HEAD_PROTOCOL,
    PROTOCOL_VERSION,
    RAW_INPUT_SCHEMA,
    TARGET_SEMANTICS,
    validate_r3_config,
)

RESUME_RUNTIME_CONFIG_PATHS = frozenset(
    {
        "training.num_workers",
        "training.inference_num_workers",
        "training.inference_batch_size",
        "training.inference_resume_chunk_rows",
    }
)
RESUME_POLICY_CONFIG_KEYS = frozenset(
    {
        "enforce_git_identity_on_resume",
        "enforce_runtime_source_identity_on_resume",
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
    project = normalized.get("project")
    if isinstance(project, dict):
        for key in RESUME_POLICY_CONFIG_KEYS:
            project.pop(key, None)
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


def _tracked_inputs(config: dict[str, Any], input_root: Path) -> dict[str, Path]:
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
        "canonical_anchors": canonical["anchors"],
        "canonical_statistics": canonical["statistics"],
        "canonical_events": canonical["events"],
        "canonical_preparation_identity": canonical["preparation_identity"],
        "canonical_anchors_identity": canonical["anchors_identity"],
        "canonical_manifest": canonical["manifest"],
    }


def _input_snapshot_matches(existing: dict[str, Any], current: dict[str, Any]) -> bool:
    existing_hashes = existing.get("hashes")
    current_hashes = current.get("hashes")
    return (
        existing.get("version") in {5, 6}
        and existing.get("protocol_version") == current.get("protocol_version")
        and existing.get("input_artifact_schema_version")
        == current.get("input_artifact_schema_version")
        and isinstance(existing_hashes, dict)
        and isinstance(current_hashes, dict)
        and all(existing_hashes.get(name) == digest for name, digest in current_hashes.items())
    )


def feature_provenance_artifact_path(
    output_root: Path, provenance: dict[str, Any]
) -> Path:
    return output_root / "feature_provenance" / f"{_canonical_hash(provenance)}.json"


def _m1_comparison_identity(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(_public_config(config))
    experiment = normalized.get("experiment", {})
    for key in ("name", "ablation_id", "variant", "state_promotion"):
        experiment.pop(key, None)
    normalized.setdefault("decoder", {})["use_semi_markov"] = False
    return normalized


def _p1_comparison_identity(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(_public_config(config))
    experiment = normalized.get("experiment", {})
    for key in ("name", "ablation_id", "variant", "motion_parent"):
        experiment.pop(key, None)
    normalized.setdefault("model", {})["use_ppg"] = False
    return normalized


def _validate_p1_parent(config: dict[str, Any], output_root: Path) -> None:
    experiment = config.get("experiment", {})
    if str(experiment.get("ablation_id", "")) != "R3-P1":
        return
    parent = experiment.get("motion_parent")
    if not isinstance(parent, dict):
        raise TypeError(
            "R3-P1 must be built from the passing motion winner with prepare_hierarchical_v4_p1.py"
        )
    run_name = str(parent.get("run_name", ""))
    source_root = output_root / "experiments" / run_name
    source_config_path = source_root / "fold_0" / "resolved_config.yaml"
    gate_path = source_root / "ablation" / "fold_0_report.json"
    expected = {
        "run_name": run_name,
        "resolved_config_sha256": sha256_file(source_config_path),
        "fold0_gate_sha256": sha256_file(gate_path),
    }
    if parent != expected:
        raise RuntimeError("R3-P1 motion-parent hashes do not match the selected run")
    gate_report = json.loads(gate_path.read_text(encoding="utf-8"))
    if not bool(gate_report.get("passed", False)) or gate_report.get("selected_run") != run_name:
        raise RuntimeError("R3-P1 motion parent is not the passing selected fold-0 run")
    verify_gate_evidence(output_root.parent, gate_report)
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8")) or {}
    if bool(source_config.get("model", {}).get("use_ppg", False)):
        raise RuntimeError("R3-P1 motion parent must not use PPG")
    if _p1_comparison_identity(config) != _p1_comparison_identity(source_config):
        raise RuntimeError("R3-P1 differs from its motion parent beyond PPG enablement")


def _validate_m1_promotion(config: dict[str, Any], output_root: Path) -> None:
    experiment = config.get("experiment", {})
    if str(experiment.get("ablation_id", "")) != "R3-M1":
        return
    decision = experiment.get("state_promotion")
    if decision is None:
        raise RuntimeError(
            "R3-M1 requires hash-locked motion/PPG promotion evidence; "
            "run prepare_hierarchical_v4_m1.py first"
        )
    if not isinstance(decision, dict):
        raise TypeError("R3-M1 state promotion evidence must be a mapping")
    if decision.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("R3-M1 state promotion evidence uses a different protocol")
    if set(decision.get("source_runs", {})) != {"MOTION", "PPG"}:
        raise RuntimeError("PPG promotion evidence must identify motion and PPG source runs")
    if "resolved_use_ppg" not in decision:
        raise RuntimeError("R3-M1 state promotion evidence has no resolved PPG decision")
    verify_gate_evidence(output_root.parent, decision)
    source_runs = decision["source_runs"]
    recomputed = evaluate_ppg_promotion(
        output_root,
        s2_run=str(source_runs["MOTION"]),
        s3_run=str(source_runs["PPG"]),
        gate=config["promotion_gate"],
    )
    if decision != recomputed:
        raise RuntimeError("R3-M1 state promotion decision does not match its locked evidence")
    configured_use_ppg = bool(config.get("model", {}).get("use_ppg", True))
    if configured_use_ppg != bool(decision["resolved_use_ppg"]):
        raise RuntimeError("R3-M1 model.use_ppg differs from its locked state promotion decision")
    selected_key = "PPG" if bool(decision["resolved_use_ppg"]) else "MOTION"
    selected_run = str(source_runs[selected_key])
    selected_path = output_root / "experiments" / selected_run / "fold_0" / "resolved_config.yaml"
    selected_config = yaml.safe_load(selected_path.read_text(encoding="utf-8")) or {}
    if _m1_comparison_identity(config) != _m1_comparison_identity(selected_config):
        raise RuntimeError("R3-M1 differs from the promoted state architecture beyond Semi-Markov")
    if not bool(config.get("decoder", {}).get("use_semi_markov", False)):
        raise RuntimeError("R3-M1 must enable the coherent Semi-Markov decoder")


def current_v4_identity(config: dict[str, Any], input_root: Path) -> dict[str, Any]:
    validate_r3_config(config)
    output_root = input_root.parent / str(config["project"]["artifact_schema_version"])
    _validate_p1_parent(config, output_root)
    _validate_m1_promotion(config, output_root)
    tracked = _tracked_inputs(config, input_root)
    missing = [name for name, path in tracked.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required StatsFusion inputs are missing: {missing}")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "pooled_head_protocol": (
            POOLED_HEAD_PROTOCOL
            if config.get("hierarchical", {}).get("downstream_mode")
            in {"pooled_heads", "pooled_deep_only"}
            else None
        ),
        "resolved_config_sha256": resume_config_hash(config),
        "git": git_worktree_identity(Path(__file__).resolve().parents[2]),
        "input_hashes": {name: sha256_file(path) for name, path in tracked.items()},
    }


def validate_v4_freeze_manifest(
    freeze_path: Path,
    output_root: Path,
    *,
    expected_resume_config_sha256: str,
    expected_git: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if not freeze_path.is_file():
        raise FileNotFoundError("V4 folds 2-4 require freeze_manifest.json")
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected_fields = {
        "code_version": CODE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "blocked_predecessors": list(BLOCKED_PREDECESSORS),
        "candidate_minimum_seconds": int(config["decoder"]["candidate_minimum_seconds"]),
        "candidate_maximum_seconds": int(config["decoder"]["candidate_maximum_seconds"]),
        "resume_config_sha256": expected_resume_config_sha256,
    }
    if bool(config.get("project", {}).get("enforce_git_identity_on_resume", False)):
        expected_fields["git"] = expected_git
    for key, expected in expected_fields.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"V4 freeze manifest has an invalid {key}")
    if payload.get("locked_after_folds") != [0, 1]:
        raise RuntimeError("V4 freeze manifest must be locked after folds 0 and 1")
    if payload.get("selected_run") != freeze_path.parent.name:
        raise RuntimeError("V4 freeze manifest identifies a different selected run")
    verify_gate_evidence(output_root.parent, payload)
    return payload


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
    validate_r3_config(config)
    if fold not in range(int(config["data"]["subject_folds"])):
        raise ValueError("Outer fold is outside the configured fold range")
    _validate_p1_parent(config, output_root)
    _validate_m1_promotion(config, output_root)
    run_root = output_root / "experiments" / run_name / f"fold_{fold}"
    manifest_path = run_root / "run_manifest.json"
    public_config = _public_config(config)
    config_hash = resume_config_hash(config)
    project_root = Path(__file__).resolve().parents[2]
    git_identity = git_worktree_identity(project_root)
    experiment_root = output_root / "experiments" / run_name
    time_constrained_single_holdout = bool(
        config.get("experiment", {}).get("time_constrained_single_holdout", False)
    )
    fold_zero_manifest = experiment_root / "fold_0" / "run_manifest.json"
    reference = None
    if fold > 0 and fold_zero_manifest.is_file():
        reference = json.loads(fold_zero_manifest.read_text(encoding="utf-8"))
        reference_hash = _saved_resume_config_hash(fold_zero_manifest.parent, reference)
        if reference_hash != config_hash:
            raise RuntimeError("V4 fold configuration differs from fold 0")
    freeze_path = experiment_root / "freeze_manifest.json"
    if fold >= 2 and not time_constrained_single_holdout:
        validate_v4_freeze_manifest(
            freeze_path,
            output_root,
            expected_resume_config_sha256=config_hash,
            expected_git=git_identity,
            config=config,
        )
    tracked = _tracked_inputs(config, input_root)
    missing = [name for name, path in tracked.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required StatsFusion inputs are missing: {missing}")
    input_hashes = {name: sha256_file(path) for name, path in tracked.items()}
    if manifest_path.is_file():
        if fresh:
            raise FileExistsError(f"V4 run already exists: {run_root}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(f"Blocked predecessor runs cannot be resumed as {PROTOCOL_VERSION}")
        saved_config_hash = _saved_resume_config_hash(run_root, payload)
        if saved_config_hash != config_hash:
            raise RuntimeError("Active v4 configuration differs from the run manifest")
        if payload.get("input_hashes") != input_hashes:
            raise RuntimeError("V2 inputs changed after the v4 run was initialized")
        if (
            fold >= 2
            and not time_constrained_single_holdout
            and payload.get("freeze_manifest_sha256") != sha256_file(freeze_path)
        ):
            raise RuntimeError("V4 freeze manifest changed after run initialization")
        run = HierarchicalRun(run_root, manifest_path, payload)
        run.verify_artifacts()
        previous_git = payload.get("git")
        if previous_git != git_identity:
            if bool(
                config.get("project", {}).get("enforce_git_identity_on_resume", False)
            ):
                raise RuntimeError("Active v4 worktree differs from the run manifest")
            payload.setdefault("source_identity_history", []).append(
                {"previous": previous_git, "active": git_identity}
            )
            payload["git"] = git_identity
        payload.pop("recompute_pretraining_scalers", None)
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
    snapshot = {
        "version": 6,
        "protocol_version": PROTOCOL_VERSION,
        "input_artifact_schema_version": "v2",
        "hashes": input_hashes,
    }
    snapshot_path = output_root / INPUT_SNAPSHOT_FILENAME
    if snapshot_path.is_file():
        existing_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if not _input_snapshot_matches(existing_snapshot, snapshot):
            raise RuntimeError("V4 input snapshot conflicts with current v2 artifacts")
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
    provenance_path = feature_provenance_artifact_path(output_root, provenance)
    if provenance_path.is_file() and (
        json.loads(provenance_path.read_text(encoding="utf-8")) != provenance
    ):
        raise RuntimeError("V4 feature provenance changed after initialization")
    from bme_eating.models.factory import build_state_model

    state_model = build_state_model(config["model"])
    parameter_count = sum(parameter.numel() for parameter in state_model.parameters())
    canonical_manifest = json.loads(tracked["canonical_manifest"].read_text(encoding="utf-8"))
    if not snapshot_path.is_file():
        write_json_atomic(snapshot_path, snapshot)
    if not provenance_path.is_file():
        write_json_atomic(provenance_path, provenance)
    run_root.mkdir(parents=True, exist_ok=True)
    write_yaml_atomic(run_root / "resolved_config.yaml", public_config)
    payload = {
        "version": 5,
        "code_version": CODE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "blocked_predecessors": list(BLOCKED_PREDECESSORS),
        "target_semantics": TARGET_SEMANTICS,
        "calibration_protocol": CALIBRATION_PROTOCOL,
        "decoder_protocol": DECODER_PROTOCOL,
        "raw_input_schema": RAW_INPUT_SCHEMA,
        "promotion_protocol": (
            "time_constrained_single_holdout_v1"
            if time_constrained_single_holdout
            else "registered_ablation_gates_v1"
        ),
        "candidate_minimum_seconds": int(config["decoder"]["candidate_minimum_seconds"]),
        "feature_code_sha256": canonical_manifest["feature_code_sha256"],
        "run_name": run_name,
        "outer_fold": int(fold),
        "stage": "CREATED",
        "git": git_identity,
        "command": [Path(value).name if Path(value).is_absolute() else value for value in sys.argv],
        "resolved_config_sha256": _canonical_hash(public_config),
        "resume_config_sha256": config_hash,
        "runtime_config": _runtime_config_snapshot(config),
        "input_hashes": input_hashes,
        "feature_provenance_path": provenance_path.relative_to(output_root).as_posix(),
        "feature_provenance_sha256": sha256_file(provenance_path),
        "random_seeds": {
            "state": [
                int(value)
                for value in config.get("final_training", {}).get(
                    "state_seeds", [config["training"]["random_seed"]]
                )
            ],
            "verifier": (
                []
                if config.get("hierarchical", {}).get("downstream_mode")
                in {"state_only", "pooled_logistic", "pooled_heads", "pooled_deep_only"}
                else [int(value) for value in config["verifier"]["seeds"]]
            ),
            "boundary": (
                []
                if config.get("hierarchical", {}).get("downstream_mode")
                in {"state_only", "pooled_logistic", "pooled_heads", "pooled_deep_only"}
                else [int(value) for value in config["boundary"]["seeds"]]
            ),
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
    if fold >= 2 and not time_constrained_single_holdout:
        payload["freeze_manifest_sha256"] = sha256_file(freeze_path)
    write_json_atomic(manifest_path, payload)
    return HierarchicalRun(run_root, manifest_path, payload)
