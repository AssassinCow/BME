from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import uuid
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_capabilities import capabilities, supported

PROTOCOL = "statsfusion-joint-event-v1"
RUNTIME_KEYS = {"num_workers", "loader_prefetch_factor", "session_reader_cache_size", "derived_cache_mib"}
STAGES = ("audit", "validate", "outer", "gates", "final", "export", "replay")
V411_PROTOCOL = "statsfusion-joint-event-v2"
V412_PROTOCOL = "statsfusion-joint-event-v3"
V413_PROTOCOL = "statsfusion-joint-event-v4"
V414_PROTOCOL = "statsfusion-joint-event-v5"


def grouped_protocol(config):
    return supported(config, "grouped")


def event_score_formula(config):
    return capabilities(config).score_formula


def resolve_reference(output_base, name):
    matches = [output_base / schema / "runs" / name for schema in ("v410", "v411", "v412", "v413", "v414")
               if (output_base / schema / "runs" / name / "identity.json").is_file()]
    if len({sha256_file(path / "identity.json") for path in matches}) > 1:
        raise RuntimeError("Reference run name has conflicting identities across schemas")
    return matches[0] if matches else output_base / "v411" / "runs" / name


def protocol_for(config):
    return config["experiment"]["protocol_version"]


def stages_for(config):
    if supported(config, "full_data_workflow"):
        return ("audit", "dev", "final39", "export", "replay")
    if execution_mode(config) == "fold0":
        return STAGES
    return ("audit", "validate", "outer", "ablation", "gates", "final", "export", "replay") if (
        protocol_for(config) == V411_PROTOCOL) else STAGES


def execution_mode(config):
    return config.get("execution", {}).get("mode", "fivefold")


def execution_folds(config):
    return [0] if execution_mode(config) == "fold0" else list(range(5))


def internal_partitions(config):
    mode = config.get("execution", {}).get("internal_mode", "oof")
    if mode not in {"single", "oof"}:
        raise ValueError("Internal mode must be single or oof")
    if mode == "single":
        if execution_mode(config) != "fold0" or not grouped_protocol(config):
            raise ValueError("Single partition is restricted to grouped fold0 diagnostics")
        return [0]
    if supported(config, "consistent_inputs") and execution_mode(config) != "fivefold":
        raise ValueError("v412 requires fold0/single or fivefold/oof")
    return list(range(3))


def calibration_scope(config):
    return {"calibration_kind": "single_partition_holdout" if internal_partitions(config) == [0]
            else "three_partition_oof", "full_internal_oof": internal_partitions(config) != [0]}


def model_run_identity(run):
    path = run / "single_partition_migration.json"
    if not path.exists():
        return sha256_file(run / "identity.json")
    from bme_eating.joint_partition_migration import verify_partition_migration
    return verify_partition_migration(run)["original_identity_sha256"]


def configure_numerics(config: dict[str, Any]) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(int(config["resources"]["cpu_threads"]))
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


class GPUResourcesBusy(RuntimeError):
    pass


def require_idle_gpu(device: str) -> None:
    if device != "cuda":
        return
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True)
    others = []
    for line in result.stdout.splitlines():
        fields = line.split(",", 1)
        if len(fields) != 2:
            continue
        process_id, name = fields[0].strip(), Path(fields[1].strip()).name.lower()
        if process_id != str(os.getpid()) and name.startswith("python"):
            others.append(process_id)
    if others:
        raise GPUResourcesBusy("Other Python processes use the GPU; resume after existing training releases resources")


def digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def config_identity(config: dict[str, Any]) -> str:
    payload = deepcopy({name: value for name, value in config.items() if not name.startswith("_")})
    for name in RUNTIME_KEYS:
        payload.get("training", {}).pop(name, None)
    return digest(payload)


def source_identity(root: Path) -> dict[str, str]:
    paths = sorted((root / "src" / "bme_eating").rglob("*.py"))
    paths.extend(sorted((root / "scripts").glob("*.py")))
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def verify_artifact_hashes(root: Path, hashes: dict[str, str]) -> None:
    root = root.resolve()
    if not hashes:
        raise RuntimeError("Artifact evidence is missing")
    for relative, expected in hashes.items():
        target = (root / relative).resolve()
        if (not target.is_relative_to(root) or not target.is_file()
                or not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256_file(target) != expected):
            raise RuntimeError(f"Artifact checksum mismatch or unsafe path: {relative}")


@contextmanager
def run_lock(run: Path):
    with (run / "execution.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another process owns this run; concurrent resume is forbidden") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def runtime_identity() -> dict[str, Any]:
    return {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
            "cuda": torch.version.cuda, "platform": platform.platform(),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def resume_identity_differences(stored: dict[str, Any], current: dict[str, Any]) -> list[str]:
    def differences(left, right, prefix):
        if isinstance(left, dict) and isinstance(right, dict):
            result = []
            for name in sorted(left.keys() | right.keys()):
                path = f"{prefix}.{name}" if prefix else name
                if path == "git_head":
                    continue
                if name not in left or name not in right:
                    result.append(path)
                else:
                    result.extend(differences(left[name], right[name], path))
            return result
        return [prefix] if left != right else []
    return differences(stored, current, "")


def record_resume_git(run: Path, stored: dict[str, Any], current: dict[str, Any]) -> None:
    path = run / "resume_git_history.json"
    with run_lock(run):
        history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "identity_sha256": sha256_file(run / "identity.json"), "observations": []}
        observation = {"stored_git_head": stored.get("git_head"), "git_head": current.get("git_head"),
                       "source_sha256": digest(current["source"])}
        if any(all(entry.get(key) == value for key, value in observation.items())
               for entry in history["observations"]):
            return
        history["observations"].append({**observation, "observed_at": datetime.now(UTC).isoformat()})
        write_json_atomic(path, history)


def validate_config(config: dict[str, Any]) -> None:
    full_data = supported(config, "full_data_workflow")
    if full_data:
        if (execution_mode(config) != "dev28_11" or config["execution"].get("internal_mode") != "development"
                or config["execution"].get("deployment_boundary_mode") != "off"
                or config["development"].get("final_calibration_kind") != "development_on_final_model"
                or not re.fullmatch(r"[0-9a-f]{64}", config["development"].get("split_sha256", ""))):
            raise ValueError("v414 requires a frozen dev28_11 split and explicit final development calibration")
    else:
        if execution_mode(config) not in {"fold0", "fivefold"}:
            raise ValueError("Execution mode must be fold0 or fivefold")
        internal_partitions(config)
    def finite_values(value):
        if isinstance(value, dict):
            return all(finite_values(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_values(item) for item in value)
        return not isinstance(value, (int, float)) or math.isfinite(value)
    if not finite_values(config):
        raise ValueError("Joint configuration contains nonfinite values")
    v411 = grouped_protocol(config)
    v412 = supported(config, "consistent_inputs")
    profile = capabilities(config)
    if config["experiment"]["protocol_version"] != profile.protocol:
        raise ValueError("Unregistered joint protocol")
    expected_version = profile.version
    if config["experiment"]["code_version"] != expected_version:
        raise ValueError(f"Joint version must be {expected_version}")
    if config["model"]["architecture"] != profile.architecture:
        raise ValueError("Joint architecture mismatch")
    if config["training"]["seeds"] != [2026] or config["project"]["seed"] != 2026:
        raise ValueError("Joint protocol uses one seed: 2026")
    if config["model"]["hidden_dim"] != 64 or config["sequence"]["step_seconds"] != 3:
        raise ValueError("Joint protocol requires hidden=64 and step=3 seconds")
    if (config["project"]["artifact_schema_version"] != profile.schema
            or config["model"].get("future_context_seconds", 0) != 0
            or config["model"]["long_pool_factor"] != config["sequence"]["long_pool_factor"]):
        raise ValueError("Joint schema, causal context or pooling identity mismatch")
    if config["sequence"]["supervised_steps"] != 256:
        raise ValueError("Joint protocol requires 256 new steps per chunk")
    if not full_data and config["training"]["partitions"] != 3:
        raise ValueError("Joint protocol requires three internal OOF partitions")
    search = config["search"]
    if full_data and search != {
            "high_thresholds": [.06, .10, .20], "low_ratios": [.50, .90], "budgets": [10, 20, 30],
            "nms_ious": [.3, .5, .7], "gap_seconds": [0, 15, 30],
            "acceptance_min": .01, "acceptance_max": .99, "acceptance_step": .01}:
        raise ValueError("v414 retains all 16038 registered operating points")
    if (not 0 <= search["acceptance_min"] <= search["acceptance_max"] <= 1
            or search["acceptance_step"] <= 0
            or any(not 0 < value < 1 for value in search["high_thresholds"])
            or any(not 0 <= value <= 1 for value in search["nms_ious"])
            or any(value not in (10, 20, 30) for value in search["budgets"])
            or any(not search[name] for name in ("high_thresholds", "nms_ious", "budgets", "gap_seconds"))):
        raise ValueError("Joint operating point search is invalid")
    if (not 0 < config["boundary"]["sigma_seconds"] or config["boundary"]["minimum_positive_events"] < 60
            or not 0.7 <= config["boundary"]["minimum_candidate_recall"] <= 1
            or any(not 0 <= value <= 1 for value in config["boundary"]["entropy_thresholds"])):
        raise ValueError("Joint Boundary qualification or entropy search is invalid")
    if config["decoder"].get("use_transition_candidates", False):
        raise ValueError("Transition candidates are disabled")
    if config["decoder"]["commit_buffer_seconds"] != 30:
        raise ValueError("Joint commit buffer is 30 seconds")
    if any(value not in (0, 15, 30) for value in config["search"]["gap_seconds"]):
        raise ValueError("Joint gaps cannot exceed the observed commit buffer")
    if config["boundary"]["maximum_end_range_seconds"] > 30:
        raise ValueError("Endpoint lookahead exceeds joint protocol")
    if "maximum_tp_to_fp_fraction" in config["gates"]:
        raise ValueError("Legacy Boundary fraction key is forbidden")
    if config["training"]["amp_dtype"] != "bfloat16":
        raise ValueError("Only registered bfloat16 AMP is supported")
    if any(int(config["training"].get(name, 1)) < 1 for name in RUNTIME_KEYS):
        raise ValueError("Runtime workers/prefetch/cache must be positive")
    if v411:
        quotas = config["training"].get("group_quotas")
        if (quotas != {"positive": 2, "near_miss": 2, "hard_false_positive": 3, "random_background": 1}
                or config["training"]["candidates_per_chunk"] != 8
                or config["training"]["versions_per_group"] != 3
                or config["training"]["maximum_groups_per_positive_event"] != 3
                or "sample_composition" in config["training"]):
            raise ValueError("v411 requires explicit 2/2/3/1 group quotas, eight groups and three versions")
    else:
        composition = config["training"]["sample_composition"]
        if (set(composition) != {"positive", "near_miss", "hard_false_positive", "random_background"}
                or any(not 0 <= value <= 1 for value in composition.values())
                or abs(sum(composition.values()) - 1) > 1e-9):
            raise ValueError("Candidate sample composition must sum to one")
    if not 0 <= config["training"]["clipping_promotion_fraction"] <= config["training"]["clipping_abort_fraction"] <= 1:
        raise ValueError("Clipping fractions are invalid")
    if (config["training"]["max_epochs"] < config["training"]["checkpoint_min_epoch"]
            or config["training"]["accumulate_sessions"] < 1 or config["training"]["candidates_per_chunk"] < 1):
        raise ValueError("Joint epoch/session/candidate counts are invalid")
    if (not 0 < config["boundary"]["maximum_start_range_seconds"] <= 900
            or not 0 < config["boundary"]["maximum_end_range_seconds"] <= 30):
        raise ValueError("Boundary ranges are invalid")
    if v411 and (not config["decoder"].get("learned_localization")
                 or config["decoder"]["quotas"] or config["decoder"]["onset_peak_threshold"] != 0.30
                 or config["training"]["validation_head_batch_size"] != 128
                 or config["boundary"]["modes"] != ["off", "start_only", "both"]
                 or not 1 <= config["resources"]["search_processes"] <= 4):
        raise ValueError("v411 localization, budget, Boundary or search profile mismatch")
    if v412:
        training = config["training"]
        if (training["lr_horizon_epochs"] != 64 or training["max_epochs"] != 64
                or training["lr_warmup_epochs"] != 2 or training["lr_minimum_fraction"] != 0.10
                or training["mining_families"] != 32 or training["mining_near_miss_families"] != 8
                or not isinstance(training["derived_cache_mib"], int) or not 1 <= training["derived_cache_mib"] <= 256
                or training["low_bandwidth_rates"] != [25.0, 26.316, 50.0]
                or any(training[name] != 0.20 for name in ("gyro_modality_dropout", "ppg_modality_dropout",
                    "rotation_augmentation_probability", "low_bandwidth_probability"))):
            raise ValueError("v412 augmentation, mining or scheduler profile mismatch")
    if profile.stabilized and (config["search"]["low_ratios"] != [0.50, 0.90]
                or config["model"]["localization_dilations"] != [1, 2]
                or config["loss"]["endpoint_background_ratio"] != 20
                or config["training"]["natural_replay_from_epoch"] != 3
                or config["training"]["qualification_deadline_epoch"] != 24
                or config["decoder"]["score_formula"] != profile.score_formula):
        raise ValueError("v413 registered stability profile mismatch")


def assert_excluded(training: set[str], prediction: set[str], outer: set[str]) -> None:
    if training & (prediction | outer):
        raise RuntimeError("Joint model subject exclusion failed")
    if not training or not prediction:
        raise ValueError("Joint fit and prediction subjects must be nonempty")


def initialize_run(config: dict[str, Any], root: Path, output: Path, run_name: str,
                   data_identity: dict[str, Any], *, fresh: bool, skip_s0: bool,
                   historical_reference: str | None, migrate_clipping_gate: bool = False) -> Path:
    validate_config(config)
    if fresh and migrate_clipping_gate:
        raise ValueError("Clipping migration requires --resume")
    if run_name in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("Unsafe run name")
    if historical_reference and (historical_reference in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", historical_reference)):
        raise ValueError("Unsafe historical reference name")
    run = output / "runs" / run_name
    identity = {"protocol": protocol_for(config), "config_sha256": config_identity(config),
                "source": source_identity(root), "data": data_identity,
                "runtime": runtime_identity(), "seed": 2026,
                "s0": {"skipped": skip_s0, "comparison_performed": False},
                "historical_reference": historical_reference}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    identity["git_head"] = git.stdout.strip() if git.returncode == 0 else None
    snapshot = root / "v412_source_snapshot.json" if protocol_for(config) == V412_PROTOCOL else root / "manifests" / "v411_source_snapshot.json"
    if supported(config, "stabilized"):
        snapshot = root / "manifests/v413_source_provenance.json"
    if supported(config, "full_data_workflow"):
        snapshot = root / "manifests/v414_source_provenance.json"
    if grouped_protocol(config) and snapshot.is_file():
        identity["parent_snapshot_sha256"] = sha256_file(snapshot)
    provenance = root / f"manifests/{capabilities(config).schema}_source_provenance.json"
    if supported(config, "consistent_inputs") and provenance.is_file():
        identity["parent_provenance_sha256"] = sha256_file(provenance)
    if fresh:
        if run.exists():
            raise FileExistsError("Fresh cannot overwrite an existing run; use --resume")
        run.mkdir(parents=True)
        write_json_atomic(run / "identity.json", identity)
        write_json_atomic(run / "execution_status.json", {"status": "CREATED", "stages": {}})
        if grouped_protocol(config):
            import shutil
            for relative, checksum in identity["source"].items():
                destination = run / "source_snapshot" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / relative, destination)
                if sha256_file(destination) != checksum:
                    raise RuntimeError("Source changed during run snapshot creation")
            write_json_atomic(run / "source_snapshot" / "SHA256SUMS.json", identity["source"])
            if snapshot.is_file():
                shutil.copyfile(snapshot, run / "parent_source_snapshot.json")
            if supported(config, "consistent_inputs") and provenance.is_file():
                shutil.copyfile(provenance, run / "parent_source_provenance.json")
    else:
        if not (run / "identity.json").is_file():
            raise FileNotFoundError("Resume requires an existing joint run")
        if (run / "single_partition_migration.json").exists():
            model_run_identity(run)
        from bme_eating.joint_resume_migration import (
            active_resume_identity,
            migrate_clipping_run,
            migrate_resume_identity,
        )
        if not grouped_protocol(config):
            migrate_resume_identity(run, identity, root)
        from bme_eating.joint_performance_migration import migrate_performance_run
        if not grouped_protocol(config):
            migrate_performance_run(run, identity, root)
        if migrate_clipping_gate and grouped_protocol(config):
            raise ValueError("v411 does not allow source/protocol migration on resume")
        if migrate_clipping_gate:
            migrate_clipping_run(run, identity, root)
        stored = active_resume_identity(run)
        differences = resume_identity_differences(stored, identity)
        if differences:
            raise RuntimeError("Resume identity changed: " + ", ".join(differences)
                               + "; create a new run")
        record_resume_git(run, stored, identity)
        if grouped_protocol(config):
            verify_artifact_hashes(run / "source_snapshot", identity["source"])
    if fresh:
        write_json_atomic(run / "resolved_config.json", config)
    return run


def checkpoint_qualified(payload: dict[str, Any], expected_identity: str,
                         expected_subjects: set[str], minimum_epoch: int) -> bool:
    return bool(payload.get("epoch_complete") and payload.get("promotion_eligible")
                and payload.get("epoch", 0) >= minimum_epoch
                and 0 <= payload.get("clipping_fraction", 1) <= 0.20
                and payload.get("identity") == expected_identity
                and set(payload.get("training_subjects", [])) == expected_subjects
                and all(torch.isfinite(value).all().item()
                        for value in payload.get("model", {}).values())
                and bool(payload.get("model")))


def verify_oof_lineage(lineages, subjects, outer, *, identities=None, expected_partitions=(0, 1, 2)):
    predictions, models, partitions = [], set(), set()
    if expected_partitions not in ((0,), (0, 1, 2)):
        raise ValueError("Unregistered internal prediction partitions")
    if len(lineages) != len(expected_partitions):
        raise RuntimeError("Joint OOF requires three complete partitions" if len(expected_partitions) == 3
                           else "Single calibration requires exactly partition 0")
    for lineage in lineages:
        fit, prediction = set(lineage["training_subjects"]), set(lineage["prediction_subjects"])
        model = lineage.get("model_identity")
        partition = lineage.get("partition")
        if not model or model in models or partition in partitions or partition not in expected_partitions:
            raise RuntimeError("Joint OOF model/partition identity is missing or duplicated")
        models.add(model)
        partitions.add(partition)
        if not fit or not prediction or fit & (prediction | outer) or fit | prediction != subjects:
            raise RuntimeError("Joint OOF model exclusion or cohort mismatch")
        if set(lineage["source_training_subjects"]) != fit or set(lineage["normalization_subjects"]) != fit:
            raise RuntimeError("Joint OOF preprocessing/source leakage")
        if set(lineage["outer_excluded_subjects"]) != outer:
            raise RuntimeError("Joint OOF outer exclusion mismatch")
        selector_fit, selector = set(lineage["selector_fit_subjects"]), set(lineage["selector_subjects"])
        if (not selector_fit or not selector or selector_fit & selector or selector_fit | selector != fit
                or not lineage.get("checkpoint_qualified")):
            raise RuntimeError("Joint selector exclusion/checkpoint qualification failed")
        if identities and any(lineage.get(name) != value for name, value in identities.items()):
            raise RuntimeError("Joint OOF config/source/data identity mismatch")
        predictions.extend(prediction)
    expected_subjects = subjects
    if expected_partitions == (0,):
        from bme_eating.training.hierarchical_v4_trainer import stacking_partitions
        assignment = stacking_partitions(subjects, 3, 2026)
        expected_subjects = {subject for subject, partition in assignment.items() if partition == 0}
    if len(predictions) != len(set(predictions)) or set(predictions) != expected_subjects:
        raise RuntimeError("Joint OOF does not have exactly one prediction model per subject")


def verify_calibration_lineage(lineages, subjects, outer, config, *, identities=None):
    verify_oof_lineage(lineages, subjects, outer, identities=identities,
                       expected_partitions=tuple(internal_partitions(config)))


def protocol_identities(config, data_identity):
    return {"config_sha256": config_identity(config),
            "source_sha256": digest(source_identity(Path(config["_config_path"]).resolve().parents[1])),
            "data_sha256": digest(data_identity)}


def verify_internal_manifest(directory, config, data_identity, subjects, outer, run_identity):
    path = directory / "internal_manifest.json"
    if not path.is_file():
        raise RuntimeError("Joint internal OOF completion manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {"protocol": protocol_for(config), "run_identity": run_identity,
                "training_subjects": sorted(subjects), "outer_excluded_subjects": sorted(outer),
                **protocol_identities(config, data_identity)}
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise RuntimeError("Joint internal OOF completion identity mismatch")
    if internal_partitions(config) == [0] and any(
            manifest.get(name) != value for name, value in calibration_scope(config).items()):
        raise RuntimeError("Single-partition calibration scope is missing or mismatched")
    required = {"operating_point.json", "internal_oof.pt", "internal_oof_windows.parquet",
                "search.json", "boundary_crossfit.json"}
    if not required.issubset(manifest.get("evidence_sha256", {})):
        raise RuntimeError("Joint internal OOF completion evidence is incomplete")
    verify_artifact_hashes(directory, manifest["evidence_sha256"])
    point = json.loads((directory / "operating_point.json").read_text(encoding="utf-8"))
    identities = protocol_identities(config, data_identity)
    verify_calibration_lineage(point["lineage"], subjects, outer, config, identities=identities)
    if internal_partitions(config) == [0] and any(
            point.get(name) != value for name, value in calibration_scope(config).items()):
        raise RuntimeError("Single-partition operating point cannot claim complete internal OOF")
    for lineage in point["lineage"]:
        train, prediction = set(lineage["training_subjects"]), set(lineage["prediction_subjects"])
        expected_model = digest({"run": run_identity, "training": sorted(train), "prediction": sorted(prediction),
                                 "outer": sorted(outer), "partition": lineage["partition"]})
        relative = f"partition_{lineage['partition']}/retrain/latest.pt"
        if lineage["model_identity"] != expected_model or relative not in manifest["evidence_sha256"]:
            raise RuntimeError("Internal checkpoint model/run evidence is missing or mismatched")
        checkpoint = torch.load(directory / relative, map_location="cpu", weights_only=True)
        if (not checkpoint_qualified(checkpoint, expected_model, train, config["training"]["checkpoint_min_epoch"])
                or checkpoint["epoch"] != lineage["epoch"]
                or set(checkpoint.get("excluded_subjects", [])) != outer | prediction
                or set(checkpoint.get("preprocessing", {}).get("training_subjects", [])) != train
                or any(checkpoint.get(name) != value for name, value in identities.items())):
            raise RuntimeError("Internal checkpoint qualification/exclusion/preprocessing identity mismatch")
        selected_name = "selected.pt" if lineage.get("selector_promotion_eligible") else "diagnostic_selected.pt"
        relative = f"partition_{lineage['partition']}/selection/{selected_name}"
        if relative not in manifest["evidence_sha256"]:
            raise RuntimeError("Internal selector checkpoint evidence is missing")
        checkpoint = torch.load(directory / relative, map_location="cpu", weights_only=True)
        fit, selector = set(lineage["selector_fit_subjects"]), set(lineage["selector_subjects"])
        qualified = checkpoint_qualified({**checkpoint, "promotion_eligible": True}, expected_model, fit,
                                          config["training"]["checkpoint_min_epoch"])
        if (not qualified or bool(checkpoint.get("promotion_eligible")) != bool(lineage.get("selector_promotion_eligible"))
                or checkpoint["epoch"] != lineage["epoch"]
                or set(checkpoint.get("excluded_subjects", [])) != outer | prediction | selector
                or set(checkpoint.get("preprocessing", {}).get("training_subjects", [])) != fit
                or any(checkpoint.get(name) != value for name, value in identities.items())):
            raise RuntimeError("Internal selector checkpoint qualification/exclusion identity mismatch")
    return point


def validate_gate_structure(payload):
    required = {"five_folds", "training_qualified", "f1", "fp_per_hour", "coverage", "truth_cohort",
                "fold_coverage", "operating_point_qualified", "stress", "boundary", "reference",
                "bootstrap", "matching_ranking", "same_recall", "different_recall"}
    checks = payload.get("checks", {})
    folds = payload.get("folds", [])
    if (not required.issubset(checks) or any(value is not True for value in checks.values())
            or len(folds) != 5 or {fold.get("fold") for fold in folds} != set(range(5))
            or not payload.get("reference", {}).get("qualified")
            or not payload.get("reference", {}).get("comparison_performed")):
        raise RuntimeError("Joint gate lock has incomplete or failed formal checks")


def execution_failure_status(error: BaseException) -> str:
    return "PAUSED" if isinstance(error, (KeyboardInterrupt, GPUResourcesBusy)) else "FAILED"


def require_gate(run: Path) -> dict[str, Any]:
    path = run / "joint_gate_manifest.json"
    if not path.is_file():
        raise RuntimeError("Joint gate manifest is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = json.loads((run / "identity.json").read_text(encoding="utf-8"))
    if payload.get("protocol") != identity.get("protocol", PROTOCOL) or not payload.get("promotable"):
        raise RuntimeError("Joint promotion gate is closed")
    if payload.get("identity_sha256") != sha256_file(run / "identity.json"):
        raise RuntimeError("Joint gate identity mismatch")
    validate_gate_structure(payload)
    if not {f"fold_{fold}/completion.json" for fold in range(5)}.issubset(payload.get("evidence_sha256", {})):
        raise RuntimeError("Joint gate lacks complete five-fold evidence")
    verify_artifact_hashes(run, payload.get("evidence_sha256", {}))
    reference = payload.get("reference", {})
    if not reference.get("qualified") or not reference.get("comparison_performed"):
        raise RuntimeError("Joint gate has no qualified paired reference")
    root = Path(reference["root"]).resolve()
    for relative, expected in reference.get("evidence_sha256", {}).items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file() or sha256_file(target) != expected:
            raise RuntimeError("Frozen reference evidence changed after promotion gate")
    if not reference.get("evidence_sha256"):
        raise RuntimeError("Frozen reference evidence is missing")
    return payload


def save_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_failure(run: Path, stage: str, error: BaseException) -> None:
    report = {"stage": stage, "error_type": type(error).__name__, "message": str(error),
              "protocol": json.loads((run / "identity.json").read_text(encoding="utf-8")).get("protocol", PROTOCOL), "deployable": False,
              "identity_sha256": sha256_file(run / "identity.json")}
    if stage in {"gates", "final", "export", "replay"}:
        for name in ("bundle", "bundle.partial", "fold0_bundle", "fold0_bundle.partial"):
            source = run / name
            if source.exists():
                destination = run / "failures" / ("nondeployable_" + name + "_" + uuid.uuid4().hex)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
                report.setdefault("quarantined_artifacts", []).append(str(destination.relative_to(run)))
    write_json_atomic(run / "failures" / f"{stage}.json", report)
    write_json_atomic(run / "failure_report.json", report)
