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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from bme_eating.artifacts import sha256_file, write_json_atomic


@dataclass(frozen=True)
class JointCapabilities:
    version: str = "v4.14"
    schema: str = "v414"
    protocol: str = "statsfusion-joint-event-v7"
    repair_revision: str = "v414-r2"
    architecture: str = "statsfusion_joint_event_v414"
    grouped: bool = True
    consistent_inputs: bool = True
    stabilized: bool = True
    feature_width: int = 201
    roi_width: int = 259
    full_data_workflow: bool = True

    @property
    def score_formula(self):
        return "calibrated_event_probability_v414"


PROFILE = JointCapabilities()
EVENT_PROBABILITY_FORMULAS = frozenset(f"calibrated_event_probability_v{version}" for version in (411, 412, 413, 414))


def capabilities(value):
    if isinstance(value, dict) and value["model"]["architecture"] != PROFILE.architecture:
        raise ValueError("Only statsfusion_joint_event_v414 is supported")
    return getattr(value, "capabilities", PROFILE)


def supported(value, name):
    return getattr(capabilities(value), name)


def calibration_discriminates(config, payload):
    return (payload.get("discriminative") is True
            and payload.get("slope", 0) > config["calibration"]["minimum_slope"]
            and (payload.get("probability_range") or 0) > config["calibration"]["minimum_probability_range"])


PROTOCOL = "statsfusion-joint-event-v7"
CHECKPOINT_STATE_VERSION = 3
RUNTIME_KEYS = {"num_workers", "loader_prefetch_factor", "session_reader_cache_size", "derived_cache_mib"}
STAGES = ("audit", "dev", "final39", "export", "replay")


def grouped_protocol(config):
    return supported(config, "grouped")


def event_score_formula(config):
    return capabilities(config).score_formula


def protocol_for(config):
    return config["experiment"]["protocol_version"]


def stages_for(config):
    capabilities(config)
    return STAGES


def execution_mode(config):
    return config.get("execution", {}).get("mode", "dev28_11")


def model_run_identity(run):
    return sha256_file(run / "identity.json")


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
    if (execution_mode(config) != "dev28_11" or config["execution"].get("internal_mode") != "development"
            or config["execution"].get("deployment_boundary_mode") != "off"
            or config["development"].get("final_calibration_kind") != "development_on_final_model"
            or not re.fullmatch(r"[0-9a-f]{64}", config["development"].get("split_sha256", ""))):
        raise ValueError("v414 requires a frozen dev28_11 split and explicit final development calibration")
    def finite_values(value):
        if isinstance(value, dict):
            return all(finite_values(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_values(item) for item in value)
        return not isinstance(value, (int, float)) or math.isfinite(value)
    if not finite_values(config):
        raise ValueError("Joint configuration contains nonfinite values")
    profile = capabilities(config)
    if config["experiment"]["protocol_version"] != profile.protocol:
        raise ValueError("Unregistered joint protocol")
    expected_version = profile.version
    if config["experiment"]["code_version"] != expected_version:
        raise ValueError(f"Joint version must be {expected_version}")
    if config["model"]["architecture"] != profile.architecture:
        raise ValueError("Joint architecture mismatch")
    if config["experiment"].get("repair_revision") != profile.repair_revision:
        raise ValueError("v414 requires the registered v414-r2 repair revision")
    if config["loss"].get("contrastive_max_history_bins") != 128:
        raise ValueError("v414-r2 requires at most 128 historical contrastive references")
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
    search = config["search"]
    if search != {
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
    quotas = config["training"].get("group_quotas")
    if (quotas != {"positive": 2, "near_miss": 2, "hard_false_positive": 3, "random_background": 1}
            or config["training"]["candidates_per_chunk"] != 8
            or config["training"]["versions_per_group"] != 3
            or config["training"]["maximum_groups_per_positive_event"] != 3
            or "sample_composition" in config["training"]):
        raise ValueError("Joint requires explicit 2/2/3/1 group quotas, eight groups and three versions")
    if not 0 <= config["training"]["clipping_promotion_fraction"] <= config["training"]["clipping_abort_fraction"] <= 1:
        raise ValueError("Clipping fractions are invalid")
    if (config["training"]["max_epochs"] < config["training"]["checkpoint_min_epoch"]
            or config["training"]["accumulate_sessions"] < 1 or config["training"]["candidates_per_chunk"] < 1):
        raise ValueError("Joint epoch/session/candidate counts are invalid")
    if (not 0 < config["boundary"]["maximum_start_range_seconds"] <= 900
            or not 0 < config["boundary"]["maximum_end_range_seconds"] <= 30):
        raise ValueError("Boundary ranges are invalid")
    if (not config["decoder"].get("learned_localization")
                 or config["decoder"]["quotas"] or config["decoder"]["onset_peak_threshold"] != 0.30
                 or config["training"]["validation_head_batch_size"] != 128
                 or config["boundary"]["modes"] != ["off", "start_only", "both"]
                 or not 1 <= config["resources"]["search_processes"] <= 4):
        raise ValueError("Joint localization, budget, Boundary or search profile mismatch")
    training = config["training"]
    if (training["lr_horizon_epochs"] != 64 or training["max_epochs"] != 64
            or training["lr_warmup_epochs"] != 2 or training["lr_minimum_fraction"] != 0.10
            or training["mining_families"] != 32 or training["mining_near_miss_families"] != 8
            or not isinstance(training["derived_cache_mib"], int) or not 1 <= training["derived_cache_mib"] <= 256
            or training["low_bandwidth_rates"] != [25.0, 26.316, 50.0]
            or any(training[name] != 0.20 for name in ("gyro_modality_dropout", "ppg_modality_dropout",
                "rotation_augmentation_probability", "low_bandwidth_probability"))):
        raise ValueError("Joint augmentation, mining or scheduler profile mismatch")
    if profile.stabilized and (config["search"]["low_ratios"] != [0.50, 0.90]
                or config["model"]["localization_dilations"] != [1, 2]
                or config["loss"]["endpoint_background_ratio"] != 20
                or config["training"]["natural_replay_from_epoch"] != 3
                or config["training"]["qualification_deadline_epoch"] != 24
                or config["decoder"]["score_formula"] != profile.score_formula):
        raise ValueError("Joint registered stability profile mismatch")


def assert_excluded(training: set[str], prediction: set[str], outer: set[str]) -> None:
    if training & (prediction | outer):
        raise RuntimeError("Joint model subject exclusion failed")
    if not training or not prediction:
        raise ValueError("Joint fit and prediction subjects must be nonempty")


def initialize_run(config, root, output, run_name, data_identity, *, fresh,
                   skip_s0, historical_reference=None):
    validate_config(config)
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
    provenance = root / "manifests/source_provenance.json"
    if provenance.is_file():
        identity["parent_provenance_sha256"] = sha256_file(provenance)
    if fresh:
        if run.exists():
            raise FileExistsError("Fresh cannot overwrite an existing run; use --resume")
        run.mkdir(parents=True)
        write_json_atomic(run / "identity.json", identity)
        write_json_atomic(run / "execution_status.json", {"status": "CREATED", "stages": {}})
        import shutil
        for relative, checksum in identity["source"].items():
            destination = run / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, destination)
            if sha256_file(destination) != checksum:
                raise RuntimeError("Source changed during run snapshot creation")
        write_json_atomic(run / "source_snapshot/SHA256SUMS.json", identity["source"])
        if provenance.is_file():
            shutil.copyfile(provenance, run / "parent_source_provenance.json")
        write_json_atomic(run / "resolved_config.json", config)
    else:
        if not (run / "identity.json").is_file():
            raise FileNotFoundError("Resume requires an existing joint run")
        stored = json.loads((run / "identity.json").read_text(encoding="utf-8"))
        differences = resume_identity_differences(stored, identity)
        if differences:
            raise RuntimeError("Resume identity changed: " + ", ".join(differences) + "; create a new run")
        record_resume_git(run, stored, identity)
        verify_artifact_hashes(run / "source_snapshot", identity["source"])
    return run


def checkpoint_qualified(payload: dict[str, Any], expected_identity: str,
                         expected_subjects: set[str], minimum_epoch: int) -> bool:
    return bool(payload.get("protocol") == PROTOCOL
                and payload.get("state_version") == CHECKPOINT_STATE_VERSION
                and payload.get("epoch_complete") and payload.get("promotion_eligible")
                and payload.get("epoch", 0) >= minimum_epoch
                and 0 <= payload.get("clipping_fraction", 1) <= 0.20
                and payload.get("identity") == expected_identity
                and set(payload.get("training_subjects", [])) == expected_subjects
                and all(torch.isfinite(value).all().item()
                        for value in payload.get("model", {}).values())
                and bool(payload.get("model")))


def protocol_identities(config, data_identity):
    return {"config_sha256": config_identity(config),
            "source_sha256": digest(source_identity(Path(config["_config_path"]).resolve().parents[1])),
            "data_sha256": digest(data_identity)}


def execution_failure_status(error: BaseException) -> str:
    return "PAUSED" if isinstance(error, (KeyboardInterrupt, GPUResourcesBusy)) else "FAILED"


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
    if stage in {"final39", "export", "replay"}:
        for name in ("bundle", "bundle.partial"):
            source = run / name
            if source.exists():
                destination = run / "failures" / ("nondeployable_" + name + "_" + uuid.uuid4().hex)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
                report.setdefault("quarantined_artifacts", []).append(str(destination.relative_to(run)))
    write_json_atomic(run / "failures" / f"{stage}.json", report)
    write_json_atomic(run / "failure_report.json", report)
