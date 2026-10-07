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
from pathlib import Path
from typing import Any

import torch

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic

PROTOCOL = "statsfusion-joint-event-v1"
RUNTIME_KEYS = {"num_workers", "loader_prefetch_factor", "session_reader_cache_size"}
STAGES = ("audit", "validate", "outer", "gates", "final", "export", "replay")


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


def validate_config(config: dict[str, Any]) -> None:
    def finite_values(value):
        if isinstance(value, dict):
            return all(finite_values(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_values(item) for item in value)
        return not isinstance(value, (int, float)) or math.isfinite(value)
    if not finite_values(config):
        raise ValueError("Joint configuration contains nonfinite values")
    if config["experiment"]["protocol_version"] != PROTOCOL:
        raise ValueError("Unregistered joint protocol")
    if config["experiment"]["code_version"] != "v4.10":
        raise ValueError("Joint version must be v4.10")
    if config["model"]["architecture"] != "statsfusion_joint_event":
        raise ValueError("Joint architecture mismatch")
    if config["training"]["seeds"] != [2026] or config["project"]["seed"] != 2026:
        raise ValueError("Joint protocol uses one seed: 2026")
    if config["model"]["hidden_dim"] != 64 or config["sequence"]["step_seconds"] != 3:
        raise ValueError("Joint protocol requires hidden=64 and step=3 seconds")
    if (config["project"]["artifact_schema_version"] != "v410"
            or config["model"].get("future_context_seconds", 0) != 0
            or config["model"]["long_pool_factor"] != config["sequence"]["long_pool_factor"]):
        raise ValueError("Joint schema, causal context or pooling identity mismatch")
    if config["sequence"]["supervised_steps"] != 256:
        raise ValueError("Joint protocol requires 256 new steps per chunk")
    if config["training"]["partitions"] != 3:
        raise ValueError("Joint protocol requires three internal OOF partitions")
    search = config["search"]
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
    if any(int(config["training"][name]) < 1 for name in RUNTIME_KEYS):
        raise ValueError("Runtime workers/prefetch/cache must be positive")
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


def assert_excluded(training: set[str], prediction: set[str], outer: set[str]) -> None:
    if training & (prediction | outer):
        raise RuntimeError("Joint model subject exclusion failed")
    if not training or not prediction:
        raise ValueError("Joint fit and prediction subjects must be nonempty")


def initialize_run(config: dict[str, Any], root: Path, output: Path, run_name: str,
                   data_identity: dict[str, Any], *, fresh: bool, skip_s0: bool,
                   historical_reference: str | None) -> Path:
    validate_config(config)
    if run_name in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("Unsafe run name")
    if historical_reference and (historical_reference in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", historical_reference)):
        raise ValueError("Unsafe historical reference name")
    run = output / "runs" / run_name
    identity = {"protocol": PROTOCOL, "config_sha256": config_identity(config),
                "source": source_identity(root), "data": data_identity,
                "runtime": runtime_identity(), "seed": 2026,
                "s0": {"skipped": skip_s0, "comparison_performed": False},
                "historical_reference": historical_reference}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    identity["git_head"] = git.stdout.strip() if git.returncode == 0 else None
    if fresh:
        if run.exists():
            raise FileExistsError("Fresh cannot overwrite an existing run; use --resume")
        run.mkdir(parents=True)
        write_json_atomic(run / "identity.json", identity)
        write_json_atomic(run / "execution_status.json", {"status": "CREATED", "stages": {}})
    else:
        if not (run / "identity.json").is_file():
            raise FileNotFoundError("Resume requires an existing joint run")
        stored = json.loads((run / "identity.json").read_text(encoding="utf-8"))
        if stored != identity:
            raise RuntimeError("Resume source/config/data/runtime/protocol identity changed; create a new run")
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


def verify_oof_lineage(lineages, subjects, outer, *, identities=None):
    predictions, models, partitions = [], set(), set()
    if len(lineages) != 3:
        raise RuntimeError("Joint OOF requires three complete partitions")
    for lineage in lineages:
        fit, prediction = set(lineage["training_subjects"]), set(lineage["prediction_subjects"])
        model = lineage.get("model_identity")
        partition = lineage.get("partition")
        if not model or model in models or partition in partitions or partition not in range(3):
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
    if len(predictions) != len(set(predictions)) or set(predictions) != subjects:
        raise RuntimeError("Joint OOF does not have exactly one prediction model per subject")


def protocol_identities(config, data_identity):
    return {"config_sha256": config_identity(config),
            "source_sha256": digest(source_identity(Path(config["_config_path"]).resolve().parents[1])),
            "data_sha256": digest(data_identity)}


def verify_internal_manifest(directory, config, data_identity, subjects, outer, run_identity):
    path = directory / "internal_manifest.json"
    if not path.is_file():
        raise RuntimeError("Joint internal OOF completion manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {"protocol": PROTOCOL, "run_identity": run_identity,
                "training_subjects": sorted(subjects), "outer_excluded_subjects": sorted(outer),
                **protocol_identities(config, data_identity)}
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise RuntimeError("Joint internal OOF completion identity mismatch")
    required = {"operating_point.json", "internal_oof.pt", "internal_oof_windows.parquet",
                "search.json", "boundary_crossfit.json"}
    if not required.issubset(manifest.get("evidence_sha256", {})):
        raise RuntimeError("Joint internal OOF completion evidence is incomplete")
    verify_artifact_hashes(directory, manifest["evidence_sha256"])
    point = json.loads((directory / "operating_point.json").read_text(encoding="utf-8"))
    identities = protocol_identities(config, data_identity)
    verify_oof_lineage(point["lineage"], subjects, outer, identities=identities)
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
    if payload.get("protocol") != PROTOCOL or not payload.get("promotable"):
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
              "protocol": PROTOCOL, "deployable": False,
              "identity_sha256": sha256_file(run / "identity.json")}
    if stage in {"gates", "final", "export", "replay"}:
        for name in ("bundle", "bundle.partial"):
            source = run / name
            if source.exists():
                destination = run / "failures" / ("nondeployable_" + name + "_" + uuid.uuid4().hex)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
                report.setdefault("quarantined_artifacts", []).append(str(destination.relative_to(run)))
    write_json_atomic(run / "failures" / f"{stage}.json", report)
    write_json_atomic(run / "failure_report.json", report)
