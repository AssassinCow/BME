from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

import torch
import yaml

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.reproducibility import git_worktree_identity
from bme_eating.stats_features import FoldRobustScaler
from bme_eating.v4_protocol import (
    BLOCKED_PREDECESSORS,
    CODE_VERSION,
    IGNORE_PROTOCOL,
    OBSERVATION_GAP_PROTOCOL,
    PROTOCOL_VERSION,
    RUNTIME_SOURCE_BINDING,
    RUNTIME_SOURCE_FILES,
    runtime_source_identity,
    validate_serialized_state_seeds,
    validate_serialized_verifier_seeds,
)

REQUIRED_MODEL_FILES = (
    "state_calibration.json",
    "statistics_scaler.json",
    "sensor_normalization.json",
    "duration_prior.json",
    "selected_pipeline.json",
    "resolved_config.yaml",
)

OPTIONAL_MODEL_FILES = (
    "logistic_verifier.json",
    "proposal_calibration.json",
    "boundary.pt",
    "boundary_range.json",
    "time_constrained_protocol.json",
)

RUNTIME_FILES = RUNTIME_SOURCE_FILES


def _selected_optional_model_files(
    selection: dict[str, object], promotion_protocol: str
) -> tuple[str, ...]:
    selected: list[str] = []
    verifier_kind = str(selection.get("verifier_kind", ""))
    if verifier_kind == "logistic":
        selected.append("logistic_verifier.json")
    elif verifier_kind == "deep":
        selected.append("proposal_calibration.json")
    if bool(selection.get("boundary_enabled", False)):
        selected.extend(("boundary.pt", "boundary_range.json"))
    if promotion_protocol == "time_constrained_single_holdout_v1":
        selected.append("time_constrained_protocol.json")
    return tuple(selected)


def _copy_sanitized_checkpoint(source: Path, target: Path, allowed: set[str]) -> None:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"V4 checkpoint is not a mapping: {source.name}")
    missing = {"model"} - set(payload)
    if missing:
        raise RuntimeError(f"V4 checkpoint is missing inference fields: {source.name}")
    sanitized = {key: value for key, value in payload.items() if key in allowed}
    forbidden = {
        "training_subjects",
        "prediction_subjects",
        "globally_excluded_subjects",
        "parent_artifact_sha256",
    }
    if forbidden & set(sanitized):
        raise RuntimeError(f"V4 checkpoint sanitization retained subject lineage: {source.name}")
    temporary = target.with_name(target.name + ".tmp")
    torch.save(sanitized, temporary)
    temporary.replace(target)


def _copy_sanitized_scaler(source: Path, target: Path) -> None:
    scaler = FoldRobustScaler.from_json(json.loads(source.read_text(encoding="utf-8")))
    sanitized = FoldRobustScaler(scaler.columns, scaler.median, scaler.iqr, ()).to_json()
    write_json_atomic(target, sanitized)


def _privacy_scan(root: Path) -> None:
    forbidden = (
        re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
        re.compile(r"\bHNU\d{5}[A-Z]?\b", re.IGNORECASE),
        re.compile(r"(?:access|secret)[_-]?key\s*[:=]", re.IGNORECASE),
        re.compile(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"),
    )
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".py", ".json", ".yaml", ".yml", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in forbidden):
            raise RuntimeError(
                f"V4 bundle privacy/dependency scan rejected {path.relative_to(root)}"
            )


def _write_bundle_zip(bundle: Path, target: Path) -> None:
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                archive.write(path, Path("model_bundle") / path.relative_to(bundle))
    temporary.replace(target)


def export_hierarchical_v4_bundle(
    project_root: Path, final_root: Path, *, fresh: bool, resume: bool
) -> Path:
    manifest_path = final_root / "final_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("V4 final manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") not in {"COMPLETE", "EXPORTED"}:
        raise RuntimeError("V4 final training must be complete before export")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"Only {PROTOCOL_VERSION} final artifacts may be exported")
    for relative, expected in manifest.get("artifact_hashes", {}).items():
        if relative == "model_bundle.zip":
            continue
        artifact = final_root / relative
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise RuntimeError(f"V4 final artifact changed before export: {relative}")
    required = [name for name in REQUIRED_MODEL_FILES if not (final_root / name).is_file()]
    if required:
        raise FileNotFoundError(f"V4 final artifacts are missing: {required}")
    selection = json.loads((final_root / "selected_pipeline.json").read_text(encoding="utf-8"))
    if selection.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"V4 selection is not {PROTOCOL_VERSION}")
    if selection.get("code_version") != CODE_VERSION:
        raise RuntimeError(f"V4 selection is not {CODE_VERSION}")
    if selection.get("blocked_predecessors") != list(BLOCKED_PREDECESSORS):
        raise RuntimeError("V4 selection does not block every predecessor protocol")
    if selection.get("selection_source") != "pooled_outer_oof":
        raise RuntimeError("V4 export requires pooled outer-OOF selection")
    required_protocols = {
        "ignore_protocol_version": IGNORE_PROTOCOL,
        "observation_gap_protocol": OBSERVATION_GAP_PROTOCOL,
        "runtime_source_binding": RUNTIME_SOURCE_BINDING,
    }
    mismatched_protocols = {
        key: selection.get(key)
        for key, expected in required_protocols.items()
        if selection.get(key) != expected
    }
    if mismatched_protocols:
        raise RuntimeError(
            f"V4 selection has incompatible protocol bindings: {mismatched_protocols}"
        )
    resume_identity = manifest.get("resume_identity", {})
    expected_git = resume_identity.get("git")
    active_git = git_worktree_identity(project_root)
    if expected_git != active_git:
        raise RuntimeError("V4 export runtime worktree differs from final training")
    source_root = project_root / "src" / "bme_eating"
    active_runtime_source = runtime_source_identity(source_root)
    if resume_identity.get("runtime_source_identity") != active_runtime_source:
        raise RuntimeError("V4 export runtime source differs from final training")
    if selection.get("candidate_minimum_seconds") != 3 or selection.get(
        "candidate_maximum_seconds"
    ) != 14_400:
        raise RuntimeError("V4 selection has invalid candidate duration bounds")
    config = yaml.safe_load((final_root / "resolved_config.yaml").read_text(encoding="utf-8"))
    promotion_hashes = selection.get("promotion_evidence_sha256", {})
    promotion_protocol = str(
        selection.get("promotion_protocol", "registered_ablation_gates_v1")
    )
    if str(manifest.get("promotion_protocol", "registered_ablation_gates_v1")) != (
        promotion_protocol
    ):
        raise RuntimeError("V4 final manifest and selection promotion protocols differ")
    required_promotion = (
        {"time_constrained_protocol"}
        if promotion_protocol == "time_constrained_single_holdout_v1"
        else {"fold0_ablation", "development_gate", "freeze_manifest", "stress_gate"}
    )
    if promotion_protocol not in {
        "registered_ablation_gates_v1",
        "time_constrained_single_holdout_v1",
    }:
        raise RuntimeError("V4 selection has an unsupported promotion protocol")
    if set(promotion_hashes) != required_promotion or any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in promotion_hashes.values()
    ):
        raise RuntimeError("V4 selection has incomplete promotion evidence hashes")
    if promotion_protocol == "time_constrained_single_holdout_v1":
        evidence_path = final_root / "time_constrained_protocol.json"
        if (
            not evidence_path.is_file()
            or sha256_file(evidence_path) != promotion_hashes["time_constrained_protocol"]
        ):
            raise RuntimeError("V4 time-constrained protocol evidence is missing or changed")
    state_seeds = validate_serialized_state_seeds(selection.get("state_seeds", []))
    configured_state = config.get("final_training", {}).get("state_seeds")
    if configured_state is not None and [int(value) for value in configured_state] != state_seeds:
        raise RuntimeError("V4 selection state seeds differ from resolved config")
    manifest_state = manifest.get("state_seeds")
    if manifest_state is not None and [int(value) for value in manifest_state] != state_seeds:
        raise RuntimeError("V4 final manifest state seeds differ from selection")
    selected_state_files = [f"state_seed_{seed}.pt" for seed in state_seeds]
    missing_state = [name for name in selected_state_files if not (final_root / name).is_file()]
    if missing_state:
        raise FileNotFoundError(f"V4 state artifacts are missing: {missing_state}")
    verifier_kind = str(selection.get("verifier_kind", ""))
    selected_verifier_files: list[str] = []
    if verifier_kind == "deep":
        verifier_seeds = validate_serialized_verifier_seeds(
            selection.get("verifier_seeds", [])
        )
        configured_verifier = config.get("verifier", {}).get("seeds")
        if configured_verifier is not None and [
            int(value) for value in configured_verifier
        ] != verifier_seeds:
            raise RuntimeError("V4 selection verifier seeds differ from resolved config")
        selected_verifier_files = [f"verifier_seed_{seed}.pt" for seed in verifier_seeds]
        missing_verifier = [
            name for name in (*selected_verifier_files, "proposal_calibration.json")
            if not (final_root / name).is_file()
        ]
        if missing_verifier:
            raise FileNotFoundError(f"Deep v4 verifier artifacts are missing: {missing_verifier}")
    elif verifier_kind == "logistic":
        if not (final_root / "logistic_verifier.json").is_file():
            raise FileNotFoundError("Logistic v4 verifier artifact is missing")
    elif verifier_kind == "state_only":
        if selection.get("score_column") not in {"generator_score", "final_score"}:
            raise RuntimeError("State-only v4 selection has an invalid score column")
    else:
        raise RuntimeError(f"Unsupported v4 verifier kind: {verifier_kind}")
    if bool(selection.get("boundary_enabled", False)):
        missing_boundary = [
            name for name in ("boundary.pt", "boundary_range.json")
            if not (final_root / name).is_file()
        ]
        if missing_boundary:
            raise FileNotFoundError(f"Selected v4 boundary artifacts are missing: {missing_boundary}")
    bundle = final_root / "model_bundle"
    if bundle.exists():
        if fresh:
            raise FileExistsError(f"V4 bundle already exists: {bundle}")
        if not resume:
            raise RuntimeError("Existing V4 bundle requires --resume")
        hashes = json.loads((bundle / "SHA256SUMS.json").read_text(encoding="utf-8"))["files"]
        for relative, expected in hashes.items():
            if sha256_file(bundle / relative) != expected:
                raise RuntimeError(f"V4 bundle artifact changed: {relative}")
        zip_path = final_root / "model_bundle.zip"
        expected_zip = manifest.get("artifact_hashes", {}).get("model_bundle.zip")
        if zip_path.is_file():
            actual_zip = sha256_file(zip_path)
            if expected_zip is not None and actual_zip != expected_zip:
                raise RuntimeError("V4 model_bundle.zip changed after export")
            if expected_zip is None:
                manifest.setdefault("artifact_hashes", {})["model_bundle.zip"] = actual_zip
                manifest["stage"] = "EXPORTED"
                write_json_atomic(manifest_path, manifest)
        else:
            _write_bundle_zip(bundle, zip_path)
            manifest.setdefault("artifact_hashes", {})["model_bundle.zip"] = sha256_file(zip_path)
            manifest["stage"] = "EXPORTED"
            write_json_atomic(manifest_path, manifest)
        return bundle
    if resume:
        raise FileNotFoundError("V4 bundle does not exist; start with --fresh")
    temporary = final_root / "model_bundle.tmp"
    if temporary.exists():
        raise RuntimeError("Stale V4 model_bundle.tmp exists")
    temporary.mkdir(parents=True)
    selected_optional_files = _selected_optional_model_files(selection, promotion_protocol)
    unexpected_optional_files = set(OPTIONAL_MODEL_FILES) - set(selected_optional_files)
    for name in (*REQUIRED_MODEL_FILES, *selected_state_files, *selected_optional_files):
        source = final_root / name
        if not source.is_file():
            continue
        target = temporary / name
        if name.startswith("state_seed_") and name.endswith(".pt"):
            _copy_sanitized_checkpoint(
                source,
                target,
                {"model", "model_config", "epochs", "seed", "training_subject_count"},
            )
        elif name == "boundary.pt":
            _copy_sanitized_checkpoint(
                source, target, {"model", "input_dim", "config", "seed", "epochs"}
            )
        elif name == "statistics_scaler.json":
            _copy_sanitized_scaler(source, target)
        else:
            shutil.copy2(source, temporary / name)
    copied_optional_files = {
        path.name for path in temporary.iterdir() if path.name in OPTIONAL_MODEL_FILES
    }
    if copied_optional_files & unexpected_optional_files:
        raise RuntimeError("V4 bundle retained an unselected optional model artifact")
    for name in selected_verifier_files:
        _copy_sanitized_checkpoint(
            final_root / name,
            temporary / name,
            {"model", "sequence_dim", "scalar_dim", "config", "seed", "epochs"},
        )
    runtime_root = temporary / "runtime" / "bme_eating"
    for relative in RUNTIME_FILES:
        target = runtime_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, target)
    (temporary / "README.md").write_text(
        "# StatsFusion v4 model bundle\n\n"
        "The bundle contains no training labels, raw data, personal paths, credentials, "
        "XGBoost model, XGBoost score, or XGBoost runtime dependency.\n",
        encoding="utf-8",
    )
    _privacy_scan(temporary)
    files = {
        path.relative_to(temporary).as_posix(): sha256_file(path)
        for path in sorted(temporary.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    write_json_atomic(temporary / "SHA256SUMS.json", {"version": 1, "files": files})
    temporary.replace(bundle)
    _write_bundle_zip(bundle, final_root / "model_bundle.zip")
    manifest["stage"] = "EXPORTED"
    manifest.setdefault("artifact_hashes", {})["model_bundle.zip"] = sha256_file(
        final_root / "model_bundle.zip"
    )
    write_json_atomic(manifest_path, manifest)
    return bundle
