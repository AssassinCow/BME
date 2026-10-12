from __future__ import annotations

import json
import os
import re
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any

from bme_eating.artifacts import sha256_file, write_json_atomic
from bme_eating.cli import command_audit, command_audit_multisection, command_preprocess
from bme_eating.config import _load_project_dotenv, require_environment_path
from bme_eating.data.manifest import _resolve_attachment_path
from bme_eating.data.quality import (
    validate_configured_expectations,
    validate_quality_invariants,
    write_quality_expectations,
)
from bme_eating.data.stats_fusion_inputs import (
    PROTOCOL_VERSION,
    canonical_input_paths,
    prepare_canonical_statsfusion_inputs,
)
from bme_eating.privacy import normalize_subject_id
from bme_eating.protocol import digest

PREPARATION_RECIPE_VERSION = "v2-audited-2026-09"
PREPARATION_RECIPE = {
    "data": {
        "invalid_subjects": ["HNU21007", "HNU21026J", "HNU21030"],
        "subject_aliases": {},
        "formal_subject_pattern": r"^HNU\d{5}$",
        "packet_timestamp_anchor": "start",
        "ppg_samples_per_row": 20,
        "ppg_available_columns": 44,
        "gap_factor": 5.0,
        "minimum_gap_ms": 2000,
        "maximum_interpolation_gap_ms": 250,
        "motion_target_hz": 100,
        "ppg_target_hz": 50,
        "minimum_segment_seconds": 30,
        "output_step_seconds": 3,
        "session_join_gap_seconds": 3,
        "subject_folds": 5,
        "split_seed": 2026,
    },
    "preprocess": {"compression": True, "workers": 8, "overwrite": False},
    "audit": {"workers": 8},
    "quality_gates": {
        "expected_records": 1112,
        "expected_subjects": 39,
        "expected_documented_text": 1029,
        "expected_recovered_text_suffix": 4,
        "expected_recovered_multisection_deduplicated": 45,
        "expected_quarantined_multisection": 34,
        "expected_preprocessed_attachments": 1078,
        "expected_ppg_samples_per_row": 20,
        "expected_ppg_available_columns": 44,
        "expected_maximum_observed_nonzero_ppg_slot": 20,
    },
}
PREPARATION_IMPLEMENTATION_FILES = (
    "cli.py", "privacy.py", "data/input_preparation.py", "data/manifest.py",
    "data/multisection.py", "data/packet_reader.py", "data/preprocess.py",
    "data/labels.py", "data/splits.py", "data/quality.py",
)


def preparation_config(config: dict[str, Any], workers: int | None = None) -> dict[str, Any]:
    project = config["project"]
    if (project.get("input_artifact_schema_version") != "v2"
            or project.get("canonical_schema_version") != "v4"
            or config["model"]["architecture"] != "statsfusion_joint_event_v414"):
        raise ValueError("Input preparation requires the complete v4.14 training configuration")
    prepared = deepcopy(PREPARATION_RECIPE)
    prepared["project"] = {
        "seed": 2026,
        "data_root_env": project["data_root_env"],
        "output_root_env": project["output_root_env"],
        "artifact_schema_version": "v2",
    }
    prepared["_config_path"] = config["_config_path"]
    if workers is not None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        prepared["preprocess"]["workers"] = workers
        prepared["audit"]["workers"] = workers
    return prepared


def _raw_identity(data_root: Path, output_base: Path, config: dict[str, Any]) -> dict[str, Any]:
    salt_path = output_base / "private/subject_salt.hex"
    env_salt = os.environ.get("BME_SUBJECT_SALT")
    if env_salt:
        salt = env_salt.encode("utf-8")
    elif salt_path.is_file():
        salt = bytes.fromhex(salt_path.read_text(encoding="ascii").strip())
        if not salt:
            raise ValueError("The private subject salt is empty")
    else:
        raise RuntimeError(
            "Reproduction requires the original BME_SUBJECT_SALT or "
            "BME_OUTPUT_ROOT/private/subject_salt.hex; do not generate new subject keys"
        )
    sources = ("manifests/download_state.json", "derived/valid_metadata/sensor.csv",
               "derived/valid_metadata/meal.csv")
    files = {relative: sha256_file(data_root / relative) for relative in sources}
    state = json.loads((data_root / sources[0]).read_text(encoding="utf-8"))
    invalid = set(config["data"]["invalid_subjects"])
    pattern = re.compile(config["data"]["formal_subject_pattern"])
    for item in state["attachments"]:
        subject = normalize_subject_id(item.get("externalid"))
        if (subject in invalid or not pattern.fullmatch(subject)
                or item.get("status") != "downloaded" or item.get("invalid_externalid")):
            continue
        path = _resolve_attachment_path(data_root, item["relative_path"])
        checksum = sha256_file(path)
        if checksum != str(item.get("sha256", "")).lower():
            raise RuntimeError("Raw attachment differs from its download manifest: " + path.name)
        files[path.relative_to(data_root).as_posix()] = checksum
    package_root = Path(__file__).resolve().parents[1]
    code = {name: sha256_file(package_root / name) for name in PREPARATION_IMPLEMENTATION_FILES}
    recipe = deepcopy({name: value for name, value in config.items() if not name.startswith("_")})
    recipe["preprocess"].pop("workers")
    recipe["audit"].pop("workers")
    return {"recipe_version": PREPARATION_RECIPE_VERSION, "recipe": recipe,
            "raw_root": str(data_root), "output_root": str(output_base),
            "raw_files": files, "subject_salt_sha256": digest(list(salt)), "source": code}


def _verify_v2(input_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    indices = input_root / "indices"
    report = json.loads((indices / "quality_report.json").read_text(encoding="utf-8"))
    expected = json.loads((indices / "quality_expectations.json").read_text(encoding="utf-8"))
    validate_configured_expectations(report, config["quality_gates"])
    validate_quality_invariants(report)
    if report != expected:
        raise RuntimeError("The v2 quality report differs from its frozen expectations")
    manifest_path = input_root / "preparation_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "COMPLETE":
            raise RuntimeError("v2 preparation is incomplete; resume its v2 stage first")
        for relative, checksum in manifest["artifacts"].items():
            path = (input_root / relative).resolve()
            if not path.is_relative_to(input_root.resolve()) or sha256_file(path) != checksum:
                raise RuntimeError("Prepared v2 artifact changed: " + relative)
    return report


def _prepare_v2(data_root, output_base, config, *, fresh, resume):
    input_root = output_base / "v2"
    manifest_path = input_root / "preparation_manifest.json"
    if fresh and input_root.exists() and any(input_root.iterdir()):
        raise FileExistsError("v2 inputs already exist; use a separate BME_OUTPUT_ROOT")
    if resume and not manifest_path.is_file():
        raise RuntimeError("v2 resume requires this entry point's preparation_manifest.json")
    identity = _raw_identity(data_root, output_base, config)
    if resume:
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored["identity"] != identity:
            raise RuntimeError("Raw data, salt or preparation recipe/source changed; resume refused")
        if stored["status"] == "COMPLETE":
            return _verify_v2(input_root, config)
    manifest = {"schema": "v2", "status": "RUNNING", "identity": identity,
                "identity_sha256": digest(identity), "workers": config["preprocess"]["workers"]}
    write_json_atomic(manifest_path, manifest)
    args = Namespace(config=config["_config_path"], workers=config["preprocess"]["workers"],
                     schema_zips="all", maximum_rows=100_000, no_resume=False,
                     overwrite=False, allow_quarantine=True)
    try:
        command_audit(args, config=config)
        command_audit_multisection(args, config=config)
        command_preprocess(args, config=config)
        write_quality_expectations(input_root)
        if _raw_identity(data_root, output_base, config) != identity:
            raise RuntimeError("Raw data, salt or preparation source changed during preprocessing")
        manifest["artifacts"] = {
            path.relative_to(input_root).as_posix(): sha256_file(path)
            for directory in (input_root / "indices", input_root / "segments")
            for path in sorted(directory.rglob("*")) if path.is_file()
        }
        manifest["status"] = "COMPLETE"
        write_json_atomic(manifest_path, manifest)
        return _verify_v2(input_root, config)
    except BaseException as error:
        manifest.update({"status": "PAUSED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                         "error_type": type(error).__name__, "error": str(error)})
        write_json_atomic(manifest_path, manifest)
        raise


def prepare_inputs(config, *, stage="all", workers=None, fresh=False, resume=False):
    if stage not in {"v2", "canonical", "all"} or fresh == resume:
        raise ValueError("Select v2/canonical/all and exactly one of fresh/resume")
    prepared = preparation_config(config, workers)
    _load_project_dotenv(config)
    output_base = require_environment_path(config, "output_root_env")
    input_root, canonical_root = output_base / "v2", output_base / "v4"
    canonical_paths = canonical_input_paths(canonical_root)
    if fresh and canonical_paths["root"].exists():
        if any(canonical_paths["root"].iterdir()):
            raise FileExistsError("Canonical inputs already exist; use a separate BME_OUTPUT_ROOT")
    if stage in {"v2", "all"}:
        data_root = require_environment_path(config, "data_root_env")
        _prepare_v2(data_root, output_base, prepared, fresh=fresh, resume=resume)
    else:
        _verify_v2(input_root, prepared)
    result = {"input_root": str(input_root), "schema": "v2", "stage": stage}
    if stage in {"canonical", "all"}:
        existing = canonical_paths["root"].is_dir() and any(canonical_paths["root"].iterdir())
        manifest = prepare_canonical_statsfusion_inputs(
            input_root, canonical_root, workers=prepared["preprocess"]["workers"],
            fresh=not existing, resume=existing,
        )
        result.update({"canonical_root": str(canonical_paths["root"]),
                       "canonical_protocol": PROTOCOL_VERSION, "canonical_manifest": manifest})
    return result
