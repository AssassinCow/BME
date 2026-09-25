from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

import torch

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.stats_features import FoldRobustScaler

REQUIRED_MODEL_FILES = (
    "state_seed_2026.pt",
    "state_seed_2027.pt",
    "state_seed_2028.pt",
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
)

RUNTIME_FILES = (
    "__init__.py",
    "calibration_v4.py",
    "hierarchical_v4_pipeline.py",
    "metrics.py",
    "proposals_v4.py",
    "stats_features.py",
    "structured_decoder.py",
    "types.py",
    "data/__init__.py",
    "data/deep_dataset.py",
    "data/session.py",
    "data/stats_fusion_preprocess.py",
    "data/stats_fusion_sequence.py",
    "features/__init__.py",
    "features/baseline.py",
    "features/signal.py",
    "models/__init__.py",
    "models/dtp_sqf.py",
    "models/endpoint_refiner.py",
    "models/event_verifier_v4.py",
    "models/factory.py",
    "models/hierarchical_state.py",
    "models/stats_fusion_state.py",
)


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


def export_hierarchical_v4_bundle(
    project_root: Path, final_root: Path, *, fresh: bool, resume: bool
) -> Path:
    manifest_path = final_root / "final_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("V4 final manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") not in {"COMPLETE", "EXPORTED"}:
        raise RuntimeError("V4 final training must be complete before export")
    if manifest.get("protocol_version") != "statsfusion-r2":
        raise RuntimeError("Only statsfusion-r2 final artifacts may be exported")
    for relative, expected in manifest.get("artifact_hashes", {}).items():
        artifact = final_root / relative
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise RuntimeError(f"V4 final artifact changed before export: {relative}")
    required = [name for name in REQUIRED_MODEL_FILES if not (final_root / name).is_file()]
    if required:
        raise FileNotFoundError(f"V4 final artifacts are missing: {required}")
    selection = json.loads((final_root / "selected_pipeline.json").read_text(encoding="utf-8"))
    if selection.get("protocol_version") != "statsfusion-r2":
        raise RuntimeError("V4 selection is not statsfusion-r2")
    if selection.get("selection_source") != "pooled_outer_oof":
        raise RuntimeError("V4 export requires pooled outer-OOF selection")
    state_seeds = [int(value) for value in selection.get("state_seeds", [])]
    if state_seeds != [2026, 2027, 2028]:
        raise RuntimeError("V4 export requires state seeds 2026/2027/2028")
    verifier_kind = str(selection.get("verifier_kind", ""))
    selected_verifier_files: list[str] = []
    if verifier_kind == "deep":
        verifier_seeds = [int(value) for value in selection.get("verifier_seeds", [])]
        if verifier_seeds != [2026, 2027, 2028]:
            raise RuntimeError("Deep v4 export requires verifier seeds 2026/2027/2028")
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
        return bundle
    if resume:
        raise FileNotFoundError("V4 bundle does not exist; start with --fresh")
    temporary = final_root / "model_bundle.tmp"
    if temporary.exists():
        raise RuntimeError("Stale V4 model_bundle.tmp exists")
    temporary.mkdir(parents=True)
    for name in (*REQUIRED_MODEL_FILES, *OPTIONAL_MODEL_FILES):
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
    for name in selected_verifier_files:
        _copy_sanitized_checkpoint(
            final_root / name,
            temporary / name,
            {"model", "sequence_dim", "scalar_dim", "config", "seed", "epochs"},
        )
    source_root = project_root / "src" / "bme_eating"
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
    zip_temporary = final_root / "model_bundle.tmp.zip"
    with zipfile.ZipFile(zip_temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                archive.write(path, Path("model_bundle") / path.relative_to(bundle))
    zip_temporary.replace(final_root / "model_bundle.zip")
    manifest["stage"] = "EXPORTED"
    manifest.setdefault("artifact_hashes", {})["model_bundle.zip"] = sha256_file(
        final_root / "model_bundle.zip"
    )
    write_json_atomic(manifest_path, manifest)
    return bundle
