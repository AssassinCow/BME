from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic

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
    "verifier.pt",
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
    "models/__init__.py",
    "models/dtp_sqf.py",
    "models/endpoint_refiner.py",
    "models/event_verifier_v4.py",
    "models/factory.py",
    "models/hierarchical_state.py",
    "models/stats_fusion_state.py",
)


def _privacy_scan(root: Path) -> None:
    forbidden = (
        re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
        re.compile(r"(?:access|secret)[_-]?key\s*[:=]", re.IGNORECASE),
        re.compile(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"),
    )
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".py", ".json", ".yaml", ".yml", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in forbidden):
            raise RuntimeError(f"V4 bundle privacy/dependency scan rejected {path.relative_to(root)}")


def export_hierarchical_v4_bundle(
    project_root: Path, final_root: Path, *, fresh: bool, resume: bool
) -> Path:
    manifest_path = final_root / "final_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("V4 final manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") not in {"COMPLETE", "EXPORTED"}:
        raise RuntimeError("V4 final training must be complete before export")
    required = [name for name in REQUIRED_MODEL_FILES if not (final_root / name).is_file()]
    if required:
        raise FileNotFoundError(f"V4 final artifacts are missing: {required}")
    if not any((final_root / name).is_file() for name in ("verifier.pt", "logistic_verifier.json")):
        raise FileNotFoundError("V4 bundle requires a deep or logistic verifier")
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
        if source.is_file():
            shutil.copy2(source, temporary / name)
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
