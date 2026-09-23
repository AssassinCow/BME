from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic

MODEL_FILES = (
    "state_seed_2026.pt",
    "state_seed_2027.pt",
    "state_seed_2028.pt",
    "xgboost_full.json",
    "xgboost_full.metadata.json",
    "xgboost_postprocess.json",
    "verifier.pt",
    "boundary.pt",
    "calibration.json",
    "selected_pipeline.json",
    "duration_bounds.json",
    "sensor_normalization.json",
    "stable_feature_normalization.json",
    "resolved_config.yaml",
)

RUNTIME_FILES = (
    "__init__.py",
    "calibration.py",
    "hierarchical_pipeline.py",
    "metrics.py",
    "postprocess.py",
    "proposals.py",
    "types.py",
    "models/__init__.py",
    "models/boundary_refiner.py",
    "models/dtp_sqf.py",
    "models/event_verifier.py",
    "models/factory.py",
    "models/hierarchical_state.py",
)


def _verify_existing_bundle(bundle_root: Path) -> None:
    manifest_path = bundle_root / "SHA256SUMS.json"
    if not manifest_path.is_file():
        raise RuntimeError("Existing model bundle has no hash manifest")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_files = set(payload["files"])
    actual_files = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS.json"
        and path.suffix.lower() != ".pyc"
        and "__pycache__" not in path.parts
    }
    if actual_files != expected_files:
        unexpected = sorted(actual_files - expected_files)
        missing = sorted(expected_files - actual_files)
        raise RuntimeError(
            f"Existing bundle file set changed; unexpected={unexpected}, missing={missing}"
        )
    for relative, expected in payload["files"].items():
        path = bundle_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Existing bundle artifact changed: {relative}")


def _scan_text_files(bundle_root: Path) -> None:
    forbidden = (
        re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
        re.compile(r"(?:access|secret)[_-]?key\s*[:=]", re.IGNORECASE),
        re.compile(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"),
    )
    for path in bundle_root.rglob("*"):
        if path.suffix.lower() not in {".py", ".json", ".yaml", ".yml", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in forbidden):
            raise RuntimeError(f"Bundle privacy scan rejected {path.relative_to(bundle_root)}")


def _write_bundle_readme(path: Path) -> None:
    path.write_text(
        """# Hierarchical eating detector model bundle

This bundle contains only model weights, frozen inference configuration, normalization
statistics, and the minimal Python runtime. It intentionally excludes training labels,
raw sensor data, subject mappings, OOF predictions, credentials, and machine-specific paths.

Install the dependencies declared by the project, add `runtime` to `PYTHONPATH`, then use
`bme_eating.hierarchical_pipeline.load_hierarchical_bundle`. The official contest test-file
schema is still unconfirmed, so the repository must provide a thin input/output adapter once
that schema is released. The detector itself enforces the frozen 60-second future-data limit.
""",
        encoding="utf-8",
    )


def export_hierarchical_bundle(
    project_root: Path,
    final_root: Path,
    *,
    fresh: bool,
    resume: bool,
) -> Path:
    final_manifest_path = final_root / "final_manifest.json"
    if not final_manifest_path.is_file():
        raise FileNotFoundError("Final training manifest is missing")
    final_manifest: dict[str, Any] = json.loads(
        final_manifest_path.read_text(encoding="utf-8")
    )
    if final_manifest.get("stage") not in {"COMPLETE", "EXPORTED"}:
        raise RuntimeError("Final training must be complete before bundle export")
    bundle_root = final_root / "model_bundle"
    if bundle_root.exists():
        if fresh:
            raise FileExistsError(f"Model bundle already exists: {bundle_root}")
        if not resume:
            raise RuntimeError("Existing model bundle requires --resume")
        _verify_existing_bundle(bundle_root)
    else:
        if resume:
            raise FileNotFoundError("Model bundle does not exist; start export with --fresh")
        missing = [name for name in MODEL_FILES if not (final_root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Final model artifacts are missing: {missing}")
        temporary = final_root / "model_bundle.tmp"
        if temporary.exists():
            raise RuntimeError("Stale model_bundle.tmp exists; inspect it before retrying")
        temporary.mkdir(parents=True)
        for name in MODEL_FILES:
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_root / name, target)
        runtime_root = temporary / "runtime" / "bme_eating"
        source_root = project_root / "src" / "bme_eating"
        for name in RUNTIME_FILES:
            source = source_root / name
            target = runtime_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        _write_bundle_readme(temporary / "README.md")
        _scan_text_files(temporary)
        files = {
            path.relative_to(temporary).as_posix(): sha256_file(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file() and path.name != "SHA256SUMS.json"
        }
        write_json_atomic(
            temporary / "SHA256SUMS.json",
            {"version": 1, "files": files},
        )
        temporary.replace(bundle_root)
        _verify_existing_bundle(bundle_root)

    zip_path = final_root / "model_bundle.zip"
    zip_temporary = final_root / "model_bundle.tmp.zip"
    registered = json.loads(
        (bundle_root / "SHA256SUMS.json").read_text(encoding="utf-8")
    )["files"]
    with zipfile.ZipFile(zip_temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in [*sorted(registered), "SHA256SUMS.json"]:
            path = bundle_root / relative
            archive.write(path, Path("model_bundle") / relative)
    zip_temporary.replace(zip_path)
    final_manifest["stage"] = "EXPORTED"
    final_manifest.setdefault("artifact_hashes", {}).update(
        {
            "model_bundle/SHA256SUMS.json": sha256_file(
                bundle_root / "SHA256SUMS.json"
            ),
            "model_bundle.zip": sha256_file(zip_path),
        }
    )
    write_json_atomic(final_manifest_path, final_manifest)
    return bundle_root
