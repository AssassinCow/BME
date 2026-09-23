from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from bme_eating.hierarchical_export import MODEL_FILES, export_hierarchical_bundle


def test_export_bundle_is_whitelisted_hashed_and_resumable(tmp_path) -> None:
    final_root = tmp_path / "final"
    final_root.mkdir()
    for name in MODEL_FILES:
        path = final_root / name
        if path.suffix in {".json"}:
            path.write_text("{}", encoding="utf-8")
        elif path.suffix in {".yaml", ".yml"}:
            path.write_text("project: {}\n", encoding="utf-8")
        else:
            path.write_bytes(b"model")
    (final_root / "final_manifest.json").write_text(
        json.dumps({"stage": "COMPLETE", "artifact_hashes": {}}),
        encoding="utf-8",
    )
    project_root = Path(__file__).resolve().parents[1]
    bundle = export_hierarchical_bundle(
        project_root, final_root, fresh=True, resume=False
    )
    assert (bundle / "SHA256SUMS.json").is_file()
    assert (final_root / "model_bundle.zip").is_file()
    assert not any("label" in path.name.lower() for path in bundle.rglob("*"))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(bundle / "runtime")
    subprocess.run(
        [sys.executable, "-c", "import bme_eating.hierarchical_pipeline"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    resumed = export_hierarchical_bundle(
        project_root, final_root, fresh=False, resume=True
    )
    assert resumed == bundle
    with zipfile.ZipFile(final_root / "model_bundle.zip") as archive:
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in archive.namelist())


def test_export_resume_finishes_zip_after_bundle_was_written(tmp_path) -> None:
    final_root = tmp_path / "final"
    final_root.mkdir()
    for name in MODEL_FILES:
        path = final_root / name
        if path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        elif path.suffix in {".yaml", ".yml"}:
            path.write_text("project: {}\n", encoding="utf-8")
        else:
            path.write_bytes(b"model")
    manifest_path = final_root / "final_manifest.json"
    manifest_path.write_text(
        json.dumps({"stage": "COMPLETE", "artifact_hashes": {}}),
        encoding="utf-8",
    )
    project_root = Path(__file__).resolve().parents[1]
    bundle = export_hierarchical_bundle(
        project_root, final_root, fresh=True, resume=False
    )
    (final_root / "model_bundle.zip").unlink()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stage"] = "COMPLETE"
    manifest["artifact_hashes"].pop("model_bundle.zip")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resumed = export_hierarchical_bundle(
        project_root, final_root, fresh=False, resume=True
    )

    assert resumed == bundle
    assert (final_root / "model_bundle.zip").is_file()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["stage"] == "EXPORTED"


def test_bundle_verification_rejects_unregistered_files(tmp_path) -> None:
    final_root = tmp_path / "final"
    final_root.mkdir()
    for name in MODEL_FILES:
        path = final_root / name
        if path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        elif path.suffix in {".yaml", ".yml"}:
            path.write_text("project: {}\n", encoding="utf-8")
        else:
            path.write_bytes(b"model")
    (final_root / "final_manifest.json").write_text(
        json.dumps({"stage": "COMPLETE", "artifact_hashes": {}}),
        encoding="utf-8",
    )
    project_root = Path(__file__).resolve().parents[1]
    bundle = export_hierarchical_bundle(
        project_root, final_root, fresh=True, resume=False
    )
    (bundle / "unexpected_labels.csv").write_text("label\n1\n", encoding="utf-8")

    try:
        export_hierarchical_bundle(
            project_root, final_root, fresh=False, resume=True
        )
    except RuntimeError as error:
        assert "file set changed" in str(error)
    else:
        raise AssertionError("Bundle verification accepted an unregistered file")
