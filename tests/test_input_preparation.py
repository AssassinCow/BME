import csv
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from bme_eating.config import load_config
from bme_eating.data import input_preparation
from bme_eating.data.packet_reader import SENSOR_COLUMNS
from bme_eating.data.stats_fusion_inputs import verify_canonical_statsfusion_inputs


@pytest.fixture
def config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v414_full_data.yaml")


@pytest.fixture
def raw_inputs(tmp_path, monkeypatch):
    data_root, output_root = tmp_path / "raw", tmp_path / "output"
    metadata = data_root / "derived/valid_metadata"
    metadata.mkdir(parents=True)
    (data_root / "manifests").mkdir()
    sensor_zip = data_root / "raw/sensor_attachments/HNU21001/sample.zip"
    sensor_zip.parent.mkdir(parents=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(SENSOR_COLUMNS)
    for timestamp in range(0, 120_000, 100):
        writer.writerow([timestamp, timestamp, timestamp] + list(range(1, 21)) + [0] * 24
                        + [1, 2, 3, 4, 5, 6])
    with zipfile.ZipFile(sensor_zip, "w") as archive:
        archive.writestr("sensor.txt", buffer.getvalue())
    duplicate_zip = sensor_zip.with_name("duplicate.zip")
    duplicate = io.StringIO()
    writer = csv.writer(duplicate, delimiter="\t", lineterminator="\n")
    writer.writerow(SENSOR_COLUMNS)
    for timestamp in range(300_000, 420_000, 100):
        writer.writerow([timestamp, timestamp, timestamp] + list(range(1, 21)) + [0] * 24
                        + [1, 2, 3, 4, 5, 6])
    with zipfile.ZipFile(duplicate_zip, "w") as archive:
        archive.writestr("sensor.txt", duplicate.getvalue() * 2)
    conflict_zip = sensor_zip.with_name("conflict.zip")
    with zipfile.ZipFile(conflict_zip, "w") as archive:
        archive.writestr("sensor.txt", buffer.getvalue()
                         + buffer.getvalue().replace("\t1\t2\t3\t4\t5\t6", "\t2\t2\t3\t4\t5\t6"))
    (metadata / "sensor.csv").write_text("externalid\nHNU21001\n", encoding="utf-8")
    (metadata / "meal.csv").write_text(
        "externalid,beforeTime,afterTime,dietaryHand,wearHand,uniqueid\n"
        "HNU21001,30000,60000,left,right,event1\n", encoding="utf-8")
    state = {"attachments": [{"externalid": "HNU21001", "status": "downloaded",
             "relative_path": str(path.relative_to(data_root)),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in (sensor_zip, duplicate_zip, conflict_zip)]}
    (data_root / "manifests/download_state.json").write_text(json.dumps(state), encoding="utf-8")
    recipe = deepcopy(input_preparation.PREPARATION_RECIPE)
    recipe["quality_gates"].update({"expected_records": 3, "expected_subjects": 1,
        "expected_documented_text": 1, "expected_recovered_text_suffix": 0,
        "expected_recovered_multisection_deduplicated": 1, "expected_quarantined_multisection": 1,
        "expected_preprocessed_attachments": 2})
    monkeypatch.setattr(input_preparation, "PREPARATION_RECIPE", recipe)
    monkeypatch.setenv("BME_DATA_ROOT", str(data_root))
    monkeypatch.setenv("BME_OUTPUT_ROOT", str(output_root))
    monkeypatch.setenv("BME_SUBJECT_SALT", "synthetic-test-salt")
    monkeypatch.setattr(input_preparation, "_load_project_dotenv", lambda config: None)
    return data_root, output_root, sensor_zip


def test_preparation_uses_original_recipe_without_changing_training(config):
    original = deepcopy(config)
    prepared = input_preparation.preparation_config(config, 1)
    assert config == original
    assert prepared["project"]["artifact_schema_version"] == "v2"
    assert prepared["data"]["motion_target_hz"] == 100
    assert prepared["data"]["ppg_target_hz"] == 50
    assert prepared["quality_gates"]["expected_quarantined_multisection"] == 34
    assert prepared["preprocess"]["workers"] == 1
    with pytest.raises(ValueError, match="workers"):
        input_preparation.preparation_config(config, 0)


def test_raw_to_v2_and_canonical_with_resume_and_overwrite_protection(config, raw_inputs):
    _, output_root, _ = raw_inputs
    result = input_preparation.prepare_inputs(config, workers=1, fresh=True)
    assert result["canonical_protocol"] == "statsfusion-r3.2"
    manifest_path = output_root / "v2/preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    report = json.loads((output_root / "v2/indices/quality_report.json").read_text(encoding="utf-8"))
    assert report["preprocessed_attachments"] == 2
    assert report["quarantined_attachments"] == 1
    verify_canonical_statsfusion_inputs(output_root / "v2", output_root / "v4")
    files = {path: (path.read_bytes(), path.stat().st_mtime_ns)
             for path in output_root.rglob("*") if path.is_file()}
    input_preparation.prepare_inputs(config, workers=2, resume=True)
    with pytest.raises(FileExistsError):
        input_preparation.prepare_inputs(config, workers=1, fresh=True)
    assert files == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}
    archive = next((output_root / "v2/segments").glob("*.npz"))
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="artifact changed"):
        input_preparation.prepare_inputs(config, stage="v2", workers=1, resume=True)


def test_v2_interruption_resume_and_changed_raw_rejected(config, raw_inputs, monkeypatch):
    data_root, output_root, _ = raw_inputs
    original = input_preparation.command_preprocess
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("injected interruption")
    monkeypatch.setattr(input_preparation, "command_preprocess", interrupt)
    with pytest.raises(KeyboardInterrupt):
        input_preparation.prepare_inputs(config, stage="v2", workers=1, fresh=True)
    manifest = json.loads((output_root / "v2/preparation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PAUSED"
    monkeypatch.setattr(input_preparation, "command_preprocess", original)
    input_preparation.prepare_inputs(config, stage="v2", workers=1, resume=True)
    monkeypatch.setenv("BME_SUBJECT_SALT", "different-salt")
    with pytest.raises(RuntimeError, match="changed"):
        input_preparation.prepare_inputs(config, stage="v2", workers=1, resume=True)
    monkeypatch.setenv("BME_SUBJECT_SALT", "synthetic-test-salt")
    meal = data_root / "derived/valid_metadata/meal.csv"
    meal.write_text(meal.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        input_preparation.prepare_inputs(config, stage="v2", workers=1, resume=True)


def test_missing_salt_and_raw_hash_mismatch_fail_before_writes(config, raw_inputs, monkeypatch):
    _, output_root, sensor_zip = raw_inputs
    monkeypatch.delenv("BME_SUBJECT_SALT")
    with pytest.raises(RuntimeError, match="original BME_SUBJECT_SALT"):
        input_preparation.prepare_inputs(config, stage="v2", workers=1, fresh=True)
    assert not output_root.exists()
    monkeypatch.setenv("BME_SUBJECT_SALT", "synthetic-test-salt")
    sensor_zip.write_bytes(sensor_zip.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="download manifest"):
        input_preparation.prepare_inputs(config, stage="v2", workers=1, fresh=True)
    assert not output_root.exists()


def test_canonical_only_preserves_v2(config, raw_inputs):
    _, output_root, _ = raw_inputs
    input_preparation.prepare_inputs(config, stage="v2", workers=1, fresh=True)
    indices = {path: (path.read_bytes(), path.stat().st_mtime_ns)
               for path in (output_root / "v2").rglob("*") if path.is_file()}
    input_preparation.prepare_inputs(config, stage="canonical", workers=1, fresh=True)
    assert indices == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in indices}


def test_preparation_audits_headers_after_100000_rows(config, raw_inputs, monkeypatch):
    data_root, output_root, sensor_zip = raw_inputs
    duplicate_zip = sensor_zip.with_name("duplicate.zip")
    header = "\t".join(SENSOR_COLUMNS) + "\n"
    data_row = "\t".join(map(str, [1000] * 3 + [1] * 20 + [0] * 24 + [1] * 6)) + "\n"
    with zipfile.ZipFile(duplicate_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("sensor.txt", header + data_row * 100_001 + header + data_row)
    state_path = data_root / "manifests/download_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for attachment in state["attachments"]:
        if Path(attachment["relative_path"]).name == duplicate_zip.name:
            attachment["sha256"] = hashlib.sha256(duplicate_zip.read_bytes()).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")

    def stop_after_schema_audit(*args, **kwargs):
        raise KeyboardInterrupt("inspect the schema audit before multisection preprocessing")

    monkeypatch.setattr(input_preparation, "command_audit_multisection", stop_after_schema_audit)
    with pytest.raises(KeyboardInterrupt):
        input_preparation.prepare_inputs(config, stage="v2", workers=1, fresh=True)

    report = json.loads((output_root / "v2/indices/schema_audit.json").read_text(encoding="utf-8"))
    assert report["header_scan_scope"] == "full_member"
    assert report["ppg_sample_maximum_rows"] == 100_000
    assert report["layout_status_counts"] == {"documented_text": 1, "repeated_header": 2}
    late_layout = next(layout for layout in report["layouts"]
                       if layout["zip_name"] == duplicate_zip.name)
    assert late_layout["status"] == "repeated_header"


def test_preparation_refuses_resume_with_old_scan_source(config, raw_inputs, monkeypatch):
    _, output_root, _ = raw_inputs

    def stop_after_schema_audit(*args, **kwargs):
        raise KeyboardInterrupt("injected interruption")

    monkeypatch.setattr(input_preparation, "command_audit_multisection", stop_after_schema_audit)
    with pytest.raises(KeyboardInterrupt):
        input_preparation.prepare_inputs(config, stage="v2", workers=1, fresh=True)
    manifest_path = output_root / "v2/preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["source"]["data/packet_reader.py"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="recipe/source changed; resume refused"):
        input_preparation.prepare_inputs(config, stage="v2", workers=1, resume=True)


def test_preparation_cli_help():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts/prepare_statsfusion_inputs.py"),
                             "--help"], capture_output=True, text=True, check=True)
    assert "v2,canonical,all" in result.stdout
