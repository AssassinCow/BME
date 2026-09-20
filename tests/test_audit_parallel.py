import csv
import io
import json
import zipfile

import pytest

from bme_eating.cli import _inspect_selected_layouts, _load_audit_checkpoint
from bme_eating.data.packet_reader import SENSOR_COLUMNS


def _write_sensor_zip(path, timestamp):
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(SENSOR_COLUMNS)
    writer.writerow(
        [timestamp, timestamp, timestamp]
        + list(range(1, 45))
        + [1, 2, 3, 4, 5, 6]
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sensor.txt", buffer.getvalue())


def test_parallel_schema_audit_preserves_order_and_checkpoint(tmp_path):
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    _write_sensor_zip(first, 1000)
    _write_sensor_zip(second, 2000)
    fingerprints = [
        {"zip_path": str(second), "zip_sha256": "b" * 64, "zip_size_bytes": second.stat().st_size},
        {"zip_path": str(first), "zip_sha256": "a" * 64, "zip_size_bytes": first.stat().st_size},
    ]
    checkpoint = tmp_path / "schema_audit.checkpoint.json"

    layouts = _inspect_selected_layouts(
        fingerprints,
        maximum_rows=100,
        workers=2,
        checkpoint_path=checkpoint,
        schema_zips="all",
        completed={},
    )

    assert [layout["zip_name"] for layout in layouts] == [second.name, first.name]
    assert all(layout["status"] == "documented_text" for layout in layouts)
    restored = _load_audit_checkpoint(
        checkpoint,
        "all",
        100,
        json.loads(json.dumps(fingerprints)),
    )
    assert set(restored) == {str(first), str(second)}


def test_parallel_schema_audit_resumes_completed_attachments(tmp_path):
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    _write_sensor_zip(first, 1000)
    _write_sensor_zip(second, 2000)
    fingerprints = [
        {"zip_path": str(first), "zip_sha256": "a" * 64, "zip_size_bytes": first.stat().st_size},
        {"zip_path": str(second), "zip_sha256": "b" * 64, "zip_size_bytes": second.stat().st_size},
    ]
    completed = {
        str(first): {
            "zip_name": first.name,
            "status": "documented_text",
            "rows_scanned": 1,
            "ppg_rows": 1,
            "nonzero_fraction_by_slot": [1.0] * 44,
        }
    }

    layouts = _inspect_selected_layouts(
        fingerprints,
        maximum_rows=100,
        workers=2,
        checkpoint_path=tmp_path / "checkpoint.json",
        schema_zips="all",
        completed=completed,
    )

    assert layouts[0] is completed[str(first)]
    assert layouts[1]["zip_name"] == second.name


def test_parallel_schema_audit_rejects_nonpositive_workers(tmp_path):
    with pytest.raises(ValueError, match="workers must be at least 1"):
        _inspect_selected_layouts(
            [],
            maximum_rows=100,
            workers=0,
            checkpoint_path=tmp_path / "checkpoint.json",
            schema_zips="all",
            completed={},
        )
