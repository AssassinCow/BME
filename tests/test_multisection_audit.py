import csv
import io
import zipfile

import pandas as pd

from bme_eating.data.multisection import (
    audit_multisection_zip,
    audit_repeated_header_attachments,
)
from bme_eating.data.packet_reader import SENSOR_COLUMNS


def _row(timestamp, value=1.0):
    return [timestamp, timestamp, timestamp] + [value] * 44 + [value] * 6


def _section(rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(SENSOR_COLUMNS)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_multisection_zip(path, sections):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sensor.txt", b"\x00\xffopaque-prefix" + b"".join(sections))


def test_multisection_audit_recognizes_exact_duplicate_overlap(tmp_path):
    path = tmp_path / "duplicate.zip"
    rows = [_row(1000), _row(1100, 2.0)]
    _write_multisection_zip(path, [_section(rows), _section(rows)])

    result = audit_multisection_zip(path, "a" * 64, 20)

    assert result["classification"] == "exact_duplicate_overlap"
    assert result["header_count"] == 2
    assert result["cross_section_conflicting_packets"] == 0
    assert result["relationship_counts"] == {"exact_duplicate_overlap": 3}


def test_multisection_audit_detects_conflicting_packet_values(tmp_path):
    path = tmp_path / "conflict.zip"
    _write_multisection_zip(
        path,
        [
            _section([_row(1000, 1.0), _row(1100, 2.0)]),
            _section([_row(1000, 1.0), _row(1100, 3.0)]),
        ],
    )

    result = audit_multisection_zip(path, "b" * 64, 20)

    assert result["classification"] == "conflicting_overlap"
    assert result["cross_section_conflicting_packets"] == 3


def test_multisection_audit_detects_partial_overlap(tmp_path):
    path = tmp_path / "partial.zip"
    _write_multisection_zip(
        path,
        [
            _section([_row(1000), _row(1100), _row(1200)]),
            _section([_row(1100), _row(1250)]),
        ],
    )

    result = audit_multisection_zip(path, "c" * 64, 20)

    assert result["classification"] == "partial_overlap"
    assert result["relationship_counts"] == {"partial_overlap": 3}


def test_multisection_audit_blocks_mixed_exact_and_disjoint_modalities(tmp_path):
    path = tmp_path / "mixed.zip"
    left = _row(1000)
    right = _row(1000)
    right[1] = 2000
    _write_multisection_zip(path, [_section([left]), _section([right])])

    result = audit_multisection_zip(path, "f" * 64, 20)

    assert result["classification"] == "mixed_relationships"


def test_multisection_audit_detects_internal_timestamp_regression(tmp_path):
    path = tmp_path / "regression.zip"
    _write_multisection_zip(
        path,
        [
            _section([_row(1000), _row(1200), _row(1100)]),
            _section([_row(1300)]),
        ],
    )

    result = audit_multisection_zip(path, "d" * 64, 20)

    assert result["classification"] == "nonmonotonic_section"
    assert result["timestamp_regressions"] == 3


def test_multisection_report_omits_paths_names_subjects_and_raw_values(tmp_path):
    path = tmp_path / "private-name.zip"
    rows = [_row(1000), _row(1100, 2.0)]
    _write_multisection_zip(path, [_section(rows), _section(rows)])
    records = pd.DataFrame(
        [
            {
                "zip_path": str(path),
                "zip_sha256": "e" * 64,
                "subject_key": "private-subject",
            }
        ]
    )
    schema_audit = {
        "layouts": [
            {
                "zip_name": path.name,
                "status": "unsupported_binary",
                "error": "standard sensor header occurs more than once",
            }
        ]
    }

    report = audit_repeated_header_attachments(records, schema_audit, 20)
    serialized = str(report)

    assert str(path) not in serialized
    assert path.name not in serialized
    assert "private-subject" not in serialized
    assert "source_zip_sha256" in report["results"][0]
