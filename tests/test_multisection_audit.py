import csv
import io
import json
import zipfile

import pandas as pd
import pytest

from bme_eating.data.multisection import (
    audit_multisection_zip,
    audit_repeated_header_attachments,
    validate_multisection_preprocess_policy,
    write_quarantine_manifest,
)
from bme_eating.data.packet_reader import (
    SENSOR_COLUMNS,
    UnsupportedSensorFormatError,
    parse_multisection_sensor_zip,
)


def _row(timestamp, value=1.0):
    return [timestamp, timestamp, timestamp] + [value] * 44 + [value] * 6


def _section(rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(SENSOR_COLUMNS)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_multisection_zip(path, sections, prefix=b"\x00\xffopaque-prefix"):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sensor.txt", prefix + b"".join(sections))


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


def test_multisection_report_includes_header_at_byte_zero(tmp_path):
    path = tmp_path / "documented-prefix.zip"
    rows = [_row(1000), _row(1100, 2.0)]
    _write_multisection_zip(path, [_section(rows), _section(rows)], prefix=b"")
    records = pd.DataFrame(
        [{"zip_path": str(path), "zip_sha256": "a" * 64}]
    )
    schema_audit = {
        "layouts": [{"zip_name": path.name, "status": "repeated_header"}]
    }

    report = audit_repeated_header_attachments(records, schema_audit, 20, workers=2)

    assert report["attachments_audited"] == 1
    assert report["classification_counts"] == {"exact_duplicate_overlap": 1}
    assert report["results"][0]["binary_prefix_bytes"] == 0


def test_multisection_parser_deduplicates_exact_expanded_samples(tmp_path):
    path = tmp_path / "duplicate.zip"
    rows = [_row(1000, 1.0), _row(1100, 2.0)]
    _write_multisection_zip(path, [_section(rows), _section(rows)])

    parsed = parse_multisection_sensor_zip(path, ppg_samples_per_row=2)

    assert parsed.parser_status == "recovered_multisection_deduplicated"
    assert parsed.left_censored is True
    assert len(parsed.acc.timestamp_ms) == 2
    assert len(parsed.gyro.timestamp_ms) == 2
    assert len(parsed.ppg.timestamp_ms) == 4
    assert len(set(parsed.acc.timestamp_ms.tolist())) == 2
    assert len(set(parsed.ppg.timestamp_ms.tolist())) == 4


def test_multisection_parser_deduplicates_packets_before_timestamp_expansion(tmp_path):
    path = tmp_path / "submillisecond-packets.zip"
    first = _row(1000, 1.0)
    first[4] = 2.0
    second = _row(1001, 3.0)
    second[4] = 4.0
    rows = [first, second]
    _write_multisection_zip(path, [_section(rows), _section(rows)], prefix=b"")

    parsed = parse_multisection_sensor_zip(path, ppg_samples_per_row=2)

    assert parsed.ppg.timestamp_ms.tolist() == [1000, 1000, 1001, 1002]
    assert parsed.ppg.values[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_multisection_parser_rejects_conflicting_expanded_samples(tmp_path):
    path = tmp_path / "conflict.zip"
    _write_multisection_zip(
        path,
        [
            _section([_row(1000, 1.0), _row(1100, 2.0)]),
            _section([_row(1000, 1.0), _row(1100, 3.0)]),
        ],
    )

    with pytest.raises(UnsupportedSensorFormatError, match="samples conflict"):
        parse_multisection_sensor_zip(path, ppg_samples_per_row=2)


def _policy_inputs(tmp_path):
    hashes = [f"{value:064x}" for value in range(1, 13)]
    records = pd.DataFrame(
        [
            {"zip_path": str(tmp_path / f"attachment-{index}.zip"), "zip_sha256": digest}
            for index, digest in enumerate(hashes)
        ]
    )
    schema = {
        "layout_files_inspected": 12,
        "layout_status_counts": {"repeated_header": 12},
        "layouts": [
            {
                "zip_name": f"attachment-{index}.zip",
                "status": "repeated_header",
                "error": "standard sensor header occurs more than once",
            }
            for index in range(12)
        ]
    }
    results = [
        {
            "source_zip_sha256": digest,
            "classification": (
                "exact_duplicate_overlap" if index == 0 else "conflicting_overlap"
            ),
        }
        for index, digest in enumerate(hashes)
    ]
    audit = {
        "attachments_audited": 12,
        "automatic_recovery_performed": False,
        "classification_counts": {
            "conflicting_overlap": 11,
            "exact_duplicate_overlap": 1,
        },
        "results": results,
    }
    return records, schema, audit, hashes


def test_multisection_policy_requires_exact_hash_and_classification_set(tmp_path):
    records, schema, audit, hashes = _policy_inputs(tmp_path)

    exact, quarantined = validate_multisection_preprocess_policy(records, schema, audit)

    assert exact == {hashes[0]}
    assert quarantined == {digest: "conflicting_overlap" for digest in hashes[1:]}

    changed = json.loads(json.dumps(audit))
    changed["results"][0]["source_zip_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="hashes do not match"):
        validate_multisection_preprocess_policy(records, schema, changed)

    changed = json.loads(json.dumps(audit))
    changed["results"][0]["classification"] = "conflicting_overlap"
    changed["classification_counts"] = {"conflicting_overlap": 12}
    with pytest.raises(RuntimeError, match="classifications changed"):
        validate_multisection_preprocess_policy(records, schema, changed)


def test_multisection_policy_quarantines_nonmonotonic_sections(tmp_path):
    records, schema, audit, hashes = _policy_inputs(tmp_path)
    audit["results"][-1]["classification"] = "nonmonotonic_section"
    audit["classification_counts"] = {
        "conflicting_overlap": 10,
        "exact_duplicate_overlap": 1,
        "nonmonotonic_section": 1,
    }

    exact, quarantined = validate_multisection_preprocess_policy(
        records, schema, audit, expected_exact=1, expected_quarantined=11
    )

    assert exact == {hashes[0]}
    assert quarantined[hashes[-1]] == "nonmonotonic_section"


def test_quarantine_manifest_contains_no_identity_or_path(tmp_path):
    quarantined = {
        "a" * 64: "conflicting_overlap",
        "b" * 64: "nonmonotonic_section",
    }
    path = write_quarantine_manifest(tmp_path, quarantined)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["attachments"] == [
        {
            "source_zip_sha256": "a" * 64,
            "classification": "conflicting_overlap",
            "status": "quarantined_unsafe_multisection",
        },
        {
            "source_zip_sha256": "b" * 64,
            "classification": "nonmonotonic_section",
            "status": "quarantined_unsafe_multisection",
        },
    ]
    serialized = json.dumps(payload)
    assert "zip_path" not in serialized
    assert "subject" not in serialized
    assert "filename" not in serialized
