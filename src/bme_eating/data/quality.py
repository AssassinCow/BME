from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from bme_eating.data.multisection import (
    EXACT_RECOVERY_CLASSIFICATION,
    QUARANTINE_CLASSIFICATION,
    QUARANTINE_STATUS,
    validate_multisection_preprocess_policy,
)


def _frame_digest(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.reindex(columns=columns).sort_values(columns).reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(selected, index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def _subject_fold_digest(index_dir: Path) -> tuple[str, int]:
    fold_path = index_dir / "subject_folds.json"
    manifest_path = index_dir / "subject_folds.manifest.json"
    assignments = json.loads(fold_path.read_text(encoding="utf-8"))
    if not isinstance(assignments, dict):
        raise RuntimeError("Subject fold assignments must be a JSON object")
    canonical = json.dumps(assignments, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("assignments_sha256") != digest:
        raise RuntimeError("Subject fold manifest fingerprint does not match assignments")
    if int(manifest.get("subjects", -1)) != len(assignments):
        raise RuntimeError("Subject fold manifest subject count does not match assignments")
    return digest, len(assignments)


def build_quality_report(output_root: Path) -> dict[str, Any]:
    index_dir = output_root / "indices"
    records = pd.read_parquet(index_dir / "records.parquet")
    segments = pd.read_parquet(index_dir / "segments.parquet")
    events = pd.read_parquet(index_dir / "events.parquet")
    schema_audit = json.loads((index_dir / "schema_audit.json").read_text(encoding="utf-8"))
    unsupported_count = int(
        schema_audit.get("layout_status_counts", {}).get("unsupported_binary", 0)
    )
    exact_hashes: set[str] = set()
    quarantine_hash_set: set[str] = set()
    multisection_audit: dict[str, Any] = {
        "attachments_audited": 0,
        "classification_counts": {},
    }
    if unsupported_count:
        multisection_audit = json.loads(
            (index_dir / "multisection_audit.json").read_text(encoding="utf-8")
        )
        quarantine_manifest = json.loads(
            (index_dir / "quarantined_attachments.json").read_text(encoding="utf-8")
        )
        exact_hashes, policy_quarantined_hashes = validate_multisection_preprocess_policy(
            records, schema_audit, multisection_audit
        )
        quarantine_entries = quarantine_manifest.get("attachments", [])
        if not isinstance(quarantine_entries, list):
            raise RuntimeError("quarantine manifest attachments must be a list")
        quarantine_hashes: list[str] = []
        for entry in quarantine_entries:
            if set(entry) != {"source_zip_sha256", "classification", "status"}:
                raise RuntimeError("quarantine manifest contains unexpected or missing fields")
            if entry["classification"] != QUARANTINE_CLASSIFICATION:
                raise RuntimeError("quarantine manifest contains an unexpected classification")
            if entry["status"] != QUARANTINE_STATUS:
                raise RuntimeError("quarantine manifest contains an unexpected status")
            quarantine_hashes.append(str(entry["source_zip_sha256"]).lower())
        if len(set(quarantine_hashes)) != len(quarantine_hashes):
            raise RuntimeError("quarantine manifest contains duplicate source ZIP hashes")
        quarantine_hash_set = set(quarantine_hashes)
        if quarantine_hash_set != policy_quarantined_hashes:
            raise RuntimeError("quarantine manifest hashes do not match the audited policy")
    fold_digest, fold_subjects = _subject_fold_digest(index_dir)
    required_segment_columns = {
        "acc_valid_fraction",
        "gyro_valid_fraction",
        "ppg_valid_fraction",
        "ppg_samples_per_row",
        "ppg_available_columns",
    }
    missing_segment_columns = required_segment_columns - set(segments.columns)
    if missing_segment_columns:
        raise RuntimeError(
            "Segments use an obsolete preprocessing schema; rerun full v2 preprocessing. "
            f"Missing columns: {sorted(missing_segment_columns)}"
        )

    ordered = segments.sort_values(["subject_key", "start_ms", "end_ms", "segment_id"])
    subject_overlaps = 0
    for _, group in ordered.groupby("subject_key", sort=False):
        prior_maximum_end = group["end_ms"].cummax().shift(1).to_numpy()
        subject_overlaps += int(
            (
                group["start_ms"].iloc[1:].to_numpy()
                < prior_maximum_end[1:]
            ).sum()
        )
    session_overlaps = 0
    for _, group in ordered.groupby(["subject_key", "session_id"], sort=False):
        group = group.sort_values(["start_ms", "end_ms", "segment_id"])
        prior_maximum_end = group["end_ms"].cummax().shift(1).to_numpy()
        session_overlaps += int(
            (
                group["start_ms"].iloc[1:].to_numpy()
                < prior_maximum_end[1:]
            ).sum()
        )
    parser_by_attachment = (
        segments[["source_zip_sha256", "parser_status"]]
        .drop_duplicates("source_zip_sha256")
        ["parser_status"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    record_hashes = set(records["zip_sha256"].astype(str).str.lower())
    segment_hashes = set(segments["source_zip_sha256"].astype(str).str.lower())
    missing_hashes = record_hashes - segment_hashes
    unaccounted_missing_hashes = missing_hashes - quarantine_hash_set
    quarantined_in_segments = quarantine_hash_set & segment_hashes
    report = {
        "artifact_schema_version": "v2",
        "records": int(len(records)),
        "subjects": int(records["subject_key"].nunique()),
        "segments": int(len(segments)),
        "sessions": int(segments["session_id"].nunique()),
        "events": int(len(events)),
        "valid_duration_events": int(events["valid_duration"].sum()),
        "evaluable_events": int(events.get("evaluable", pd.Series(False, index=events.index)).sum()),
        "coverage_counts": events["coverage"].value_counts().sort_index().to_dict(),
        "parser_status_by_attachment": parser_by_attachment,
        "preprocessed_attachments": len(segment_hashes),
        "missing_preprocessed_attachments": len(missing_hashes),
        "quarantined_attachments": len(quarantine_hash_set),
        "unaccounted_missing_attachments": len(unaccounted_missing_hashes),
        "quarantined_attachments_in_segments": len(quarantined_in_segments),
        "unexpected_preprocessed_attachments": len(segment_hashes - record_hashes),
        "subject_time_overlaps": subject_overlaps,
        "session_time_overlaps": session_overlaps,
        "modality_valid_fraction": {
            modality: float(segments[f"{modality}_valid_fraction"].mean())
            for modality in ("acc", "gyro", "ppg")
        },
        "zero_valid_segments": {
            modality: int((segments[f"{modality}_valid_fraction"] <= 0).sum())
            for modality in ("acc", "gyro", "ppg")
        },
        "ppg_samples_per_row": sorted(
            int(value) for value in segments["ppg_samples_per_row"].dropna().unique()
        ),
        "ppg_available_columns": sorted(
            int(value) for value in segments["ppg_available_columns"].dropna().unique()
        ),
        "schema_layout_files_inspected": int(schema_audit["layout_files_inspected"]),
        "schema_maximum_observed_nonzero_ppg_slot": int(
            schema_audit["maximum_observed_nonzero_ppg_slot"]
        ),
        "schema_layout_status_counts": {
            str(key): int(value)
            for key, value in sorted(schema_audit["layout_status_counts"].items())
        },
        "multisection_attachments_audited": int(
            multisection_audit.get("attachments_audited", -1)
        ),
        "multisection_classification_counts": {
            str(key): int(value)
            for key, value in sorted(
                multisection_audit.get("classification_counts", {}).items()
            )
        },
        "multisection_exact_recovery_attachments": len(exact_hashes),
        "invalid_source_hashes": int(
            (~segments["source_zip_sha256"].astype(str).str.fullmatch(r"[0-9a-fA-F]{64}")).sum()
        ),
        "subject_folds_digest": fold_digest,
        "subject_folds_subjects": fold_subjects,
        "records_digest": _frame_digest(
            records, ["subject_key", "zip_sha256", "zip_size_bytes"]
        ),
        "segments_digest": _frame_digest(
            segments,
            ["subject_key", "segment_id", "session_id", "start_ms", "end_ms", "source_zip_sha256"],
        ),
        "events_digest": _frame_digest(
            events,
            ["subject_key", "event_id", "start_ms", "end_ms", "coverage", "evaluable"],
        ),
    }
    path = index_dir / "quality_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def validate_quality_invariants(report: dict[str, Any]) -> None:
    failures: list[str] = []
    if int(report["subject_time_overlaps"]) != 0:
        failures.append("a subject has overlapping source segments")
    if int(report["invalid_source_hashes"]) != 0:
        failures.append("source ZIP hashes are missing or invalid")
    if int(report["missing_preprocessed_attachments"]) != int(
        report["quarantined_attachments"]
    ):
        failures.append("missing preprocessed attachments are not exactly the quarantined set")
    if int(report["unaccounted_missing_attachments"]) != 0:
        failures.append("some non-quarantined attachments produced no segments")
    if int(report["quarantined_attachments_in_segments"]) != 0:
        failures.append("a quarantined attachment appears in segments")
    if int(report["unexpected_preprocessed_attachments"]) != 0:
        failures.append("segments reference unknown attachments")
    if int(report["subject_folds_subjects"]) != int(report["subjects"]):
        failures.append("subject fold assignments do not cover all subjects")
    if int(report["schema_layout_files_inspected"]) != int(report["records"]):
        failures.append("schema audit did not inspect every attachment")
    allowed_statuses = {"documented_text", "recovered_text_suffix", "unsupported_binary"}
    unexpected_statuses = {
        key: value
        for key, value in report["schema_layout_status_counts"].items()
        if key not in allowed_statuses and int(value) > 0
    }
    if unexpected_statuses:
        failures.append("schema audit contains unsupported attachment statuses")
    if int(report["schema_layout_status_counts"].get("unsupported_binary", 0)) != int(
        report["multisection_attachments_audited"]
    ):
        failures.append("unsupported schema attachments are not fully accounted for")
    if int(report["schema_layout_status_counts"].get("unsupported_binary", 0)):
        if report["multisection_classification_counts"] != {
            EXACT_RECOVERY_CLASSIFICATION: 1,
            QUARANTINE_CLASSIFICATION: 11,
        }:
            failures.append("multisection classification policy changed")
        if int(report["quarantined_attachments"]) != 11:
            failures.append("quarantined attachment count changed")
    if failures:
        raise RuntimeError("Data quality invariants failed: " + "; ".join(failures))


def write_quality_expectations(output_root: Path) -> Path:
    report = build_quality_report(output_root)
    validate_quality_invariants(report)
    path = output_root / "indices" / "quality_expectations.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def validate_configured_expectations(
    report: dict[str, Any],
    expectations: dict[str, Any],
) -> None:
    checks = {
        "records": int(expectations["expected_records"]),
        "subjects": int(expectations["expected_subjects"]),
        "preprocessed_attachments": int(expectations["expected_preprocessed_attachments"]),
        "quarantined_attachments": int(
            expectations["expected_quarantined_conflicting_multisection"]
        ),
    }
    mismatches = [
        f"{name}: observed={report.get(name)} expected={expected}"
        for name, expected in checks.items()
        if int(report.get(name, -1)) != expected
    ]
    for status in (
        "documented_text",
        "recovered_text_suffix",
        "recovered_multisection_deduplicated",
    ):
        observed = int(report.get("parser_status_by_attachment", {}).get(status, 0))
        expected = int(expectations[f"expected_{status}"])
        if observed != expected:
            mismatches.append(f"{status}: observed={observed} expected={expected}")
    for name in ("ppg_samples_per_row", "ppg_available_columns"):
        expected = [int(expectations[f"expected_{name}"])]
        observed = [int(value) for value in report.get(name, [])]
        if observed != expected:
            mismatches.append(f"{name}: observed={observed} expected={expected}")
    observed_slot = int(report.get("schema_maximum_observed_nonzero_ppg_slot", -1))
    expected_slot = int(expectations["expected_maximum_observed_nonzero_ppg_slot"])
    if observed_slot != expected_slot:
        mismatches.append(
            "maximum_observed_nonzero_ppg_slot: "
            f"observed={observed_slot} expected={expected_slot}"
        )
    if mismatches:
        raise RuntimeError("Configured data quality expectations failed: " + "; ".join(mismatches))


def validate_quality_gate(output_root: Path) -> dict[str, Any]:
    expected_path = output_root / "indices" / "quality_expectations.json"
    if not expected_path.exists():
        raise RuntimeError(
            "Quality expectations are not frozen. Review quality_report.json, then run "
            "scripts/validate_data.py --write-expectations on the RTX 4080 computer."
        )
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    observed = build_quality_report(output_root)
    validate_quality_invariants(observed)
    if observed != expected:
        differences = sorted(
            key for key in set(expected) | set(observed) if expected.get(key) != observed.get(key)
        )
        raise RuntimeError(
            "Data quality gate failed; changed fields: " + ", ".join(differences)
        )
    return observed
