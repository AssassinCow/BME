import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bme_eating.data.deep_dataset import SegmentBalancedBatchSampler
from bme_eating.data.labels import build_anchor_index, classify_event_coverage
from bme_eating.data.preprocess import _antialias_series, assign_virtual_sessions, _save_segment
from bme_eating.data.quality import build_quality_report, validate_quality_invariants
from bme_eating.data.session import SessionWindowReader
from bme_eating.features.signal import masked_spectral_summary, ppg_quality_features
from bme_eating.metrics import partition_evaluation_events
from bme_eating.postprocess import probabilities_to_events
from bme_eating.types import SensorSeries


def _write_segment(path: Path, timestamps: np.ndarray, acc_valid: np.ndarray) -> None:
    motion_mask = np.repeat(acc_valid[:, None], 6, axis=1)
    _save_segment(
        path,
        timestamps,
        np.repeat(timestamps[:, None].astype(np.float32), 6, axis=1),
        motion_mask,
        timestamps,
        timestamps[:, None].astype(np.float32),
        acc_valid[:, None],
        compressed=False,
    )


def test_session_reader_crosses_segment_boundary_but_not_session(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    other = tmp_path / "other.npz"
    _write_segment(first, np.asarray([0, 1000, 2000]), np.ones(3, dtype=bool))
    _write_segment(second, np.asarray([3000, 4000]), np.ones(2, dtype=bool))
    _write_segment(other, np.asarray([5000, 6000]), np.ones(2, dtype=bool))
    segments = pd.DataFrame(
        [
            {"session_id": "a", "segment_id": "s1", "segment_path": first, "start_ms": 0, "end_ms": 2000},
            {"session_id": "a", "segment_id": "s2", "segment_path": second, "start_ms": 3000, "end_ms": 4000},
            {"session_id": "b", "segment_id": "s3", "segment_path": other, "start_ms": 5000, "end_ms": 6000},
        ]
    )
    payload = SessionWindowReader(segments).read("a", 1000, 5000)
    assert payload["motion_timestamp_ms"].tolist() == [1000, 2000, 3000, 4000]


def test_virtual_session_assignment_respects_gap_and_is_path_independent():
    segments = pd.DataFrame(
        [
            {"subject_key": "subject", "segment_id": "hash_s000", "segment_path": "C:/a", "start_ms": 0, "end_ms": 1000},
            {"subject_key": "subject", "segment_id": "hash_s001", "segment_path": "D:/b", "start_ms": 3000, "end_ms": 4000},
            {"subject_key": "subject", "segment_id": "hash_s002", "segment_path": "D:/c", "start_ms": 8001, "end_ms": 9000},
        ]
    )
    assigned = assign_virtual_sessions(segments, maximum_gap_ms=3000)
    assert assigned.loc[0, "session_id"] == assigned.loc[1, "session_id"]
    assert assigned.loc[1, "session_id"] != assigned.loc[2, "session_id"]
    assert assigned.loc[1, "previous_segment_id"] == "hash_s000"


def test_coverage_uses_acc_mask_and_reports_boundary_observability(tmp_path):
    path = tmp_path / "segment.npz"
    timestamps = np.arange(0, 11_000, 1000, dtype=np.int64)
    valid = np.ones(len(timestamps), dtype=bool)
    valid[5:7] = False
    _write_segment(path, timestamps, valid)
    segments = pd.DataFrame(
        [{"subject_key": "s", "segment_id": "x", "segment_path": path, "start_ms": 0, "end_ms": 10_000}]
    )
    events = pd.DataFrame(
        [{"event_id": "e", "subject_key": "s", "start_ms": 1000, "end_ms": 9000, "valid_duration": True}]
    )
    classified = classify_event_coverage(events, segments, output_step_seconds=1)
    assert classified.iloc[0].coverage == "partial"
    assert classified.iloc[0].start_observed
    assert classified.iloc[0].end_observed
    assert classified.iloc[0].max_gap_seconds == 2.0


def test_event_distance_uses_all_subject_events_not_only_intersections(tmp_path):
    segments = pd.DataFrame(
        [{"subject_key": "s", "segment_id": "x", "session_id": "a", "segment_path": tmp_path / "missing.npz", "start_ms": 0, "end_ms": 30_000}]
    )
    events = pd.DataFrame(
        [{"event_id": "e", "subject_key": "s", "start_ms": 31_000, "end_ms": 40_000, "hand_relation": "same", "valid_duration": True, "start_observed": False, "end_observed": False}]
    )
    anchors = build_anchor_index(segments, events, 3, tmp_path / "anchors.parquet")
    assert anchors.iloc[-1].distance_to_event_seconds == 1.0


def test_balanced_sampler_has_exact_positive_fraction_per_batch():
    rows = []
    for index in range(20):
        positive = index < 8
        rows.append(
            {
                "segment_id": "s",
                "event_id": f"e{index % 4}" if positive else "",
                "state_target": float(positive),
                "distance_to_event_seconds": 60.0 if index < 14 else 3600.0,
            }
        )
    anchors = pd.DataFrame(rows)
    sampler = SegmentBalancedBatchSampler(anchors, 10, 5, 0.4, 2026)
    for batch in sampler:
        assert int((anchors.loc[batch, "state_target"] > 0).sum()) == 4


def test_downsampling_filter_attenuates_above_nyquist_energy():
    timestamps = np.arange(0, 4000, 2, dtype=np.int64)
    seconds = timestamps / 1000.0
    values = np.sin(2 * np.pi * 100.0 * seconds).astype(np.float32)[:, None]
    filtered, source_hz = _antialias_series(
        SensorSeries(timestamps, values), target_hz=50.0, maximum_gap_ms=250
    )
    assert 499.0 <= source_hz <= 501.0
    assert float(np.std(filtered.values)) < float(np.std(values)) * 0.1


def test_ppg_quality_handles_many_isolated_valid_samples():
    values = np.sin(np.arange(120, dtype=np.float64))
    mask = np.zeros(120, dtype=bool)
    mask[::2] = True
    features, quality = ppg_quality_features(values, mask, sampling_hz=50.0)
    assert np.isfinite(features).all()
    assert features[5] == 0.0
    assert np.isfinite(quality)


def test_spectral_summary_does_not_compress_across_missing_gap():
    values = np.sin(np.arange(64, dtype=np.float64))
    mask = np.zeros(64, dtype=bool)
    mask[0:12] = True
    mask[40:52] = True
    summary = masked_spectral_summary(values, mask, sampling_hz=50.0)
    assert summary["dominant_frequency"] == 0.0
    assert summary["spectral_entropy"] == 0.0


def test_dtp_evaluation_partitions_non_evaluable_events_as_ignore():
    events = pd.DataFrame(
        [
            {
                "subject_key": "validation",
                "event_id": "visible",
                "start_ms": 0,
                "end_ms": 1000,
                "valid_duration": True,
                "evaluable": True,
            },
            {
                "subject_key": "validation",
                "event_id": "partial",
                "start_ms": 2000,
                "end_ms": 3000,
                "valid_duration": True,
                "evaluable": False,
            },
            {
                "subject_key": "test",
                "event_id": "held_out",
                "start_ms": 4000,
                "end_ms": 5000,
                "valid_duration": True,
                "evaluable": True,
            },
        ]
    )
    truth, ignore = partition_evaluation_events(events, {"validation"})
    assert truth["event_id"].tolist() == ["visible"]
    assert ignore["event_id"].tolist() == ["partial"]


def test_quality_report_detects_overlaps_across_different_sessions(tmp_path):
    index_dir = tmp_path / "indices"
    index_dir.mkdir()
    pd.DataFrame(
        [{"subject_key": "s", "zip_sha256": "a" * 64, "zip_size_bytes": 1}]
    ).to_parquet(index_dir / "records.parquet", index=False)
    pd.DataFrame(
        [
            {
                "subject_key": "s",
                "segment_id": "a",
                "session_id": "session-a",
                "start_ms": 0,
                "end_ms": 2000,
                "source_zip_sha256": "a" * 64,
                "parser_status": "documented_text",
                "acc_valid_fraction": 1.0,
                "gyro_valid_fraction": 0.5,
                "ppg_valid_fraction": 0.75,
                "ppg_samples_per_row": 20,
                "ppg_available_columns": 44,
            },
            {
                "subject_key": "s",
                "segment_id": "b",
                "session_id": "session-b",
                "start_ms": 1000,
                "end_ms": 3000,
                "source_zip_sha256": "b" * 64,
                "parser_status": "documented_text",
                "acc_valid_fraction": 1.0,
                "gyro_valid_fraction": 0.5,
                "ppg_valid_fraction": 0.75,
                "ppg_samples_per_row": 20,
                "ppg_available_columns": 44,
            },
        ]
    ).to_parquet(index_dir / "segments.parquet", index=False)
    pd.DataFrame(
        [
            {
                "subject_key": "s",
                "event_id": "e",
                "start_ms": 0,
                "end_ms": 1000,
                "valid_duration": True,
                "coverage": "full",
                "evaluable": True,
            }
        ]
    ).to_parquet(index_dir / "events.parquet", index=False)
    assignments = {"s": 0}
    canonical = json.dumps(assignments, sort_keys=True, separators=(",", ":")).encode()
    (index_dir / "subject_folds.json").write_text(
        json.dumps(assignments), encoding="utf-8"
    )
    (index_dir / "subject_folds.manifest.json").write_text(
        json.dumps(
            {
                "assignments_sha256": hashlib.sha256(canonical).hexdigest(),
                "subjects": 1,
            }
        ),
        encoding="utf-8",
    )
    (index_dir / "schema_audit.json").write_text(
        json.dumps(
            {
                "layout_files_inspected": 1,
                "maximum_observed_nonzero_ppg_slot": 20,
                "layout_status_counts": {"documented_text": 1},
            }
        ),
        encoding="utf-8",
    )

    report = build_quality_report(tmp_path)
    assert report["subject_time_overlaps"] == 1
    assert report["session_time_overlaps"] == 0
    with pytest.raises(RuntimeError, match="overlapping source segments"):
        validate_quality_invariants(report)


def test_cross_attachment_session_produces_one_event_without_subject_leakage(tmp_path):
    first = tmp_path / "subject-a-first.npz"
    second = tmp_path / "subject-a-second.npz"
    other = tmp_path / "subject-b.npz"
    _write_segment(first, np.asarray([0, 1000, 2000]), np.ones(3, dtype=bool))
    _write_segment(second, np.asarray([3000, 4000, 5000]), np.ones(3, dtype=bool))
    _write_segment(other, np.asarray([0, 1000, 2000]), np.ones(3, dtype=bool))
    segments = pd.DataFrame(
        [
            {
                "subject_key": "a",
                "session_id": "a-session",
                "segment_id": "a1",
                "segment_path": first,
                "start_ms": 0,
                "end_ms": 2000,
            },
            {
                "subject_key": "a",
                "session_id": "a-session",
                "segment_id": "a2",
                "segment_path": second,
                "start_ms": 3000,
                "end_ms": 5000,
            },
            {
                "subject_key": "b",
                "session_id": "b-session",
                "segment_id": "b1",
                "segment_path": other,
                "start_ms": 0,
                "end_ms": 2000,
            },
        ]
    )
    history = SessionWindowReader(segments).read("a-session", 0, 5000)
    assert history["motion_timestamp_ms"].tolist() == [0, 1000, 2000, 3000, 4000, 5000]
    predictions = pd.DataFrame(
        {
            "subject_key": ["a"] * 6,
            "session_id": ["a-session"] * 6,
            "timestamp_ms": [0, 1000, 2000, 3000, 4000, 5000],
            "state_probability": [0.0, 0.9, 0.9, 0.9, 0.9, 0.0],
            "start_probability": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "end_probability": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    )
    events = probabilities_to_events(predictions, 0.1, 0.6, 0.3, 0, 0, 2)
    assert len(events) == 1
    assert events.iloc[0].subject_key == "a"
