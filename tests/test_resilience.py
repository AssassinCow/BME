import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from bme_eating.cli import _feature_cache_path, _feature_job, _preprocess_job
from bme_eating.data.deep_dataset import _load_segment_archive
from bme_eating.data.preprocess import _collapse_duplicate_timestamps, _save_segment, _split_ranges
from bme_eating.types import SensorSeries


def test_save_segment_is_atomic_and_readable(tmp_path):
    output_path = tmp_path / "segment.npz"
    _save_segment(
        output_path,
        np.array([1000, 1010], dtype=np.int64),
        np.ones((2, 6), dtype=np.float64),
        np.ones((2, 6), dtype=bool),
        np.array([1000, 1020], dtype=np.int64),
        np.ones((2, 1), dtype=np.float64),
        np.ones((2, 1), dtype=bool),
        compressed=True,
    )

    with np.load(output_path) as payload:
        assert payload["motion_values"].shape == (2, 6)
        assert payload["ppg_values"].shape == (2, 1)
    assert not list(tmp_path.glob("*.tmp"))


def test_segment_ranges_use_sorted_unique_timestamps():
    series = SensorSeries(
        np.array([1000, 1010, 900, 910, 1010], dtype=np.int64),
        np.arange(5, dtype=np.float32).reshape(-1, 1),
    )
    canonical = _collapse_duplicate_timestamps(series)

    assert canonical.timestamp_ms.tolist() == [900, 910, 1000, 1010]
    assert len(_split_ranges(canonical.timestamp_ms, gap_factor=5.0, minimum_gap_ms=2000)) == 1


def test_feature_job_reports_corrupt_segment(tmp_path):
    segment_path = tmp_path / "corrupt.npz"
    segment_path.write_bytes(b"not an npz archive")

    frame, issue = _feature_job(str(segment_path), pd.DataFrame(), {})

    assert frame.empty
    assert issue is not None
    assert issue["status"] == "feature_extraction_error"
    assert issue["segment_path"] == str(segment_path)
    assert issue["error_type"] in {"ValueError", "BadZipFile", "KeyError"}


def test_feature_job_reuses_atomic_segment_cache(tmp_path, monkeypatch):
    segment_path = tmp_path / "segment.npz"
    segment_path.write_bytes(b"segment fingerprint")
    anchors = pd.DataFrame({"segment_id": ["s"], "timestamp_ms": [1000]})
    feature_config = {
        "window_seconds": 15,
        "include_dyadic": False,
        "motion_bucket_seconds": [3],
        "ppg_bucket_seconds": [15],
    }
    cache_path = _feature_cache_path(
        tmp_path / "cache", str(segment_path), anchors, feature_config
    )
    expected = pd.DataFrame({"feature": [1.0]})
    monkeypatch.setattr(
        "bme_eating.cli.build_segment_features", lambda *args, **kwargs: expected
    )
    first, first_issue = _feature_job(
        str(segment_path), anchors, feature_config, str(cache_path)
    )
    monkeypatch.setattr(
        "bme_eating.cli.build_segment_features",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache was ignored")),
    )
    second, second_issue = _feature_job(
        str(segment_path), anchors, feature_config, str(cache_path)
    )

    assert first_issue is None
    assert second_issue is None
    pd.testing.assert_frame_equal(first, second)
    assert not list(cache_path.parent.glob("*.tmp"))


def test_deep_dataset_reports_corrupt_segment_path(tmp_path):
    segment_path = tmp_path / "corrupt.npz"
    segment_path.write_bytes(b"not an npz archive")

    with pytest.raises(RuntimeError, match=r"corrupt\.npz"):
        _load_segment_archive(segment_path)


def test_preprocess_job_reports_binary_attachment(tmp_path):
    zip_path = tmp_path / "binary.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("info.json", json.dumps({"metadataType": "SensorOriginalData"}))
        archive.writestr("sensor.txt", b"\x00\xff\x99\x05")

    rows, issue = _preprocess_job(
        {
            "zip_path": str(zip_path),
            "subject_key": "S01",
            "zip_sha256": "abc",
        },
        str(tmp_path / "segments"),
        {"ppg_samples_per_row": 20, "packet_timestamp_anchor": "start"},
        compressed=True,
        overwrite=False,
    )

    assert rows == []
    assert issue is not None
    assert issue["status"] == "unsupported_binary"
    assert issue["member_name"] == "sensor.txt"


def test_deep_dataset_rejects_unsorted_segment_cache(tmp_path):
    segment_path = tmp_path / "unsorted.npz"
    np.savez(
        segment_path,
        motion_timestamp_ms=np.array([2, 1]),
        motion_values=np.zeros((2, 6), dtype=np.float32),
        motion_mask=np.ones((2, 6), dtype=np.uint8),
        ppg_timestamp_ms=np.array([1, 2]),
        ppg_values=np.zeros((2, 1), dtype=np.float32),
        ppg_mask=np.ones((2, 1), dtype=np.uint8),
    )
    with pytest.raises(RuntimeError, match="strictly increasing"):
        _load_segment_archive(segment_path)
