import numpy as np

from bme_eating.features.baseline import (
    _compact_motion_bucket_features,
    _compact_ppg_bucket_features,
    _history_slice,
)


def test_history_slices_are_right_closed_without_shared_boundaries():
    timestamps = np.arange(0, 5001, 1000, dtype=np.int64)

    recent = timestamps[_history_slice(timestamps, 2000, 5000)]
    older = timestamps[_history_slice(timestamps, 0, 2000)]

    np.testing.assert_array_equal(recent, np.array([3000, 4000, 5000]))
    np.testing.assert_array_equal(older, np.array([1000, 2000]))
    assert set(recent).isdisjoint(older)


def test_compact_motion_bucket_keeps_terminal_value_and_real_time_trend():
    values = np.zeros((5, 6), dtype=np.float32)
    values[:, 0] = np.array([0.0, 1.0, 50.0, 50.0, 10.0])
    mask = np.zeros((5, 6), dtype=bool)
    mask[[0, 1, 4], :3] = True

    features = _compact_motion_bucket_features(values, mask, "bucket")

    assert features["bucket_acc_last"] == 10.0
    assert features["bucket_acc_valid_fraction"] == 0.6
    valid_positions = np.array([-1.0, -0.5, 1.0])
    valid_values = np.array([0.0, 1.0, 10.0])
    expected_slope = np.dot(
        valid_positions - valid_positions.mean(),
        valid_values - valid_values.mean(),
    ) / np.square(valid_positions - valid_positions.mean()).sum()
    assert np.isclose(features["bucket_acc_slope"], expected_slope)


def test_compact_ppg_bucket_reports_validity_terminal_and_zero_runs():
    values = np.array([4.0, 0.0, 0.0, 99.0, 8.0], dtype=np.float32)
    mask = np.array([True, True, True, False, True])

    features = _compact_ppg_bucket_features(values, mask, "bucket")

    assert features["bucket_ppg_last"] == 8.0
    assert features["bucket_ppg_valid_fraction"] == 0.8
    assert features["bucket_ppg_zero_fraction"] == 0.5
    assert features["bucket_ppg_longest_zero_run_ratio"] == 0.4
