from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bme_eating.features.signal import ppg_quality_features, robust_statistics, spectral_summary


def _slice(timestamp_ms: np.ndarray, start_ms: int, end_ms: int) -> slice:
    return slice(
        int(np.searchsorted(timestamp_ms, start_ms, side="left")),
        int(np.searchsorted(timestamp_ms, end_ms, side="right")),
    )


def _add_statistics(output: dict[str, float], prefix: str, values: np.ndarray) -> None:
    for name, value in robust_statistics(values).items():
        output[f"{prefix}_{name}"] = value


def _motion_window_features(
    values: np.ndarray,
    mask: np.ndarray,
    sampling_hz: float,
    prefix: str = "local",
) -> dict[str, float]:
    output: dict[str, float] = {}
    names = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
    cleaned = values.copy()
    cleaned[~mask] = np.nan
    for column, name in enumerate(names):
        valid = cleaned[:, column]
        valid = valid[np.isfinite(valid)]
        _add_statistics(output, f"{prefix}_{name}", valid)
        output[f"{prefix}_{name}_valid_fraction"] = float(mask[:, column].mean())
    for start, sensor_name in ((0, "acc_mag"), (3, "gyro_mag")):
        valid_rows = mask[:, start : start + 3].all(axis=1)
        magnitude = np.linalg.norm(values[valid_rows, start : start + 3], axis=1)
        _add_statistics(output, f"{prefix}_{sensor_name}", magnitude)
        for name, value in spectral_summary(magnitude, sampling_hz).items():
            output[f"{prefix}_{sensor_name}_{name}"] = value
        if len(magnitude) > 1:
            jerk = np.diff(magnitude) * sampling_hz
            output[f"{prefix}_{sensor_name}_jerk_std"] = float(np.std(jerk))
            output[f"{prefix}_{sensor_name}_jerk_rms"] = float(
                np.sqrt(np.mean(np.square(jerk)))
            )
            output[f"{prefix}_{sensor_name}_jerk_p95"] = float(
                np.percentile(np.abs(jerk), 95)
            )
        else:
            output[f"{prefix}_{sensor_name}_jerk_std"] = 0.0
            output[f"{prefix}_{sensor_name}_jerk_rms"] = 0.0
            output[f"{prefix}_{sensor_name}_jerk_p95"] = 0.0
    for start, sensor_name in ((0, "acc"), (3, "gyro")):
        block = cleaned[:, start : start + 3]
        for left, right, suffix in ((0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")):
            valid = np.isfinite(block[:, left]) & np.isfinite(block[:, right])
            correlation = (
                float(np.corrcoef(block[valid, left], block[valid, right])[0, 1])
                if valid.sum() >= 3
                and np.std(block[valid, left]) > 0
                and np.std(block[valid, right]) > 0
                else 0.0
            )
            output[f"{prefix}_{sensor_name}_corr_{suffix}"] = correlation
    return output


def _ppg_window_features(
    values: np.ndarray,
    mask: np.ndarray,
    sampling_hz: float,
    prefix: str = "local",
) -> dict[str, float]:
    output: dict[str, float] = {}
    valid = values[mask]
    _add_statistics(output, f"{prefix}_ppg", valid)
    for name, value in spectral_summary(valid, sampling_hz).items():
        output[f"{prefix}_ppg_{name}"] = value
    quality_features, quality = ppg_quality_features(values, mask, sampling_hz)
    quality_names = (
        "valid_fraction",
        "maximum_gap_ratio",
        "clipping_ratio",
        "derivative_outlier_ratio",
        "pulse_band_concentration",
        "autocorrelation_peak",
        "flatline_ratio",
        "robust_snr",
    )
    for name, value in zip(quality_names, quality_features):
        output[f"{prefix}_ppg_{name}"] = float(value)
    output[f"{prefix}_ppg_sqi"] = quality
    return output


def _compact_motion_bucket_features(
    motion_values: np.ndarray,
    motion_mask: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    output: dict[str, float] = {}
    for start, name in ((0, "acc"), (3, "gyro")):
        valid = motion_mask[:, start : start + 3].all(axis=1)
        magnitude = np.linalg.norm(motion_values[valid, start : start + 3], axis=1)
        stats = robust_statistics(magnitude)
        for statistic in ("mean", "std", "rms", "maximum", "slope"):
            output[f"{prefix}_{name}_{statistic}"] = stats[statistic]
        output[f"{prefix}_{name}_valid_fraction"] = float(valid.mean()) if len(valid) else 0.0
    return output


def _compact_ppg_bucket_features(
    ppg_values: np.ndarray,
    ppg_mask: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    output: dict[str, float] = {}
    ppg_valid = ppg_values[ppg_mask]
    stats = robust_statistics(ppg_valid)
    for statistic in ("mean", "std", "rms", "range", "slope"):
        output[f"{prefix}_ppg_{statistic}"] = stats[statistic]
    _, quality = ppg_quality_features(ppg_values, ppg_mask, 50.0)
    output[f"{prefix}_ppg_sqi"] = quality
    return output


def build_segment_features(
    segment_path: str | Path,
    anchors: pd.DataFrame,
    window_seconds: int,
    include_dyadic: bool,
    motion_bucket_seconds: list[int],
    ppg_bucket_seconds: list[int],
) -> pd.DataFrame:
    with np.load(segment_path) as payload:
        motion_time = payload["motion_timestamp_ms"]
        motion_values = payload["motion_values"]
        motion_mask = payload["motion_mask"].astype(bool)
        ppg_time = payload["ppg_timestamp_ms"]
        ppg_values = payload["ppg_values"].reshape(-1)
        ppg_mask = payload["ppg_mask"].astype(bool).reshape(-1)
    rows: list[dict[str, object]] = []
    for anchor in anchors.itertuples(index=False):
        end_ms = int(anchor.timestamp_ms)
        start_ms = end_ms - int(window_seconds * 1000)
        motion_slice = _slice(motion_time, start_ms, end_ms)
        ppg_slice = _slice(ppg_time, start_ms, end_ms)
        features = _motion_window_features(
            motion_values[motion_slice], motion_mask[motion_slice], 100.0
        )
        features.update(
            _ppg_window_features(ppg_values[ppg_slice], ppg_mask[ppg_slice], 50.0)
        )
        if include_dyadic:
            motion_cursor_ms = end_ms
            for bucket_index, width_seconds in enumerate(motion_bucket_seconds):
                bucket_start_ms = motion_cursor_ms - int(width_seconds * 1000)
                motion_bucket = _slice(motion_time, bucket_start_ms, motion_cursor_ms)
                features.update(
                    _compact_motion_bucket_features(
                        motion_values[motion_bucket],
                        motion_mask[motion_bucket],
                        f"motion_bucket_{bucket_index}",
                    )
                )
                motion_cursor_ms = bucket_start_ms
            ppg_cursor_ms = end_ms
            for bucket_index, width_seconds in enumerate(ppg_bucket_seconds):
                bucket_start_ms = ppg_cursor_ms - int(width_seconds * 1000)
                ppg_bucket = _slice(ppg_time, bucket_start_ms, ppg_cursor_ms)
                features.update(
                    _compact_ppg_bucket_features(
                        ppg_values[ppg_bucket],
                        ppg_mask[ppg_bucket],
                        f"ppg_bucket_{bucket_index}",
                    )
                )
                ppg_cursor_ms = bucket_start_ms
        features.update(
            {
                "segment_id": anchor.segment_id,
                "subject_key": anchor.subject_key,
                "timestamp_ms": end_ms,
                "state_target": float(anchor.state_target),
                "start_target": float(anchor.start_target),
                "end_target": float(anchor.end_target),
                "start_loss_mask": float(anchor.start_loss_mask),
                "end_loss_mask": float(anchor.end_loss_mask),
                "distance_to_event_seconds": float(anchor.distance_to_event_seconds),
                "hand_relation": anchor.hand_relation,
            }
        )
        rows.append(features)
    return pd.DataFrame(rows)
