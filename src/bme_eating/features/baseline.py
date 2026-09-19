from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bme_eating.data.deep_dataset import _load_segment_archive
from bme_eating.data.session import SessionWindowReader
from bme_eating.features.signal import (
    longest_false_run,
    masked_spectral_summary,
    ppg_quality_features,
    robust_statistics,
)


def _slice(timestamp_ms: np.ndarray, start_ms: int, end_ms: int) -> slice:
    return slice(
        int(np.searchsorted(timestamp_ms, start_ms, side="left")),
        int(np.searchsorted(timestamp_ms, end_ms, side="right")),
    )


def _history_slice(timestamp_ms: np.ndarray, start_ms: int, end_ms: int) -> slice:
    """Return a right-closed history interval so adjacent buckets do not overlap."""
    return slice(
        int(np.searchsorted(timestamp_ms, start_ms, side="right")),
        int(np.searchsorted(timestamp_ms, end_ms, side="right")),
    )


def _masked_slope(values: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1) & np.isfinite(values)
    if mask.sum() < 2:
        return 0.0
    positions = np.linspace(-1.0, 1.0, len(values), dtype=np.float64)[mask]
    selected = values[mask]
    centered_positions = positions - positions.mean()
    centered_values = selected - selected.mean()
    denominator = float(np.dot(centered_positions, centered_positions))
    return float(np.dot(centered_positions, centered_values) / max(denominator, 1e-12))


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
        magnitude_all = np.linalg.norm(values[:, start : start + 3], axis=1)
        magnitude = magnitude_all[valid_rows]
        _add_statistics(output, f"{prefix}_{sensor_name}", magnitude)
        for name, value in masked_spectral_summary(
            magnitude_all, valid_rows, sampling_hz
        ).items():
            output[f"{prefix}_{sensor_name}_{name}"] = value
        adjacent_valid = valid_rows[:-1] & valid_rows[1:]
        if adjacent_valid.any():
            jerk = np.diff(magnitude_all)[adjacent_valid] * sampling_hz
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
    for name, value in masked_spectral_summary(values, mask, sampling_hz).items():
        output[f"{prefix}_ppg_{name}"] = value
    zero_mask = mask & (values == 0)
    output[f"{prefix}_ppg_zero_fraction"] = float(zero_mask.sum() / max(mask.sum(), 1))
    output[f"{prefix}_ppg_longest_zero_run_ratio"] = float(
        longest_false_run(~zero_mask) / max(len(zero_mask), 1)
    )
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
        magnitude = np.linalg.norm(motion_values[:, start : start + 3], axis=1)
        stats = robust_statistics(magnitude[valid])
        for statistic in ("mean", "std", "rms", "maximum", "slope"):
            output[f"{prefix}_{name}_{statistic}"] = stats[statistic]
        output[f"{prefix}_{name}_slope"] = _masked_slope(magnitude, valid)
        output[f"{prefix}_{name}_last"] = float(magnitude[np.flatnonzero(valid)[-1]]) if valid.any() else 0.0
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
    output[f"{prefix}_ppg_slope"] = _masked_slope(ppg_values, ppg_mask)
    output[f"{prefix}_ppg_last"] = (
        float(ppg_values[np.flatnonzero(ppg_mask)[-1]]) if ppg_mask.any() else 0.0
    )
    output[f"{prefix}_ppg_valid_fraction"] = (
        float(ppg_mask.mean()) if len(ppg_mask) else 0.0
    )
    zero_mask = ppg_mask & (ppg_values == 0)
    output[f"{prefix}_ppg_zero_fraction"] = float(
        zero_mask.sum() / max(ppg_mask.sum(), 1)
    )
    output[f"{prefix}_ppg_longest_zero_run_ratio"] = float(
        longest_false_run(~zero_mask) / max(len(zero_mask), 1)
    )
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
    context_segments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if context_segments is None:
        payload = _load_segment_archive(segment_path)
        first = anchors.iloc[0]
        context_segments = pd.DataFrame(
            [
                {
                    "session_id": str(getattr(first, "session_id", first.segment_id)),
                    "segment_id": str(first.segment_id),
                    "segment_path": str(segment_path),
                    "start_ms": int(payload["motion_timestamp_ms"][0]),
                    "end_ms": int(payload["motion_timestamp_ms"][-1]),
                }
            ]
        )
    reader = SessionWindowReader(context_segments)
    maximum_history_seconds = max(
        int(window_seconds),
        sum(motion_bucket_seconds) if include_dyadic else 0,
        sum(ppg_bucket_seconds) if include_dyadic else 0,
    )
    rows: list[dict[str, object]] = []
    for anchor in anchors.itertuples(index=False):
        end_ms = int(anchor.timestamp_ms)
        session_id = str(getattr(anchor, "session_id", anchor.segment_id))
        payload = reader.read(
            session_id, end_ms - maximum_history_seconds * 1000, end_ms
        )
        motion_time = payload["motion_timestamp_ms"]
        motion_values = payload["motion_values"]
        motion_mask = payload["motion_mask"].astype(bool)
        ppg_time = payload["ppg_timestamp_ms"]
        ppg_values = payload["ppg_values"].reshape(-1)
        ppg_mask = payload["ppg_mask"].astype(bool).reshape(-1)
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
                motion_bucket = _history_slice(
                    motion_time, bucket_start_ms, motion_cursor_ms
                )
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
                ppg_bucket = _history_slice(ppg_time, bucket_start_ms, ppg_cursor_ms)
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
                "session_id": session_id,
                "subject_key": anchor.subject_key,
                "timestamp_ms": end_ms,
                "state_target": float(anchor.state_target),
                "state_loss_mask": float(getattr(anchor, "state_loss_mask", 1.0)),
                "censor_mask": float(getattr(anchor, "censor_mask", 0.0)),
                "start_target": float(anchor.start_target),
                "end_target": float(anchor.end_target),
                "start_loss_mask": float(anchor.start_loss_mask),
                "end_loss_mask": float(anchor.end_loss_mask),
                "distance_to_event_seconds": float(anchor.distance_to_event_seconds),
                "hand_relation": anchor.hand_relation,
                "event_id": str(getattr(anchor, "event_id", "")),
                "motion_history_available_seconds": float(
                    getattr(anchor, "motion_history_available_seconds", 0.0)
                ),
                "ppg_history_available_seconds": float(
                    getattr(anchor, "ppg_history_available_seconds", 0.0)
                ),
            }
        )
        rows.append(features)
    return pd.DataFrame(rows)
