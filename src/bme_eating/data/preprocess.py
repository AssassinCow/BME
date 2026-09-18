from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from bme_eating.data.packet_reader import ParsedAttachment, parse_sensor_zip
from bme_eating.types import SensorSeries


def _positive_median_difference(timestamp_ms: np.ndarray) -> float:
    differences = np.diff(timestamp_ms.astype(np.float64))
    differences = differences[differences > 0]
    return float(np.median(differences)) if len(differences) else 1.0


def _split_ranges(
    timestamp_ms: np.ndarray,
    gap_factor: float,
    minimum_gap_ms: int,
) -> list[tuple[int, int]]:
    if len(timestamp_ms) == 0:
        return []
    median_difference = _positive_median_difference(timestamp_ms)
    gap_threshold = max(gap_factor * median_difference, float(minimum_gap_ms))
    differences = np.diff(timestamp_ms.astype(np.float64))
    split_points = np.flatnonzero((differences <= 0) | (differences > gap_threshold)) + 1
    boundaries = np.concatenate(([0], split_points, [len(timestamp_ms)]))
    return [(int(start), int(end)) for start, end in zip(boundaries[:-1], boundaries[1:])]


def _collapse_duplicate_timestamps(series: SensorSeries) -> SensorSeries:
    if len(series.timestamp_ms) == 0:
        return series
    order = np.argsort(series.timestamp_ms, kind="stable")
    timestamps = series.timestamp_ms[order]
    values = series.values[order]
    unique, inverse, counts = np.unique(timestamps, return_inverse=True, return_counts=True)
    if len(unique) == len(timestamps):
        return SensorSeries(unique, values)
    sums = np.zeros((len(unique), values.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, values)
    return SensorSeries(unique, (sums / counts[:, None]).astype(np.float32))


def _interpolate_with_mask(
    series: SensorSeries,
    target_timestamp_ms: np.ndarray,
    maximum_gap_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    dimensions = series.values.shape[1] if series.values.ndim == 2 else 1
    output = np.zeros((len(target_timestamp_ms), dimensions), dtype=np.float32)
    mask = np.zeros((len(target_timestamp_ms), dimensions), dtype=bool)
    if len(series.timestamp_ms) < 2 or len(target_timestamp_ms) == 0:
        return output, mask
    series = _collapse_duplicate_timestamps(series)
    source_time = series.timestamp_ms.astype(np.float64)
    target_time = target_timestamp_ms.astype(np.float64)
    positions = np.searchsorted(source_time, target_time, side="left")
    left = np.clip(positions - 1, 0, len(source_time) - 1)
    right = np.clip(positions, 0, len(source_time) - 1)
    bracket_gap = source_time[right] - source_time[left]
    nearest_distance = np.minimum(
        np.abs(target_time - source_time[left]), np.abs(source_time[right] - target_time)
    )
    valid = (
        (target_time >= source_time[0])
        & (target_time <= source_time[-1])
        & (bracket_gap <= maximum_gap_ms)
        & (nearest_distance <= maximum_gap_ms)
    )
    for dimension in range(dimensions):
        output[:, dimension] = np.interp(
            target_time,
            source_time,
            series.values[:, dimension],
            left=0.0,
            right=0.0,
        ).astype(np.float32)
    mask[valid, :] = True
    output[~mask] = 0.0
    return output, mask


def _record_token(zip_path: Path) -> str:
    return hashlib.sha256(str(zip_path).encode("utf-8")).hexdigest()[:16]


def _save_segment(
    output_path: Path,
    motion_timestamp_ms: np.ndarray,
    motion_values: np.ndarray,
    motion_mask: np.ndarray,
    ppg_timestamp_ms: np.ndarray,
    ppg_values: np.ndarray,
    ppg_mask: np.ndarray,
    compressed: bool,
) -> None:
    writer = np.savez_compressed if compressed else np.savez
    writer(
        output_path,
        motion_timestamp_ms=motion_timestamp_ms,
        motion_values=motion_values.astype(np.float32),
        motion_mask=motion_mask.astype(np.uint8),
        ppg_timestamp_ms=ppg_timestamp_ms,
        ppg_values=ppg_values.astype(np.float32),
        ppg_mask=ppg_mask.astype(np.uint8),
    )


def preprocess_attachment(
    record: dict[str, object],
    output_dir: Path,
    data_config: dict[str, object],
    compressed: bool = True,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    zip_path = Path(str(record["zip_path"]))
    parsed: ParsedAttachment = parse_sensor_zip(
        zip_path,
        ppg_samples_per_row=int(data_config["ppg_samples_per_row"]),
        timestamp_anchor=str(data_config["packet_timestamp_anchor"]),
    )
    ranges = _split_ranges(
        parsed.acc.timestamp_ms,
        float(data_config["gap_factor"]),
        int(data_config["minimum_gap_ms"]),
    )
    motion_period_ms = 1000.0 / float(data_config["motion_target_hz"])
    ppg_period_ms = 1000.0 / float(data_config["ppg_target_hz"])
    minimum_duration_ms = int(float(data_config.get("minimum_segment_seconds", 30)) * 1000)
    maximum_gap_ms = int(data_config["maximum_interpolation_gap_ms"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    token = _record_token(zip_path)

    for segment_number, (start_index, end_index) in enumerate(ranges):
        acc = SensorSeries(
            parsed.acc.timestamp_ms[start_index:end_index],
            parsed.acc.values[start_index:end_index],
        )
        if len(acc.timestamp_ms) < 2:
            continue
        start_ms = int(acc.timestamp_ms[0])
        end_ms = int(acc.timestamp_ms[-1])
        if end_ms - start_ms < minimum_duration_ms:
            continue
        motion_time = np.rint(
            np.arange(start_ms, end_ms + motion_period_ms / 2, motion_period_ms)
        ).astype(np.int64)
        acc_values, acc_mask = _interpolate_with_mask(acc, motion_time, maximum_gap_ms)
        gyro_values, gyro_mask = _interpolate_with_mask(
            parsed.gyro, motion_time, maximum_gap_ms
        )
        motion_values = np.concatenate((acc_values, gyro_values), axis=1)
        motion_mask = np.concatenate((acc_mask, gyro_mask), axis=1)

        ppg_time = np.rint(
            np.arange(start_ms, end_ms + ppg_period_ms / 2, ppg_period_ms)
        ).astype(np.int64)
        ppg_values, ppg_mask = _interpolate_with_mask(
            parsed.ppg, ppg_time, maximum_gap_ms
        )
        segment_id = f"{token}_s{segment_number:03d}"
        output_path = output_dir / f"{segment_id}.npz"
        if overwrite or not output_path.exists():
            _save_segment(
                output_path,
                motion_time,
                motion_values,
                motion_mask,
                ppg_time,
                ppg_values,
                ppg_mask,
                compressed,
            )
        rows.append(
            {
                "segment_id": segment_id,
                "subject_key": str(record["subject_key"]),
                "segment_path": str(output_path),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_seconds": (end_ms - start_ms) / 1000.0,
                "motion_valid_fraction": float(motion_mask.mean()),
                "ppg_valid_fraction": float(ppg_mask.mean()),
                "source_zip_sha256": str(record.get("zip_sha256", "")),
            }
        )
    return rows


def write_preprocess_summary(rows: Iterable[dict[str, object]], output_path: Path) -> None:
    frame = pd.DataFrame(rows)
    frame.to_parquet(output_path, index=False)
    summary = {
        "segments": int(len(frame)),
        "subjects": int(frame["subject_key"].nunique()) if len(frame) else 0,
        "duration_hours": float(frame["duration_seconds"].sum() / 3600) if len(frame) else 0,
        "mean_motion_valid_fraction": float(frame["motion_valid_fraction"].mean())
        if len(frame)
        else 0,
        "mean_ppg_valid_fraction": float(frame["ppg_valid_fraction"].mean())
        if len(frame)
        else 0,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

