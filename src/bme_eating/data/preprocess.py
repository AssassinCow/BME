from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from bme_eating.data.packet_reader import (
    ParsedAttachment,
    parse_multisection_sensor_zip,
    parse_sensor_zip,
)
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_token(zip_path: Path, declared_sha256: object) -> str:
    value = str(declared_sha256 or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        value = _file_sha256(zip_path)
    return value[:20]


def _antialias_series(
    series: SensorSeries,
    target_hz: float,
    maximum_gap_ms: int,
) -> tuple[SensorSeries, float]:
    if len(series.timestamp_ms) < 2:
        return series, 0.0
    median_period_ms = _positive_median_difference(series.timestamp_ms)
    source_hz = 1000.0 / median_period_ms if median_period_ms > 0 else 0.0
    if source_hz <= target_hz * 1.05:
        return series, source_hz
    normalized_cutoff = min(0.99, 0.8 * target_hz / source_hz)
    sos = butter(4, normalized_cutoff, btype="lowpass", output="sos")
    values = series.values.astype(np.float64, copy=True)
    differences = np.diff(series.timestamp_ms.astype(np.float64))
    split_points = np.flatnonzero((differences <= 0) | (differences > maximum_gap_ms)) + 1
    boundaries = np.concatenate(([0], split_points, [len(series.timestamp_ms)]))
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 16:
            continue
        try:
            values[start:end] = sosfiltfilt(sos, values[start:end], axis=0)
        except ValueError:
            continue
    return SensorSeries(series.timestamp_ms, values.astype(np.float32)), source_hz


def assign_virtual_sessions(segments: pd.DataFrame, maximum_gap_ms: int) -> pd.DataFrame:
    if maximum_gap_ms < 0:
        raise ValueError("maximum_gap_ms must be non-negative")
    if segments.empty:
        result = segments.copy()
        for column in (
            "session_id",
            "session_position",
            "previous_segment_id",
            "next_segment_id",
            "gap_from_previous_ms",
            "left_censored",
            "right_censored",
        ):
            result[column] = pd.Series(dtype="object")
        return result
    required = {"segment_id", "subject_key", "start_ms", "end_ms"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(f"Segments are missing columns: {sorted(missing)}")
    ordered = segments.sort_values(
        ["subject_key", "start_ms", "end_ms", "segment_id"]
    ).reset_index(drop=True)
    output: list[pd.DataFrame] = []
    for subject_key, subject_segments in ordered.groupby("subject_key", sort=True):
        rows = subject_segments.reset_index(drop=True).copy()
        groups: list[int] = []
        group_number = -1
        previous_end: int | None = None
        gaps: list[float] = []
        for row in rows.itertuples(index=False):
            gap = float("nan") if previous_end is None else int(row.start_ms) - previous_end
            if previous_end is None or gap < 0 or gap > maximum_gap_ms:
                group_number += 1
                gaps.append(float("nan"))
            else:
                gaps.append(float(gap))
            groups.append(group_number)
            previous_end = int(row.end_ms)
        rows["_session_group"] = groups
        rows["gap_from_previous_ms"] = gaps
        for _, session in rows.groupby("_session_group", sort=False):
            session = session.copy().reset_index(drop=True)
            first_segment = str(session.iloc[0]["segment_id"])
            digest = hashlib.sha256(
                f"session-v2|{subject_key}|{first_segment}".encode("utf-8")
            ).hexdigest()[:20]
            segment_ids = session["segment_id"].astype(str).tolist()
            session["session_id"] = digest
            session["session_position"] = np.arange(len(session), dtype=np.int32)
            session["previous_segment_id"] = [""] + segment_ids[:-1]
            session["next_segment_id"] = segment_ids[1:] + [""]
            source_left = session.get(
                "source_left_censored", pd.Series(False, index=session.index)
            ).astype(bool)
            session["left_censored"] = source_left | (session.index == 0)
            session["right_censored"] = session.index == len(session) - 1
            output.append(session.drop(columns="_session_group"))
    return pd.concat(output, ignore_index=True)


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
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary_path.open("wb") as handle:
            writer(
                handle,
                motion_timestamp_ms=motion_timestamp_ms,
                motion_values=motion_values.astype(np.float32),
                motion_mask=motion_mask.astype(np.uint8),
                ppg_timestamp_ms=ppg_timestamp_ms,
                ppg_values=ppg_values.astype(np.float32),
                ppg_mask=ppg_mask.astype(np.uint8),
            )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def preprocess_attachment(
    record: dict[str, object],
    output_dir: Path,
    data_config: dict[str, object],
    compressed: bool = True,
    overwrite: bool = False,
    parser_mode: str = "standard",
) -> list[dict[str, object]]:
    zip_path = Path(str(record["zip_path"]))
    if parser_mode not in {"standard", "exact_multisection"}:
        raise ValueError(f"unsupported parser mode: {parser_mode}")
    parser = parse_multisection_sensor_zip if parser_mode == "exact_multisection" else parse_sensor_zip
    parsed: ParsedAttachment = parser(
        zip_path,
        ppg_samples_per_row=int(data_config["ppg_samples_per_row"]),
        timestamp_anchor=str(data_config["packet_timestamp_anchor"]),
    )
    parsed = ParsedAttachment(
        acc=_collapse_duplicate_timestamps(parsed.acc),
        gyro=_collapse_duplicate_timestamps(parsed.gyro),
        ppg=_collapse_duplicate_timestamps(parsed.ppg),
        source_name=parsed.source_name,
        info=parsed.info,
        parser_status=parsed.parser_status,
        text_offset_bytes=parsed.text_offset_bytes,
        left_censored=parsed.left_censored,
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
    acc_filtered, acc_source_hz = _antialias_series(
        parsed.acc, float(data_config["motion_target_hz"]), maximum_gap_ms
    )
    gyro_filtered, gyro_source_hz = _antialias_series(
        parsed.gyro, float(data_config["motion_target_hz"]), maximum_gap_ms
    )
    ppg_filtered, ppg_source_hz = _antialias_series(
        parsed.ppg, float(data_config["ppg_target_hz"]), maximum_gap_ms
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    token = _record_token(zip_path, record.get("zip_sha256"))
    if overwrite:
        for stale_path in output_dir.glob(f"{token}_s*.npz"):
            stale_path.unlink(missing_ok=True)
        for stale_path in output_dir.glob(f"{token}_s*.npz.tmp"):
            stale_path.unlink(missing_ok=True)

    for segment_number, (start_index, end_index) in enumerate(ranges):
        acc = SensorSeries(
            acc_filtered.timestamp_ms[start_index:end_index],
            acc_filtered.values[start_index:end_index],
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
            gyro_filtered, motion_time, maximum_gap_ms
        )
        motion_values = np.concatenate((acc_values, gyro_values), axis=1)
        motion_mask = np.concatenate((acc_mask, gyro_mask), axis=1)

        ppg_time = np.rint(
            np.arange(start_ms, end_ms + ppg_period_ms / 2, ppg_period_ms)
        ).astype(np.int64)
        ppg_values, ppg_mask = _interpolate_with_mask(
            ppg_filtered, ppg_time, maximum_gap_ms
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
                "acc_valid_fraction": float(acc_mask.mean()),
                "gyro_valid_fraction": float(gyro_mask.mean()),
                "motion_valid_fraction": float(motion_mask.mean()),
                "ppg_valid_fraction": float(ppg_mask.mean()),
                "ppg_samples_per_row": int(data_config["ppg_samples_per_row"]),
                "ppg_available_columns": int(data_config["ppg_available_columns"]),
                "source_zip_sha256": str(record.get("zip_sha256", "")),
                "parser_status": parsed.parser_status,
                "text_offset_bytes": int(parsed.text_offset_bytes),
                "source_left_censored": bool(parsed.left_censored and segment_number == 0),
                "acc_source_hz": float(acc_source_hz),
                "gyro_source_hz": float(gyro_source_hz),
                "ppg_source_hz": float(ppg_source_hz),
            }
        )
    return rows


def write_preprocess_summary(rows: Iterable[dict[str, object]], output_path: Path) -> None:
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_parquet(output_path, index=False)
    summary = {
        "segments": int(len(frame)),
        "subjects": int(frame["subject_key"].nunique()) if len(frame) else 0,
        "duration_hours": float(frame["duration_seconds"].sum() / 3600) if len(frame) else 0,
        "mean_acc_valid_fraction": float(frame["acc_valid_fraction"].mean())
        if len(frame)
        else 0,
        "mean_gyro_valid_fraction": float(frame["gyro_valid_fraction"].mean())
        if len(frame)
        else 0,
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

