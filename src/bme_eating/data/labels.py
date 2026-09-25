from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from bme_eating.data.stats_fusion_preprocess import session_right_endpoint_grid


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _acc_observed_intervals(segments: pd.DataFrame) -> dict[str, list[tuple[int, int]]]:
    by_subject: dict[str, list[tuple[int, int]]] = {}
    for segment in segments.itertuples(index=False):
        intervals: list[tuple[int, int]] = []
        path = Path(str(getattr(segment, "segment_path", "")))
        if path.is_file():
            with np.load(path) as payload:
                timestamps = payload["motion_timestamp_ms"].astype(np.int64)
                mask = payload["motion_mask"].astype(bool)
            valid = mask[:, :3].all(axis=1) if mask.ndim == 2 and mask.shape[1] >= 3 else np.zeros(len(timestamps), dtype=bool)
            valid_times = timestamps[valid]
            if len(valid_times):
                period = round(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 10
                period = max(period, 1)
                split = np.flatnonzero(np.diff(valid_times) > period * 1.5) + 1
                boundaries = np.concatenate(([0], split, [len(valid_times)]))
                intervals.extend(
                    (int(valid_times[start]), int(valid_times[end - 1]) + period)
                    for start, end in pairwise(boundaries)
                    if end > start
                )
        else:
            intervals.append((int(segment.start_ms), int(segment.end_ms)))
        by_subject.setdefault(str(segment.subject_key), []).extend(intervals)
    return {key: _merge_intervals(value) for key, value in by_subject.items()}


def _point_is_observed(point_ms: int, intervals: list[tuple[int, int]], tolerance_ms: int) -> bool:
    return any(start - tolerance_ms <= point_ms <= end + tolerance_ms for start, end in intervals)


def _maximum_gap_ms(
    event_start: int,
    event_end: int,
    intervals: list[tuple[int, int]],
) -> int:
    clipped = _merge_intervals(
        [
            (max(event_start, start), min(event_end, end))
            for start, end in intervals
            if end > event_start and start < event_end
        ]
    )
    cursor = event_start
    maximum = 0
    for start, end in clipped:
        maximum = max(maximum, start - cursor)
        cursor = max(cursor, end)
    return max(maximum, event_end - cursor)


def _overlap_fraction(
    interval_starts: np.ndarray,
    interval_ends: np.ndarray,
    event_start: float,
    event_end: float,
) -> np.ndarray:
    overlap = np.maximum(
        0.0, np.minimum(interval_ends, event_end) - np.maximum(interval_starts, event_start)
    )
    return overlap / np.maximum(interval_ends - interval_starts, 1.0)


def _minimum_event_distance(timestamp_ms: np.ndarray, events: pd.DataFrame) -> np.ndarray:
    if events.empty:
        return np.full(len(timestamp_ms), np.inf, dtype=np.float32)
    distance = np.full(len(timestamp_ms), np.inf, dtype=np.float64)
    for event in events.itertuples(index=False):
        before = np.maximum(float(event.start_ms) - timestamp_ms, 0.0)
        after = np.maximum(timestamp_ms - float(event.end_ms), 0.0)
        distance = np.minimum(distance, before + after)
    return (distance / 1000.0).astype(np.float32)


def build_anchor_index(
    segments: pd.DataFrame,
    events: pd.DataFrame,
    output_step_seconds: int,
    output_path: Path,
) -> pd.DataFrame:
    anchor_columns = [
        "segment_id",
        "session_id",
        "segment_path",
        "subject_key",
        "timestamp_ms",
        "state_target",
        "state_loss_mask",
        "censor_mask",
        "start_target",
        "end_target",
        "start_loss_mask",
        "end_loss_mask",
        "distance_to_event_seconds",
        "hand_relation",
        "event_id",
        "motion_history_available_seconds",
        "ppg_history_available_seconds",
    ]
    if output_step_seconds <= 0:
        raise ValueError("output_step_seconds must be positive")
    event_columns = [
        "event_id",
        "subject_key",
        "start_ms",
        "end_ms",
        "hand_relation",
        "valid_duration",
    ]
    if events.empty:
        events = pd.DataFrame(columns=event_columns)
    else:
        missing = set(event_columns) - set(events.columns)
        if missing:
            raise ValueError(f"Events are missing columns: {sorted(missing)}")
    step_ms = int(output_step_seconds * 1000)
    sigma_ms = 6000.0
    rows: list[pd.DataFrame] = []
    valid_events = events[events["valid_duration"]].copy()
    ordered_segments = segments.sort_values(
        ["subject_key", "session_id", "start_ms", "end_ms", "segment_id"]
    )
    accumulated_history: dict[str, tuple[float, float]] = {}
    for segment in ordered_segments.itertuples(index=False):
        anchor_time = np.arange(
            int(segment.start_ms) + step_ms,
            int(segment.end_ms) + 1,
            step_ms,
            dtype=np.int64,
        )
        if len(anchor_time) == 0:
            continue
        interval_start = anchor_time - step_ms
        all_subject_events = valid_events[
            valid_events["subject_key"] == segment.subject_key
        ]
        subject_events = all_subject_events[
            (all_subject_events["end_ms"] > segment.start_ms)
            & (all_subject_events["start_ms"] < segment.end_ms)
        ]
        observable = np.ones(len(anchor_time), dtype=np.float32)
        session_id = str(getattr(segment, "session_id", segment.segment_id))
        previous_motion_seconds, previous_ppg_seconds = accumulated_history.get(
            session_id, (0.0, 0.0)
        )
        motion_history = np.full(
            len(anchor_time), previous_motion_seconds, dtype=np.float32
        )
        ppg_history = np.full(
            len(anchor_time), previous_ppg_seconds, dtype=np.float32
        )
        segment_motion_seconds = max(0.0, (int(segment.end_ms) - int(segment.start_ms)) / 1000.0)
        segment_ppg_seconds = segment_motion_seconds
        segment_path = Path(str(segment.segment_path))
        if segment_path.is_file():
            with np.load(segment_path) as payload:
                motion_time = payload["motion_timestamp_ms"].astype(np.int64)
                motion_mask = payload["motion_mask"].astype(bool)
                ppg_time = payload["ppg_timestamp_ms"].astype(np.int64)
                ppg_mask = payload["ppg_mask"].astype(bool)
            acc_valid = motion_mask[:, :3].all(axis=1)
            positions = np.searchsorted(motion_time, anchor_time, side="left")
            positions = np.clip(positions, 0, max(len(motion_time) - 1, 0))
            observable = acc_valid[positions].astype(np.float32) if len(motion_time) else np.zeros(len(anchor_time), dtype=np.float32)
            motion_valid = motion_mask.any(axis=1)
            ppg_valid = ppg_mask.any(axis=1)
            motion_period_seconds = (
                float(np.median(np.diff(motion_time))) / 1000.0
                if len(motion_time) > 1
                else 0.01
            )
            ppg_period_seconds = (
                float(np.median(np.diff(ppg_time))) / 1000.0
                if len(ppg_time) > 1
                else 0.02
            )
            motion_prefix = np.concatenate(([0], np.cumsum(motion_valid, dtype=np.int64)))
            ppg_prefix = np.concatenate(([0], np.cumsum(ppg_valid, dtype=np.int64)))
            motion_positions = np.searchsorted(motion_time, anchor_time, side="right")
            ppg_positions = np.searchsorted(ppg_time, anchor_time, side="right")
            motion_history += (
                motion_prefix[motion_positions] * max(motion_period_seconds, 0.0)
            ).astype(np.float32)
            ppg_history += (
                ppg_prefix[ppg_positions] * max(ppg_period_seconds, 0.0)
            ).astype(np.float32)
            segment_motion_seconds = float(motion_valid.sum()) * max(
                motion_period_seconds, 0.0
            )
            segment_ppg_seconds = float(ppg_valid.sum()) * max(ppg_period_seconds, 0.0)
        state = np.zeros(len(anchor_time), dtype=np.float32)
        start_target = np.zeros(len(anchor_time), dtype=np.float32)
        end_target = np.zeros(len(anchor_time), dtype=np.float32)
        start_loss_mask = observable.copy()
        end_loss_mask = observable.copy()
        hand_relation = np.full(len(anchor_time), "background", dtype=object)
        event_id = np.full(len(anchor_time), "", dtype=object)
        for event in subject_events.itertuples(index=False):
            state = np.maximum(
                state,
                _overlap_fraction(
                    interval_start,
                    anchor_time,
                    float(event.start_ms),
                    float(event.end_ms),
                ).astype(np.float32),
            )
            start_distance = np.abs(anchor_time - float(event.start_ms))
            end_distance = np.abs(anchor_time - float(event.end_ms))
            start_gaussian = np.exp(-0.5 * (start_distance / sigma_ms) ** 2)
            end_gaussian = np.exp(-0.5 * (end_distance / sigma_ms) ** 2)
            start_gaussian[start_distance > 15_000] = 0.0
            end_gaussian[end_distance > 15_000] = 0.0
            start_target = np.maximum(start_target, start_gaussian.astype(np.float32))
            end_target = np.maximum(end_target, end_gaussian.astype(np.float32))
            inside = (anchor_time > event.start_ms) & (interval_start < event.end_ms)
            hand_relation[inside] = event.hand_relation
            event_id[inside] = str(event.event_id)
            if not bool(getattr(event, "start_observed", True)):
                start_loss_mask[start_distance <= 15_000] = 0.0
            if not bool(getattr(event, "end_observed", True)):
                end_loss_mask[end_distance <= 15_000] = 0.0
        rows.append(
            pd.DataFrame(
                {
                    "segment_id": segment.segment_id,
                    "session_id": session_id,
                    "segment_path": str(segment.segment_path),
                    "subject_key": segment.subject_key,
                    "timestamp_ms": anchor_time,
                    "state_target": state,
                    "state_loss_mask": observable,
                    "censor_mask": 1.0 - observable,
                    "start_target": start_target,
                    "end_target": end_target,
                    "start_loss_mask": start_loss_mask,
                    "end_loss_mask": end_loss_mask,
                    "distance_to_event_seconds": _minimum_event_distance(
                        anchor_time, all_subject_events
                    ),
                    "hand_relation": hand_relation,
                    "event_id": event_id,
                    "motion_history_available_seconds": motion_history,
                    "ppg_history_available_seconds": ppg_history,
                }
            )
        )
        accumulated_history[session_id] = (
            previous_motion_seconds + segment_motion_seconds,
            previous_ppg_seconds + segment_ppg_seconds,
        )
    anchors = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=anchor_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anchors.to_parquet(output_path, index=False)
    return anchors


def build_statsfusion_session_anchor_index(
    segments: pd.DataFrame,
    events: pd.DataFrame,
    output_step_seconds: int = 3,
    output_path: Path | None = None,
) -> pd.DataFrame:
    required_segments = {
        "segment_id",
        "session_id",
        "segment_path",
        "subject_key",
        "start_ms",
        "end_ms",
    }
    missing_segments = required_segments - set(segments.columns)
    if missing_segments:
        raise ValueError(f"StatsFusion segments are missing columns: {sorted(missing_segments)}")
    required_events = {"event_id", "subject_key", "start_ms", "end_ms", "valid_duration"}
    missing_events = required_events - set(events.columns)
    if len(events) and missing_events:
        raise ValueError(f"StatsFusion events are missing columns: {sorted(missing_events)}")
    if output_step_seconds <= 0:
        raise ValueError("StatsFusion output step must be positive")
    step_ms = int(output_step_seconds * 1000)
    valid_events = events[events["valid_duration"].astype(bool)].copy()
    rows: list[pd.DataFrame] = []
    ordered = segments.sort_values(
        ["subject_key", "session_id", "start_ms", "end_ms", "segment_id"], kind="stable"
    )
    for (subject_key, session_id), session_segments in ordered.groupby(
        ["subject_key", "session_id"], sort=False
    ):
        session_segments = session_segments.sort_values(
            ["start_ms", "end_ms", "segment_id"], kind="stable"
        )
        observed_starts: list[int] = []
        observed_ends: list[int] = []
        observed_bounds: dict[str, tuple[int, int]] = {}
        for segment in session_segments.itertuples(index=False):
            path = Path(str(segment.segment_path))
            if not path.is_file():
                continue
            segment_starts: list[int] = []
            segment_ends: list[int] = []
            with np.load(path) as payload:
                for name in ("motion_timestamp_ms", "ppg_timestamp_ms"):
                    timestamps = payload[name]
                    if len(timestamps):
                        segment_starts.append(int(timestamps[0]))
                        segment_ends.append(int(timestamps[-1]))
            if segment_starts:
                segment_start = min(segment_starts)
                segment_end = max(segment_ends)
                observed_starts.append(segment_start)
                observed_ends.append(segment_end)
                observed_bounds[str(segment.segment_id)] = (segment_start, segment_end)
        first_ms = (
            min(observed_starts)
            if observed_starts
            else int(session_segments["start_ms"].min())
        )
        last_ms = (
            max(observed_ends)
            if observed_ends
            else int(session_segments["end_ms"].max())
        )
        anchor_time = session_right_endpoint_grid(first_ms, last_ms, step_ms)
        if not len(anchor_time):
            continue
        interval_start = anchor_time - step_ms
        observable = np.zeros(len(anchor_time), dtype=np.float32)
        segment_ids = np.full(len(anchor_time), "", dtype=object)
        segment_paths = np.full(len(anchor_time), "", dtype=object)
        motion_history = np.zeros(len(anchor_time), dtype=np.float32)
        ppg_history = np.zeros(len(anchor_time), dtype=np.float32)
        accumulated_motion = 0.0
        accumulated_ppg = 0.0
        for segment in session_segments.itertuples(index=False):
            segment_start, segment_end = observed_bounds.get(
                str(segment.segment_id), (int(segment.start_ms), int(segment.end_ms))
            )
            in_segment = (anchor_time > segment_start) & (anchor_time <= segment_end)
            after_segment = anchor_time > segment_end
            segment_ids[in_segment] = str(segment.segment_id)
            segment_paths[in_segment] = str(segment.segment_path)
            path = Path(str(segment.segment_path))
            if path.is_file():
                with np.load(path) as payload:
                    motion_time = payload["motion_timestamp_ms"].astype(np.int64)
                    motion_mask = payload["motion_mask"].astype(bool)
                    ppg_time = payload["ppg_timestamp_ms"].astype(np.int64)
                    ppg_mask = payload["ppg_mask"].astype(bool).reshape(-1)
                motion_valid = motion_mask.any(axis=1)
                ppg_valid = ppg_mask.astype(bool)
                motion_period = (
                    max(float(np.median(np.diff(motion_time))) / 1000.0, 0.0)
                    if len(motion_time) > 1
                    else 0.01
                )
                ppg_period = (
                    max(float(np.median(np.diff(ppg_time))) / 1000.0, 0.0)
                    if len(ppg_time) > 1
                    else 0.02
                )
                motion_prefix = np.concatenate(([0], np.cumsum(motion_valid, dtype=np.int64)))
                ppg_prefix = np.concatenate(([0], np.cumsum(ppg_valid, dtype=np.int64)))
                selected_times = anchor_time[in_segment]
                if len(selected_times):
                    motion_positions = np.searchsorted(motion_time, selected_times, side="right")
                    ppg_positions = np.searchsorted(ppg_time, selected_times, side="right")
                    motion_history[in_segment] = accumulated_motion + (
                        motion_prefix[motion_positions] * motion_period
                    ).astype(np.float32)
                    ppg_history[in_segment] = accumulated_ppg + (
                        ppg_prefix[ppg_positions] * ppg_period
                    ).astype(np.float32)
                    nearest = np.searchsorted(motion_time, selected_times, side="left")
                    nearest = np.clip(nearest, 0, max(len(motion_time) - 1, 0))
                    observable[in_segment] = (
                        motion_mask[nearest, :3].all(axis=1).astype(np.float32)
                        if len(motion_time)
                        else 0.0
                    )
                segment_motion = float(motion_valid.sum()) * motion_period
                segment_ppg = float(ppg_valid.sum()) * ppg_period
            else:
                segment_motion = max(0.0, (int(segment.end_ms) - int(segment.start_ms)) / 1000.0)
                segment_ppg = segment_motion
                observable[in_segment] = 1.0
                motion_history[in_segment] = accumulated_motion + (
                    (anchor_time[in_segment] - segment_start) / 1000.0
                ).astype(np.float32)
                ppg_history[in_segment] = accumulated_ppg + (
                    (anchor_time[in_segment] - segment_start) / 1000.0
                ).astype(np.float32)
            motion_history[after_segment] = accumulated_motion + segment_motion
            ppg_history[after_segment] = accumulated_ppg + segment_ppg
            accumulated_motion += segment_motion
            accumulated_ppg += segment_ppg

        subject_events = valid_events[
            valid_events["subject_key"].astype(str).eq(str(subject_key))
            & (valid_events["end_ms"].astype(np.int64) > first_ms)
            & (valid_events["start_ms"].astype(np.int64) < last_ms)
        ]
        state = np.zeros(len(anchor_time), dtype=np.float32)
        onset = np.zeros(len(anchor_time), dtype=np.float32)
        offset = np.zeros(len(anchor_time), dtype=np.float32)
        start_loss_mask = observable.copy()
        end_loss_mask = observable.copy()
        hand_relation = np.full(len(anchor_time), "background", dtype=object)
        event_id = np.full(len(anchor_time), "", dtype=object)
        for event in subject_events.itertuples(index=False):
            state = np.maximum(
                state,
                _overlap_fraction(
                    interval_start,
                    anchor_time,
                    float(event.start_ms),
                    float(event.end_ms),
                ).astype(np.float32),
            )
            start_delta = anchor_time - int(event.start_ms)
            end_delta = anchor_time - int(event.end_ms)
            onset = np.maximum(
                onset,
                np.where(
                    (start_delta >= 0) & (start_delta <= 30_000),
                    1.0 - start_delta / 30_000.0,
                    0.0,
                ).astype(np.float32),
            )
            offset = np.maximum(
                offset,
                np.where(
                    (end_delta >= 0) & (end_delta <= 60_000),
                    1.0 - end_delta / 60_000.0,
                    0.0,
                ).astype(np.float32),
            )
            inside = (anchor_time > int(event.start_ms)) & (
                interval_start < int(event.end_ms)
            )
            hand_relation[inside] = str(getattr(event, "hand_relation", "unknown"))
            event_id[inside] = str(event.event_id)
            if not bool(getattr(event, "start_observed", True)):
                start_loss_mask[(start_delta >= 0) & (start_delta <= 30_000)] = 0.0
            if not bool(getattr(event, "end_observed", True)):
                end_loss_mask[(end_delta >= 0) & (end_delta <= 60_000)] = 0.0
        frame = pd.DataFrame(
            {
                "segment_id": segment_ids,
                "session_id": str(session_id),
                "segment_path": segment_paths,
                "subject_key": str(subject_key),
                "timestamp_ms": anchor_time,
                "state_target": state,
                "state_loss_mask": observable,
                "censor_mask": 1.0 - observable,
                "start_target": onset,
                "end_target": offset,
                "start_loss_mask": start_loss_mask,
                "end_loss_mask": end_loss_mask,
                "distance_to_event_seconds": _minimum_event_distance(
                    anchor_time, subject_events
                ),
                "hand_relation": hand_relation,
                "event_id": event_id,
                "motion_history_available_seconds": motion_history,
                "ppg_history_available_seconds": ppg_history,
            }
        )
        rows.append(frame)
    anchors = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(anchors) and anchors.duplicated(["subject_key", "session_id", "timestamp_ms"]).any():
        raise RuntimeError("StatsFusion canonical anchors contain duplicate session timestamps")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        anchors.to_parquet(temporary, index=False)
        temporary.replace(output_path)
    return anchors


def classify_event_coverage(
    events: pd.DataFrame,
    segments: pd.DataFrame,
    output_step_seconds: int = 3,
) -> pd.DataFrame:
    if output_step_seconds <= 0:
        raise ValueError("output_step_seconds must be positive")
    classified = events.copy()
    if classified.empty:
        for column, dtype in (
            ("coverage", "object"),
            ("coverage_ratio", "float64"),
            ("start_observed", "bool"),
            ("end_observed", "bool"),
            ("max_gap_seconds", "float64"),
            ("evaluable", "bool"),
        ):
            classified[column] = pd.Series(dtype=dtype)
        return classified
    if "valid_duration" not in classified.columns:
        classified["valid_duration"] = False
    observed_by_subject = _acc_observed_intervals(segments)
    coverage: list[str] = []
    ratios: list[float] = []
    starts_observed: list[bool] = []
    ends_observed: list[bool] = []
    maximum_gaps: list[float] = []
    evaluable: list[bool] = []
    tolerance_ms = int(output_step_seconds * 1000)
    for event in classified.itertuples(index=False):
        if not event.valid_duration:
            coverage.append("invalid_duration")
            ratios.append(0.0)
            starts_observed.append(False)
            ends_observed.append(False)
            maximum_gaps.append(float("nan"))
            evaluable.append(False)
            continue
        intervals = observed_by_subject.get(str(event.subject_key), [])
        merged = _merge_intervals(
            [
                (max(int(event.start_ms), start), min(int(event.end_ms), end))
                for start, end in intervals
                if end > event.start_ms and start < event.end_ms
            ]
        )
        overlap = sum(end - start for start, end in merged)
        duration = int(event.end_ms) - int(event.start_ms)
        ratio = min(1.0, max(0.0, overlap / duration))
        start_seen = _point_is_observed(int(event.start_ms), intervals, tolerance_ms)
        end_seen = _point_is_observed(int(event.end_ms), intervals, tolerance_ms)
        maximum_gap_ms = _maximum_gap_ms(int(event.start_ms), int(event.end_ms), intervals)
        can_evaluate = start_seen and end_seen and maximum_gap_ms <= tolerance_ms
        if overlap <= 0:
            coverage.append("none")
        elif can_evaluate:
            coverage.append("full")
        else:
            coverage.append("partial")
        ratios.append(ratio)
        starts_observed.append(start_seen)
        ends_observed.append(end_seen)
        maximum_gaps.append(maximum_gap_ms / 1000.0)
        evaluable.append(can_evaluate)
    classified["coverage"] = coverage
    classified["coverage_ratio"] = ratios
    classified["start_observed"] = starts_observed
    classified["end_observed"] = ends_observed
    classified["max_gap_seconds"] = maximum_gaps
    classified["evaluable"] = evaluable
    return classified
