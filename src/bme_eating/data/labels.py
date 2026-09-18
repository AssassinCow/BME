from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


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
    step_ms = int(output_step_seconds * 1000)
    sigma_ms = 6000.0
    rows: list[pd.DataFrame] = []
    valid_events = events[events["valid_duration"]].copy()
    for segment in segments.itertuples(index=False):
        anchor_time = np.arange(
            int(segment.start_ms) + step_ms,
            int(segment.end_ms) + 1,
            step_ms,
            dtype=np.int64,
        )
        if len(anchor_time) == 0:
            continue
        interval_start = anchor_time - step_ms
        subject_events = valid_events[
            (valid_events["subject_key"] == segment.subject_key)
            & (valid_events["end_ms"] > segment.start_ms)
            & (valid_events["start_ms"] < segment.end_ms)
        ]
        state = np.zeros(len(anchor_time), dtype=np.float32)
        start_target = np.zeros(len(anchor_time), dtype=np.float32)
        end_target = np.zeros(len(anchor_time), dtype=np.float32)
        start_loss_mask = np.ones(len(anchor_time), dtype=np.float32)
        end_loss_mask = np.ones(len(anchor_time), dtype=np.float32)
        hand_relation = np.full(len(anchor_time), "background", dtype=object)
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
            if not (segment.start_ms <= event.start_ms <= segment.end_ms):
                start_loss_mask[inside] = 0.0
            if not (segment.start_ms <= event.end_ms <= segment.end_ms):
                end_loss_mask[inside] = 0.0
        rows.append(
            pd.DataFrame(
                {
                    "segment_id": segment.segment_id,
                    "segment_path": segment.segment_path,
                    "subject_key": segment.subject_key,
                    "timestamp_ms": anchor_time,
                    "state_target": state,
                    "start_target": start_target,
                    "end_target": end_target,
                    "start_loss_mask": start_loss_mask,
                    "end_loss_mask": end_loss_mask,
                    "distance_to_event_seconds": _minimum_event_distance(
                        anchor_time, subject_events
                    ),
                    "hand_relation": hand_relation,
                }
            )
        )
    anchors = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anchors.to_parquet(output_path, index=False)
    return anchors


def classify_event_coverage(events: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    classified = events.copy()
    coverage: list[str] = []
    for event in classified.itertuples(index=False):
        if not event.valid_duration:
            coverage.append("invalid_duration")
            continue
        subject_segments = segments[segments["subject_key"] == event.subject_key]
        intervals = []
        for segment in subject_segments.itertuples(index=False):
            start = max(int(segment.start_ms), int(event.start_ms))
            end = min(int(segment.end_ms), int(event.end_ms))
            if end > start:
                intervals.append((start, end))
        intervals.sort()
        merged: list[list[int]] = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        overlap = sum(end - start for start, end in merged)
        duration = event.end_ms - event.start_ms
        if overlap <= 0:
            coverage.append("none")
        elif math.isclose(float(overlap), float(duration), rel_tol=0.0, abs_tol=1.0):
            coverage.append("full")
        else:
            coverage.append("partial")
    classified["coverage"] = coverage
    return classified
