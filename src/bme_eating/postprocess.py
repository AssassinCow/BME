from __future__ import annotations

import math
from itertools import product

import numpy as np
import pandas as pd

from bme_eating.metrics import evaluate_events


def causal_ema(values: np.ndarray, step_seconds: float, half_life_seconds: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    alpha = 1.0 - math.exp(-math.log(2.0) * step_seconds / max(half_life_seconds, 1e-6))
    output = np.empty_like(values)
    output[0] = values[0]
    for index in range(1, len(values)):
        output[index] = alpha * values[index] + (1.0 - alpha) * output[index - 1]
    return output


def probabilities_to_events(
    predictions: pd.DataFrame,
    ema_half_life_seconds: float,
    high_threshold: float,
    low_threshold: float,
    minimum_event_seconds: float,
    merge_gap_seconds: float,
    boundary_lookback_seconds: float,
) -> pd.DataFrame:
    required = {
        "subject_key",
        "segment_id",
        "timestamp_ms",
        "state_probability",
        "start_probability",
        "end_probability",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    events: list[dict[str, object]] = []
    for (subject_key, segment_id), group in predictions.groupby(
        ["subject_key", "segment_id"], sort=False
    ):
        group = group.sort_values("timestamp_ms").reset_index(drop=True)
        timestamps = group["timestamp_ms"].to_numpy(dtype=np.int64)
        if len(timestamps) == 0:
            continue
        step_seconds = float(np.median(np.diff(timestamps)) / 1000.0) if len(timestamps) > 1 else 3.0
        smoothed = causal_ema(
            group["state_probability"].to_numpy(), step_seconds, ema_half_life_seconds
        )
        start_prob = group["start_probability"].to_numpy()
        end_prob = group["end_probability"].to_numpy()
        lookback_rows = max(1, int(round(boundary_lookback_seconds / step_seconds)))
        active = False
        start_index = 0
        candidates: list[dict[str, object]] = []
        for index, probability in enumerate(smoothed):
            if not active and probability >= high_threshold:
                search_start = max(0, index - lookback_rows)
                local = int(np.argmax(start_prob[search_start : index + 1]))
                start_index = search_start + local
                active = True
            elif active and probability < low_threshold:
                search_start = max(start_index, index - lookback_rows)
                local = int(np.argmax(end_prob[search_start : index + 1]))
                end_index = search_start + local
                if end_index <= start_index:
                    end_index = index
                candidates.append(
                    {
                        "subject_key": subject_key,
                        "segment_id": segment_id,
                        "start_ms": int(timestamps[start_index]),
                        "end_ms": int(timestamps[end_index]),
                        "score": float(np.mean(smoothed[start_index : end_index + 1])),
                    }
                )
                active = False
        if active:
            candidates.append(
                {
                    "subject_key": subject_key,
                    "segment_id": segment_id,
                    "start_ms": int(timestamps[start_index]),
                    "end_ms": int(timestamps[-1]),
                    "score": float(np.mean(smoothed[start_index:])),
                }
            )
        minimum_ms = int(minimum_event_seconds * 1000)
        candidates = [
            event for event in candidates if event["end_ms"] - event["start_ms"] >= minimum_ms
        ]
        merged: list[dict[str, object]] = []
        maximum_gap_ms = int(merge_gap_seconds * 1000)
        for event in candidates:
            if merged and event["start_ms"] - merged[-1]["end_ms"] <= maximum_gap_ms:
                previous = merged[-1]
                previous["end_ms"] = max(previous["end_ms"], event["end_ms"])
                previous["score"] = max(previous["score"], event["score"])
            else:
                merged.append(event)
        events.extend(merged)
    return pd.DataFrame(events, columns=["subject_key", "segment_id", "start_ms", "end_ms", "score"])


def tune_postprocess_parameters(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    search: dict[str, list[float]],
    iou_threshold: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    names = (
        "ema_half_life_seconds",
        "high_threshold",
        "low_threshold",
        "minimum_event_seconds",
        "merge_gap_seconds",
        "boundary_lookback_seconds",
    )
    rows: list[dict[str, float]] = []
    best_parameters: dict[str, float] | None = None
    best_key = (-1.0, -float("inf"), -float("inf"))
    values = [search[name] for name in names]
    for combination in product(*values):
        parameters = {name: float(value) for name, value in zip(names, combination)}
        if parameters["low_threshold"] >= parameters["high_threshold"]:
            continue
        events = probabilities_to_events(predictions, **parameters)
        metrics, _ = evaluate_events(
            truth, events, iou_threshold=iou_threshold, method="hungarian"
        )
        boundary_values = [
            value
            for value in (metrics["start_mae_seconds"], metrics["end_mae_seconds"])
            if np.isfinite(value)
        ]
        boundary_mae = float(np.mean(boundary_values)) if boundary_values else float("inf")
        key = (metrics["f1"], -boundary_mae, metrics["precision"])
        rows.append({**parameters, **metrics, "boundary_mae_seconds": boundary_mae})
        if key > best_key:
            best_key = key
            best_parameters = parameters
    if best_parameters is None:
        raise RuntimeError("Postprocessing search produced no valid parameter combination")
    return best_parameters, pd.DataFrame(rows).sort_values(
        ["f1", "boundary_mae_seconds", "precision"],
        ascending=[False, True, False],
    )
