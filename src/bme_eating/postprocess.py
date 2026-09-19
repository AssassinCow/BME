from __future__ import annotations

import hashlib
import json
import math
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from bme_eating.metrics import evaluate_events

_POSTPROCESS_PARAMETER_NAMES = (
    "ema_half_life_seconds",
    "high_threshold",
    "low_threshold",
    "minimum_event_seconds",
    "merge_gap_seconds",
    "boundary_lookback_seconds",
)


def _frame_digest(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.reindex(columns=columns)
    row_hashes = pd.util.hash_pandas_object(selected, index=True).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update(json.dumps(columns, separators=(",", ":")).encode())
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _postprocess_search_signature(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    search: dict[str, list[float]],
    iou_threshold: float,
) -> str:
    prediction_columns = [
        "subject_key",
        "segment_id",
        "timestamp_ms",
        "state_probability",
        "start_probability",
        "end_probability",
    ]
    truth_columns = ["subject_key", "start_ms", "end_ms"]
    payload = {
        "version": 1,
        "search": {name: [float(value) for value in search[name]] for name in _POSTPROCESS_PARAMETER_NAMES},
        "iou_threshold": float(iou_threshold),
        "predictions": _frame_digest(predictions, prediction_columns),
        "truth": _frame_digest(truth, truth_columns),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _parameter_key(parameters: dict[str, Any]) -> tuple[float, ...]:
    return tuple(float(parameters[name]) for name in _POSTPROCESS_PARAMETER_NAMES)


def _load_search_checkpoint(path: Path, signature: str) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            header = json.loads(handle.readline())
            if header != {"version": 1, "signature": signature}:
                return []
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and all(
                    name in row for name in _POSTPROCESS_PARAMETER_NAMES
                ):
                    rows.append(row)
    except (OSError, json.JSONDecodeError, TypeError):
        return []
    return rows


def _initialize_search_checkpoint(path: Path, signature: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps({"version": 1, "signature": signature}) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _append_search_checkpoint(path: Path, row: dict[str, float]) -> None:
    needs_separator = path.exists() and path.stat().st_size > 0
    if needs_separator:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            needs_separator = handle.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as handle:
        prefix = "\n" if needs_separator else ""
        handle.write(
            prefix + json.dumps(row, allow_nan=True, separators=(",", ":")) + "\n"
        )
        handle.flush()


def causal_ema(values: np.ndarray, step_seconds: float, half_life_seconds: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("step_seconds must be positive and finite")
    if not np.isfinite(half_life_seconds) or half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be positive and finite")
    if len(values) == 0:
        return values
    alpha = 1.0 - math.exp(-math.log(2.0) * step_seconds / half_life_seconds)
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
    parameters = {
        "ema_half_life_seconds": ema_half_life_seconds,
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "minimum_event_seconds": minimum_event_seconds,
        "merge_gap_seconds": merge_gap_seconds,
        "boundary_lookback_seconds": boundary_lookback_seconds,
    }
    if not all(np.isfinite(float(value)) for value in parameters.values()):
        raise ValueError("Postprocessing parameters must be finite")
    if ema_half_life_seconds <= 0:
        raise ValueError("ema_half_life_seconds must be positive")
    if minimum_event_seconds < 0 or merge_gap_seconds < 0 or boundary_lookback_seconds < 0:
        raise ValueError("Postprocessing durations must be non-negative")
    if not 0 <= low_threshold < high_threshold <= 1:
        raise ValueError("Expected 0 <= low_threshold < high_threshold <= 1")
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
    if predictions.empty:
        return pd.DataFrame(
            columns=["subject_key", "segment_id", "start_ms", "end_ms", "score"]
        )
    numeric_columns = [
        "timestamp_ms",
        "state_probability",
        "start_probability",
        "end_probability",
    ]
    numeric = predictions[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Prediction timestamps and probabilities must be finite")
    if ((numeric[:, 1:] < 0) | (numeric[:, 1:] > 1)).any():
        raise ValueError("Prediction probabilities must be in [0, 1]")
    events: list[dict[str, object]] = []
    for (subject_key, segment_id), group in predictions.groupby(
        ["subject_key", "segment_id"], sort=False
    ):
        group = (
            group.sort_values("timestamp_ms")
            .groupby("timestamp_ms", as_index=False)[
                ["state_probability", "start_probability", "end_probability"]
            ]
            .mean()
            .reset_index(drop=True)
        )
        timestamps = group["timestamp_ms"].to_numpy(dtype=np.int64)
        if len(timestamps) == 0:
            continue
        step_seconds = float(np.median(np.diff(timestamps)) / 1000.0) if len(timestamps) > 1 else 3.0
        smoothed = causal_ema(
            group["state_probability"].to_numpy(), step_seconds, ema_half_life_seconds
        )
        start_prob = group["start_probability"].to_numpy()
        end_prob = group["end_probability"].to_numpy()
        lookback_rows = max(1, round(boundary_lookback_seconds / step_seconds))
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
    checkpoint_path: Path | None = None,
    show_progress: bool = True,
) -> tuple[dict[str, float], pd.DataFrame]:
    values = [search[name] for name in _POSTPROCESS_PARAMETER_NAMES]
    combinations = [
        tuple(float(value) for value in combination)
        for combination in product(*values)
        if float(combination[2]) < float(combination[1])
    ]
    signature = _postprocess_search_signature(predictions, truth, search, iou_threshold)
    rows = _load_search_checkpoint(checkpoint_path, signature) if checkpoint_path else []
    completed = {_parameter_key(row) for row in rows}
    if checkpoint_path is not None and not completed:
        _initialize_search_checkpoint(checkpoint_path, signature)

    progress = tqdm(
        total=len(combinations),
        initial=len(completed),
        desc="Tuning postprocess",
        unit="combination",
        disable=not show_progress,
    )
    for combination in combinations:
        if combination in completed:
            continue
        parameters = dict(zip(_POSTPROCESS_PARAMETER_NAMES, combination))
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
        row = {
            **parameters,
            **{name: float(value) for name, value in metrics.items()},
            "boundary_mae_seconds": boundary_mae,
        }
        rows.append(row)
        completed.add(combination)
        if checkpoint_path is not None:
            _append_search_checkpoint(checkpoint_path, row)
        progress.update(1)
        progress.set_postfix(f1=f"{metrics['f1']:.4f}")
    progress.close()

    if not rows:
        raise RuntimeError("Postprocessing search produced no valid parameter combination")
    trials = pd.DataFrame(rows).drop_duplicates(
        subset=list(_POSTPROCESS_PARAMETER_NAMES), keep="last"
    )
    trials = trials.sort_values(
        ["f1", "boundary_mae_seconds", "precision"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    best_parameters = {
        name: float(trials.iloc[0][name]) for name in _POSTPROCESS_PARAMETER_NAMES
    }
    return best_parameters, trials
