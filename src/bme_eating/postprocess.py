from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from bme_eating.metrics import evaluate_events

_LEGACY_PARAMETER_NAMES = (
    "ema_half_life_seconds",
    "high_threshold",
    "low_threshold",
    "minimum_event_seconds",
    "merge_gap_seconds",
    "boundary_lookback_seconds",
)

_DUAL_EMA_STAGE1_NAMES = (
    "fast_ema_half_life_seconds",
    "slow_ema_half_life_seconds",
    "fast_high_threshold",
    "slow_high_threshold",
    "exit_threshold_ratio",
    "off_duration_seconds",
)

_DUAL_EMA_STAGE2_NAMES = (
    "minimum_event_seconds",
    "merge_gap_seconds",
    "boundary_lookback_seconds",
)

_DUAL_EMA_PARAMETER_NAMES = _DUAL_EMA_STAGE1_NAMES + _DUAL_EMA_STAGE2_NAMES
_POSTPROCESS_PARAMETER_NAMES = _LEGACY_PARAMETER_NAMES

_POSTPROCESS_WORKER_CONTEXT: tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    float,
    str,
] | None = None

_DUAL_WORKER_CONTEXT: tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    float,
    str,
] | None = None


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
    ignore: pd.DataFrame,
    search: dict[str, list[float]],
    iou_threshold: float,
) -> str:
    prediction_columns = [
        "subject_key",
        "session_id",
        "timestamp_ms",
        "state_probability",
        "start_probability",
        "end_probability",
    ]
    truth_columns = ["subject_key", "start_ms", "end_ms"]
    payload = {
        "version": 3,
        "detector_mode": str(search.get("detector_mode", "hysteresis_v1")),
        "boundary_model_version": str(search.get("boundary_model_version", "derivative_v1")),
        "search": {key: value for key, value in search.items() if key != "workers"},
        "iou_threshold": float(iou_threshold),
        "predictions": _frame_digest(predictions, prediction_columns),
        "truth": _frame_digest(truth, truth_columns),
        "ignore": _frame_digest(ignore, truth_columns),
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
    ema_half_life_seconds: float = 12.0,
    high_threshold: float = 0.6,
    low_threshold: float = 0.3,
    minimum_event_seconds: float = 30.0,
    merge_gap_seconds: float = 60.0,
    boundary_lookback_seconds: float = 30.0,
    *,
    detector_mode: str = "hysteresis_v1",
    fast_ema_half_life_seconds: float | None = None,
    slow_ema_half_life_seconds: float | None = None,
    fast_high_threshold: float | None = None,
    slow_high_threshold: float | None = None,
    exit_threshold_ratio: float = 0.5,
    off_duration_seconds: float = 15.0,
) -> pd.DataFrame:
    if detector_mode == "dual_ema":
        dual_parameters = {
            "fast_ema_half_life_seconds": fast_ema_half_life_seconds,
            "slow_ema_half_life_seconds": slow_ema_half_life_seconds,
            "fast_high_threshold": fast_high_threshold,
            "slow_high_threshold": slow_high_threshold,
        }
        missing = [name for name, value in dual_parameters.items() if value is None]
        if missing:
            raise ValueError(f"Missing dual_ema parameters: {missing}")
        return _dual_ema_probabilities_to_events(
            predictions,
            fast_ema_half_life_seconds=float(fast_ema_half_life_seconds),
            slow_ema_half_life_seconds=float(slow_ema_half_life_seconds),
            fast_high_threshold=float(fast_high_threshold),
            slow_high_threshold=float(slow_high_threshold),
            exit_threshold_ratio=float(exit_threshold_ratio),
            off_duration_seconds=float(off_duration_seconds),
            minimum_event_seconds=float(minimum_event_seconds),
            merge_gap_seconds=float(merge_gap_seconds),
            boundary_lookback_seconds=float(boundary_lookback_seconds),
        )
    if detector_mode != "hysteresis_v1":
        raise ValueError(f"Unknown detector mode: {detector_mode}")
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
        "session_id",
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
            columns=["subject_key", "session_id", "start_ms", "end_ms", "score"]
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
    for (subject_key, session_id), group in predictions.groupby(
        ["subject_key", "session_id"], sort=False
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
                        "session_id": session_id,
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
                    "session_id": session_id,
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
    return pd.DataFrame(events, columns=["subject_key", "session_id", "start_ms", "end_ms", "score"])


def _validate_predictions(predictions: pd.DataFrame) -> None:
    required = {
        "subject_key",
        "session_id",
        "timestamp_ms",
        "state_probability",
        "start_probability",
        "end_probability",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    numeric = predictions[
        ["timestamp_ms", "state_probability", "start_probability", "end_probability"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Prediction timestamps and probabilities must be finite")
    if len(numeric) and ((numeric[:, 1:] < 0) | (numeric[:, 1:] > 1)).any():
        raise ValueError("Prediction probabilities must be in [0, 1]")


def _dual_ema_probabilities_to_events(
    predictions: pd.DataFrame,
    *,
    fast_ema_half_life_seconds: float,
    slow_ema_half_life_seconds: float,
    fast_high_threshold: float,
    slow_high_threshold: float,
    exit_threshold_ratio: float,
    off_duration_seconds: float,
    minimum_event_seconds: float,
    merge_gap_seconds: float,
    boundary_lookback_seconds: float,
) -> pd.DataFrame:
    values = (
        fast_ema_half_life_seconds,
        slow_ema_half_life_seconds,
        fast_high_threshold,
        slow_high_threshold,
        exit_threshold_ratio,
        off_duration_seconds,
        minimum_event_seconds,
        merge_gap_seconds,
        boundary_lookback_seconds,
    )
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("Postprocessing parameters must be finite")
    if fast_ema_half_life_seconds <= 0 or slow_ema_half_life_seconds <= 0:
        raise ValueError("EMA half lives must be positive")
    if not 0 < exit_threshold_ratio < 1:
        raise ValueError("exit_threshold_ratio must be in (0, 1)")
    if not 0 <= fast_high_threshold <= 1 or not 0 <= slow_high_threshold <= 1:
        raise ValueError("Dual EMA thresholds must be in [0, 1]")
    if min(
        off_duration_seconds,
        minimum_event_seconds,
        merge_gap_seconds,
        boundary_lookback_seconds,
    ) < 0:
        raise ValueError("Postprocessing durations must be non-negative")
    _validate_predictions(predictions)
    if predictions.empty:
        return pd.DataFrame(
            columns=["subject_key", "session_id", "start_ms", "end_ms", "score"]
        )

    events: list[dict[str, object]] = []
    for (subject_key, session_id), group in predictions.groupby(
        ["subject_key", "session_id"], sort=False
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
        if not len(timestamps):
            continue
        step_seconds = (
            float(np.median(np.diff(timestamps)) / 1000.0) if len(timestamps) > 1 else 3.0
        )
        state = group["state_probability"].to_numpy(dtype=np.float64)
        # A zero initial state prevents one sample at a session boundary from
        # being treated as a fully warmed-up slow-channel response.
        fast = causal_ema(
            np.concatenate(([0.0], state)), step_seconds, fast_ema_half_life_seconds
        )[1:]
        slow = causal_ema(
            np.concatenate(([0.0], state)), step_seconds, slow_ema_half_life_seconds
        )[1:]
        start_probability = group["start_probability"].to_numpy(dtype=np.float64)
        end_probability = group["end_probability"].to_numpy(dtype=np.float64)
        lookback_rows = max(1, round(boundary_lookback_seconds / step_seconds))
        off_rows = max(1, math.ceil(off_duration_seconds / step_seconds))
        fast_exit = fast_high_threshold * exit_threshold_ratio
        slow_exit = slow_high_threshold * exit_threshold_ratio
        active = False
        start_index = 0
        below_count = 0
        candidates: list[dict[str, object]] = []
        for index in range(len(timestamps)):
            recent_fast = fast[max(0, index - 2) : index + 1]
            fast_votes = int(np.count_nonzero(recent_fast >= fast_high_threshold))
            should_start = fast_votes >= 2 or slow[index] >= slow_high_threshold
            if not active and should_start:
                search_start = max(0, index - lookback_rows)
                local = int(np.argmax(start_probability[search_start : index + 1]))
                start_index = search_start + local
                active = True
                below_count = 0
                continue
            if not active:
                continue
            if fast[index] < fast_exit and slow[index] < slow_exit:
                below_count += 1
            else:
                below_count = 0
            if below_count < off_rows:
                continue
            exit_start = index - below_count + 1
            search_start = max(start_index, exit_start - lookback_rows)
            local = int(np.argmax(end_probability[search_start : index + 1]))
            end_index = search_start + local
            if end_index <= start_index:
                end_index = exit_start
            candidates.append(
                {
                    "subject_key": subject_key,
                    "session_id": session_id,
                    "start_ms": int(timestamps[start_index]),
                    "end_ms": int(timestamps[end_index]),
                    "score": float(np.mean(np.maximum(fast, slow)[start_index : end_index + 1])),
                }
            )
            active = False
            below_count = 0
        if active:
            candidates.append(
                {
                    "subject_key": subject_key,
                    "session_id": session_id,
                    "start_ms": int(timestamps[start_index]),
                    "end_ms": int(timestamps[-1]),
                    "score": float(np.mean(np.maximum(fast, slow)[start_index:])),
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
                merged[-1]["end_ms"] = max(merged[-1]["end_ms"], event["end_ms"])
                merged[-1]["score"] = max(merged[-1]["score"], event["score"])
            else:
                merged.append(event)
        events.extend(merged)
    return pd.DataFrame(
        events, columns=["subject_key", "session_id", "start_ms", "end_ms", "score"]
    )


def _evaluate_postprocess_combination(
    combination: tuple[float, ...],
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    ignore: pd.DataFrame,
    iou_threshold: float,
    matching_method: str,
) -> dict[str, float]:
    parameters = dict(zip(_POSTPROCESS_PARAMETER_NAMES, combination))
    events = probabilities_to_events(predictions, **parameters)
    metrics, _ = evaluate_events(
        truth,
        events,
        iou_threshold=iou_threshold,
        method=matching_method,
        ignore=ignore,
    )
    boundary_values = [
        value
        for value in (metrics["start_mae_seconds"], metrics["end_mae_seconds"])
        if np.isfinite(value)
    ]
    boundary_mae = float(np.mean(boundary_values)) if boundary_values else float("inf")
    return {
        **parameters,
        **{name: float(value) for name, value in metrics.items()},
        "boundary_mae_seconds": boundary_mae,
    }


def _initialize_postprocess_worker(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    ignore: pd.DataFrame,
    iou_threshold: float,
    matching_method: str,
) -> None:
    global _POSTPROCESS_WORKER_CONTEXT
    _POSTPROCESS_WORKER_CONTEXT = (
        predictions,
        truth,
        ignore,
        iou_threshold,
        matching_method,
    )


def _postprocess_worker(combination: tuple[float, ...]) -> dict[str, float]:
    if _POSTPROCESS_WORKER_CONTEXT is None:
        raise RuntimeError("postprocess worker was not initialized")
    return _evaluate_postprocess_combination(
        combination,
        *_POSTPROCESS_WORKER_CONTEXT,
    )


def tune_postprocess_parameters(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    search: dict[str, list[float]],
    iou_threshold: float,
    checkpoint_path: Path | None = None,
    show_progress: bool = True,
    ignore: pd.DataFrame | None = None,
    matching_method: str = "max_cardinality_iou",
    workers: int = 1,
) -> tuple[dict[str, float], pd.DataFrame]:
    detector_mode = str(search.get("detector_mode", "hysteresis_v1"))
    if detector_mode == "dual_ema":
        return tune_dual_ema_parameters(
            predictions,
            truth,
            search,
            iou_threshold,
            checkpoint_path=checkpoint_path,
            show_progress=show_progress,
            ignore=ignore,
            matching_method=matching_method,
            workers=workers,
        )
    if detector_mode != "hysteresis_v1":
        raise ValueError(f"Unknown detector mode: {detector_mode}")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if ignore is None:
        ignore = pd.DataFrame(columns=["subject_key", "start_ms", "end_ms"])
    high_values = [float(value) for value in search.get("high_threshold", [])]
    probabilities = predictions["state_probability"].to_numpy(dtype=np.float64)
    for quantile in search.get("high_threshold_quantiles", []):
        high_values.append(float(np.quantile(probabilities, float(quantile))))
    high_values = sorted({min(1.0, max(0.0, value)) for value in high_values})
    threshold_pairs: set[tuple[float, float]] = set()
    for high in high_values:
        for low in search.get("low_threshold", []):
            if float(low) < high:
                threshold_pairs.add((high, float(low)))
        for ratio in search.get("low_threshold_ratios", []):
            low = high * float(ratio)
            if 0 <= low < high:
                threshold_pairs.add((high, low))
    other_names = [
        name for name in _POSTPROCESS_PARAMETER_NAMES if name not in {"high_threshold", "low_threshold"}
    ]
    combinations = []
    for other_values in product(*(search[name] for name in other_names)):
        other = dict(zip(other_names, (float(value) for value in other_values)))
        for high, low in sorted(threshold_pairs):
            parameters = {**other, "high_threshold": high, "low_threshold": low}
            combinations.append(tuple(parameters[name] for name in _POSTPROCESS_PARAMETER_NAMES))
    signature = _postprocess_search_signature(
        predictions, truth, ignore, search, iou_threshold
    )
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
    pending = [combination for combination in combinations if combination not in completed]

    def record_result(combination: tuple[float, ...], row: dict[str, float]) -> None:
        rows.append(row)
        completed.add(combination)
        if checkpoint_path is not None:
            _append_search_checkpoint(checkpoint_path, row)
        progress.update(1)
        progress.set_postfix(f1=f"{row['f1']:.4f}")

    if workers == 1:
        for combination in pending:
            row = _evaluate_postprocess_combination(
                combination,
                predictions,
                truth,
                ignore,
                iou_threshold,
                matching_method,
            )
            record_result(combination, row)
    elif pending:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(pending)),
            initializer=_initialize_postprocess_worker,
            initargs=(predictions, truth, ignore, iou_threshold, matching_method),
        ) as executor:
            futures = {
                executor.submit(_postprocess_worker, combination): combination
                for combination in pending
            }
            for future in as_completed(futures):
                combination = futures[future]
                record_result(combination, future.result())
    progress.close()

    if not rows:
        raise RuntimeError("Postprocessing search produced no valid parameter combination")
    trials = pd.DataFrame(rows).drop_duplicates(
        subset=list(_POSTPROCESS_PARAMETER_NAMES), keep="last"
    )
    trials = trials.sort_values(
        ["f1", "boundary_mae_seconds", "precision", *_POSTPROCESS_PARAMETER_NAMES],
        ascending=[False, True, False, *([True] * len(_POSTPROCESS_PARAMETER_NAMES))],
    ).reset_index(drop=True)
    best_parameters = {
        name: float(trials.iloc[0][name]) for name in _POSTPROCESS_PARAMETER_NAMES
    }
    return best_parameters, trials


def _dual_search_signature(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    ignore: pd.DataFrame,
    search: dict[str, Any],
    iou_threshold: float,
) -> str:
    payload = {
        "version": 1,
        "detector_mode": "dual_ema",
        "boundary_model_version": str(search.get("boundary_model_version", "derivative_v1")),
        "search": {key: value for key, value in search.items() if key != "workers"},
        "iou_threshold": float(iou_threshold),
        "predictions": _frame_digest(
            predictions,
            [
                "subject_key",
                "session_id",
                "timestamp_ms",
                "state_probability",
                "start_probability",
                "end_probability",
                "calibration_fold",
            ],
        ),
        "truth": _frame_digest(truth, ["subject_key", "start_ms", "end_ms"]),
        "ignore": _frame_digest(ignore, ["subject_key", "start_ms", "end_ms"]),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _evaluate_dual_combination(
    parameters: dict[str, float],
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    ignore: pd.DataFrame,
    iou_threshold: float,
    matching_method: str,
) -> dict[str, float]:
    events = probabilities_to_events(
        predictions,
        detector_mode="dual_ema",
        **parameters,
    )
    metrics, matches = evaluate_events(
        truth,
        events,
        iou_threshold=iou_threshold,
        method=matching_method,
        ignore=ignore,
    )
    fold_scores: list[float] = []
    if "calibration_fold" in predictions.columns:
        subject_fold = (
            predictions[["subject_key", "calibration_fold"]]
            .drop_duplicates()
            .set_index("subject_key")["calibration_fold"]
        )
        if subject_fold.index.has_duplicates:
            raise ValueError("A calibration subject appears in more than one inner fold")
        for fold in sorted(subject_fold.unique()):
            subjects = set(subject_fold[subject_fold == fold].index.astype(str))
            fold_metrics, _ = evaluate_events(
                truth[truth["subject_key"].astype(str).isin(subjects)],
                events[events["subject_key"].astype(str).isin(subjects)],
                iou_threshold=iou_threshold,
                method=matching_method,
                ignore=ignore[ignore["subject_key"].astype(str).isin(subjects)],
            )
            fold_scores.append(float(fold_metrics["f1"]))
    if not fold_scores:
        fold_scores = [float(metrics["f1"])]
    truth_relation = truth.get(
        "hand_relation", pd.Series("unknown", index=truth.index, dtype=object)
    )
    match_relation = matches.get(
        "hand_relation", pd.Series("unknown", index=matches.index, dtype=object)
    )
    different_truth = int((truth_relation == "different").sum())
    different_matches = int((match_relation == "different").sum())
    boundary_values = [
        float(metrics[name])
        for name in ("start_mae_seconds", "end_mae_seconds")
        if np.isfinite(float(metrics[name]))
    ]
    return {
        **parameters,
        **{name: float(value) for name, value in metrics.items()},
        "mean_fold_f1": float(np.mean(fold_scores)),
        "minimum_fold_f1": float(np.min(fold_scores)),
        "different_sensitivity": (
            different_matches / different_truth if different_truth else float("nan")
        ),
        "boundary_mae_seconds": (
            float(np.mean(boundary_values)) if boundary_values else float("inf")
        ),
    }


def _initialize_dual_worker(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    ignore: pd.DataFrame,
    iou_threshold: float,
    matching_method: str,
) -> None:
    global _DUAL_WORKER_CONTEXT
    _DUAL_WORKER_CONTEXT = (
        predictions,
        truth,
        ignore,
        iou_threshold,
        matching_method,
    )


def _dual_worker(parameters: dict[str, float]) -> dict[str, float]:
    if _DUAL_WORKER_CONTEXT is None:
        raise RuntimeError("dual EMA worker was not initialized")
    return _evaluate_dual_combination(parameters, *_DUAL_WORKER_CONTEXT)


def _rank_dual_trials(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return trials
    maximum = float(trials["mean_fold_f1"].max())
    trials = trials.copy()
    trials["within_mean_f1_tolerance"] = trials["mean_fold_f1"] >= maximum - 0.01
    return trials.sort_values(
        [
            "within_mean_f1_tolerance",
            "minimum_fold_f1",
            "different_sensitivity",
            "boundary_mae_seconds",
            "mean_fold_f1",
            *_DUAL_EMA_PARAMETER_NAMES,
        ],
        ascending=[False, False, False, True, False, *([True] * len(_DUAL_EMA_PARAMETER_NAMES))],
    ).reset_index(drop=True)


def tune_dual_ema_parameters(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    search: dict[str, Any],
    iou_threshold: float,
    checkpoint_path: Path | None = None,
    show_progress: bool = True,
    ignore: pd.DataFrame | None = None,
    matching_method: str = "max_cardinality_iou",
    workers: int = 1,
) -> tuple[dict[str, float], pd.DataFrame]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if ignore is None:
        ignore = pd.DataFrame(columns=["subject_key", "start_ms", "end_ms"])
    for name in _DUAL_EMA_PARAMETER_NAMES:
        if name not in search or not search[name]:
            raise ValueError(f"dual_ema search is missing non-empty parameter grid: {name}")
    signature = _dual_search_signature(predictions, truth, ignore, search, iou_threshold)
    rows: list[dict[str, float]] = []
    if checkpoint_path and checkpoint_path.exists():
        try:
            lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
            if lines and json.loads(lines[0]) == {"version": 2, "signature": signature}:
                for line in lines[1:]:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict) and all(
                        name in row for name in _DUAL_EMA_PARAMETER_NAMES
                    ):
                        rows.append(row)
        except (OSError, TypeError, json.JSONDecodeError):
            rows = []
    if checkpoint_path and not rows:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps({"version": 2, "signature": signature}) + "\n",
            encoding="utf-8",
        )
    completed = {
        tuple(float(row[name]) for name in _DUAL_EMA_PARAMETER_NAMES) for row in rows
    }
    fixed_stage2 = {
        name: float(search[name][0]) for name in _DUAL_EMA_STAGE2_NAMES
    }
    stage1_parameters = [
        {**dict(zip(_DUAL_EMA_STAGE1_NAMES, map(float, values))), **fixed_stage2}
        for values in product(*(search[name] for name in _DUAL_EMA_STAGE1_NAMES))
    ]

    def record(parameters: dict[str, float], row: dict[str, float], progress: tqdm) -> None:
        key = tuple(float(parameters[name]) for name in _DUAL_EMA_PARAMETER_NAMES)
        rows.append(row)
        completed.add(key)
        if checkpoint_path:
            _append_search_checkpoint(checkpoint_path, row)
        progress.update(1)
        progress.set_postfix(f1=f"{row['mean_fold_f1']:.4f}")

    def run_stage(parameters_list: list[dict[str, float]], description: str) -> None:
        progress = tqdm(
            total=len(parameters_list),
            desc=description,
            unit="combination",
            disable=not show_progress,
        )
        pending: list[dict[str, float]] = []
        for parameters in parameters_list:
            key = tuple(float(parameters[name]) for name in _DUAL_EMA_PARAMETER_NAMES)
            if key in completed:
                progress.update(1)
            else:
                pending.append(parameters)
        if workers == 1:
            for parameters in pending:
                row = _evaluate_dual_combination(
                    parameters, predictions, truth, ignore, iou_threshold, matching_method
                )
                record(parameters, row, progress)
        elif pending:
            with ProcessPoolExecutor(
                max_workers=min(workers, len(pending)),
                initializer=_initialize_dual_worker,
                initargs=(predictions, truth, ignore, iou_threshold, matching_method),
            ) as executor:
                futures = {
                    executor.submit(_dual_worker, parameters): parameters
                    for parameters in pending
                }
                for future in as_completed(futures):
                    parameters = futures[future]
                    record(parameters, future.result(), progress)
        progress.close()

    run_stage(stage1_parameters, "Dual EMA stage 1")
    stage1 = pd.DataFrame(rows)
    stage1 = stage1[
        np.logical_and.reduce(
            [stage1[name] == value for name, value in fixed_stage2.items()]
        )
    ]
    top_count = max(1, int(search.get("stage1_keep", 20)))
    top_stage1 = _rank_dual_trials(stage1).head(top_count)
    stage2_parameters: list[dict[str, float]] = []
    for stage1_row in top_stage1.itertuples(index=False):
        first = {name: float(getattr(stage1_row, name)) for name in _DUAL_EMA_STAGE1_NAMES}
        for values in product(*(search[name] for name in _DUAL_EMA_STAGE2_NAMES)):
            stage2_parameters.append(
                {**first, **dict(zip(_DUAL_EMA_STAGE2_NAMES, map(float, values)))}
            )
    run_stage(stage2_parameters, "Dual EMA stage 2")
    trials = pd.DataFrame(rows).drop_duplicates(
        subset=list(_DUAL_EMA_PARAMETER_NAMES), keep="last"
    )
    trials = _rank_dual_trials(trials)
    if trials.empty:
        raise RuntimeError("Dual EMA search produced no valid parameter combination")
    best = {name: float(trials.iloc[0][name]) for name in _DUAL_EMA_PARAMETER_NAMES}
    best["detector_mode"] = "dual_ema"
    return best, trials


def parameters_at_search_boundary(
    selected: dict[str, Any], trials: pd.DataFrame
) -> dict[str, bool]:
    """Return every genuinely searched parameter selected at a finite grid edge."""

    flags: dict[str, bool] = {}
    ignored = {
        "f1",
        "precision",
        "sensitivity",
        "true_positive",
        "false_positive",
        "false_negative",
        "ignored_predictions",
        "start_mae_seconds",
        "end_mae_seconds",
        "boundary_mae_seconds",
        "mean_fold_f1",
        "minimum_fold_f1",
        "different_sensitivity",
        "within_mean_f1_tolerance",
    }
    for name, value in selected.items():
        if name in ignored or name not in trials.columns or not isinstance(value, (int, float)):
            continue
        values = sorted(
            float(item) for item in trials[name].dropna().unique() if np.isfinite(float(item))
        )
        if len(values) > 1:
            flags[name] = bool(np.isclose(float(value), values[0]) or np.isclose(float(value), values[-1]))
    return flags
