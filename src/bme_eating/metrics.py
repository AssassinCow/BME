from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class Match:
    truth_index: int
    prediction_index: int
    iou: float


def masked_average_precision(
    targets: np.ndarray, probabilities: np.ndarray, mask: np.ndarray
) -> float:
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    if not (len(targets) == len(probabilities) == len(mask)):
        raise ValueError("Targets, probabilities, and mask must have equal lengths")
    eligible = (mask > 0) & np.isfinite(targets) & np.isfinite(probabilities)
    if not eligible.any():
        return 0.0
    binary_targets = targets[eligible] > 0
    if not binary_targets.any():
        return 0.0
    return float(average_precision_score(binary_targets, probabilities[eligible]))


def partition_evaluation_events(
    events: pd.DataFrame, subject_keys: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"subject_key", "valid_duration"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Events are missing columns: {sorted(missing)}")
    selected = events[events["subject_key"].isin(subject_keys) & events["valid_duration"]]
    if "evaluable" in selected.columns:
        evaluable = selected["evaluable"].fillna(False).astype(bool)
    elif "coverage" in selected.columns:
        evaluable = selected["coverage"].eq("full")
    else:
        raise ValueError("Events must contain evaluable or coverage")
    return selected[evaluable].copy(), selected[~evaluable].copy()


def interval_iou_matrix(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.size == 0:
        truth = np.empty((0, 2), dtype=np.float64)
    if prediction.size == 0:
        prediction = np.empty((0, 2), dtype=np.float64)
    for name, intervals in (("truth", truth), ("prediction", prediction)):
        if intervals.ndim != 2 or intervals.shape[1] != 2:
            raise ValueError(f"{name} intervals must have shape (n, 2)")
        if not np.isfinite(intervals).all():
            raise ValueError(f"{name} intervals must be finite")
        if (intervals[:, 1] <= intervals[:, 0]).any():
            raise ValueError(f"{name} intervals must have positive duration")
    if len(truth) == 0 or len(prediction) == 0:
        return np.zeros((len(truth), len(prediction)), dtype=np.float64)
    truth_start = truth[:, 0][:, None]
    truth_end = truth[:, 1][:, None]
    prediction_start = prediction[:, 0][None, :]
    prediction_end = prediction[:, 1][None, :]
    intersection = np.maximum(
        0.0, np.minimum(truth_end, prediction_end) - np.maximum(truth_start, prediction_start)
    )
    union = np.maximum(truth_end, prediction_end) - np.minimum(truth_start, prediction_start)
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_events(
    truth: np.ndarray,
    prediction: np.ndarray,
    iou_threshold: float = 0.25,
    method: str = "max_cardinality_iou",
) -> list[Match]:
    if not np.isfinite(iou_threshold) or not 0 <= iou_threshold < 1:
        raise ValueError("iou_threshold must be in [0, 1)")
    iou = interval_iou_matrix(truth, prediction)
    if iou.size == 0:
        return []
    matches: list[Match] = []
    if method == "max_cardinality_iou":
        truth_count, prediction_count = iou.shape
        size = truth_count + prediction_count
        cardinality_bonus = float(min(truth_count, prediction_count) + 1)
        score = np.zeros((size, size), dtype=np.float64)
        valid = iou > iou_threshold
        score[:truth_count, :prediction_count] = np.where(
            valid, cardinality_bonus + iou, -cardinality_bonus
        )
        truth_indices, prediction_indices = linear_sum_assignment(-score)
        for truth_index, prediction_index in zip(truth_indices, prediction_indices):
            if truth_index >= truth_count or prediction_index >= prediction_count:
                continue
            value = float(iou[truth_index, prediction_index])
            if value > iou_threshold:
                matches.append(Match(int(truth_index), int(prediction_index), value))
    elif method in {"hungarian", "hungarian_iou_legacy"}:
        truth_indices, prediction_indices = linear_sum_assignment(-iou)
        for truth_index, prediction_index in zip(truth_indices, prediction_indices):
            value = float(iou[truth_index, prediction_index])
            if value > iou_threshold:
                matches.append(Match(int(truth_index), int(prediction_index), value))
    elif method == "greedy":
        candidates = [
            (float(iou[row, column]), row, column)
            for row in range(iou.shape[0])
            for column in range(iou.shape[1])
            if iou[row, column] > iou_threshold
        ]
        used_truth: set[int] = set()
        used_prediction: set[int] = set()
        for value, truth_index, prediction_index in sorted(candidates, reverse=True):
            if truth_index in used_truth or prediction_index in used_prediction:
                continue
            matches.append(Match(truth_index, prediction_index, value))
            used_truth.add(truth_index)
            used_prediction.add(prediction_index)
    else:
        raise ValueError(f"Unknown matching method: {method}")
    return matches


def evaluate_events(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    iou_threshold: float = 0.25,
    method: str = "max_cardinality_iou",
    ignore: pd.DataFrame | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    for name, frame in (("truth", truth), ("prediction", prediction)):
        required = {"subject_key", "start_ms", "end_ms"}
        missing = required - set(frame.columns)
        if missing and len(frame):
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if truth.empty and "subject_key" not in truth.columns:
        truth = pd.DataFrame(columns=["subject_key", "start_ms", "end_ms"])
    if prediction.empty and "subject_key" not in prediction.columns:
        prediction = pd.DataFrame(columns=["subject_key", "start_ms", "end_ms"])
    if ignore is None or ignore.empty and "subject_key" not in ignore.columns:
        ignore = pd.DataFrame(columns=["subject_key", "start_ms", "end_ms"])
    matches_output: list[dict[str, object]] = []
    true_positive = 0
    total_truth = 0
    total_prediction = 0
    ignored_predictions = 0
    subjects = (
        set(truth.get("subject_key", []))
        | set(prediction.get("subject_key", []))
        | set(ignore.get("subject_key", []))
    )
    for subject_key in sorted(subjects):
        subject_truth = truth[truth["subject_key"] == subject_key].reset_index(drop=True)
        subject_prediction = prediction[prediction["subject_key"] == subject_key].reset_index(drop=True)
        truth_intervals = subject_truth[["start_ms", "end_ms"]].to_numpy(dtype=np.float64)
        prediction_intervals = subject_prediction[["start_ms", "end_ms"]].to_numpy(
            dtype=np.float64
        )
        matches = match_events(truth_intervals, prediction_intervals, iou_threshold, method)
        matched_prediction_indices = {match.prediction_index for match in matches}
        subject_ignore = ignore[ignore["subject_key"] == subject_key]
        ignored_indices: set[int] = set()
        if len(subject_ignore):
            ignore_intervals = subject_ignore[["start_ms", "end_ms"]].to_numpy(
                dtype=np.float64
            )
            for prediction_index, interval in enumerate(prediction_intervals):
                if prediction_index in matched_prediction_indices:
                    continue
                overlap = np.maximum(
                    0.0,
                    np.minimum(interval[1], ignore_intervals[:, 1])
                    - np.maximum(interval[0], ignore_intervals[:, 0]),
                )
                if np.any(overlap > 0):
                    ignored_indices.add(prediction_index)
        true_positive += len(matches)
        total_truth += len(subject_truth)
        total_prediction += len(subject_prediction) - len(ignored_indices)
        ignored_predictions += len(ignored_indices)
        for match in matches:
            truth_row = subject_truth.iloc[match.truth_index]
            prediction_row = subject_prediction.iloc[match.prediction_index]
            matches_output.append(
                {
                    "subject_key": subject_key,
                    "event_id": truth_row.get("event_id", ""),
                    "hand_relation": truth_row.get("hand_relation", "unknown"),
                    "truth_start_ms": int(truth_row.start_ms),
                    "truth_end_ms": int(truth_row.end_ms),
                    "prediction_start_ms": int(prediction_row.start_ms),
                    "prediction_end_ms": int(prediction_row.end_ms),
                    "prediction_event_id": prediction_row.get("proposal_id", ""),
                    "iou": match.iou,
                    "start_absolute_error_ms": abs(
                        int(prediction_row.start_ms) - int(truth_row.start_ms)
                    ),
                    "end_absolute_error_ms": abs(
                        int(prediction_row.end_ms) - int(truth_row.end_ms)
                    ),
                    "start_signed_error_ms": (
                        int(prediction_row.start_ms) - int(truth_row.start_ms)
                    ),
                    "end_signed_error_ms": (
                        int(prediction_row.end_ms) - int(truth_row.end_ms)
                    ),
                }
            )
    false_positive = total_prediction - true_positive
    false_negative = total_truth - true_positive
    precision = true_positive / total_prediction if total_prediction else 0.0
    sensitivity = true_positive / total_truth if total_truth else 0.0
    f1 = (
        2.0 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity
        else 0.0
    )
    matched = pd.DataFrame(matches_output)
    metrics = {
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "ignored_predictions": float(ignored_predictions),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "f1": float(f1),
        "start_mae_seconds": float(matched["start_absolute_error_ms"].mean() / 1000.0)
        if len(matched)
        else float("nan"),
        "end_mae_seconds": float(matched["end_absolute_error_ms"].mean() / 1000.0)
        if len(matched)
        else float("nan"),
        "start_signed_error_seconds": float(matched["start_signed_error_ms"].mean() / 1000.0)
        if len(matched)
        else float("nan"),
        "end_signed_error_seconds": float(matched["end_signed_error_ms"].mean() / 1000.0)
        if len(matched)
        else float("nan"),
    }
    return metrics, matched

