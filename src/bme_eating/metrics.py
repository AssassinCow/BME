from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Match:
    truth_index: int
    prediction_index: int
    iou: float


def interval_iou_matrix(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
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
    method: str = "hungarian",
) -> list[Match]:
    iou = interval_iou_matrix(truth, prediction)
    if iou.size == 0:
        return []
    matches: list[Match] = []
    if method == "hungarian":
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
    method: str = "hungarian",
) -> tuple[dict[str, float], pd.DataFrame]:
    matches_output: list[dict[str, object]] = []
    true_positive = 0
    total_truth = 0
    total_prediction = 0
    for subject_key in sorted(set(truth.get("subject_key", [])) | set(prediction.get("subject_key", []))):
        subject_truth = truth[truth["subject_key"] == subject_key].reset_index(drop=True)
        subject_prediction = prediction[prediction["subject_key"] == subject_key].reset_index(drop=True)
        truth_intervals = subject_truth[["start_ms", "end_ms"]].to_numpy(dtype=np.float64)
        prediction_intervals = subject_prediction[["start_ms", "end_ms"]].to_numpy(
            dtype=np.float64
        )
        matches = match_events(truth_intervals, prediction_intervals, iou_threshold, method)
        true_positive += len(matches)
        total_truth += len(subject_truth)
        total_prediction += len(subject_prediction)
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
                    "iou": match.iou,
                    "start_absolute_error_ms": abs(
                        int(prediction_row.start_ms) - int(truth_row.start_ms)
                    ),
                    "end_absolute_error_ms": abs(
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
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "f1": float(f1),
        "start_mae_seconds": float(matched["start_absolute_error_ms"].mean() / 1000.0)
        if len(matched)
        else float("nan"),
        "end_mae_seconds": float(matched["end_absolute_error_ms"].mean() / 1000.0)
        if len(matched)
        else float("nan"),
    }
    return metrics, matched

