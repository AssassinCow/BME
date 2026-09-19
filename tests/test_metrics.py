import numpy as np
import pandas as pd
import pytest

from bme_eating.metrics import evaluate_events, masked_average_precision, match_events


def test_iou_must_be_strictly_greater_than_threshold():
    truth = np.asarray([[0.0, 100.0]])
    prediction_at_threshold = np.asarray([[0.0, 25.0]])
    prediction_above_threshold = np.asarray([[0.0, 26.0]])
    assert match_events(truth, prediction_at_threshold, 0.25) == []
    assert len(match_events(truth, prediction_above_threshold, 0.25)) == 1


def test_duplicate_predictions_match_one_truth_once():
    truth = pd.DataFrame(
        [{"subject_key": "s", "start_ms": 0, "end_ms": 1000, "event_id": "e"}]
    )
    prediction = pd.DataFrame(
        [
            {"subject_key": "s", "start_ms": 0, "end_ms": 1000},
            {"subject_key": "s", "start_ms": 50, "end_ms": 950},
        ]
    )
    metrics, _ = evaluate_events(truth, prediction)
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1


def test_metrics_reject_reversed_intervals():
    with pytest.raises(ValueError, match="positive duration"):
        match_events(np.asarray([[100.0, 0.0]]), np.asarray([[0.0, 100.0]]))


def test_maximum_cardinality_precedes_total_iou():
    truth = np.asarray([[0.0, 1000.0], [1000.0, 2000.0]])
    prediction = np.asarray([[0.0, 1670.0], [0.0, 260.0]])
    matches = match_events(truth, prediction, 0.25, method="max_cardinality_iou")
    assert len(matches) == 2


def test_predictions_overlapping_ignore_intervals_are_not_false_positives():
    truth = pd.DataFrame(columns=["subject_key", "start_ms", "end_ms"])
    prediction = pd.DataFrame(
        [{"subject_key": "s", "start_ms": 100, "end_ms": 200}]
    )
    ignore = pd.DataFrame(
        [{"subject_key": "s", "start_ms": 50, "end_ms": 250}]
    )
    metrics, _ = evaluate_events(truth, prediction, ignore=ignore)
    assert metrics["false_positive"] == 0
    assert metrics["ignored_predictions"] == 1


def test_window_auprc_excludes_state_masked_anchors():
    auprc = masked_average_precision(
        targets=np.array([1.0, 0.0, 0.0]),
        probabilities=np.array([0.9, 0.1, 0.99]),
        mask=np.array([1.0, 1.0, 0.0]),
    )

    assert np.isclose(auprc, 1.0)


def test_window_auprc_returns_zero_without_eligible_positive():
    assert masked_average_precision(
        targets=np.array([0.0, 1.0]),
        probabilities=np.array([0.2, 0.8]),
        mask=np.array([1.0, 0.0]),
    ) == 0.0


def test_window_auprc_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        masked_average_precision(
            targets=np.array([1.0]),
            probabilities=np.array([0.9, 0.1]),
            mask=np.array([1.0]),
        )

