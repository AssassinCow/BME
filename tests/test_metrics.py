import numpy as np
import pandas as pd
import pytest

from bme_eating.metrics import evaluate_events, match_events


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

