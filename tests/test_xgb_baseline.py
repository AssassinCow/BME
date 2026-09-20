import numpy as np
import pandas as pd

from bme_eating.models.xgb_baseline import (
    _event_balanced_weights,
    _sample_boundary_rows,
    feature_columns,
    load_xgboost_fold,
    predict_xgboost,
    train_xgboost_fold,
)


def test_positive_sample_weights_are_balanced_by_event():
    frame = pd.DataFrame(
        {
            "state_target": [1.0, 1.0, 1.0, 1.0, 0.0],
            "event_id": ["long", "long", "long", "short", ""],
        }
    )
    weights = _event_balanced_weights(frame)
    assert np.isclose(weights[:3].sum(), weights[3])
    assert weights[4] == 1.0


def test_different_hand_weight_is_normalized_and_never_a_feature():
    frame = pd.DataFrame(
        {
            "state_target": [1.0, 1.0, 1.0, 0.0],
            "event_id": ["same", "different", "different", ""],
            "hand_relation": ["same", "different", "different", "background"],
            "feature_a": [1.0, 2.0, 3.0, 4.0],
        }
    )
    weights = _event_balanced_weights(frame, different_hand_weight=2.0)
    assert np.isclose(weights[:3].sum(), 3.0)
    assert weights[1:3].sum() > weights[0]
    assert "hand_relation" not in feature_columns(frame)


def test_censored_boundaries_are_excluded_from_boundary_training():
    frame = pd.DataFrame(
        {
            "start_target": [1.0, 1.0, 0.0, 0.0],
            "start_loss_mask": [1.0, 0.0, 1.0, 0.0],
        }
    )
    selected, remaining = _sample_boundary_rows(
        frame, "start_target", "start_loss_mask", 20.0, 2026
    )
    assert set(selected.index) == {0, 2}
    assert remaining.empty


def test_xgboost_search_checkpoint_resumes_completed_trials(tmp_path):
    rows = []
    subject_folds = {}
    for subject_index in range(12):
        subject_key = f"s{subject_index}"
        subject_folds[subject_key] = 0 if subject_index < 2 else 1
        for row_index in range(4):
            positive = row_index % 2
            rows.append(
                {
                    "subject_key": subject_key,
                    "segment_id": f"{subject_key}-segment",
                    "session_id": f"{subject_key}-session",
                    "timestamp_ms": row_index * 3000,
                    "state_target": float(positive),
                    "state_loss_mask": 1.0,
                    "event_id": f"{subject_key}-event" if positive else "",
                    "hand_relation": "same" if subject_index % 2 else "different",
                    "distance_to_event_seconds": 0.0 if positive else 3600.0,
                    "feature_a": float(subject_index + row_index),
                    "feature_b": float(positive),
                }
            )
    features = pd.DataFrame(rows)
    config = {
        "device": "cpu",
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "n_estimators": 10,
        "early_stopping_rounds": 2,
        "trials": 2,
        "near_event_minutes": 30,
        "far_negative_to_positive_ratio": 100,
        "hard_negative_to_positive_ratio": 2,
        "random_seed": 2026,
    }

    train_xgboost_fold(features, subject_folds, 0, config, tmp_path)
    checkpoint = tmp_path / "xgboost_search.checkpoint.jsonl"
    first_checkpoint = checkpoint.read_text(encoding="utf-8")
    assert len(pd.read_csv(tmp_path / "trials.csv")) == 2

    with checkpoint.open("a", encoding="utf-8") as handle:
        handle.write('{"interrupted":')
    train_xgboost_fold(features, subject_folds, 0, config, tmp_path)

    assert checkpoint.read_text(encoding="utf-8").startswith(first_checkpoint)
    assert len(pd.read_csv(tmp_path / "trials.csv")) == 2
    loaded, columns = load_xgboost_fold(tmp_path)
    predictions = predict_xgboost(loaded, features.iloc[:2], columns)
    assert list(predictions.columns) == [
        "subject_key",
        "segment_id",
        "session_id",
        "timestamp_ms",
        "state_probability",
        "start_probability",
        "end_probability",
    ]
