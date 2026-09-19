import pandas as pd

from bme_eating.models.xgb_baseline import train_xgboost_fold


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
                    "timestamp_ms": row_index * 3000,
                    "state_target": float(positive),
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
