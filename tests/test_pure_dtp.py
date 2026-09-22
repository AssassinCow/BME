import numpy as np
import pandas as pd
import pytest

from bme_eating.pure_dtp import validate_dtp_splits, window_diagnostics


def _predictions(subjects):
    return pd.DataFrame(
        {
            "subject_key": subjects,
            "segment_id": [f"segment_{index}" for index in range(len(subjects))],
            "session_id": [f"session_{index}" for index in range(len(subjects))],
            "timestamp_ms": [index * 3000 for index in range(len(subjects))],
            "state_probability": [0.2] * len(subjects),
            "start_probability": [0.1] * len(subjects),
            "end_probability": [0.1] * len(subjects),
        }
    )


def test_validate_dtp_splits_requires_disjoint_complete_subjects():
    folds = {"a": 1, "b": 1, "c": 1, "d": 0}
    oof = _predictions(["a", "b", "c"])
    oof["calibration_fold"] = [0, 1, 2]
    test = _predictions(["d"])
    validated_oof, validated_test = validate_dtp_splits(oof, test, oof, test, folds, 0)
    assert len(validated_oof) == 3
    assert len(validated_test) == 1

    with pytest.raises(ValueError, match="disagree with frozen subject folds"):
        validate_dtp_splits(oof, test, oof, test, {**folds, "d": 1}, 0)


def test_validate_dtp_splits_rejects_partition_reuse_and_duplicate_timeline():
    folds = {"a": 1, "b": 1, "c": 1, "d": 0}
    oof = _predictions(["a", "b", "c"])
    oof["calibration_fold"] = [0, 0, 2]
    test = _predictions(["d"])
    with pytest.raises(ValueError, match="three inner partitions"):
        validate_dtp_splits(oof, test, oof, test, folds, 0)
    oof = pd.concat([oof, oof.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate alignment keys"):
        validate_dtp_splits(oof, test, oof.iloc[:3], test, folds, 0)


def test_window_diagnostics_masks_invalid_labels_and_requires_alignment():
    predictions = _predictions(["a", "a", "a"])
    anchors = predictions[["subject_key", "session_id", "timestamp_ms"]].copy()
    anchors["state_target"] = [1.0, 0.0, 1.0]
    anchors["state_loss_mask"] = [1.0, 1.0, 0.0]
    metrics = window_diagnostics(predictions, anchors)
    assert metrics["eligible_windows"] == 2
    assert metrics["positive_windows"] == 1
    assert np.isfinite(metrics["auprc"])
    with pytest.raises(ValueError, match="without anchor labels"):
        window_diagnostics(predictions, anchors.iloc[:2])
