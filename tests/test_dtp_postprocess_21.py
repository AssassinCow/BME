import json
from argparse import Namespace

import pandas as pd
import pytest

import bme_eating.dtp_postprocess_21 as module


def _settings():
    return {
        "detector_mode": "dual_ema",
        "fast_ema_half_life_seconds": 12,
        "slow_ema_half_life_seconds": 36,
        "fast_high_threshold": 0.55,
        "slow_high_threshold": 0.3,
        "exit_threshold_ratio": 0.25,
        "off_duration_seconds": 60,
        "minimum_event_seconds": 30,
        "merge_gap_seconds": 60,
        "boundary_lookback_seconds": 60,
    }


def test_legacy_crossfit_fits_each_threshold_without_heldout_subject(monkeypatch):
    predictions = pd.DataFrame(
        [
            (subject, "day", offset * 3000, 0.9, 0.0, 0.0, fold)
            for fold, subject in enumerate(("first", "second", "third"))
            for offset in range(12)
        ],
        columns=[
            "subject_key",
            "session_id",
            "timestamp_ms",
            "state_probability",
            "start_probability",
            "end_probability",
            "calibration_fold",
        ],
    )
    truth = pd.DataFrame(
        [(subject, 0, 33000, True, True, "different") for subject in ("first", "second", "third")],
        columns=[
            "subject_key",
            "start_ms",
            "end_ms",
            "valid_duration",
            "evaluable",
            "hand_relation",
        ],
    )
    scores = {"first": 0.1, "second": 0.5, "third": 0.9}
    calls = []

    def generate(frame, **kwargs):
        calls.append(set(frame.subject_key))
        return pd.DataFrame(
            [(subject, "day", 0, 33000, scores[subject]) for subject in frame.subject_key.unique()],
            columns=["subject_key", "session_id", "start_ms", "end_ms", "score"],
        )

    real_fit = module.fit_score_threshold
    training_subjects = []

    def fit(frame, quantile):
        training_subjects.append(set(frame.subject_key))
        return real_fit(frame, quantile)

    monkeypatch.setattr(module, "probabilities_to_events", generate)
    monkeypatch.setattr(module, "fit_score_threshold", fit)
    record, selected, fixed_candidates, trials, generated = module.crossfit_legacy_generator(
        predictions, truth, _settings(), [0.5], 0.25, "max_cardinality_iou"
    )
    assert calls == [{"first", "second", "third"}]
    assert training_subjects[:3] == [{"second", "third"}, {"first", "third"}, {"first", "second"}]
    assert record["scopes"][0]["selected_threshold"] == 0.5
    assert record["scopes"][2]["selected_threshold"] == 0.1
    assert (
        set(selected.subject_key)
        == set(fixed_candidates[0.5].subject_key)
        == {
            "second",
            "third",
        }
    )
    assert set(generated.subject_key) == {"first", "second", "third"}
    assert trials.selected_within_train.all()


def test_frozen_inference_does_not_fit_threshold_or_search(monkeypatch):
    monkeypatch.setattr(module, "fit_score_threshold", lambda *args: pytest.fail("threshold fit"))
    predictions = pd.DataFrame(
        {
            "subject_key": ["one"] * 100,
            "session_id": ["day"] * 100,
            "timestamp_ms": [offset * 3000 for offset in range(100)],
            "state_probability": [0.9] * 100,
            "start_probability": [0.0] * 100,
            "end_probability": [0.0] * 100,
        }
    )
    events = module.frozen_events(predictions, _settings(), 0.7)
    assert not events.empty
    assert events.score.ge(0.7).all()
    with pytest.raises(ValueError, match="threshold"):
        module.frozen_events(predictions, _settings(), 1.1)


def test_selection_loader_rejects_modified_artifact(tmp_path, monkeypatch):
    directory = tmp_path / "dtp_postprocess_21" / "fold_0"
    directory.mkdir(parents=True)
    selected = directory / "selected_dtp_postprocess.json"
    selected.write_text(
        json.dumps({"protocol_version": "2.1", "selection_run": "dtp_postprocess_21"})
    )
    (directory / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment": {"name": "dtp_postprocess_21", "fold": 0},
                "git": {"commit": "commit", "dirty": False},
                "artifact_hashes": {selected.name: "bad-hash"},
            }
        )
    )
    monkeypatch.setattr(module, "_git_identity", lambda: {"commit": "commit", "dirty": False})
    with pytest.raises(RuntimeError, match="artifact hash changed"):
        module._load_selection(directory, "dtp_postprocess_21")


def test_component_screen_accepts_unique_rescue_without_pure_dtp_f1_requirement():
    truth = pd.DataFrame(
        [
            ("one", 0, 100, "different"),
            ("one", 200, 300, "same"),
        ],
        columns=["subject_key", "start_ms", "end_ms", "hand_relation"],
    )
    baseline = pd.DataFrame([("one", 200, 300)], columns=["subject_key", "start_ms", "end_ms"])
    candidate = pd.DataFrame(
        [("one", 0, 100), ("one", 400, 500), ("one", 600, 700)],
        columns=["subject_key", "start_ms", "end_ms"],
    )
    diagnostic = module.complementarity(
        truth, truth.iloc[:0], baseline, candidate, 0.25, "max_cardinality_iou"
    )
    assert diagnostic["common_true_positives"] == 0
    assert diagnostic["rescued_xgboost_false_negatives"] == 1
    assert diagnostic["rescued_different_hand_events"] == 1
    assert diagnostic["xgboost_unique_true_positives"] == 1
    screen = module._component_screen(
        diagnostic,
        [diagnostic],
        {
            "minimum_unique_rescues": 1,
            "minimum_different_hand_rescues": 1,
            "minimum_partition_rescues": 1,
            "maximum_standalone_false_positives_per_rescue": 2,
        },
    )
    assert screen["passed_for_fusion_ablation"]
    assert screen["not_a_standalone_or_fusion_promotion_gate"]


def test_standalone_outer_evaluation_is_not_component_promotion(monkeypatch):
    monkeypatch.setattr(module, "_source", lambda *args: pytest.fail("read outer predictions"))
    with pytest.raises(RuntimeError, match="not a standalone promotion"):
        module.evaluate_legacy_frozen(Namespace(), {})
