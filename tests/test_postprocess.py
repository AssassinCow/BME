import pandas as pd
import pytest

import bme_eating.postprocess as postprocess_module
from bme_eating.postprocess import (
    parameters_at_search_boundary,
    probabilities_to_events,
    tune_postprocess_parameters,
)


def test_hysteresis_creates_one_event():
    frame = pd.DataFrame(
        {
            "subject_key": ["s"] * 8,
            "segment_id": ["x"] * 8,
            "session_id": ["session"] * 8,
            "timestamp_ms": [3000 * index for index in range(1, 9)],
            "state_probability": [0.0, 0.1, 0.8, 0.9, 0.8, 0.2, 0.1, 0.0],
            "start_probability": [0.0, 0.1, 0.9, 0.2, 0.1, 0.0, 0.0, 0.0],
            "end_probability": [0.0, 0.0, 0.0, 0.1, 0.2, 0.9, 0.1, 0.0],
        }
    )
    events = probabilities_to_events(
        frame,
        ema_half_life_seconds=0.1,
        high_threshold=0.6,
        low_threshold=0.3,
        minimum_event_seconds=3,
        merge_gap_seconds=0,
        boundary_lookback_seconds=6,
    )
    assert len(events) == 1
    assert events.iloc[0].end_ms > events.iloc[0].start_ms


def test_duplicate_prediction_timestamps_are_collapsed():
    frame = pd.DataFrame(
        {
            "subject_key": ["s", "s", "s"],
            "segment_id": ["x", "x", "x"],
            "session_id": ["session", "session", "session"],
            "timestamp_ms": [0, 0, 3000],
            "state_probability": [0.8, 0.8, 0.0],
            "start_probability": [0.9, 0.7, 0.0],
            "end_probability": [0.0, 0.0, 0.9],
        }
    )
    events = probabilities_to_events(frame, 1, 0.6, 0.3, 0, 0, 3)
    assert len(events) == 1


def test_postprocess_rejects_nonfinite_probability():
    frame = pd.DataFrame(
        {
            "subject_key": ["s"],
            "segment_id": ["x"],
            "session_id": ["session"],
            "timestamp_ms": [0],
            "state_probability": [float("nan")],
            "start_probability": [0.0],
            "end_probability": [0.0],
        }
    )
    with pytest.raises(ValueError, match="finite"):
        probabilities_to_events(frame, 1, 0.6, 0.3, 0, 0, 3)


def test_postprocess_search_resumes_from_matching_checkpoint(tmp_path, monkeypatch):
    predictions = pd.DataFrame(
        {
            "subject_key": ["s"] * 4,
            "segment_id": ["x"] * 4,
            "session_id": ["session"] * 4,
            "timestamp_ms": [0, 3000, 6000, 9000],
            "state_probability": [0.0, 0.9, 0.8, 0.0],
            "start_probability": [0.0, 1.0, 0.0, 0.0],
            "end_probability": [0.0, 0.0, 0.0, 1.0],
        }
    )
    truth = pd.DataFrame(
        {"subject_key": ["s"], "start_ms": [3000], "end_ms": [9000]}
    )
    search = {
        "ema_half_life_seconds": [0.1],
        "high_threshold": [0.6, 0.7],
        "low_threshold": [0.3],
        "minimum_event_seconds": [0],
        "merge_gap_seconds": [0],
        "boundary_lookback_seconds": [3],
    }
    checkpoint = tmp_path / "postprocess.checkpoint.jsonl"
    calls = 0
    original = postprocess_module.evaluate_events

    def counted_evaluate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(postprocess_module, "evaluate_events", counted_evaluate)
    first_best, first_trials = tune_postprocess_parameters(
        predictions,
        truth,
        search,
        0.25,
        checkpoint_path=checkpoint,
        show_progress=False,
    )
    checkpoint_lines = checkpoint.read_text(encoding="utf-8").splitlines()
    checkpoint.write_text(
        "\n".join(checkpoint_lines[:2]) + '\n{"interrupted":', encoding="utf-8"
    )
    second_best, second_trials = tune_postprocess_parameters(
        predictions,
        truth,
        search,
        0.25,
        checkpoint_path=checkpoint,
        show_progress=False,
    )
    third_best, third_trials = tune_postprocess_parameters(
        predictions,
        truth,
        search,
        0.25,
        checkpoint_path=checkpoint,
        show_progress=False,
    )

    assert calls == 3
    assert first_best == second_best == third_best
    pd.testing.assert_frame_equal(first_trials, second_trials)
    pd.testing.assert_frame_equal(first_trials, third_trials)


def test_parallel_postprocess_search_matches_serial_results():
    predictions = pd.DataFrame(
        {
            "subject_key": ["s"] * 4,
            "session_id": ["session"] * 4,
            "timestamp_ms": [0, 3000, 6000, 9000],
            "state_probability": [0.0, 0.9, 0.8, 0.0],
            "start_probability": [0.0, 1.0, 0.0, 0.0],
            "end_probability": [0.0, 0.0, 0.0, 1.0],
        }
    )
    truth = pd.DataFrame(
        {"subject_key": ["s"], "start_ms": [3000], "end_ms": [9000]}
    )
    search = {
        "ema_half_life_seconds": [0.1],
        "high_threshold": [0.6, 0.7],
        "low_threshold": [0.3],
        "minimum_event_seconds": [0],
        "merge_gap_seconds": [0],
        "boundary_lookback_seconds": [3],
    }

    serial_best, serial_trials = tune_postprocess_parameters(
        predictions, truth, search, 0.25, show_progress=False, workers=1
    )
    parallel_best, parallel_trials = tune_postprocess_parameters(
        predictions, truth, search, 0.25, show_progress=False, workers=2
    )

    assert serial_best == parallel_best
    pd.testing.assert_frame_equal(serial_trials, parallel_trials)


def test_postprocess_merges_predictions_across_segments_in_one_session():
    frame = pd.DataFrame(
        {
            "subject_key": ["s"] * 4,
            "segment_id": ["a", "a", "b", "b"],
            "session_id": ["session"] * 4,
            "timestamp_ms": [3000, 6000, 9000, 12000],
            "state_probability": [0.8, 0.8, 0.8, 0.0],
            "start_probability": [1.0, 0.0, 0.0, 0.0],
            "end_probability": [0.0, 0.0, 0.0, 1.0],
        }
    )
    events = probabilities_to_events(frame, 0.1, 0.6, 0.3, 0, 0, 3)
    assert len(events) == 1
    assert events.iloc[0].session_id == "session"


def _dual_frame(values, sessions=None):
    sessions = sessions or ["session"] * len(values)
    return pd.DataFrame(
        {
            "subject_key": ["s"] * len(values),
            "segment_id": ["x"] * len(values),
            "session_id": sessions,
            "timestamp_ms": [index * 3000 for index in range(len(values))],
            "state_probability": values,
            "start_probability": values,
            "end_probability": [1.0 - value for value in values],
        }
    )


def _dual_events(frame, **overrides):
    parameters = {
        "fast_ema_half_life_seconds": 0.1,
        "slow_ema_half_life_seconds": 60.0,
        "fast_high_threshold": 0.6,
        "slow_high_threshold": 0.95,
        "exit_threshold_ratio": 0.5,
        "off_duration_seconds": 6,
        "minimum_event_seconds": 0,
        "merge_gap_seconds": 0,
        "boundary_lookback_seconds": 3,
    }
    parameters.update(overrides)
    return probabilities_to_events(frame, detector_mode="dual_ema", **parameters)


def test_dual_ema_rejects_one_spike_and_accepts_short_sustained_burst():
    assert _dual_events(_dual_frame([0.0, 0.9, 0.0, 0.0, 0.0])).empty
    assert _dual_events(_dual_frame([0.9, 0.0, 0.0, 0.0])).empty
    events = _dual_events(_dual_frame([0.0, 0.9, 0.9, 0.0, 0.0]))
    assert len(events) == 1


def test_dual_ema_requires_persistent_exit_and_resets_at_session_boundary():
    events = _dual_events(_dual_frame([0.8, 0.8, 0.0, 0.8, 0.0, 0.0]))
    assert len(events) == 1
    assert events.iloc[0].end_ms >= 12000
    split = _dual_frame(
        [0.8, 0.8, 0.8, 0.8], sessions=["first", "first", "second", "second"]
    )
    events = _dual_events(split)
    assert len(events) == 2
    assert set(events["session_id"]) == {"first", "second"}


def test_every_searched_postprocess_parameter_can_trigger_boundary_gate():
    trials = pd.DataFrame(
        {
            "high_threshold": [0.2, 0.4, 0.6],
            "low_threshold": [0.1, 0.2, 0.3],
            "merge_gap_seconds": [15, 30, 60],
            "f1": [0.1, 0.2, 0.3],
        }
    )
    flags = parameters_at_search_boundary(
        {"high_threshold": 0.4, "low_threshold": 0.1, "merge_gap_seconds": 60}, trials
    )
    assert flags == {
        "high_threshold": False,
        "low_threshold": True,
        "merge_gap_seconds": True,
    }

