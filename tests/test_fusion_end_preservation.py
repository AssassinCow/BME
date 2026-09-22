import pandas as pd
import pytest

from bme_eating.fusion_end_preservation import (
    apply_frozen_end_preservation,
    preserve_matched_ends,
)
from bme_eating.metrics import Match


def frame(rows):
    return pd.DataFrame(rows, columns=["subject_key", "session_id", "start_ms", "end_ms", "score"])


def test_empty_rescue_and_no_match_preserve_values():
    original = frame([("a", "s", 0, 100, 0.5)])
    empty, decisions = preserve_matched_ends(original, frame([]))
    assert empty.empty and decisions.empty
    rescued = frame([("a", "s", 200, 300, 0.9)])
    result, decisions = preserve_matched_ends(original, rescued)
    pd.testing.assert_frame_equal(result, rescued)
    assert decisions.reason.tolist() == ["no_iou_match"]


def test_unique_one_to_one_and_unchanged_start_score_count():
    original = frame([("a", "s", 0, 100, 0.4)])
    rescued = frame([("a", "s", 0, 60, 0.2), ("a", "s", 60, 110, 0.8)])
    result, decisions = preserve_matched_ends(original, rescued)
    assert len(result) == 2
    assert result.start_ms.tolist() == rescued.start_ms.tolist()
    assert result.score.tolist() == rescued.score.tolist()
    assert decisions.original_index.notna().sum() == 1
    assert "assignment_conflict" in decisions.reason.tolist()


def test_session_and_subject_isolation():
    original = frame([("a", "one", 0, 100, 0.7)])
    rescued = frame([("a", "two", 0, 90, 0.8), ("b", "one", 0, 90, 0.9)])
    result, decisions = preserve_matched_ends(original, rescued)
    pd.testing.assert_frame_equal(result, rescued)
    assert set(decisions.reason) == {"no_iou_match"}


def test_invalid_duration_and_overlap_are_skipped():
    invalid_original = frame([("a", "s", 0, 50, 0.6)])
    invalid_rescue = frame([("a", "s", 60, 90, 0.6)])
    invalid, decisions = preserve_matched_ends(invalid_original, invalid_rescue)
    assert invalid.end_ms.tolist() == [90]
    assert decisions.reason.tolist() == ["no_iou_match"]

    original = frame([("a", "s", 0, 100, 0.6)])
    rescued = frame([("a", "s", 0, 70, 0.6), ("a", "s", 80, 110, 0.8)])
    result, decisions = preserve_matched_ends(original, rescued)
    assert result.end_ms.tolist() == [70, 110]
    assert decisions.reason.iloc[0] == "would_overlap"


def test_zero_rescue_reproduces_original_bitwise():
    original = frame([("a", "s", 0, 100, 0.6), ("a", "s", 200, 300, 0.9)])
    result, decisions = preserve_matched_ends(original, original)
    pd.testing.assert_frame_equal(result, original)
    assert set(decisions.reason) == {"unchanged"}


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError, match="invalid intervals"):
        preserve_matched_ends(frame([("a", "s", 100, 100, 0.5)]), frame([]))


def test_invalid_copied_end_is_skipped(monkeypatch):
    import bme_eating.fusion_end_preservation as module

    monkeypatch.setattr(module, "match_events", lambda *args, **kwargs: [Match(0, 0, 0.5)])
    original = frame([("a", "s", 0, 50, 0.7)])
    rescued = frame([("a", "s", 60, 100, 0.6)])
    result, decisions = preserve_matched_ends(original, rescued)
    pd.testing.assert_frame_equal(result, rescued)
    assert decisions.reason.tolist() == ["invalid_duration"]


def test_inference_refuses_failed_gate_without_reading_predictions():
    with pytest.raises(RuntimeError, match="both development gates"):
        apply_frozen_end_preservation(
            pd.DataFrame(),
            pd.DataFrame(),
            {"promotion_allowed": False},
            {},
            {},
            inference_scope="outer",
        )


def test_outer_inference_rejects_training_subject_reuse():
    with pytest.raises(RuntimeError, match="overlap"):
        apply_frozen_end_preservation(
            pd.DataFrame(),
            pd.DataFrame({"subject_key": ["trained"]}),
            {"promotion_allowed": True, "deployment": {"training_subjects": ["trained"]}},
            {},
            {},
            inference_scope="outer",
        )
