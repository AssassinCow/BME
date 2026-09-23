from __future__ import annotations

import numpy as np
import pandas as pd

from bme_eating.proposals import (
    SOURCE_HINT,
    SOURCE_STATE,
    SOURCE_XGBOOST,
    generate_event_candidates,
    label_event_candidates,
)


def _config() -> dict[str, object]:
    return {
        "high_threshold": 0.6,
        "low_threshold": 0.3,
        "hint_threshold": 0.7,
        "jitter_seconds": [-30, -15, 0, 15, 30],
        "maximum_variants_per_event": 12,
        "maximum_candidates_per_hour": 20,
        "deduplication_iou": 0.9,
    }


def _predictions() -> pd.DataFrame:
    timestamps = np.arange(0, 600_000, 3000, dtype=np.int64)
    state = np.full(len(timestamps), 0.05, dtype=np.float32)
    state[30:90] = 0.8
    start = np.zeros(len(timestamps), dtype=np.float32)
    end = np.zeros(len(timestamps), dtype=np.float32)
    start[30] = 0.9
    end[90] = 0.9
    return pd.DataFrame(
        {
            "subject_key": "s1",
            "session_id": "session",
            "timestamp_ms": timestamps,
            "state_probability": state,
            "start_probability": start,
            "end_probability": end,
        }
    )


def test_candidates_are_deterministic_deduplicated_and_budgeted() -> None:
    first = generate_event_candidates(
        _predictions(), _config(), (15.0, 600.0), split_role="oof"
    )
    second = generate_event_candidates(
        _predictions(), _config(), (15.0, 600.0), split_role="oof"
    )
    pd.testing.assert_frame_equal(first, second)
    assert first["proposal_id"].is_unique
    assert len(first) <= 4
    assert set(first["split_role"]) == {"oof"}


def test_exact_iou_threshold_is_not_positive() -> None:
    proposals = pd.DataFrame(
        [
            {
                "proposal_id": "p",
                "subject_key": "s",
                "session_id": "x",
                "coarse_start_ms": 0,
                "coarse_end_ms": 100_000,
                "source_mask": 1,
                "generator_score": 1.0,
                "rank_within_session": 0,
                "split_role": "oof",
            }
        ]
    )
    events = pd.DataFrame(
        [{"event_id": "e", "subject_key": "s", "start_ms": 0, "end_ms": 25_000}]
    )
    labeled = label_event_candidates(proposals, events, 0.25)
    assert labeled.loc[0, "max_iou"] == 0.25
    assert not bool(labeled.loc[0, "is_positive"])


def test_empty_candidate_labels_preserve_schema() -> None:
    proposals = generate_event_candidates(
        _predictions().iloc[:0], _config(), (15.0, 600.0), split_role="oof"
    )
    labeled = label_event_candidates(proposals, pd.DataFrame(), 0.25)
    assert labeled.empty
    assert {"max_iou", "matched_event_id", "is_positive", "negative_type"} <= set(
        labeled.columns
    )


def test_budget_retains_each_available_primary_source() -> None:
    config = _config()
    config["maximum_candidates_per_hour"] = 20
    xgb = pd.DataFrame(
        [
            {
                "subject_key": "s1",
                "session_id": "session",
                "start_ms": 120_000,
                "end_ms": 300_000,
                "score": 0.95,
            }
        ]
    )
    proposals = generate_event_candidates(
        _predictions(), config, (15.0, 600.0), split_role="oof", xgb_events=xgb
    )
    combined_mask = int(np.bitwise_or.reduce(proposals["source_mask"].to_numpy(dtype=int)))
    assert combined_mask & SOURCE_STATE
    assert combined_mask & SOURCE_HINT
    assert combined_mask & SOURCE_XGBOOST
