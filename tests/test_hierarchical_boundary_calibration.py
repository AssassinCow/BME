from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from bme_eating.calibration import crossfit_calibrate_scores
from bme_eating.metrics import evaluate_events
from bme_eating.models.boundary_refiner import decode_boundaries, soft_offset_target


def _boundary_config() -> dict[str, object]:
    return {
        "start_range_seconds": 600,
        "end_range_seconds": 360,
        "coarse_bin_seconds": 30,
        "fine_range_seconds": 15,
        "safety_gap_seconds": 3,
    }


def test_soft_boundary_target_interpolates_between_bins() -> None:
    bins = np.asarray([-30.0, 0.0, 30.0], dtype=np.float32)
    target = soft_offset_target(15.0, bins)
    np.testing.assert_allclose(target, [0.0, 0.5, 0.5])


def test_high_entropy_boundary_falls_back_without_changing_identity() -> None:
    proposals = pd.DataFrame(
        [
            {
                "proposal_id": "a",
                "subject_key": "s",
                "session_id": "x",
                "coarse_start_ms": 100_000,
                "coarse_end_ms": 200_000,
            }
        ]
    )
    output = {
        "start_distribution_logit": torch.zeros(1, 41),
        "end_distribution_logit": torch.zeros(1, 25),
        "start_fine_seconds": torch.zeros(1),
        "end_fine_seconds": torch.zeros(1),
    }
    refined = decode_boundaries(
        proposals,
        output,
        _boundary_config(),
        0.55,
        {("s", "x"): 260_000},
    )
    assert refined.loc[0, "proposal_id"] == "a"
    assert refined.loc[0, "refined_start_ms"] == 100_000
    assert refined.loc[0, "refined_end_ms"] == 200_000
    assert bool(refined.loc[0, "boundary_fallback"])


def test_score_calibration_is_subject_fold_crossfit() -> None:
    rows = []
    for fold in range(3):
        for index in range(10):
            positive = index % 2 == 0
            rows.append(
                {
                    "proposal_id": f"{fold}-{index}",
                    "calibration_fold": fold,
                    "event_logit": 2.0 if positive else -2.0,
                    "iou_logit": 1.5 if positive else -1.5,
                    "state_score": 0.8 if positive else 0.2,
                    "is_positive": positive,
                    "max_iou": 0.8 if positive else 0.05,
                }
            )
    calibrated, bundle = crossfit_calibrate_scores(pd.DataFrame(rows))
    assert calibrated["proposal_id"].is_unique
    assert calibrated["final_score"].between(0, 1).all()
    assert calibrated["calibrated_state_probability"].between(0, 1).all()
    assert bundle.event.temperature > 0
    assert bundle.state.temperature > 0


def test_future_end_offsets_are_masked_and_clamped() -> None:
    proposals = pd.DataFrame(
        [
            {
                "proposal_id": "a",
                "subject_key": "s",
                "session_id": "x",
                "coarse_start_ms": 100_000,
                "coarse_end_ms": 200_000,
            }
        ]
    )
    output = {
        "start_distribution_logit": torch.full((1, 41), -20.0),
        "end_distribution_logit": torch.full((1, 25), -20.0),
        "start_fine_seconds": torch.zeros(1),
        "end_fine_seconds": torch.full((1,), 15.0),
    }
    output["start_distribution_logit"][0, 20] = 20.0
    output["end_distribution_logit"][0, -1] = 20.0
    refined = decode_boundaries(
        proposals,
        output,
        _boundary_config(),
        1.0,
        {("s", "x"): 230_000},
    )
    assert refined.loc[0, "refined_end_ms"] <= 230_000


def test_event_matches_preserve_prediction_identity_for_boundary_gates() -> None:
    truth = pd.DataFrame(
        [{"event_id": "truth", "subject_key": "s", "start_ms": 0, "end_ms": 1000}]
    )
    prediction = pd.DataFrame(
        [
            {
                "proposal_id": "proposal",
                "subject_key": "s",
                "start_ms": 0,
                "end_ms": 1000,
            }
        ]
    )

    _, matches = evaluate_events(truth, prediction, iou_threshold=0.25)

    assert matches.loc[0, "prediction_event_id"] == "proposal"
