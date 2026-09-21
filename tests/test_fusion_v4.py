import numpy as np
import pandas as pd
import pytest

from bme_eating import fusion_v4
from bme_eating.fusion_v4 import (
    PlattCalibrator,
    _tune_scope,
    evaluate_v4_gate,
    fit_platt_calibrator,
    fuse_gated_prediction_frames,
    paired_subject_bootstrap,
    run_meta_crossfit_selection,
)


def _predictions(subject="subject-a", probability=0.2):
    return pd.DataFrame(
        {
            "subject_key": subject,
            "segment_id": "segment-a",
            "session_id": "session-a",
            "timestamp_ms": [0, 3000, 6000, 9000, 12000],
            "state_probability": probability,
            "start_probability": [0.0, 1.0, 0.0, 0.0, 0.0],
            "end_probability": [0.0, 0.0, 0.0, 1.0, 0.0],
        }
    )


def _parameters(**overrides):
    return {
        "candidate_id": "test",
        "alpha_positive": 0.5,
        "alpha_negative": 0.25,
        "baseline_support_min": 0.05,
        "baseline_support_max": 0.5,
        "dtp_on_threshold": 0.55,
        "dtp_off_threshold": 0.2,
        "persistence_seconds": 12.0,
        "use_quality_weighting": False,
        **overrides,
    }


def _anchors(frame, targets):
    result = frame[["subject_key", "session_id", "timestamp_ms"]].copy()
    result["state_target"] = targets
    result["state_loss_mask"] = 1.0
    return result


def test_positive_slope_platt_calibration_is_finite_and_monotonic():
    dtp = pd.concat(
        [
            _predictions("subject-a", probability=0.1),
            _predictions("subject-b", probability=0.9),
        ],
        ignore_index=True,
    )
    anchors = _anchors(dtp, [0] * 5 + [1] * 5)

    calibrator = fit_platt_calibrator(dtp, anchors)

    assert calibrator.slope > 0
    assert np.isfinite([calibrator.slope, calibrator.intercept]).all()
    assert calibrator.training_subjects == 2


def test_platt_calibration_accepts_soft_window_coverage_targets():
    dtp = _predictions(probability=0.5)
    anchors = _anchors(dtp, [0.0, 0.25, 0.5, 0.75, 1.0])

    calibrator = fit_platt_calibrator(dtp, anchors)

    assert calibrator.slope > 0


def test_zero_gated_weights_exactly_reproduce_baseline():
    baseline = _predictions(probability=0.25)
    dtp = _predictions(probability=0.9)
    calibrator = PlattCalibrator(1.0, 0.0, 0.0, 5, 1)

    fused = fuse_gated_prediction_frames(
        baseline,
        dtp,
        calibrator,
        _parameters(alpha_positive=0.0, alpha_negative=0.0),
    )

    pd.testing.assert_frame_equal(fused, baseline)


def test_positive_gate_requires_causal_persistence():
    baseline = _predictions(probability=0.2)
    dtp = _predictions(probability=0.9)
    calibrator = PlattCalibrator(1.0, 0.0, 0.0, 5, 1)

    fused = fuse_gated_prediction_frames(
        baseline,
        dtp,
        calibrator,
        _parameters(alpha_negative=0.0),
    )

    np.testing.assert_allclose(fused["state_probability"].iloc[:3], 0.2)
    assert (fused["state_probability"].iloc[3:] > 0.2).all()
    np.testing.assert_array_equal(fused["start_probability"], baseline["start_probability"])
    np.testing.assert_array_equal(fused["end_probability"], baseline["end_probability"])


def test_negative_gate_can_suppress_persistent_high_confidence_disagreement():
    baseline = _predictions(probability=0.8)
    dtp = _predictions(probability=0.01)
    calibrator = PlattCalibrator(1.0, 0.0, 0.0, 5, 1)

    fused = fuse_gated_prediction_frames(
        baseline,
        dtp,
        calibrator,
        _parameters(alpha_positive=0.0, alpha_negative=0.25),
    )

    np.testing.assert_allclose(fused["state_probability"].iloc[:3], 0.8)
    assert (fused["state_probability"].iloc[3:] < 0.8).all()


def test_quality_weighting_requires_inference_diagnostics():
    with pytest.raises(ValueError, match="diagnostic columns"):
        fuse_gated_prediction_frames(
            _predictions(),
            _predictions(probability=0.9),
            PlattCalibrator(1.0, 0.0, 0.0, 5, 1),
            _parameters(use_quality_weighting=True),
        )


def test_v4_gate_checks_each_meta_partition():
    baseline = {
        "f1": 0.5,
        "different_sensitivity": 0.2,
        "strict_no_ignore_f1": 0.45,
        "false_positives_per_observed_hour": 1.0,
        "start_mae_seconds": 100.0,
        "end_mae_seconds": 100.0,
    }
    candidate = {
        **baseline,
        "f1": 0.53,
        "different_sensitivity": 0.24,
        "strict_no_ignore_f1": 0.46,
        "false_positives_per_observed_hour": 1.1,
        "start_mae_seconds": 105.0,
        "end_mae_seconds": 105.0,
    }
    config = {
        "minimum_f1_improvement": 0.02,
        "minimum_different_sensitivity_improvement": 0.03,
        "maximum_fp_per_hour_ratio": 1.2,
        "maximum_boundary_mae_ratio": 1.1,
        "maximum_partition_f1_drop": 0.01,
        "maximum_partition_strict_f1_drop": 0.01,
    }
    partitions = [
        {
            "candidate_metrics": {"f1": 0.49, "strict_no_ignore_f1": 0.44},
            "baseline_metrics": {"f1": 0.5, "strict_no_ignore_f1": 0.45},
        },
        {
            "candidate_metrics": {"f1": 0.6, "strict_no_ignore_f1": 0.5},
            "baseline_metrics": {"f1": 0.55, "strict_no_ignore_f1": 0.48},
        },
    ]

    assert evaluate_v4_gate(candidate, baseline, config, partitions)["passed"]
    partitions[0]["candidate_metrics"]["f1"] = 0.489
    gate = evaluate_v4_gate(candidate, baseline, config, partitions)
    assert not gate["passed"]
    assert not gate["checks"]["partition_f1_stability"]


def test_meta_crossfit_never_tunes_on_heldout_subjects(tmp_path, monkeypatch):
    baselines = []
    dtps = []
    anchors = []
    events = []
    subject_fold = {}
    for fold in range(3):
        subject = f"subject-{fold}"
        baseline = _predictions(subject, probability=0.8)
        dtp = _predictions(subject, probability=0.8)
        dtp["calibration_fold"] = fold
        baselines.append(baseline)
        dtps.append(dtp)
        anchors.append(_anchors(dtp, [0, 1, 1, 1, 0]))
        events.append(
            {
                "subject_key": subject,
                "start_ms": 3000,
                "end_ms": 12000,
                "event_id": f"event-{fold}",
                "hand_relation": "different",
                "coverage_ratio": 1.0,
                "valid_duration": True,
                "evaluable": True,
            }
        )
        subject_fold[subject] = fold
    observed_training_subjects = []
    postprocess = {
        "ema_half_life_seconds": 0.1,
        "high_threshold": 0.6,
        "low_threshold": 0.3,
        "minimum_event_seconds": 0,
        "merge_gap_seconds": 0,
        "boundary_lookback_seconds": 3,
        "iou_threshold": 0.25,
        "matching_method": "max_cardinality_iou",
    }

    def fake_tune_scope(baseline, dtp, *_args, **_kwargs):
        scope = _args[-2]
        subjects = set(baseline["subject_key"].astype(str).unique())
        observed_training_subjects.append((scope, subjects))
        return (
            {
                "scope": scope,
                "calibrator": PlattCalibrator(1.0, 0.0, 0.0, len(dtp), len(subjects)).as_dict(),
                "parameters": _parameters(alpha_positive=0.0, alpha_negative=0.0),
                "postprocess": postprocess,
                "postprocess_source": "frozen_hysteresis_control",
                "training_metrics": {},
                "training_baseline_metrics": {},
            },
            pd.DataFrame([{"scope": scope, "candidate_id": "baseline_identity"}]),
        )

    monkeypatch.setattr(fusion_v4, "_tune_scope", fake_tune_scope)
    config = {
        "crossfit_partitions": 3,
        "probability_epsilon": 1e-6,
        "residual_clip": 2.0,
        "gated_residual": {"gate_ema_half_life_seconds": 12},
        "meta_selection": {"bootstrap_replicates": 10, "bootstrap_seed": 2026},
        "promotion_gate": {
            "minimum_f1_improvement": 0.02,
            "minimum_different_sensitivity_improvement": 0.03,
            "maximum_fp_per_hour_ratio": 1.2,
            "maximum_boundary_mae_ratio": 1.1,
            "maximum_partition_f1_drop": 0.01,
            "maximum_partition_strict_f1_drop": 0.01,
        },
    }

    selection, predictions, _, _ = run_meta_crossfit_selection(
        pd.concat(baselines, ignore_index=True),
        pd.concat(dtps, ignore_index=True),
        pd.concat(anchors, ignore_index=True),
        pd.DataFrame(events),
        postprocess,
        config,
        tmp_path,
        workers=1,
    )

    for scope, training_subjects in observed_training_subjects[:3]:
        heldout = int(scope.rsplit("_", 1)[1])
        assert f"subject-{heldout}" not in training_subjects
    assert len(predictions) == 15
    assert selection["protocol_version"] == 4
    assert selection["meta_folds"][0]["training_scope"]["subjects"] == [
        "subject-1",
        "subject-2",
    ]
    assert selection["meta_folds"][0]["validation_scope"]["subjects"] == ["subject-0"]


def test_tune_scope_searches_three_fusion_candidates_beside_identity(tmp_path, monkeypatch):
    baseline = _predictions(probability=0.2)
    dtp = _predictions(probability=0.9)
    anchors = _anchors(dtp, [0.0, 0.25, 0.5, 0.75, 1.0])
    events = pd.DataFrame(
        [
            {
                "subject_key": "subject-a",
                "start_ms": 3000,
                "end_ms": 9000,
                "valid_duration": True,
                "evaluable": True,
            }
        ]
    )
    postprocess = {
        "ema_half_life_seconds": 1.0,
        "high_threshold": 0.6,
        "low_threshold": 0.3,
        "minimum_event_seconds": 0.0,
        "merge_gap_seconds": 0.0,
        "boundary_lookback_seconds": 3.0,
        "iou_threshold": 0.25,
        "matching_method": "max_cardinality_iou",
    }
    config = {
        "calibration": {"regularization": 0.0001, "maximum_slope": 20.0},
        "probability_epsilon": 1e-6,
        "residual_clip": 2.0,
        "gated_residual": {
            "alpha_positive_candidates": [0.25, 0.5],
            "alpha_negative_candidates": [0.0],
            "baseline_support_min_candidates": [0.05, 0.1],
            "baseline_support_max": 0.5,
            "dtp_on_threshold_candidates": [0.55],
            "dtp_off_threshold_candidates": [0.1],
            "persistence_seconds_candidates": [12],
            "gate_ema_half_life_seconds": 12,
            "quality_weighting_candidates": [False],
        },
        "meta_selection": {"top_fusion_candidates": 3, "show_progress": False},
        "postprocess_search": {
            "fast_ema_half_life_seconds": [6],
            "slow_ema_half_life_seconds": [24],
            "fast_high_threshold": [0.45],
            "slow_high_threshold": [0.2],
            "exit_threshold_ratio": [0.25],
            "off_duration_seconds": [18],
            "minimum_event_seconds": [15],
            "merge_gap_seconds": [30],
            "boundary_lookback_seconds": [30],
        },
    }
    dual_calls = []

    def fake_evaluate(predictions, *_args, **_kwargs):
        identity = np.array_equal(
            predictions["state_probability"].to_numpy(),
            baseline["state_probability"].to_numpy(),
        )
        score = 1.0 if identity else float(predictions["state_probability"].mean())
        return (
            {
                "f1": score,
                "strict_no_ignore_f1": score,
                "different_sensitivity": score,
                "false_positives_per_observed_hour": 0.0,
                "boundary_mae_seconds": 0.0,
                "start_mae_seconds": 0.0,
                "end_mae_seconds": 0.0,
            },
            pd.DataFrame(),
        )

    def fake_dual(predictions, *_args, **_kwargs):
        dual_calls.append(predictions.copy())
        selected = {
            "detector_mode": "dual_ema",
            "fast_ema_half_life_seconds": 6.0,
            "slow_ema_half_life_seconds": 24.0,
            "fast_high_threshold": 0.45,
            "slow_high_threshold": 0.2,
            "exit_threshold_ratio": 0.25,
            "off_duration_seconds": 18.0,
            "minimum_event_seconds": 15.0,
            "merge_gap_seconds": 30.0,
            "boundary_lookback_seconds": 30.0,
        }
        return selected, pd.DataFrame([selected])

    monkeypatch.setattr(fusion_v4, "evaluate_fusion_predictions", fake_evaluate)
    monkeypatch.setattr(fusion_v4, "tune_dual_ema_parameters", fake_dual)

    _tune_scope(
        baseline,
        dtp,
        anchors,
        events,
        postprocess,
        config,
        tmp_path,
        "test_scope",
        workers=1,
    )

    assert len(dual_calls) == 3
    assert all(
        not np.array_equal(
            frame["state_probability"].to_numpy(),
            baseline["state_probability"].to_numpy(),
        )
        for frame in dual_calls
    )


def test_paired_bootstrap_keeps_boundary_intervals_with_zero_hit_subject():
    baseline = pd.concat(
        [_predictions("subject-hit", 0.9), _predictions("subject-miss", 0.1)],
        ignore_index=True,
    )
    candidate = baseline.copy()
    truth = pd.DataFrame(
        [
            {
                "subject_key": subject,
                "start_ms": 3000,
                "end_ms": 9000,
                "hand_relation": "different",
            }
            for subject in ("subject-hit", "subject-miss")
        ]
    )
    events = pd.DataFrame(
        [
            {
                "subject_key": "subject-hit",
                "segment_id": "segment-a",
                "start_ms": 3000,
                "end_ms": 9000,
                "score": 0.9,
            }
        ]
    )
    ignore = pd.DataFrame(columns=["subject_key", "start_ms", "end_ms"])

    intervals = paired_subject_bootstrap(
        candidate,
        events,
        baseline,
        events,
        truth,
        ignore,
        iou_threshold=0.25,
        matching_method="max_cardinality_iou",
        replicates=20,
        seed=2026,
    )

    assert "start_mae_seconds" in intervals
    assert "end_mae_seconds" in intervals
