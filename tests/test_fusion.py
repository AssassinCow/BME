import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

import bme_eating.fusion as fusion_module
from bme_eating.cli import _require_prior_fusion_folds
from bme_eating.fusion import (
    FROZEN_BASELINE_FILES,
    FusionValidator,
    align_prediction_frames,
    assemble_crossfit_predictions,
    average_prediction_frames,
    evaluate_fold0_gate,
    evaluate_internal_gate,
    fuse_prediction_frames,
    prepare_fusion_run_root,
    sha256_file,
    validate_clean_baseline_experiment,
    validate_frozen_baseline_fold,
    validate_fusion_run_name,
)


def _predictions(
    probabilities=(0.4, 0.4, 0.4, 0.4, 0.4),
    *,
    subject="subject-a",
    session="session-a",
):
    return pd.DataFrame(
        {
            "subject_key": subject,
            "segment_id": ["segment-a", "segment-a", "segment-b", "segment-b", "segment-b"],
            "session_id": session,
            "timestamp_ms": [0, 3000, 6000, 9000, 12000],
            "state_probability": probabilities,
            "start_probability": [0.0, 0.7, 0.0, 0.0, 0.0],
            "end_probability": [0.0, 0.0, 0.0, 0.8, 0.0],
        }
    )


POSTPROCESS = {
    "ema_half_life_seconds": 0.1,
    "high_threshold": 0.6,
    "low_threshold": 0.3,
    "minimum_event_seconds": 0,
    "merge_gap_seconds": 0,
    "boundary_lookback_seconds": 3,
    "iou_threshold": 0.25,
    "matching_method": "max_cardinality_iou",
}


def _truth(subject="subject-a"):
    return pd.DataFrame(
        {
            "subject_key": [subject],
            "start_ms": [3000],
            "end_ms": [12000],
            "event_id": ["event-a"],
            "hand_relation": ["different"],
        }
    )


def _ignore():
    return pd.DataFrame(columns=["subject_key", "start_ms", "end_ms"])


def test_zero_weight_fusion_exactly_reproduces_baseline():
    baseline = _predictions((0.0, 0.25, 0.5, 0.75, 1.0))
    dtp = _predictions((1.0, 0.75, 0.5, 0.25, 0.0))

    fused = fuse_prediction_frames(baseline, dtp, alpha=0.0, beta=0.0)

    pd.testing.assert_frame_equal(fused, baseline)


def test_fusion_run_name_requires_an_isolated_safe_suffix():
    assert (
        validate_fusion_run_name(
            "baseline_dtp_fusion", "baseline_dtp_fusion_clean_20260920a"
        )
        == "baseline_dtp_fusion_clean_20260920a"
    )
    with pytest.raises(ValueError, match="must start"):
        validate_fusion_run_name("baseline_dtp_fusion", "../shared")
    with pytest.raises(ValueError, match="must remain"):
        validate_fusion_run_name("changed", None)


def test_fresh_fusion_run_never_reuses_an_existing_directory(tmp_path):
    experiments = tmp_path / "experiments"
    run_name = "baseline_dtp_fusion_clean_20260920a"

    run_root = prepare_fusion_run_root(
        experiments, run_name, fresh=True, named_run=True
    )

    assert run_root.is_dir()
    with pytest.raises(RuntimeError, match="will not be reused"):
        prepare_fusion_run_root(experiments, run_name, fresh=True, named_run=True)
    with pytest.raises(FileNotFoundError, match="start fold 0"):
        prepare_fusion_run_root(
            experiments,
            "baseline_dtp_fusion_missing",
            fresh=False,
            named_run=True,
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_fusion_rejects_nonfinite_probabilities(value):
    baseline = _predictions()
    baseline.loc[0, "state_probability"] = value

    with pytest.raises(ValueError, match="NaN or infinite"):
        fuse_prediction_frames(baseline, _predictions(), alpha=0.25, beta=0.0)


def test_fusion_rejects_duplicate_missing_and_session_mismatched_keys():
    baseline = _predictions()
    duplicated = pd.concat([_predictions(), _predictions().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        align_prediction_frames(baseline, duplicated)

    with pytest.raises(ValueError, match="not one-to-one"):
        align_prediction_frames(baseline, _predictions().iloc[:-1])

    wrong_session = _predictions(session="session-b")
    with pytest.raises(ValueError, match="session_id"):
        align_prediction_frames(baseline, wrong_session)


def test_residual_clip_bounds_logit_change_and_preserves_baseline_boundaries():
    baseline = _predictions((0.4,) * 5)
    dtp = _predictions((1.0,) * 5)

    fused = fuse_prediction_frames(baseline, dtp, alpha=0.5, beta=0.0, residual_clip=2.0)

    base_logit = math.log(0.4 / 0.6)
    fused_logit = np.log(
        fused["state_probability"].to_numpy() / (1.0 - fused["state_probability"].to_numpy())
    )
    assert np.all(fused_logit - base_logit <= 1.0 + 1e-6)
    np.testing.assert_array_equal(fused["start_probability"], baseline["start_probability"])
    np.testing.assert_array_equal(fused["end_probability"], baseline["end_probability"])


def test_validator_evaluates_exactly_nine_fixed_combinations(tmp_path):
    baseline = _predictions()
    validator = FusionValidator(
        baseline=baseline,
        truth=_truth(),
        ignore=_ignore(),
        postprocess=POSTPROCESS,
        fusion_config={
            "alpha_candidates": [0.0, 0.25, 0.5],
            "beta_candidates": [-0.25, 0.0, 0.25],
            "residual_clip": 2.0,
            "probability_epsilon": 1e-6,
        },
        output_dir=tmp_path,
        forbidden_subjects={"outer-subject"},
    )

    selection = validator(_predictions((0.99, 0.99, 0.99, 0.01, 0.01)), epoch=2)

    trials = pd.read_csv(tmp_path / "fusion_trials.csv")
    assert len(trials) == 9
    assert set(trials["alpha"]) == {0.0, 0.25, 0.5}
    assert set(trials["beta"]) == {-0.25, 0.0, 0.25}
    assert selection["epoch"] == 2
    assert selection["baseline_metrics"]["f1"] == 0.0
    assert selection["beta_only_beta"] in {-0.25, 0.0, 0.25}
    assert "f1" in selection["beta_only_metrics"]
    assert selection["metrics"]["f1"] == 1.0
    assert selection["alpha"] == 0.5
    assert selection["metrics"]["predicted_events"] == 1.0

    repeated = validator(_predictions((0.99, 0.99, 0.99, 0.01, 0.01)), epoch=2)
    repeated_trials = pd.read_csv(tmp_path / "fusion_trials.csv")
    assert len(repeated_trials) == 9
    assert repeated["alpha"] == selection["alpha"]
    assert repeated["beta"] == selection["beta"]


def test_validator_rejects_reference_event_mismatch(tmp_path, monkeypatch):
    original = fusion_module.evaluate_fusion_predictions
    call_count = 0

    def altered_evaluation(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        metrics, events = original(*args, **kwargs)
        if call_count == 2:
            events = events.copy()
            events.loc[events.index[0], "start_ms"] += 3000
        return metrics, events

    monkeypatch.setattr(fusion_module, "evaluate_fusion_predictions", altered_evaluation)
    baseline = _predictions((0.99, 0.99, 0.99, 0.01, 0.01))
    validator = FusionValidator(
        baseline=baseline,
        truth=_truth(),
        ignore=_ignore(),
        postprocess=POSTPROCESS,
        fusion_config={
            "alpha_candidates": [0.0],
            "beta_candidates": [0.0],
            "residual_clip": 2.0,
            "probability_epsilon": 1e-6,
        },
        output_dir=tmp_path,
        forbidden_subjects=set(),
    )

    with pytest.raises(RuntimeError, match="baseline events"):
        validator(baseline, epoch=0)


def test_validator_rejects_outer_subject_leakage(tmp_path):
    with pytest.raises(ValueError, match="outer-fold"):
        FusionValidator(
            baseline=_predictions(subject="held-out"),
            truth=_truth(subject="held-out"),
            ignore=_ignore(),
            postprocess=POSTPROCESS,
            fusion_config={
                "alpha_candidates": [0.0, 0.25, 0.5],
                "beta_candidates": [-0.25, 0.0, 0.25],
                "residual_clip": 2.0,
                "probability_epsilon": 1e-6,
            },
            output_dir=tmp_path,
            forbidden_subjects={"held-out"},
        )


def test_internal_and_fold0_gates_are_deterministic():
    baseline = {
        "f1": 0.50,
        "different_sensitivity": 0.10,
        "strict_no_ignore_f1": 0.45,
        "false_positives_per_observed_hour": 0.02,
    }
    selection = {
        "alpha": 0.25,
        "metrics": {
            "f1": 0.52,
            "different_sensitivity": 0.14,
            "strict_no_ignore_f1": 0.45,
            "false_positives_per_observed_hour": 0.02,
        },
        "baseline_metrics": baseline,
        "beta_only_metrics": baseline,
    }
    internal = evaluate_internal_gate(
        selection,
        {
            "minimum_f1_improvement": 0.015,
            "minimum_f1_improvement_over_beta_only": 0.005,
            "minimum_different_sensitivity_improvement": 0.03,
            "maximum_fp_per_hour_ratio": 1.2,
        },
    )
    assert internal["passed"]

    candidate = {
        **selection["metrics"],
        "sensitivity": 0.6,
        "same_sensitivity": 0.7,
        "start_mae_seconds": 105.0,
        "end_mae_seconds": 55.0,
    }
    baseline_fold = {
        **baseline,
        "sensitivity": 0.5,
        "same_sensitivity": 0.7,
        "start_mae_seconds": 100.0,
        "end_mae_seconds": 50.0,
    }
    outer = evaluate_fold0_gate(
        candidate,
        baseline_fold,
        {
            "minimum_f1_improvement": 0.02,
            "maximum_fp_per_hour_ratio": 1.2,
            "maximum_boundary_mae_ratio": 1.1,
        },
    )
    assert outer["passed"]


def test_internal_gate_rejects_alpha_candidate_whose_gain_is_only_beta():
    baseline = {
        "f1": 0.40,
        "different_sensitivity": 0.10,
        "strict_no_ignore_f1": 0.40,
        "false_positives_per_observed_hour": 0.02,
    }
    beta_only = {
        "f1": 0.52,
        "different_sensitivity": 0.15,
        "strict_no_ignore_f1": 0.45,
        "false_positives_per_observed_hour": 0.021,
    }
    selection = {
        "alpha": 0.25,
        "metrics": dict(beta_only),
        "baseline_metrics": baseline,
        "beta_only_metrics": beta_only,
    }

    gate = evaluate_internal_gate(
        selection,
        {
            "minimum_f1_improvement": 0.015,
            "minimum_f1_improvement_over_beta_only": 0.005,
            "minimum_different_sensitivity_improvement": 0.03,
            "maximum_fp_per_hour_ratio": 1.2,
        },
    )

    assert not gate["passed"]
    assert not gate["checks"]["f1_improvement_over_beta_only"]


def test_crossfit_oof_is_disjoint_and_test_predictions_are_mean_ensemble():
    validation_frames = {
        partition: _predictions(
            tuple([0.1 + 0.1 * partition] * 5),
            subject=f"subject-{partition}",
            session=f"session-{partition}",
        )
        for partition in range(3)
    }
    test_frames = {
        partition: _predictions(
            tuple([0.2 + 0.2 * partition] * 5), subject="held-out"
        )
        for partition in range(3)
    }

    oof, ensemble = assemble_crossfit_predictions(
        validation_frames,
        test_frames,
        {partition: {f"subject-{partition}"} for partition in range(3)},
        {"held-out"},
    )

    assert set(oof["subject_key"]) == {"subject-0", "subject-1", "subject-2"}
    assert set(oof.groupby("subject_key")["calibration_fold"].first()) == {0, 1, 2}
    np.testing.assert_allclose(ensemble["state_probability"], 0.4)
    pd.testing.assert_frame_equal(
        ensemble,
        average_prediction_frames([test_frames[index] for index in range(3)]),
    )


def test_crossfit_rejects_subject_leakage_between_partitions():
    validation_frames = {partition: _predictions(subject="same") for partition in range(3)}
    test_frames = {partition: _predictions(subject="held-out") for partition in range(3)}

    with pytest.raises(RuntimeError, match="more than one partition"):
        assemble_crossfit_predictions(
            validation_frames,
            test_frames,
            {partition: {"same"} for partition in range(3)},
            {"held-out"},
        )


def test_failed_outer_diagnostics_do_not_block_later_folds(tmp_path):
    experiment_name = "baseline_dtp_fusion_clean_test"
    fold_dir = tmp_path / "experiments" / experiment_name / "fold_0"
    fold_dir.mkdir(parents=True)
    (fold_dir / "test_metrics.json").write_text(
        json.dumps({"primary_method": "max_cardinality_iou"}), encoding="utf-8"
    )
    selection_path = fold_dir / "selected_fusion.json"
    selection_path.write_text(
        json.dumps(
            {
                "version": 3,
                "run_name": experiment_name,
                "crossfit_partitions": 3,
                "crossfit_models": [
                    {"selection_signature": str(partition) * 64}
                    for partition in (1, 2, 3)
                ],
                "internal_gate": {"passed": False},
                "outer_fold_gate": {"passed": False, "diagnostic_only": True},
            }
        ),
        encoding="utf-8",
    )
    metrics_path = fold_dir / "test_metrics.json"
    (fold_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment": {"name": experiment_name, "fold": 0},
                "git": {"commit": "abc1234"},
                "resolved_config_sha256": "config-hash",
                "artifact_hashes": {
                    "test_metrics.json": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                    "selected_fusion.json": hashlib.sha256(
                        selection_path.read_bytes()
                    ).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    _require_prior_fusion_folds(tmp_path, experiment_name, requested_fold=1)


def test_frozen_baseline_gate_detects_artifact_changes(tmp_path):
    output_root = tmp_path / "v2"
    fold_dir = output_root / "experiments" / "baseline" / "fold_0"
    indices = output_root / "indices"
    fold_dir.mkdir(parents=True)
    indices.mkdir(parents=True)
    for name in FROZEN_BASELINE_FILES:
        if name != "run_manifest.json":
            (fold_dir / name).write_bytes(name.encode())
    tracked = {
        "quality_report": indices / "quality_report.json",
        "quality_expectations": indices / "quality_expectations.json",
        "subject_folds": indices / "subject_folds.json",
        "subject_folds_manifest": indices / "subject_folds.manifest.json",
        "events": indices / "events.parquet",
        "anchors": indices / "anchors.parquet",
        "segments": indices / "segments.parquet",
    }
    for name, path in tracked.items():
        path.write_bytes(name.encode())
    artifact_hashes = {
        name: sha256_file(fold_dir / name)
        for name in FROZEN_BASELINE_FILES
        if name != "run_manifest.json"
    }
    manifest = {
        "git": {"commit": "7ca651a"},
        "artifact_provenance": {
            "claimed_source_commit": "3ca55bb",
            "verification": "team-declared",
        },
        "hashes": {name: sha256_file(path) for name, path in tracked.items()},
        "artifact_hashes": artifact_hashes,
        "resolved_config_sha256": "config-hash",
    }
    (fold_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    validate_frozen_baseline_fold(output_root, "baseline", 0, "3ca55bb")
    with pytest.raises(RuntimeError, match="backfilled provenance"):
        validate_frozen_baseline_fold(
            output_root, "baseline", 0, "3ca55bb", require_clean=True
        )

    manifest.pop("artifact_provenance")
    manifest["git"] = {"commit": "3ca55bb", "dirty": False}
    (fold_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    validate_frozen_baseline_fold(
        output_root, "baseline", 0, "3ca55bb", require_clean=True
    )
    clean = validate_clean_baseline_experiment(
        output_root, "baseline", "3ca55bb", number_of_folds=1
    )
    assert set(clean) == {0}
    (fold_dir / "test_predictions.parquet").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="artifact hash changed"):
        validate_frozen_baseline_fold(
            output_root, "baseline", 0, "3ca55bb", require_clean=True
        )
