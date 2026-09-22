import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from bme_eating.dtp_postprocess import (
    SERIALIZATION_FAILURE_SIGNATURE,
    Decoder,
    _crossfit,
    _event_diagnostics,
    _legacy_control,
    _metrics,
    _observed_hours,
    _save_json,
    _search_scope,
    _validate_selection,
    _validate_serialization_recovery,
    correct_boundaries,
    evaluate_dtp_postprocess,
    event_gate,
    filter_and_merge,
    fit_background_thresholds,
    generate_core_events,
    module_gate,
    weighted_quantile,
)
from bme_eating.fusion import json_safe, sha256_file
from bme_eating.fusion_v4 import PlattCalibrator, fuse_gated_prediction_frames
from bme_eating.postprocess import probabilities_to_events


def _predictions(values, *, subject="one", session="first", offset=0):
    return pd.DataFrame(
        {
            "subject_key": [subject] * len(values),
            "session_id": [session] * len(values),
            "segment_id": ["segment"] * len(values),
            "timestamp_ms": np.arange(len(values)) * 3000 + offset,
            "state_probability": values,
            "start_probability": [0.0] * len(values),
            "end_probability": [0.0] * len(values),
        }
    )


def _decoder(mode="fast_persistent", persistence=6):
    return Decoder(
        "raw_control", 0.1, 1000, mode, 0.9, 0.9, persistence, exit_ratio=0.5, off_duration=6
    )


def test_json_safe_recursively_converts_numpy_boolean(tmp_path):
    payload = {"nested": [np.bool_(True), {"flag": np.bool_(False)}]}
    assert json_safe(payload) == {"nested": [True, {"flag": False}]}
    output = tmp_path / "payload.json"
    _save_json(output, payload)
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "nested": [True, {"flag": False}]
    }


def test_serialization_recovery_requires_exact_signature_and_complete_checkpoints(tmp_path):
    (tmp_path / "search_trials.csv").write_text("scope,key\n", encoding="utf-8")
    (tmp_path / "meta_oof_metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "module_ablations.csv").write_text("", encoding="utf-8")
    for fold in range(3):
        (tmp_path / f"search.heldout_{fold}.checkpoint.jsonl").write_text(
            json.dumps({"scope": f"heldout_{fold}", "key": "one"}) + "\n",
            encoding="utf-8",
        )
    _validate_serialization_recovery(tmp_path, SERIALIZATION_FAILURE_SIGNATURE)
    with pytest.raises(RuntimeError, match="Cannot resume"):
        _validate_serialization_recovery(tmp_path, "wrong-signature")


def test_serialization_recovery_rejects_corrupt_checkpoint(tmp_path):
    (tmp_path / "search_trials.csv").write_text("scope,key\n", encoding="utf-8")
    (tmp_path / "meta_oof_metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "module_ablations.csv").write_text("", encoding="utf-8")
    for fold in range(3):
        text = "{" if fold == 1 else json.dumps({"scope": "x", "key": "y"})
        (tmp_path / f"search.heldout_{fold}.checkpoint.jsonl").write_text(text, encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint is corrupt"):
        _validate_serialization_recovery(tmp_path, SERIALIZATION_FAILURE_SIGNATURE)


def test_subject_equal_weight_quantile():
    values = np.asarray([0.1] * 100 + [0.9])
    subjects = np.asarray(["long"] * 100 + ["short"])
    assert weighted_quantile(values, subjects, 0.75) == pytest.approx(0.9)
    assert weighted_quantile(values, subjects, 1.0) == pytest.approx(0.9)
    with pytest.raises(ValueError, match="training values"):
        weighted_quantile(np.asarray([]), np.asarray([]), 0.5)


def test_legacy_control_reproduces_existing_decoder():
    predictions = _predictions([0.0] * 10 + [0.9] * 30 + [0.0] * 40)
    original = probabilities_to_events(
        predictions,
        detector_mode="dual_ema",
        fast_ema_half_life_seconds=12,
        slow_ema_half_life_seconds=36,
        fast_high_threshold=0.55,
        slow_high_threshold=0.30,
        exit_threshold_ratio=0.25,
        off_duration_seconds=60,
        minimum_event_seconds=30,
        merge_gap_seconds=60,
        boundary_lookback_seconds=60,
    )
    pd.testing.assert_frame_equal(_legacy_control(predictions, {}), original)


def test_slow_channel_cannot_start_persistent_fast_event():
    predictions = _predictions([0.6] * 20)
    limits = {"fast": 0.95, "slow": 0.01}
    assert generate_core_events(predictions, _decoder(), limits).empty
    assert not generate_core_events(predictions, _decoder("legacy_or"), limits).empty


def test_persistence_is_causal_and_gap_resets():
    beginning = _predictions([1.0] * 2, offset=0)
    tail = _predictions([1.0] * 10, offset=15000)
    predictions = pd.concat([beginning, tail], ignore_index=True)
    decoder = _decoder(persistence=9)
    events = generate_core_events(
        predictions, decoder, {"fast": 0.5, "slow": 0.2}, minimum_seconds=9
    )
    assert len(events) == 1
    assert events.iloc[0].start_ms == 15000
    assert events.iloc[0].end_ms == 42000
    assert generate_core_events(
        beginning, decoder, {"fast": 0.5, "slow": 0.2}, minimum_seconds=0
    ).empty


def test_session_boundary_and_missing_background_fail_closed():
    first = _predictions([0.9] * 2)
    second = _predictions([0.9] * 2, session="second", offset=6000)
    predictions = pd.concat([first, second], ignore_index=True)
    assert generate_core_events(
        predictions,
        _decoder(persistence=9),
        {"fast": 0.5, "slow": 0.2},
        minimum_seconds=0,
    ).empty
    labels = first[["subject_key", "session_id", "timestamp_ms"]].copy()
    labels["state_target"] = 1
    labels["state_loss_mask"] = 1
    with pytest.raises(ValueError, match="No eligible negative"):
        fit_background_thresholds(first, labels, _decoder(), 3)


def test_gap_cannot_be_rejoined_by_merge_or_boundary_head():
    first = _predictions([1.0] * 12)
    second = _predictions([1.0] * 12, offset=45000)
    predictions = pd.concat([first, second], ignore_index=True)
    core = generate_core_events(
        predictions,
        _decoder(),
        {"fast": 0.5, "slow": 0.2},
        minimum_seconds=0,
    )
    assert len(core) == 2
    assert len(filter_and_merge(core, None, 60)) == 2
    spanning = pd.DataFrame(
        [("one", "first", 0, 78000, 0.7)],
        columns=["subject_key", "session_id", "start_ms", "end_ms", "score"],
    )
    pd.testing.assert_frame_equal(correct_boundaries(predictions, spanning), spanning)


def test_filter_precedes_merge_and_score_is_duration_weighted():
    core = pd.DataFrame(
        [
            ("person", "s", 0, 30_000, 0.9),
            ("person", "s", 45_000, 75_000, 0.1),
            ("person", "s", 90_000, 150_000, 0.5),
        ],
        columns=["subject_key", "session_id", "start_ms", "end_ms", "score"],
    )
    filtered = filter_and_merge(core, 0.6, 60)
    assert len(filtered) == 1
    assert filtered.iloc[0].end_ms == 30_000
    merged = filter_and_merge(core.iloc[[0, 2]], None, 60)
    assert merged.iloc[0].score == pytest.approx((0.9 * 30 + 0.5 * 60) / 90)
    assert len(filter_and_merge(core.iloc[[0, 2]], None, 0)) == 2


def test_event_gate_uses_train_subject_equal_weights():
    predictions = _predictions([0.9] * 3)
    events = pd.DataFrame(
        [("one", "first", 0, 6000, 0.5)],
        columns=["subject_key", "session_id", "start_ms", "end_ms", "score"],
    )
    reference = pd.DataFrame(
        {
            "subject_key": ["long"] * 10 + ["short"],
            "score": [0.1] * 10 + [0.9],
        }
    )
    gate = event_gate(predictions, events, reference)
    assert gate.event_gate.tolist() == pytest.approx([0.5] * 3)
    assert (event_gate(predictions, events.iloc[:0], reference).event_gate == 0).all()


def test_cached_exposure_preserves_event_metrics():
    predictions = _predictions([0.1] * 10)
    events = pd.DataFrame(
        [("one", "first", 0, 15000, 0.9)],
        columns=["subject_key", "session_id", "start_ms", "end_ms", "score"],
    )
    truth = events.assign(hand_relation="different")
    ignore = events.iloc[:0]
    regular = _metrics(predictions, events, truth, ignore, 0.25, "max_cardinality_iou")
    cached = _metrics(
        predictions,
        events,
        truth,
        ignore,
        0.25,
        "max_cardinality_iou",
        observed_hours=_observed_hours(predictions),
    )
    assert regular.keys() == cached.keys()
    for name in regular:
        assert np.isclose(regular[name], cached[name], equal_nan=True), name
    failures, by_subject, by_hand = _event_diagnostics(
        predictions, events, truth, ignore, 0.25, "max_cardinality_iou"
    )
    assert failures.empty
    assert by_subject.iloc[0].f1 == pytest.approx(1.0)
    assert by_hand["different"]["sensitivity"] == pytest.approx(1.0)


def test_module_gate_rejects_fold_degradation():
    config = {
        "module_minimum_f1_improvement": 0.01,
        "module_minimum_fp_reduction": 0.15,
        "module_minimum_large_f1_improvement": 0.02,
        "module_maximum_different_sensitivity_drop": 0.1,
        "module_maximum_strict_f1_drop": 0.01,
        "module_maximum_partition_f1_drop": 0.02,
    }
    original = {
        "f1": 0.4,
        "false_positives_per_observed_hour": 0.3,
        "different_sensitivity": 0.5,
        "strict_no_ignore_f1": 0.4,
    }
    candidate = {
        "f1": 0.43,
        "false_positives_per_observed_hour": 0.25,
        "different_sensitivity": 0.48,
        "strict_no_ignore_f1": 0.41,
    }
    assert module_gate(candidate, original, [candidate], [original], config)["passed"]
    assert not module_gate(
        candidate, original, [{**candidate, "f1": 0.2}], [{**original, "f1": 0.4}], config
    )["passed"]


def test_boundary_head_only_edits_existing_event():
    predictions = _predictions([0.5] * 20)
    predictions.loc[:, "start_probability"] = 0
    predictions.loc[:, "end_probability"] = 0
    predictions.loc[4, "start_probability"] = 1
    predictions.loc[15, "end_probability"] = 1
    events = pd.DataFrame(
        [("one", "first", 9000, 45000, 0.7)],
        columns=["subject_key", "session_id", "start_ms", "end_ms", "score"],
    )
    corrected = correct_boundaries(predictions, events, 15)
    assert len(corrected) == len(events)
    assert corrected.iloc[0].start_ms == 12000
    assert corrected.iloc[0].end_ms == 45000
    assert correct_boundaries(predictions, events.iloc[:0]).empty


def test_event_gate_only_changes_positive_fusion_residual():
    baseline = _predictions([0.4, 0.4])
    dtp = _predictions([0.9, 0.1])
    calibrator = PlattCalibrator(1, 0, 0, 2, 1)
    parameters = {
        "alpha_positive": 0.5,
        "alpha_negative": 0.25,
        "baseline_support_min": 0,
        "baseline_support_max": 1,
        "dtp_on_threshold": 0,
        "dtp_off_threshold": 1,
        "persistence_seconds": 3,
    }
    gate = baseline[["subject_key", "session_id", "timestamp_ms"]].copy()
    gate["event_gate"] = 0.0
    gated = fuse_gated_prediction_frames(
        baseline,
        dtp,
        calibrator,
        parameters,
        event_gate=gate,
        gate_ema_half_life_seconds=0.1,
    )
    negative_only = fuse_gated_prediction_frames(
        baseline,
        dtp,
        calibrator,
        {**parameters, "alpha_positive": 0},
        gate_ema_half_life_seconds=0.1,
    )
    np.testing.assert_array_equal(gated.state_probability, negative_only.state_probability)
    assert gated.state_probability.iloc[1] < 0.4
    gate.loc[0, "event_gate"] = 1.0
    positive = fuse_gated_prediction_frames(
        baseline,
        dtp,
        calibrator,
        parameters,
        event_gate=gate,
        gate_ema_half_life_seconds=0.1,
    )
    assert positive.state_probability.iloc[0] > gated.state_probability.iloc[0]
    assert positive.state_probability.iloc[1] == gated.state_probability.iloc[1]
    gate.loc[0, "event_gate"] = float("nan")
    with pytest.raises(ValueError, match="missing prediction timestamps"):
        fuse_gated_prediction_frames(baseline, dtp, calibrator, parameters, event_gate=gate)


def test_manifest_hash_blocks_tampered_selection(tmp_path, monkeypatch):
    import bme_eating.dtp_postprocess as module

    monkeypatch.setattr(module, "_git_identity", lambda: {"commit": "head", "dirty": False})
    selection = tmp_path / "selected_dtp_postprocess.json"
    selection.write_text(json.dumps({"protocol_version": 2, "selection_run": "dtp_postprocess_a"}))
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "experiment": {"name": "dtp_postprocess_a", "fold": 0},
                "artifact_hashes": {selection.name: sha256_file(selection)},
                "git": {"commit": "head", "dirty": False},
            }
        )
    )
    assert _validate_selection(tmp_path, "dtp_postprocess_a", 0)["protocol_version"] == 2
    selection.write_text("{}")
    with pytest.raises(RuntimeError, match="hash changed"):
        _validate_selection(tmp_path, "dtp_postprocess_a", 0)


@pytest.mark.parametrize("force_boundary", [False, True])
def test_three_fold_search_fits_only_other_subjects(tmp_path, monkeypatch, force_boundary):
    import bme_eating.dtp_postprocess as module

    if force_boundary:
        def shift_boundaries(predictions, detected, lookback_seconds=60):
            shifted = detected.copy()
            if len(shifted):
                shifted["start_ms"] += 3000
            return shifted

        def approve_boundary(predictions, detected, events, folds, iou, method):
            adjusted = shift_boundaries(predictions, detected)
            truth, ignore = module.partition_evaluation_events(
                events, set(predictions.subject_key.astype(str))
            )
            metrics = module._metrics(predictions, adjusted, truth, ignore, iou, method)
            return True, {"before": metrics, "after": metrics, "checks": {}}, adjusted

        monkeypatch.setattr(module, "correct_boundaries", shift_boundaries)
        monkeypatch.setattr(module, "_boundary_gate", approve_boundary)
    frames = []
    anchors = []
    truths = []
    for partition in range(3):
        for member in range(2):
            subject = f"person_{partition}_{member}"
            values = [0.05] * 12 + [0.9] * 12 + [0.05] * 12
            frame = _predictions(values, subject=subject)
            frame["calibration_fold"] = partition
            frames.append(frame)
            labels = frame[["subject_key", "session_id", "timestamp_ms"]].copy()
            labels["state_target"] = [0] * 12 + [1] * 12 + [0] * 12
            labels["state_loss_mask"] = 1
            anchors.append(labels)
            truths.append(
                {
                    "subject_key": subject,
                    "start_ms": 36000,
                    "end_ms": 69000,
                    "valid_duration": True,
                    "evaluable": True,
                    "hand_relation": "different",
                }
            )
    predictions = pd.concat(frames, ignore_index=True)
    labels = pd.concat(anchors, ignore_index=True)
    events = pd.DataFrame(truths)
    config = {
        "output_step_seconds": 3,
        "probability_spaces": ["raw_control"],
        "fast_ema_half_life_seconds": [0.1],
        "slow_ema_half_life_seconds": [6],
        "start_modes": ["fast_persistent"],
        "fast_background_quantiles": [0.9],
        "slow_background_quantiles": [0.9],
        "persistence_seconds": [6],
        "exit_ratios": [0.5],
        "off_duration_seconds": [6],
        "minimum_event_seconds": 3,
        "score_quantiles": [None],
        "merge_gap_seconds": [0],
        "stage1_keep": 1,
        "final_keep": 1,
        "bootstrap_replicates": 4,
        "bootstrap_seed": 2026,
        "minimum_f1_improvement": -1,
        "module_minimum_f1_improvement": 0.01,
        "module_minimum_fp_reduction": 0.15,
        "module_minimum_large_f1_improvement": 0.02,
        "module_maximum_different_sensitivity_drop": 0.1,
        "module_maximum_strict_f1_drop": 0.01,
        "module_maximum_partition_f1_drop": 0.02,
        "maximum_xgb_fp_ratio": 1.2,
        "minimum_different_sensitivity_gain": 0.03,
    }
    baseline_postprocess = {
        "detector_mode": "hysteresis_v1",
        "ema_half_life_seconds": 12,
        "high_threshold": 0.95,
        "low_threshold": 0.5,
        "minimum_event_seconds": 30,
        "merge_gap_seconds": 0,
        "boundary_lookback_seconds": 0,
        "iou_threshold": 0.25,
        "matching_method": "max_cardinality_iou",
    }
    selection, outputs, _ = _crossfit(
        predictions,
        labels,
        events,
        predictions,
        baseline_postprocess,
        config,
        tmp_path / "search.checkpoint.jsonl",
    )
    for gate in selection["calibration_gate"]:
        heldout = set(
            predictions.loc[predictions.calibration_fold == gate["heldout_fold"], "subject_key"]
        )
        for inner in gate.get("inner_folds", []):
            assert set(inner["fit_subjects"]).isdisjoint(heldout)
            assert set(inner["validation_subjects"]).isdisjoint(heldout)
    for point in selection["working_points"].values():
        for fold in point.get("folds", []):
            assert set(fold["train_subjects"]).isdisjoint(fold["validation_subjects"])
            assert {item["subject_key"] for item in fold["fitted"]["score_reference"]}.isdisjoint(
                fold["validation_subjects"]
            )
    primary = selection["primary_working_point"]
    assert primary is not None
    point = selection["working_points"][primary]
    assert point["deployment_trials"]
    assert all(row["decoder"] == point["decoder"] for row in point["folds"])
    if force_boundary:
        assert point["use_boundary_head"]
        selected_gate = outputs[f"{primary}_gate"]
        selected_events = outputs[f"{primary}_events"]
        assert not selected_events.empty
        for event in selected_events.itertuples(index=False):
            member_gate = selected_gate[
                (selected_gate.subject_key == event.subject_key)
                & (selected_gate.session_id == event.session_id)
            ]
            assert member_gate.loc[
                member_gate.timestamp_ms == event.start_ms - 3000, "event_gate"
            ].eq(0).all()
            assert member_gate.loc[
                member_gate.timestamp_ms == event.start_ms, "event_gate"
            ].gt(0).all()


def test_failed_meta_gate_blocks_outer_read(tmp_path, monkeypatch):
    import bme_eating.dtp_postprocess as module

    monkeypatch.setattr(module, "load_config", lambda _: {"dtp_postprocess": {}})
    monkeypatch.setattr(module, "resolve_roots", lambda _: (tmp_path, tmp_path))
    monkeypatch.setattr(module, "require_clean_git_worktree", lambda: "head")
    monkeypatch.setattr(
        module,
        "_validate_selection",
        lambda *args: {
            "meta_gate_passed": False,
            "primary_working_point": None,
        },
    )
    monkeypatch.setattr(module, "_source", lambda *args: pytest.fail("outer source was read"))
    with pytest.raises(RuntimeError, match="Meta-OOF gate failed"):
        evaluate_dtp_postprocess(
            Namespace(
                config="unused",
                fold=1,
                source_run="baseline_dtp_fusion_a",
                selection_run="dtp_postprocess_selection",
                run_name="dtp_postprocess_eval",
            )
        )


@pytest.mark.parametrize("last_line", ['{"scope":"other","key":"old"}', '{"scope":'])
def test_search_checkpoint_repairs_unterminated_or_partial_tail(tmp_path, monkeypatch, last_line):
    import bme_eating.dtp_postprocess as module

    monkeypatch.setattr(
        module,
        "_metrics",
        lambda *args, **kwargs: {
            "f1": 0.2,
            "strict_no_ignore_f1": 0.2,
            "false_positives_per_observed_hour": 0.5,
            "different_sensitivity": 0.2,
        },
    )
    predictions = _predictions([0.1] * 12)
    predictions["calibration_fold"] = 0
    anchors = predictions[["subject_key", "session_id", "timestamp_ms"]].copy()
    anchors["state_target"] = 0
    anchors["state_loss_mask"] = 1
    truth = pd.DataFrame(columns=["subject_key"])
    checkpoint = tmp_path / "search.checkpoint.jsonl"
    checkpoint.write_text(last_line, encoding="utf-8")
    config = {
        "output_step_seconds": 3,
        "probability_spaces": ["raw_control"],
        "fast_ema_half_life_seconds": [6],
        "slow_ema_half_life_seconds": [36],
        "start_modes": ["fast_persistent"],
        "fast_background_quantiles": [0.9],
        "slow_background_quantiles": [0.9],
        "persistence_seconds": [6],
        "minimum_event_seconds": 3,
        "score_quantiles": [None],
        "merge_gap_seconds": [0],
        "exit_ratios": [0.5],
        "off_duration_seconds": [6],
        "stage1_keep": 1,
        "final_keep": 1,
        "maximum_xgb_fp_ratio": 1.2,
        "minimum_different_sensitivity_gain": 0.03,
    }

    def search():
        return _search_scope(
            predictions,
            anchors,
            truth,
            truth,
            config,
            0.25,
            "max_cardinality_iou",
            {
                "false_positives_per_observed_hour": 1,
                "strict_no_ignore_f1": 0,
                "different_sensitivity": 0,
            },
            {"raw_control"},
            checkpoint,
            "unit",
        )

    search()
    first_pass = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()]
    assert any(row["scope"] == "unit" for row in first_pass)
    search()
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == len(first_pass)


def test_outer_evaluation_rejects_duplicate_selection_across_run_names(tmp_path, monkeypatch):
    import bme_eating.dtp_postprocess as module

    monkeypatch.setattr(module, "load_config", lambda _: {"dtp_postprocess": {}})
    monkeypatch.setattr(module, "resolve_roots", lambda _: (tmp_path, tmp_path))
    monkeypatch.setattr(module, "require_clean_git_worktree", lambda: "head")
    monkeypatch.setattr(
        module,
        "_validate_selection",
        lambda *args: {"meta_gate_passed": True, "primary_working_point": "robust_f1"},
    )
    monkeypatch.setattr(module, "_source", lambda *args: pytest.fail("outer source was read"))
    older = tmp_path / "experiments" / "dtp_postprocess_old" / "fold_1"
    older.mkdir(parents=True)
    (older / "applied_selection.json").write_text(
        json.dumps({"selection_run": "dtp_postprocess_other"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="already evaluated"):
        evaluate_dtp_postprocess(
            Namespace(
                config="unused",
                fold=1,
                source_run="baseline_dtp_fusion_a",
                selection_run="dtp_postprocess_selection",
                run_name="dtp_postprocess_new",
            )
        )


def test_fold_zero_cannot_be_used_for_formal_outer_evaluation(tmp_path, monkeypatch):
    import bme_eating.dtp_postprocess as module

    monkeypatch.setattr(module, "load_config", lambda _: {"dtp_postprocess": {}})
    monkeypatch.setattr(module, "resolve_roots", lambda _: (tmp_path, tmp_path))
    monkeypatch.setattr(module, "require_clean_git_worktree", lambda: "head")
    monkeypatch.setattr(module, "_source", lambda *args: pytest.fail("outer source was read"))
    with pytest.raises(ValueError, match="development-only"):
        evaluate_dtp_postprocess(
            Namespace(
                config="unused",
                fold=0,
                source_run="baseline_dtp_fusion_a",
                selection_run="dtp_postprocess_selection",
                run_name="dtp_postprocess_new",
            )
        )
