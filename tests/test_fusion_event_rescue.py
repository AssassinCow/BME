from __future__ import annotations

from argparse import Namespace
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from bme_eating import fusion_event_rescue as rescue
from bme_eating.config import load_config
from bme_eating.dtp_postprocess import fit_score_threshold
from bme_eating.fusion import sha256_file
from bme_eating.postprocess import probabilities_to_events


def events(rows=()):
    return pd.DataFrame(rows, columns=rescue.EVENT_COLUMNS)


def predictions(subjects=("a", "b", "c"), probability=0.9):
    return pd.DataFrame(
        [
            (subject, "segment", "session", t, probability, 0.0, 0.0, partition)
            for partition, subject in enumerate(subjects)
            for t in range(0, 600_001, 3000)
        ],
        columns=[
            "subject_key",
            "segment_id",
            "session_id",
            "timestamp_ms",
            "state_probability",
            "start_probability",
            "end_probability",
            "calibration_fold",
        ],
    )


POST = {
    "ema_half_life_seconds": 12,
    "high_threshold": 0.6,
    "low_threshold": 0.3,
    "minimum_event_seconds": 30,
    "merge_gap_seconds": 60,
    "boundary_lookback_seconds": 30,
    "iou_threshold": 0.25,
    "matching_method": "max_cardinality_iou",
}


def labels(subjects=("a", "b", "c")):
    return pd.DataFrame(
        [
            {
                "subject_key": s,
                "start_ms": 0,
                "end_ms": 600000,
                "valid_duration": True,
                "evaluable": True,
                "hand_relation": "different",
                "event_id": s,
            }
            for s in subjects
        ]
    )


def test_distance_boundary_session_scope_and_baseline_identity():
    base = events([("a", "s", 500000, 600000, 0.123456789012345)])
    candidate = events(
        [
            ("a", "s", 100000, 379999, 0.95),  # 120001 ms before
            ("a", "s", 200000, 380000, 0.95),  # exactly 120 s before
            ("a", "s", 550000, 650000, 0.95),  # overlap
            ("a", "s", 720000, 750000, 0.95),  # exactly 120 s after
            ("a", "s", 720001, 750001, 0.95),
            ("a", "other", 500000, 600000, 0.95),
            ("other", "s", 500000, 600000, 0.95),
            ("a", "s", 900000, 950000, 0.899),
        ]
    )
    combined, audit = rescue.append_rescue_events(base, candidate, 0.90)
    assert len(combined) == 5
    assert audit.reason.value_counts().to_dict() == {
        "accepted": 4,
        "baseline_within_120s": 3,
        "below_score_threshold": 1,
    }
    pd.testing.assert_frame_equal(
        combined[combined.score == base.score.iloc[0]].reset_index(drop=True), base
    )
    assert combined.equals(combined.sort_values(rescue.ORDER, kind="stable").reset_index(drop=True))


def test_empty_dtp_reproduces_baseline_and_empty_baseline_accepts():
    base = events([("a", "s", 0, 30000, 0.123456789)])
    combined, _ = rescue.append_rescue_events(base, events(), 0.9)
    pd.testing.assert_frame_equal(combined, base)
    combined, _ = rescue.append_rescue_events(events(), base, 0.1)
    pd.testing.assert_frame_equal(combined, base, check_dtype=False)
    assert rescue.append_rescue_events(events(), events(), 0.9)[0].empty


@pytest.mark.parametrize(
    "row",
    [
        ("a", "s", 30, 0, 0.9),
        ("a", "s", 0, 0, 0.9),
        ("a", "s", 0, 30, np.nan),
        ("a", "s", 0.1, 30, 0.9),
    ],
)
def test_invalid_event_fails(row):
    with pytest.raises(ValueError):
        rescue.append_rescue_events(events(), events([row]), 0.9)


def test_duplicate_events_fail_instead_of_silently_editing_baseline():
    duplicate = events([("a", "s", 0, 30, 0.9)] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        rescue.append_rescue_events(duplicate, events(), 0.9)
    with pytest.raises(ValueError, match="duplicate"):
        rescue.append_rescue_events(events(), duplicate, 0.9)


def test_linear_scan_matches_bruteforce_for_nested_intervals_and_many_candidates():
    rng = np.random.default_rng(2026)
    starts = np.sort(rng.choice(10_000_000, size=100, replace=False))
    base = events([("a", "s", int(t), int(t + rng.integers(1, 500000)), 0.5) for t in starts])
    starts = rng.choice(10_000_000, size=1000, replace=False)
    candidate = events([("a", "s", int(t), int(t + 1000), 0.95) for t in starts])
    combined, audit = rescue.append_rescue_events(base, candidate, 0.9)
    expected = {
        int(t)
        for t in starts
        if not any(
            int(t) <= b.end_ms + 120000 and int(t) + 1000 >= b.start_ms - 120000
            for b in base.itertuples()
        )
    }
    assert set(audit.loc[audit.reason == "accepted", "start_ms"]) == expected
    assert len(combined) == len(base) + len(expected)


def test_subject_equal_threshold():
    frame = events([("a", "s", i, i + 1, 0.1) for i in range(100)] + [("b", "s", 0, 1, 0.9)])
    assert fit_score_threshold(frame, 0.90) == 0.9


def test_crossfit_threshold_never_sees_heldout_subject(monkeypatch):
    dtp = predictions()
    base = predictions(probability=0.1)
    original = rescue.fit_score_threshold
    seen = []

    def observe(frame, quantile):
        seen.append(set(frame.subject_key))
        return original(frame, quantile)

    monkeypatch.setattr(rescue, "fit_score_threshold", observe)
    record, combined, baseline_events, _ = rescue.crossfit_event_rescue(base, dtp, labels(), POST)
    assert seen == [{"b", "c"}, {"a", "c"}, {"a", "b"}, {"a", "b", "c"}]
    assert record["outer_predictions_read"] is False
    assert len(combined) == 3 and baseline_events.empty
    assert all(
        not set(r["training_subjects"]) & set(r["validation_subjects"])
        for r in record["partitions"]
    )


def test_no_candidates_and_missing_labels_fail():
    negative = predictions(probability=0.0)
    with pytest.raises(ValueError, match="No training candidate"):
        rescue.crossfit_event_rescue(negative, negative, labels(), POST)
    with pytest.raises(ValueError, match="missing columns"):
        rescue.crossfit_event_rescue(predictions(), predictions(), pd.DataFrame(), POST)


def test_invalid_partition_fails_and_gaps_preserve_frozen_session_semantics():
    dtp = predictions()
    dtp.loc[0, "calibration_fold"] = 1
    with pytest.raises(ValueError, match="subject-disjoint"):
        rescue.crossfit_event_rescue(predictions(), dtp, labels(), POST)
    dtp = predictions().drop(index=[1, 2])
    selection = {"protocol": rescue.SETTINGS, "baseline_postprocess": POST, "score_threshold": 0.9}
    actual, baseline, _ = rescue.apply_frozen_event_rescue(dtp, dtp, selection)
    expected = probabilities_to_events(dtp, **rescue._event_parameters(POST))
    pd.testing.assert_frame_equal(baseline, expected)
    pd.testing.assert_frame_equal(actual, expected.sort_values(rescue.ORDER).reset_index(drop=True))


def test_frozen_inference_decodes_only_twice_and_has_no_label_access(monkeypatch):
    base, dtp = predictions(), predictions(probability=0.01)
    expected = probabilities_to_events(base, **rescue._event_parameters(POST))
    original = rescue.probabilities_to_events
    calls = []

    def decode(frame, **parameters):
        calls.append(parameters)
        return original(frame, **parameters)

    def forbidden(*args, **kwargs):
        pytest.fail("Inference read labels, fitted thresholds or ran bootstrap")

    monkeypatch.setattr(rescue, "probabilities_to_events", decode)
    for name in ("fit_score_threshold", "partition_evaluation_events", "paired_subject_bootstrap"):
        monkeypatch.setattr(rescue, name, forbidden)
    result, _, _ = rescue.apply_frozen_event_rescue(
        base,
        dtp,
        {
            "protocol": rescue.SETTINGS,
            "baseline_postprocess": POST,
            "score_threshold": 0.90,
        },
    )
    pd.testing.assert_frame_equal(result, expected.sort_values(rescue.ORDER).reset_index(drop=True))
    assert len(calls) == 2 and calls[1] == rescue.GENERATOR


def test_fixed_config_and_development_failure_not_rounded_to_pass():
    config = load_config("configs/dtp_fusion_event_rescue.yaml")
    rescue.validate_protocol(config)
    config["event_rescue"]["exclusion_seconds"] = 119
    with pytest.raises(ValueError, match="fixed"):
        rescue.validate_protocol(config)
    bm = {
        "f1": 0.53,
        "different_sensitivity": 0.4,
        "strict_no_ignore_f1": 0.46,
        "false_positives_per_observed_hour": 62,
        "start_mae_seconds": 289,
        "end_mae_seconds": 82,
        "true_positive": 69,
        "sensitivity": 0.53,
    }
    cm = {
        **bm,
        "f1": 0.56,
        "different_sensitivity": 0.5,
        "strict_no_ignore_f1": 0.48,
        "false_positives_per_observed_hour": 75,
        "true_positive": 79,
        "sensitivity": 0.61,
    }
    gate = rescue.rescue_gate(cm, bm)
    assert gate["passed"] is False
    assert gate["checks"]["fp_per_hour_within_ratio"] is False


def _fixture_inputs(tmp_path, monkeypatch):
    config = load_config("configs/dtp_fusion_event_rescue.yaml")
    source = "baseline_dtp_fusion_test"
    indices = tmp_path / "indices"
    indices.mkdir()
    rescue._json(indices / "subject_folds.json", {"a": 0, "b": 1, "c": 3, "test": 2})
    rescue._json(indices / "subject_folds.manifest.json", {"version": 2})
    labels(("a", "b", "c", "test")).to_parquet(indices / "events.parquet")
    inputs = {
        k: sha256_file(indices / n)
        for k, n in (
            ("events", "events.parquet"),
            ("subject_folds", "subject_folds.json"),
            ("subject_folds_manifest", "subject_folds.manifest.json"),
        )
    }
    for run, role in (("baseline", "baseline"), (source, "dtp")):
        directory = tmp_path / "experiments" / run / "fold_2"
        directory.mkdir(parents=True)
        name = (
            "validation_predictions.parquet"
            if role == "baseline"
            else "dtp_oof_predictions.parquet"
        )
        predictions(probability=0.1 if role == "baseline" else 0.9).to_parquet(directory / name)
        artifact_hashes = {name: sha256_file(directory / name)}
        test_name = (
            "test_predictions.parquet" if role == "baseline" else "dtp_test_predictions.parquet"
        )
        predictions(("test",), probability=0.1 if role == "baseline" else 0.9).to_parquet(
            directory / test_name
        )
        artifact_hashes[test_name] = sha256_file(directory / test_name)
        if role == "baseline":
            rescue._json(directory / "selected_postprocess.json", POST)
            artifact_hashes["selected_postprocess.json"] = sha256_file(
                directory / "selected_postprocess.json"
            )
        else:
            for partition, subject in enumerate(("a", "b", "c")):
                path = directory / f"crossfit_{partition}" / "metadata.json"
                path.parent.mkdir()
                rescue._json(
                    path,
                    {
                        "outer_fold": 2,
                        "inner_validation_partition": partition,
                        "train_subjects": sorted({"a", "b", "c"} - {subject}),
                        "validation_subjects": [subject],
                    },
                )
                artifact_hashes[path.relative_to(directory).as_posix()] = sha256_file(path)
        rescue._json(
            directory / "run_manifest.json",
            {
                "experiment": {"name": run, "fold": 2},
                "git": {"dirty": False, "commit": "3ca55bb"},
                "hashes": inputs,
                "artifact_hashes": artifact_hashes,
            },
        )
    identity = {"test": "identity"}
    monkeypatch.setattr(rescue, "_identity", lambda *a, **k: identity)
    monkeypatch.setattr(rescue, "resolve_roots", lambda _config: (tmp_path, tmp_path))
    monkeypatch.setattr(rescue, "require_registration", lambda *a: {"registered": True})
    args = Namespace(
        config="configs/dtp_fusion_event_rescue.yaml",
        source_run=source,
        run_name="fusion_event_rescue_test",
        fold=2,
        fresh=True,
        resume=False,
    )
    return config, args


def test_fit_never_reads_or_hashes_outer_predictions(tmp_path, monkeypatch):
    _, args = _fixture_inputs(tmp_path, monkeypatch)
    original_read, original_hash = pd.read_parquet, rescue.sha256_file

    def safe_read(path, *a, **k):
        assert "test_predictions" not in str(path)
        return original_read(path, *a, **k)

    def safe_hash(path):
        assert "test_predictions" not in str(path)
        return original_hash(path)

    monkeypatch.setattr(pd, "read_parquet", safe_read)
    monkeypatch.setattr(rescue, "sha256_file", safe_hash)
    directory = rescue.fit_event_rescue(args)
    assert (directory / "selected_event_rescue.json").exists()
    record = rescue._read(directory / "selected_event_rescue.json")
    assert record["outer_predictions_read"] is False
    args.fresh, args.resume = False, True
    assert rescue.fit_event_rescue(args) == directory
    (directory / "accepted_dtp_events.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact changed"):
        rescue.fit_event_rescue(args)


def test_resume_signature_rejects_changed_inputs_and_code(tmp_path, monkeypatch):
    _, args = _fixture_inputs(tmp_path, monkeypatch)
    directory = rescue.fit_event_rescue(args)
    args.fresh, args.resume = False, True
    monkeypatch.setattr(rescue, "_identity", lambda *a, **k: {"test": "new-code"})
    with pytest.raises(RuntimeError, match="Cannot resume"):
        rescue.fit_event_rescue(args)
    assert rescue._read(directory / "run_manifest.json")["identity"] == {"test": "identity"}


def test_outer_atomic_once_and_no_working_point_switch(tmp_path, monkeypatch):
    _, args = _fixture_inputs(tmp_path, monkeypatch)
    directory = rescue.fit_event_rescue(args)
    outer = rescue.evaluate_event_rescue(args)
    assert outer == directory / "outer"
    assert not (directory / "outer.pending").exists()
    assert (outer / "run_manifest.json").exists()
    with pytest.raises(FileExistsError, match="already evaluated"):
        rescue.evaluate_event_rescue(args)
    args.fold = 1
    with pytest.raises(ValueError, match="restricted"):
        rescue.evaluate_event_rescue(args)


def test_outer_interruption_preserves_audit_and_rejects_restart(tmp_path, monkeypatch):
    _, args = _fixture_inputs(tmp_path, monkeypatch)
    directory = rescue.fit_event_rescue(args)

    def interrupted(*a, **k):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(rescue, "apply_frozen_event_rescue", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        rescue.evaluate_event_rescue(args)
    assert not (directory / "outer").exists()
    assert (directory / "outer.pending" / "attempt.json").exists()
    with pytest.raises(RuntimeError, match="Interrupted outer"):
        rescue.evaluate_event_rescue(args)


def test_no_training_scope_leakage(tmp_path, monkeypatch):
    config, args = _fixture_inputs(tmp_path, monkeypatch)
    directory = tmp_path / "experiments" / args.source_run / "fold_2"
    path = directory / "crossfit_0" / "metadata.json"
    data = rescue._read(path)
    data["train_subjects"].append("test")
    rescue._json(path, data)
    manifest = rescue._read(directory / "run_manifest.json")
    manifest["artifact_hashes"]["crossfit_0/metadata.json"] = sha256_file(path)
    rescue._json(directory / "run_manifest.json", manifest)
    with pytest.raises(RuntimeError, match="scope leaks"):
        rescue._inputs(tmp_path, config, args.source_run, 2)


def test_registration_required_before_training_mutation(tmp_path, monkeypatch):
    from bme_eating import cli

    config = load_config("configs/dtp_fusion_event_rescue.yaml")
    monkeypatch.setattr(cli, "load_config", lambda _: deepcopy(config))
    monkeypatch.setattr(cli, "require_clean_git_worktree", lambda: "commit")
    monkeypatch.setattr(cli, "resolve_roots", lambda _: (tmp_path, tmp_path))
    monkeypatch.setattr(cli, "validate_quality_gate", lambda _: None)
    args = Namespace(
        config="unused",
        fold=2,
        run_name="baseline_dtp_fusion_test",
        fresh=False,
        resume=None,
        event_rescue_run="v5_locked",
    )
    with pytest.raises(FileNotFoundError, match="protocol_registration"):
        cli.command_train_fusion(args)
    assert not (tmp_path / "experiments").exists()


def test_registration_checks_code_source_and_development_hashes(tmp_path, monkeypatch):
    config = load_config("configs/dtp_fusion_event_rescue.yaml")
    run = tmp_path / "experiments" / "v5_test"
    identity = {"test": "code-a"}
    monkeypatch.setattr(rescue, "_identity", lambda *a, **k: identity)
    monkeypatch.setattr(rescue, "_load_fit", lambda *a: ({}, {}))
    hashes = {}
    for f in (0, 1):
        directory = run / f"fold_{f}"
        directory.mkdir(parents=True)
        rescue._json(directory / "run_manifest.json", {"fold": f})
        hashes[str(f)] = sha256_file(directory / "run_manifest.json")
    rescue._json(
        run / "protocol_registration.json",
        {
            "identity": identity,
            "source_run": "source",
            "run_name": run.name,
            "development_manifests": hashes,
        },
    )
    rescue.require_registration(tmp_path, run.name, config, "source")
    with pytest.raises(RuntimeError, match="source run differs"):
        rescue.require_registration(tmp_path, run.name, config, "another_source")
    (run / "fold_1" / "run_manifest.json").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="development evidence changed"):
        rescue.require_registration(tmp_path, run.name, config, "source")
    identity["test"] = "code-b"
    with pytest.raises(RuntimeError, match="code/config/Git changed"):
        rescue.require_registration(tmp_path, run.name, config, "source")


def test_stress_summary_pools_counts_not_fold_f1_and_is_immutable(tmp_path, monkeypatch):
    run = tmp_path / "experiments" / "v5_summary"
    run.mkdir(parents=True)
    identity = {"test": "code"}
    registration = {"source_run": "source"}
    rescue._json(run / "protocol_registration.json", registration)
    monkeypatch.setattr(rescue, "_identity", lambda *a, **k: identity)
    monkeypatch.setattr(rescue, "resolve_roots", lambda _: (tmp_path, tmp_path))
    monkeypatch.setattr(rescue, "require_registration", lambda *a: registration)
    fit_manifest = {"frozen": True}
    selections = {}
    for f in (2, 3, 4):
        subject = f"subject_{f}"
        p = predictions((subject,))
        y = labels((subject,))
        y.loc[0, "end_ms"] = 100000
        if f == 3:
            y = pd.concat(
                [
                    y,
                    y.assign(start_ms=200000, end_ms=300000),
                    y.assign(start_ms=400000, end_ms=500000),
                ],
                ignore_index=True,
            )
        base = events([(subject, "session", 0, 100000, 0.9)])
        if f == 2:
            base = pd.concat(
                [base, events([(subject, "session", 500000, 550000, 0.9)])], ignore_index=True
            )
        candidate = base.copy()
        if f == 3:
            candidate = pd.concat(
                [
                    candidate,
                    events(
                        [
                            (subject, "session", 200000, 300000, 0.9),
                            (subject, "session", 400000, 500000, 0.9),
                        ]
                    ),
                ],
                ignore_index=True,
            )
        bm = rescue._metrics(p, base, y, POST)
        cm = rescue._metrics(p, candidate, y, POST)
        selections[f] = {"partitions": [{"baseline_metrics": bm, "candidate_metrics": cm}]}
        directory = run / f"fold_{f}" / "outer"
        directory.mkdir(parents=True)
        candidate.to_parquet(directory / "events.parquet")
        base.to_parquet(directory / "baseline_events.parquet")
        p.to_parquet(directory / "exposure.parquet")
        y.to_parquet(directory / "evaluation_labels.parquet")
        rescue._json(
            directory / "metrics.json",
            {
                "fold": f,
                "baseline": bm,
                "candidate": cm,
                "registration_sha256": rescue._digest(registration),
            },
        )
        rescue._manifest(
            directory,
            identity,
            {
                "fit_manifest_sha256": rescue._digest(fit_manifest),
                "registration_sha256": rescue._digest(registration),
            },
        )
    monkeypatch.setattr(
        rescue, "_load_fit", lambda run, fold, identity: (selections[fold], fit_manifest)
    )
    args = Namespace(config="configs/dtp_fusion_event_rescue.yaml", run_name=run.name)
    path = rescue.summarize_event_rescue(args)
    report = rescue._read(path)
    assert report["baseline"]["true_positive"] == 3
    assert report["baseline"]["f1"] == pytest.approx(6 / 9)
    assert report["candidate"]["f1"] == pytest.approx(10 / 11)
    assert report["gate"]["passed"] is True
    assert rescue.summarize_event_rescue(args) == path
    report["submission_candidate"] = "tampered"
    rescue._json(path, report)
    with pytest.raises(RuntimeError, match="Immutable stress summary"):
        rescue.summarize_event_rescue(args)


def test_training_failure_does_not_disable_existing_v4_gate(tmp_path, monkeypatch):
    from bme_eating import cli

    config = load_config("configs/dtp_fusion.yaml")
    monkeypatch.setattr(cli, "load_config", lambda _: deepcopy(config))
    monkeypatch.setattr(cli, "require_clean_git_worktree", lambda: "commit")
    monkeypatch.setattr(cli, "resolve_roots", lambda _: (tmp_path, tmp_path))
    monkeypatch.setattr(cli, "validate_quality_gate", lambda _: None)
    args = Namespace(
        config="unused",
        fold=2,
        run_name="baseline_dtp_fusion_test",
        fresh=False,
        resume=None,
        event_rescue_run="v5_locked",
        confirmation_run=None,
    )
    with pytest.raises(ValueError, match="Protocol-v4 source"):
        cli.command_train_fusion(args)
