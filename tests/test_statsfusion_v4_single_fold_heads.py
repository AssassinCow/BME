from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from bme_eating.metrics import evaluate_events
from bme_eating.training import hierarchical_v4_single_fold_heads as diagnostic


def test_single_fold_diagnostic_worktree_records_source_without_enforcing_parent(
    monkeypatch, tmp_path
) -> None:
    for relative in diagnostic.DIAGNOSTIC_SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    monkeypatch.setattr(
        diagnostic,
        "git_worktree_identity",
        lambda _root: {"commit": "new-commit", "dirty": True, "worktree_sha256": "new"},
    )
    identity = diagnostic._verify_diagnostic_worktree(tmp_path, "parent-commit")
    assert identity["commit"] == "new-commit"
    assert identity["parent_commit"] == "parent-commit"
    assert identity["matches_parent_commit"] is False
    assert set(identity["source_sha256"]) == set(diagnostic.DIAGNOSTIC_SOURCE_PATHS)


def test_single_fold_head_partitions_are_deterministic_and_subject_disjoint() -> None:
    proposals = pd.DataFrame(
        {
            "subject_key": [f"subject-{index}" for index in range(11)],
        }
    )
    config = {
        "hierarchical": {"verifier_crossfit_partitions": 3},
        "training": {"random_seed": 2026},
    }
    first = diagnostic._subject_partitions(proposals, config)
    second = diagnostic._subject_partitions(proposals.sample(frac=1.0), config)
    assert first == second
    assert set(first) == set(proposals["subject_key"])
    assert set(first.values()) == {0, 1, 2}
    for partition in set(first.values()):
        training = {subject for subject, value in first.items() if value != partition}
        prediction = {subject for subject, value in first.items() if value == partition}
        assert training.isdisjoint(prediction)


def _cohort_fixture():
    windows = pd.DataFrame({"subject_key": ["a", "b", "no-candidates"]})
    events = pd.DataFrame(
        {
            "subject_key": ["a", "b", *(["no-candidates"] * 7), "outer"],
            "session_id": ["session"] * 10,
            "event_id": list(range(10)),
            "start_ms": [index * 20_000 for index in range(10)],
            "end_ms": [index * 20_000 + 10_000 for index in range(10)],
            "valid_duration": [True] * 10,
            "evaluable": [True] * 8 + [False, True],
            "hand_relation": ["same"] * 10,
        }
    )
    run = SimpleNamespace(
        payload={
            "state_oof_prediction_subjects": ["a", "b", "no-candidates"],
            "outer_test_subjects": ["outer"],
        }
    )
    return run, windows, events


@pytest.mark.parametrize("candidate_subjects", [["a", "b"], []])
def test_single_fold_cohort_keeps_truth_and_ignore_for_zero_candidate_subjects(
    candidate_subjects,
) -> None:
    run, windows, events = _cohort_fixture()
    proposals = pd.DataFrame({"subject_key": candidate_subjects})
    truth, ignore, cohort = diagnostic._evaluation_cohort(run, windows, proposals, events)
    assert len(truth) == 8
    assert len(ignore) == 1
    assert "outer" not in set(truth.subject_key)
    assert cohort["subject_count"] == 3
    assert cohort["truth_without_candidates_count"] == (6 if candidate_subjects else 8)
    predictions = events[events.subject_key.isin(candidate_subjects)].copy()
    metrics, _ = evaluate_events(truth, predictions, method="max_cardinality_iou", ignore=ignore)
    assert metrics["false_negative"] == (6 if candidate_subjects else 8)
    assert metrics["sensitivity"] == (0.25 if candidate_subjects else 0.0)


@pytest.mark.parametrize("invalid", ["missing_timeline_subject", "outside_candidate", "outer"])
def test_single_fold_cohort_rejects_wrong_evaluation_population(invalid) -> None:
    run, windows, events = _cohort_fixture()
    proposals = pd.DataFrame({"subject_key": ["a"]})
    if invalid == "missing_timeline_subject":
        windows = windows.iloc[:2]
    elif invalid == "outside_candidate":
        proposals = pd.DataFrame({"subject_key": ["unknown"]})
    else:
        run.payload["outer_test_subjects"] = ["a"]
    with pytest.raises(RuntimeError):
        diagnostic._evaluation_cohort(run, windows, proposals, events)


def test_single_fold_boundary_reports_complete_cohort_even_without_accepted_events(
    monkeypatch, tmp_path
) -> None:
    run, windows, events = _cohort_fixture()
    run.root = tmp_path / "run"
    (run.root / "oof").mkdir(parents=True)
    windows.to_parquet(run.root / "oof" / "window_predictions.parquet", index=False)
    root = run.root / "diagnostics" / "single_fold_heads"
    root.mkdir(parents=True)
    scores = pd.DataFrame({"subject_key": ["a"], "max_iou": [0.0]})
    scores.to_parquet(root / "oof_proposal_scores.parquet", index=False)
    monkeypatch.setattr(diagnostic.core, "_accepted_from_point", lambda *_args: scores.iloc[:0])
    monkeypatch.setattr(
        diagnostic.core,
        "_attach_truth_boundaries",
        lambda *_args: pd.DataFrame(columns=["subject_key", "session_id", "matched_event_id"]),
    )
    _, report = diagnostic._fit_boundary(
        root,
        run,
        {"boundary": {"minimum_independent_events": 60}},
        SimpleNamespace(events=events),
        {"selected_point": {}},
        resume=False,
    )
    assert report["evaluation_cohort"]["truth_count"] == 8
    assert report["evaluation_cohort"]["ignore_count"] == 1
    assert report["evaluation_cohort"]["truth_without_candidates_count"] == 7


def test_completed_legacy_diagnostics_resume_reselects_and_preserves_evidence(
    monkeypatch, tmp_path
) -> None:
    run = SimpleNamespace(root=tmp_path)
    root = tmp_path / "diagnostics" / "single_fold_heads"
    (root / "verifier").mkdir(parents=True)
    checkpoint = root / "verifier" / "partition_0_seed_2026.pt"
    checkpoint.write_bytes(b"unchanged verifier weights")
    (root / "boundary").mkdir()
    boundary = root / "boundary" / "partition_0_seed_2026.pt"
    boundary.write_bytes(b"boundary depends on old selected scores")
    identity = {
        "protocol": diagnostic.DIAGNOSTIC_PROTOCOL,
        "evaluation_cohort_protocol": diagnostic.EVALUATION_COHORT_PROTOCOL,
        "parent_sha256": {"oof_windows": "frozen-parent"},
        "diagnostic_worktree": {"commit": "current"},
    }
    legacy = {key: value for key, value in identity.items() if key != "evaluation_cohort_protocol"}
    legacy.update(
        stage="COMPLETE",
        artifact_sha256={
            "verifier/partition_0_seed_2026.pt": diagnostic.sha256_file(checkpoint),
            "boundary/partition_0_seed_2026.pt": diagnostic.sha256_file(boundary),
        },
    )
    old_manifest = json.dumps(legacy)
    (root / "manifest.json").write_text(old_manifest, encoding="utf-8")
    monkeypatch.setattr(diagnostic, "_parent_identity", lambda *_args: identity.copy())
    new_root, path, active, stage = diagnostic._load_or_initialize(
        run, {}, fresh=False, resume=True
    )
    assert stage == "CREATED"
    assert new_root == root
    archive = run.root / active["evaluation_cohort_repair"]["archive_path"]
    assert (archive / "manifest.json").read_text(encoding="utf-8") == old_manifest
    assert (archive / "boundary" / boundary.name).read_bytes() == (
        b"boundary depends on old selected scores"
    )
    assert (root / "verifier" / checkpoint.name).read_bytes() == b"unchanged verifier weights"
    assert not (root / "boundary").exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["evaluation_cohort_protocol"] == diagnostic.EVALUATION_COHORT_PROTOCOL
    assert saved["evaluation_cohort_repair"]["state_retraining_required"] is False
    assert saved["evaluation_cohort_repair"]["reused_verifier_sha256"] == {
        "verifier/partition_0_seed_2026.pt": diagnostic.sha256_file(checkpoint),
    }
    _, _, resumed, resumed_stage = diagnostic._load_or_initialize(
        run, {}, fresh=False, resume=True
    )
    assert resumed_stage == "CREATED"
    assert resumed["evaluation_cohort_repair"] == active["evaluation_cohort_repair"]
    assert (archive / "verifier" / checkpoint.name).read_bytes() == b"unchanged verifier weights"
    assert len(list(root.parent.glob("single_fold_heads_before_cohort_fix_*"))) == 1


def test_single_fold_resume_rejects_corrupted_report_before_migration(monkeypatch, tmp_path) -> None:
    root = tmp_path / "diagnostics" / "single_fold_heads"
    root.mkdir(parents=True)
    report = root / "verifier_report.json"
    report.write_text("{}", encoding="utf-8")
    identity = {"protocol": diagnostic.DIAGNOSTIC_PROTOCOL}
    saved = {**identity, "stage": "COMPLETE", "artifact_sha256": {report.name: "wrong-hash"}}
    (root / "manifest.json").write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(
        diagnostic,
        "_parent_identity",
        lambda *_args: {
            **identity,
            "evaluation_cohort_protocol": diagnostic.EVALUATION_COHORT_PROTOCOL,
        },
    )
    with pytest.raises(RuntimeError, match="artifact changed"):
        diagnostic._load_or_initialize(SimpleNamespace(root=tmp_path), {}, fresh=False, resume=True)
    assert report.exists()
    assert not list(root.parent.glob("single_fold_heads_before_cohort_fix_*"))


def test_single_fold_heads_read_outer_labels_only_after_head_lock(monkeypatch, tmp_path) -> None:
    root = tmp_path / "diagnostic"
    root.mkdir()
    (root / "verifier_report.json").write_text(
        json.dumps({"winner": "state_only"}), encoding="utf-8"
    )
    (root / "boundary_report.json").write_text(
        json.dumps({"diagnostic_enabled": False}), encoding="utf-8"
    )
    manifest = root / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    run = SimpleNamespace(
        stage="PROPOSALS_COMPLETE",
        payload={"outer_fold": 0},
    )
    calls: list[str] = []
    outer_inputs = object()

    monkeypatch.setattr(
        diagnostic,
        "_load_or_initialize",
        lambda *_args, **_kwargs: (root, manifest, {"identity": "locked"}, "BOUNDARY_COMPLETE"),
    )

    def load_inputs(*_args, **kwargs):
        assert kwargs["event_role"] == "outer_test"
        assert kwargs["allow_outer_labels"] is True
        calls.append("load_outer")
        return outer_inputs

    def evaluate(*_args, **_kwargs):
        assert calls == ["load_outer"]
        assert _args[3] is outer_inputs
        calls.append("evaluate")
        return []

    monkeypatch.setattr(diagnostic.core, "load_v4_inputs", load_inputs)
    monkeypatch.setattr(diagnostic, "_evaluate_outer", evaluate)
    monkeypatch.setattr(diagnostic, "_write_manifest", lambda *_args, **_kwargs: None)

    diagnostic.train_single_fold_heads_v4(
        run,
        {},
        object(),
        tmp_path,
        fresh=False,
        resume=True,
    )
    assert calls == ["load_outer", "evaluate"]
