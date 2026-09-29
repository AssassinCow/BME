from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

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
