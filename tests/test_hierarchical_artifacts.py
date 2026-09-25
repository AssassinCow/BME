from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

import bme_eating.hierarchical_artifacts as artifacts
from bme_eating.hierarchical_artifacts import (
    HierarchicalRun,
    OuterLabelGuard,
    assert_disjoint_subjects,
    initialize_hierarchical_run,
    write_json_atomic,
)


def test_subject_overlap_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="Subject leakage"):
        assert_disjoint_subjects(train={"a", "b"}, test={"b", "c"})


def test_outer_labels_are_blocked_until_evaluation() -> None:
    events = pd.DataFrame(
        [{"subject_key": "outer", "start_ms": 0, "end_ms": 1, "valid_duration": True}]
    )
    guard = OuterLabelGuard(frozenset({"outer"}), allow_outer_labels=False)
    with pytest.raises(RuntimeError, match="Outer-fold labels"):
        guard.select_labels(events, {"outer"})
    evaluation_guard = OuterLabelGuard(frozenset({"outer"}), allow_outer_labels=True)
    assert len(evaluation_guard.select_labels(events, {"outer"})) == 1


def test_atomic_json_accepts_numpy_scalars(tmp_path) -> None:
    path = tmp_path / "selection.json"
    write_json_atomic(path, {"threshold": np.float64(0.5), "count": np.int64(3)})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "threshold": 0.5,
        "count": 3,
    }


def test_manifested_upstream_artifact_tampering_is_rejected(tmp_path) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    write_json_atomic(manifest_path, {"stage": "CREATED", "artifact_hashes": {}})
    run = HierarchicalRun(
        tmp_path,
        manifest_path,
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    artifact = tmp_path / "oof" / "window_predictions.parquet"
    artifact.parent.mkdir()
    artifact.write_bytes(b"registered")
    run.transition("STATE_COMPLETE", [artifact])
    artifact.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="changed or is missing"):
        run.require_stage("STATE_COMPLETE")


def test_created_run_allows_resumable_migration_artifacts_to_change(tmp_path) -> None:
    state = tmp_path / "crossfit_1" / "state"
    state.mkdir(parents=True)
    last = state / "last.pt"
    history = state / "history.csv"
    normalization = state / "normalization.json"
    last.write_bytes(b"migrated-checkpoint")
    history.write_text("epoch\n8\n", encoding="utf-8")
    normalization.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "run_manifest.json"
    payload = {
        "stage": "CREATED",
        "artifact_hashes": {
            "crossfit_1/state/last.pt": artifacts.sha256_file(last),
            "crossfit_1/state/history.csv": artifacts.sha256_file(history),
            "crossfit_1/state/normalization.json": artifacts.sha256_file(normalization),
        },
        "state_checkpoint_migrations": {
            "1": {"training_will_resume": True},
        },
    }
    write_json_atomic(manifest_path, payload)
    run = HierarchicalRun(tmp_path, manifest_path, payload)

    last.write_bytes(b"continued-training-checkpoint")
    history.write_text("epoch\n8\n9\n", encoding="utf-8")
    run.require_stage("CREATED")

    normalization.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="normalization.json"):
        run.require_stage("CREATED")


def test_modified_freeze_manifest_is_rejected_after_stress_run_creation(
    tmp_path, monkeypatch
) -> None:
    input_root = tmp_path / "v2"
    output_root = tmp_path / "v3"
    for relative in (
        "indices/anchors.parquet",
        "indices/events.parquet",
        "indices/segments.parquet",
        "indices/subject_folds.json",
        "indices/subject_folds.manifest.json",
        "indices/quality_report.json",
        "features/baseline.parquet",
    ):
        path = input_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    config = {
        "project": {
            "input_artifact_schema_version": "v2",
            "artifact_schema_version": "v3",
            "strict_resume_identity": True,
        },
        "data": {"subject_folds": 5},
        "features": {"artifact_name": "baseline"},
        "model": {},
        "training": {"random_seed": 2026},
        "verifier": {"seeds": [2026, 2027, 2028]},
        "boundary": {"seeds": [2026, 2027, 2028]},
        "hierarchical": {"maximum_event_latency_seconds": 60},
        "postprocess": {
            "iou_threshold": 0.25,
            "matching_method": "max_cardinality_iou",
        },
    }
    run_name = "stress-test"
    experiment_root = output_root / "experiments" / run_name
    experiment_root.mkdir(parents=True)
    freeze_path = experiment_root / "freeze_manifest.json"
    write_json_atomic(
        freeze_path,
        {
            "resolved_config_sha256": artifacts._canonical_hash(config),
            "git_commit": "commit",
        },
    )
    monkeypatch.setattr(
        artifacts,
        "git_worktree_identity",
        lambda _root: {
            "commit": "commit",
            "dirty": False,
            "worktree_sha256": "snapshot",
        },
    )
    monkeypatch.setattr(
        "bme_eating.models.factory.build_state_model",
        lambda _config: torch.nn.Linear(1, 1),
    )
    initialize_hierarchical_run(
        config, input_root, output_root, run_name, 2, fresh=True
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["tampered"] = True
    write_json_atomic(freeze_path, freeze)

    with pytest.raises(RuntimeError, match="changed after run creation"):
        initialize_hierarchical_run(
            config, input_root, output_root, run_name, 2, fresh=False
        )


def test_relaxed_resume_updates_active_config_and_git_identity(tmp_path, monkeypatch) -> None:
    input_root = tmp_path / "v2"
    output_root = tmp_path / "v3"
    for relative in (
        "indices/anchors.parquet",
        "indices/events.parquet",
        "indices/segments.parquet",
        "indices/subject_folds.json",
        "indices/subject_folds.manifest.json",
        "indices/quality_report.json",
        "features/baseline.parquet",
    ):
        path = input_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    config = {
        "project": {
            "input_artifact_schema_version": "v2",
            "artifact_schema_version": "v3",
            "strict_resume_identity": False,
        },
        "data": {"subject_folds": 5},
        "features": {"artifact_name": "baseline"},
        "model": {},
        "training": {"random_seed": 2026, "num_workers": 2},
        "verifier": {"seeds": [2026, 2027, 2028]},
        "boundary": {"seeds": [2026, 2027, 2028]},
        "hierarchical": {"maximum_event_latency_seconds": 60},
        "postprocess": {
            "iou_threshold": 0.25,
            "matching_method": "max_cardinality_iou",
        },
    }
    identities = iter(
        (
            {"commit": "old", "dirty": False, "worktree_sha256": "old-tree"},
            {"commit": "new", "dirty": True, "worktree_sha256": "new-tree"},
        )
    )
    monkeypatch.setattr(artifacts, "git_worktree_identity", lambda _root: next(identities))
    monkeypatch.setattr(
        "bme_eating.models.factory.build_state_model",
        lambda _config: torch.nn.Linear(1, 1),
    )
    initialize_hierarchical_run(
        config, input_root, output_root, "relaxed-run", 0, fresh=True
    )
    config["training"]["num_workers"] = 8

    with pytest.warns(RuntimeWarning, match="active configuration and worktree"):
        resumed = initialize_hierarchical_run(
            config, input_root, output_root, "relaxed-run", 0, fresh=False
        )

    assert resumed.payload["git"]["commit"] == "new"
    assert resumed.payload["resume_history"][0]["previous_git"]["commit"] == "old"
    resolved = artifacts.yaml.safe_load(
        (resumed.root / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    assert resolved["training"]["num_workers"] == 8
