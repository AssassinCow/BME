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
