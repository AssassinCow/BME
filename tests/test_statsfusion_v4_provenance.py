from __future__ import annotations

import torch

import bme_eating.hierarchical_v4_artifacts as v4_artifacts
from bme_eating.hierarchical_v4_artifacts import initialize_v4_run
from bme_eating.stats_features import STATS_FEATURE_COLUMNS, audit_feature_provenance


def test_feature_provenance_hashes_sources_and_marks_stress_evidence(tmp_path) -> None:
    project = tmp_path / "project"
    inputs = tmp_path / "v2"
    project.mkdir()
    (project / "selection.md").write_text("five-fold gain", encoding="utf-8")
    source = inputs / "experiments" / "baseline" / "fold_0" / "metadata.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    result = audit_feature_provenance(
        project_root=project,
        input_root=inputs,
        source_paths=["experiments/baseline/fold_0/metadata.json"],
        selection_note_path="selection.md",
        assumed_used_all_outer_folds=True,
    )
    assert tuple(result["feature_columns"]) == STATS_FEATURE_COLUMNS
    assert result["evidence_classification"] == "development_stress_only"
    assert all(len(source["sha256"]) == 64 for source in result["sources"])


def test_v4_resume_rejects_worktree_identity_change(tmp_path, monkeypatch) -> None:
    input_root = tmp_path / "v2"
    output_root = tmp_path / "v4"
    for relative in (
        "indices/anchors.parquet",
        "indices/events.parquet",
        "indices/segments.parquet",
        "indices/subject_folds.json",
        "indices/subject_folds.manifest.json",
        "indices/quality_report.json",
        "features/baseline.parquet",
        "experiments/baseline/fold_0/metadata.json",
    ):
        path = input_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    config = {
        "project": {
            "artifact_schema_version": "v4",
            "input_artifact_schema_version": "v2",
            "strict_resume_identity": True,
        },
        "data": {"subject_folds": 5},
        "features": {"artifact_name": "baseline"},
        "model": {},
        "training": {"random_seed": 2026},
        "verifier": {"seeds": [2026]},
        "boundary": {"seeds": [2026]},
        "hierarchical": {"maximum_event_latency_seconds": 60},
        "feature_provenance": {
            "source_paths": ["experiments/baseline/fold_0/metadata.json"],
            "selection_note_path": None,
            "assumed_used_all_outer_folds": True,
        },
    }
    identities = iter(
        (
            {"commit": "a", "dirty": False, "worktree_sha256": "one"},
            {"commit": "b", "dirty": True, "worktree_sha256": "two"},
        )
    )
    monkeypatch.setattr(v4_artifacts, "git_worktree_identity", lambda _root: next(identities))
    monkeypatch.setattr(
        "bme_eating.models.factory.build_state_model", lambda _config: torch.nn.Linear(1, 1)
    )
    initialize_v4_run(config, input_root, output_root, "strict-v4", 0, fresh=True)
    try:
        initialize_v4_run(config, input_root, output_root, "strict-v4", 0, fresh=False)
    except RuntimeError as error:
        assert "worktree differs" in str(error)
    else:
        raise AssertionError("V4 resume accepted a changed worktree identity")
