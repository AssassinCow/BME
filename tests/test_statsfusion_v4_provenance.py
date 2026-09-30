from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import torch

import bme_eating.data.stats_fusion_inputs as statsfusion_inputs
import bme_eating.hierarchical_v4_artifacts as v4_artifacts
from bme_eating.config import load_config
from bme_eating.data.stats_fusion_inputs import (
    _preparation_identity,
    _segment_archive_digest,
    _segment_archive_stat_digest,
    _session_cache_path,
    prepare_canonical_statsfusion_inputs,
    verify_canonical_statsfusion_inputs,
)
from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.hierarchical_v4_artifacts import (
    _input_snapshot_matches,
    feature_provenance_artifact_path,
    initialize_v4_run,
    resume_config_hash,
)
from bme_eating.stats_features import STATS_FEATURE_COLUMNS, audit_feature_provenance
from bme_eating.v4_protocol import INPUT_SNAPSHOT_FILENAME


def _r3_config() -> dict:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "hierarchical_v4_statsfusion_r3.yaml")


def test_input_snapshot_identity_ignores_code_version_but_not_input_hashes() -> None:
    current = {
        "version": 6,
        "protocol_version": "statsfusion-r3.2",
        "input_artifact_schema_version": "v2",
        "hashes": {"anchors": "a" * 64},
    }
    legacy = {
        **current,
        "version": 5,
        "code_version": "v4.4.2",
    }
    assert _input_snapshot_matches(legacy, current)
    legacy_with_retired_input = deepcopy(legacy)
    legacy_with_retired_input["hashes"]["retired_feature_artifact"] = "c" * 64
    assert _input_snapshot_matches(legacy_with_retired_input, current)
    assert not _input_snapshot_matches(
        {**legacy, "hashes": {"anchors": "b" * 64}}, current
    )
    assert not _input_snapshot_matches(
        {**legacy, "hashes": {"retired_feature_artifact": "c" * 64}}, current
    )


def test_canonical_identity_includes_archive_reader_and_session_dependencies(
    monkeypatch,
) -> None:
    assert "data/deep_dataset.py" in statsfusion_inputs.FEATURE_IMPLEMENTATION_FILES
    assert "data/session.py" in statsfusion_inputs.FEATURE_IMPLEMENTATION_FILES
    baseline = statsfusion_inputs.feature_implementation_identity()
    original = statsfusion_inputs.sha256_file

    def changed(path):
        if Path(path).name == "deep_dataset.py":
            return "f" * 64
        return original(path)

    monkeypatch.setattr(statsfusion_inputs, "sha256_file", changed)
    modified = statsfusion_inputs.feature_implementation_identity()
    assert modified["sha256"] != baseline["sha256"]


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


def test_v4_resume_records_and_allows_worktree_identity_change(tmp_path, monkeypatch) -> None:
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
    segment_archive = tmp_path / "segment.npz"
    segment_archive.write_bytes(b"segment")
    segment_frame = pd.DataFrame(
        {
            "session_id": ["session"],
            "segment_id": ["segment"],
            "segment_path": [str(segment_archive)],
        }
    )
    segment_frame.to_parquet(input_root / "indices" / "segments.parquet", index=False)
    canonical_root = output_root / "canonical_input_r3_2"
    canonical_root.mkdir(parents=True)
    canonical_anchors = canonical_root / "anchors.parquet"
    canonical_statistics = canonical_root / "statistics.parquet"
    canonical_events = canonical_root / "events_with_session.parquet"
    canonical_anchors.write_bytes(b"canonical anchors")
    canonical_statistics.write_bytes(b"canonical statistics")
    canonical_events.write_bytes(b"canonical events")
    preparation_identity = _preparation_identity(
        input_root / "indices" / "segments.parquet",
        input_root / "indices" / "events.parquet",
        segment_frame,
    )
    preparation_path = canonical_root / "preparation_identity.json"
    anchors_identity_path = canonical_root / "anchors.sha256.json"
    write_json_atomic(preparation_path, preparation_identity)
    write_json_atomic(
        anchors_identity_path,
        {
            "anchors_sha256": sha256_file(canonical_anchors),
            "preparation_identity_sha256": sha256_file(preparation_path),
        },
    )
    (canonical_root / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": "statsfusion-r3.2",
                "anchor_semantics": "right_endpoint_half_open",
                "feature_code_sha256": preparation_identity["feature_implementation"]["sha256"],
                "preparation_identity_sha256": sha256_file(preparation_path),
                "source_sha256": preparation_identity["source_sha256"],
                "output_sha256": {
                    "anchors": sha256_file(canonical_anchors),
                    "statistics": sha256_file(canonical_statistics),
                    "events": sha256_file(canonical_events),
                    "anchors_identity": sha256_file(anchors_identity_path),
                },
            }
        ),
        encoding="utf-8",
    )
    config = _r3_config()
    config["training"]["num_workers"] = 8
    config["feature_provenance"] = {
        "source_paths": ["experiments/baseline/fold_0/metadata.json"],
        "selection_note_path": None,
        "assumed_used_all_outer_folds": True,
    }
    identities = iter(
        (
            {"commit": "a", "dirty": False, "worktree_sha256": "one"},
            {"commit": "a", "dirty": False, "worktree_sha256": "one"},
            {"commit": "a", "dirty": True, "worktree_sha256": "two"},
            {"commit": "b", "dirty": True, "worktree_sha256": "three"},
        )
    )
    monkeypatch.setattr(v4_artifacts, "git_worktree_identity", lambda _root: next(identities))
    monkeypatch.setattr(
        "bme_eating.models.factory.build_state_model", lambda _config: torch.nn.Linear(1, 1)
    )
    legacy_snapshot = {"code_version": "v4.3", "protocol_version": "statsfusion-r3"}
    write_json_atomic(output_root / "input_snapshot_r3.json", legacy_snapshot)
    legacy_provenance_path = output_root / "feature_provenance.json"
    write_json_atomic(legacy_provenance_path, {"legacy": True})
    initial = initialize_v4_run(config, input_root, output_root, "strict-v4", 0, fresh=True)
    assert json.loads((output_root / "input_snapshot_r3.json").read_text(encoding="utf-8")) == (
        legacy_snapshot
    )
    assert (output_root / INPUT_SNAPSHOT_FILENAME).is_file()
    assert json.loads(legacy_provenance_path.read_text(encoding="utf-8")) == {"legacy": True}
    active_provenance_path = output_root / initial.payload["feature_provenance_path"]
    assert active_provenance_path.is_file()
    assert active_provenance_path == feature_provenance_artifact_path(
        output_root,
        json.loads(active_provenance_path.read_text(encoding="utf-8")),
    )
    legacy = dict(initial.payload)
    legacy.pop("runtime_config")
    legacy.pop("resume_config_sha256")
    write_json_atomic(initial.manifest_path, legacy)
    runtime_changed = deepcopy(config)
    runtime_changed["training"]["num_workers"] = 0
    resumed = initialize_v4_run(
        runtime_changed, input_root, output_root, "strict-v4", 0, fresh=False
    )
    assert resumed.payload["runtime_config"]["training.num_workers"] == 0
    runtime_history = resumed.payload["runtime_config_history"]
    assert len(runtime_history) == 1
    assert runtime_history[0]["previous"]["training.num_workers"] == 8
    assert runtime_history[0]["active"]["training.num_workers"] == 0
    original_config_hash = resumed.payload["resolved_config_sha256"]
    migrated = initialize_v4_run(
        runtime_changed, input_root, output_root, "strict-v4", 0, fresh=False
    )
    assert "recompute_pretraining_scalers" not in migrated.payload
    assert migrated.payload["resolved_config_sha256"] == original_config_hash
    assert len(migrated.payload["source_identity_history"]) == 1
    checkpoint_path = migrated.root / "crossfit" / "partition_0" / "state" / "seed_2026.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"model")
    resumed_with_checkpoint = initialize_v4_run(
        runtime_changed, input_root, output_root, "strict-v4", 0, fresh=False
    )
    assert resumed_with_checkpoint.payload["git"]["commit"] == "b"
    assert len(resumed_with_checkpoint.payload["source_identity_history"]) == 2
    assert checkpoint_path.read_bytes() == b"model"


def test_v4_snapshot_conflict_does_not_create_partial_run(tmp_path, monkeypatch) -> None:
    input_root = tmp_path / "v2"
    output_root = tmp_path / "v4"
    input_root.mkdir()
    output_root.mkdir()
    tracked = {}
    for name in ("anchors", "canonical_manifest"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}", encoding="utf-8")
        tracked[name] = path
    monkeypatch.setattr(v4_artifacts, "_tracked_inputs", lambda _config, _root: tracked)
    monkeypatch.setattr(
        v4_artifacts,
        "git_worktree_identity",
        lambda _root: {"commit": "test", "dirty": False, "worktree_sha256": "clean"},
    )
    write_json_atomic(output_root / INPUT_SNAPSHOT_FILENAME, {"conflict": True})

    run_root = output_root / "experiments" / "snapshot-conflict" / "fold_0"
    with pytest.raises(RuntimeError, match="input snapshot conflicts"):
        initialize_v4_run(_r3_config(), input_root, output_root, "snapshot-conflict", 0, fresh=True)

    assert not run_root.exists()


def test_v4_resume_hash_ignores_execution_only_settings() -> None:
    config = {
        "project": {"strict_resume_identity": True},
        "training": {
            "batch_size": 16,
            "learning_rate": 0.0003,
            "num_workers": 8,
            "inference_batch_size": 16,
        },
        "model": {"hidden_dim": 64},
    }
    runtime_changed = deepcopy(config)
    runtime_changed["training"].update(
        {
            "num_workers": 0,
            "inference_batch_size": 64,
            "inference_num_workers": 0,
        }
    )
    assert resume_config_hash(runtime_changed) == resume_config_hash(config)

    policy_changed = deepcopy(config)
    policy_changed["project"].update(
        {
            "enforce_git_identity_on_resume": False,
            "enforce_runtime_source_identity_on_resume": False,
        }
    )
    assert resume_config_hash(policy_changed) == resume_config_hash(config)

    result_changed = deepcopy(config)
    result_changed["training"]["learning_rate"] = 0.0001
    assert resume_config_hash(result_changed) != resume_config_hash(config)


def test_v4_rejects_blocked_predecessor_protocol(tmp_path) -> None:
    config = {
        "project": {
            "artifact_schema_version": "v4",
            "input_artifact_schema_version": "v2",
            "strict_resume_identity": True,
        },
        "experiment": {"protocol_version": "statsfusion-r1"},
        "features": {"artifact_name": "baseline"},
    }
    with pytest.raises(ValueError, match="statsfusion-r3.2"):
        v4_artifacts.current_v4_identity(config, tmp_path)


@pytest.mark.parametrize(
    "blocked_protocol",
    (
        "statsfusion-r0-blocked",
        "statsfusion-r1-blocked",
        "statsfusion-r2-blocked",
        "statsfusion-r3-blocked",
        "statsfusion-r3.1-blocked",
    ),
)
def test_v4_identity_rejects_all_blocked_predecessors(tmp_path, blocked_protocol) -> None:
    config = _r3_config()
    config["experiment"]["protocol_version"] = blocked_protocol
    with pytest.raises(ValueError, match="statsfusion-r3.2"):
        v4_artifacts.current_v4_identity(config, tmp_path / "v2")


def test_r3_freeze_manifest_hash_locks_promotion_and_selection_evidence(tmp_path) -> None:
    output_root = tmp_path / "outputs" / "v4"
    experiment_root = output_root / "experiments" / "winner"
    evidence_path = experiment_root / "fold_0" / "selection" / "selected_pipeline.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("{}", encoding="utf-8")
    git_identity = {"commit": "abc", "dirty": False}
    config = {"decoder": {"candidate_minimum_seconds": 3, "candidate_maximum_seconds": 14_400}}
    freeze_path = experiment_root / "freeze_manifest.json"
    payload = {
        "code_version": "v4.7.1",
        "protocol_version": "statsfusion-r3.2",
        "blocked_predecessors": [
            "statsfusion-r0-blocked",
            "statsfusion-r1-blocked",
            "statsfusion-r2-blocked",
            "statsfusion-r3-blocked",
            "statsfusion-r3.1-blocked",
        ],
        "selected_run": "winner",
        "locked_after_folds": [0, 1],
        "candidate_minimum_seconds": 3,
        "candidate_maximum_seconds": 14_400,
        "resume_config_sha256": "config-sha",
        "git": git_identity,
        "evidence_sha256": {
            "fold_0_selected_pipeline": {
                "relative_path": evidence_path.relative_to(output_root.parent).as_posix(),
                "sha256": sha256_file(evidence_path),
            }
        },
    }
    write_json_atomic(freeze_path, payload)
    assert (
        v4_artifacts.validate_v4_freeze_manifest(
            freeze_path,
            output_root,
            expected_resume_config_sha256="config-sha",
            expected_git=git_identity,
            config=config,
        )
        == payload
    )
    assert (
        v4_artifacts.validate_v4_freeze_manifest(
            freeze_path,
            output_root,
            expected_resume_config_sha256="config-sha",
            expected_git={"commit": "changed", "dirty": True},
            config=config,
        )
        == payload
    )
    evidence_path.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Gate evidence changed"):
        v4_artifacts.validate_v4_freeze_manifest(
            freeze_path,
            output_root,
            expected_resume_config_sha256="config-sha",
            expected_git=git_identity,
            config=config,
        )


def test_session_cache_key_uses_archive_content_not_size_or_mtime(tmp_path) -> None:
    archive = tmp_path / "segment.bin"
    archive.write_bytes(b"AAAA")
    original_stat = archive.stat()
    segments = pd.DataFrame(
        {
            "session_id": ["session"],
            "segment_id": ["segment"],
            "segment_path": [str(archive)],
        }
    )
    anchors = pd.DataFrame(
        {"session_id": ["session"], "subject_key": ["subject"], "timestamp_ms": [3_000]}
    )
    first_path = _session_cache_path(tmp_path, "session", anchors, segments)
    first_content_digest = _segment_archive_digest(segments)
    first_stat_digest = _segment_archive_stat_digest(segments)

    archive.write_bytes(b"BBBB")
    os.utime(archive, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    second_path = _session_cache_path(tmp_path, "session", anchors, segments)
    assert second_path != first_path
    assert _segment_archive_digest(segments) != first_content_digest
    assert _segment_archive_stat_digest(segments) == first_stat_digest


def _partial_canonical_fixture(tmp_path):
    input_root = tmp_path / "v2"
    output_root = tmp_path / "v4"
    indices = input_root / "indices"
    indices.mkdir(parents=True)
    archive = tmp_path / "segment.bin"
    archive.write_bytes(b"segment-content")
    segments = pd.DataFrame(
        {
            "session_id": ["session"],
            "segment_id": ["segment"],
            "segment_path": [str(archive)],
            "subject_key": ["subject"],
            "start_ms": [0],
            "end_ms": [6_000],
        }
    )
    events = pd.DataFrame(
        columns=[
            "event_id",
            "subject_key",
            "session_id",
            "start_ms",
            "end_ms",
            "valid_duration",
        ]
    )
    segments_path = indices / "segments.parquet"
    events_path = indices / "events.parquet"
    segments.to_parquet(segments_path, index=False)
    events.to_parquet(events_path, index=False)
    root = output_root / "canonical_input_r3_2"
    root.mkdir(parents=True)
    anchors = pd.DataFrame(
        {
            "segment_id": ["segment"],
            "session_id": ["session"],
            "subject_key": ["subject"],
            "timestamp_ms": [3_000],
            "state_loss_mask": [1.0],
        }
    )
    anchors_path = root / "anchors.parquet"
    anchors.to_parquet(anchors_path, index=False)
    identity = _preparation_identity(segments_path, events_path, segments)
    identity_path = root / "preparation_identity.json"
    write_json_atomic(identity_path, identity)
    write_json_atomic(
        root / "anchors.sha256.json",
        {
            "anchors_sha256": sha256_file(anchors_path),
            "preparation_identity_sha256": sha256_file(identity_path),
        },
    )
    return input_root, output_root, segments, events


@pytest.mark.parametrize("changed_source", ["events", "segments", "archive"])
def test_partial_canonical_resume_rejects_changed_source(tmp_path, changed_source) -> None:
    input_root, output_root, segments, events = _partial_canonical_fixture(tmp_path)
    if changed_source == "events":
        changed = events.copy()
        changed.loc[0] = ["event", "subject", "session", 0, 3_000, True]
        changed.to_parquet(input_root / "indices" / "events.parquet", index=False)
    elif changed_source == "segments":
        changed = segments.copy()
        changed.loc[0, "end_ms"] = 9_000
        changed.to_parquet(input_root / "indices" / "segments.parquet", index=False)
    else:
        archive = statsfusion_inputs.Path(str(segments.iloc[0].segment_path))
        original_stat = archive.stat()
        archive.write_bytes(b"altered-content")
        os.utime(archive, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(RuntimeError, match="source identity changed"):
        prepare_canonical_statsfusion_inputs(
            input_root,
            output_root,
            workers=1,
            fresh=False,
            resume=True,
        )


def test_partial_canonical_resume_reuses_matching_anchors(tmp_path, monkeypatch) -> None:
    input_root, output_root, _, _ = _partial_canonical_fixture(tmp_path)

    class ImmediateFuture:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class ImmediateExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, *args):
            return ImmediateFuture(function(*args))

    def build_statistics(anchors, _segments, cache_path):
        frame = anchors[["segment_id", "session_id", "subject_key", "timestamp_ms"]].copy()
        for column in STATS_FEATURE_COLUMNS:
            frame[column] = 0.0
        path = statsfusion_inputs.Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return str(path), None

    monkeypatch.setattr(statsfusion_inputs, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(statsfusion_inputs, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(statsfusion_inputs, "_build_session_statistics", build_statistics)
    before = sha256_file(output_root / "canonical_input_r3_2" / "anchors.parquet")
    manifest = prepare_canonical_statsfusion_inputs(
        input_root,
        output_root,
        workers=1,
        fresh=False,
        resume=True,
    )
    assert sha256_file(output_root / "canonical_input_r3_2" / "anchors.parquet") == before
    assert manifest == verify_canonical_statsfusion_inputs(input_root, output_root)
    archive = statsfusion_inputs.Path(
        pd.read_parquet(input_root / "indices" / "segments.parquet").iloc[0].segment_path
    )
    original_stat = archive.stat()
    archive.write_bytes(b"altered-content")
    os.utime(archive, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(RuntimeError, match="source identity changed"):
        verify_canonical_statsfusion_inputs(input_root, output_root)
