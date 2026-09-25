from __future__ import annotations

import json
import os

import pandas as pd
import pytest
import torch

import bme_eating.data.stats_fusion_inputs as statsfusion_inputs
import bme_eating.hierarchical_v4_artifacts as v4_artifacts
from bme_eating.data.stats_fusion_inputs import (
    _preparation_identity,
    _segment_archive_digest,
    _segment_archive_stat_digest,
    _session_cache_path,
    prepare_canonical_statsfusion_inputs,
    verify_canonical_statsfusion_inputs,
)
from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
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
    canonical_root = output_root / "canonical_input"
    canonical_root.mkdir(parents=True)
    canonical_anchors = canonical_root / "anchors.parquet"
    canonical_statistics = canonical_root / "statistics.parquet"
    canonical_anchors.write_bytes(b"canonical anchors")
    canonical_statistics.write_bytes(b"canonical statistics")
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
                "protocol_version": "statsfusion-r2",
                "anchor_semantics": "right_endpoint_half_open",
                "preparation_identity_sha256": sha256_file(preparation_path),
                "source_sha256": preparation_identity["source_sha256"],
                "output_sha256": {
                    "anchors": sha256_file(canonical_anchors),
                    "statistics": sha256_file(canonical_statistics),
                    "anchors_identity": sha256_file(anchors_identity_path),
                },
            }
        ),
        encoding="utf-8",
    )
    config = {
        "project": {
            "artifact_schema_version": "v4",
            "input_artifact_schema_version": "v2",
            "strict_resume_identity": True,
        },
        "data": {"subject_folds": 5},
        "experiment": {"protocol_version": "statsfusion-r2"},
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


def test_v4_rejects_r1_protocol(tmp_path) -> None:
    config = {
        "project": {
            "artifact_schema_version": "v4",
            "input_artifact_schema_version": "v2",
            "strict_resume_identity": True,
        },
        "experiment": {"protocol_version": "statsfusion-r1"},
        "features": {"artifact_name": "baseline"},
    }
    with pytest.raises(ValueError, match="statsfusion-r2"):
        v4_artifacts.current_v4_identity(config, tmp_path)


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
        columns=["event_id", "subject_key", "start_ms", "end_ms", "valid_duration"]
    )
    segments_path = indices / "segments.parquet"
    events_path = indices / "events.parquet"
    segments.to_parquet(segments_path, index=False)
    events.to_parquet(events_path, index=False)
    root = output_root / "canonical_input"
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
        changed.loc[0] = ["event", "subject", 0, 3_000, True]
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
    before = sha256_file(output_root / "canonical_input" / "anchors.parquet")
    manifest = prepare_canonical_statsfusion_inputs(
        input_root,
        output_root,
        workers=1,
        fresh=False,
        resume=True,
    )
    assert sha256_file(output_root / "canonical_input" / "anchors.parquet") == before
    assert manifest == verify_canonical_statsfusion_inputs(input_root, output_root)
    archive = statsfusion_inputs.Path(
        pd.read_parquet(input_root / "indices" / "segments.parquet").iloc[0].segment_path
    )
    original_stat = archive.stat()
    archive.write_bytes(b"altered-content")
    os.utime(archive, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(RuntimeError, match="source identity changed"):
        verify_canonical_statsfusion_inputs(input_root, output_root)
