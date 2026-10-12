from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from bme_eating.config import load_config
from bme_eating.data.joint_sessions import JointInputs
from bme_eating.data.stats_fusion_inputs import canonical_input_paths
from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_evaluation import NonnegativePlatt, apply_online_budget, commit_candidates
from bme_eating.joint_protocol import (
    PROTOCOL,
    digest,
    initialize_run,
    protocol_identities,
    run_lock,
    save_torch_atomic,
    validate_gate_structure,
    verify_artifact_hashes,
    verify_internal_manifest,
    verify_oof_lineage,
)
from bme_eating.joint_stream import (
    FrozenSourceCDF,
    ImmutableCommitter,
    JointCandidate,
    OnlineBudget,
)
from bme_eating.joint_workflow import verify_outer_completion
from bme_eating.training.statsfusion_joint_trainer import boundary_pair_gate


@pytest.fixture
def config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v410_joint_event.yaml")


def candidate(start=0, end=9000, identity="proposal"):
    return JointCandidate("subject", "session", start, end, end + 30_000, "proposal", 3, .8,
        start, end, identity, identity, identity, "hash", "model", {}, end + 30_000,
        event_logit=2, predicted_quality=.8)


def test_boundary_replacement_is_not_a_paired_tp(config):
    truth = pd.DataFrame({"subject_key": ["subject"], "session_id": ["session"],
        "event_id": ["event"], "start_ms": [3000], "end_ms": [12000], "hand_relation": ["same"]})
    windows = pd.DataFrame({"subject_key": "subject", "session_id": "session",
        "timestamp_ms": np.arange(3000, 45001, 3000), "valid": 1})
    before = candidate()
    after = candidate(3000, 12000, "replacement")
    report = boundary_pair_gate([before], [after], truth, truth.iloc[:0], windows, config, require_sixty=False)
    assert report["tp_to_fp_rate"] == 1
    assert report["paired_positive_events"] == 0
    assert not report["passed"]


def test_boundary_clipping_includes_candidates_rejected_by_threshold(config):
    truth = pd.DataFrame({"subject_key": ["subject"], "session_id": ["session"],
        "event_id": ["event"], "start_ms": [3000], "end_ms": [12000], "hand_relation": ["same"]})
    windows = pd.DataFrame({"subject_key": "subject", "session_id": "session",
        "timestamp_ms": np.arange(3000, 45001, 3000), "valid": 1})
    before = candidate()
    after = replace(before, start_ms=3000, end_ms=12000, refined=True, unrefined_start_ms=0,
                    unrefined_end_ms=9000)
    rejected = replace(after, proposal_id="rejected", boundary_clipped=True)
    report = boundary_pair_gate([before], [after], truth, truth.iloc[:0], windows, config,
                                require_sixty=False, refined_candidates=[after, rejected])
    assert report["paired_positive_events"] == 1
    assert report["clipping_rate"] == .5
    assert report["diagnostic_candidate_count"] == 2
    assert report["changed_endpoints"] == 4
    assert not report["passed"]


def test_boundary_neighbor_conflict_restores_coarse_probability_and_coordinates():
    previous = candidate(0, 9000, "previous")
    current = candidate(12000, 21000, "current")
    refined = replace(current, start_ms=6000, refined=True, unrefined_start_ms=12000,
        unrefined_end_ms=21000, coarse_event_logit=-10, coarse_quality=.8)
    calibrator = NonnegativePlatt(1, 0, True)
    chosen = commit_candidates([previous, refined], calibrator, .1, .7)
    assert [item.proposal_id for item in chosen] == ["previous"]
    chosen = commit_candidates([previous, refined], calibrator, 0, .7)
    assert chosen[1].start_ms == current.start_ms
    assert chosen[1].boundary_conflict
    assert chosen[1].rank_score < .001
    assert chosen[1].acceptance_score == chosen[1].rank_score
    assert chosen[1].quality_score == chosen[1].predicted_quality
    assert not chosen[1].refined


def test_committed_candidate_is_independent_of_later_search_mutations():
    item = candidate()
    item.rank_score = .5
    committer = ImmutableCommitter(.1, .5)
    committer.submit([item])
    item.rank_score = .9
    item.generation_parameters["changed"] = True
    assert committer.committed[0].rank_score == .5
    assert committer.committed[0].generation_parameters == {}


@pytest.mark.parametrize("field", ["event_logit", "predicted_quality", "rank_score", "missing_fraction"])
def test_nonfinite_candidates_fail_before_budget_or_acceptance(field):
    item = candidate()
    setattr(item, field, float("nan"))
    with pytest.raises(RuntimeError, match="Nonfinite"):
        ImmutableCommitter(.1, .5).submit([item])
    tracker = OnlineBudget(20, {"proposal": .2, "short": .15, "expansion": .15})
    tracker.observe(3600, 1)
    with pytest.raises(RuntimeError, match="Nonfinite"):
        tracker.select([item], FrozenSourceCDF({"proposal": [0, 1]}, ["fit"]))
    assert tracker.spent == 0


def test_budget_rejects_duplicate_proposal_across_decisions():
    tracker = OnlineBudget(20, {"proposal": .2, "short": .15, "expansion": .15})
    tracker.observe(3600, 1)
    mapping = FrozenSourceCDF({"proposal": [0, 1]}, ["fit"])
    tracker.select([candidate()], mapping)
    with pytest.raises(RuntimeError, match="Duplicate"):
        tracker.select([candidate()], mapping)


def test_offline_budget_matches_incremental_exposure_with_missing_bins():
    windows = pd.DataFrame({"subject_key": "subject", "session_id": "session",
        "timestamp_ms": np.arange(3000, 600001, 3000), "valid": np.tile([1, 0], 100)})
    items = [replace(candidate(0, decision - 30000, str(decision)), decision_time_ms=decision,
                     max_input_time_ms=decision) for decision in (60000, 120000, 600000)]
    mapping = FrozenSourceCDF({"proposal": [0, 1]}, ["fit"])
    quotas = {"proposal": .2, "short": .15, "expansion": .15}
    chosen, audit = apply_online_budget(items, windows, {"model": mapping}, 20, quotas)
    tracker = OnlineBudget(20, quotas)
    incremental = []
    for row in windows.itertuples(index=False):
        tracker.observe(3, row.valid)
        current = [item for item in items if item.decision_time_ms == row.timestamp_ms]
        if current:
            incremental.extend(tracker.select(current, mapping))
    assert [item.proposal_id for item in chosen] == [item.proposal_id for item in incremental]
    assert audit[0]["records"] == tracker.audit


def test_artifact_hash_rejects_path_escape_and_run_lock_is_exclusive(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    external = tmp_path / "outside.txt"
    external.write_text("outside", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsafe path"):
        verify_artifact_hashes(root, {"../outside.txt": sha256_file(external)})
    with run_lock(root), pytest.raises(RuntimeError, match="concurrent resume"), run_lock(root):
        pytest.fail("Concurrent writer entered the run")
    with run_lock(root):
        pass


@pytest.mark.parametrize("name", [".", ".."])
def test_run_name_cannot_escape_output_directory(tmp_path, config, name):
    with pytest.raises(ValueError, match="Unsafe run"):
        initialize_run(config, tmp_path, tmp_path, name, {}, fresh=True, skip_s0=True, historical_reference=None)


def test_internal_oof_resume_requires_complete_identity_bound_manifest(tmp_path, config):
    subjects = {f"subject_{index}" for index in range(6)}
    outer = {"outer"}
    data = {"dataset": "fixture"}
    identities = protocol_identities(config, data)
    lineages = []
    for partition in range(3):
        prediction = set(sorted(subjects)[partition * 2:partition * 2 + 2])
        fit = subjects - prediction
        model_identity = digest({"run": "run", "training": sorted(fit), "prediction": sorted(prediction),
                                 "outer": sorted(outer), "partition": partition})
        lineages.append({"partition": partition, "model_identity": model_identity, "epoch": 8,
            "selector_promotion_eligible": True,
            "training_subjects": sorted(fit), "prediction_subjects": sorted(prediction),
            "source_training_subjects": sorted(fit), "normalization_subjects": sorted(fit),
            "selector_fit_subjects": sorted(fit)[:-1], "selector_subjects": sorted(fit)[-1:],
            "outer_excluded_subjects": sorted(outer), "checkpoint_qualified": True, **identities})
        save_torch_atomic(tmp_path / f"partition_{partition}/retrain/latest.pt", {"model": {"weight": torch.ones(1)},
            "identity": model_identity, "epoch": 8, "epoch_complete": True, "promotion_eligible": True,
            "clipping_fraction": 0, "training_subjects": sorted(fit), "excluded_subjects": sorted(outer | prediction),
            "preprocessing": {"training_subjects": sorted(fit)}, **identities})
        save_torch_atomic(tmp_path / f"partition_{partition}/selection/selected.pt", {"model": {"weight": torch.ones(1)},
            "identity": model_identity, "epoch": 8, "epoch_complete": True, "promotion_eligible": True,
            "clipping_fraction": 0, "training_subjects": sorted(fit)[:-1],
            "excluded_subjects": sorted(outer | prediction | set(sorted(fit)[-1:])),
            "preprocessing": {"training_subjects": sorted(fit)[:-1]}, **identities})
    point = {"lineage": lineages}
    write_json_atomic(tmp_path / "operating_point.json", point)
    with pytest.raises(RuntimeError, match="manifest is missing"):
        verify_internal_manifest(tmp_path, config, data, subjects, outer, "run")
    for relative in ("internal_oof.pt", "internal_oof_windows.parquet", "search.json", "boundary_crossfit.json"):
        (tmp_path / relative).write_bytes(b"test evidence")
    evidence = {path.relative_to(tmp_path).as_posix(): sha256_file(path) for path in tmp_path.rglob("*") if path.is_file()}
    write_json_atomic(tmp_path / "internal_manifest.json", {"protocol": PROTOCOL, "run_identity": "run",
        "training_subjects": sorted(subjects), "outer_excluded_subjects": sorted(outer),
        **identities, "evidence_sha256": evidence})
    assert verify_internal_manifest(tmp_path, config, data, subjects, outer, "run") == point
    with pytest.raises(RuntimeError, match="identity mismatch"):
        verify_internal_manifest(tmp_path, config, data, subjects, outer, "different_run")
    invalid = copy.deepcopy(lineages)
    invalid[1]["model_identity"] = invalid[0]["model_identity"]
    with pytest.raises(RuntimeError, match="duplicated"):
        verify_oof_lineage(invalid, subjects, outer)
    (tmp_path / "internal_oof.pt").write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="checksum"):
        verify_internal_manifest(tmp_path, config, data, subjects, outer, "run")


def test_evaluation_json_alone_does_not_complete_outer_fold(tmp_path, config):
    write_json_atomic(tmp_path / "fold_0" / "evaluation.json", {"checkpoint_qualified": True})
    with pytest.raises(RuntimeError, match="completion evidence is missing"):
        verify_outer_completion(SimpleNamespace(), config, tmp_path, 0)


def test_promotable_boolean_cannot_bypass_formal_checks():
    with pytest.raises(RuntimeError, match="formal checks"):
        validate_gate_structure({"promotable": True, "checks": {"f1": True}})


@pytest.mark.parametrize("logits,targets", [([float("nan")], [1]), ([0, 1], [0]), ([0, 1], [0, .5])])
def test_platt_rejects_nonfinite_or_misaligned_targets(logits, targets):
    with pytest.raises(ValueError):
        NonnegativePlatt.fit(np.asarray(logits), np.asarray(targets))


@pytest.mark.parametrize("changed", ["canonical", "raw"])
def test_inputs_detect_archive_and_canonical_mutations_after_audit(tmp_path, changed):
    input_root, canonical = tmp_path / "input", tmp_path / "canonical"
    paths = canonical_input_paths(canonical)
    paths["root"].mkdir(parents=True)
    paths["anchors"].write_bytes(b"canonical fixture")
    archive = tmp_path / "raw.npz"
    archive.write_bytes(b"raw fixture")
    (input_root / "indices").mkdir(parents=True)
    pd.DataFrame({"subject_key": ["subject"], "session_id": ["session"], "segment_path": [str(archive)]}).to_parquet(
        input_root / "indices" / "segments.parquet", index=False)
    inputs = JointInputs(input_root, canonical, {"subject": 0}, {"anchors": sha256_file(paths["anchors"])})
    (paths["anchors"] if changed == "canonical" else archive).write_bytes(b"changed identity")
    with pytest.raises(RuntimeError, match="input identity changed|sensor archive changed"):
        inputs.frames({"subject"}, labels=True)
