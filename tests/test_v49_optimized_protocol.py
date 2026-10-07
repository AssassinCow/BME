from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from bme_eating.config import load_config
from bme_eating.hierarchical_artifacts import write_json_atomic
from bme_eating.hierarchical_v4_artifacts import resume_config_hash
from bme_eating.integrated_v49 import (
    candidate_coverage,
    validate_v49_raw_imu_gate,
    write_failure_report,
    write_v49_raw_imu_gate,
)
from bme_eating.models.event_verifier_v4 import (
    EventVerifierV4,
    ProposalFeatureBatchV4,
    _raw_imu_proposal_snippets,
    fit_raw_imu_normalization,
)
from bme_eating.proposals_v4 import budget_v49_candidates
from bme_eating.training import hierarchical_v4_trainer as trainer
from bme_eating.v4_protocol import (
    V49_EXECUTION_SOURCE_FILES,
    execution_source_identity,
    protocol_git_identity,
    validate_r3_config,
)

PROJECT = Path(__file__).parents[1]


def _config():
    return load_config(PROJECT / "configs/hierarchical_v4_v49_integrated_optimized.yaml")


def _script(name, monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value", [None, 0.01])
def test_optimized_rejects_explicit_legacy_boundary_key(tmp_path, value):
    config = _config()
    validate_r3_config(config)
    config["promotion_gate"]["maximum_tp_to_fp_fraction"] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({name: value for name, value in config.items() if not name.startswith("_")}), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy fraction"):
        load_config(path)


def test_optimized_resume_allows_workers_but_binds_model():
    config = _config()
    original = resume_config_hash(config)
    config["training"].update({"num_workers": 2, "preprocess_num_workers": 2, "loader_prefetch_factor": 2})
    assert resume_config_hash(config) == original
    config["verifier"]["seeds"] = [2026]
    assert resume_config_hash(config) != original


def test_optimized_git_identity_allows_runtime_config_edits(monkeypatch):
    from bme_eating import reproducibility

    config = _config()
    states = iter([{"commit": "same", "worktree_sha256": "before"}, {"commit": "same", "worktree_sha256": "after"}])
    monkeypatch.setattr(reproducibility, "git_worktree_identity", lambda *_args: next(states))
    original = protocol_git_identity(PROJECT, config)
    config["training"]["num_workers"] = 2
    assert protocol_git_identity(PROJECT, config) == original


@pytest.mark.parametrize("relative", V49_EXECUTION_SOURCE_FILES)
def test_execution_identity_detects_script_changes(tmp_path, relative):
    for name in V49_EXECUTION_SOURCE_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("original", encoding="utf-8")
    package = tmp_path / "src/bme_eating/core.py"
    package.parent.mkdir(parents=True)
    package.write_text("core", encoding="utf-8")
    original = execution_source_identity(tmp_path)
    (tmp_path / relative).write_text("changed", encoding="utf-8")
    assert execution_source_identity(tmp_path)["sha256"] != original["sha256"]


def test_quota_second_pass_reserves_short_and_expansion_variants():
    frame = pd.DataFrame({
        "proposal_id": [f"candidate-{index}" for index in range(8)],
        "proposal_family_id": ["shared", "shared", "shared", "f3", "f4", "f5", "f6", "f7"],
        "source_mask": [64, 32, 1, 1, 1, 1, 1, 1],
        "generator_score": [100, 90, 80, 70, 60, 50, 40, 30],
        "coarse_start_ms": np.arange(8) * 1000,
        "coarse_end_ms": np.arange(8) * 1000 + np.array([100, 100, 10, 100, 100, 100, 100, 100]) * 1000,
    })
    result = budget_v49_candidates(frame, 4, {})
    assert {"candidate-0", "candidate-1", "candidate-2"} <= set(result.proposal_id)
    diagnostics = json.loads(result.budget_diagnostics.iloc[0])
    assert all(row["shortfall"] == 0 for row in diagnostics["quotas"].values())
    assert len(result) == 4


def test_coverage_reports_short_and_unknown_hand_events():
    truth = pd.DataFrame({"subject_key": ["subject", "subject"], "session_id": ["session", "session"],
                          "event_id": ["short", "long"], "start_ms": [0, 100000], "end_ms": [10000, 500000],
                          "hand_relation": ["unknown", "same"]})
    proposals = pd.DataFrame({"subject_key": ["subject"], "session_id": ["session"],
                              "coarse_start_ms": [0], "coarse_end_ms": [10000], "source_mask": [64], "proposal_family_id": ["family"]})
    coverage = candidate_coverage(proposals, truth)
    assert coverage["candidate_recall"] == 0.5
    assert coverage["strata"]["duration"]["short_le_30s"]["covered"] == 1
    assert coverage["strata"]["hand_relation"]["unknown"]["truth_count"] == 1


def test_candidate_gate_does_not_hide_a_failed_fold():
    config = _config()
    config["promotion_gate"]["candidate_coverage_minimum"] = 4
    config["promotion_gate"]["v48_frozen_baseline"].update(same_candidate_covered=4, different_candidate_covered=0)
    truth = pd.DataFrame([{"subject_key": f"s-{fold}", "session_id": "session", "event_id": f"e-{fold}",
                           "start_ms": 0, "end_ms": 10000, "hand_relation": "same", "outer_fold": fold} for fold in range(5)])
    proposals = truth.iloc[:4].rename(columns={"start_ms": "coarse_start_ms", "end_ms": "coarse_end_ms"}).assign(source_mask=64)
    report = trainer._v48_candidate_gate(proposals, truth, config)
    assert report["checks"]["overall_coverage"] is True
    assert report["checks"]["fold_candidate_recall_floor"] is False
    assert report["fold_coverage"][4]["truth_count"] == 1
    assert report["passed"] is False


def test_downstream_selector_filters_unqualified_then_prefers_event_f1():
    def epoch(number, f1, **extra):
        return {"epoch": number, "robust_event_f1": f1, "robust_candidate_recall": 0.8,
                "robust_subject_macro_soft_bce": 0.2, "robust_soft_bce_standard_error": 0.01,
                "robust_state_fragment_count": 5, "robust_ece": 0.01, "promotion_eligible": True, **extra}
    values = [epoch(3, 0.99, promotion_eligible=False), epoch(4, 0.2), epoch(5, 0.3), epoch(6, 0.3)]
    selected, passed = trainer._choose_conservative_state_epoch(values, values, minimum_delta=0.001, minimum_epoch=3,
                                                               require_qualified=True, downstream_proxy=True)
    assert passed and selected["epoch"] == 5


def test_raw_imu_is_causal_masked_and_training_normalized(tmp_path):
    timestamps = np.arange(0, 100000, 100, dtype=np.int64)
    values = np.tile(np.arange(6, dtype=np.float32), (len(timestamps), 1))
    values[:, 0] = np.sin(timestamps / 1000)
    mask = np.ones_like(values, dtype=bool)
    mask[:, 3:] = False
    raw = _raw_imu_proposal_snippets(timestamps, values, mask, 30000, 60000, normalize_locally=False)
    changed = values.copy()
    changed[timestamps > 75000] = 999
    np.testing.assert_array_equal(raw, _raw_imu_proposal_snippets(timestamps, changed, mask, 30000, 60000, normalize_locally=False))
    assert raw.shape == (3, 12, 300)
    assert not raw[:, 3:6].any() and not raw[:, 9:].any()
    features = ProposalFeatureBatchV4(np.array(["a", "b"]), np.zeros((2, 4, 3), np.float32),
                                     np.ones((2, 4), bool), np.zeros((2, 2), np.float32), raw_imu=np.stack([raw, raw]))
    model = EventVerifierV4(3, 2, _config()["verifier"]).eval()
    fit_raw_imu_normalization(model, features)
    batch = {name: torch.from_numpy(getattr(features, name).astype(np.float32) if name == "raw_imu" else getattr(features, name))
             for name in ("sequence", "sequence_mask", "scalar", "raw_imu")}
    with torch.no_grad():
        first = model(batch)["event_logit"]
        batch["raw_imu"][:, :, 3:6] = 999
        second = model(batch)["event_logit"]
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, atol=1e-6, rtol=0)
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    repeated = EventVerifierV4(3, 2, _config()["verifier"]).eval()
    repeated.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(first, repeated(batch)["event_logit"], atol=1e-6, rtol=0)


@pytest.mark.parametrize("deep_passed", [True, False])
def test_raw_gate_hashes_failed_and_passed_deep_evidence(tmp_path, deep_passed):
    config = _config()
    lineage = [{"outer_fold": fold, "seeds": [2026, 2027, 2028], "training_subjects": [f"train-{fold}"],
                "upstream_lineage": {"upstream_isolation": "fully_excluded_nested_state_oof_v1"},
                "raw_imu": {"shape": [2, 3, 12, 300], "normalization": "training_fold_robust_v1", "maximum_future_seconds": 15,
                            "normalization_training_subjects": [f"train-{fold}"]}} for fold in range(5)]
    write_json_atomic(tmp_path / "deep_crossfit.json", {"lineage": lineage})
    for name in ("deep_crossfit_scores.parquet", "resolved_config.yaml", "v49_deep_gate.json"):
        (tmp_path / name).write_text("evidence", encoding="utf-8")
    path = write_v49_raw_imu_gate(tmp_path, config, deep_passed)
    assert json.loads(path.read_text())["passed"] is deep_passed
    if deep_passed:
        validate_v49_raw_imu_gate(tmp_path)
        (tmp_path / "resolved_config.yaml").write_text("tampered", encoding="utf-8")
        with pytest.raises(RuntimeError, match="changed"):
            validate_v49_raw_imu_gate(tmp_path)
    else:
        with pytest.raises(RuntimeError, match="failed"):
            validate_v49_raw_imu_gate(tmp_path)


def test_boundary_entropy_search_and_locked_oof_refinement():
    config = _config()
    positive = pd.DataFrame({"coarse_start_ms": [10000], "coarse_end_ms": [30000], "truth_start_ms": [12000], "truth_end_ms": [28000]})
    selected, search = trainer._select_boundary_entropy(positive, (np.array([2.0]), np.array([-2.0]), np.array([0.6]), np.array([0.6])), config)
    assert selected == 0.65
    assert [row["threshold"] for row in search] == [0.55, 0.65, 0.75]
    accepted = positive.assign(subject_key="subject", session_id="session", proposal_id="id")
    scores = pd.DataFrame({"proposal_id": ["id"], "start_offset_seconds": [2.0], "end_offset_seconds": [-2.0],
                           "start_entropy": [0.6], "end_entropy": [0.6], "locked_entropy_threshold": [0.55]})
    refined = trainer._refine_locked_boundary(accepted, scores, 3)
    assert refined.refined_start_ms.tolist() == [10000]
    assert refined.refined_end_ms.tolist() == [30000]


@pytest.mark.parametrize("failure", [None, "oom", "interrupt"])
def test_runner_worker_ten_writes_terminal_status_and_can_restart(tmp_path, monkeypatch, failure):
    runner = _script("run_hierarchical_v49_integrated", monkeypatch)
    monkeypatch.setattr(runner, "_monitor_resources", lambda *_args: None)
    project = tmp_path / "project"
    script = project / "scripts" / "stage.py"
    script.parent.mkdir(parents=True)
    script.write_text("raise MemoryError('OOM')" if failure == "oom" else "print('ok')", encoding="utf-8")
    root = tmp_path / "run"
    original_popen = runner.subprocess.Popen
    if failure == "interrupt":
        def interrupt(*_args, **_kwargs):
            raise KeyboardInterrupt()
        monkeypatch.setattr(runner.subprocess, "Popen", interrupt)
    if failure:
        with pytest.raises(KeyboardInterrupt if failure == "interrupt" else RuntimeError):
            runner.execute_stage(root, project, "config.yaml", dict(os.environ), "stage.py", ["--workers", "10"], "prepare", {"preprocess_num_workers": 10})
        status = json.loads((root / "execution_status.json").read_text())
        assert status["status"] == ("PAUSED" if failure == "interrupt" else "FAILED")
        if failure == "oom":
            assert status["failure_kind"] == "OOM"
        monkeypatch.setattr(runner.subprocess, "Popen", original_popen)
        script.write_text("print('ok')", encoding="utf-8")
    runner.execute_stage(root, project, "config.yaml", dict(os.environ), "stage.py", ["--workers", "10"], "prepare", {"preprocess_num_workers": 10})
    status = json.loads((root / "execution_status.json").read_text())
    assert status["stage_completed"] is True
    assert status["runtime_parameters"]["preprocess_num_workers"] == 10
    assert status["return_code"] == 0


@pytest.mark.parametrize("missing", ["nested", "boundary", "source", "raw_gate"])
def test_replay_fails_closed_on_missing_deployment_evidence(tmp_path, monkeypatch, missing):
    replay = _script("replay_hierarchical_v4_raw_session", monkeypatch)
    from bme_eating.hierarchical_artifacts import sha256_file

    config = _config()
    (tmp_path / "resolved_config.yaml").write_text("config", encoding="utf-8")
    monkeypatch.setattr(replay, "execution_source_identity", lambda *_args: {"sha256": "source"})
    monkeypatch.setattr(replay, "execution_environment_identity", lambda: {"python": "test"})
    lock = {"protocol": "v49_deployment_lock_v2", "execution_source_identity": {"sha256": "source"},
            "execution_environment": {"python": "test"}, "config_sha256": sha256_file(tmp_path / "resolved_config.yaml"),
            "nested_state_protocol": "fully_excluded_nested_state_oof_v1", "boundary_passed": True,
            "state_seeds": [2026], "verifier_seeds": [2026, 2027, 2028]}
    if missing == "nested":
        lock["nested_state_protocol"] = None
    elif missing == "boundary":
        lock["boundary_passed"] = False
    elif missing == "source":
        lock["execution_source_identity"] = {"sha256": "changed"}
    write_json_atomic(tmp_path / "v49_deployment_protocol.json", lock)
    write_json_atomic(tmp_path / "v49_deep_gate.json", {"protocol": "v49_deep_promotion_v1", "passed": True})
    write_json_atomic(tmp_path / "v49_raw_imu_gate.json", {"protocol": "v49_raw_imu_nested_crossfit_gate_v2", "passed": missing != "raw_gate"})
    with pytest.raises(RuntimeError):
        replay.validate_replay_lock(tmp_path, config)


def test_failed_export_quarantines_bundle_and_retains_evidence(tmp_path):
    bundle = tmp_path / "model_bundle"
    bundle.mkdir()
    (bundle / "model.pt").write_text("old", encoding="utf-8")
    write_json_atomic(tmp_path / "lineage.json", {"source": "evidence"})
    report_path = write_failure_report(tmp_path, "export", RuntimeError("Deep gate failed"))
    report = json.loads(report_path.read_text())
    assert not bundle.exists()
    assert report["quarantined_bundles"]
    assert (tmp_path / report["quarantined_bundles"][0] / "model.pt").is_file()
