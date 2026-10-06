from __future__ import annotations

import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from bme_eating.config import load_config
from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.hierarchical_v4_artifacts import resume_config_hash
from bme_eating.integrated_v49 import (
    _evaluation_cohort_identity,
    candidate_coverage,
    validate_proposal_lineage,
    validate_sensor_only_reference,
    validate_v49_gate_manifest,
    write_failure_report,
)
from bme_eating.models.event_verifier_v4 import (
    EventVerifierV4,
    HardNegativeBatchSampler,
    _proposal_sampling_metadata,
    build_proposal_features_v4,
)
from bme_eating.proposals_v4 import generate_event_candidates_v4
from bme_eating.reproducibility import git_worktree_identity
from bme_eating.structured_decoder import FixedLagSemiMarkovDecoder, TruncatedLogNormalDurationPrior
from bme_eating.training import hierarchical_v4_trainer as trainer
from bme_eating.v4_protocol import (
    BLOCKED_PREDECESSORS,
    IGNORE_PROTOCOL,
    OBSERVATION_GAP_PROTOCOL,
    PROTOCOL_VERSION,
    RUNTIME_SOURCE_BINDING,
    RUNTIME_SOURCE_FILES,
    runtime_source_identity,
)


def _config():
    return load_config(Path(__file__).parents[1] / "configs/hierarchical_v4_v49_integrated_repair.yaml")


def _epoch(epoch=3, **changes):
    return {"epoch": epoch, "robust_candidate_recall": 0.9,
            "robust_subject_macro_soft_bce": 0.2, "robust_soft_bce_standard_error": 0.01,
            "robust_state_fragment_count": 1.0, "robust_ece": 0.01, **changes}


@pytest.mark.parametrize("epochs,qualified", [
    ([_epoch(1)], [_epoch(1)]),
    ([_epoch()], []),
    ([_epoch(promotion_eligible=False)], [_epoch(promotion_eligible=False)]),
    ([_epoch(robust_ece=float("nan"))], [_epoch(robust_ece=float("nan"))]),
])
def test_v49_selector_rejects_unqualified_epochs(epochs, qualified):
    with pytest.raises((ValueError, RuntimeError)):
        trainer._choose_conservative_state_epoch(
            epochs, qualified, minimum_delta=0.0, minimum_epoch=3, require_qualified=True,
        )


def test_v49_resume_hash_binds_runtime_settings():
    config = _config()
    expected = resume_config_hash(config)
    config["training"]["num_workers"] += 1
    assert resume_config_hash(config) != expected


def test_nonfinite_checkpoint_is_not_written(tmp_path):
    checkpoint = tmp_path / "state.pt"
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer._save_torch_atomic(checkpoint, {"model": {"weight": torch.tensor([float("nan")])}})
    assert not checkpoint.exists()


def test_checkpoint_rejects_subject_lineage_leak(tmp_path):
    checkpoint = tmp_path / "state.pt"
    with pytest.raises(RuntimeError, match="lineage"):
        trainer._save_torch_atomic(checkpoint, {
            "model": {"weight": torch.ones(1)}, "training_subjects": ["subject"],
            "globally_excluded_subjects": ["subject"],
        })
    assert not checkpoint.exists()


@pytest.mark.parametrize("issue", ["missing_identity", "config_hash", "clipping", "finite"])
def test_v49_promotion_checkpoint_requires_qualification(monkeypatch, issue):
    config = _config()
    model = torch.nn.Linear(1, 1)
    if issue == "missing_identity":
        with pytest.raises(RuntimeError, match="identity"):
            trainer._state_promotion_evidence(model, config)
        return
    if issue == "config_hash":
        config["_v49_checkpoint_identity"] = {"input_hashes": {"events": "hash"}, "resolved_config_sha256": "wrong"}
        with pytest.raises(RuntimeError, match="config hash"):
            trainer._state_promotion_evidence(model, config)
        return
    monkeypatch.setattr(trainer, "_validate_state_checkpoint_binding", lambda *_args: None)
    model._bme_training_monitor = {
        "finite_gate_passed": issue != "finite", "clipping_fraction": 0.21 if issue == "clipping" else 0.0,
    }
    with pytest.raises(RuntimeError, match="qualification"):
        trainer._state_promotion_evidence(model, config)


def _event_frames():
    truth = pd.DataFrame([{"subject_key": "truth", "session_id": "session", "event_id": "event",
                           "start_ms": 0, "end_ms": 10_000, "hand_relation": "same"}])
    proposals = pd.DataFrame([
        {"subject_key": "truth", "session_id": "session", "proposal_id": "positive",
         "coarse_start_ms": 0, "coarse_end_ms": 10_000, "source_mask": 1,
         "final_score": 0.99, "locked_acceptance": True},
        {"subject_key": "fp-only", "session_id": "session", "proposal_id": "negative",
         "coarse_start_ms": 0, "coarse_end_ms": 10_000, "source_mask": 1,
         "final_score": 0.99, "locked_acceptance": False},
    ])
    windows = pd.DataFrame({"subject_key": ["truth", "fp-only", "window-only"],
                            "session_id": ["session"] * 3, "timestamp_ms": [10_000] * 3})
    return truth, truth.iloc[:0].copy(), proposals, windows


def test_v49_locked_evaluation_does_not_tune_on_holdout(monkeypatch):
    truth, ignore, scores, windows = _event_frames()
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Outer holdout operating-point search was called")
    monkeypatch.setattr(trainer, "_best_verifier_operating_point", forbidden)
    report = {"lineage": [{"locked_operating_point": {
        "acceptance_threshold": threshold, "nms_iou_threshold": 0.5,
    }} for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]]}
    diagnostics, accepted = trainer._v49_locked_deep_diagnostics(scores, truth, ignore, windows, report)
    assert diagnostics["point"]["f1"] == 1.0
    assert diagnostics["point"]["acceptance_threshold"] == pytest.approx(0.3)
    assert accepted.proposal_id.tolist() == ["positive"]


def test_v49_candidate_coverage_uses_truth_events():
    truth, _, proposals, _ = _event_frames()
    result = candidate_coverage(proposals, truth)
    assert result["covered"] == 1
    assert result["same_covered"] == 1
    assert result["source_wise_covered"] == {"1": 1}


def test_bootstrap_retains_prediction_only_subject():
    truth, ignore, proposals, _ = _event_frames()
    result = trainer._v48_subject_bootstrap(proposals, truth, ignore, replicates=20)
    assert result["subjects"] == ["fp-only", "truth"]


def test_hard_negative_strata_use_candidate_duration_and_gyro_quality():
    _, _, proposals, _ = _event_frames()
    proposals["proposal_family_id"] = ["family-a", "family-b"]
    proposals.loc[1, "coarse_end_ms"] = 600_000
    windows = pd.DataFrame({"subject_key": ["truth", "fp-only"], "session_id": ["session"] * 2,
                            "timestamp_ms": [3_000] * 2, "gyro_valid_fraction": [1.0, 0.0],
                            "hand_relation": ["different", "same"]})
    metadata = _proposal_sampling_metadata(proposals, windows)
    assert metadata["duration_bin"][0] != metadata["duration_bin"][1]
    assert metadata["gyro_missingness_bin"].tolist() == ["observed", "missing"]
    assert metadata["hand_relation"].tolist() == ["different", "same"]


def test_sampler_rejects_family_crossing_subjects():
    _, _, proposals, windows = _event_frames()
    proposals["proposal_family_id"] = "leaked-family"
    with pytest.raises(ValueError, match="family crosses"):
        HardNegativeBatchSampler(
            np.asarray(["hard_false_positive"] * 2), batch_size=2, steps_per_epoch=1, seed=2026,
            ratios={"positive": 0.0, "near_miss": 0.0, "hard_false_positive": 1.0, "random_background": 0.0},
            sampling_metadata=_proposal_sampling_metadata(proposals, windows),
        )


def _locked_manifest(tmp_path):
    root = tmp_path / "experiments" / "run"
    identity = {"resolved_config_sha256": "config", "input_hashes": {"events": "data"}, "git": {"commit": "source"}}
    evidence = {}
    for fold in range(5):
        path = root / f"fold_{fold}" / "run_manifest.json"
        write_json_atomic(path, {"fold": fold})
        evidence[path.relative_to(root).as_posix()] = sha256_file(path)
    reference = tmp_path / "final" / "reference" / "deep_crossfit.json"
    write_json_atomic(reference, {"f1": 0.5375})
    manifest = {"protocol": "v49_gate_manifest_v1", "candidate_run": "run", "folds": list(range(5)),
                "passed": True, "crossfold_gate": {"passed": True}, "identity": identity,
                "evidence_sha256": evidence,
                "reference_evidence_sha256": {reference.relative_to(tmp_path).as_posix(): sha256_file(reference)}}
    write_json_atomic(root / "v49_gate_manifest.json", manifest)
    return root, identity, manifest


@pytest.mark.parametrize("tampering", ["missing_fold", "changed_fold", "identity", "failed", "reference"])
def test_v49_gate_lock_rejects_incomplete_or_changed_evidence(tmp_path, tampering):
    root, identity, manifest = _locked_manifest(tmp_path)
    if tampering == "changed_fold":
        write_json_atomic(root / "fold_4" / "run_manifest.json", {"changed": True})
    elif tampering == "missing_fold":
        manifest["evidence_sha256"].pop("fold_4/run_manifest.json")
    elif tampering == "identity":
        identity = {**identity, "input_hashes": {"events": "changed"}}
    elif tampering == "reference":
        write_json_atomic(tmp_path / "final" / "reference" / "deep_crossfit.json", {"changed": True})
    else:
        manifest["passed"] = False
    write_json_atomic(root / "v49_gate_manifest.json", manifest)
    with pytest.raises(RuntimeError):
        validate_v49_gate_manifest(tmp_path, "run", identity)


def test_v49_gate_lock_accepts_complete_evidence(tmp_path):
    _, identity, _ = _locked_manifest(tmp_path)
    assert validate_v49_gate_manifest(tmp_path, "run", identity)["passed"] is True


def test_v49_failure_archives_diagnostics_without_bundle(tmp_path):
    write_json_atomic(tmp_path / "diagnostics" / "lineage.json", {"subject": "excluded"})
    path = write_failure_report(tmp_path, "deep", RuntimeError("promotion failed"))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["deployable"] is False
    assert (tmp_path / report["evidence_archive"] / "diagnostics" / "lineage.json").is_file()
    assert not (tmp_path / "model_bundle").exists()


@pytest.mark.parametrize("failed_gate", ["deep", "boundary"])
def test_v49_failed_formal_gate_blocks_export(monkeypatch, tmp_path, failed_gate):
    from bme_eating import hierarchical_v4_export as export

    final_root = tmp_path / "final" / "run"
    config = {"decoder": {"candidate_protocol": "v4.9"}, "model": {}, "verifier": {}}
    write_json_atomic(final_root / "final_manifest.json", {"stage": "COMPLETE", "protocol_version": PROTOCOL_VERSION})
    selection = {
        "protocol_version": PROTOCOL_VERSION, "code_version": "v4.9", "candidate_protocol": "v4.9",
        "blocked_predecessors": list(BLOCKED_PREDECESSORS), "selection_source": "pooled_outer_oof",
        "ignore_protocol_version": IGNORE_PROTOCOL, "observation_gap_protocol": OBSERVATION_GAP_PROTOCOL,
        "runtime_source_binding": RUNTIME_SOURCE_BINDING,
    }
    if failed_gate == "boundary":
        selection.update({"boundary_enabled": True, "boundary_selection_diagnostics": {"selected": {"passed": False}}})
    write_json_atomic(final_root / "selected_pipeline.json", selection)
    (final_root / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    write_json_atomic(tmp_path / "experiments" / "run" / "v49_gate_manifest.json", {
        "protocol": "v49_gate_manifest_v1", "passed": True,
    })
    write_json_atomic(final_root / "v49_deep_gate.json", {
        "protocol": "v49_deep_promotion_v1", "passed": failed_gate != "deep",
        "evidence_sha256": {"resolved_config.yaml": sha256_file(final_root / "resolved_config.yaml")},
    })
    monkeypatch.setattr(export, "REQUIRED_MODEL_FILES", ())
    monkeypatch.setattr(export, "validate_v49_gate_manifest", lambda *_args: {})
    with pytest.raises(RuntimeError, match="Deep promotion failed" if failed_gate == "deep" else "Boundary"):
        export.export_hierarchical_v4_bundle(tmp_path, final_root, fresh=True, resume=False)
    assert not (final_root / "model_bundle").exists()


def test_v49_lineage_requires_parent_evidence():
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_proposal_lineage(pd.DataFrame({"proposal_id": ["legacy"]}))


def _synthetic_v49_candidates():
    config = _config()
    timestamps = np.arange(3_000, 1_803_000, 3_000)
    active = (timestamps >= 900_000) & (timestamps <= 990_000)
    windows = pd.DataFrame({
        "subject_key": "subject", "session_id": "session", "timestamp_ms": timestamps,
        "state_probability": np.where(active, 0.12, 0.01),
        "onset_probability": 0.0, "offset_probability": 0.0,
        "proposal_logit": np.where(timestamps == 930_000, 8.0, -8.0),
    })
    for name in (
        "ppg_gate", "statistics_gate", "long_gate", "gyro_gate", "invariant_gate",
        "missing_fraction", "acc_valid_fraction", "gyro_valid_fraction", "ppg_valid_fraction",
        "statistics_missing_fraction", "state_probability_derivative", "stat_example",
        *(f"state_hidden_{index:03d}" for index in range(64)),
    ):
        windows[name] = 0.0
    windows["acc_valid_fraction"] = 1.0
    prior = TruncatedLogNormalDurationPrior.fit(np.array([60.0, 120.0, 300.0]))
    decoder = FixedLagSemiMarkovDecoder(prior, grid_seconds=15, fixed_lag_seconds=60)
    proposals = generate_event_candidates_v4(windows, decoder, config["decoder"], split_role="synthetic_v49")
    return config, proposals, windows


@pytest.mark.parametrize("tampering", [None, "parent_hash", "source", "boundary", "duplicate"])
def test_v49_generated_lineage_is_verified(tampering):
    _, proposals, _ = _synthetic_v49_candidates()
    assert not proposals.empty
    if tampering == "parent_hash":
        proposals.loc[0, "parent_candidate_hash"] = "changed"
    elif tampering == "source":
        proposals.loc[0, "source_mask"] = 128
    elif tampering == "boundary":
        proposals.loc[0, "coarse_start_ms"] += 3_000
    elif tampering == "duplicate":
        proposals = pd.concat([proposals, proposals.iloc[:1]], ignore_index=True)
    if tampering is None:
        validate_proposal_lineage(proposals)
        proposals["proposal_id"] = "deployment:outer0:" + proposals["proposal_id"]
        validate_proposal_lineage(proposals)
    else:
        with pytest.raises(RuntimeError):
            validate_proposal_lineage(proposals)


def test_v49_verifier_runtime_works_without_training_package_or_xgboost(tmp_path):
    config, proposals, windows = _synthetic_v49_candidates()
    features = build_proposal_features_v4(proposals, windows, ["stat_example"], config["verifier"])
    torch.manual_seed(2026)
    model = EventVerifierV4(features.sequence.shape[-1], features.scalar.shape[-1], config["verifier"]).eval()
    batch = {name: torch.from_numpy(getattr(features, name)) for name in ("sequence", "sequence_mask", "scalar")}
    with torch.no_grad():
        expected = model(batch)["event_logit"].numpy()
    package = tmp_path / "runtime" / "bme_eating"
    source = Path(__file__).parents[1] / "src" / "bme_eating"
    for relative in RUNTIME_SOURCE_FILES:
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    proposals.to_parquet(tmp_path / "proposals.parquet", index=False)
    windows.to_parquet(tmp_path / "windows.parquet", index=False)
    write_json_atomic(tmp_path / "config.json", config)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    script = """
import importlib.abc
import json
from pathlib import Path
import sys
import pandas as pd
import torch
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'runtime'))
class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = ('xgboost', 'bme_eating.integrated_v49', 'bme_eating.training', 'bme_eating.hierarchical_artifacts')
        if any(fullname == name or fullname.startswith(name + '.') for name in forbidden):
            raise ImportError('Training dependency entered the inference runtime: ' + fullname)
sys.meta_path.insert(0, BlockTrainingImports())
from bme_eating.models.event_verifier_v4 import EventVerifierV4, build_proposal_features_v4
config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
proposals = pd.read_parquet(root / 'proposals.parquet')
windows = pd.read_parquet(root / 'windows.parquet')
features = build_proposal_features_v4(proposals, windows, ['stat_example'], config['verifier'])
model = EventVerifierV4(features.sequence.shape[-1], features.scalar.shape[-1], config['verifier']).eval()
model.load_state_dict(torch.load(root / 'model.pt', weights_only=True))
batch = {name: torch.from_numpy(getattr(features, name)) for name in ('sequence', 'sequence_mask', 'scalar')}
with torch.no_grad():
    first = model(batch)['event_logit']
    second = model(batch)['event_logit']
assert torch.isfinite(first).all()
assert float((first - second).abs().max()) <= 1e-6
print(json.dumps(first.tolist()))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path)], cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", check=True, timeout=60,
    )
    np.testing.assert_allclose(json.loads(result.stdout), expected, atol=1e-6, rtol=0)


def _sensor_only_fixture(tmp_path):
    truth, ignore, proposals, windows = _event_frames()
    config = {"experiment": {"ablation_id": "S0"}, "model": {"use_statistics": False}}
    inputs = {"events": "data", "subject_folds": "split"}
    for fold in range(5):
        root = tmp_path / "experiments" / "sensor-only" / f"fold_{fold}"
        root.mkdir(parents=True)
        paths = {"resolved_config.yaml": None, "evaluation/truth_events.parquet": truth,
                 "evaluation/ignore_events.parquet": ignore, "outer/window_predictions.parquet": windows,
                 "outer/proposals.parquet": proposals}
        for relative, frame in paths.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if frame is None:
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
            else:
                frame.to_parquet(path, index=False)
        write_json_atomic(root / "run_manifest.json", {
            "outer_fold": fold, "stage": "EVALUATED", "input_hashes": inputs,
            "artifact_hashes": {name: sha256_file(root / name) for name in paths},
        })
    return inputs


@pytest.mark.parametrize("issue", [None, "data", "statistics", "missing_fold"])
def test_v49_sensor_only_reference_requires_verified_same_data(tmp_path, issue):
    inputs = _sensor_only_fixture(tmp_path)
    if issue == "data":
        inputs = {**inputs, "events": "changed"}
    elif issue == "statistics":
        path = tmp_path / "experiments/sensor-only/fold_0/resolved_config.yaml"
        path.write_text(yaml.safe_dump({"experiment": {"ablation_id": "S0"}, "model": {"use_statistics": True}}))
    elif issue == "missing_fold":
        (tmp_path / "experiments/sensor-only/fold_4/run_manifest.json").unlink()
    if issue is None:
        result = validate_sensor_only_reference(tmp_path, "sensor-only", expected_input_hashes=inputs)
        assert len(result["folds"]) == 5
    else:
        with pytest.raises(RuntimeError):
            validate_sensor_only_reference(tmp_path, "sensor-only", expected_input_hashes=inputs)


def test_v49_cohort_identity_includes_ignore_and_prediction_only_subjects(tmp_path):
    _sensor_only_fixture(tmp_path)
    root = tmp_path / "experiments/sensor-only/fold_0"
    expected = _evaluation_cohort_identity(root)
    windows = pd.read_parquet(root / "outer/window_predictions.parquet")
    windows = windows.loc[windows.subject_key != "fp-only"]
    windows.to_parquet(root / "outer/window_predictions.parquet", index=False)
    assert _evaluation_cohort_identity(root)["timeline"] != expected["timeline"]
    _, _, proposals, _ = _event_frames()
    ignored = proposals.rename(columns={"coarse_start_ms": "start_ms", "coarse_end_ms": "end_ms"})
    ignored.to_parquet(root / "evaluation/ignore_events.parquet", index=False)
    assert _evaluation_cohort_identity(root)["ignore"] != expected["ignore"]


def test_v49_pooled_heads_fail_without_fully_excluded_nested_state():
    frames = [pd.DataFrame({"subject_key": [f"subject-{fold}"]}) for fold in range(5)]
    with pytest.raises(RuntimeError, match="fully excluded nested state OOF"):
        trainer._build_isolated_pooled_fold_data(frames, frames, frames, frames, _config())


def test_v49_disabled_boundary_returns_coarse_boundaries():
    from bme_eating.hierarchical_v4_pipeline import HierarchicalEatingDetectorV4

    _, _, proposals, _ = _event_frames()
    detector = object.__new__(HierarchicalEatingDetectorV4)
    detector.boundary = None
    detector.boundary_range = None
    refined = detector._refine(proposals, pd.DataFrame())
    assert refined.refined_start_ms.tolist() == proposals.coarse_start_ms.tolist()
    assert refined.refined_end_ms.tolist() == proposals.coarse_end_ms.tolist()
    assert refined.boundary_fallback.all()


@pytest.mark.parametrize("issue", [None, "lineage", "input", "source"])
def test_v49_checkpoint_binding_matches_actual_project_identity(tmp_path, issue):
    config = _config()
    root = Path(__file__).parents[1]
    data = tmp_path / "events.json"
    write_json_atomic(data, {"event": "original"})
    config["_v49_checkpoint_identity"] = {
        "resolved_config_sha256": resume_config_hash(config),
        "input_hashes": {"events": sha256_file(data)}, "git": git_worktree_identity(root),
        "trainer_sha256": sha256_file(Path(trainer.__file__)),
        "runtime_sha256": runtime_source_identity(root / "src/bme_eating")["sha256"],
        "input_file_signatures": {str(data): [data.stat().st_size, data.stat().st_mtime_ns]},
    }
    subjects = {"training": {"train"}, "excluded": {"test"}}
    if issue == "lineage":
        subjects["training"].add("test")
    elif issue == "input":
        write_json_atomic(data, {"event": "changed identity"})
    elif issue == "source":
        config["_v49_checkpoint_identity"]["trainer_sha256"] = "wrong"
    if issue is None:
        trainer._validate_state_checkpoint_binding(config, subjects)
    else:
        with pytest.raises(RuntimeError):
            trainer._validate_state_checkpoint_binding(config, subjects)


@pytest.mark.parametrize("issue", [None, "nan", "identity", "prediction"])
def test_v49_replay_checks_intermediate_predictions_even_with_no_events(monkeypatch, issue):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    repeat_error = runpy.run_path(str(scripts / "replay_hierarchical_v4_raw_session.py"))["_repeat_frame_error"]
    first = pd.DataFrame({"proposal_id": ["candidate"], "final_score": [0.1]})
    second = first.copy()
    if issue == "nan":
        second.loc[0, "final_score"] = float("nan")
    elif issue == "identity":
        second.loc[0, "proposal_id"] = "changed"
    elif issue == "prediction":
        second.loc[0, "final_score"] += 0.01
    if issue in {"nan", "identity"}:
        with pytest.raises(RuntimeError):
            repeat_error(first, second, ["proposal_id"])
    else:
        assert repeat_error(first, second, ["proposal_id"]) == pytest.approx(0.01 if issue else 0.0)
