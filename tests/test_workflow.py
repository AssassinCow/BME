from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from helpers import candidate, truth_fixture

from bme_eating.artifacts import sha256_file, write_json_atomic
from bme_eating.bundle import StatsFusionJointFinalDataDetector, export_bundle
from bme_eating.candidates import FrozenSourceCDF
from bme_eating.config import load_config
from bme_eating.data.deep_dataset import Normalization
from bme_eating.data.joint_sessions import make_joint_dataset
from bme_eating.development_split import (
    load_split,
    validate_development_calibration,
    validate_split,
)
from bme_eating.diagnostics import locked_subject_bootstrap, stratified_diagnostics
from bme_eating.evaluation import NonnegativePlatt, commit_candidates
from bme_eating.models.statsfusion_joint_event import build_joint_model
from bme_eating.protocol import (
    capabilities,
    config_identity,
    digest,
    protocol_for,
    save_torch_atomic,
    source_identity,
    stages_for,
    validate_config,
)
from bme_eating.stats_features import STATS_FEATURE_COLUMNS, FoldRobustScaler
from bme_eating.training.statsfusion_joint_trainer import decoder_settings, load_trained
from bme_eating.validation import replay_locked_bundle
from bme_eating.workflow import (
    bind_manifest,
    checkpoint_contract,
    evaluate_development,
    evaluation_paths,
    execute_dev,
    execute_final39,
    final39_training_progress,
    stage_verified,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    torch.set_num_threads(2)
    result = load_config(ROOT / "configs/statsfusion_v414_full_data.yaml")
    result["training"]["device"] = "cpu"
    return result


def payload_for(config, split, subjects, excluded, identity, epoch):
    torch.manual_seed(2026)
    model = build_joint_model(config)
    scaler = FoldRobustScaler(STATS_FEATURE_COLUMNS, np.zeros(12), np.ones(12), tuple(sorted(subjects)))
    return {"identity": identity, "epoch": epoch, "model": model.state_dict(), "epoch_complete": True,
        "promotion_eligible": True, "clipping_fraction": 0, "training_subjects": sorted(subjects),
        "excluded_subjects": sorted(excluded), "config_sha256": config_identity(config),
        "source_sha256": digest(source_identity(ROOT)), "data_sha256": split["data_sha256"],
        "preprocessing": {"training_subjects": sorted(subjects), "scaler": scaler.to_json(),
            "normalization": Normalization(np.zeros(6), np.ones(6), 0, 1).to_json()},
        "boundary": {"qualified": True, "training_qualification_separate_from_oof_enablement": True}}


def test_registered_model_and_complete_search(config):
    validate_config(config)
    assert capabilities(config).schema == "v414"
    assert capabilities(config).feature_width == 201
    assert len(decoder_settings(config)) * 3 * 3 * 99 == 16038
    assert stages_for(config) == ("audit", "dev", "final39", "export", "replay")
    torch.manual_seed(2026)
    first = build_joint_model(config).eval()
    torch.manual_seed(2026)
    second = build_joint_model(config).eval()
    for name, tensor in first.state_dict().items():
        torch.testing.assert_close(tensor, second.state_dict()[name], atol=0, rtol=0)


def test_split_constraints_and_data_change_rejected(config):
    split = load_split(config)
    assert split["training_statistics"]["events"] == 116
    assert split["development_statistics"]["events"] == 45
    assert split["training_statistics"]["sessions"] + split["development_statistics"]["sessions"] == 1153
    changed = copy.deepcopy(split)
    changed["development_subjects"][0] = changed["training_subjects"][0]
    with pytest.raises(RuntimeError, match="partition"):
        validate_split(changed)
    changed = copy.deepcopy(split)
    changed["development_statistics"]["events"] += 1
    with pytest.raises(RuntimeError, match="statistics"):
        validate_split(changed)
    altered = copy.deepcopy(config)
    altered["development"]["split_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="split hash"):
        load_split(altered)
    assert config_identity(altered) != config_identity(config)
    for mode in ("final39", "fold0", "fivefold"):
        altered = copy.deepcopy(config)
        altered["execution"]["mode"] = mode
        with pytest.raises(ValueError):
            validate_config(altered)
    altered = copy.deepcopy(config)
    altered["search"]["budgets"] = [20]
    with pytest.raises(ValueError, match="16038"):
        validate_config(altered)


def test_seen_development_requires_exact_explicit_scope_and_reads_no_labels(config):
    split = load_split(config)
    payload = payload_for(config, split, split["all_subjects"], [], "final", 8)
    evidence = {"kind": "development_on_final_model", "split_sha256": config["development"]["split_sha256"],
                "data_sha256": split["data_sha256"]}
    requests = []
    def frames(subjects, *, labels):
        requests.append((subjects, labels))
        raise LookupError("unlabeled data boundary reached")
    inputs = SimpleNamespace(identity=split["data_identity"], frames=frames)
    subjects = set(split["development_subjects"])
    validate_development_calibration(inputs, subjects, config, payload["preprocessing"], evidence, training=False)
    with pytest.raises(LookupError, match="unlabeled"):
        make_joint_dataset(inputs, subjects, config, payload["preprocessing"], training=False,
                           development_calibration=evidence)
    assert requests == [(subjects, False)]
    for changed in ({**evidence, "kind": "independent"}, {**evidence, "data_sha256": "invalid"}):
        with pytest.raises(RuntimeError):
            validate_development_calibration(inputs, subjects, config, payload["preprocessing"], changed, training=False)
    with pytest.raises(RuntimeError):
        validate_development_calibration(inputs, subjects, config, payload["preprocessing"], evidence, training=True)
    with pytest.raises(RuntimeError):
        validate_development_calibration(inputs, set(split["training_subjects"]), config, payload["preprocessing"], evidence, training=False)


def test_selected_checkpoint_instead_of_latest_and_old_protocol_rejected(config, tmp_path):
    split = load_split(config)
    subjects, excluded = set(split["training_subjects"]), set(split["development_subjects"])
    selected = payload_for(config, split, subjects, excluded, "dev", 8)
    save_torch_atomic(tmp_path / "selected.pt", selected)
    save_torch_atomic(tmp_path / "latest.pt", {**selected, "epoch": 18})
    _, payload = load_trained(tmp_path, config, "dev", subjects, split["data_identity"], checkpoint_name="selected.pt")
    assert payload["epoch"] == 8
    checkpoint_contract(payload, config, SimpleNamespace(identity=split["data_identity"]), "dev", subjects, excluded, 8)
    older = copy.deepcopy(config)
    older["experiment"]["protocol_version"] = "statsfusion-joint-event-v4"
    selected["config_sha256"] = config_identity(older)
    save_torch_atomic(tmp_path / "selected.pt", selected)
    with pytest.raises(RuntimeError, match="identity"):
        load_trained(tmp_path, config, "dev", subjects, split["data_identity"], checkpoint_name="selected.pt")


def test_new_score_identity_does_not_use_quality_product(config):
    item = candidate()
    item.score_formula = capabilities(config).score_formula
    item.event_logit, item.predicted_quality = 2, .001
    predictions = commit_candidates([item], NonnegativePlatt(1, 0, True), .5, .3)
    assert len(predictions) == 1
    assert predictions[0].acceptance_score > .8


@pytest.fixture
def completed_run(config, tmp_path, monkeypatch):
    from bme_eating import workflow
    split = load_split(config)
    run = tmp_path / "run"
    run.mkdir()
    identity = {"protocol": protocol_for(config), "config_sha256": config_identity(config),
                "source": source_identity(ROOT), "data": split["data_identity"]}
    write_json_atomic(run / "identity.json", identity)
    write_json_atomic(run / "resolved_config.json", config)
    write_json_atomic(run / "split_manifest.json", split)
    for relative in identity["source"]:
        target = run / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    calls = []
    def train(inputs, subjects, configuration, directory, model_identity, *, epochs, resume, selector=None, deployment_epoch=None):
        calls.append({"subjects": subjects, "epochs": epochs, "selector": selector, "resume": resume,
                      "deployment_epoch": deployment_epoch})
        directory.mkdir(parents=True, exist_ok=True)
        epoch = 8 if selector else deployment_epoch
        payload = payload_for(config, split, subjects, selector or set(), model_identity, epoch)
        save_torch_atomic(directory / ("selected.pt" if selector else "latest.pt"), payload)
        if selector:
            save_torch_atomic(directory / "latest.pt", {**payload, "epoch": 12})
        else:
            for retained in (epoch - 4, epoch - 2, epoch, epoch + 2, epoch + 4):
                save_torch_atomic(directory / "nearby" / f"epoch_{retained:03d}.pt", {**payload, "epoch": retained})
        write_json_atomic(directory / "preprocessing.json", payload["preprocessing"])
        write_json_atomic(directory / "training_history.json", [{"epoch": index, "clipping_fraction": 0} for index in range(1, (epoch if selector else epochs) + 1)])
        return epoch if selector else epochs
    def evaluate(model, payload, inputs, configuration, directory, partition, phase, model_identity):
        scope = "subject_excluded_development" if phase == "dev" else "development_on_final_model"
        checkpoint = run / phase / "model" / ("selected.pt" if phase == "dev" else "latest.pt")
        point = {"setting": "high_0.1_low_0.9_gap_15", "budget": 20, "acceptance": .5, "nms": .3,
            "promotion_eligible": True, "promotable": False, "boundary_enabled": False,
            "boundary_mode": "off", "boundary_entropy": None, "calibration_kind": scope,
            "independent_calibration": False, "score_formula": capabilities(config).score_formula,
            "model_identity": model_identity, "checkpoint_sha256": sha256_file(checkpoint),
            "calibration": {"slope": 1, "intercept": 0, "qualified": True,
                            "discriminative": True, "probability_range": .5}}
        report = {"metrics": {"f1": .5}, "coverage": {"recall": .8}, "strata": {},
                  "bootstrap": {}, "operating_point_stability": {}}
        for relative in evaluation_paths(phase):
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(path, {})
        write_json_atomic(run / phase / "evaluation/operating_point.json", point)
        write_json_atomic(run / phase / "evaluation/report.json", report)
        write_json_atomic(run / phase / "evaluation/source_fit/source_mapping.json", {
            "distributions": {"proposal": [0, 1]}, "training_subjects": payload["training_subjects"], "fallback": True})
        return point, report
    monkeypatch.setattr(workflow, "audit", lambda *args: split)
    monkeypatch.setattr(workflow, "train_joint_model", train)
    monkeypatch.setattr(workflow, "evaluate_development", evaluate)
    inputs = SimpleNamespace(identity=split["data_identity"])
    selection = execute_dev(inputs, config, run)
    final = execute_final39(inputs, config, run)
    return run, config, inputs, split, calls, selection, final


def test_two_fresh_trainings_fixed_epoch_and_resume_does_not_retrain(completed_run):
    run, config, inputs, split, calls, selection, final = completed_run
    assert [len(call["subjects"]) for call in calls] == [28, 39]
    assert [call["epochs"] for call in calls] == [64, 12]
    assert [call["deployment_epoch"] for call in calls] == [None, 8]
    assert calls[0]["selector"] == set(split["development_subjects"])
    assert calls[1]["selector"] is None
    assert not any(call["resume"] for call in calls)
    assert selection["selected_epoch"] == final["fixed_epoch"] == 8
    training_manifest = json.loads((run / "final39_training_manifest.json").read_text())
    assert training_manifest["executed_epochs"] == 12
    assert training_manifest["retained_checkpoint_epochs"] == [4, 6, 8, 10, 12]
    assert [record["epoch"] for record in training_manifest["neighbor_checkpoints"]] == [4, 6, 8, 10, 12]
    assert all(training_manifest["evidence_sha256"][record["path"]] == record["sha256"]
               for record in training_manifest["neighbor_checkpoints"])
    assert torch.load(run / "final39/model/latest.pt", weights_only=True)["epoch"] == 8
    assert torch.load(run / "final39/model/nearby/epoch_012.pt", weights_only=True)["epoch"] == 12
    assert final["official_test_candidate"] and not final["promotable"]
    assert final["labels_participated_in_training"] and not final["independent_calibration"]
    execute_final39(inputs, config, run, resume=True)
    assert len(calls) == 2
    with pytest.raises(FileExistsError):
        execute_final39(inputs, config, run, resume=False)
    assert stage_verified(run, config, "dev") and stage_verified(run, config, "final39")
    assert final39_training_progress(run, config)["extension_training_verified"]
    checkpoint = run / "final39/model/latest.pt"
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checksum"):
        execute_final39(inputs, config, run, resume=True)


def test_final_bundle_raw_replay_has_no_label_fit_or_search(completed_run, monkeypatch):
    run, config, *_ = completed_run
    export_bundle(config, run, ROOT)
    detector = StatsFusionJointFinalDataDetector(run / "bundle")
    assert torch.load(run / "bundle/checkpoint.pt", weights_only=True)["epoch"] == 8
    assert len(list((run / "bundle/final39/model/nearby").glob("*.pt"))) == 5
    def forbidden(*args, **kwargs):
        raise AssertionError("Official inference must not read labels or fit/search")
    from bme_eating.data.joint_sessions import JointInputs
    monkeypatch.setattr(JointInputs, "frames", forbidden)
    monkeypatch.setattr(NonnegativePlatt, "fit", forbidden)
    import bme_eating.operating_points as search
    monkeypatch.setattr(search, "choose_point", forbidden)
    report = replay_locked_bundle(detector)
    assert report["repeat_max_error"] <= 1e-6
    assert report["incremental_identity_alignment"]
    with pytest.raises(FileExistsError):
        export_bundle(config, run, ROOT)
    assert export_bundle(config, run, ROOT, resume=True)["resumed"]


def test_bundle_rejects_independent_claim_or_boundary_tamper(completed_run):
    run, config, *_ = completed_run
    export_bundle(config, run, ROOT)
    bundle = run / "bundle"
    manifest = json.loads((bundle / "final_manifest.json").read_text())
    manifest["independent_calibration"] = True
    write_json_atomic(bundle / "final_manifest.json", manifest)
    hashes = json.loads((bundle / "SHA256SUMS.json").read_text())
    hashes["final_manifest.json"] = sha256_file(bundle / "final_manifest.json")
    write_json_atomic(bundle / "SHA256SUMS.json", hashes)
    with pytest.raises(RuntimeError, match="scope"):
        StatsFusionJointFinalDataDetector(bundle)


def test_stratified_diagnostics_empty_and_locked_subject_bootstrap(config):
    split = load_split(config)
    subject = split["development_subjects"][0]
    truth = truth_fixture().assign(subject_key=subject, event_id="event", hand_relation="same")
    item = candidate()
    item.subject_key = subject
    windows = pd.DataFrame({"subject_key": [subject] * 20, "session_id": ["session"] * 20,
                            "timestamp_ms": np.arange(20) * 3000, "valid": 1., "gyro_valid": 1.})
    report = stratified_diagnostics(truth, [item], [item], truth.iloc[:0], windows, split)
    assert report["different_hand"]["support_events"] == 0
    bootstrap = locked_subject_bootstrap(truth, [item], truth.iloc[:0], windows, [subject], replicates=10)
    assert not bootstrap["selection_optimism_corrected"]
    assert bootstrap["intervals_95"]["f1"] == [1., 1.]


def test_failed_calibration_cannot_export(config, tmp_path):
    write_json_atomic(tmp_path / "identity.json", {})
    manifest = bind_manifest(tmp_path, config, "final39", {"export_eligible": False}, ["identity.json"])
    write_json_atomic(tmp_path / "final_manifest.json", manifest)
    with pytest.raises(RuntimeError, match="no deployment"):
        export_bundle(config, tmp_path, ROOT)
    assert not (tmp_path / "bundle").exists()


def test_development_evaluation_full_search_streaming_and_artifact_paths(config, tmp_path, monkeypatch):
    from bme_eating import workflow
    split = load_split(config)
    subject = split["development_subjects"][0]
    truth = truth_fixture().assign(subject_key=subject, start_ms=2_070_000, end_ms=2_130_000,
                                   evaluable=True, valid_duration=True, hand_relation="same")
    positive = candidate(start=2_070_000, end=2_130_000, identity="positive", parent="positive")
    negative = candidate(start=2_100_000, end=2_115_000, identity="negative", parent="negative")
    for item, logit in ((positive, 5), (negative, -5)):
        item.subject_key, item.model_identity = subject, "model"
        item.decision_time_ms = item.max_input_time_ms = 2_160_000
        item.score_formula, item.event_logit = capabilities(config).score_formula, logit
    windows = pd.DataFrame({"subject_key": subject, "session_id": "session",
        "timestamp_ms": np.arange(721) * 3000, "valid": 1., "gyro_valid": 1.,
        "onset_probability": .01, "offset_probability": .01})
    source_candidates = copy.deepcopy([positive, negative])
    for item in source_candidates:
        item.subject_key = split["training_subjects"][0]
    mapping = FrozenSourceCDF.fit(source_candidates, set(split["training_subjects"]))
    def fit(*args, **kwargs):
        directory = args[-2]
        write_json_atomic(directory / "source_mapping.json", mapping.to_json())
        write_json_atomic(directory / "performance.json", {})
        return mapping
    def predict(*args, **kwargs):
        write_json_atomic(args[5] / "performance.json", {})
        assert kwargs["development_calibration"] is None
        assert args[7]["search_eligible"] is False
        return {"coarse": {name: copy.deepcopy([positive, negative]) for name in decoder_settings(config)},
                "windows": windows, "refined": {}}
    def frames(subjects, *, labels):
        assert subjects == set(split["development_subjects"]) and labels
        return None, None, truth, None
    monkeypatch.setattr(workflow, "fit_source_mapping", fit)
    monkeypatch.setattr(workflow, "predict_joint_model", predict)
    config["resources"]["search_processes"] = 1
    directory = tmp_path / "dev/model"
    directory.mkdir(parents=True)
    save_torch_atomic(directory / "selected.pt", {"fixture": True})
    point, report = evaluate_development(None, {"preprocessing": {}, "boundary": {}},
        SimpleNamespace(frames=frames), config, tmp_path, split, "dev", "model")
    assert point["promotion_eligible"] and point["f1"] == 1
    assert point["boundary_mode"] == "off" and not point["independent_calibration"]
    assert report["registered_operating_points"] == 16038
    assert report["strata"]["same_hand"]["recall"] == 1
    assert all((tmp_path / name).exists() for name in evaluation_paths("dev"))


@pytest.mark.parametrize("failure", [KeyboardInterrupt, torch.OutOfMemoryError])
def test_v414_training_contribution_and_natural_replay_recovery(config, tmp_path, monkeypatch, failure):
    from test_training import test_replay_and_contribution_chunk_resume
    test_replay_and_contribution_chunk_resume(config, tmp_path, monkeypatch, failure)


@pytest.mark.parametrize("failure,interruption_epoch", [(KeyboardInterrupt, 9), (torch.OutOfMemoryError, 11)])
def test_neighbor_extension_chunk_resume_matches_continuous(config, tmp_path, monkeypatch, failure, interruption_epoch):
    from test_training import test_replay_and_contribution_chunk_resume
    test_replay_and_contribution_chunk_resume(config, tmp_path, monkeypatch, failure,
        epochs=12, deployment_epoch=8, interruption_epoch=interruption_epoch)


@pytest.mark.parametrize("change", ["missing", "corrupted", "epoch", "identity", "subject", "finite", "incomplete", "metadata"])
def test_neighbor_checkpoint_damage_rejected_before_resume_or_export(completed_run, change):
    run, config, *_ = completed_run
    manifest_path = run / "final39_training_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["neighbor_checkpoints"][-1]
    checkpoint = run / record["path"]
    if change == "missing":
        checkpoint.unlink()
    elif change == "corrupted":
        checkpoint.write_bytes(b"corrupted")
    elif change == "metadata":
        manifest["executed_epochs"] = 8
        write_json_atomic(manifest_path, manifest)
    else:
        payload = torch.load(checkpoint, weights_only=True)
        if change == "epoch":
            payload["epoch"] = 10
        elif change == "identity":
            payload["identity"] = "another_model"
        elif change == "subject":
            payload["training_subjects"] = payload["training_subjects"][:-1]
        elif change == "finite":
            next(iter(payload["model"].values())).reshape(-1)[0] = float("nan")
        else:
            payload["epoch_complete"] = False
        save_torch_atomic(checkpoint, payload)
        record["sha256"] = manifest["evidence_sha256"][record["path"]] = sha256_file(checkpoint)
        write_json_atomic(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="checksum|neighbor"):
        stage_verified(run, config, "final39")
    with pytest.raises(RuntimeError):
        export_bundle(config, run, ROOT)
    assert not (run / "bundle").exists()


def test_neighbor_checkpoints_can_remain_diagnostic(completed_run):
    run, config, *_ = completed_run
    manifest_path = run / "final39_training_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["neighbor_checkpoints"][0]
    checkpoint = run / record["path"]
    payload = torch.load(checkpoint, weights_only=True)
    payload.update({"promotion_eligible": False, "clipping_fraction": .3})
    save_torch_atomic(checkpoint, payload)
    record.update({"sha256": sha256_file(checkpoint), "promotion_eligible": False, "clipping_fraction": .3})
    manifest["evidence_sha256"][record["path"]] = record["sha256"]
    write_json_atomic(manifest_path, manifest)
    assert verify_manifest(run, config, manifest_path.name, "final39")["training_qualified"]


def test_bundle_rechecks_neighbor_contract(completed_run):
    run, config, *_ = completed_run
    export_bundle(config, run, ROOT)
    bundle = run / "bundle"
    relative = "final39_training_manifest.json"
    manifest = json.loads((bundle / relative).read_text())
    manifest["retained_checkpoint_epochs"] = [8]
    write_json_atomic(bundle / relative, manifest)
    hashes = json.loads((bundle / "SHA256SUMS.json").read_text())
    hashes[relative] = sha256_file(bundle / relative)
    write_json_atomic(bundle / "SHA256SUMS.json", hashes)
    with pytest.raises(RuntimeError, match="neighbor"):
        StatsFusionJointFinalDataDetector(bundle)


@pytest.fixture
def runner_context(completed_run, tmp_path, monkeypatch):
    import bme_eating.development_split as split_module

    run, config, inputs, split, *_ = completed_run
    output = tmp_path / "outputs"
    target = output / "v414/runs/run"
    shutil.copytree(run, target)
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("v414_runner_failure_test", ROOT / "scripts/run_statsfusion_joint.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    write_json_atomic(target / "execution_status.json", {"status": "COMPLETE", "current_stage": "final39",
        "stages": {"audit": "COMPLETE", "dev": "COMPLETE", "final39": "COMPLETE"}})
    monkeypatch.setattr(runner, "_load_project_dotenv", lambda *args: None)
    monkeypatch.setattr(runner, "require_environment_path", lambda *args: output)
    monkeypatch.setattr(runner.JointInputs, "audit", lambda *args: inputs)
    monkeypatch.setattr(runner, "initialize_run", lambda *args, **kwargs: target)
    monkeypatch.setattr(split_module, "load_split", lambda *args: split)
    arguments = SimpleNamespace(evaluation_mode="dev28_11", internal_mode=None, fold=None,
        migrate_clipping_gate=False, stage="final39", max_epochs=None, run_name="run",
        fresh=False, resume=True, reference_run=None, historical_reference=None)
    return runner, target, config, arguments


@pytest.mark.parametrize("phase", ["input_audit", "run_audit", "completed_stage"])
def test_preflight_failure_updates_status_and_preserves_checkpoints(runner_context, monkeypatch, phase):
    from bme_eating import workflow

    runner, run, config, arguments = runner_context
    checkpoints = {path: sha256_file(path) for path in (run / "final39/model").rglob("*.pt")}
    error = RuntimeError("injected audit failure")
    def fail(*args, **kwargs):
        raise error
    if phase == "input_audit":
        monkeypatch.setattr(runner.JointInputs, "audit", fail)
    elif phase == "run_audit":
        monkeypatch.setattr(workflow, "audit", fail)
    else:
        monkeypatch.setattr(workflow, "stage_verified", fail)
    with pytest.raises(RuntimeError) as raised:
        runner.run_v414(arguments, config, SimpleNamespace(error=pytest.fail))
    assert raised.value is error
    stage = "final39" if phase == "completed_stage" else "audit"
    status = json.loads((run / "execution_status.json").read_text())
    report = json.loads((run / "failure_report.json").read_text())
    assert status["status"] == status["stages"][stage] == "FAILED"
    assert report["stage"] == stage and report["message"] == str(error)
    assert not report["final_model_complete"]
    assert all(sha256_file(path) == checksum for path, checksum in checkpoints.items())


def test_corrupted_neighbor_resume_reports_failure_and_saved_deployment(runner_context):
    runner, run, config, arguments = runner_context
    neighbor = run / "final39/model/nearby/epoch_012.pt"
    neighbor.write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="checksum"):
        runner.run_v414(arguments, config, SimpleNamespace(error=pytest.fail))
    status = json.loads((run / "execution_status.json").read_text())
    report = json.loads((run / "failure_report.json").read_text())
    assert status["status"] == status["stages"]["final39"] == "FAILED"
    progress = report["final39_training_progress"]
    assert progress["deployment_checkpoint_saved"] and progress["deployment_checkpoint_qualified"]
    assert not progress["extension_training_verified"] and "epoch_012.pt" in progress["extension_verification_error"]["message"]


@pytest.mark.parametrize("failure,expected_status", [(KeyboardInterrupt, "PAUSED"), (torch.OutOfMemoryError, "FAILED")])
def test_extension_failure_records_saved_epoch_without_completing_stage(runner_context, monkeypatch, failure, expected_status):
    from bme_eating import workflow

    runner, run, config, arguments = runner_context
    for name in ("final39_training_manifest.json", "calibration_manifest.json", "final_manifest.json"):
        (run / name).unlink()
    history = run / "final39/model/training_history.json"
    write_json_atomic(history, json.loads(history.read_text())[:9])
    status = json.loads((run / "execution_status.json").read_text())
    status["status"] = status["stages"]["final39"] = "PAUSED"
    write_json_atomic(run / "execution_status.json", status)
    error = failure("injected extension failure")
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(workflow, "execute_final39", fail)
    checkpoint = run / "final39/model/latest.pt"
    checksum = sha256_file(checkpoint)
    with pytest.raises(failure) as raised:
        runner.run_v414(arguments, config, SimpleNamespace(error=pytest.fail))
    assert raised.value is error
    status = json.loads((run / "execution_status.json").read_text())
    report = json.loads((run / "failure_report.json").read_text())
    assert status["status"] == status["stages"]["final39"] == expected_status
    assert report["final39_training_progress"] == {"fixed_epoch": 8, "target_epochs": 12,
        "deployment_checkpoint_saved": True, "deployment_checkpoint_qualified": True,
        "last_recorded_epoch": 9, "extension_training_verified": False}
    assert not report["final_model_complete"] and sha256_file(checkpoint) == checksum


def test_failure_report_write_error_does_not_replace_original_error(runner_context, monkeypatch):
    from bme_eating import workflow

    runner, run, config, arguments = runner_context
    error = RuntimeError("original validation error")
    def fail(*args, **kwargs):
        raise error
    def fail_report(*args, **kwargs):
        raise OSError("report write error")
    monkeypatch.setattr(workflow, "stage_verified", fail)
    monkeypatch.setattr(runner, "write_failure", fail_report)
    with pytest.raises(RuntimeError) as raised:
        runner.run_v414(arguments, config, SimpleNamespace(error=pytest.fail))
    assert raised.value is error
    report = json.loads((run / "failure_report.json").read_text())
    assert report["message"] == str(error) and report["status"] == "FAILED"
    assert report["reporting_errors"][0]["operation"] == "failure_details"


def test_final39_cli_alias_preserves_frozen_identity(config, tmp_path, monkeypatch):
    import bme_eating.bundle as bundle_module
    import bme_eating.development_split as split_module
    from bme_eating import validation, workflow

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("v414_runner_alias_test", ROOT / "scripts/run_statsfusion_joint.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    split = load_split(config)
    run = tmp_path / "run"
    run.mkdir()
    write_json_atomic(run / "execution_status.json", {
        "status": "PAUSED", "current_stage": "dev", "stages": {"audit": "COMPLETE", "dev": "COMPLETE"}})
    original_identity = config_identity(config)
    inputs = SimpleNamespace(identity=split["data_identity"])
    calls = []
    monkeypatch.setattr(runner, "_load_project_dotenv", lambda *args: None)
    monkeypatch.setattr(runner, "require_environment_path", lambda *args: tmp_path)
    monkeypatch.setattr(runner.JointInputs, "audit", lambda *args: inputs)
    monkeypatch.setattr(runner, "initialize_run", lambda *args, **kwargs: run)
    monkeypatch.setattr(split_module, "load_split", lambda *args: split)
    monkeypatch.setattr(workflow, "audit", lambda *args: split)
    monkeypatch.setattr(workflow, "execute_final39", lambda *args, **kwargs: calls.append("final39"))
    monkeypatch.setattr(bundle_module, "export_bundle", lambda *args, **kwargs: calls.append("export"))
    def replay(*args, **kwargs):
        calls.append("replay")
        return {"no_xgboost_replay": True}
    monkeypatch.setattr(validation, "replay_joint_bundle", replay)
    arguments = SimpleNamespace(evaluation_mode="final39", internal_mode=None, fold=None,
        migrate_clipping_gate=False, stage="all", max_epochs=None, run_name="run", fresh=False,
        resume=True, reference_run=None, historical_reference=None)
    monkeypatch.setattr(sys, "argv", ["runner", "--evaluation-mode", "final39", "--resume"])
    runner.run_v414(arguments, config, SimpleNamespace(error=lambda message: pytest.fail(message)))
    assert calls == ["final39", "export", "replay"]
    assert config["execution"]["mode"] == "dev28_11"
    assert config_identity(config) == original_identity
    assert json.loads((run / "execution_status.json").read_text())["status"] == "COMPLETE"
