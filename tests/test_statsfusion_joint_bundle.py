from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bme_eating.config import load_config
from bme_eating.data.deep_dataset import Normalization
from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_pipeline import StatsFusionJointDetector
from bme_eating.joint_protocol import (
    config_identity,
    digest,
    protocol_for,
    save_torch_atomic,
    source_identity,
)
from bme_eating.joint_validation import replay_joint_bundle
from bme_eating.joint_workflow import export_joint_bundle
from bme_eating.models.statsfusion_joint_event_v411 import build_joint_model
from bme_eating.stats_features import STATS_FEATURE_COLUMNS, FoldRobustScaler


def convert_to_single_fold_bundle(bundle, config):
    from bme_eating.joint_protocol import calibration_scope, config_identity

    config["execution"] = {"mode": "fold0", **({"internal_mode": "single"}
        if config["experiment"]["code_version"] in {"v4.12", "v4.13"} else {})}
    identity = json.loads((bundle / "identity.json").read_text(encoding="utf-8"))
    identity["config_sha256"] = config_identity(config)
    write_json_atomic(bundle / "identity.json", identity)
    write_json_atomic(bundle / "resolved_config.json", config)
    manifest = json.loads((bundle / "final_manifest.json").read_text(encoding="utf-8"))
    test_subjects = ["outer_test"]
    scope = {}
    if config["execution"].get("internal_mode") == "single":
        from bme_eating.training.hierarchical_v4_trainer import stacking_partitions
        subjects = set(manifest["training_subjects"])
        assignment = stacking_partitions(subjects, 3, 2026)
        prediction = {subject for subject, partition in assignment.items() if partition == 0}
        train = subjects - prediction
        selector = {max(train)}
        manifest["lineage"] = [{**manifest["lineage"][0], "training_subjects": sorted(train),
            "prediction_subjects": sorted(prediction), "source_training_subjects": sorted(train),
            "normalization_subjects": sorted(train), "selector_subjects": sorted(selector),
            "selector_fit_subjects": sorted(train - selector)}]
        scope = calibration_scope(config)
    for lineage in manifest["lineage"]:
        lineage["outer_excluded_subjects"] = test_subjects
        lineage["config_sha256"] = config_identity(config)
    point = json.loads((bundle / "operating_point.json").read_text(encoding="utf-8"))
    point["lineage"] = manifest["lineage"]
    point.update(scope)
    write_json_atomic(bundle / "operating_point.json", point)
    payload = torch.load(bundle / "checkpoint.pt", weights_only=True)
    payload["excluded_subjects"] = test_subjects
    payload["config_sha256"] = config_identity(config)
    save_torch_atomic(bundle / "checkpoint.pt", payload)
    gate = {"protocol": protocol_for(config), "promotable": False, "five_folds_complete": False,
        "artifact_kind": "single_fold_diagnostic", "evaluation_mode": "fold0",
        "diagnostic_complete": True,
        "export_eligible": True, "checks": {name: True for name in
            ("training_qualified", "operating_point_qualified", "boundary")}, "fold": 0,
        "identity_sha256": sha256_file(bundle / "identity.json"),
        "training_subjects": manifest["training_subjects"], "test_subjects": test_subjects}
    gate.update(scope)
    write_json_atomic(bundle / "fold0_gate_manifest.json", gate)
    manifest.update({"promotable": False, "export_eligible": True, "fold": 0,
        "artifact_kind": "single_fold_diagnostic", "evaluation_mode": "fold0", "test_subjects": test_subjects,
        "uses_test_labels_for_selection": False,
        "identity_sha256": gate["identity_sha256"],
        "gate_sha256": sha256_file(bundle / "fold0_gate_manifest.json"),
        "evidence_sha256": {name: sha256_file(bundle / name) for name in
                           ("checkpoint.pt", "operating_point.json", "source_mapping.json")}, **scope})
    write_json_atomic(bundle / "final_manifest.json", manifest)
    update_hashes(bundle)
    hashes = json.loads((bundle / "SHA256SUMS.json").read_text(encoding="utf-8"))
    hashes["fold0_gate_manifest.json"] = sha256_file(bundle / "fold0_gate_manifest.json")
    write_json_atomic(bundle / "SHA256SUMS.json", hashes)
    return config


@pytest.fixture(params=["v410", "v411", "v412", "v413"])
def synthetic_bundle(tmp_path, request):
    source = Path(__file__).resolve().parents[1]
    filenames = {"v410": "statsfusion_v410_joint_event.yaml", "v411": "statsfusion_v411_learned_boundary.yaml",
                 "v412": "statsfusion_v412_integrated_repair.yaml",
                 "v413": "statsfusion_v413_integrated_repair.yaml"}
    config = load_config(source / "configs" / filenames[request.param])
    if request.param in {"v412", "v413"}:
        config["execution"] = {"mode": "fivefold", "internal_mode": "oof"}
    bundle = tmp_path / "synthetic_bundle"
    bundle.mkdir()
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    model = build_joint_model(config)
    subjects = [f"fit_{index}" for index in range(6)]
    identity = {"synthetic_test_fixture": True, "source": source_identity(source), "data": {"fixture": "synthetic"},
                "protocol": protocol_for(config)}
    scaler = FoldRobustScaler(STATS_FEATURE_COLUMNS, np.zeros(12), np.ones(12), tuple(subjects))
    payload = {"model": model.state_dict(), "identity": "synthetic_fixture", "epoch": 8,
               "epoch_complete": True, "promotion_eligible": True, "clipping_fraction": 0,
               "training_subjects": subjects, "config_sha256": config_identity(config), "excluded_subjects": [],
               "source_sha256": digest(identity["source"]), "data_sha256": digest(identity["data"]),
               "preprocessing": {"scaler": scaler.to_json(), "training_subjects": subjects,
                   "normalization": Normalization(np.zeros(6), np.ones(6), 0, 1).to_json()},
               "boundary": {"qualified": False}}
    save_torch_atomic(bundle / "checkpoint.pt", payload)
    write_json_atomic(bundle / "identity.json", identity)
    identity_hash = sha256_file(bundle / "identity.json")
    gate = {"protocol": protocol_for(config), "promotable": True, "identity_sha256": identity_hash,
            "synthetic_test_fixture": True,
            "folds": [{"fold": fold, "prediction_subjects": subjects[fold::5]} for fold in range(5)],
            "reference": {"qualified": True, "comparison_performed": True},
            "checks": {name: True for name in ("five_folds", "training_qualified", "f1", "fp_per_hour", "coverage",
                "truth_cohort", "fold_coverage", "operating_point_qualified", "stress", "boundary", "reference",
                "bootstrap", "matching_ranking", "same_recall", "different_recall")}}
    write_json_atomic(bundle / "joint_gate_manifest.json", gate)
    write_json_atomic(bundle / "resolved_config.json", config)
    write_json_atomic(bundle / "operating_point.json", {"setting": ("high_0.1_low_0.9_gap_15"
        if request.param == "v413" else "high_0.1_gap_15"), "budget": 30,
        "acceptance": .1, "nms": .5, "promotion_eligible": True, "boundary_enabled": False,
        "boundary_mode": "off", "score_formula": (f"calibrated_event_probability_{request.param}"
            if request.param != "v410" else "event_probability_times_quality_v410"),
        "calibration": {"slope": 1, "intercept": 0, "qualified": True,
            **({"optimization_succeeded": True, "discriminative": True, "probability_range": .5}
               if request.param == "v413" else {})}})
    write_json_atomic(bundle / "source_mapping.json", {"distributions": {"proposal": [0, 1]},
                                                       "training_subjects": subjects})
    lineages = []
    for partition in range(3):
        prediction = subjects[2 * partition:2 * partition + 2]
        fit = sorted(set(subjects) - set(prediction))
        lineages.append({"partition": partition, "model_identity": f"synthetic_partition_{partition}",
            "config_sha256": payload["config_sha256"], "source_sha256": payload["source_sha256"],
            "data_sha256": payload["data_sha256"], "training_subjects": fit, "prediction_subjects": prediction,
            "source_training_subjects": fit, "normalization_subjects": fit, "outer_excluded_subjects": [],
            "selector_fit_subjects": fit[:-1], "selector_subjects": fit[-1:], "checkpoint_qualified": True})
    write_json_atomic(bundle / "final_manifest.json", {"protocol": protocol_for(config), "promotable": True,
        "model_identity": "synthetic_fixture", "identity_sha256": identity_hash,
        "training_subjects": subjects,
        "gate_sha256": sha256_file(bundle / "joint_gate_manifest.json"), "boundary_enabled": False,
        "boundary_mode": "off", "score_formula": (f"calibrated_event_probability_{request.param}"
            if request.param != "v410" else "event_probability_times_quality_v410"),
        "internal_oof_sha256": "synthetic_fixture", "lineage": lineages,
        "checkpoint_sha256": sha256_file(bundle / "checkpoint.pt"),
        "operating_point_sha256": sha256_file(bundle / "operating_point.json"),
        "source_mapping_sha256": sha256_file(bundle / "source_mapping.json")})
    shutil.copytree(source / "src" / "bme_eating", bundle / "src" / "bme_eating",
                    ignore=shutil.ignore_patterns("__pycache__"))
    hashes = {str(path.relative_to(bundle)).replace("\\", "/"): sha256_file(path)
              for path in bundle.rglob("*") if path.is_file()}
    write_json_atomic(bundle / "SHA256SUMS.json", hashes)
    return bundle, config


def update_hashes(bundle):
    path = bundle / "SHA256SUMS.json"
    hashes = json.loads(path.read_text(encoding="utf-8"))
    for relative in hashes:
        hashes[relative] = sha256_file(bundle / relative)
    write_json_atomic(path, hashes)


def test_bundle_replay_without_xgboost(synthetic_bundle):
    bundle, config = synthetic_bundle
    report = replay_joint_bundle(bundle, config)
    assert report["no_xgboost_load"]
    assert report["event_count"] > 0


def test_single_fold_bundle_replay_is_separate_from_formal_promotion(synthetic_bundle):
    from bme_eating.joint_single_fold import StatsFusionJointFoldDetector

    bundle, config = synthetic_bundle
    if config["experiment"]["code_version"] != "v4.11":
        pytest.skip("Single-fold continuation targets v411")
    config = convert_to_single_fold_bundle(bundle, config)
    detector = StatsFusionJointFoldDetector(bundle)
    assert detector.identity == "synthetic_fixture"
    with pytest.raises(RuntimeError, match="qualified"):
        StatsFusionJointDetector(bundle)
    report = replay_joint_bundle(bundle, config)
    assert report["no_xgboost_replay"]


def test_single_partition_bundle_keeps_holdout_lineage_and_cannot_claim_full_oof(synthetic_bundle):
    from bme_eating.joint_protocol import calibration_scope
    from bme_eating.joint_single_fold import StatsFusionJointFoldDetector
    from bme_eating.training.hierarchical_v4_trainer import stacking_partitions

    bundle, config = synthetic_bundle
    if config["experiment"]["code_version"] not in {"v4.11", "v4.12", "v4.13"}:
        pytest.skip("Single partition requires grouped candidates")
    config = convert_to_single_fold_bundle(bundle, config)
    config["execution"]["internal_mode"] = "single"
    manifest = json.loads((bundle / "final_manifest.json").read_text(encoding="utf-8"))
    subjects = set(manifest["training_subjects"])
    assignment = stacking_partitions(subjects, 3, 2026)
    prediction = {subject for subject, partition in assignment.items() if partition == 0}
    train = subjects - prediction
    selector = {max(train)}
    lineage = {**manifest["lineage"][0], "training_subjects": sorted(train),
        "prediction_subjects": sorted(prediction), "source_training_subjects": sorted(train),
        "normalization_subjects": sorted(train), "selector_subjects": sorted(selector),
        "selector_fit_subjects": sorted(train - selector), "config_sha256": config_identity(config)}
    scope = calibration_scope(config)
    identity = json.loads((bundle / "identity.json").read_text(encoding="utf-8"))
    identity["config_sha256"] = config_identity(config)
    write_json_atomic(bundle / "identity.json", identity)
    write_json_atomic(bundle / "resolved_config.json", config)
    payload = torch.load(bundle / "checkpoint.pt", weights_only=True)
    payload["config_sha256"] = config_identity(config)
    save_torch_atomic(bundle / "checkpoint.pt", payload)
    point = json.loads((bundle / "operating_point.json").read_text(encoding="utf-8"))
    point.update({"lineage": [lineage], **scope})
    write_json_atomic(bundle / "operating_point.json", point)
    gate = json.loads((bundle / "fold0_gate_manifest.json").read_text(encoding="utf-8"))
    gate.update({"identity_sha256": sha256_file(bundle / "identity.json"), **scope})
    write_json_atomic(bundle / "fold0_gate_manifest.json", gate)
    manifest.update({"lineage": [lineage], **scope, "identity_sha256": gate["identity_sha256"],
        "gate_sha256": sha256_file(bundle / "fold0_gate_manifest.json"),
        "evidence_sha256": {name: sha256_file(bundle / name) for name in
                            ("checkpoint.pt", "operating_point.json", "source_mapping.json")}})
    write_json_atomic(bundle / "final_manifest.json", manifest)
    update_hashes(bundle)
    assert StatsFusionJointFoldDetector(bundle).identity == "synthetic_fixture"
    assert replay_joint_bundle(bundle, config)["no_xgboost_replay"]
    manifest["full_internal_oof"] = True
    write_json_atomic(bundle / "final_manifest.json", manifest)
    update_hashes(bundle)
    with pytest.raises(RuntimeError, match="calibration scope"):
        StatsFusionJointFoldDetector(bundle)


def test_single_fold_bundle_rejects_test_subject_training_leakage(synthetic_bundle):
    from bme_eating.joint_single_fold import StatsFusionJointFoldDetector

    bundle, config = synthetic_bundle
    if config["experiment"]["code_version"] != "v4.11":
        pytest.skip("Single-fold continuation targets v411")
    convert_to_single_fold_bundle(bundle, config)
    manifest = json.loads((bundle / "final_manifest.json").read_text(encoding="utf-8"))
    manifest["test_subjects"] = [manifest["training_subjects"][0]]
    write_json_atomic(bundle / "final_manifest.json", manifest)
    update_hashes(bundle)
    with pytest.raises(RuntimeError, match="lineage"):
        StatsFusionJointFoldDetector(bundle)


def test_single_fold_export_flag_cannot_replace_required_qualifications(synthetic_bundle):
    from bme_eating.joint_single_fold import StatsFusionJointFoldDetector

    bundle, config = synthetic_bundle
    if config["experiment"]["code_version"] != "v4.11":
        pytest.skip("Single-fold continuation targets v411")
    convert_to_single_fold_bundle(bundle, config)
    gate = json.loads((bundle / "fold0_gate_manifest.json").read_text(encoding="utf-8"))
    gate["checks"].pop("boundary")
    write_json_atomic(bundle / "fold0_gate_manifest.json", gate)
    manifest = json.loads((bundle / "final_manifest.json").read_text(encoding="utf-8"))
    manifest["gate_sha256"] = sha256_file(bundle / "fold0_gate_manifest.json")
    write_json_atomic(bundle / "final_manifest.json", manifest)
    update_hashes(bundle)
    with pytest.raises(RuntimeError, match="manifest lock"):
        StatsFusionJointFoldDetector(bundle)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("single_fold", [False, True])
def test_gpu_v413_bundle_locked_replay(synthetic_bundle, single_fold):
    bundle, config = synthetic_bundle
    if config["experiment"]["code_version"] != "v4.13":
        pytest.skip("v413 GPU bundle probe")
    if single_fold:
        convert_to_single_fold_bundle(bundle, config)
    report = replay_joint_bundle(bundle, config, device="cuda")
    assert report["device"] == "cuda"
    assert report["gpu_validation_performed"]
    assert report["no_xgboost_replay"]


def test_single_fold_save_export_resume_and_partial_recovery(synthetic_bundle, tmp_path, monkeypatch):
    from bme_eating.joint_single_fold import StatsFusionJointFoldDetector, export_fold_bundle

    fixture, config = synthetic_bundle
    if config["experiment"]["code_version"] != "v4.11":
        pytest.skip("Single-fold continuation targets v411")
    convert_to_single_fold_bundle(fixture, config)
    run, source = tmp_path / "single_run", Path(__file__).resolve().parents[1]
    run.mkdir()
    for name in ("identity.json", "resolved_config.json"):
        shutil.copyfile(fixture / name, run / name)
    manifest = json.loads((fixture / "final_manifest.json").read_text(encoding="utf-8"))
    point = json.loads((fixture / "operating_point.json").read_text(encoding="utf-8"))
    checkpoint = run / "fold_0/model/latest.pt"
    checkpoint.parent.mkdir(parents=True)
    shutil.copyfile(fixture / "checkpoint.pt", checkpoint)
    mapping = run / "fold_0/source_fit/source_mapping.json"
    mapping.parent.mkdir(parents=True)
    shutil.copyfile(fixture / "source_mapping.json", mapping)
    report = {"identity": manifest["model_identity"], "epoch": 8, "operating_point": point,
        "checkpoint_qualified": True, "checkpoint_sha256": sha256_file(checkpoint),
        "training_subjects": manifest["training_subjects"], "prediction_subjects": manifest["test_subjects"],
        "boundary": {"passed": True}, "stress_f1_drop": 0,
        "metrics": {"f1": .6, "fp_per_hour": .05}, "coverage": {"recall": .8}}
    write_json_atomic(run / "fold_0/evaluation.json", report)
    write_json_atomic(run / "fold_0/completion.json", {"evidence_sha256": {
        "evaluation.json": sha256_file(run / "fold_0/evaluation.json"),
        "model/latest.pt": sha256_file(checkpoint)}})
    monkeypatch.setattr("bme_eating.joint_workflow.verify_outer_completion", lambda *args: report)
    write_json_atomic(run / "fold0_bundle.partial/incomplete.json", {"partial": True})
    exported = export_fold_bundle(None, config, run, source, resume=True)
    assert not (run / "fold0_bundle.partial").exists()
    assert list((run / "failures").glob("incomplete_fold0_bundle_*"))
    detector = StatsFusionJointFoldDetector(exported["bundle"])
    assert detector.identity == report["identity"]
    assert sha256_file(run / "fold0_final/checkpoint.pt") == report["checkpoint_sha256"]
    assert export_fold_bundle(None, config, run, source, resume=True)["resumed"]
    final_checkpoint = run / "fold0_final/checkpoint.pt"
    with final_checkpoint.open("ab") as handle:
        handle.write(b"changed")
    final_manifest = json.loads((run / "fold0_final/final_manifest.json").read_text(encoding="utf-8"))
    final_manifest["evidence_sha256"]["checkpoint.pt"] = sha256_file(final_checkpoint)
    write_json_atomic(run / "fold0_final/final_manifest.json", final_manifest)
    with pytest.raises(RuntimeError, match="weights"):
        export_fold_bundle(None, config, run, source, resume=True)


def test_bundle_false_gate_cannot_load(synthetic_bundle):
    bundle, _ = synthetic_bundle
    gate_path = bundle / "joint_gate_manifest.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["promotable"] = False
    write_json_atomic(gate_path, gate)
    update_hashes(bundle)
    with pytest.raises(RuntimeError, match="lock"):
        StatsFusionJointDetector(bundle)


def test_bundle_checkpoint_hash_change_is_detected(synthetic_bundle):
    bundle, _ = synthetic_bundle
    with (bundle / "checkpoint.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(RuntimeError, match="checksum"):
        StatsFusionJointDetector(bundle)


def test_declared_boundary_cannot_bypass_qualification(synthetic_bundle):
    bundle, _ = synthetic_bundle
    point_path = bundle / "operating_point.json"
    point = json.loads(point_path.read_text(encoding="utf-8"))
    point["boundary_enabled"] = True
    write_json_atomic(point_path, point)
    manifest_path = bundle / "final_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["operating_point_sha256"] = sha256_file(point_path)
    write_json_atomic(manifest_path, manifest)
    update_hashes(bundle)
    with pytest.raises(RuntimeError, match="Boundary"):
        StatsFusionJointDetector(bundle)


class UnsafeCheckpoint:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return Path.write_text, (self.marker, "unsafe pickle executed")


def test_bundle_safe_load_never_executes_pickle(synthetic_bundle, tmp_path):
    bundle, _ = synthetic_bundle
    marker = tmp_path / "pickle_execution.txt"
    torch.save(UnsafeCheckpoint(marker), bundle / "checkpoint.pt")
    update_hashes(bundle)
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        StatsFusionJointDetector(bundle)
    assert not marker.exists()


def test_bundle_rejects_unlisted_python_source(synthetic_bundle):
    bundle, _ = synthetic_bundle
    (bundle / "src" / "unlisted.py").write_text("unlisted artifact", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unlisted"):
        StatsFusionJointDetector(bundle)


def test_bundle_promotable_flag_cannot_replace_complete_gate_checks(synthetic_bundle):
    bundle, _ = synthetic_bundle
    gate_path = bundle / "joint_gate_manifest.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["checks"].pop("bootstrap")
    write_json_atomic(gate_path, gate)
    manifest_path = bundle / "final_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gate_sha256"] = sha256_file(gate_path)
    write_json_atomic(manifest_path, manifest)
    update_hashes(bundle)
    with pytest.raises(RuntimeError, match="formal checks"):
        StatsFusionJointDetector(bundle)


def test_actual_export_builds_safe_bundle_and_revalidates_resume(synthetic_bundle, tmp_path):
    fixture, config = synthetic_bundle
    source = Path(__file__).resolve().parents[1]
    run, final = tmp_path / "export_run", tmp_path / "export_run" / "final"
    internal = final / "internal"
    run.mkdir()
    for name in ("identity.json", "resolved_config.json"):
        shutil.copyfile(fixture / name, run / name)
    run_identity = sha256_file(run / "identity.json")
    payload = torch.load(fixture / "checkpoint.pt", weights_only=True)
    subjects = set(payload["training_subjects"])
    identities = {name: payload[name] for name in ("config_sha256", "source_sha256", "data_sha256")}
    lineages = []
    for partition in range(3):
        prediction = set(sorted(subjects)[partition * 2:partition * 2 + 2])
        train = subjects - prediction
        fit, selector = set(sorted(train)[:-1]), set(sorted(train)[-1:])
        model_identity = digest({"run": run_identity, "training": sorted(train), "prediction": sorted(prediction),
                                 "outer": [], "partition": partition})
        lineages.append({"partition": partition, "model_identity": model_identity, "epoch": 8,
            "training_subjects": sorted(train), "prediction_subjects": sorted(prediction),
            "outer_excluded_subjects": [], "source_training_subjects": sorted(train),
            "normalization_subjects": sorted(train), "selector_fit_subjects": sorted(fit),
            "selector_subjects": sorted(selector), "checkpoint_qualified": True,
            "selector_promotion_eligible": True, **identities})
        for relative, training, excluded in (("retrain/latest.pt", train, prediction),
                                             ("selection/selected.pt", fit, prediction | selector)):
            checkpoint = {**payload, "identity": model_identity, "training_subjects": sorted(training),
                "excluded_subjects": sorted(excluded),
                "preprocessing": {**payload["preprocessing"], "training_subjects": sorted(training)}}
            save_torch_atomic(internal / f"partition_{partition}" / relative, checkpoint)
    point = json.loads((fixture / "operating_point.json").read_text(encoding="utf-8"))
    point["lineage"] = lineages
    write_json_atomic(internal / "operating_point.json", point)
    write_json_atomic(internal / "boundary_crossfit.json", [])
    write_json_atomic(internal / "search.json", [])
    pd.DataFrame({"subject_key": sorted(subjects)}).to_parquet(internal / "internal_oof_windows.parquet", index=False)
    save_torch_atomic(internal / "internal_oof.pt", {"lineages": lineages})
    evidence = {path.relative_to(internal).as_posix(): sha256_file(path)
                for path in internal.rglob("*") if path.is_file()}
    write_json_atomic(internal / "internal_manifest.json", {"protocol": protocol_for(config),
        "run_identity": run_identity, "training_subjects": sorted(subjects), "outer_excluded_subjects": [],
        **identities, "evidence_sha256": evidence})
    save_torch_atomic(final / "model" / "latest.pt", payload)
    write_json_atomic(final / "operating_point.json", point)
    shutil.copyfile(fixture / "source_mapping.json", final / "source_mapping.json")
    gate = json.loads((fixture / "joint_gate_manifest.json").read_text(encoding="utf-8"))
    reference = tmp_path / "reference"
    write_json_atomic(reference / "evidence.json", {"synthetic_test_fixture": True})
    gate["reference"].update({"root": str(reference),
                              "evidence_sha256": {"evidence.json": sha256_file(reference / "evidence.json")}})
    gate["evidence_sha256"] = {}
    for fold in range(5):
        relative = f"fold_{fold}/completion.json"
        write_json_atomic(run / relative, {"synthetic_test_fixture": True})
        gate["evidence_sha256"][relative] = sha256_file(run / relative)
    write_json_atomic(run / "joint_gate_manifest.json", gate)
    manifest = json.loads((fixture / "final_manifest.json").read_text(encoding="utf-8"))
    manifest.update({"gate_sha256": sha256_file(run / "joint_gate_manifest.json"), "lineage": lineages,
        "checkpoint_sha256": sha256_file(final / "model" / "latest.pt"),
        "operating_point_sha256": sha256_file(final / "operating_point.json"),
        "internal_oof_sha256": sha256_file(internal / "internal_oof.pt"),
        "internal_manifest_sha256": sha256_file(internal / "internal_manifest.json")})
    write_json_atomic(final / "final_manifest.json", manifest)
    exported = export_joint_bundle(config, run, source)
    detector = StatsFusionJointDetector(exported["bundle"])
    assert detector.identity == "synthetic_fixture"
    assert export_joint_bundle(config, run, source, resume=True)["resumed"]
    with pytest.raises(FileExistsError, match="cannot overwrite"):
        export_joint_bundle(config, run, source)
    changed_payload = {**payload, "model": {name: value.clone() for name, value in payload["model"].items()}}
    first_weight = next(value for value in changed_payload["model"].values() if value.is_floating_point())
    first_weight.add_(.01)
    save_torch_atomic(final / "model" / "latest.pt", changed_payload)
    manifest["checkpoint_sha256"] = sha256_file(final / "model" / "latest.pt")
    write_json_atomic(final / "final_manifest.json", manifest)
    with pytest.raises(RuntimeError, match="weights differ"):
        export_joint_bundle(config, run, source, resume=True)
    save_torch_atomic(final / "model" / "latest.pt", payload)
    manifest["checkpoint_sha256"] = sha256_file(final / "model" / "latest.pt")
    write_json_atomic(final / "final_manifest.json", manifest)
    (run / "bundle" / "unexpected.py").write_text("unlisted source", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unlisted"):
        export_joint_bundle(config, run, source, resume=True)
