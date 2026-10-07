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
    PROTOCOL,
    config_identity,
    digest,
    save_torch_atomic,
    source_identity,
)
from bme_eating.joint_validation import replay_joint_bundle
from bme_eating.joint_workflow import export_joint_bundle
from bme_eating.models.statsfusion_joint_event import StatsFusionJointEvent
from bme_eating.stats_features import STATS_FEATURE_COLUMNS, FoldRobustScaler


@pytest.fixture
def synthetic_bundle(tmp_path):
    source = Path(__file__).resolve().parents[1]
    config = load_config(source / "configs/statsfusion_v410_joint_event.yaml")
    bundle = tmp_path / "synthetic_bundle"
    bundle.mkdir()
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    model = StatsFusionJointEvent(config)
    subjects = [f"fit_{index}" for index in range(6)]
    identity = {"synthetic_test_fixture": True, "source": source_identity(source), "data": {"fixture": "synthetic"}}
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
    gate = {"protocol": PROTOCOL, "promotable": True, "identity_sha256": identity_hash,
            "synthetic_test_fixture": True,
            "folds": [{"fold": fold, "prediction_subjects": subjects[fold::5]} for fold in range(5)],
            "reference": {"qualified": True, "comparison_performed": True},
            "checks": {name: True for name in ("five_folds", "training_qualified", "f1", "fp_per_hour", "coverage",
                "truth_cohort", "fold_coverage", "operating_point_qualified", "stress", "boundary", "reference",
                "bootstrap", "matching_ranking", "same_recall", "different_recall")}}
    write_json_atomic(bundle / "joint_gate_manifest.json", gate)
    write_json_atomic(bundle / "resolved_config.json", config)
    write_json_atomic(bundle / "operating_point.json", {"setting": "high_0.1_gap_15", "budget": 30,
        "acceptance": .1, "nms": .5, "promotion_eligible": True, "boundary_enabled": False,
        "calibration": {"slope": 1, "intercept": 0, "qualified": True}})
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
    write_json_atomic(bundle / "final_manifest.json", {"protocol": PROTOCOL, "promotable": True,
        "model_identity": "synthetic_fixture", "identity_sha256": identity_hash,
        "training_subjects": subjects,
        "gate_sha256": sha256_file(bundle / "joint_gate_manifest.json"), "boundary_enabled": False,
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
    write_json_atomic(internal / "internal_manifest.json", {"protocol": PROTOCOL,
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
    (run / "bundle" / "unexpected.py").write_text("unlisted source", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unlisted"):
        export_joint_bundle(config, run, source, resume=True)
