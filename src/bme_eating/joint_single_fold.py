from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import pandas as pd
import torch

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_capabilities import calibration_discriminates, supported
from bme_eating.joint_pipeline import StatsFusionJointDetector
from bme_eating.joint_protocol import (
    calibration_scope,
    checkpoint_qualified,
    config_identity,
    configure_numerics,
    digest,
    event_score_formula,
    execution_mode,
    protocol_for,
    source_identity,
    validate_config,
    verify_artifact_hashes,
    verify_calibration_lineage,
)
from bme_eating.joint_stream import FrozenSourceCDF


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fold_reference_comparison(inputs, config, run):
    from bme_eating.joint_evaluation import paired_subject_bootstrap
    from bme_eating.joint_workflow import audit_reference
    from bme_eating.metrics import evaluate_events

    previous = run / "reference_audit.json"
    audit = read_json(previous) if previous.exists() else {}
    if not audit.get("root"):
        return {"qualified": False, "comparison_performed": False,
                "reasons": audit.get("reasons", ["Completed reference unavailable"])}
    reference = Path(audit["root"])
    audit = audit_reference(reference, inputs, folds=[0])
    if not audit["qualified"]:
        return audit
    current = run / "fold_0"
    other = reference / "fold_0"
    for name, keys in (("truth.parquet", None), ("ignore.parquet", None),
                       ("outer_windows.parquet", ["subject_key", "session_id", "timestamp_ms", "valid"])):
        left, right = pd.read_parquet(current / name), pd.read_parquet(other / name)
        keys = keys or sorted(left.columns)
        if (not set(keys).issubset(right.columns)
                or digest(left[keys].sort_values(keys).to_dict("records"))
                != digest(right[keys].sort_values(keys).to_dict("records"))):
            return {**audit, "qualified": False, "comparison_performed": False,
                    "reasons": ["Reference truth/ignore/exposure differs: " + name]}
    truth, ignore = pd.read_parquet(current / "truth.parquet"), pd.read_parquet(current / "ignore.parquet")
    predictions = pd.read_parquet(other / "outer_predictions.parquet")
    metrics, _ = evaluate_events(truth, predictions, ignore=ignore)
    evidence = torch.load(current / "outer_evidence.pt", map_location="cpu", weights_only=False)
    paired = paired_subject_bootstrap(truth, evidence["prediction"], predictions, ignore,
        pd.read_parquet(current / "outer_windows.parquet"), config["gates"]["bootstrap_replicates"])
    return {**audit, "comparison_performed": True, "reference_metrics": metrics,
            "paired_bootstrap": paired, "evidence_scope": "subject_excluded_development_only"}


def execute_fold_gate(inputs, config, run):
    from bme_eating.joint_workflow import verify_outer_completion

    if execution_mode(config) != "fold0":
        raise RuntimeError("Single-fold artifacts require explicit fold0 mode")
    report = verify_outer_completion(inputs, config, run, 0)
    point = report["operating_point"]
    checks = {
        "training_qualified": report["checkpoint_qualified"],
        "operating_point_qualified": point["promotion_eligible"],
        "boundary": not point["boundary_enabled"] or report["boundary"]["passed"],
    }
    metric_checks = {
        "f1": report["metrics"]["f1"] >= config["gates"]["minimum_f1"],
        "fp_per_hour": report["metrics"]["fp_per_hour"] <= config["gates"]["maximum_fp_per_hour"],
        "candidate_recall": report["coverage"]["recall"] >= config["gates"]["minimum_fold_candidate_recall"],
        "stress": report["stress_f1_drop"] <= config["gates"]["maximum_stress_f1_drop"],
    }
    completion = read_json(run / "fold_0" / "completion.json")
    evidence = {"fold_0/completion.json": sha256_file(run / "fold_0" / "completion.json")}
    evidence.update({"fold_0/" + name: value for name, value in completion["evidence_sha256"].items()})
    if (run / "single_partition_migration.json").exists():
        evidence["single_partition_migration.json"] = sha256_file(run / "single_partition_migration.json")
    gate = {"protocol": protocol_for(config), "evaluation_mode": "fold0",
        "artifact_kind": "single_fold_diagnostic", "fold": 0,
        **calibration_scope(config),
        "identity_sha256": sha256_file(run / "identity.json"),
        "diagnostic_complete": True, "promotable": False, "five_folds_complete": False,
        "comparison_performed": False, "export_eligible": all(checks.values()),
        "checks": checks, "metric_checks": metric_checks, "metrics": report["metrics"],
        "coverage": report["coverage"], "training_subjects": report["training_subjects"],
        "test_subjects": report["prediction_subjects"], "evidence_sha256": evidence}
    if supported(config, "consistent_inputs"):
        comparison = fold_reference_comparison(inputs, config, run)
        gate.update({"reference": comparison, "comparison_performed": comparison["comparison_performed"],
                     "s0": read_json(run / "identity.json")["s0"],
                     "evidence_scope": "subject_excluded_development_only"})
    write_json_atomic(run / "fold0_gate_manifest.json", gate)
    if not gate["export_eligible"]:
        raise RuntimeError("Single-fold checkpoint, operating point or Boundary qualification failed")
    return gate


def save_fold_final(inputs, config, run, *, resume=False):
    gate = execute_fold_gate(inputs, config, run)
    report = read_json(run / "fold_0" / "evaluation.json")
    final = run / "fold0_final"
    manifest_path = final / "final_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Single-fold final already exists")
        manifest = read_json(manifest_path)
        verify_artifact_hashes(final, manifest["evidence_sha256"])
        if (manifest["gate_sha256"] != sha256_file(run / "fold0_gate_manifest.json")
                or manifest["identity_sha256"] != sha256_file(run / "identity.json")
                or manifest["model_identity"] != report["identity"]
                or manifest["evidence_sha256"].get("checkpoint.pt") != report["checkpoint_sha256"]
                or read_json(final / "operating_point.json") != report["operating_point"]
                or sha256_file(final / "source_mapping.json") != sha256_file(run / "fold_0/source_fit/source_mapping.json")):
            raise RuntimeError("Single-fold final gate, weights or operating point changed")
        return manifest
    final.mkdir(parents=True, exist_ok=True)
    source_checkpoint = run / "fold_0" / "model" / "latest.pt"
    shutil.copyfile(source_checkpoint, final / "checkpoint.pt")
    write_json_atomic(final / "operating_point.json", report["operating_point"])
    shutil.copyfile(run / "fold_0" / "source_fit" / "source_mapping.json",
                    final / "source_mapping.json")
    manifest = {"protocol": protocol_for(config), "artifact_kind": "single_fold_diagnostic",
        "evaluation_mode": "fold0", "fold": 0, "promotable": False,
        **calibration_scope(config),
        "export_eligible": gate["export_eligible"], "model_identity": report["identity"],
        "identity_sha256": sha256_file(run / "identity.json"),
        "gate_sha256": sha256_file(run / "fold0_gate_manifest.json"),
        "epoch": report["epoch"], "training_subjects": report["training_subjects"],
        "test_subjects": report["prediction_subjects"], "metrics": report["metrics"],
        "lineage": report["operating_point"]["lineage"],
        "boundary_enabled": report["operating_point"]["boundary_enabled"],
        "boundary_mode": report["operating_point"].get("boundary_mode", "off"),
        "score_formula": report["operating_point"]["score_formula"],
        "uses_test_labels_for_selection": False,
        "final_training": "saved_outer_fold0_model_no_test_subject_retraining"}
    manifest["evidence_sha256"] = {name: sha256_file(final / name) for name in
                                   ("checkpoint.pt", "operating_point.json", "source_mapping.json")}
    write_json_atomic(manifest_path, manifest)
    return manifest


def export_fold_bundle(inputs, config, run, source_root, *, resume=False):
    manifest = save_fold_final(inputs, config, run, resume=True)
    bundle = run / "fold0_bundle"
    if bundle.exists():
        if not resume:
            raise FileExistsError("Single-fold bundle already exists")
        detector = StatsFusionJointFoldDetector(bundle)
        if (detector.identity != manifest["model_identity"]
                or sha256_file(bundle / "final_manifest.json") != sha256_file(run / "fold0_final" / "final_manifest.json")
                or any(sha256_file(bundle / name) != expected for name, expected in manifest["evidence_sha256"].items())):
            raise RuntimeError("Existing single-fold bundle weights or manifest differ")
        return {"bundle": str(bundle), "resumed": True}
    staging = run / "fold0_bundle.partial"
    if staging.exists():
        if not resume:
            raise RuntimeError("Incomplete single-fold bundle requires --resume")
        backup = run / "failures" / ("incomplete_fold0_bundle_" + uuid.uuid4().hex)
        backup.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, backup)
    staging.mkdir()
    final = run / "fold0_final"
    for name in ("checkpoint.pt", "operating_point.json", "source_mapping.json", "final_manifest.json"):
        shutil.copyfile(final / name, staging / name)
    for name in ("identity.json", "resolved_config.json", "fold0_gate_manifest.json"):
        shutil.copyfile(run / name, staging / name)
    for relative, checksum in source_identity(source_root).items():
        if relative.startswith("src/"):
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_root / relative, target)
            if sha256_file(target) != checksum:
                raise RuntimeError("Source changed during single-fold export")
    hashes = {path.relative_to(staging).as_posix(): sha256_file(path)
              for path in staging.rglob("*") if path.is_file()}
    write_json_atomic(staging / "SHA256SUMS.json", hashes)
    StatsFusionJointFoldDetector(staging)
    os.replace(staging, bundle)
    return {"bundle": str(bundle), "promotable": False, "artifact_kind": "single_fold_diagnostic"}


class StatsFusionJointFoldDetector(StatsFusionJointDetector):
    def __init__(self, bundle: str | Path, *, device="cpu"):
        root = Path(bundle)
        hashes = read_json(root / "SHA256SUMS.json")
        required = {"checkpoint.pt", "operating_point.json", "source_mapping.json", "final_manifest.json",
                    "identity.json", "resolved_config.json", "fold0_gate_manifest.json"}
        if not required.issubset(hashes):
            raise RuntimeError("Incomplete single-fold bundle")
        verify_artifact_hashes(root, hashes)
        files = {path.relative_to(root).as_posix() for path in root.rglob("*")
                 if path.is_file() and path.name != "SHA256SUMS.json" and "__pycache__" not in path.parts}
        if files != set(hashes):
            raise RuntimeError("Single-fold checksum coverage differs")
        manifest, gate = read_json(root / "final_manifest.json"), read_json(root / "fold0_gate_manifest.json")
        self.config = read_json(root / "resolved_config.json")
        validate_config(self.config)
        identity = read_json(root / "identity.json")
        qualification_checks = {"training_qualified", "operating_point_qualified", "boundary"}
        if (execution_mode(self.config) != "fold0" or manifest.get("artifact_kind") != "single_fold_diagnostic"
                or gate.get("artifact_kind") != "single_fold_diagnostic"
                or manifest.get("evaluation_mode") != "fold0" or gate.get("evaluation_mode") != "fold0"
                or manifest.get("promotable") is not False or gate.get("promotable") is not False
                or gate.get("five_folds_complete") is not False
                or manifest.get("uses_test_labels_for_selection") is not False
                or gate.get("diagnostic_complete") is not True
                or manifest.get("export_eligible") is not True or gate.get("export_eligible") is not True
                or not qualification_checks.issubset(gate.get("checks", {}))
                or any(value is not True for value in gate["checks"].values())
                or gate.get("fold") != 0 or manifest.get("fold") != 0
                or manifest.get("protocol") != protocol_for(self.config)
                or gate.get("protocol") != manifest.get("protocol")
                or identity.get("protocol") != manifest.get("protocol")
                or identity.get("config_sha256") != config_identity(self.config)
                or manifest.get("identity_sha256") != sha256_file(root / "identity.json")
                or gate.get("identity_sha256") != manifest["identity_sha256"]
                or manifest.get("gate_sha256") != sha256_file(root / "fold0_gate_manifest.json")):
            raise RuntimeError("Single-fold diagnostic manifest lock failed")
        verify_artifact_hashes(root, manifest["evidence_sha256"])
        self.point = read_json(root / "operating_point.json")
        payload = torch.load(root / "checkpoint.pt", map_location=device, weights_only=True)
        train, test = set(manifest["training_subjects"]), set(manifest["test_subjects"])
        identities = {"config_sha256": config_identity(self.config),
                      "source_sha256": digest(identity["source"]), "data_sha256": digest(identity["data"])}
        if (not train or not test or train & test
                or set(gate["training_subjects"]) != train or set(gate["test_subjects"]) != test
                or set(payload.get("excluded_subjects", [])) != test
                or set(payload["preprocessing"]["training_subjects"]) != train
                or not checkpoint_qualified(payload, manifest["model_identity"], train,
                                            self.config["training"]["checkpoint_min_epoch"])
                or any(payload.get(key) != value for key, value in identities.items())
                or self.point.get("lineage") != manifest["lineage"]
                or not self.point.get("promotion_eligible") or not self.point["calibration"].get("qualified")
                or not calibration_discriminates(self.config, self.point["calibration"])
                or self.point.get("score_formula") != event_score_formula(self.config)
                or self.point.get("boundary_mode") not in {"off", "start_only", "both"}
                or self.point["boundary_enabled"] != manifest["boundary_enabled"]
                or self.point["boundary_enabled"] != (self.point["boundary_mode"] != "off")
                or self.point["boundary_mode"] != manifest["boundary_mode"]
                or self.point["boundary_enabled"] and not payload["boundary"]["qualified"]):
            raise RuntimeError("Single-fold checkpoint, exclusion or operating-point lineage failed")
        verify_calibration_lineage(manifest["lineage"], train, test, self.config, identities=identities)
        if not calibration_scope(self.config)["full_internal_oof"] and any(
                manifest.get(name) != value or gate.get(name) != value or self.point.get(name) != value
                for name, value in calibration_scope(self.config).items()):
            raise RuntimeError("Single-fold calibration scope mismatch")
        for relative, expected in identity["source"].items():
            if relative.startswith("src/") and hashes.get(relative) != expected:
                raise RuntimeError("Single-fold bundle source differs from frozen identity")
        configure_numerics(self.config)
        from bme_eating.models.statsfusion_joint_event_v411 import build_joint_model
        self.model = build_joint_model(self.config).to(device).eval()
        self.model.load_state_dict(payload["model"])
        self.preprocessing, self.boundary = payload["preprocessing"], payload["boundary"]
        self.identity = payload["identity"]
        self.mapping = FrozenSourceCDF(**read_json(root / "source_mapping.json"))
        if self.mapping.training_subjects != sorted(train):
            raise RuntimeError("Single-fold source normalization cohort differs")
