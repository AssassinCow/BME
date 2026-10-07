from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_evaluation import candidate_coverage
from bme_eating.joint_protocol import PROTOCOL, require_gate, write_failure
from bme_eating.joint_stream import JointCandidate
from bme_eating.joint_workflow import audit_reference


@pytest.fixture
def reference_evidence(tmp_path):
    folds = {f"subject_{index}": index // 2 for index in range(10)}
    inputs = SimpleNamespace(subject_folds=folds, identity={"subject_folds": "split"})
    write_json_atomic(tmp_path / "identity.json", {"seed": 2026})
    pd.DataFrame(columns=["subject_key", "start_ms", "end_ms"]).to_parquet(tmp_path / "predictions.parquet")
    pd.DataFrame({"subject_key": sorted(folds), "session_id": "session", "timestamp_ms": 3000,
                  "valid": 1}).to_parquet(tmp_path / "exposure.parquet")
    payload = {"truth_sha256": "truth", "ignore_sha256": "ignore", "subject_folds_sha256": "split",
               "matching": "max_cardinality_iou", "iou_threshold": .25, "strict_iou": True,
               "seed": 2026, "candidate_budget": 20, "full_outer_subjects": sorted(folds),
               "model_lineages": [], "calibration_lineage": []}
    for name, relative in (("identity", "identity.json"), ("predictions", "predictions.parquet"),
                           ("exposure", "exposure.parquet")):
        payload[name + "_file"] = relative
        payload[name + "_sha256"] = sha256_file(tmp_path / relative)
    for fold in range(5):
        prediction = sorted(subject for subject, index in folds.items() if index == fold)
        train = sorted(set(folds) - set(prediction))
        for kind in ("model_lineages", "calibration_lineage"):
            relative = f"{kind}_{fold}.json"
            write_json_atomic(tmp_path / relative, {"training_subjects": train})
            lineage = {"fold": fold, "training_subjects": train, "prediction_subjects": prediction,
                       "artifact_file": relative, "artifact_sha256": sha256_file(tmp_path / relative)}
            if kind == "calibration_lineage":
                lineage["oof_lineages"] = [
                    {"training_subjects": train[4:], "prediction_subjects": train[:4]},
                    {"training_subjects": train[:4], "prediction_subjects": train[4:]}]
            payload[kind].append(lineage)
    path = tmp_path / "joint_reference_evidence.json"
    write_json_atomic(path, payload)
    return tmp_path, path, payload, inputs


def test_reference_requires_complete_audited_fold_and_calibration_exclusion(reference_evidence):
    root, path, payload, inputs = reference_evidence
    assert audit_reference(root, inputs)["qualified"]
    payload["model_lineages"] = payload["model_lineages"][:1]
    write_json_atomic(path, payload)
    report = audit_reference(root, inputs)
    assert not report["qualified"]
    assert any("five outer folds" in reason for reason in report["reasons"])


def test_reference_artifact_and_inner_oof_leakage_cannot_be_claimed_qualified(reference_evidence):
    root, path, payload, inputs = reference_evidence
    payload["calibration_lineage"][0]["oof_lineages"][0]["training_subjects"].append("subject_0")
    write_json_atomic(path, payload)
    assert not audit_reference(root, inputs)["qualified"]
    payload["calibration_lineage"][0]["oof_lineages"][0]["training_subjects"].remove("subject_0")
    write_json_atomic(path, payload)
    write_json_atomic(root / "model_lineages_0.json", {"training_subjects": ["subject_0"]})
    assert not audit_reference(root, inputs)["qualified"]


def test_gate_rechecks_frozen_reference_hashes(reference_evidence):
    root, _, _, inputs = reference_evidence
    run = root / "run"
    run.mkdir()
    write_json_atomic(run / "identity.json", {"seed": 2026})
    write_json_atomic(run / "fold.json", {"complete": True})
    evidence = {}
    for fold in range(5):
        relative = f"fold_{fold}/completion.json"
        write_json_atomic(run / relative, {"fold": fold})
        evidence[relative] = sha256_file(run / relative)
    reference = audit_reference(root, inputs)
    reference["comparison_performed"] = True
    write_json_atomic(run / "joint_gate_manifest.json", {
        "protocol": PROTOCOL, "promotable": True, "reference": reference,
        "identity_sha256": sha256_file(run / "identity.json"),
        "folds": [{"fold": fold} for fold in range(5)],
        "checks": {name: True for name in ("five_folds", "training_qualified", "f1", "fp_per_hour", "coverage",
            "truth_cohort", "fold_coverage", "operating_point_qualified", "stress", "boundary", "reference",
            "bootstrap", "matching_ranking", "same_recall", "different_recall")},
        "evidence_sha256": evidence})
    assert require_gate(run)["promotable"]
    write_json_atomic(root / "identity.json", {"seed": 2027})
    with pytest.raises(RuntimeError, match="reference evidence changed"):
        require_gate(run)


def test_failed_promotion_quarantines_bundle_and_keeps_diagnostics(tmp_path):
    write_json_atomic(tmp_path / "identity.json", {"seed": 2026})
    for name in ("bundle", "bundle.partial"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "evidence.txt").write_text("diagnostic evidence", encoding="utf-8")
    write_failure(tmp_path, "gates", RuntimeError("promotion failed"))
    assert not (tmp_path / "bundle").exists()
    assert not (tmp_path / "bundle.partial").exists()
    report = json.loads((tmp_path / "failure_report.json").read_text(encoding="utf-8"))
    assert not report["deployable"]
    assert len(report["quarantined_artifacts"]) == 2
    for relative in report["quarantined_artifacts"]:
        assert (tmp_path / relative / "evidence.txt").read_text(encoding="utf-8") == "diagnostic evidence"


def test_unreadable_reference_closes_gate_without_aborting_new_model_diagnostics(tmp_path):
    (tmp_path / "joint_reference_evidence.json").write_text("{invalid", encoding="utf-8")
    report = audit_reference(tmp_path, SimpleNamespace())
    assert not report["qualified"]
    assert "Reference audit failed" in report["reasons"][0]


def test_coverage_reports_duration_hand_and_gyro_missingness():
    truth = pd.DataFrame({"subject_key": "subject", "session_id": "session",
        "start_ms": [0, 100_000, 300_000], "end_ms": [30_000, 160_000, 1_200_000],
        "source_event_key": ["short", "medium", "long"], "hand_relation": ["unknown", "same", "different"]})
    windows = pd.DataFrame({"subject_key": "subject", "session_id": "session",
        "timestamp_ms": [3000, 103_000, 106_000, 303_000], "valid": 1, "gyro_valid": [0, 0, 1, 1]})
    prediction = JointCandidate("subject", "session", 0, 30_000, 60_000, "proposal", 3, .8,
        0, 30_000, "family", "proposal", "parent", "hash", "model", {}, 60_000)
    report = candidate_coverage([prediction], truth, windows)
    assert report["strata"]["duration_short"] == {"truth": 1, "covered": 1}
    assert report["strata"]["duration_medium"]["truth"] == 1
    assert report["strata"]["duration_long"]["truth"] == 1
    assert report["strata"]["gyro_missing"]["covered"] == 1
    assert report["strata"]["gyro_mixed"]["truth"] == 1
    assert report["strata"]["gyro_present"]["truth"] == 1
    assert report["strata"]["hand_unknown"]["covered"] == 1
