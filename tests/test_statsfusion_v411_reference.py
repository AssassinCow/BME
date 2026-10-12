from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_protocol import config_identity, digest, save_torch_atomic
from bme_eating.joint_workflow import audit_reference


@pytest.fixture
def completed_reference(tmp_path):
    subjects = {f"subject_{index}": index // 2 for index in range(10)}
    inputs = SimpleNamespace(subject_folds=subjects, identity={"subject_folds": "split"})
    source = {"model.py": "a" * 64}
    reference_config = {"experiment": {"protocol_version": "statsfusion-joint-event-v1"},
                        "execution": {"mode": "fivefold", "internal_mode": "oof"}}
    write_json_atomic(tmp_path / "resolved_config.json", reference_config)
    identity = {"protocol": "statsfusion-joint-event-v1", "seed": 2026,
                "data": inputs.identity, "source": source, "config_sha256": config_identity(reference_config)}
    write_json_atomic(tmp_path / "identity.json", identity)
    write_json_atomic(tmp_path / "joint_gate_manifest.json", {"promotable": False})
    identity_hash = sha256_file(tmp_path / "identity.json")
    identities = {"config_sha256": identity["config_sha256"], "source_sha256": digest(source),
                  "data_sha256": digest(inputs.identity)}
    for fold in range(5):
        directory = tmp_path / f"fold_{fold}"
        holdout = {subject for subject, number in subjects.items() if number == fold}
        training = set(subjects) - holdout
        payload = {"epoch": 8, "epoch_complete": True, "promotion_eligible": True,
                   "clipping_fraction": .1, "identity": f"outer_{fold}",
                   "model": {"weight": torch.ones(1)}, "training_subjects": sorted(training),
                   "excluded_subjects": sorted(holdout), **identities,
                   "preprocessing": {"training_subjects": sorted(training)}}
        save_torch_atomic(directory / "model" / "latest.pt", payload)
        lineages = []
        for partition in range(3):
            prediction = set(sorted(training)[partition::3])
            fit_all = training - prediction
            selector = set(sorted(fit_all)[:1])
            fit = fit_all - selector
            model_identity = f"internal_{fold}_{partition}"
            lineage = {"partition": partition, "model_identity": model_identity,
                "training_subjects": sorted(fit_all), "prediction_subjects": sorted(prediction),
                "outer_excluded_subjects": sorted(holdout), "source_training_subjects": sorted(fit_all),
                "normalization_subjects": sorted(fit_all), "selector_fit_subjects": sorted(fit),
                "selector_subjects": sorted(selector), "checkpoint_qualified": True,
                "selector_promotion_eligible": False, "epoch": 8, **identities}
            lineages.append(lineage)
            for relative, selected_training, excluded in (("retrain/latest.pt", fit_all, holdout | prediction),
                    ("selection/diagnostic_selected.pt", fit, holdout | prediction | selector)):
                checkpoint = {**payload, "identity": model_identity, "training_subjects": sorted(selected_training),
                    "excluded_subjects": sorted(excluded), "preprocessing": {"training_subjects": sorted(selected_training)},
                    "promotion_eligible": relative.startswith("retrain")}
                save_torch_atomic(directory / "internal" / f"partition_{partition}" / relative, checkpoint)
        point = {"lineage": lineages, "budget": 20, "calibration": {"qualified": True, "slope": 1, "intercept": 0},
                 "promotion_eligible": False}
        write_json_atomic(directory / "internal" / "operating_point.json", point)
        internal = directory / "internal"
        write_json_atomic(internal / "internal_manifest.json", {"evidence_sha256": {
            path.relative_to(internal).as_posix(): sha256_file(path) for path in internal.rglob("*") if path.is_file()}})
        write_json_atomic(directory / "evaluation.json", {"identity": payload["identity"],
            "training_subjects": sorted(training), "prediction_subjects": sorted(holdout), "operating_point": point,
            "coverage": {"covered": 999}})
        frame = pd.DataFrame([{"subject_key": subject, "session_id": "session", "event_id": subject,
                              "start_ms": 0, "end_ms": 6000} for subject in sorted(holdout)])
        for name in ("truth", "outer_predictions", "outer_candidates"):
            frame.to_parquet(directory / (name + ".parquet"), index=False)
        frame.iloc[:0].to_parquet(directory / "ignore.parquet", index=False)
        pd.DataFrame([{"subject_key": subject, "session_id": "session", "timestamp_ms": 6000, "valid": 1}
                      for subject in sorted(holdout)]).to_parquet(directory / "outer_windows.parquet", index=False)
        evidence = {path.relative_to(directory).as_posix(): sha256_file(path) for path in directory.rglob("*") if path.is_file()}
        write_json_atomic(directory / "completion.json", {"protocol": identity["protocol"],
                                                          "run_identity": identity_hash, "evidence_sha256": evidence})
    return inputs


def test_completed_v410_reference_qualifies_without_old_promotion(tmp_path, completed_reference):
    inputs = completed_reference
    result = audit_reference(tmp_path, inputs)
    assert result["qualified"]
    assert result["payload"]["candidate_covered"] == 10
    assert not result["comparison_performed"]
    checkpoint = tmp_path / "fold_4" / "model" / "latest.pt"
    checkpoint.write_bytes(b"changed")
    assert not audit_reference(tmp_path, inputs)["qualified"]


def test_reference_rejects_changed_resolved_configuration(tmp_path, completed_reference):
    import json

    path = tmp_path / "resolved_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["execution"]["internal_mode"] = "single"
    write_json_atomic(path, config)
    result = audit_reference(tmp_path, completed_reference)
    assert not result["qualified"] and not result["comparison_performed"]


@pytest.mark.parametrize("relative", ["retrain/latest.pt", "selection/diagnostic_selected.pt"])
@pytest.mark.parametrize("field,value", [("source_sha256", "c" * 64), ("config_sha256", "c" * 64),
    ("data_sha256", "c" * 64), ("epoch", 10)])
def test_reference_rejects_internal_identity_mismatch_with_valid_hashes(tmp_path, completed_reference, relative, field, value):
    import json
    directory = tmp_path / "fold_0"
    internal = directory / "internal"
    checkpoint = internal / "partition_0" / relative
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload[field] = value
    save_torch_atomic(checkpoint, payload)
    manifest = json.loads((internal / "internal_manifest.json").read_text(encoding="utf-8"))
    manifest["evidence_sha256"][checkpoint.relative_to(internal).as_posix()] = sha256_file(checkpoint)
    write_json_atomic(internal / "internal_manifest.json", manifest)
    completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    completion["evidence_sha256"].update({checkpoint.relative_to(directory).as_posix(): sha256_file(checkpoint),
        "internal/internal_manifest.json": sha256_file(internal / "internal_manifest.json")})
    write_json_atomic(directory / "completion.json", completion)
    assert not audit_reference(tmp_path, completed_reference)["qualified"]


def test_incomplete_v410_reference_fails_closed(tmp_path):
    write_json_atomic(tmp_path / "identity.json", {"protocol": "statsfusion-joint-event-v1",
                      "data": {}, "source": {}, "config_sha256": "hash"})
    result = audit_reference(tmp_path, SimpleNamespace(identity={}, subject_folds={"subject": 0}))
    assert not result["qualified"]
    assert not result["comparison_performed"]
