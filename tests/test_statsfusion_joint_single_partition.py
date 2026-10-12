from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from bme_eating.config import load_config
from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_partition_migration import (
    MIGRATION,
    migrate_single_partition,
    state_fingerprint,
    verify_training_unchanged,
)
from bme_eating.joint_protocol import (
    calibration_scope,
    config_identity,
    digest,
    initialize_run,
    internal_partitions,
    model_run_identity,
    save_torch_atomic,
    source_identity,
    validate_config,
    verify_calibration_lineage,
    verify_oof_lineage,
)
from bme_eating.training.hierarchical_v4_trainer import stacking_partitions

ROOT = Path(__file__).resolve().parents[1]


def single_config():
    return load_config(ROOT / "configs/statsfusion_v411_fold0_single_partition.yaml")


def lineage_for(subjects, outer, config):
    assignment = stacking_partitions(subjects, 3, 2026)
    prediction = {subject for subject, partition in assignment.items() if partition == 0}
    train = subjects - prediction
    selector = {max(train)}
    return {"partition": 0, "model_identity": "partition0", "training_subjects": sorted(train),
            "prediction_subjects": sorted(prediction), "outer_excluded_subjects": sorted(outer),
            "source_training_subjects": sorted(train), "normalization_subjects": sorted(train),
            "selector_fit_subjects": sorted(train - selector), "selector_subjects": sorted(selector),
            "checkpoint_qualified": True}


def test_single_partition_keeps_existing_split_and_cannot_claim_complete_oof():
    config = single_config()
    validate_config(config)
    assert internal_partitions(config) == [0]
    assert config["training"]["partitions"] == 3
    assert not calibration_scope(config)["full_internal_oof"]
    subjects, outer = {f"subject_{index}" for index in range(9)}, {"outer"}
    lineage = lineage_for(subjects, outer, config)
    verify_calibration_lineage([lineage], subjects, outer, config)
    with pytest.raises(RuntimeError, match="three complete"):
        verify_oof_lineage([lineage], subjects, outer)
    lineage["source_training_subjects"].append("outer")
    with pytest.raises(RuntimeError, match="leakage"):
        verify_calibration_lineage([lineage], subjects, outer, config)
    config["execution"]["mode"] = "fivefold"
    with pytest.raises(ValueError, match="restricted"):
        validate_config(config)


def test_single_partition_does_not_calibrate_on_other_subjects(monkeypatch, tmp_path):
    from bme_eating.training import statsfusion_joint_trainer as trainer

    config = single_config()
    subjects, outer = {f"subject_{index}" for index in range(9)}, {"outer"}
    lineage = lineage_for(subjects, outer, config)
    prediction = set(lineage["prediction_subjects"])
    train = subjects - prediction
    selector = {max(train)}
    calls, calibration = [], []
    model = SimpleNamespace()
    mapping = SimpleNamespace(training_subjects=sorted(train))
    identities = {"config_sha256": "config", "source_sha256": "source", "data_sha256": "data"}
    inputs = SimpleNamespace(identity={"data": "test"}, frames=lambda cohort, labels:
        (pd.DataFrame(), pd.DataFrame(), pd.DataFrame({"subject_key": sorted(cohort),
         "session_id": "session", "start_ms": 0, "end_ms": 60000,
         "valid_duration": True, "evaluable": True}), pd.DataFrame()))
    monkeypatch.setattr(trainer, "_selector_split", lambda *args, **kwargs: (train - selector, selector, {}))

    def training(inputs, cohort, config, directory, identity, **kwargs):
        calls.append((directory.name, set(cohort), kwargs))
        if directory.name == "selection":
            write_json_atomic(directory / "training_history.json", [{"epoch": 38,
                              "selector": {"promotion_eligible": True}}])
        return 38

    monkeypatch.setattr(trainer, "train_joint_model", training)
    monkeypatch.setattr(trainer, "load_trained", lambda *args: (model,
        {"boundary": {}, "preprocessing": {"training_subjects": sorted(train)}, **identities}))
    monkeypatch.setattr(trainer, "fit_source_mapping", lambda *args: mapping)
    monkeypatch.setattr(trainer, "predict_joint_model", lambda *args, **kwargs:
        {"windows": pd.DataFrame({"subject_key": sorted(prediction)}), "coarse": {"setting": []}})

    def choose(coarse, windows, truth, ignore, mappings, config):
        calibration.append((set(truth.subject_key), set(windows.subject_key)))
        return {"setting": "setting", "budget": 20, "acceptance": .5, "nms": .3,
                "calibration": {"slope": 1, "intercept": 0, "qualified": True},
                "promotion_eligible": True}, []

    monkeypatch.setattr(trainer, "choose_operating_point", choose)
    monkeypatch.setattr(trainer, "apply_online_budget", lambda *args: ([], {}))
    monkeypatch.setattr(trainer, "commit_candidates", lambda *args: [])
    monkeypatch.setattr("bme_eating.joint_v411_boundary.lock_boundary_modes",
        lambda point, *args: ({**point, "boundary_enabled": False, "boundary_entropy": None}, [], []))
    monkeypatch.setattr(trainer, "protocol_identities", lambda *args: identities)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    epoch, point = trainer.train_partition_protocol(inputs, subjects, outer, config, tmp_path, "run", True)
    assert epoch == 38 and [name for name, _, _ in calls] == ["selection", "retrain"]
    assert calls[1][1] == train and calls[1][2]["resume"]
    assert calibration == [(prediction, prediction)]
    assert point["calibration_subjects"] == sorted(prediction)
    assert not point["full_internal_oof"] and len(point["lineage"]) == 1
    assert all(call[2]["outer_excluded"] == outer | prediction for call in calls)


def prepare_migration(tmp_path, monkeypatch):
    monkeypatch.setattr("bme_eating.joint_protocol.runtime_identity", lambda: {"runtime": "same"})
    monkeypatch.setattr("bme_eating.joint_partition_migration.runtime_identity", lambda: {"runtime": "same"})
    root = tmp_path / "source"
    trainer = root / "src/bme_eating/training/statsfusion_joint_trainer.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("def train_joint_model():\n    return 1\ndef train_partition_protocol():\n    return 3\n")
    config = single_config()
    config["execution"].pop("internal_mode")
    run = initialize_run(config, root, tmp_path / "outputs", "paused", {"data": "same"},
                         fresh=True, skip_s0=True, historical_reference=None)
    write_json_atomic(run / "execution_status.json", {"status": "PAUSED", "stages": {"validate": "COMPLETE"}})
    payload = {"identity": "original_partition", "requested_epochs": 38, "training_subjects": ["fit"],
        "selector_subjects": [], "excluded_subjects": ["holdout", "outer"],
        "config_sha256": config_identity(config), "source_sha256": digest(source_identity(root)),
        "data_sha256": digest({"data": "same"}), "model": {"weight": torch.tensor([1.25])},
        "optimizer": {"state": {0: {"step": torch.tensor(21), "exp_avg": torch.tensor([.2])}}},
        "gradients": {"weight": torch.tensor([.3])}, "torch_rng": torch.get_rng_state(),
        "python_rng": (1, 2), "numpy_rng": (np.arange(4),), "cuda_rng": [],
        "state": {"epoch": 21, "next_chunk": 2075, "session_count": 3,
            "positive_gradients": {"event": {"gradients": {"weight": torch.tensor([.4])}}},
            "journal_offsets": {"sampling": 234}, "cache_rows": 17}}
    for directory in ("selection", "retrain"):
        save_torch_atomic(run / f"fold_0/internal/partition_0/{directory}/resume.pt", payload)
    trainer.write_text("def train_joint_model():\n    return 1\ndef train_partition_protocol():\n    return 1\n")
    return run, root, payload


def test_migration_preserves_resume_state_is_idempotent_and_resumes(tmp_path, monkeypatch):
    run, root, payload = prepare_migration(tmp_path, monkeypatch)
    identity_before = sha256_file(run / "identity.json")
    receipt = migrate_single_partition(run, single_config(), root, {"data": "same"})
    assert receipt["status"] == "COMPLETE" and receipt["migration"] == MIGRATION
    assert model_run_identity(run) == identity_before
    assert sha256_file(run / "identity.json") != identity_before
    for directory in ("selection", "retrain"):
        restored = torch.load(run / f"fold_0/internal/partition_0/{directory}/resume.pt",
                              map_location="cpu", weights_only=False)
        assert state_fingerprint(restored) == state_fingerprint(payload)
        assert restored["state"]["next_chunk"] == 2075
        assert restored["config_sha256"] == config_identity(single_config())
    assert migrate_single_partition(run, single_config(), root, {"data": "same"}) == receipt
    assert initialize_run(single_config(), root, tmp_path / "outputs", "paused", {"data": "same"},
                          fresh=False, skip_s0=True, historical_reference=None) == run


def test_migration_rejects_unrelated_training_changes(tmp_path, monkeypatch):
    run, root, _ = prepare_migration(tmp_path, monkeypatch)
    path = root / "src/bme_eating/training/statsfusion_joint_trainer.py"
    path.write_text(path.read_text().replace("return 1", "return 2", 1))
    with pytest.raises(RuntimeError, match="training or prediction"):
        migrate_single_partition(run, single_config(), root, {"data": "same"})
    changed = copy.deepcopy(single_config())
    changed["training"]["head_learning_rate"] *= 2
    with pytest.raises(RuntimeError, match="only change"):
        migrate_single_partition(run, changed, root, {"data": "same"})


def test_prepared_migration_recovers_interrupted_checkpoint_commit(tmp_path, monkeypatch):
    run, root, payload = prepare_migration(tmp_path, monkeypatch)
    from bme_eating import joint_partition_migration as migration
    original = migration.save_torch_atomic
    calls = 0

    def interrupted(path, checkpoint):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("interrupted")
        original(path, checkpoint)

    monkeypatch.setattr(migration, "save_torch_atomic", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        migrate_single_partition(run, single_config(), root, {"data": "same"})
    assert json.loads((run / "single_partition_migration.json").read_text())["status"] == "PREPARED"
    monkeypatch.setattr(migration, "save_torch_atomic", original)
    assert migrate_single_partition(run, single_config(), root, {"data": "same"})["status"] == "COMPLETE"
    restored = torch.load(run / "fold_0/internal/partition_0/retrain/resume.pt",
                          map_location="cpu", weights_only=False)
    assert state_fingerprint(restored) == state_fingerprint(payload)


def test_training_semantics_check_only_allows_orchestration_changes():
    previous = "def train_joint_model():\n    return 1\ndef train_partition_protocol():\n    return 3\n"
    verify_training_unchanged(previous, previous.replace("return 3", "return 1"))
    with pytest.raises(RuntimeError, match="training or prediction"):
        verify_training_unchanged(previous, previous.replace("return 1", "return 2"))
