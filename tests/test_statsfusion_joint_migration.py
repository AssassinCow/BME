from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bme_eating import joint_performance_migration as performance
from bme_eating.config import load_config
from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_protocol import config_identity, digest, initialize_run, source_identity
from bme_eating.joint_resume_migration import (
    ACTIVE,
    BACKUP,
    IDENTITY_BACKUP,
    IDENTITY_RECEIPT,
    IDENTITY_REPAIR,
    IDENTITY_SPEC,
    MIGRATION,
    RECEIPT,
    SPEC,
    active_resume_identity,
    authorize_checkpoint_migration,
    migrate_clipping_run,
    migration_evidence,
    source_changes,
)


@pytest.fixture
def migration_case(tmp_path, monkeypatch):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v410_joint_event.yaml")
    config["training"]["device"] = "cpu"
    root = tmp_path / "source"
    script = root / "scripts" / "run_statsfusion_joint.py"
    script.parent.mkdir(parents=True)
    script.write_text("old source", encoding="utf-8")
    monkeypatch.setattr("bme_eating.joint_protocol.runtime_identity", lambda: {"python": "test"})
    output = tmp_path / "output"
    data = {"data": "hash"}
    run = initialize_run(config, root, output, "failed_run", data,
                         fresh=True, skip_s0=True, historical_reference=None)
    original = json.loads((run / "identity.json").read_text(encoding="utf-8"))
    status = {"status": "FAILED", "current_stage": "outer", "current_fold": 0,
              "stages": {"audit": "COMPLETE", "validate": "COMPLETE", "outer": "FAILED"},
              "error": {"type": "RuntimeError", "message": "Joint clipping abort gate failed: 1.0000"}}
    write_json_atomic(run / "execution_status.json", status)
    cursor = {"epoch": 8, "next_chunk": 29, "session_count": 3, "optimizer_steps": 0, "clipped_steps": 0}
    checkpoint = run / "fold_0/internal/partition_0/selection/resume.pt"
    checkpoint.parent.mkdir(parents=True)
    payload = {"source_sha256": digest(original["source"]), "config_sha256": config_identity(config),
               "data_sha256": digest(data), "identity": "model_identity", "training_subjects": ["fit"],
               "model": {"weight": torch.ones(1)}, "gradients": {"weight": torch.tensor([7.73])},
               "optimizer": {"state": "unchanged"}, "torch_rng": torch.get_rng_state(),
               "state": {**cursor, "epoch_complete": False, "training_complete": False}}
    torch.save(payload, checkpoint)
    script.write_text("fixed clipping source", encoding="utf-8")
    target = {**original, "source": source_identity(root)}
    spec = {"migration": MIGRATION, "run_name": run.name,
            "original_identity_sha256": sha256_file(run / "identity.json"),
            "previous_source_sha256": digest(original["source"]),
            "current_source_sha256": digest(target["source"]),
            "checkpoint": checkpoint.relative_to(run).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint), "resume_cursor": cursor}
    write_json_atomic(root / "configs" / SPEC, spec)
    return config, root, output, data, run, target, checkpoint, payload, spec


def test_registered_migration_preserves_original_identity_checkpoint_and_gradients(migration_case):
    config, root, output, data, run, target, checkpoint, payload, spec = migration_case
    initialize_run(config, root, output, run.name, data, fresh=False, skip_s0=True,
                   historical_reference=None, migrate_clipping_gate=True)
    assert sha256_file(run / "identity.json") == spec["original_identity_sha256"]
    assert sha256_file(checkpoint) == spec["checkpoint_sha256"]
    assert sha256_file(run / BACKUP / spec["checkpoint"]) == spec["checkpoint_sha256"]
    restored = torch.load(checkpoint, weights_only=False)
    torch.testing.assert_close(restored["gradients"]["weight"], payload["gradients"]["weight"], rtol=0, atol=0)
    assert restored["state"] == payload["state"] and restored["optimizer"] == payload["optimizer"]
    torch.testing.assert_close(restored["torch_rng"], payload["torch_rng"], rtol=0, atol=0)
    assert active_resume_identity(run) == target
    evidence = authorize_checkpoint_migration(checkpoint.parent, checkpoint, payload, target["source"])
    assert evidence["resume_cursor"] == spec["resume_cursor"]
    assert evidence["receipt_sha256"] == sha256_file(run / RECEIPT)
    status = json.loads((run / "execution_status.json").read_text(encoding="utf-8"))
    assert status["status"] == status["stages"]["outer"] == status["stages"]["validate"] == "PAUSED"
    assert initialize_run(config, root, output, run.name, data, fresh=False, skip_s0=True,
                          historical_reference=None) == run
    assert set(migration_evidence(run)) == {RECEIPT, ACTIVE}


@pytest.mark.parametrize("change", ["source", "model", "data", "runtime", "checkpoint", "completed"])
def test_migration_refuses_changes_outside_registered_repair(migration_case, change):
    config, root, output, data, run, target, checkpoint, payload, spec = migration_case
    if change == "source":
        (root / "scripts/run_statsfusion_joint.py").write_text("unregistered edit", encoding="utf-8")
    elif change == "model":
        config["model"]["dropout"] = .2
    elif change == "data":
        data["data"] = "different split"
    elif change == "runtime":
        target["runtime"] = {"python": "different"}
    elif change == "checkpoint":
        payload["model"]["weight"] *= 2
        torch.save(payload, checkpoint)
    else:
        write_json_atomic(run / "fold_0/completion.json", {})
    with pytest.raises(RuntimeError):
        if change == "runtime":
            migrate_clipping_run(run, target, root)
        else:
            initialize_run(config, root, output, run.name, data, fresh=False, skip_s0=True,
                           historical_reference=None, migrate_clipping_gate=True)
    assert not (run / RECEIPT).exists()
    assert sha256_file(run / "identity.json") == spec["original_identity_sha256"]


@pytest.mark.parametrize("change", ["active", "backup", "checkpoint", "source", "other_checkpoint"])
def test_migrated_run_still_fails_closed_on_tampering(migration_case, change):
    _config, root, _output, _data, run, target, checkpoint, payload, spec = migration_case
    migrate_clipping_run(run, target, root)
    if change == "active":
        changed = copy.deepcopy(target)
        changed["source"] = {}
        write_json_atomic(run / ACTIVE, changed)
    elif change == "backup":
        (run / BACKUP / spec["checkpoint"]).write_bytes(b"tampered backup")
    elif change == "checkpoint":
        torch.save({**payload, "identity": "different"}, checkpoint)
    elif change == "source":
        target = {**target, "source": {}}
    else:
        alternate = checkpoint.with_name("other.pt")
        alternate.write_bytes(checkpoint.read_bytes())
        checkpoint = alternate
    with pytest.raises(RuntimeError):
        authorize_checkpoint_migration(checkpoint.parent, checkpoint, payload, target["source"])


def test_migration_commit_recovery_and_idempotence(migration_case):
    config, root, output, data, run, target, _checkpoint, _payload, _spec = migration_case
    migrate_clipping_run(run, target, root)
    receipt_hash = sha256_file(run / RECEIPT)
    (run / ACTIVE).unlink()
    with pytest.raises(RuntimeError, match="commit is incomplete"):
        active_resume_identity(run)
    original_status = json.loads((run / BACKUP / "execution_status.json").read_text(encoding="utf-8"))
    write_json_atomic(run / "execution_status.json", original_status)
    migrate_clipping_run(run, target, root)
    migrate_clipping_run(run, target, root)
    assert sha256_file(run / RECEIPT) == receipt_hash
    assert active_resume_identity(run) == target
    assert json.loads((run / "execution_status.json").read_text(encoding="utf-8"))["status"] == "PAUSED"
    changed = copy.deepcopy(config)
    changed["training"]["loader_prefetch_factor"] = 2
    assert initialize_run(changed, root, output, run.name, data, fresh=False, skip_s0=True,
                          historical_reference=None) == run


def test_missing_registration_and_unauthorized_old_checkpoint_are_rejected(migration_case):
    _config, root, _output, _data, run, target, checkpoint, payload, _spec = migration_case
    (root / "configs" / SPEC).unlink()
    with pytest.raises(RuntimeError, match="not registered"):
        migrate_clipping_run(run, target, root)
    with pytest.raises(RuntimeError, match="no authorized migration"):
        authorize_checkpoint_migration(checkpoint.parent, checkpoint, payload, target["source"])


@pytest.fixture
def identity_repair_case(migration_case):
    config, root, output, data, run, previous, checkpoint, payload, spec = migration_case
    migrate_clipping_run(run, previous, root)
    parent_receipt = (run / RECEIPT).read_bytes()
    protocol = root / "src/bme_eating/joint_protocol.py"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("git metadata comparison repair", encoding="utf-8")
    current = {**previous, "source": source_identity(root), "git_head": "new_commit"}
    registration = {"migration": IDENTITY_REPAIR, "run_name": run.name,
                    "previous_migration_receipt_sha256": sha256_file(run / RECEIPT),
                    "previous_source_sha256": digest(previous["source"]),
                    "current_source_sha256": digest(current["source"]),
                    "source_changes": source_changes(previous["source"], current["source"])}
    write_json_atomic(root / "configs" / IDENTITY_SPEC, registration)
    return config, root, output, data, run, current, checkpoint, payload, spec, parent_receipt


def test_identity_repair_chain_preserves_checkpoint_and_parent_receipt(identity_repair_case, monkeypatch):
    config, root, output, data, run, current, checkpoint, payload, spec, parent = identity_repair_case
    monkeypatch.setattr("bme_eating.joint_protocol.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=current["git_head"]))
    for _attempt in range(2):
        assert initialize_run(config, root, output, run.name, data, fresh=False, skip_s0=True,
                              historical_reference=None) == run
    assert active_resume_identity(run) == current
    assert (run / RECEIPT).read_bytes() == parent
    assert sha256_file(checkpoint) == spec["checkpoint_sha256"]
    assert sha256_file(run / "identity.json") == spec["original_identity_sha256"]
    assert (run / IDENTITY_BACKUP / RECEIPT).read_bytes() == parent
    evidence = authorize_checkpoint_migration(checkpoint.parent, checkpoint, payload, current["source"])
    assert evidence["identity_repair_receipt_sha256"] == sha256_file(run / IDENTITY_RECEIPT)
    assert evidence["resume_cursor"] == spec["resume_cursor"]
    assert IDENTITY_RECEIPT in migration_evidence(run)
    status = json.loads((run / "execution_status.json").read_text(encoding="utf-8"))
    assert status["stages"]["validate"] == "PAUSED"
    repair_hash = sha256_file(run / IDENTITY_RECEIPT)
    status["stages"]["validate"] = "COMPLETE"
    write_json_atomic(run / "execution_status.json", status)
    initialize_run(config, root, output, run.name, data, fresh=False, skip_s0=True, historical_reference=None)
    assert sha256_file(run / IDENTITY_RECEIPT) == repair_hash
    assert json.loads((run / "execution_status.json").read_text(encoding="utf-8"))["stages"]["validate"] == "COMPLETE"
    status.pop("resume_identity_repair")
    write_json_atomic(run / "execution_status.json", status)
    initialize_run(config, root, output, run.name, data, fresh=False, skip_s0=True, historical_reference=None)
    assert json.loads((run / "execution_status.json").read_text(encoding="utf-8"))["stages"]["validate"] == "PAUSED"
    monkeypatch.setattr("bme_eating.joint_protocol.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="another_commit_same_bytes"))
    initialize_run(config, root, output, run.name, data, fresh=False, skip_s0=True, historical_reference=None)
    assert sha256_file(run / IDENTITY_RECEIPT) == repair_hash
    assert active_resume_identity(run) == current


@pytest.mark.parametrize("change", ["source", "runtime", "data"])
def test_identity_repaired_run_still_rejects_later_changes(identity_repair_case, monkeypatch, change):
    config, root, output, data, run, _current, _checkpoint, _payload, _spec, _parent = identity_repair_case
    initialize_run(config, root, output, run.name, data, fresh=False, skip_s0=True, historical_reference=None)
    if change == "source":
        (root / "src/bme_eating/joint_protocol.py").write_text("later unregistered edit", encoding="utf-8")
    elif change == "runtime":
        monkeypatch.setattr("bme_eating.joint_protocol.runtime_identity", lambda: {"python": "different"})
    else:
        data["data"] = "different"
    with pytest.raises(RuntimeError, match="identity changed"):
        initialize_run(config, root, output, run.name, data, fresh=False, skip_s0=True, historical_reference=None)


@pytest.mark.parametrize("change", ["source", "model", "runtime", "data", "registration"])
def test_identity_repair_does_not_allow_unregistered_changes(identity_repair_case, monkeypatch, change):
    config, root, output, data, run, _current, _checkpoint, _payload, _spec, _parent = identity_repair_case
    if change == "source":
        (root / "scripts/run_statsfusion_joint.py").write_text("unregistered", encoding="utf-8")
    elif change == "model":
        config["model"]["dropout"] = .2
    elif change == "runtime":
        monkeypatch.setattr("bme_eating.joint_protocol.runtime_identity", lambda: {"python": "different"})
    elif change == "data":
        data["data"] = "different"
    else:
        (root / "configs" / IDENTITY_SPEC).unlink()
    with pytest.raises(RuntimeError, match="identity changed"):
        initialize_run(config, root, output, run.name, data, fresh=False, skip_s0=True, historical_reference=None)
    assert not (run / IDENTITY_RECEIPT).exists()


@pytest.mark.parametrize("change", ["backup", "registration", "receipt"])
def test_identity_repair_chain_detects_tampered_evidence(identity_repair_case, change):
    config, root, output, data, run, _current, _checkpoint, _payload, _spec, _parent = identity_repair_case
    initialize_run(config, root, output, run.name, data, fresh=False, skip_s0=True, historical_reference=None)
    if change == "backup":
        (run / IDENTITY_BACKUP / ACTIVE).write_bytes(b"changed")
    elif change == "registration":
        (run / IDENTITY_BACKUP / IDENTITY_SPEC).write_bytes(b"changed")
    else:
        receipt = json.loads((run / IDENTITY_RECEIPT).read_text(encoding="utf-8"))
        receipt["target_identity"]["runtime"] = {"python": "different"}
        write_json_atomic(run / IDENTITY_RECEIPT, receipt)
    with pytest.raises(RuntimeError, match="Resume identity repair"):
        active_resume_identity(run)


@pytest.fixture
def performance_case(identity_repair_case):
    config, root, output, data, run, _current, checkpoint, payload, _spec, _parent = identity_repair_case
    initialize_run(config, root, output, run.name, data, fresh=False, skip_s0=True, historical_reference=None)
    previous = active_resume_identity(run)
    payload = copy.deepcopy(payload)
    payload["source_sha256"] = digest(previous["source"])
    torch.save(payload, checkpoint)
    loader = root / "src/bme_eating/data/joint_sessions.py"
    loader.parent.mkdir(parents=True)
    loader.write_text("equivalent persistent worker performance", encoding="utf-8")
    evidence = "performance_validation.json"
    write_json_atomic(run / evidence, {"passed": True})
    spec = {"migration": performance.MIGRATION, "run_name": run.name,
            "previous_receipt_sha256": sha256_file(run / IDENTITY_RECEIPT),
            "previous_source_sha256": digest(previous["source"]),
            "current_source_sha256": digest(source_identity(root)),
            "source_changes": source_changes(previous["source"], source_identity(root)),
            "checkpoint": checkpoint.relative_to(run).as_posix(), "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_model_identity": payload["identity"],
            "training_subjects_sha256": digest(payload.get("training_subjects")),
            "selector_subjects_sha256": digest(payload.get("selector_subjects")),
            "excluded_subjects_sha256": digest(payload.get("excluded_subjects")),
            "resume_cursor": {key: payload["state"][key] for key in
                ("epoch", "next_chunk", "session_count", "optimizer_steps", "clipped_steps")},
            "equivalence_checks": {key: True for key in performance.EQUIVALENCE_CHECKS},
            "validation_evidence": {evidence: sha256_file(run / evidence)}}
    write_json_atomic(root / "configs" / performance.SPEC, spec)
    return config, root, output, data, run, checkpoint, payload, spec


def test_performance_migration_keeps_checkpoint_and_parent_chain(performance_case):
    config, root, output, data, run, checkpoint, payload, spec = performance_case
    parent_hash = sha256_file(run / IDENTITY_RECEIPT)
    original_hash = sha256_file(run / "identity.json")
    for _attempt in range(2):
        assert initialize_run(config, root, output, run.name, data, fresh=False, skip_s0=True,
                              historical_reference=None) == run
    assert sha256_file(checkpoint) == spec["checkpoint_sha256"]
    assert sha256_file(run / IDENTITY_RECEIPT) == parent_hash
    assert sha256_file(run / "identity.json") == original_hash
    assert active_resume_identity(run)["source"] == source_identity(root)
    evidence = authorize_checkpoint_migration(checkpoint.parent, checkpoint, payload, source_identity(root))
    assert evidence["migration"] == performance.MIGRATION
    assert evidence["resume_cursor"] == spec["resume_cursor"]
    assert performance.RECEIPT in migration_evidence(run)


@pytest.mark.parametrize("change", ["source", "config", "runtime", "checkpoint", "evidence"])
def test_performance_migration_rejects_unregistered_changes(performance_case, monkeypatch, change):
    config, root, output, data, run, checkpoint, _payload, _spec = performance_case
    if change == "source":
        (root / "scripts/run_statsfusion_joint.py").write_text("changed", encoding="utf-8")
    elif change == "config":
        config["loss"]["state"] = 2.
    elif change == "runtime":
        monkeypatch.setattr("bme_eating.joint_protocol.runtime_identity", lambda: {"python": "different"})
    elif change == "checkpoint":
        checkpoint.write_bytes(b"tampered")
    else:
        (run / "performance_validation.json").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError):
        initialize_run(config, root, output, run.name, data, fresh=False, skip_s0=True, historical_reference=None)
    assert not (run / performance.RECEIPT).exists()


def test_performance_migration_requires_validated_equivalence(performance_case):
    config, root, output, data, run, _checkpoint, _payload, spec = performance_case
    spec["equivalence_checks"]["loss_gradients_weights_exact"] = False
    write_json_atomic(root / "configs" / performance.SPEC, spec)
    with pytest.raises(RuntimeError, match="complete equivalence evidence"):
        initialize_run(config, root, output, run.name, data, fresh=False, skip_s0=True, historical_reference=None)
    assert not (run / performance.RECEIPT).exists()


def test_performance_migration_detects_modified_backup_and_later_source(performance_case):
    config, root, output, data, run, checkpoint, payload, _spec = performance_case
    initialize_run(config, root, output, run.name, data, fresh=False, skip_s0=True, historical_reference=None)
    source = source_identity(root)
    (root / "src/bme_eating/data/joint_sessions.py").write_text("later change", encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity changed"):
        initialize_run(config, root, output, run.name, data, fresh=False, skip_s0=True, historical_reference=None)
    (run / performance.BACKUP / "performance_validation.json").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Performance migration backup"):
        authorize_checkpoint_migration(checkpoint.parent, checkpoint, payload, source)
