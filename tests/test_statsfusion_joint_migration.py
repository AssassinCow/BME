from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from bme_eating.config import load_config
from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_protocol import config_identity, digest, initialize_run, source_identity
from bme_eating.joint_resume_migration import (
    ACTIVE,
    BACKUP,
    MIGRATION,
    RECEIPT,
    SPEC,
    active_resume_identity,
    authorize_checkpoint_migration,
    migrate_clipping_run,
    migration_evidence,
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
