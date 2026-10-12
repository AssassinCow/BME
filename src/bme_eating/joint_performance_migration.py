from __future__ import annotations

import shutil
from pathlib import Path

import torch

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_protocol import digest, resume_identity_differences, run_lock, source_identity
from bme_eating.joint_resume_migration import (
    IDENTITY_RECEIPT,
    identity_resume_identity,
    read_json,
    repair_invariants,
    source_changes,
)

MIGRATION = "numerically_equivalent_performance_v1"
SPEC = "statsfusion_v410_performance_resume_migration.json"
RECEIPT = "performance_resume_migration.json"
BACKUP = Path("resume_migrations") / MIGRATION
SOURCE_PATHS = {"src/bme_eating/joint_protocol.py", "src/bme_eating/joint_resume_migration.py",
                "src/bme_eating/joint_performance_migration.py", "src/bme_eating/data/joint_sessions.py",
                "src/bme_eating/training/statsfusion_joint_trainer.py"}
EQUIVALENCE_CHECKS = ("proposal_targets_exact", "loss_gradients_weights_exact", "persistent_loader_exact",
                      "resume_state_preserved", "gpu_memory_qualified", "checkpoint_snapshot_exact")


def performance_resume_identity(run, previous):
    if not (run / RECEIPT).exists():
        return previous
    receipt = read_json(run / RECEIPT)
    registration = run / BACKUP / SPEC
    if (receipt.get("migration") != MIGRATION or not registration.is_file()
            or sha256_file(registration) != receipt.get("registration_sha256")):
        raise RuntimeError("Performance migration registration is missing or changed")
    spec = read_json(registration)
    current = receipt.get("target_identity", {})
    changes = source_changes(previous["source"], current.get("source", {}))
    checks = spec.get("equivalence_checks", {})
    if (spec.get("migration") != MIGRATION or spec.get("run_name") != run.name
            or sha256_file(run / IDENTITY_RECEIPT) != spec.get("previous_receipt_sha256")
            or digest(previous) != receipt.get("previous_identity_sha256")
            or repair_invariants(previous) != repair_invariants(current)
            or digest(previous["source"]) != spec.get("previous_source_sha256")
            or digest(current.get("source")) != spec.get("current_source_sha256")
            or not changes or not set(changes).issubset(SOURCE_PATHS) or changes != spec.get("source_changes")
            or not all(checks.get(key) is True for key in EQUIVALENCE_CHECKS)):
        raise RuntimeError("Performance migration exceeds registered equivalent changes")
    hashes = receipt.get("backup_sha256", {})
    if not {IDENTITY_RECEIPT, "execution_status.json", spec["checkpoint"]}.issubset(hashes):
        raise RuntimeError("Performance migration backup evidence is incomplete")
    validation = spec.get("validation_evidence", {})
    if not validation or any(hashes.get(name) != expected for name, expected in validation.items()):
        raise RuntimeError("Performance migration validation backup evidence is incomplete")
    for relative, expected in hashes.items():
        target = (run / BACKUP / relative).resolve()
        if (not target.is_relative_to((run / BACKUP).resolve()) or not target.is_file()
                or sha256_file(target) != expected):
            raise RuntimeError("Performance migration backup is missing or changed")
    if (hashes[IDENTITY_RECEIPT] != spec["previous_receipt_sha256"]
            or hashes[spec["checkpoint"]] != spec["checkpoint_sha256"]):
        raise RuntimeError("Performance migration parent or checkpoint evidence changed")
    return current


def record_performance_status(run, receipt):
    status = read_json(run / "execution_status.json")
    evidence = {"migration": MIGRATION, "receipt_sha256": sha256_file(run / RECEIPT),
                "resume_cursor": receipt["resume_cursor"]}
    if status.get("performance_resume") == evidence:
        return
    status["stages"]["validate"] = "PAUSED"
    status["performance_resume"] = evidence
    write_json_atomic(run / "execution_status.json", status)


def migrate_performance_run(run, identity, source_root):
    if (run / RECEIPT).exists():
        with run_lock(run):
            active = performance_resume_identity(run, identity_resume_identity(run))
            if not resume_identity_differences(active, identity):
                record_performance_status(run, read_json(run / RECEIPT))
        return
    registration = source_root / "configs" / SPEC
    if not (run / IDENTITY_RECEIPT).exists() or not registration.is_file():
        return
    spec = read_json(registration)
    if spec.get("run_name") != run.name:
        return
    with run_lock(run):
        previous = identity_resume_identity(run)
        changes = source_changes(previous["source"], identity["source"])
        if (spec.get("migration") != MIGRATION
                or sha256_file(run / IDENTITY_RECEIPT) != spec.get("previous_receipt_sha256")
                or digest(previous["source"]) != spec.get("previous_source_sha256")
                or digest(identity["source"]) != spec.get("current_source_sha256")
                or identity["source"] != source_identity(source_root)
                or repair_invariants(previous) != repair_invariants(identity)
                or not changes or not set(changes).issubset(SOURCE_PATHS) or changes != spec.get("source_changes")):
            return
        if not all(spec.get("equivalence_checks", {}).get(key) is True for key in EQUIVALENCE_CHECKS):
            raise RuntimeError("Performance migration requires complete equivalence evidence")
        status = read_json(run / "execution_status.json")
        if (status.get("status") != "PAUSED"
                or any((run / f"fold_{fold}/completion.json").exists() for fold in range(5))
                or any((run / name).exists() for name in ("final", "bundle", "joint_gate_manifest.json"))):
            raise RuntimeError("Equivalent performance migration requires the paused incomplete run")
        checkpoint = (run / spec["checkpoint"]).resolve()
        if (not checkpoint.is_relative_to(run.resolve()) or not checkpoint.is_file()
                or sha256_file(checkpoint) != spec.get("checkpoint_sha256")):
            raise RuntimeError("Performance resume checkpoint changed")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["state"]
        if (payload.get("source_sha256") != spec["previous_source_sha256"]
                or payload.get("config_sha256") != previous["config_sha256"]
                or payload.get("data_sha256") != digest(previous["data"])
                or state.get("epoch_complete") or state.get("training_complete")
                or payload.get("identity") != spec.get("checkpoint_model_identity")
                or digest(payload.get("training_subjects")) != spec.get("training_subjects_sha256")
                or digest(payload.get("excluded_subjects")) != spec.get("excluded_subjects_sha256")
                or digest(payload.get("selector_subjects")) != spec.get("selector_subjects_sha256")
                or any(state.get(key) != value for key, value in spec["resume_cursor"].items())
                or not all(torch.isfinite(value).all() for value in payload["model"].values())
                or any(value is not None and not torch.isfinite(value).all()
                       for value in payload["gradients"].values())):
            raise RuntimeError("Performance checkpoint qualification or cursor changed")
        hashes = {}
        for relative in (IDENTITY_RECEIPT, "execution_status.json", "implementation_validation.json",
                         spec["checkpoint"]):
            source = run / relative
            if not source.is_file():
                continue
            destination = run / BACKUP / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            expected = sha256_file(source)
            if destination.exists() and sha256_file(destination) != expected:
                raise RuntimeError("Performance backup differs; refuse overwrite")
            if not destination.exists():
                shutil.copyfile(source, destination)
            hashes[relative] = expected
        evidence = spec.get("validation_evidence", {})
        if not evidence:
            raise RuntimeError("Performance validation evidence is missing")
        for relative, expected in evidence.items():
            source = (run / relative).resolve()
            if not source.is_relative_to(run.resolve()) or not source.is_file() or sha256_file(source) != expected:
                raise RuntimeError("Performance validation evidence changed")
            destination = run / BACKUP / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha256_file(destination) != expected:
                raise RuntimeError("Performance validation backup differs")
            if not destination.exists():
                shutil.copyfile(source, destination)
            hashes[relative] = expected
        destination = run / BACKUP / SPEC
        if destination.exists() and sha256_file(destination) != sha256_file(registration):
            raise RuntimeError("Performance registration differs; refuse overwrite")
        shutil.copyfile(registration, destination)
        receipt = {"migration": MIGRATION, "authorization": "User requested maximum equivalent acceleration",
                   "previous_identity_sha256": digest(previous), "target_identity": identity,
                   "registration_sha256": sha256_file(destination), "backup_sha256": hashes,
                   "checkpoint": spec["checkpoint"], "checkpoint_sha256": spec["checkpoint_sha256"],
                   "resume_cursor": spec["resume_cursor"]}
        write_json_atomic(run / RECEIPT, receipt)
        performance_resume_identity(run, previous)
        record_performance_status(run, receipt)


def authorize_performance_checkpoint(directory, checkpoint, payload, current_source):
    for run in (directory, *directory.parents):
        if not (run / RECEIPT).is_file():
            continue
        receipt = read_json(run / RECEIPT)
        previous = identity_resume_identity(run)
        active = performance_resume_identity(run, previous)
        if (current_source != active["source"] or payload.get("source_sha256") != digest(previous["source"])
                or checkpoint.resolve() != (run / receipt["checkpoint"]).resolve()
                or sha256_file(checkpoint) != receipt["checkpoint_sha256"]
                or payload.get("config_sha256") != active["config_sha256"]
                or payload.get("data_sha256") != digest(active["data"])):
            raise RuntimeError("Performance checkpoint migration evidence does not match")
        return {"migration": MIGRATION, "receipt_sha256": sha256_file(run / RECEIPT),
                "previous_source_sha256": payload["source_sha256"], "resume_cursor": receipt["resume_cursor"],
                "previous_migration": payload["state"].get("resume_migration")}
    return None
