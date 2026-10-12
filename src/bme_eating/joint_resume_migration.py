from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_protocol import digest, resume_identity_differences, run_lock, source_identity

MIGRATION = "clipping_epoch_gate_v1"
SPEC = "statsfusion_v410_clipping_resume_migration.json"
BACKUP = Path("resume_migrations") / MIGRATION
RECEIPT = "resume_migration.json"
ACTIVE = "active_identity.json"
IDENTITY_REPAIR = "resume_identity_comparison_v1"
IDENTITY_SPEC = "statsfusion_v410_resume_identity_migration.json"
IDENTITY_RECEIPT = "resume_identity_migration.json"
IDENTITY_BACKUP = Path("resume_migrations") / IDENTITY_REPAIR
IDENTITY_SOURCE_PATHS = {"src/bme_eating/joint_protocol.py", "src/bme_eating/joint_resume_migration.py"}


def repair_invariants(identity):
    return {key: value for key, value in identity.items() if key not in {"source", "git_head"}}


def source_changes(previous, current):
    return {name: {"before": previous.get(name), "after": current.get(name)}
            for name in sorted(previous.keys() | current.keys()) if previous.get(name) != current.get(name)}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def record_migration_status(run, receipt):
    status = read_json(run / "execution_status.json")
    receipt_hash = sha256_file(run / RECEIPT)
    if status.get("resume_migration", {}).get("receipt_sha256") == receipt_hash:
        return
    status["status"] = status["stages"]["outer"] = "PAUSED"
    status["stages"]["validate"] = "PAUSED"
    status.setdefault("outer_folds", {})[str(status["current_fold"])] = "PAUSED"
    status["resume_migration"] = {"migration": MIGRATION, "receipt_sha256": receipt_hash,
                                  "resume_cursor": receipt["resume_cursor"]}
    status.pop("error", None)
    write_json_atomic(run / "execution_status.json", status)


def clipping_resume_identity(run):
    if not (run / RECEIPT).exists():
        if (run / ACTIVE).exists():
            raise RuntimeError("Active resume identity lacks a migration receipt")
        return read_json(run / "identity.json")
    receipt = read_json(run / RECEIPT)
    registration = run / BACKUP / SPEC
    if (receipt.get("migration") != MIGRATION or not registration.is_file()
            or sha256_file(registration) != receipt.get("registration_sha256")):
        raise RuntimeError("Resume migration registration is missing or changed")
    spec = read_json(registration)
    if (spec.get("run_name") != run.name
            or sha256_file(run / "identity.json") != spec.get("original_identity_sha256")
            or receipt.get("original_identity_sha256") != spec.get("original_identity_sha256")):
        raise RuntimeError("Original run identity differs from registered resume migration")
    for relative, expected in receipt.get("backup_sha256", {}).items():
        target = (run / BACKUP / relative).resolve()
        if (not target.is_relative_to((run / BACKUP).resolve()) or not target.is_file()
                or sha256_file(target) != expected):
            raise RuntimeError("Resume migration backup is missing or changed")
    required = {"identity.json", "execution_status.json", spec["checkpoint"]}
    if not required.issubset(receipt.get("backup_sha256", {})):
        raise RuntimeError("Resume migration backup evidence is incomplete")
    original = read_json(run / "identity.json")
    target_identity = receipt.get("target_identity", {})
    if (repair_invariants(original) != repair_invariants(target_identity)
            or digest(original["source"]) != spec.get("previous_source_sha256")
            or digest(target_identity.get("source")) != spec.get("current_source_sha256")
            or receipt["backup_sha256"][spec["checkpoint"]] != spec["checkpoint_sha256"]):
        raise RuntimeError("Resume migration exceeds the registered clipping repair")
    if not (run / ACTIVE).is_file() or read_json(run / ACTIVE) != target_identity:
        raise RuntimeError("Resume migration commit is incomplete; use --migrate-clipping-gate")
    return target_identity


def identity_resume_identity(run):
    previous = clipping_resume_identity(run)
    if not (run / IDENTITY_RECEIPT).exists():
        return previous
    receipt = read_json(run / IDENTITY_RECEIPT)
    registration = run / IDENTITY_BACKUP / IDENTITY_SPEC
    if (receipt.get("migration") != IDENTITY_REPAIR or not registration.is_file()
            or sha256_file(registration) != receipt.get("registration_sha256")):
        raise RuntimeError("Resume identity repair registration is missing or changed")
    spec = read_json(registration)
    current = receipt.get("target_identity", {})
    changes = source_changes(previous["source"], current.get("source", {}))
    if (spec.get("migration") != IDENTITY_REPAIR or spec.get("run_name") != run.name
            or sha256_file(run / RECEIPT) != spec.get("previous_migration_receipt_sha256")
            or digest(previous) != receipt.get("previous_identity_sha256")
            or repair_invariants(previous) != repair_invariants(current)
            or digest(previous["source"]) != spec.get("previous_source_sha256")
            or digest(current.get("source")) != spec.get("current_source_sha256")
            or not changes or not set(changes).issubset(IDENTITY_SOURCE_PATHS)
            or changes != spec.get("source_changes")):
        raise RuntimeError("Resume identity repair exceeds its registered source changes")
    hashes = receipt.get("backup_sha256", {})
    if not {RECEIPT, ACTIVE, "execution_status.json"}.issubset(hashes):
        raise RuntimeError("Resume identity repair backup evidence is incomplete")
    for relative, expected in hashes.items():
        target = (run / IDENTITY_BACKUP / relative).resolve()
        if (not target.is_relative_to((run / IDENTITY_BACKUP).resolve()) or not target.is_file()
                or sha256_file(target) != expected):
            raise RuntimeError("Resume identity repair backup is missing or changed")
    if hashes[RECEIPT] != spec["previous_migration_receipt_sha256"]:
        raise RuntimeError("Resume identity repair parent receipt changed")
    return current


def active_resume_identity(run):
    from bme_eating.joint_performance_migration import performance_resume_identity
    return performance_resume_identity(run, identity_resume_identity(run))


def record_identity_repair_status(run):
    status = read_json(run / "execution_status.json")
    repair = {"migration": IDENTITY_REPAIR, "receipt_sha256": sha256_file(run / IDENTITY_RECEIPT)}
    if status.get("resume_identity_repair") == repair:
        return
    status["stages"]["validate"] = "PAUSED"
    status["resume_identity_repair"] = repair
    write_json_atomic(run / "execution_status.json", status)


def migrate_resume_identity(run, identity, source_root):
    spec_path = source_root / "configs" / IDENTITY_SPEC
    if (run / IDENTITY_RECEIPT).exists():
        with run_lock(run):
            active = active_resume_identity(run)
            if not resume_identity_differences(active, identity):
                record_identity_repair_status(run)
        return
    if not (run / RECEIPT).exists() or not spec_path.is_file():
        return
    spec = read_json(spec_path)
    if spec.get("run_name") != run.name:
        return
    with run_lock(run):
        if (run / IDENTITY_RECEIPT).exists():
            return
        previous = clipping_resume_identity(run)
        changes = source_changes(previous["source"], identity["source"])
        if (spec.get("migration") != IDENTITY_REPAIR
                or sha256_file(run / RECEIPT) != spec.get("previous_migration_receipt_sha256")
                or digest(previous["source"]) != spec.get("previous_source_sha256")
                or digest(identity["source"]) != spec.get("current_source_sha256")
                or identity["source"] != source_identity(source_root)
                or repair_invariants(previous) != repair_invariants(identity)
                or not changes or not set(changes).issubset(IDENTITY_SOURCE_PATHS)
                or changes != spec.get("source_changes")):
            return
        status = read_json(run / "execution_status.json")
        if (status.get("status") not in {"PAUSED", "FAILED"}
                or any((run / f"fold_{fold}/completion.json").exists() for fold in range(5))
                or any((run / name).exists() for name in ("final", "bundle", "joint_gate_manifest.json"))):
            raise RuntimeError("Resume identity repair requires the incomplete paused run")
        parent = read_json(run / RECEIPT)
        if sha256_file(run / parent["checkpoint"]) != parent["checkpoint_sha256"]:
            raise RuntimeError("Registered resume checkpoint changed before identity repair")
        hashes = {}
        for relative in (RECEIPT, ACTIVE, "execution_status.json", "implementation_validation.json"):
            source = run / relative
            if not source.is_file():
                continue
            destination = run / IDENTITY_BACKUP / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            expected = sha256_file(source)
            if destination.exists() and sha256_file(destination) != expected:
                raise RuntimeError("Existing identity repair backup differs; refuse overwrite")
            if not destination.exists():
                shutil.copyfile(source, destination)
            hashes[relative] = expected
        registration = run / IDENTITY_BACKUP / IDENTITY_SPEC
        if registration.exists() and sha256_file(registration) != sha256_file(spec_path):
            raise RuntimeError("Existing identity repair registration differs; refuse overwrite")
        shutil.copyfile(spec_path, registration)
        receipt = {"migration": IDENTITY_REPAIR,
                   "reason": "Git commit metadata blocked resume despite identical source bytes",
                   "authorization": "Explicit user request to resume the existing run",
                   "previous_identity_sha256": digest(previous),
                   "registration_sha256": sha256_file(registration), "backup_sha256": hashes,
                   "target_identity": identity}
        write_json_atomic(run / IDENTITY_RECEIPT, receipt)
        active_resume_identity(run)
        record_identity_repair_status(run)


def migrate_clipping_run(run, identity, source_root):
    with run_lock(run):
        if (run / RECEIPT).exists():
            receipt = read_json(run / RECEIPT)
            if not (run / ACTIVE).exists() and receipt.get("target_identity") == identity:
                write_json_atomic(run / ACTIVE, identity)
            if resume_identity_differences(active_resume_identity(run), identity):
                raise RuntimeError("Resume migration target identity changed")
            record_migration_status(run, receipt)
            return
        spec_path = source_root / "configs" / SPEC
        if not spec_path.is_file():
            raise RuntimeError("Clipping resume migration is not registered")
        spec = read_json(spec_path)
        original = read_json(run / "identity.json")
        if (spec.get("migration") != MIGRATION or spec.get("run_name") != run.name
                or sha256_file(run / "identity.json") != spec.get("original_identity_sha256")
                or digest(original["source"]) != spec.get("previous_source_sha256")
                or digest(identity["source"]) != spec.get("current_source_sha256")
                or identity["source"] != source_identity(source_root)
                or repair_invariants(original) != repair_invariants(identity)):
            raise RuntimeError("Source/config/data/runtime differs from registered clipping repair")
        status = read_json(run / "execution_status.json")
        if (status.get("status") != "FAILED" or status.get("current_stage") != "outer"
                or status.get("error", {}).get("message") != "Joint clipping abort gate failed: 1.0000"
                or any((run / f"fold_{fold}" / "completion.json").exists() for fold in range(5))
                or any((run / name).exists() for name in ("final", "bundle", "joint_gate_manifest.json"))):
            raise RuntimeError("Only the incomplete clipping-aborted run can be migrated")
        checkpoint = (run / spec["checkpoint"]).resolve()
        if (not checkpoint.is_relative_to(run.resolve()) or not checkpoint.is_file()
                or sha256_file(checkpoint) != spec["checkpoint_sha256"]):
            raise RuntimeError("Registered clipping resume checkpoint changed")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["state"]
        if (payload.get("source_sha256") != spec["previous_source_sha256"]
                or payload.get("config_sha256") != original["config_sha256"]
                or payload.get("data_sha256") != digest(original["data"])
                or state.get("epoch_complete") or state.get("training_complete")
                or any(state.get(key) != value for key, value in spec["resume_cursor"].items())
                or not all(torch.isfinite(value).all() for value in payload["model"].values())
                or any(value is not None and not torch.isfinite(value).all()
                       for value in payload["gradients"].values())):
            raise RuntimeError("Clipping resume checkpoint eligibility or cursor changed")
        files = ["identity.json", "resolved_config.json", "execution_status.json", "failure_report.json",
                 "implementation_validation.json", spec["checkpoint"],
                 str(Path(spec["checkpoint"]).with_name("latest.pt")).replace("\\", "/"),
                 str(Path(spec["checkpoint"]).with_name("training_history.json")).replace("\\", "/")]
        hashes = {}
        for relative in files:
            source = run / relative
            if not source.is_file():
                continue
            destination = run / BACKUP / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            expected = sha256_file(source)
            if destination.exists() and sha256_file(destination) != expected:
                raise RuntimeError("Existing migration backup differs; refuse overwrite")
            if not destination.exists():
                shutil.copyfile(source, destination)
            hashes[relative] = sha256_file(destination)
        registration = run / BACKUP / SPEC
        if registration.exists() and sha256_file(registration) != sha256_file(spec_path):
            raise RuntimeError("Existing migration registration differs; refuse overwrite")
        shutil.copyfile(spec_path, registration)
        receipt = {"migration": MIGRATION, "reason": "First-update clipping fraction used an incomplete denominator",
                   "authorization": "Explicit user request to resume the existing run after clipping repair",
                   "original_identity_sha256": spec["original_identity_sha256"],
                   "registration_sha256": sha256_file(registration), "backup_sha256": hashes,
                   "target_identity": identity, "checkpoint": spec["checkpoint"],
                   "checkpoint_sha256": spec["checkpoint_sha256"], "resume_cursor": spec["resume_cursor"]}
        write_json_atomic(run / RECEIPT, receipt)
        write_json_atomic(run / ACTIVE, identity)
        active_resume_identity(run)
        record_migration_status(run, receipt)


def authorize_checkpoint_migration(directory, checkpoint, payload, current_source):
    from bme_eating.joint_performance_migration import authorize_performance_checkpoint
    performance = authorize_performance_checkpoint(directory, checkpoint, payload, current_source)
    if performance is not None:
        return performance
    for run in (directory, *directory.parents):
        if not (run / RECEIPT).is_file():
            continue
        active = active_resume_identity(run)
        receipt = read_json(run / RECEIPT)
        original = read_json(run / "identity.json")
        if (current_source != active["source"] or payload.get("source_sha256") != digest(original["source"])
                or checkpoint.resolve() != (run / receipt["checkpoint"]).resolve()
                or sha256_file(checkpoint) != receipt["checkpoint_sha256"]
                or payload.get("config_sha256") != active["config_sha256"]
                or payload.get("data_sha256") != digest(active["data"])):
            raise RuntimeError("Checkpoint source migration evidence does not match")
        evidence = {"migration": MIGRATION, "receipt_sha256": sha256_file(run / RECEIPT),
                    "previous_source_sha256": payload["source_sha256"], "resume_cursor": receipt["resume_cursor"]}
        if (run / IDENTITY_RECEIPT).exists():
            evidence["identity_repair_receipt_sha256"] = sha256_file(run / IDENTITY_RECEIPT)
        return evidence
    raise RuntimeError("Training checkpoint source identity mismatch; no authorized migration")


def migration_evidence(run):
    if not (run / RECEIPT).exists():
        return {}
    active_resume_identity(run)
    evidence = {RECEIPT: sha256_file(run / RECEIPT), ACTIVE: sha256_file(run / ACTIVE)}
    if (run / IDENTITY_RECEIPT).exists():
        evidence[IDENTITY_RECEIPT] = sha256_file(run / IDENTITY_RECEIPT)
    from bme_eating.joint_performance_migration import RECEIPT as PERFORMANCE_RECEIPT
    if (run / PERFORMANCE_RECEIPT).exists():
        evidence[PERFORMANCE_RECEIPT] = sha256_file(run / PERFORMANCE_RECEIPT)
    return evidence
