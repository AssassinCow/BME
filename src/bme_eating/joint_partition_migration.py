from __future__ import annotations

import ast
import copy
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_protocol import (
    V411_PROTOCOL,
    config_identity,
    digest,
    internal_partitions,
    run_lock,
    runtime_identity,
    save_torch_atomic,
    source_identity,
    validate_config,
    verify_artifact_hashes,
)
from bme_eating.joint_resume_migration import source_changes

MIGRATION = "v411_single_partition_holdout_v1"
RECEIPT = "single_partition_migration.json"
BACKUP = Path("resume_migrations") / MIGRATION
ALLOWED_SOURCES = {
    "scripts/run_statsfusion_joint.py", "scripts/migrate_statsfusion_single_partition.py",
    "src/bme_eating/joint_protocol.py", "src/bme_eating/joint_partition_migration.py",
    "src/bme_eating/joint_single_fold.py", "src/bme_eating/joint_workflow.py",
    "src/bme_eating/training/statsfusion_joint_trainer.py",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def state_fingerprint(payload):
    def encode(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            return ("tensor", str(tensor.dtype), tuple(tensor.shape),
                    tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        if isinstance(value, np.ndarray):
            return ("array", str(value.dtype), value.shape, value.tobytes())
        if isinstance(value, dict):
            return ("dict", tuple(sorted(((encode(key), encode(item)) for key, item in value.items()),
                                         key=lambda item: pickle.dumps(item[0]))))
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, tuple(encode(item) for item in value))
        if isinstance(value, (set, frozenset)):
            return (type(value).__name__, tuple(sorted((encode(item) for item in value), key=pickle.dumps)))
        if hasattr(value, "__dict__"):
            return (type(value).__module__, type(value).__qualname__, encode(vars(value)))
        return value

    state = {key: value for key, value in payload.items() if key not in {"config_sha256", "source_sha256"}}
    return hashlib.sha256(pickle.dumps(encode(state), protocol=5)).hexdigest()


def verify_training_unchanged(previous, current):
    def body(source):
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
                     and not (isinstance(node, ast.FunctionDef) and node.name == "train_partition_protocol")]
        return ast.dump(tree, include_attributes=False)

    if body(previous) != body(current):
        raise RuntimeError("Single-partition migration changes model training or prediction semantics")


def verify_partition_migration(run):
    receipt = read_json(run / RECEIPT)
    if receipt.get("migration") != MIGRATION or receipt.get("status") != "COMPLETE":
        raise RuntimeError("Single-partition migration is incomplete; rerun its migration script")
    if receipt.get("target_identity_sha256") != sha256_file(run / "identity.json"):
        raise RuntimeError("Single-partition migration identity changed")
    required = {"identity.json", "resolved_config.json", "execution_status.json"}
    if not required.issubset(receipt.get("backup_sha256", {})):
        raise RuntimeError("Single-partition migration backup evidence is missing")
    verify_artifact_hashes(run / BACKUP, receipt["backup_sha256"])
    if sha256_file(run / BACKUP / "identity.json") != receipt["original_identity_sha256"]:
        raise RuntimeError("Single-partition model ancestry changed")
    current = read_json(run / "identity.json")
    previous = read_json(run / BACKUP / "identity.json")
    immutable = set(previous) - {"config_sha256", "source", "git_head"}
    if any(current.get(key) != previous[key] for key in immutable):
        raise RuntimeError("Single-partition migration data/runtime/reference identity changed")
    if config_identity(read_json(run / "resolved_config.json")) != current["config_sha256"]:
        raise RuntimeError("Single-partition migration configuration changed")
    return receipt


def migrate_single_partition(run, config, root, data_identity):
    validate_config(config)
    if internal_partitions(config) != [0]:
        raise ValueError("Migration requires single-partition fold0 mode")
    run, root = run.resolve(), root.resolve()
    with run_lock(run):
        receipt_path, backup = run / RECEIPT, run / BACKUP
        if receipt_path.exists() and read_json(receipt_path).get("status") == "COMPLETE":
            receipt = verify_partition_migration(run)
            current = read_json(run / "identity.json")
            if (current["config_sha256"] != config_identity(config) or current["source"] != source_identity(root)
                    or current["runtime"] != runtime_identity() or current["data"] != data_identity):
                raise RuntimeError("Completed single-partition migration cannot authorize further changes")
            return receipt
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            verify_artifact_hashes(backup, receipt["backup_sha256"])
            previous = read_json(backup / "identity.json")
            prior_config = read_json(backup / "resolved_config.json")
            status = read_json(backup / "execution_status.json")
        else:
            previous = read_json(run / "identity.json")
            prior_config = read_json(run / "resolved_config.json")
            status = read_json(run / "execution_status.json")
        if (previous["protocol"] != V411_PROTOCOL or status.get("status") != "PAUSED"
                or prior_config.get("execution", {}).get("mode") != "fold0"
                or internal_partitions(prior_config) != [0, 1, 2]):
            raise RuntimeError("Migration requires a paused v411 fold0 three-partition run")
        comparable = copy.deepcopy(config)
        comparable["execution"].pop("internal_mode")
        if config_identity(comparable) != config_identity(prior_config):
            raise RuntimeError("Migration may only change internal_mode; training/model/data parameters differ")
        if previous["config_sha256"] != config_identity(prior_config):
            raise RuntimeError("Frozen configuration differs from original run identity")
        if previous["runtime"] != runtime_identity() or previous["data"] != data_identity:
            raise RuntimeError("Single-partition migration requires unchanged data and runtime")
        current_source = source_identity(root)
        changes = source_changes(previous["source"], current_source)
        if set(changes) - ALLOWED_SOURCES or any(value["after"] is None for value in changes.values()):
            raise RuntimeError("Single-partition migration includes unauthorized source changes")
        original_root = backup if receipt_path.exists() else run
        verify_artifact_hashes(original_root / "source_snapshot", previous["source"])
        trainer = "src/bme_eating/training/statsfusion_joint_trainer.py"
        verify_training_unchanged((original_root / "source_snapshot" / trainer).read_text(encoding="utf-8"),
                                  (root / trainer).read_text(encoding="utf-8"))
        internal = run / "fold_0/internal"
        if any((internal / f"partition_{partition}").exists() for partition in (1, 2)):
            raise RuntimeError("Migration requires only partition 0; other partitions already exist")
        if (internal / "internal_manifest.json").exists() or any((run / name).exists() for name in
                ("fold_0/model", "fold_0/completion.json", "fold0_final", "fold0_bundle", "joint_gate_manifest.json")):
            raise RuntimeError("Migration cannot relabel completed calibration or outer/final artifacts")
        checkpoints = sorted(path.relative_to(run).as_posix() for path in internal.glob("partition_0/*/*.pt"))
        if receipt_path.exists():
            checkpoints = sorted(receipt["checkpoint_states"])
        if "fold_0/internal/partition_0/selection/resume.pt" not in checkpoints:
            raise RuntimeError("Partition 0 selector resume checkpoint is missing")
        target = {**previous, "config_sha256": config_identity(config), "source": current_source}
        if receipt_path.exists() and receipt["target_identity"] != target:
            raise RuntimeError("Prepared migration source/config changed; restore registered sources")
        checkpoint_states = {}
        for relative in checkpoints:
            payload = torch.load(original_root / relative, map_location="cpu", weights_only=False)
            if (payload.get("config_sha256") != previous["config_sha256"]
                    or payload.get("source_sha256") != digest(previous["source"])
                    or payload.get("data_sha256") != digest(previous["data"])):
                raise RuntimeError(f"Original checkpoint identity mismatch: {relative}")
            checkpoint_states[relative] = {"state_sha256": state_fingerprint(payload),
                "model_identity": payload["identity"], "epoch": payload.get("state", {}).get("epoch", payload.get("epoch")),
                "next_chunk": payload.get("state", {}).get("next_chunk"),
                "requested_epochs": payload.get("requested_epochs")}
        if not receipt_path.exists():
            backup.mkdir(parents=True, exist_ok=True)
            paths = ["identity.json", "resolved_config.json", "execution_status.json", *checkpoints]
            for relative in paths:
                destination = backup / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(run / relative, destination)
                if sha256_file(run / relative) != sha256_file(destination):
                    raise RuntimeError("Migration backup changed while copying")
            shutil.copytree(run / "source_snapshot", backup / "source_snapshot", dirs_exist_ok=True)
            receipt = {"migration": MIGRATION, "status": "PREPARED", "target_identity": target,
                "original_identity_sha256": sha256_file(run / "identity.json"),
                "source_changes": changes, "checkpoint_states": checkpoint_states,
                "model_optimizer_gradients_rng_cursor_preserved": True,
                "partition_assignment_count": 3, "executed_partitions": [0], "full_internal_oof": False,
                "backup_sha256": {path.relative_to(backup).as_posix(): sha256_file(path)
                                  for path in backup.rglob("*") if path.is_file()}}
            write_json_atomic(receipt_path, receipt)
        for relative in checkpoints:
            payload = torch.load(backup / relative, map_location="cpu", weights_only=False)
            payload.update(config_sha256=target["config_sha256"], source_sha256=digest(current_source))
            save_torch_atomic(run / relative, payload)
            restored = torch.load(run / relative, map_location="cpu", weights_only=False)
            if state_fingerprint(restored) != receipt["checkpoint_states"][relative]["state_sha256"]:
                raise RuntimeError("Migration changed training state; restore original checkpoint backup")
        for relative, expected in current_source.items():
            destination = run / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, destination)
            if sha256_file(destination) != expected:
                raise RuntimeError("Source changed during single-partition snapshot creation")
        write_json_atomic(run / "source_snapshot/SHA256SUMS.json", current_source)
        write_json_atomic(run / "resolved_config.json", config)
        status.update(status="PAUSED", internal_mode="single", single_partition_migration=MIGRATION)
        status.setdefault("stages", {})["validate"] = "PAUSED"
        status.pop("error", None)
        write_json_atomic(run / "execution_status.json", status)
        write_json_atomic(run / "identity.json", target)
        receipt.update(status="COMPLETE", target_identity_sha256=sha256_file(run / "identity.json"),
                       migrated_checkpoint_sha256={relative: sha256_file(run / relative) for relative in checkpoints})
        write_json_atomic(receipt_path, receipt)
        return verify_partition_migration(run)
