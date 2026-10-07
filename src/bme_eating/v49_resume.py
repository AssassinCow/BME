from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from bme_eating.hierarchical_artifacts import HierarchicalRun, sha256_file, write_json_atomic
from bme_eating.hierarchical_v4_artifacts import (
    _runtime_config_snapshot,
    _saved_resume_config_hash,
    _tracked_inputs,
    resume_config_hash,
)
from bme_eating.reproducibility import git_worktree_identity
from bme_eating.v4_protocol import runtime_source_identity

RECOVERY_RUN = "hierarchical_v4_v49_integrated_repair_20261006c"
ORIGINAL_GIT = {
    "commit": "dd2737282132215268382884b560c498a7763de3",
    "dirty": True,
    "worktree_sha256": "53cec04b39568d77d3baec20ad158994eb30bdc2a6c48ec3d019dc932d42036c",
}
ORIGINAL_CONFIG_HASH = "6395b8b88ff9d14c31759fa580b986ee31004b926ae118e3ed0985d57bb9c53c"
MIGRATION_FILE = "resume_migration.json"


def _contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError("v4.9 recovery evidence escapes the run directory")
    return path


def validate_resume_migration(
    root: Path, config: dict[str, Any], input_hashes: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    report_path = root / MIGRATION_FILE
    if not report_path.is_file():
        if any(root.glob(f"fold_*/{MIGRATION_FILE}")):
            raise RuntimeError("v4.9 recovery migration report is missing")
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    project = Path(__file__).resolve().parents[2]
    active_git = git_worktree_identity(project)
    active_runtime = runtime_source_identity(project / "src/bme_eating")
    if (
        report.get("version") != 1
        or report.get("run_name") != root.name
        or root.name != RECOVERY_RUN
        or report.get("original_git") != ORIGINAL_GIT
        or report.get("original_config_sha256") != ORIGINAL_CONFIG_HASH
        or report.get("folds") != [0, 1]
        or report.get("resume_config_sha256") != resume_config_hash(config)
        or report.get("active_git") != active_git
        or report.get("runtime_source_identity") != active_runtime
        or (input_hashes is not None and report.get("input_hashes") != input_hashes)
        or not report.get("backup_sha256")
    ):
        raise RuntimeError("v4.9 recovery migration identity changed")
    for relative, digest in report["backup_sha256"].items():
        path = _contained(root, relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"v4.9 recovery backup changed: {relative}")
    report_hash = sha256_file(report_path)
    for fold in report["folds"]:
        fold_root = root / f"fold_{fold}"
        manifest = json.loads((fold_root / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("git") == report["active_git"] and (
            manifest.get("artifact_hashes", {}).get(MIGRATION_FILE) != report_hash
            or not (fold_root / MIGRATION_FILE).is_file()
            or sha256_file(fold_root / MIGRATION_FILE) != report_hash
        ):
            raise RuntimeError("v4.9 recovery report differs from the fold evidence")
    return report


def _apply_report(root: Path, report: dict[str, Any]) -> None:
    report_hash = sha256_file(root / MIGRATION_FILE)
    backup_root = root / "resume_backup_20261007"
    for fold in report["folds"]:
        fold_root = root / f"fold_{fold}"
        manifest_path = fold_root / "run_manifest.json"
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current.get("git") == report["active_git"]:
            if current.get("artifact_hashes", {}).get(MIGRATION_FILE) != report_hash:
                raise RuntimeError("v4.9 recovery manifest lost its migration binding")
            HierarchicalRun(fold_root, manifest_path, current).verify_artifacts()
            continue
        backup_path = backup_root / f"fold_{fold}" / "run_manifest.json"
        if sha256_file(manifest_path) != sha256_file(backup_path):
            raise RuntimeError("v4.9 recovery refuses a changed original manifest")
        payload = deepcopy(current)
        payload["training_source_git"] = ORIGINAL_GIT
        payload["git"] = report["active_git"]
        payload.setdefault("source_identity_history", []).append({
            "previous": ORIGINAL_GIT, "active": report["active_git"],
            "migration_sha256": report_hash, "reason": report["reason"],
        })
        payload.setdefault("runtime_config_history", []).append({
            "previous": payload["runtime_config"], "active": report["active_runtime_config"],
        })
        payload["runtime_config"] = report["active_runtime_config"]
        payload["resume_config_sha256"] = report["resume_config_sha256"]
        shutil.copy2(root / MIGRATION_FILE, fold_root / MIGRATION_FILE)
        payload["artifact_hashes"][MIGRATION_FILE] = report_hash
        write_json_atomic(manifest_path, payload)


def legacy_state_checkpoint_matches(
    path: Path, checkpoint: dict[str, Any], config: dict[str, Any],
) -> bool:
    root = next((parent for parent in path.parents if parent.name == RECOVERY_RUN), None)
    if root is None:
        return False
    binding = config.get("_v49_checkpoint_identity", {})
    report = validate_resume_migration(root, config, binding.get("input_hashes"))
    if report is None:
        return False
    entry = report["legacy_checkpoints"].get(path.resolve().relative_to(root.resolve()).as_posix())
    return bool(
        entry
        and entry["sha256"] == sha256_file(path)
        and entry["resume_config_sha256"] == checkpoint.get("resume_config_sha256")
        and entry["epoch"] == checkpoint.get("epoch")
    )


def prepare_resume_migration(
    config: dict[str, Any], input_root: Path, output_root: Path, run_name: str, *, apply: bool,
) -> dict[str, Any]:
    if run_name != RECOVERY_RUN or config.get("decoder", {}).get("candidate_protocol") != "v4.9":
        raise RuntimeError("This recovery is restricted to the audited 20261006c run")
    root = output_root / "experiments" / run_name
    tracked = _tracked_inputs(config, input_root)
    input_hashes = {name: sha256_file(path) for name, path in tracked.items()}
    existing = validate_resume_migration(root, config, input_hashes)
    if existing is not None:
        if apply:
            _apply_report(root, existing)
        report_hash = sha256_file(root / MIGRATION_FILE)
        for fold in existing["folds"]:
            fold_root = root / f"fold_{fold}"
            path = fold_root / "run_manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("git") != existing["active_git"]
                or payload.get("artifact_hashes", {}).get(MIGRATION_FILE) != report_hash
            ):
                raise RuntimeError("v4.9 recovery was not completely applied")
            HierarchicalRun(fold_root, path, payload).verify_artifacts()
        return existing
    if (root / "v49_gate_manifest.json").exists() or (output_root / "final" / run_name).exists():
        raise RuntimeError("Cannot migrate a locked or final v4.9 run")
    protocol_path = root / "execution_protocol.json"
    if json.loads(protocol_path.read_text(encoding="utf-8")) != {
        "run_name": run_name, "s0_run": None, "skip_s0": True,
    }:
        raise RuntimeError("v4.9 recovery cannot change the original S0 policy")
    fold_roots = sorted(root.glob("fold_*"))
    if [path.name for path in fold_roots] != ["fold_0", "fold_1"]:
        raise RuntimeError("v4.9 recovery requires the audited two-fold progress")
    manifests = {}
    originals = [protocol_path]
    checkpoints = {}
    for fold, fold_root in enumerate(fold_roots):
        path = fold_root / "run_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("git") != ORIGINAL_GIT
            or payload.get("resolved_config_sha256") != ORIGINAL_CONFIG_HASH
            or payload.get("input_hashes") != input_hashes
            or payload.get("stage") != ("EVALUATED" if fold == 0 else "CREATED")
            or _saved_resume_config_hash(fold_root, payload) != resume_config_hash(config)
        ):
            raise RuntimeError(f"v4.9 recovery cannot accept changed fold {fold} identity")
        HierarchicalRun(fold_root, path, payload).verify_artifacts()
        manifests[fold] = payload
        originals.extend([path, fold_root / "resolved_config.yaml", *fold_root.rglob("*.pt")])
        for checkpoint_path in fold_root.rglob("*_last.pt"):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint.get("resume_config_sha256") != ORIGINAL_CONFIG_HASH:
                raise RuntimeError("v4.9 legacy checkpoint config identity differs")
            if any(not torch.isfinite(value).all() for value in checkpoint["model"].values()):
                raise RuntimeError("v4.9 recovery refuses non-finite weights")
            checkpoints[checkpoint_path.relative_to(root).as_posix()] = {
                "sha256": sha256_file(checkpoint_path), "epoch": int(checkpoint["epoch"]),
                "resume_config_sha256": checkpoint["resume_config_sha256"],
            }
    selector = "fold_1/crossfit/partition_0/state/selector_seed_2026_last.pt"
    if checkpoints.get(selector, {}).get("epoch") != 4:
        raise RuntimeError("v4.9 recovery requires the audited fold 1 epoch 4 checkpoint")
    project = Path(__file__).resolve().parents[2]
    report = {
        "version": 1, "run_name": run_name, "created_at": datetime.now(UTC).isoformat(),
        "reason": "Correct epoch clipping denominator and resume runtime-only loader changes",
        "original_git": ORIGINAL_GIT, "active_git": git_worktree_identity(project),
        "original_config_sha256": ORIGINAL_CONFIG_HASH,
        "resume_config_sha256": resume_config_hash(config), "input_hashes": input_hashes,
        "runtime_source_identity": runtime_source_identity(project / "src/bme_eating"),
        "previous_runtime_config": {str(fold): payload["runtime_config"] for fold, payload in manifests.items()},
        "active_runtime_config": _runtime_config_snapshot(config),
        "folds": list(manifests), "legacy_checkpoints": checkpoints,
        "continuation": {"fold": 1, "completed_epoch": 4, "next_epoch": 5},
        "source_review_limit": "Original dirty worktree is identified by hash; its full source snapshot is unavailable",
    }
    if not apply:
        return report
    backup_root = root / "resume_backup_20261007"
    backup_root.mkdir(exist_ok=True)
    backups = {}
    for source in originals:
        target = backup_root / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = sha256_file(source)
        if sha256_file(target) != digest:
            raise RuntimeError("v4.9 recovery backup verification failed")
        backups[target.relative_to(root).as_posix()] = digest
    source_diff = backup_root / "active_source.diff"
    source_diff.write_bytes(subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"], cwd=project, check=True, capture_output=True,
    ).stdout)
    backups[source_diff.relative_to(root).as_posix()] = sha256_file(source_diff)
    changed_files = subprocess.run(
        ["git", "diff", "--name-only", "-z", "HEAD", "--"],
        cwd=project, check=True, capture_output=True,
    ).stdout
    untracked_files = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=project, check=True, capture_output=True,
    ).stdout
    for relative in set((changed_files + untracked_files).decode("utf-8").split("\0")) - {""}:
        source = project / relative
        if not source.is_file():
            continue
        target = backup_root / "active_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        backups[target.relative_to(root).as_posix()] = sha256_file(target)
    report["backup_sha256"] = backups
    write_json_atomic(root / MIGRATION_FILE, report)
    validate_resume_migration(root, config, input_hashes)
    _apply_report(root, report)
    return report
