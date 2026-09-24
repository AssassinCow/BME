from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from bme_eating.hierarchical_artifacts import (
    HierarchicalRun,
    sha256_file,
    write_json_atomic,
)
from bme_eating.training.hierarchical_trainer import (
    _early_stopping_migration_compatibility,
    _model_state_sha256,
    migrate_completed_state_partition,
)


def _config(
    inference_batch_size: int,
    include_chunks: bool,
    patience_checks: int = 4,
) -> dict[str, object]:
    training = {
        "random_seed": 2026,
        "max_epochs": 32,
        "early_stopping_patience_checks": patience_checks,
        "inference_batch_size": inference_batch_size,
        "inference_num_workers": 2,
        "batch_size": 16,
    }
    if include_chunks:
        training["inference_resume_chunk_rows"] = 32768
    return {
        "experiment": {"name": "hierarchical_v3"},
        "model": {"architecture": "hierarchical_state"},
        "training": training,
        "loss": {"focal_gamma": 2.0},
    }


def _checkpoint(config: dict[str, object], epoch: int, patience: int) -> dict[str, object]:
    return {
        "epoch": epoch,
        "patience": patience,
        "outer_fold": 0,
        "inner_validation_partition": 0,
        "selection_signature": "source-signature",
        "checkpoint_selection_metric": "event_f1",
        "model_config": config["model"],
        "training_config": config["training"],
        "model": {"weight": torch.arange(4, dtype=torch.float32)},
    }


def _run(root: Path, payload: dict[str, object]) -> HierarchicalRun:
    manifest = root / "run_manifest.json"
    write_json_atomic(manifest, payload)
    return HierarchicalRun(root, manifest, payload)


def test_completed_state_checkpoint_migration_preserves_weights(tmp_path) -> None:
    source_config = _config(16, include_chunks=False)
    target_config = _config(64, include_chunks=True, patience_checks=3)
    source_root = tmp_path / "source" / "fold_0"
    source_state = source_root / "crossfit_0" / "state"
    source_state.mkdir(parents=True)
    resolved = source_root / "resolved_config.yaml"
    resolved.write_text(yaml.safe_dump(source_config, sort_keys=False), encoding="utf-8")
    source_payload = {
        "run_name": "source",
        "outer_fold": 0,
        "stage": "CREATED",
        "git": {
            "commit": "old",
            "dirty": True,
            "worktree_sha256": "source-snapshot",
        },
        "input_hashes": {"anchors": "same"},
        "artifact_hashes": {"resolved_config.yaml": sha256_file(resolved)},
    }
    write_json_atomic(source_root / "run_manifest.json", source_payload)
    history = [
        {
            "epoch": epoch,
            "validation_f1": score,
            "early_stopping_patience": patience,
        }
        for epoch, score, patience in (
            (5, 0.39, 0),
            (7, 0.27, 1),
            (9, 0.32, 2),
            (11, 0.35, 3),
            (13, 0.34, 4),
        )
    ]
    best_checkpoint = _checkpoint(source_config, epoch=5, patience=0)
    best_checkpoint["history"] = history[:1]
    last_checkpoint = _checkpoint(source_config, epoch=13, patience=4)
    last_checkpoint["history"] = history
    torch.save(best_checkpoint, source_state / "best.pt")
    torch.save(last_checkpoint, source_state / "last.pt")
    (source_state / "normalization.json").write_text("{}", encoding="utf-8")
    pd.DataFrame(history).to_csv(source_state / "history.csv", index=False)

    target_root = tmp_path / "target" / "fold_0"
    target_root.mkdir(parents=True)
    target_payload = {
        "run_name": "target",
        "outer_fold": 0,
        "stage": "CREATED",
        "git": {
            "commit": "new",
            "dirty": True,
            "worktree_sha256": "target-snapshot",
        },
        "input_hashes": {"anchors": "same"},
        "artifact_hashes": {},
    }
    target = _run(target_root, target_payload)

    migration = migrate_completed_state_partition(
        source_root,
        target,
        source_config,
        target_config,
        partition=0,
    )

    migrated_best = torch.load(
        target_root / "crossfit_0" / "state" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    source_best = torch.load(source_state / "best.pt", map_location="cpu", weights_only=False)
    assert _model_state_sha256(migrated_best["model"]) == _model_state_sha256(
        source_best["model"]
    )
    assert migrated_best["training_config"]["inference_batch_size"] == 64
    assert migrated_best["selection_signature"] != "source-signature"
    assert migration["model_weights_unchanged"] is True
    assert migration["training_will_not_resume"] is True
    assert migration["source_worktree_sha256"] == "source-snapshot"
    assert migration["target_worktree_sha256"] == "target-snapshot"
    assert migration["allowed_config_changes"] == {
        "early_stopping_patience_checks": {"source": 4, "target": 3},
        "inference_batch_size": {"source": 16, "target": 64},
        "inference_resume_chunk_rows": {"source": None, "target": 32768},
    }
    assert migration["early_stopping_compatibility"] == {
        "source_patience_checks": 4,
        "target_patience_checks": 3,
        "best_checkpoint_epoch": 5,
        "counterfactual_stop_epoch": 11,
        "same_best_checkpoint": True,
    }
    persisted = json.loads(target.manifest_path.read_text(encoding="utf-8"))
    assert "0" in persisted["state_checkpoint_migrations"]
    target.verify_artifacts()


def test_checkpoint_migration_rejects_non_runtime_changes(tmp_path) -> None:
    source_config = _config(16, include_chunks=False)
    target_config = _config(64, include_chunks=True)
    target_config["model"] = {"architecture": "different"}
    source_root = tmp_path / "source"
    source_root.mkdir()
    resolved = source_root / "resolved_config.yaml"
    resolved.write_text(yaml.safe_dump(source_config), encoding="utf-8")
    source_payload = {
        "run_name": "source",
        "outer_fold": 0,
        "stage": "CREATED",
        "git": {"commit": "old", "dirty": False},
        "input_hashes": {"anchors": "same"},
        "artifact_hashes": {"resolved_config.yaml": sha256_file(resolved)},
    }
    write_json_atomic(source_root / "run_manifest.json", source_payload)
    target_root = tmp_path / "target"
    target_root.mkdir()
    target = _run(
        target_root,
        {
            "run_name": "target",
            "outer_fold": 0,
            "stage": "CREATED",
            "git": {"commit": "new", "dirty": False},
            "input_hashes": {"anchors": "same"},
            "artifact_hashes": {},
        },
    )

    with pytest.raises(RuntimeError, match="only permits inference"):
        migrate_completed_state_partition(
            source_root,
            target,
            source_config,
            target_config,
            partition=0,
        )


def test_patience_migration_rejects_best_checkpoint_after_target_stop() -> None:
    source_config = _config(16, include_chunks=False)
    target_config = _config(16, include_chunks=False, patience_checks=3)
    history = [
        {
            "epoch": epoch,
            "validation_f1": score,
            "early_stopping_patience": patience,
        }
        for epoch, score, patience in (
            (5, 0.39, 0),
            (7, 0.27, 1),
            (9, 0.32, 2),
            (11, 0.35, 3),
            (13, 0.41, 0),
            (15, 0.30, 1),
            (17, 0.31, 2),
            (19, 0.32, 3),
            (21, 0.33, 4),
        )
    ]
    checkpoints = {
        "best.pt": {"epoch": 13},
        "last.pt": {
            "history": history,
            "checkpoint_selection_metric": "event_f1",
        },
    }

    with pytest.raises(RuntimeError, match="after training would have stopped"):
        _early_stopping_migration_compatibility(
            checkpoints,
            source_config,
            target_config,
        )
