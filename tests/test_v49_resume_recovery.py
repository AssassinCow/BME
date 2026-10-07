from __future__ import annotations

import json
import shutil

import pytest
import torch
import yaml

from bme_eating import v49_resume as recovery
from bme_eating.hierarchical_artifacts import _canonical_hash, sha256_file, write_json_atomic
from bme_eating.hierarchical_v4_artifacts import resume_config_hash
from bme_eating.training import hierarchical_v4_trainer as trainer


@pytest.fixture
def interrupted_run(tmp_path, monkeypatch):
    config = {
        "decoder": {"candidate_protocol": "v4.9"},
        "training": {
            "device": "cpu", "learning_rate": 0.01, "weight_decay": 0.0,
            "num_workers": 10, "inference_num_workers": 10,
            "loader_prefetch_factor": 1, "session_reader_cache_size": 1,
        },
    }
    saved = json.loads(json.dumps(config))
    saved["training"].update(num_workers=12, inference_num_workers=12)
    saved["training"].pop("loader_prefetch_factor")
    saved["training"].pop("session_reader_cache_size")
    original_hash = _canonical_hash(saved)
    monkeypatch.setattr(recovery, "ORIGINAL_CONFIG_HASH", original_hash)
    data = tmp_path / "events.json"
    write_json_atomic(data, {"event": "original"})
    inputs = {"events": sha256_file(data)}
    monkeypatch.setattr(recovery, "_tracked_inputs", lambda *_args: {"events": data})
    root = tmp_path / "experiments" / recovery.RECOVERY_RUN
    for fold in range(2):
        fold_root = root / f"fold_{fold}"
        fold_root.mkdir(parents=True)
        config_path = fold_root / "resolved_config.yaml"
        config_path.write_text(yaml.safe_dump(saved), encoding="utf-8")
        write_json_atomic(fold_root / "run_manifest.json", {
            "git": recovery.ORIGINAL_GIT, "resolved_config_sha256": original_hash,
            "input_hashes": inputs, "stage": "EVALUATED" if fold == 0 else "CREATED",
            "runtime_config": {"training.num_workers": 12, "training.inference_num_workers": 12},
            "artifact_hashes": {"resolved_config.yaml": sha256_file(config_path)},
        })
    write_json_atomic(root / "execution_protocol.json", {
        "run_name": recovery.RECOVERY_RUN, "s0_run": None, "skip_s0": True,
    })
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    for epoch in range(4):
        optimizer.zero_grad()
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        scheduler.step()
    checkpoint = {
        "kind": "selector", "epoch": 4, "seed": 2026,
        "subjects": {"fit": ["train"], "selector": ["validation"]},
        "resume_config_sha256": original_hash, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": None,
    }
    checkpoint_path = root / "fold_1/crossfit/partition_0/state/selector_seed_2026_last.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(checkpoint, checkpoint_path)
    return tmp_path, root, config, checkpoint_path, checkpoint, data


def _migrate(interrupted_run, *, apply=True):
    output, _, config, *_ = interrupted_run
    return recovery.prepare_resume_migration(config, output, output, recovery.RECOVERY_RUN, apply=apply)


def test_recovery_dry_run_preserves_progress(interrupted_run):
    _, root, _, checkpoint_path, *_ = interrupted_run
    originals = {path: sha256_file(path) for path in root.rglob("*") if path.is_file()}
    report = _migrate(interrupted_run, apply=False)
    assert report["continuation"] == {"fold": 1, "completed_epoch": 4, "next_epoch": 5}
    assert report["legacy_checkpoints"][checkpoint_path.relative_to(root).as_posix()]["epoch"] == 4
    assert {path: sha256_file(path) for path in root.rglob("*") if path.is_file()} == originals


def test_recovery_preserves_weights_and_restores_optimizer_scheduler_rng(interrupted_run):
    _, root, config, path, checkpoint, _ = interrupted_run
    original_sha = sha256_file(path)
    report = _migrate(interrupted_run)
    assert sha256_file(path) == original_sha
    assert _migrate(interrupted_run) == report
    manifest = json.loads((root / "fold_0/run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "EVALUATED"
    assert manifest["training_source_git"] == recovery.ORIGINAL_GIT
    model = torch.nn.Linear(1, 1)
    epoch, optimizer, scheduler, _ = trainer._load_state_epoch(
        path, model, config, kind="selector", seed=2026,
        subjects={"fit": {"train"}, "selector": {"validation"}}, maximum_epochs=32,
    )
    assert epoch == 4
    assert scheduler == checkpoint["scheduler"]
    assert torch.equal(torch.get_rng_state(), checkpoint["torch_rng_state"])
    assert optimizer.param_groups[0]["lr"] == checkpoint["optimizer"]["param_groups"][0]["lr"]
    for key, value in checkpoint["model"].items():
        assert torch.equal(model.state_dict()[key], value)
    for parameter, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            assert torch.equal(value, checkpoint["optimizer"]["state"][parameter][name])
    assert report["resume_config_sha256"] == resume_config_hash(config)


def test_recovery_application_can_finish_after_manifest_interruption(interrupted_run):
    _, root, *_ = interrupted_run
    report = _migrate(interrupted_run)
    shutil.copy2(
        root / "resume_backup_20261007/fold_1/run_manifest.json",
        root / "fold_1/run_manifest.json",
    )
    (root / "fold_1" / recovery.MIGRATION_FILE).unlink()
    assert _migrate(interrupted_run) == report
    manifest = json.loads((root / "fold_1/run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["git"] == report["active_git"]


def test_recovery_rejects_model_settings_before_writing(interrupted_run):
    _, root, config, *_ = interrupted_run
    config["training"]["learning_rate"] = 0.02
    with pytest.raises(RuntimeError, match="changed fold"):
        _migrate(interrupted_run)
    assert not (root / recovery.MIGRATION_FILE).exists()
    assert not (root / "resume_backup_20261007").exists()


@pytest.mark.parametrize("issue", ["checkpoint", "copy", "config", "source", "data", "backup", "missing_report"])
def test_recovery_fails_closed_on_changed_evidence(interrupted_run, monkeypatch, issue):
    _, root, config, path, checkpoint, data = interrupted_run
    report = _migrate(interrupted_run)
    if issue == "checkpoint":
        checkpoint["epoch"] = 5
        torch.save(checkpoint, path)
    elif issue == "copy":
        copied = path.with_name("unregistered_last.pt")
        shutil.copy2(path, copied)
        path = copied
    elif issue == "config":
        config["training"]["learning_rate"] = 0.02
    elif issue == "source":
        monkeypatch.setattr(recovery, "git_worktree_identity", lambda *_args: {"commit": "changed"})
    elif issue == "data":
        write_json_atomic(data, {"event": "changed"})
    elif issue == "backup":
        (root / next(iter(report["backup_sha256"]))).write_text("changed", encoding="utf-8")
    elif issue == "missing_report":
        (root / recovery.MIGRATION_FILE).unlink()
    with pytest.raises(RuntimeError):
        if issue in {"data", "missing_report"}:
            _migrate(interrupted_run)
        else:
            trainer._load_state_epoch(
                path, torch.nn.Linear(1, 1), config, kind="selector", seed=2026,
                subjects={"fit": {"train"}, "selector": {"validation"}}, maximum_epochs=32,
            )
