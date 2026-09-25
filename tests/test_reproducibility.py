import hashlib
import json
import subprocess

import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from bme_eating import reproducibility
from bme_eating.reproducibility import (
    require_clean_git_worktree,
    require_git_worktree,
    write_run_manifest,
)
from bme_eating.training.dtp_trainer import _apply_active_optimizer_config


def test_resume_optimizer_uses_active_learning_rates_and_weight_decay() -> None:
    encoder = torch.nn.Parameter(torch.tensor([1.0]))
    head = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = AdamW(
        [
            {"params": [encoder], "lr": 1e-3},
            {"params": [head], "lr": 2e-3},
        ],
        weight_decay=0.01,
    )
    scheduler = LambdaLR(optimizer, lambda _step: 0.5)
    optimizer.step()
    scheduler.step()

    _apply_active_optimizer_config(
        optimizer,
        scheduler,
        {
            "encoder_learning_rate": 1e-4,
            "learning_rate": 3e-4,
            "weight_decay": 0.02,
        },
    )

    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [5e-5, 1.5e-4]
    )
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.02, 0.02]


def test_git_output_is_decoded_as_utf8(monkeypatch, tmp_path):
    observed = {}

    def fake_run(*args, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, stdout="中文差异\n", stderr="")

    monkeypatch.setattr(reproducibility.subprocess, "run", fake_run)

    assert reproducibility._git_value(tmp_path, "diff") == "中文差异"
    assert observed["encoding"] == "utf-8"
    assert observed["errors"] == "strict"


def test_formal_training_rejects_dirty_or_unverifiable_git(monkeypatch, tmp_path):
    values = {
        ("rev-parse", "HEAD"): "abc1234",
        ("status", "--porcelain=v1", "--untracked-files=all"): " M file.py",
        ("diff", "--binary", "HEAD", "--"): "diff",
        ("ls-files", "--others", "--exclude-standard", "-z"): "",
    }
    monkeypatch.setattr(
        reproducibility,
        "_git_value",
        lambda _root, *arguments: values.get(arguments),
    )
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        require_clean_git_worktree(tmp_path)

    with pytest.warns(RuntimeWarning, match="execution is allowed"):
        assert require_git_worktree(tmp_path) == "abc1234"

    values[("status", "--porcelain=v1", "--untracked-files=all")] = ""
    assert require_clean_git_worktree(tmp_path) == "abc1234"

    values[("rev-parse", "HEAD")] = None
    with pytest.raises(RuntimeError, match="Cannot verify"):
        require_clean_git_worktree(tmp_path)


def test_run_manifest_hashes_inputs_without_absolute_paths(tmp_path):
    config_path = tmp_path / "candidate.yaml"
    config_path.write_text("project: {}\n", encoding="utf-8")
    output_root = tmp_path / "outputs"
    index_dir = output_root / "indices"
    index_dir.mkdir(parents=True)
    (index_dir / "subject_folds.json").write_text("{}", encoding="utf-8")
    experiment = output_root / "experiments" / "candidate" / "fold_0"
    experiment.mkdir(parents=True)
    (experiment / "test_predictions.parquet").write_bytes(b"prediction")
    crossfit = experiment / "crossfit_0"
    crossfit.mkdir()
    (crossfit / "best.pt").write_bytes(b"checkpoint")
    config = {
        "_config_path": str(config_path),
        "project": {"seed": 2026},
        "xgboost": {"random_seed": 2026},
    }
    path = write_run_manifest(
        experiment,
        config,
        output_root,
        command=[str(tmp_path / "private" / "train.py"), "--fold", "0"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["experiment"] == {"name": "candidate", "fold": 0}
    assert payload["command"] == ["train.py", "--fold", "0"]
    assert "subject_folds" in payload["hashes"]
    assert (
        payload["artifact_hashes"]["test_predictions.parquet"]
        == hashlib.sha256(b"prediction").hexdigest()
    )
    assert (
        payload["artifact_hashes"]["crossfit_0/best.pt"]
        == hashlib.sha256(b"checkpoint").hexdigest()
    )
    assert str(tmp_path) not in path.read_text(encoding="utf-8")
