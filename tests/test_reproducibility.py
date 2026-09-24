import hashlib
import json
import subprocess

import pytest

from bme_eating import reproducibility
from bme_eating.reproducibility import (
    require_clean_git_worktree,
    require_git_worktree,
    write_run_manifest,
)


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
