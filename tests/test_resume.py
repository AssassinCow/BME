from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from bme_eating.config import load_config
from bme_eating.protocol import (
    checkpoint_qualified,
    execution_failure_status,
    initialize_run,
    resume_identity_differences,
)


def test_failure_status_ctrl_c_oom():
    assert execution_failure_status(KeyboardInterrupt()) == "PAUSED"
    assert execution_failure_status(torch.OutOfMemoryError("synthetic OOM")) == "FAILED"
    assert execution_failure_status(RuntimeError("child exit 1")) == "FAILED"


def test_checkpoint_filters_clipping_finite_identity_lineage():
    payload = {"epoch_complete": True, "promotion_eligible": True, "epoch": 8,
               "clipping_fraction": .1, "identity": "identity", "training_subjects": ["fit"],
               "model": {"parameter": torch.ones(2)}}
    assert checkpoint_qualified(payload, "identity", {"fit"}, 8)
    for changes in ({"promotion_eligible": False}, {"epoch": 7}, {"epoch_complete": False},
                    {"clipping_fraction": .21}, {"identity": "wrong"},
                    {"training_subjects": ["outer"]}, {"model": {"parameter": torch.tensor([float("nan")])}}):
        assert not checkpoint_qualified({**payload, **changes}, "identity", {"fit"}, 8)


def test_run_resume_workers_allowed_source_changes_rejected(tmp_path, monkeypatch):
    from pathlib import Path
    config = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v414_full_data.yaml")
    monkeypatch.setattr("bme_eating.protocol.runtime_identity", lambda: {"python": "test"})
    source = tmp_path / "source"
    script = source / "scripts" / "run_statsfusion_joint.py"
    script.parent.mkdir(parents=True)
    script.write_text("original", encoding="utf-8")
    run = initialize_run(config, source, tmp_path / "outputs", "run", {"data": "hash"},
                         fresh=True, skip_s0=True, historical_reference=None)
    changed = copy.deepcopy(config)
    changed["training"]["num_workers"] = 10
    changed["training"]["loader_prefetch_factor"] = 2
    assert initialize_run(changed, source, tmp_path / "outputs", "run", {"data": "hash"},
                          fresh=False, skip_s0=True, historical_reference=None) == run
    script.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity changed"):
        initialize_run(changed, source, tmp_path / "outputs", "run", {"data": "hash"},
                       fresh=False, skip_s0=True, historical_reference=None)
    with pytest.raises(FileExistsError):
        initialize_run(config, source, tmp_path / "outputs", "run", {"data": "hash"},
                       fresh=True, skip_s0=True, historical_reference=None)


def test_git_commit_change_with_identical_source_can_resume(tmp_path, monkeypatch):
    from pathlib import Path
    config = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v414_full_data.yaml")
    source = tmp_path / "source"
    source.mkdir()
    package = source / "src/bme_eating"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    git_head = {"value": "original"}
    monkeypatch.setattr("bme_eating.protocol.runtime_identity", lambda: {"python": "test"})
    monkeypatch.setattr("bme_eating.protocol.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=git_head["value"]))
    arguments = (config, source, tmp_path / "outputs", "run", {"data": "hash"})
    run = initialize_run(*arguments, fresh=True, skip_s0=True, historical_reference=None)
    original = (run / "identity.json").read_bytes()
    git_head["value"] = "committed_same_source"
    for _attempt in range(2):
        assert initialize_run(*arguments, fresh=False, skip_s0=True, historical_reference=None) == run
    assert (run / "identity.json").read_bytes() == original
    history = json.loads((run / "resume_git_history.json").read_text(encoding="utf-8"))
    assert len(history["observations"]) == 1
    assert history["observations"][0]["git_head"] == git_head["value"]
    assert history["observations"][0]["stored_git_head"] == "original"
    monkeypatch.setattr("bme_eating.protocol.runtime_identity", lambda: {"python": "different"})
    with pytest.raises(RuntimeError, match=r"identity changed: runtime.python"):
        initialize_run(*arguments, fresh=False, skip_s0=True, historical_reference=None)
    assert (run / "identity.json").read_bytes() == original


def test_resume_reports_source_data_and_config_differences():
    stored = {"git_head": "before", "source": {"model.py": "before"},
              "runtime": {"cuda": "12.4"}, "data": {"folds": "before"}, "config_sha256": "before"}
    current = {"git_head": "after", "source": {"model.py": "after"},
               "runtime": {"cuda": "12.5"}, "data": {"folds": "after"}, "config_sha256": "after"}
    assert resume_identity_differences(stored, current) == [
        "config_sha256", "data.folds", "runtime.cuda", "source.model.py"]


