from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bme_eating.config import load_config
from bme_eating.protocol import (
    CHECKPOINT_STATE_VERSION,
    PROTOCOL,
    checkpoint_qualified,
    execution_failure_status,
    initialize_run,
    resume_identity_differences,
    source_identity,
    training_source_identity,
)


def test_failure_status_ctrl_c_oom():
    assert execution_failure_status(KeyboardInterrupt()) == "PAUSED"
    assert execution_failure_status(torch.OutOfMemoryError("synthetic OOM")) == "FAILED"
    assert execution_failure_status(RuntimeError("child exit 1")) == "FAILED"


def test_checkpoint_filters_clipping_finite_identity_lineage():
    payload = {"epoch_complete": True, "promotion_eligible": True, "epoch": 8,
               "protocol": PROTOCOL, "state_version": CHECKPOINT_STATE_VERSION,
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


@pytest.fixture
def output_only_run(tmp_path, monkeypatch):
    from bme_eating.artifacts import write_json_atomic

    config = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v414_full_data.yaml")
    monkeypatch.setattr("bme_eating.protocol.runtime_identity", lambda: {"python": "test"})
    root = tmp_path / "source"
    package = root / "src/bme_eating"
    package.mkdir(parents=True)
    (package / "model.py").write_text("training = 1\n", encoding="utf-8")
    (package / "display.py").write_text("display = 1\n", encoding="utf-8")
    output, data = tmp_path / "outputs", {"data": "fixture"}
    run = initialize_run(config, root, output, "run", data, fresh=True, skip_s0=True)
    def normalized(directory):
        return {relative: hashlib.sha256((directory / relative).read_text(
            encoding="utf-8").encode("utf-8")).hexdigest() for relative in source_identity(root)}
    baseline = normalized(root)
    (package / "display.py").write_text("display = 2\n", encoding="utf-8")
    write_json_atomic(root / "manifests/v414_output_only_resume.json", {
        "policy": "v414_dev_checkpoints_10_every_4_quiet_replay_v1",
        "training_source": baseline, "execution_source": normalized(root)})
    return config, root, output, data, run


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_exact_output_only_changes_resume_without_rewriting_training_identity(output_only_run, newline):
    from bme_eating.artifacts import sha256_file

    config, root, output, data, run = output_only_run
    display = root / "src/bme_eating/display.py"
    display.write_text(display.read_text(encoding="utf-8"), encoding="utf-8", newline=newline)
    identity_before = (run / "identity.json").read_bytes()
    stored_source = json.loads(identity_before)["source"]
    snapshot_before = {relative: sha256_file(run / "source_snapshot" / relative) for relative in stored_source}
    for _attempt in range(2):
        assert initialize_run(config, root, output, "run", data, fresh=False, skip_s0=True) == run
        assert training_source_identity(root, config) == stored_source
    assert (run / "identity.json").read_bytes() == identity_before
    assert {relative: sha256_file(run / "source_snapshot" / relative) for relative in stored_source} == snapshot_before
    assert source_identity(root) != stored_source
    history = json.loads((run / "runtime_source_history.json").read_text(encoding="utf-8"))
    assert len(history["observations"]) == 1
    compatibility = history["observations"][0]
    assert compatibility["training_source"] == stored_source
    assert compatibility["execution_source"] == source_identity(root)
    snapshots = list((run / "runtime_source_snapshots").iterdir())
    assert len(snapshots) == 1
    assert json.loads((snapshots[0] / "SHA256SUMS.json").read_text()) == source_identity(root)
    (root / "src/bme_eating/model.py").write_text("training = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Execution source changed"):
        training_source_identity(root, config)


@pytest.mark.parametrize("change", ["model", "display", "config", "data", "runtime", "snapshot"])
def test_output_only_resume_rejects_unregistered_changes(output_only_run, monkeypatch, change):
    config, root, output, data, run = output_only_run
    identity_before = (run / "identity.json").read_bytes()
    if change in {"model", "display"}:
        (root / f"src/bme_eating/{change}.py").write_text("unregistered = 3\n", encoding="utf-8")
    elif change == "config":
        config["training"]["weight_decay"] += .001
    elif change == "data":
        data = {"data": "different"}
    elif change == "runtime":
        monkeypatch.setattr("bme_eating.protocol.runtime_identity", lambda: {"python": "different"})
    else:
        (run / "source_snapshot/src/bme_eating/model.py").write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity changed|checksum mismatch"):
        initialize_run(config, root, output, "run", data, fresh=False, skip_s0=True)
    assert (run / "identity.json").read_bytes() == identity_before
    assert not (run / "runtime_source_history.json").exists()
    assert "_output_only_resume" not in config


def test_output_only_registry_matches_delivered_source():
    root = Path(__file__).resolve().parents[1]
    policy = json.loads((root / "manifests/v414_output_only_resume.json").read_text(encoding="utf-8"))
    current = {relative: hashlib.sha256((root / relative).read_text(
        encoding="utf-8").encode("utf-8")).hexdigest() for relative in source_identity(root)}
    assert current == policy["execution_source"]


