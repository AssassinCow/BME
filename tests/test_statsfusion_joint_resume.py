from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from bme_eating.config import load_config
from bme_eating.joint_protocol import (
    checkpoint_qualified,
    execution_failure_status,
    initialize_run,
    resume_identity_differences,
)
from bme_eating.joint_validation import synthetic_batch
from bme_eating.training import statsfusion_joint_trainer as trainer


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
    config = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v410_joint_event.yaml")
    monkeypatch.setattr("bme_eating.joint_protocol.runtime_identity", lambda: {"python": "test"})
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
    config = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v410_joint_event.yaml")
    source = tmp_path / "source"
    source.mkdir()
    git_head = {"value": "original"}
    monkeypatch.setattr("bme_eating.joint_protocol.runtime_identity", lambda: {"python": "test"})
    monkeypatch.setattr("bme_eating.joint_protocol.subprocess.run", lambda *args, **kwargs:
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
    monkeypatch.setattr("bme_eating.joint_protocol.runtime_identity", lambda: {"python": "different"})
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


@pytest.mark.parametrize("failure", [KeyboardInterrupt, torch.OutOfMemoryError])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("version", ["v410", "v412"])
def test_chunk_resume_restores_partial_gradients_and_session_cache(tmp_path, monkeypatch, failure, device, version):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from pathlib import Path
    name = "statsfusion_v412_integrated_repair.yaml" if version == "v412" else "statsfusion_v410_joint_event.yaml"
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / name)
    config["training"]["device"] = device
    config["training"]["loader_prefetch_factor"] = 1
    config["training"]["session_reader_cache_size"] = 1
    config["training"]["gradient_diagnostics_every_chunks"] = 1000
    config["training"]["accumulate_sessions"] = 4
    config["training"]["gradient_clip_norm"] = 1000
    empty_events = pd.DataFrame(columns=["subject_key", "session_id", "start_ms", "end_ms", "valid_duration", "evaluable"])
    groups = {("fit", name): pd.DataFrame({"_row_id": np.arange(index * 8, (index + 1) * 8)})
              for index, name in enumerate(("first", "second"))}
    dataset = SimpleNamespace(events=empty_events, ignore_events=empty_events,
        session_groups=groups, geometry=SimpleNamespace(supervised_steps=4))
    preprocessing = {"training_subjects": ["fit"]}
    monkeypatch.setattr(trainer, "make_joint_dataset", lambda *args, **kwargs: (dataset, preprocessing))
    cache_class = trainer.TimelineCache
    constructed_caches = []
    def tracked_cache(*args, **kwargs):
        constructed_caches.append(args[0])
        return cache_class(*args, **kwargs)
    monkeypatch.setattr(trainer, "TimelineCache", tracked_cache)
    normal_loader = trainer.session_loader
    should_fail = {"enabled": False}
    def fake_loader(dataset, sampler, config):
        for position in range(sampler.start_chunk, len(sampler.chunks)):
            if should_fail["enabled"] and position == 1:
                should_fail["enabled"] = False
                raise failure("synthetic interruption")
            row, _, _, key, _, _ = sampler.chunks[position]
            batch = synthetic_batch(config, 20)
            session_position = row % 8
            shift = (session_position - 19) * 3000
            batch["timestamp_ms"] += shift
            batch["timeline_position"] += session_position - 19
            for name in ("state_target", "onset_target", "offset_target"):
                batch[name] = torch.zeros(1, 20)
            for name in ("state_loss_mask", "onset_loss_mask", "offset_loss_mask", "smooth_mask", "smooth_loss_mask"):
                batch[name] = torch.ones(1, 20)
            batch["subject_key"], batch["session_id"] = key
            if version == "v412":
                batch["augmentation_identity"] = "fixed_" + key[1]
                batch["augmentation_description"] = {"identity": key[1], "gyro_dropout": False}
                for name in ("onset", "offset"):
                    batch[name + "_positive_count"] = torch.tensor(1.0)
                    batch[name + "_background_count"] = torch.tensor(8.0)
            yield batch
    monkeypatch.setattr(trainer, "session_loader", fake_loader)
    inputs = SimpleNamespace(identity={"test": "data"})
    trainer.train_joint_model(inputs, {"fit"}, config, tmp_path / "continuous", "identity",
                              epochs=1, resume=False)
    assert len(constructed_caches) == len(groups)
    should_fail["enabled"] = True
    with pytest.raises(failure):
        trainer.train_joint_model(inputs, {"fit"}, config, tmp_path / "interrupted", "identity",
                                  epochs=1, resume=False)
    saved = torch.load(tmp_path / "interrupted" / "resume.pt", weights_only=False)
    assert saved["state"]["next_chunk"] == 1
    assert saved["state"]["cache_rows"] == 4
    assert saved["state"]["cache_sha256"] == json.loads(
        (constructed_caches[-1] / "checksums.jsonl").read_text().splitlines()[0])["sha256"]
    assert any(value is not None for value in saved["gradients"].values())
    config["training"]["loader_prefetch_factor"] = 2
    config["training"]["session_reader_cache_size"] = 2
    trainer.train_joint_model(inputs, {"fit"}, config, tmp_path / "interrupted", "identity",
                              epochs=1, resume=True)
    complete = torch.load(tmp_path / "continuous" / "latest.pt", weights_only=False)
    resumed = torch.load(tmp_path / "interrupted" / "latest.pt", weights_only=False)
    for name in complete["model"]:
        torch.testing.assert_close(complete["model"][name], resumed["model"][name], rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="resume epoch/selector/exclusion"):
        trainer.train_joint_model(inputs, {"fit"}, config, tmp_path / "interrupted", "identity",
                                  epochs=2, resume=True)
    monkeypatch.setattr(trainer, "session_loader", normal_loader)
