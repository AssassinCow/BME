from __future__ import annotations

import copy
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


@pytest.mark.parametrize("failure", [KeyboardInterrupt, torch.OutOfMemoryError])
def test_chunk_resume_restores_partial_gradients_and_session_cache(tmp_path, monkeypatch, failure):
    from pathlib import Path
    config = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v410_joint_event.yaml")
    config["training"]["device"] = "cpu"
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
            yield batch
    monkeypatch.setattr(trainer, "session_loader", fake_loader)
    inputs = SimpleNamespace(identity={"test": "data"})
    trainer.train_joint_model(inputs, {"fit"}, config, tmp_path / "continuous", "identity",
                              epochs=1, resume=False)
    should_fail["enabled"] = True
    with pytest.raises(failure):
        trainer.train_joint_model(inputs, {"fit"}, config, tmp_path / "interrupted", "identity",
                                  epochs=1, resume=False)
    saved = torch.load(tmp_path / "interrupted" / "resume.pt", weights_only=False)
    assert saved["state"]["next_chunk"] == 1
    assert saved["state"]["cache_rows"] == 4
    assert any(value is not None for value in saved["gradients"].values())
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
