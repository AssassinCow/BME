from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from bme_eating.config import load_config
from bme_eating.joint_validation import synthetic_batch
from bme_eating.training import statsfusion_joint_trainer as trainer


def test_first_clipped_update_does_not_fail_a_recoverable_epoch():
    report = trainer.epoch_clipping_report(8, 1, 1, 87, .5)
    assert report["clipping_fraction"] == 1
    assert report["minimum_epoch_clipping_fraction"] == pytest.approx(1 / 87)
    assert not report["abort_failed"]
    assert trainer.epoch_clipping_report(8, 1, 87, 87, .5)["clipping_fraction"] == pytest.approx(1 / 87)


@pytest.mark.parametrize("planned", [1, 2, 3, 5, 87])
@pytest.mark.parametrize("threshold", [0, .2, .5, 1])
def test_abort_exactly_when_no_remaining_updates_can_meet_epoch_limit(planned, threshold):
    for observed in range(planned + 1):
        for clipped in range(observed + 1):
            report = trainer.epoch_clipping_report(8, clipped, observed, planned, threshold)
            possible_final_fractions = [(clipped + extra) / planned
                                        for extra in range(planned - observed + 1)]
            assert report["abort_failed"] == all(value > threshold for value in possible_final_fractions)
            if observed == planned:
                assert report["abort_failed"] == (report["clipping_fraction"] > threshold)


def test_warmup_and_exact_threshold_keep_existing_semantics():
    assert not trainer.epoch_clipping_report(5, 5, 5, 5, .5)["abort_failed"]
    assert not trainer.epoch_clipping_report(8, 2, 2, 4, .5)["abort_failed"]
    assert trainer.epoch_clipping_report(8, 3, 3, 4, .5)["abort_failed"]


@pytest.mark.parametrize("counts", [(1, 0, 87), (-1, 1, 87), (1, 88, 87), (0, 0, 0)])
def test_invalid_clipping_counters_fail_closed(counts):
    with pytest.raises(RuntimeError, match="clipping counters"):
        trainer.epoch_clipping_report(8, *counts, .5)


@pytest.fixture
def training_case(monkeypatch):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v410_joint_event.yaml")
    config["training"]["device"] = "cpu"
    config["training"]["gradient_diagnostics_every_chunks"] = 1000
    events = pd.DataFrame(columns=["subject_key", "session_id", "start_ms", "end_ms",
                                  "valid_duration", "evaluable"])
    groups = {("fit", f"session_{index}"): pd.DataFrame({"_row_id": np.arange(index * 4, (index + 1) * 4)})
              for index in range(17)}
    dataset = SimpleNamespace(events=events, ignore_events=events, session_groups=groups,
                              geometry=SimpleNamespace(supervised_steps=4))
    monkeypatch.setattr(trainer, "make_joint_dataset", lambda *args, **kwargs:
                        (dataset, {"training_subjects": ["fit"]}))
    monkeypatch.setattr(trainer, "generate_chunk", lambda *args: [])
    monkeypatch.setattr(trainer, "random_background_candidates", lambda *args: [])
    interrupt = {"enabled": False}

    def fake_loader(dataset, sampler, config):
        for position in range(sampler.start_chunk, len(sampler.chunks)):
            if interrupt["enabled"] and sampler.epoch == 8 and position == 5:
                interrupt["enabled"] = False
                raise KeyboardInterrupt("clipping counter recovery probe")
            key = sampler.chunks[position][3]
            batch = synthetic_batch(config, 20)
            batch["timestamp_ms"] -= 16 * 3000
            batch["timeline_position"] -= 16
            for name in ("state_target", "onset_target", "offset_target"):
                batch[name] = torch.zeros(1, 20)
            for name in ("state_loss_mask", "onset_loss_mask", "offset_loss_mask",
                         "smooth_mask", "smooth_loss_mask"):
                batch[name] = torch.ones(1, 20)
            batch["subject_key"], batch["session_id"] = key
            yield batch

    monkeypatch.setattr(trainer, "session_loader", fake_loader)
    return config, SimpleNamespace(identity={"test": "data"}), interrupt


def test_real_training_and_resume_preserve_clipping_denominator(tmp_path, monkeypatch, training_case):
    config, inputs, interrupt = training_case
    original_clip = torch.nn.utils.clip_grad_norm_
    calls = {"updates": 0}

    def clipping_probe(*args, **kwargs):
        original_clip(*args, **kwargs)
        calls["updates"] += 1
        return torch.tensor(6.0 if calls["updates"] == 36 else 1.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clipping_probe)
    trainer.train_joint_model(inputs, {"fit"}, config, tmp_path / "continuous", "identity",
                              epochs=8, resume=False)
    calls["updates"] = 0
    interrupt["enabled"] = True
    with pytest.raises(KeyboardInterrupt):
        trainer.train_joint_model(inputs, {"fit"}, config, tmp_path / "resumed", "identity",
                                  epochs=8, resume=False)
    saved = torch.load(tmp_path / "resumed" / "resume.pt", weights_only=False)
    assert saved["state"]["epoch"] == 8
    assert saved["state"]["optimizer_steps"] == saved["state"]["clipped_steps"] == 1
    assert saved["state"]["session_count"] == 1
    assert saved["state"]["next_chunk"] == 5
    trainer.train_joint_model(inputs, {"fit"}, config, tmp_path / "resumed", "identity",
                              epochs=8, resume=True)
    complete = torch.load(tmp_path / "continuous" / "latest.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resumed" / "latest.pt", weights_only=False)
    assert resumed["clipping_fraction"] == .2
    assert resumed["promotion_eligible"]
    for name in complete["model"]:
        torch.testing.assert_close(complete["model"][name], resumed["model"][name], rtol=0, atol=0)
    history = json.loads((tmp_path / "resumed" / "training_history.json").read_text(encoding="utf-8"))
    assert history[-1]["clipping_gate"]["expected_optimizer_steps"] == 5
    assert history[-1]["maximum_gradient_norm"] == 6


def test_excessive_epoch_clipping_still_fails_before_promotion(tmp_path, monkeypatch, training_case):
    config, inputs, _ = training_case
    original_clip = torch.nn.utils.clip_grad_norm_

    def clipping_probe(*args, **kwargs):
        original_clip(*args, **kwargs)
        return torch.tensor(6.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clipping_probe)
    with pytest.raises(RuntimeError, match="3/5 planned updates clipped"):
        trainer.train_joint_model(inputs, {"fit"}, config, tmp_path, "identity", epochs=8, resume=False)
    report = json.loads((tmp_path / "clipping_failure_report.json").read_text(encoding="utf-8"))
    assert report["epoch"] == 6
    assert report["minimum_epoch_clipping_fraction"] == .6
    assert not report["epoch_complete"] and not report["promotion_eligible"]
    saved = torch.load(tmp_path / "resume.pt", weights_only=False)
    assert saved["state"]["epoch"] == 6
    assert saved["state"]["optimizer_steps"] == 2
    assert saved["state"]["next_chunk"] == report["committed_next_chunk"]
    latest = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert latest["epoch"] == 5 and not latest["promotion_eligible"]
    assert not (tmp_path / "selected.pt").exists()
