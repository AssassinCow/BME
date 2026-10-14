from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from helpers import truth_fixture

from bme_eating.candidates import NoSourceCandidatesError
from bme_eating.config import load_config
from bme_eating.proposal_stream import make_stream
from bme_eating.protocol import CHECKPOINT_STATE_VERSION, PROFILE, validate_config
from bme_eating.training import statsfusion_joint_trainer as trainer
from bme_eating.validation import synthetic_batch


@pytest.fixture
def config():
    torch.set_num_threads(2)
    value = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v414_full_data.yaml")
    value["training"]["device"] = "cpu"
    return value


def stream_candidates(config, peaks=(), *, steps=150, identity="model"):
    stream = make_stream("subject", "session", identity, copy.deepcopy(config["decoder"]))
    output = []
    for position in range(1, steps + 1):
        output.extend(stream.observe(position * 3000,
            {"state": .9 if 8 <= position <= 25 else 0}, 1,
            onset_probability=.8 if position * 3000 in peaks else 0))
    return output


@pytest.mark.parametrize("peaks", [(27000,), (27000, 45000)])
def test_learned_onset_preserves_compatibility_coverage(config, peaks):
    baseline = stream_candidates(config)
    actual = stream_candidates(config, peaks)

    def overlap(candidate):
        intersection = max(0, min(candidate.end_ms, 39000) - max(candidate.start_ms, 15000))
        return intersection / (max(candidate.end_ms, 39000) - min(candidate.start_ms, 15000))

    assert max(map(overlap, baseline)) > .25
    assert max(map(overlap, actual)) > .25
    assert len(actual) == 4
    assert sum(candidate.source == "learned_onset" for candidate in actual) == 1
    assert any(candidate.generation_parameters["compatibility_guard"] for candidate in actual)
    extended = stream_candidates(config, peaks, steps=200)
    assert [candidate.to_dict() for candidate in actual] == [candidate.to_dict() for candidate in extended[:len(actual)]]
    other_identity = stream_candidates(config, peaks, identity="another_model")
    assert [(candidate.start_ms, candidate.end_ms, candidate.source) for candidate in actual] == [
        (candidate.start_ms, candidate.end_ms, candidate.source) for candidate in other_identity]


def test_duplicate_learned_onset_does_not_waste_compatibility_slots(config):
    baseline = stream_candidates(config)
    actual = stream_candidates(config, (21000,))
    assert {(candidate.start_ms, candidate.end_ms) for candidate in actual} == {
        (candidate.start_ms, candidate.end_ms) for candidate in baseline}
    assert len(actual) == 3


@pytest.mark.parametrize("mode,validation_every", [
    ("qualification_deadline", 1), ("qualified_patience", 1), ("qualified_patience", 2)])
@pytest.mark.parametrize("failure", [KeyboardInterrupt, torch.OutOfMemoryError])
def test_early_stop_commit_resume_is_terminal(config, tmp_path, monkeypatch, mode,
                                             validation_every, failure):
    config["training"].update({"checkpoint_min_epoch": 1,
        "validation_every_epochs": validation_every, "patience_checks": 1,
        "gradient_clip_norm": 1000, "natural_replay_from_epoch": 100,
        "qualification_deadline_epoch": 1 if mode == "qualification_deadline" else 24})
    truth = truth_fixture().assign(subject_key="fit", session_id="session", start_ms=0,
                                  end_ms=6000, evaluable=True, valid_duration=True)

    class Dataset:
        training = True
        geometry = SimpleNamespace(supervised_steps=4)
        events, ignore_events = truth, truth.iloc[:0]

        def __init__(self):
            self.session_groups = {("fit", "session"): pd.DataFrame({"_row_id": np.arange(8)})}

        def endpoint_counts(self, key, name):
            return 1, 8

        def __getitem__(self, item):
            row, _, _ = item
            batch = synthetic_batch(config, 20)
            batch["timestamp_ms"] += (row % 8 - 19) * 3000
            batch["timeline_position"] += row % 8 - 19
            batch["motion_blocks"][:, :, 6:] = 1
            for name in ("state_target", "onset_target", "offset_target"):
                batch[name] = torch.zeros(1, 20)
            for name in ("state_loss_mask", "onset_loss_mask", "offset_loss_mask", "smooth_mask", "smooth_loss_mask"):
                batch[name] = torch.ones(1, 20)
            batch["augmentation_identity"] = "fixed"
            batch["augmentation_description"] = {"view": "fixed"}
            for name in ("state", "proposal_3", "proposal_15", "proposal_60", "smooth"):
                batch[name + "_session_count"] = torch.tensor(8.)
            for name in ("onset", "offset"):
                batch[name + "_positive_count"] = torch.tensor(1.)
                batch[name + "_background_count"] = torch.tensor(8.)
            return {name: value[0] if isinstance(value, torch.Tensor) and value.ndim > 0 else value
                    for name, value in batch.items()}

    def loader(dataset, sampler, configuration):
        for position in range(sampler.start_chunk, len(sampler.chunks)):
            row, epoch, mask, *_ = sampler.chunks[position]
            batch = dataset[(row, epoch, mask)]
            yield {name: value[None] if isinstance(value, torch.Tensor) else value for name, value in batch.items()}

    inputs = SimpleNamespace(identity={"data": "synthetic_boundary_fixture"},
        frames=lambda *args, **kwargs: (None, None,
            truth_fixture().assign(subject_key="selector", evaluable=True, valid_duration=True), None))
    armed = {"value": False}
    real_save = trainer.save_torch_atomic
    stop_epoch = 1 if mode == "qualification_deadline" else 2 * validation_every

    def save(path, payload):
        real_save(path, payload)
        if armed["value"] and path.name == "resume.pt":
            state = torch.load(path, map_location="cpu", weights_only=False)["state"]
            if state["epoch"] == stop_epoch + 1 and state["history"]:
                armed["value"] = False
                raise failure("after committed early-stop cursor")

    def fit(*args, **kwargs):
        if mode == "qualification_deadline":
            raise NoSourceCandidatesError("synthetic empty source pool")
        return SimpleNamespace()

    def train(path, resume, *, expect_interruption=False):
        if mode == "qualification_deadline" and not expect_interruption:
            with pytest.raises(RuntimeError, match="selection failed|No finite/clipping-qualified"):
                trainer.train_joint_model(inputs, {"fit"}, config, path, "boundary_test",
                    epochs=6, resume=resume, selector={"selector"})
        else:
            trainer.train_joint_model(inputs, {"fit"}, config, path, "boundary_test",
                epochs=6, resume=resume, selector={"selector"})

    monkeypatch.setattr(trainer, "make_joint_dataset", lambda *args, **kwargs: (Dataset(), {"training_subjects": ["fit"]}))
    monkeypatch.setattr(trainer, "session_loader", loader)
    monkeypatch.setattr(trainer, "generate_chunk", lambda *args, **kwargs: [])
    monkeypatch.setattr(trainer, "fit_source_mapping", fit)
    monkeypatch.setattr(trainer, "predict_joint_model", lambda *args, **kwargs: {"coarse": [], "windows": pd.DataFrame()})
    monkeypatch.setattr(trainer, "choose_operating_point", lambda *args, **kwargs: ({"f1": .5, "promotion_eligible": True}, []))
    monkeypatch.setattr(trainer, "print_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, "save_torch_atomic", save)
    continuous, interrupted = tmp_path / "continuous", tmp_path / "interrupted"
    train(continuous, False)
    armed["value"] = True
    with pytest.raises(failure):
        train(interrupted, False, expect_interruption=True)
    saved = torch.load(interrupted / "resume.pt", map_location="cpu", weights_only=False)
    assert saved["state"]["training_complete"]
    assert saved["state"]["stop_requested"]
    assert saved["state"]["stop_reason"] == (mode if mode == "qualification_deadline" else "f1_patience")
    assert saved["state"]["completed_epoch"] == stop_epoch
    train(interrupted, True)
    normal = torch.load(continuous / "resume.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(interrupted / "resume.pt", map_location="cpu", weights_only=False)
    assert [item["epoch"] for item in normal["state"]["history"]] == list(range(1, stop_epoch + 1))
    assert [item["epoch"] for item in resumed["state"]["history"]] == list(range(1, stop_epoch + 1))
    for name in normal["model"]:
        torch.testing.assert_close(normal["model"][name], resumed["model"][name], rtol=0, atol=0)
    for parameter, values in normal["optimizer"]["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, resumed["optimizer"]["state"][parameter][name], rtol=0, atol=0)
    if mode == "qualification_deadline":
        assert json.loads((continuous / "selector_failure_report.json").read_text()) == json.loads(
            (interrupted / "selector_failure_report.json").read_text())


def test_new_candidate_and_terminal_protocol_rejects_r2(config):
    assert PROFILE.protocol == config["experiment"]["protocol_version"]
    assert PROFILE.repair_revision == config["experiment"]["repair_revision"]
    assert CHECKPOINT_STATE_VERSION == 4
    old = copy.deepcopy(config)
    old["experiment"].update({"protocol_version": "statsfusion-joint-event-v7", "repair_revision": "v414-r2"})
    with pytest.raises(ValueError):
        validate_config(old)
