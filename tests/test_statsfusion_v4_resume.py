from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

import bme_eating.training.hierarchical_v4_trainer as trainer


class ToyDataset(Dataset):
    def __init__(self) -> None:
        self.anchors = pd.DataFrame({"subject_key": ["fit"] * 6})
        self.geometry = SimpleNamespace(supervised_steps=6)

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.tensor([index / 6], dtype=torch.float32),
            "target": torch.tensor([1 - index / 6], dtype=torch.float32),
        }


class ToySampler:
    def __init__(self, _anchors, *, samples_per_epoch, **_kwargs) -> None:
        self.count = samples_per_epoch
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        return iter((index + self.epoch) % 6 for index in range(self.count))


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, 8), nn.Dropout(0.3), nn.Linear(8, 1))

    def forward(self, batch):
        return self.network(batch["x"])


class ToyLoss:
    def __call__(self, output, batch):
        return torch.nn.functional.mse_loss(output, batch["target"]), {}


def _config() -> dict:
    return {
        "training": {
            "device": "cpu",
            "batch_size": 2,
            "steps_per_epoch": 3,
            "num_workers": 0,
            "gradient_accumulation": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "warmup_fraction": 0.1,
            "max_epochs": 4,
            "validation_every_epochs": 2,
            "selector_rolling_epochs": 2,
            "early_stopping_min_epochs": 4,
            "early_stopping_patience_checks": 10,
            "early_stopping_min_delta": 0.003,
        },
        "sequence": {"sampling_mixture": {"uniform": 1, "event": 0, "boundary": 0}},
        "promotion_gate": {"minimum_candidate_recall": 0.5},
        "model": {"hidden_dim": 8},
    }


def _patch_toy_training(monkeypatch) -> None:
    monkeypatch.setattr(trainer, "ClipMixtureSampler", ToySampler)
    monkeypatch.setattr(trainer, "_state_loss", lambda _config: ToyLoss())


def _weights(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint, {key: value.clone() for key, value in checkpoint["model"].items()}


def test_retraining_resume_matches_uninterrupted_and_checks_identity(tmp_path, monkeypatch):
    _patch_toy_training(monkeypatch)
    config = _config()
    subjects = {"train": {"fit"}, "holdout": {"test"}}
    dataset = ToyDataset()
    continuous_path = tmp_path / "continuous.pt"
    interrupted_path = tmp_path / "interrupted.pt"

    torch.manual_seed(45)
    continuous = ToyModel()
    trainer._train_state_with_checkpoints(
        continuous, dataset, config, epochs=4, seed=45, subjects=subjects,
        checkpoint_path=continuous_path, resume=False, progress_label="continuous",
    )
    assert len(_weights(continuous_path)[0]["training_metrics"]) == 4
    assert _weights(continuous_path)[0]["training_metrics"][0]["optimizer_updates"] == 2

    original = trainer._train_state_epochs

    def interrupt(*args, **kwargs):
        if kwargs["epoch_offset"] == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    torch.manual_seed(45)
    interrupted = ToyModel()
    monkeypatch.setattr(trainer, "_train_state_epochs", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        trainer._train_state_with_checkpoints(
            interrupted, dataset, config, epochs=4, seed=45, subjects=subjects,
            checkpoint_path=interrupted_path, resume=False, progress_label="interrupted",
        )
    monkeypatch.setattr(trainer, "_train_state_epochs", original)
    assert _weights(interrupted_path)[0]["epoch"] == 2

    changed = deepcopy(config)
    changed["training"]["learning_rate"] = 0.02
    with pytest.raises(RuntimeError, match="identity mismatch"):
        trainer._train_state_with_checkpoints(
            ToyModel(), dataset, changed, epochs=4, seed=45, subjects=subjects,
            checkpoint_path=interrupted_path, resume=True, progress_label="changed",
        )

    torch.manual_seed(99)
    resumed = ToyModel()
    trainer._train_state_with_checkpoints(
        resumed, dataset, config, epochs=4, seed=45, subjects=subjects,
        checkpoint_path=interrupted_path, resume=True, progress_label="resumed",
    )
    assert _weights(interrupted_path)[0]["epoch"] == 4
    for name, value in _weights(continuous_path)[1].items():
        assert torch.equal(value, _weights(interrupted_path)[1][name])

    runtime_changed = deepcopy(config)
    runtime_changed["training"]["inference_num_workers"] = 2

    def unexpected_training(*_args, **_kwargs):
        raise AssertionError("A completed checkpoint must not repeat training")

    monkeypatch.setattr(trainer, "_train_state_epochs", unexpected_training)
    trainer._train_state_with_checkpoints(
        ToyModel(), dataset, runtime_changed, epochs=4, seed=45, subjects=subjects,
        checkpoint_path=interrupted_path, resume=True, progress_label="completed",
    )


def test_selector_resume_preserves_epoch_metrics(tmp_path, monkeypatch):
    _patch_toy_training(monkeypatch)
    config = _config()
    anchors = pd.DataFrame({"subject_key": ["fit"] * 6 + ["selector"] * 6})
    events = pd.DataFrame({"subject_key": ["fit", "selector"]})
    inputs = trainer.V4Inputs(anchors, pd.DataFrame(), events, pd.DataFrame(), {})
    monkeypatch.setattr(trainer, "_fit_scaler_and_transform", lambda *_args: (None, anchors))
    monkeypatch.setattr(trainer, "compute_normalization", lambda *_args: None)
    monkeypatch.setattr(trainer, "_make_dataset", lambda *_args, **_kwargs: ToyDataset())

    def seeded(_config, seed):
        torch.manual_seed(seed)
        return ToyModel()

    monkeypatch.setattr(trainer, "_build_seeded_state_model", seeded)
    monkeypatch.setattr(
        trainer,
        "_selector_score",
        lambda model, *_args, **_kwargs: {
            "candidate_recall": 0.8,
            "event_f1": float(model.network[0].weight.detach().abs().mean()),
            "state_fragment_count": 1,
            "ece": 0.02,
            "window_auprc": 0.4,
            "calibration_passed": True,
        },
    )
    fit = anchors[anchors["subject_key"] == "fit"]
    selector = anchors[anchors["subject_key"] == "selector"]
    continuous_path = tmp_path / "selector_continuous.pt"
    interrupted_path = tmp_path / "selector_interrupted.pt"
    selected, report = trainer._select_epoch(
        fit, selector, {"fit"}, inputs, config, 46,
        checkpoint_path=continuous_path,
    )

    original = trainer._train_state_epochs

    def interrupt(*args, **kwargs):
        if kwargs["epoch_offset"] == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer, "_train_state_epochs", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        trainer._select_epoch(
            fit, selector, {"fit"}, inputs, config, 46,
            checkpoint_path=interrupted_path,
        )
    monkeypatch.setattr(trainer, "_train_state_epochs", original)
    assert _weights(interrupted_path)[0]["epoch"] == 2
    resumed_selected, resumed_report = trainer._select_epoch(
        fit, selector, {"fit"}, inputs, config, 46,
        checkpoint_path=interrupted_path, resume=True,
    )
    assert resumed_selected == selected
    assert resumed_report == report
    assert len(resumed_report["training_metrics"]) == 4
    assert set(resumed_report["training_metrics"][0]) >= {
        "train_loss_mean",
        "state_loss_mean",
        "onset_loss_mean",
        "offset_loss_mean",
        "smooth_loss_mean",
        "learning_rate_last",
        "gradient_norm_mean",
    }
    for name, value in _weights(continuous_path)[1].items():
        assert torch.equal(value, _weights(interrupted_path)[1][name])


def _interrupt_after_second_head_epoch(monkeypatch):
    original_save = trainer._save_head_epoch

    def interrupt(*args, **kwargs):
        original_save(*args, **kwargs)
        if kwargs["epoch"] == 2:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(trainer, "_save_head_epoch", interrupt)
    return original_save


def _assert_same_head_weights(first, second):
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


def _verifier_inputs():
    features = trainer.ProposalFeatureBatchV4(
        proposal_ids=np.array([f"p{index}" for index in range(8)]),
        sequence=np.arange(8 * 4 * 2, dtype=np.float32).reshape(8, 4, 2) / 64,
        sequence_mask=np.ones((8, 4), dtype=bool),
        scalar=np.ones((8, 2), dtype=np.float32),
        event_target=np.array([0, 1] * 4, dtype=np.float32),
        iou_target=np.array([0, 0.6] * 4, dtype=np.float32),
        sample_weight=np.ones(8, dtype=np.float32),
    )
    categories = np.array(["random_background", "positive"] * 4)
    config = _config()
    config["verifier"] = {
        "batch_size": 4,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "batch_composition": {
            "positive": 0.5,
            "near_miss": 0.0,
            "hard_false_positive": 0.0,
            "random_background": 0.5,
        },
        "iou_loss_weight": 0.5,
        "max_epochs": 4,
        "patience": 10,
        "hidden_channels": 8,
        "hidden_dim": 16,
        "dropout": 0.2,
    }
    return features, categories, config


def test_verifier_retrain_resume_matches_uninterrupted(tmp_path, monkeypatch):
    features, categories, config = _verifier_inputs()
    identity = {"training_subjects": ["fit"], "parent_artifact_sha256": {"input": "abc"}}
    continuous = trainer._train_verifier_model_v4(
        features, categories, config, seed=52, epochs=4,
        checkpoint_path=tmp_path / "continuous.pt", identity=identity,
    )
    original_save = _interrupt_after_second_head_epoch(monkeypatch)
    path = tmp_path / "interrupted.pt"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        trainer._train_verifier_model_v4(
            features, categories, config, seed=52, epochs=4,
            checkpoint_path=path, identity=identity,
        )
    monkeypatch.setattr(trainer, "_save_head_epoch", original_save)
    resumed = trainer._train_verifier_model_v4(
        features, categories, config, seed=52, epochs=4,
        checkpoint_path=path, resume=True, identity=identity,
    )
    _assert_same_head_weights(continuous, resumed)
    assert torch.load(path, weights_only=False)["epoch"] == 4
    with pytest.raises(RuntimeError, match="identity mismatch"):
        trainer._train_verifier_model_v4(
            features, categories, config, seed=52, epochs=4,
            checkpoint_path=path, resume=True, identity={"training_subjects": ["other"]},
        )


def test_verifier_selector_resume_preserves_history(tmp_path, monkeypatch):
    features, categories, config = _verifier_inputs()
    proposals = pd.DataFrame({"proposal_id": features.proposal_ids})
    monkeypatch.setattr(
        trainer,
        "_best_verifier_operating_point",
        lambda scored, *_args: {
            "f1": float(scored["final_score"].mean()), "fp_per_hour": 0.0,
        },
    )
    arguments = (
        features, categories, features, proposals, pd.DataFrame(),
        pd.DataFrame(), pd.DataFrame(), config,
    )
    selected, history = trainer._select_verifier_epoch(
        *arguments, seed=53, checkpoint_path=tmp_path / "continuous.pt", identity={"fit": ["s"]},
    )
    original_save = _interrupt_after_second_head_epoch(monkeypatch)
    path = tmp_path / "interrupted.pt"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        trainer._select_verifier_epoch(
            *arguments, seed=53, checkpoint_path=path, identity={"fit": ["s"]},
        )
    monkeypatch.setattr(trainer, "_save_head_epoch", original_save)
    resumed_selected, resumed_history = trainer._select_verifier_epoch(
        *arguments, seed=53, checkpoint_path=path, resume=True, identity={"fit": ["s"]},
    )
    assert (resumed_selected, resumed_history) == (selected, history)


def _boundary_inputs():
    positions = np.arange(8 * 5 * 2, dtype=np.float32).reshape(8, 5, 2) / 80
    target = np.zeros((8, 5), dtype=np.float32)
    target[:, 2] = 1
    features = SimpleNamespace(
        sample_ids=np.array([f"e{index}" for index in range(8)]),
        start_sequence=positions,
        end_sequence=positions.copy(),
        start_mask=np.ones((8, 5), dtype=bool),
        end_mask=np.ones((8, 5), dtype=bool),
        start_target=target,
        end_target=target.copy(),
        sample_weight=np.ones(8, dtype=np.float32),
        start_weight=np.ones(8, dtype=np.float32),
        end_weight=np.ones(8, dtype=np.float32),
        start_offsets_seconds=np.arange(-2, 3, dtype=np.float32),
        end_offsets_seconds=np.arange(-2, 3, dtype=np.float32),
    )
    config = _config()
    config["boundary"] = {
        "batch_size": 4, "learning_rate": 0.01, "weight_decay": 0.0,
        "max_epochs": 4, "patience": 10, "hidden_dim": 8, "dropout": 0.2,
        "local_softargmax_radius_bins": 1,
    }
    return features, config


def test_boundary_retrain_resume_matches_uninterrupted(tmp_path, monkeypatch):
    features, config = _boundary_inputs()
    identity = {"training_subjects": ["fit"], "range": {"seconds": 60}}
    continuous = trainer._train_endpoint_model(
        features, config, seed=54, epochs=4,
        checkpoint_path=tmp_path / "continuous.pt", identity=identity,
    )
    original_save = _interrupt_after_second_head_epoch(monkeypatch)
    path = tmp_path / "interrupted.pt"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        trainer._train_endpoint_model(
            features, config, seed=54, epochs=4, checkpoint_path=path, identity=identity,
        )
    monkeypatch.setattr(trainer, "_save_head_epoch", original_save)
    resumed = trainer._train_endpoint_model(
        features, config, seed=54, epochs=4,
        checkpoint_path=path, resume=True, identity=identity,
    )
    _assert_same_head_weights(continuous, resumed)
    assert torch.load(path, weights_only=False)["numpy_rng_state"] is not None


def test_boundary_selector_resume_preserves_history(tmp_path, monkeypatch):
    features, config = _boundary_inputs()
    arguments = (features, features, config)
    selected, history = trainer._select_boundary_epoch(
        *arguments, seed=55, checkpoint_path=tmp_path / "continuous.pt", identity={"fit": ["s"]},
    )
    original_save = _interrupt_after_second_head_epoch(monkeypatch)
    path = tmp_path / "interrupted.pt"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        trainer._select_boundary_epoch(
            *arguments, seed=55, checkpoint_path=path, identity={"fit": ["s"]},
        )
    monkeypatch.setattr(trainer, "_save_head_epoch", original_save)
    resumed_selected, resumed_history = trainer._select_boundary_epoch(
        *arguments, seed=55, checkpoint_path=path, resume=True, identity={"fit": ["s"]},
    )
    assert (resumed_selected, resumed_history) == (selected, history)
