import warnings

import numpy as np
import pandas as pd
import pytest
import torch

import bme_eating.data.deep_dataset as deep_dataset_module
from bme_eating.data.deep_dataset import DTPDataset, Normalization, _corrupt_ppg
from bme_eating.models.dtp_sqf import DTPSQF, DyadicPool, logits_to_probability_arrays
from bme_eating.reproducibility import epoch_random_seed, should_validate_epoch
from bme_eating.training.dtp_trainer import (
    checkpoint_selection_rank,
    resolve_focal_positive_alpha,
    select_checkpoint_validation_anchors,
)

MODEL_CONFIG = {
    "motion_block_seconds": 3,
    "ppg_block_seconds": 15,
    "motion_bucket_counts": [1, 2, 4, 8, 16, 32, 64],
    "ppg_bucket_counts": [1, 2, 4, 8, 16],
    "embedding_dim": 128,
    "token_dim": 192,
    "transformer_layers": 2,
    "attention_heads": 4,
    "feedforward_dim": 384,
    "dropout": 0.0,
    "future_context_seconds": 0,
}


def test_dyadic_pool_propagates_missing_block_validity():
    embedding = torch.tensor([[[1.0, 10.0], [999.0, 999.0], [3.0, 30.0]]])
    valid = torch.tensor([[True, False, True]])

    summary = DyadicPool.summarize(embedding, valid)

    torch.testing.assert_close(summary[:, 0:2], torch.tensor([[2.0, 20.0]]))
    torch.testing.assert_close(summary[:, 4:6], torch.tensor([[3.0, 30.0]]))
    torch.testing.assert_close(summary[:, 6:8], torch.tensor([[3.0, 30.0]]))
    torch.testing.assert_close(summary[:, -1], torch.tensor([2.0 / 3.0]))


def test_dtp_geometry_is_derived_from_configuration():
    config = {
        **MODEL_CONFIG,
        "motion_block_seconds": 2,
        "ppg_block_seconds": 10,
        "motion_bucket_counts": [1, 2],
        "ppg_bucket_counts": [1, 2],
    }

    model = DTPSQF(config)

    assert model.motion_bucket_durations == (2, 4)
    assert model.motion_bucket_ages == (0, 2)
    assert model.ppg_bucket_durations == (10, 20)
    assert model.ppg_bucket_ages == (0, 10)


def test_dtp_rejects_future_context_incompatible_with_block_geometry():
    config = {**MODEL_CONFIG, "future_context_seconds": 10}

    with pytest.raises(ValueError, match="divisible"):
        DTPSQF(config)


def test_dtp_dataset_shapes_follow_configured_history(tmp_path, monkeypatch):
    motion_time = np.arange(0, 40_001, 10, dtype=np.int64)
    ppg_time = np.arange(0, 40_001, 20, dtype=np.int64)
    segment_path = tmp_path / "segment.npz"
    np.savez_compressed(
        segment_path,
        motion_timestamp_ms=motion_time,
        motion_values=np.zeros((len(motion_time), 6), dtype=np.float32),
        motion_mask=np.ones((len(motion_time), 6), dtype=bool),
        ppg_timestamp_ms=ppg_time,
        ppg_values=np.zeros((len(ppg_time), 1), dtype=np.float32),
        ppg_mask=np.ones((len(ppg_time), 1), dtype=bool),
    )
    segments = pd.DataFrame(
        [
            {
                "session_id": "session",
                "segment_id": "segment",
                "segment_path": str(segment_path),
                "start_ms": 0,
                "end_ms": 40_000,
            }
        ]
    )
    anchors = pd.DataFrame(
        [
            {
                "session_id": "session",
                "segment_id": "segment",
                "subject_key": "subject",
                "timestamp_ms": 40_000,
                "state_target": 0.0,
                "state_loss_mask": 1.0,
                "start_target": 0.0,
                "end_target": 0.0,
                "start_loss_mask": 1.0,
                "end_loss_mask": 1.0,
            }
        ]
    )
    normalization = Normalization(np.zeros(6), np.ones(6), 0.0, 1.0)
    dataset = DTPDataset(
        anchors,
        segments,
        normalization,
        motion_block_seconds=2,
        ppg_block_seconds=10,
        motion_bucket_counts=[1, 2],
        ppg_bucket_counts=[1, 2],
    )

    item = dataset[0]

    assert item["motion_blocks"].shape == (3, 12, 200)
    assert item["ppg_blocks"].shape == (3, 2, 500)

    def deterministic_corruption(values, mask, rng):
        return (values + rng.random()).astype(np.float32), mask

    monkeypatch.setattr(deep_dataset_module, "_corrupt_ppg", deterministic_corruption)
    training_dataset = DTPDataset(
        anchors,
        segments,
        normalization,
        training=True,
        ppg_augmentation_probability=1.0,
        seed=2026,
        motion_block_seconds=2,
        ppg_block_seconds=10,
        motion_bucket_counts=[1, 2],
        ppg_bucket_counts=[1, 2],
    )
    epoch_two_first = training_dataset[(0, 2)]["ppg_blocks"]
    epoch_two_second = training_dataset[(0, 2)]["ppg_blocks"]
    epoch_three = training_dataset[(0, 3)]["ppg_blocks"]

    torch.testing.assert_close(epoch_two_first, epoch_two_second)
    assert not torch.equal(epoch_two_first, epoch_three)


def test_ppg_corruption_keeps_fully_missing_blocks_finite_without_warnings():
    values = np.zeros(750, dtype=np.float32)
    mask = np.zeros(750, dtype=bool)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outputs = [
            _corrupt_ppg(values, mask, np.random.default_rng(seed))
            for seed in range(60)
        ]

    assert not caught
    assert all(np.isfinite(corrupted).all() for corrupted, _ in outputs)
    assert all(not corrupted_mask.any() for _, corrupted_mask in outputs)
    assert all(not corrupted.any() for corrupted, _ in outputs)


def test_checkpoint_validation_subset_is_deterministic_stratified_and_evaluable():
    anchors = pd.DataFrame(
        {
            "timestamp_ms": np.arange(1000),
            "state_target": [1.0] * 100 + [0.0] * 900,
            "state_loss_mask": [1.0] * 950 + [0.0] * 50,
        }
    )

    first = select_checkpoint_validation_anchors(anchors, maximum_rows=100, seed=2026)
    second = select_checkpoint_validation_anchors(anchors, maximum_rows=100, seed=2026)
    different_seed = select_checkpoint_validation_anchors(
        anchors, maximum_rows=100, seed=2027
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 100
    assert first["state_loss_mask"].eq(1.0).all()
    assert first["state_target"].sum() == 11.0
    assert not first.equals(different_seed)


def test_epoch_randomness_and_validation_schedule_are_resume_stable():
    assert epoch_random_seed(2026, 0, 4) == epoch_random_seed(2026, 0, 4)
    assert epoch_random_seed(2026, 0, 4) != epoch_random_seed(2026, 0, 5)
    assert epoch_random_seed(2026, 0, 4) != epoch_random_seed(2026, 1, 4)
    assert [epoch for epoch in range(6) if should_validate_epoch(epoch, 2)] == [0, 1, 3, 5]


def test_balanced_sampling_requires_neutral_focal_alpha():
    assert (
        resolve_focal_positive_alpha(
            {"positive_sampling_fraction": 0.5, "focal_positive_alpha": 0.5}
        )
        == 0.5
    )
    with pytest.raises(ValueError, match="double class compensation"):
        resolve_focal_positive_alpha(
            {"positive_sampling_fraction": 0.5, "focal_positive_alpha": 0.94}
        )


def test_crossfit_checkpoint_selection_uses_dtp_only_auprc():
    low_f1_high_auprc = checkpoint_selection_rank(
        "window_auprc",
        0.60,
        {"f1": 0.1, "start_mae_seconds": 100.0, "end_mae_seconds": 100.0},
        4,
    )
    high_f1_low_auprc = checkpoint_selection_rank(
        "window_auprc",
        0.55,
        {"f1": 0.9, "start_mae_seconds": 1.0, "end_mae_seconds": 1.0},
        2,
    )

    assert low_f1_high_auprc > high_f1_low_auprc


def test_dtp_output_shapes():
    model = DTPSQF(MODEL_CONFIG).eval()
    batch = {
        "motion_blocks": torch.randn(1, 127, 12, 300),
        "motion_valid": torch.ones(1, 127),
        "ppg_blocks": torch.randn(1, 31, 2, 750),
        "ppg_quality": torch.rand(1, 31, 8),
        "ppg_valid": torch.ones(1, 31),
    }
    with torch.inference_mode():
        output = model(batch)
    assert output["state_logit"].shape == (1,)
    assert output["start_logit"].shape == (1,)
    assert output["end_logit"].shape == (1,)
    assert output["ppg_gate"].shape == (1, 31)


def test_causal_model_ignores_unprovided_future():
    model = DTPSQF(MODEL_CONFIG).eval()
    batch = {
        "motion_blocks": torch.randn(1, 127, 12, 300),
        "motion_valid": torch.ones(1, 127),
        "ppg_blocks": torch.randn(1, 31, 2, 750),
        "ppg_quality": torch.rand(1, 31, 8),
        "ppg_valid": torch.ones(1, 31),
    }
    with torch.inference_mode():
        first = model(batch)["state_logit"]
        second = model({key: value.clone() for key, value in batch.items()})["state_logit"]
    torch.testing.assert_close(first, second)


def test_bfloat16_logits_are_converted_before_numpy():
    logits = torch.zeros(2, dtype=torch.bfloat16)
    state, start, end = logits_to_probability_arrays(
        {"state_logit": logits, "start_logit": logits, "end_logit": logits}
    )

    assert state.dtype == np.float32
    assert state.tolist() == [0.5, 0.5]
    assert start.tolist() == [0.5, 0.5]
    assert end.tolist() == [0.5, 0.5]
