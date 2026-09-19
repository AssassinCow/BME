import numpy as np
import pandas as pd
import pytest
import torch

from bme_eating.data.deep_dataset import DTPDataset, Normalization
from bme_eating.models.dtp_sqf import DTPSQF, DyadicPool


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


def test_dtp_dataset_shapes_follow_configured_history(tmp_path):
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
