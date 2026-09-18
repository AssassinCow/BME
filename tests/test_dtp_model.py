import torch

from bme_eating.models.dtp_sqf import DTPSQF


MODEL_CONFIG = {
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

