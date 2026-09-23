from __future__ import annotations

import torch

from bme_eating.models.hierarchical_state import (
    CausalTemporalBranch,
    HierarchicalStateModel,
)


def _config(use_stable: bool = False) -> dict[str, object]:
    return {
        "motion_block_seconds": 3,
        "ppg_block_seconds": 15,
        "motion_bucket_counts": [1, 2, 4, 8, 16, 32, 64],
        "ppg_bucket_counts": [1, 2, 4, 8, 16],
        "motion_dilations": [1, 2, 4, 8, 16, 32],
        "ppg_dilations": [1, 2, 4, 8],
        "local_embedding_dim": 16,
        "tcn_channels": 16,
        "state_embedding_dim": 32,
        "dropout": 0.0,
        "future_context_seconds": 0,
        "use_stable_state_features": use_stable,
        "stable_feature_columns": [f"feature_{index}" for index in range(12)],
    }


def _batch(include_stable: bool = False) -> dict[str, torch.Tensor]:
    batch = {
        "motion_blocks": torch.randn(1, 127, 12, 300),
        "motion_valid": torch.ones(1, 127),
        "ppg_blocks": torch.randn(1, 31, 2, 750),
        "ppg_quality": torch.rand(1, 31, 8),
        "ppg_valid": torch.ones(1, 31),
    }
    if include_stable:
        batch["stable_features"] = torch.randn(1, 12)
    return batch


def test_hierarchical_state_output_contract() -> None:
    model = HierarchicalStateModel(_config())
    with torch.inference_mode():
        output = model(_batch())
    assert output["state_logit"].shape == (1,)
    assert output["start_logit"].shape == (1,)
    assert output["end_logit"].shape == (1,)
    assert output["state_embedding"].shape == (1, 32)
    assert output["state_history_logit"].shape == (1, 127)
    assert output["ppg_gate"].shape == (1, 31)


def test_state_and_verifier_mode_accepts_stable_feature_token() -> None:
    model = HierarchicalStateModel(_config(use_stable=True))
    with torch.inference_mode():
        output = model(_batch(include_stable=True))
    assert torch.isfinite(output["state_logit"]).all()


def test_temporal_branch_is_strictly_causal() -> None:
    torch.manual_seed(1)
    branch = CausalTemporalBranch(4, 8, [1, 2, 4], 0.0).eval()
    original = torch.randn(2, 20, 4)
    changed = original.clone()
    changed[:, 11:] += 100.0
    with torch.inference_mode():
        before = branch(original)
        after = branch(changed)
    assert torch.allclose(before[:, :11], after[:, :11], atol=1e-6, rtol=0)


def test_ppg_all_missing_produces_finite_output() -> None:
    model = HierarchicalStateModel(_config()).eval()
    batch = _batch()
    batch["ppg_valid"].zero_()
    batch["ppg_blocks"].zero_()
    with torch.inference_mode():
        output = model(batch)
    assert torch.isfinite(output["state_embedding"]).all()
    assert torch.count_nonzero(output["ppg_gate"]) == 0
