from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from bme_eating.config import load_config
from bme_eating.data.joint_sessions import SessionChunkSampler, reuse_session_loader, session_loader
from bme_eating.training import statsfusion_joint_trainer as trainer


def legacy_proposal_supervision(batch, cpu_batch, valid, scale_valid, factor):
    target = batch["state_target"].float()
    scale_target, scale_mask = torch.zeros_like(target), torch.zeros_like(valid)
    for endpoint in range(factor - 1, target.shape[1]):
        if int((batch["timeline_position"][0, endpoint] + 1) % factor) != 0:
            continue
        members = slice(endpoint - factor + 1, endpoint + 1)
        labels_valid = batch["state_loss_mask"][:, members]
        scale_target[:, endpoint] = ((target[:, members] * labels_valid).sum(1)
                                     / labels_valid.sum(1).clamp_min(1))
        scale_mask[:, endpoint] = valid[:, endpoint] * (scale_valid[:, endpoint] > 0) * (labels_valid.sum(1) > 0)
    return scale_target, scale_mask


@pytest.mark.parametrize("factor", [1, 5, 20])
@pytest.mark.parametrize("kind", ["binary", "soft", "missing"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_proposal_supervision_bitwise_equivalent(factor, kind, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator().manual_seed(2026)
    target = torch.randint(0, 2, (2, 83), generator=generator).float()
    weights = torch.randint(0, 2, (2, 83), generator=generator).float()
    if kind == "soft":
        target[:, 12:29] = torch.rand(2, 17, generator=generator)
        weights[:, 24:41] = torch.rand(2, 17, generator=generator)
    if kind == "missing":
        weights.zero_()
    cpu_batch = {"state_target": target, "state_loss_mask": weights,
                 "timeline_position": torch.arange(-39, 44)[None].expand(2, -1)}
    batch = {name: value.to(device) for name, value in cpu_batch.items()}
    valid = torch.rand(2, 83, generator=generator).to(device)
    scale_valid = torch.randint(0, 2, (2, 83), generator=generator).float().to(device)
    expected = legacy_proposal_supervision(batch, cpu_batch, valid, scale_valid, factor)
    actual = trainer.proposal_supervision(batch, cpu_batch, valid, scale_valid, factor)
    for baseline, optimized in zip(expected, actual):
        torch.testing.assert_close(baseline, optimized, rtol=0, atol=0)


def test_proposal_shorter_than_scale_and_finite_gate():
    batch = {"state_target": torch.ones(1, 3), "state_loss_mask": torch.ones(1, 3),
             "timeline_position": torch.arange(3)[None]}
    result = trainer.proposal_supervision(batch, batch, torch.ones(1, 3), torch.ones(1, 3), 20)
    assert all(not value.any() for value in result)
    assert trainer.tensors_finite([torch.ones(1), torch.ones(2, dtype=torch.long)])
    assert trainer.tensors_finite([])
    assert not trainer.tensors_finite([torch.ones(1), torch.tensor([float("nan")])])
    assert not trainer.tensors_finite([torch.tensor([float("inf")])])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_checkpoint_snapshot_values_alias_and_optimizer_roundtrip(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    values = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4).transpose(0, 1)
    payload = {"model": {"weight": values, "alias": values,
                         "indices": torch.arange(4, device=device), "empty": torch.empty(0, device=device)},
               "optimizer": {"state": {0: {"exp_avg": values.clone(), "step": torch.tensor(8.)}},
                             "param_groups": [{"params": [0], "lr": .0001}]},
               "gradients": {"weight": values.clone(), "unused": None},
               "state": {"epoch": 8, "next_chunk": 1481, "history": ["preserved"]},
               "torch_rng": torch.get_rng_state()}
    actual = trainer.cpu_checkpoint_snapshot(payload)
    assert actual["state"] is payload["state"]
    assert actual["model"]["weight"] is actual["model"]["alias"]
    for name in payload["model"]:
        assert actual["model"][name].device.type == "cpu"
        torch.testing.assert_close(actual["model"][name], payload["model"][name].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(actual["optimizer"]["state"][0]["exp_avg"], values.cpu(), rtol=0, atol=0)
    buffer = io.BytesIO()
    torch.save(actual, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)
    torch.testing.assert_close(restored["gradients"]["weight"], values.cpu(), rtol=0, atol=0)
    assert restored["gradients"]["unused"] is None
    assert restored["state"] == payload["state"]
    assert payload["model"]["weight"].device.type == device


def test_packed_training_state_loads_as_original_mapping_with_rng_and_numpy():
    state = {"epoch": 8, "next_chunk": 1481, "sample_audit": [{"category": "positive", "score": np.float64(.37)}],
             "numpy": np.random.default_rng(2026).bit_generator.state, "history": np.arange(12),
             "partial": {"session": ("subject", "session"), "families": {"first", "second"}}}
    payload = {"model": {"weight": torch.ones(1)}, "optimizer": {}, "gradients": {"weight": torch.ones(1)},
               "state": state, "torch_rng": torch.get_rng_state()}
    snapshot = trainer.training_checkpoint_snapshot(payload)
    buffer = io.BytesIO()
    torch.save(snapshot, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    assert isinstance(restored["state"], dict)
    assert restored["state"]["epoch"] == 8 and restored["state"]["next_chunk"] == 1481
    assert restored["state"]["sample_audit"] == state["sample_audit"]
    assert restored["state"]["numpy"] == state["numpy"]
    np.testing.assert_array_equal(restored["state"]["history"], state["history"])
    assert restored["state"]["partial"] == state["partial"]
    torch.testing.assert_close(restored["torch_rng"], payload["torch_rng"], rtol=0, atol=0)


class EpochProbeDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.session_groups = {("subject", "session"): pd.DataFrame({"_row_id": np.arange(16)})}
        self.geometry = SimpleNamespace(supervised_steps=4)

    def __getitem__(self, index):
        row, epoch, _ = index
        generator = np.random.default_rng(np.random.SeedSequence([2026, row, epoch]))
        return {"row": torch.tensor(row), "epoch": torch.tensor(epoch),
                "values": torch.tensor(generator.normal(size=4))}

    def __len__(self):
        return 16


def test_persistent_loader_epoch_rng_order_and_partition_exclusion():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v414_full_data.yaml")
    config["training"]["num_workers"] = 2
    config["training"]["device"] = "cpu"
    dataset = EpochProbeDataset()
    loader = session_loader(dataset, SessionChunkSampler(dataset, 8), config)
    list(loader)
    worker_ids = [worker.pid for worker in loader._iterator._workers]
    next_sampler = SessionChunkSampler(dataset, 9, start_chunk=1)
    reused = reuse_session_loader(loader, dataset, next_sampler, config)
    actual = list(reused)
    assert [worker.pid for worker in reused._iterator._workers] == worker_ids
    expected = list(session_loader(dataset, next_sampler, config))
    for baseline, optimized in zip(expected, actual):
        for name in baseline:
            torch.testing.assert_close(baseline[name], optimized[name], rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="cannot cross dataset"):
        reuse_session_loader(loader, EpochProbeDataset(), next_sampler, config)


