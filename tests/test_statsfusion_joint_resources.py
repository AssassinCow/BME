from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from bme_eating.config import load_config
from bme_eating.data.joint_sessions import JointInputs, SessionChunkSampler, session_loader
from bme_eating.data.stats_fusion_inputs import canonical_input_paths
from bme_eating.hierarchical_artifacts import sha256_file
from bme_eating.joint_protocol import require_idle_gpu
from bme_eating.joint_validation import validate_joint_implementation
from bme_eating.stats_features import STATS_FEATURE_COLUMNS
from bme_eating.training.statsfusion_joint_trainer import seed_model, train_joint_model

RESOURCE_ENABLED = os.environ.get("BME_V410_RESOURCE_TESTS") == "1"


class ResourceProbeDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.session_groups = {("probe", "session"): pd.DataFrame({"_row_id": np.arange(16)})}
        self.geometry = SimpleNamespace(supervised_steps=4)

    def __len__(self):
        return 16

    def __getitem__(self, key):
        row, epoch, _ = key
        return {"row": torch.tensor(row), "epoch": torch.tensor(epoch)}


@pytest.mark.skipif(not RESOURCE_ENABLED, reason="Resource probe waits for current training to finish")
def test_real_ten_worker_chunk_cursor_recovery():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v410_joint_event.yaml")
    dataset = ResourceProbeDataset()
    sampler = SessionChunkSampler(dataset, 1)
    iterator = iter(session_loader(dataset, sampler, config))
    first = [int(next(iterator)["row"]) for _ in range(2)]
    del iterator
    gc.collect()
    resumed = [int(batch["row"]) for batch in session_loader(
        dataset, SessionChunkSampler(dataset, 1, start_chunk=2), config)]
    assert first + resumed == [3, 7, 11, 15]


@pytest.mark.skipif(not RESOURCE_ENABLED or not torch.cuda.is_available(), reason="GPU resource probe deferred")
def test_gpu_full_receptive_field_memory_and_backward():
    require_idle_gpu("cuda")
    config = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v410_joint_event.yaml")
    torch.cuda.reset_peak_memory_stats()
    report = validate_joint_implementation(config, device="cuda")
    assert report["peak_gpu_gb"] < 10.5
    assert report["repeat_max_error"] <= 1e-6


@pytest.mark.skipif(not RESOURCE_ENABLED or not torch.cuda.is_available(), reason="GPU data pipeline probe deferred")
def test_native_archive_joint_training_bfloat16_and_safe_checkpoint(tmp_path):
    require_idle_gpu("cuda")
    config = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v410_joint_event.yaml")
    config["training"]["num_workers"] = 1
    config["training"]["gradient_diagnostics_every_chunks"] = 1
    config["training"]["gradient_clip_norm"] = 1000
    config["training"]["ppg_modality_dropout"] = 0
    config["training"]["gyro_modality_dropout"] = 0
    config["training"]["rotation_augmentation_probability"] = 0
    input_root, canonical_root = tmp_path / "input", tmp_path / "canonical"
    paths = canonical_input_paths(canonical_root)
    paths["root"].mkdir(parents=True)
    (input_root / "indices").mkdir(parents=True)
    archive = input_root / "session.npz"
    generator = np.random.default_rng(2026)
    motion_times = np.arange(0, 120001, 10, dtype=np.int64)
    ppg_times = np.arange(0, 120001, 20, dtype=np.int64)
    motion_mask = np.ones((len(motion_times), 6), dtype=bool)
    motion_mask[6000:, 3:] = False
    np.savez(archive, motion_timestamp_ms=motion_times,
        motion_values=generator.normal(size=(len(motion_times), 6)).astype(np.float32), motion_mask=motion_mask,
        ppg_timestamp_ms=ppg_times, ppg_values=generator.normal(size=len(ppg_times)).astype(np.float32),
        ppg_mask=np.ones(len(ppg_times), bool))
    times = np.arange(3000, 120001, 3000, dtype=np.int64)
    anchors = pd.DataFrame({"subject_key": "fit", "session_id": "session", "segment_id": "segment",
        "segment_path": str(archive), "timestamp_ms": times, "state_target": ((times > 9000) & (times <= 39000)).astype(float),
        "start_target": 0., "end_target": 0., "state_loss_mask": 1., "start_loss_mask": 1.,
        "end_loss_mask": 1., "smooth_loss_mask": 1., "motion_history_available_seconds": times / 1000,
        "ppg_history_available_seconds": times / 1000})
    statistics = anchors[["subject_key", "session_id", "timestamp_ms"]].copy()
    for column in STATS_FEATURE_COLUMNS:
        statistics[column] = generator.normal(size=len(times))
    events = pd.DataFrame({"subject_key": ["fit"], "session_id": ["session"], "event_id": ["truth"],
        "source_event_key": ["truth"], "start_ms": [9000], "end_ms": [39000], "valid_duration": [True],
        "evaluable": [True], "hand_relation": ["unknown"]})
    segments = pd.DataFrame({"subject_key": ["fit"], "session_id": ["session"], "segment_id": ["segment"],
        "segment_path": [str(archive)], "start_ms": [0], "end_ms": [120000]})
    for name, frame in (("anchors", anchors), ("statistics", statistics), ("events", events)):
        frame.to_parquet(paths[name], index=False)
    segments.to_parquet(input_root / "indices" / "segments.parquet", index=False)
    (input_root / "indices" / "subject_folds.json").write_text('{"fit": 0}', encoding="utf-8")
    inputs = JointInputs(input_root, canonical_root, {"fit": 0},
                         {name: sha256_file(paths[name]) for name in ("anchors", "statistics", "events")})
    initial = seed_model(config).encoder.short_fusion[0].weight.detach().cpu().clone()
    directory = tmp_path / "training_probe"
    train_joint_model(inputs, {"fit"}, config, directory, "native_pipeline_probe", epochs=3, resume=False)
    payload = torch.load(directory / "latest.pt", map_location="cpu", weights_only=True)
    assert not torch.equal(initial, payload["model"]["encoder.short_fusion.0.weight"])
    assert all(torch.isfinite(value).all() for value in payload["model"].values())
    gradients = json.loads((directory / "gradients.json").read_text(encoding="utf-8"))
    assert any(row["event_weighted_norm"] > 0 and row["quality_weighted_norm"] > 0
               for row in gradients if row["epoch"] == 3)
    sampling = json.loads((directory / "sampling.json").read_text(encoding="utf-8"))
    assert any(row["truth_jitter_candidates"] > 0 for row in sampling if row["epoch"] == 3)
    assert train_joint_model(inputs, {"fit"}, config, directory, "native_pipeline_probe", epochs=3, resume=True) == 3
    assert torch.cuda.max_memory_reserved() / 1024 ** 3 < 10.5
