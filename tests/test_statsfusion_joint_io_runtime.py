from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bme_eating.config import load_config
from bme_eating.data import deep_dataset
from bme_eating.data.deep_dataset import Normalization
from bme_eating.data.joint_sessions import JointSessionDataset, SessionChunkSampler, session_loader
from bme_eating.data.session import SessionWindowReader
from bme_eating.data.stats_fusion_sequence import SequenceGeometry
from bme_eating.joint_protocol import config_identity, initialize_run
from bme_eating.stats_features import STATS_FEATURE_COLUMNS

STATISTICS_COLUMNS = [*(f"stat_{name}" for name in STATS_FEATURE_COLUMNS),
                      *(f"stat_{name}_missing" for name in STATS_FEATURE_COLUMNS)]


def native_session_frames(directory):
    generator = np.random.default_rng(2026)
    segment_rows, anchor_frames, event_rows = [], [], []
    for subject in ("first", "second"):
        for segment_index, (start, end) in enumerate(((0, 60_000), (60_010, 120_000))):
            archive = directory / f"{subject}_{segment_index}.npz"
            motion_times = np.arange(start, end + 1, 10, dtype=np.int64)
            ppg_times = np.arange(start, end + 1, 20, dtype=np.int64)
            motion_mask = np.ones((len(motion_times), 6), dtype=bool)
            motion_mask[::7, 3:] = False
            np.savez_compressed(archive, motion_timestamp_ms=motion_times,
                motion_values=generator.normal(size=(len(motion_times), 6)).astype(np.float32),
                motion_mask=motion_mask, ppg_timestamp_ms=ppg_times,
                ppg_values=generator.normal(size=len(ppg_times)).astype(np.float32),
                ppg_mask=np.arange(len(ppg_times)) % 13 != 0)
            segment_rows.append({"subject_key": subject, "session_id": "session",
                "segment_id": f"{subject}_{segment_index}", "segment_path": str(archive),
                "start_ms": start, "end_ms": end})
        times = np.arange(3000, 120_001, 3000, dtype=np.int64)
        frame = pd.DataFrame({"subject_key": subject, "session_id": "session",
            "timestamp_ms": times, "state_target": ((times > 12_000) & (times <= 45_000)).astype(float),
            "state_loss_mask": 1.})
        for name in STATS_FEATURE_COLUMNS:
            frame[f"stat_{name}"] = generator.normal(size=len(times))
            frame[f"stat_{name}_missing"] = 0.
        anchor_frames.append(frame)
        event_rows.append({"subject_key": subject, "session_id": "session", "start_ms": 12_000,
                           "end_ms": 45_000, "valid_duration": True, "evaluable": True})
    return pd.concat(anchor_frames, ignore_index=True), pd.DataFrame(segment_rows), pd.DataFrame(event_rows)


def assert_equal_batches(first, second):
    assert len(first) == len(second)
    for original, changed in zip(first, second):
        assert original.keys() == changed.keys()
        for name, value in original.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, changed[name], rtol=0, atol=0)
            else:
                assert value == changed[name]


def test_larger_reader_cache_preserves_samples_and_avoids_repeated_decompression(tmp_path, monkeypatch):
    _, segments, _ = native_session_frames(tmp_path)
    readers = [SessionWindowReader(segments, cache_size=size) for size in (1, 2)]
    archive_load = deep_dataset._load_segment_archive
    loads = []

    def counted_load(path):
        loads.append(path)
        return archive_load(path)

    monkeypatch.setattr(deep_dataset, "_load_segment_archive", counted_load)
    results, counts = [], []
    for reader in readers:
        before = len(loads)
        results.append([reader.read("session", 55_000, 65_000, subject_key="first")
                        for _attempt in range(4)])
        counts.append(len(loads) - before)
    assert counts == [8, 2]
    for baseline, optimized in zip(*results):
        for name in baseline:
            np.testing.assert_array_equal(baseline[name], optimized[name])


@pytest.mark.parametrize("version", ["v410", "v412"])
def test_augmented_chunks_identical_after_prefetch_cache_change_and_resume(tmp_path, version):
    anchors, segments, events = native_session_frames(tmp_path)
    geometry = SequenceGeometry(supervised_steps=4, short_receptive_field_steps=7,
        fused_short_receptive_field_steps=7, long_receptive_field_tokens=3)
    filename = "statsfusion_v412_integrated_repair.yaml" if version == "v412" else "statsfusion_v410_joint_event.yaml"
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / filename)
    config["training"]["num_workers"] = 2

    def collect(cache, prefetch, start_chunk):
        from bme_eating.data.joint_v412_augmentation import JointSessionDatasetV412
        from bme_eating.stats_features import FoldRobustScaler
        dataset_class = JointSessionDatasetV412 if version == "v412" else JointSessionDataset
        dataset = dataset_class(anchors, segments, events,
            Normalization(np.zeros(6), np.ones(6), 0., 1.),
            statistics_columns=STATISTICS_COLUMNS, geometry=geometry,
            training=True, seed=2026, reader_cache_size=cache,
            ppg_modality_dropout=.2, gyro_modality_dropout=.1, rotation_augmentation_probability=.2)
        if version == "v412":
            dataset.configure(config, FoldRobustScaler(STATS_FEATURE_COLUMNS, np.zeros(12), np.ones(12),
                                                      ("first", "second")), {"fixture": "natural"})
        sampler = SessionChunkSampler(dataset, 8, start_chunk=start_chunk)
        settings = copy.deepcopy(config)
        settings["training"]["loader_prefetch_factor"] = prefetch
        settings["training"]["session_reader_cache_size"] = cache
        return list(session_loader(dataset, sampler, settings)), sampler.identity

    baseline, identity = collect(1, 1, 0)
    optimized, changed_identity = collect(4, 4, 0)
    resumed, resumed_identity = collect(4, 4, 3)
    assert identity == changed_identity == resumed_identity
    assert_equal_batches(baseline, optimized)
    assert_equal_batches(baseline[3:], resumed)


def test_io_profile_can_resume_without_changing_training_identity(tmp_path, monkeypatch):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v410_joint_event.yaml")
    baseline = copy.deepcopy(config)
    baseline["training"]["loader_prefetch_factor"] = 1
    baseline["training"]["session_reader_cache_size"] = 1
    assert config["training"]["num_workers"] == 10
    assert config["training"]["loader_prefetch_factor"] == 4
    assert config["training"]["session_reader_cache_size"] == 4
    assert config_identity(config) == config_identity(baseline)
    monkeypatch.setattr("bme_eating.joint_protocol.runtime_identity", lambda: {"python": "same"})
    source = tmp_path / "source"
    source.mkdir()
    run = initialize_run(baseline, source, tmp_path / "output", "run", {"data": "same"},
                         fresh=True, skip_s0=True, historical_reference=None)
    identity = (run / "identity.json").read_bytes()
    assert initialize_run(config, source, tmp_path / "output", "run", {"data": "same"},
                          fresh=False, skip_s0=True, historical_reference=None) == run
    assert (run / "identity.json").read_bytes() == identity
