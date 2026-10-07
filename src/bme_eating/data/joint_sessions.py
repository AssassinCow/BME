from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

from bme_eating.data.deep_dataset import Normalization, compute_normalization
from bme_eating.data.stats_fusion_inputs import (
    canonical_input_paths,
    verify_canonical_statsfusion_inputs,
)
from bme_eating.data.stats_fusion_sequence import (
    StatsFusionSequenceDataset,
    sequence_geometry_from_config,
)
from bme_eating.hierarchical_artifacts import sha256_file
from bme_eating.joint_protocol import digest
from bme_eating.models.statsfusion_joint_event import CompletedScale
from bme_eating.stats_features import STATS_FEATURE_COLUMNS, FoldRobustScaler


@dataclass
class JointInputs:
    input_root: Path
    canonical_root: Path
    subject_folds: dict[str, int]
    identity: dict[str, Any]

    def __post_init__(self):
        self.bound_files = self.file_signature()
        path = self.input_root / "indices" / "segments.parquet"
        self.bound_sessions = {}
        if path.is_file():
            segments = pd.read_parquet(path, columns=["subject_key", "session_id", "segment_path"])
            for key, group in segments.groupby(["subject_key", "session_id"]):
                files = {Path(value) for value in group.segment_path}
                self.bound_sessions[tuple(map(str, key))] = {
                    str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in files}

    def assert_session_unchanged(self, key):
        for relative, signature in self.bound_sessions.get(key, {}).items():
            path = Path(relative)
            if (path.stat().st_size, path.stat().st_mtime_ns) != signature:
                raise RuntimeError("Runtime raw sensor archive changed")

    def file_signature(self):
        paths = canonical_input_paths(self.canonical_root)
        paths.update({"subject_folds": self.input_root / "indices" / "subject_folds.json",
                      "segments": self.input_root / "indices" / "segments.parquet"})
        return {name: (path.stat().st_size, path.stat().st_mtime_ns) for name, path in paths.items()
                if name in self.identity}

    @classmethod
    def audit(cls, input_root: Path, canonical_root: Path) -> JointInputs:
        verify_canonical_statsfusion_inputs(input_root, canonical_root)
        paths = canonical_input_paths(canonical_root)
        split = input_root / "indices" / "subject_folds.json"
        folds = {str(subject): int(fold) for subject, fold in
                 json.loads(split.read_text(encoding="utf-8")).items()}
        if set(folds.values()) != set(range(5)):
            raise RuntimeError("Expected the original five subject folds")
        identity = {name: sha256_file(paths[name]) for name in
                    ("anchors", "statistics", "events", "manifest", "preparation_identity")}
        identity["subject_folds"] = sha256_file(split)
        identity["segments"] = sha256_file(input_root / "indices" / "segments.parquet")
        return cls(input_root, canonical_root, folds, identity)

    def frames(self, subjects: set[str], *, labels: bool):
        if self.file_signature() != self.bound_files:
            raise RuntimeError("Runtime canonical input identity changed")
        for key in self.bound_sessions:
            if key[0] in subjects:
                self.assert_session_unchanged(key)
        filters = [("subject_key", "in", sorted(subjects))]
        paths = canonical_input_paths(self.canonical_root)
        unlabeled_columns = ["segment_id", "session_id", "segment_path", "subject_key", "timestamp_ms",
                             "motion_history_available_seconds", "ppg_history_available_seconds"]
        anchors = pd.read_parquet(paths["anchors"], filters=filters,
                                   columns=None if labels else unlabeled_columns)
        statistics = pd.read_parquet(paths["statistics"], filters=filters)
        events = pd.read_parquet(paths["events"], filters=filters) if labels else pd.DataFrame()
        if not labels:
            for column in ("state_target", "start_target", "end_target"):
                anchors[column] = 0.0
            for column in ("state_loss_mask", "start_loss_mask", "end_loss_mask", "smooth_loss_mask"):
                anchors[column] = 0.0
        segments = pd.read_parquet(self.input_root / "indices" / "segments.parquet", filters=filters)
        return anchors, statistics, events, segments


def joint_geometry(config: dict[str, Any]):
    geometry = sequence_geometry_from_config(config)
    proposal_receptive_field = CompletedScale.receptive_field_steps
    extra = (proposal_receptive_field - 1) * (60 // geometry.step_seconds) + int(
        config["model"].get("local_attention_window_steps", 0))
    return replace(geometry, fused_short_receptive_field_steps=
                   geometry.fused_short_receptive_field_steps + extra)


class JointSessionDataset(StatsFusionSequenceDataset):
    def __getitem__(self, index):
        row, epoch, importance = index
        subject, session = self.row_location[int(row)]
        seed_text = f"{self.seed}|{epoch}|{subject}|{session}"
        self.session_augmentation_seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8])
        result = super().__getitem__((row, epoch, importance))
        step_ms = self.geometry.step_seconds * 1000
        result["timeline_position"] = (
            (result["timestamp_ms"] - self.session_origins[(subject, session)]) // step_ms - 1)
        return result

    def _motion_blocks(self, payload, first_timestamp_ms, steps, rng):
        return super()._motion_blocks(payload, first_timestamp_ms, steps,
                                       np.random.default_rng(self.session_augmentation_seed))

    def _ppg_blocks(self, payload, timestamps, session_origin_ms, rng):
        return super()._ppg_blocks(payload, timestamps, session_origin_ms,
                                   np.random.default_rng(self.session_augmentation_seed + 1))


def make_joint_dataset(inputs: JointInputs, subjects: set[str], config: dict[str, Any],
                       preprocessing: dict[str, Any] | None = None, *, training: bool):
    anchors, statistics, events, segments = inputs.frames(subjects, labels=training)
    if preprocessing is None:
        if not training:
            raise RuntimeError("Prediction requires training-bound preprocessing")
        scaler = FoldRobustScaler.fit(statistics, training_subjects=subjects)
        normal = compute_normalization(segments, subjects, require_motion=True, require_ppg=True)
        preprocessing = {"scaler": scaler.to_json(), "normalization": normal.to_json(),
                         "training_subjects": sorted(subjects)}
    else:
        scaler = FoldRobustScaler.from_json(preprocessing["scaler"])
        normal = Normalization.from_json(preprocessing["normalization"])
        if not training:
            scaler.assert_unseen(subjects)
    transformed = scaler.transform_frame(statistics)
    keys = ["subject_key", "session_id", "timestamp_ms"]
    columns = [*(f"stat_{name}" for name in STATS_FEATURE_COLUMNS),
               *(f"stat_{name}_missing" for name in STATS_FEATURE_COLUMNS)]
    anchors = anchors.merge(transformed[[*keys, *columns]], on=keys, how="left", validate="one_to_one")
    if anchors[columns].isna().any().any():
        raise RuntimeError("Canonical statistics have missing anchor alignment rows")
    dataset = JointSessionDataset(
        anchors, segments, events, normal, statistics_columns=columns,
        geometry=joint_geometry(config), training=training, seed=2026,
        reader_cache_size=int(config["training"]["session_reader_cache_size"]),
        ppg_modality_dropout=float(config["training"]["ppg_modality_dropout"]) if training else 0,
        gyro_modality_dropout=float(config["training"]["gyro_modality_dropout"]) if training else 0,
        rotation_augmentation_probability=float(config["training"]["rotation_augmentation_probability"])
        if training else 0)
    return dataset, preprocessing


class SessionChunkSampler(Sampler):
    def __init__(self, dataset: JointSessionDataset, epoch: int, start_chunk: int = 0,
                 training: bool = True) -> None:
        self.dataset, self.epoch = dataset, epoch
        groups: dict[str, list[tuple[str, str]]] = {}
        for key in sorted(dataset.session_groups):
            groups.setdefault(key[0], []).append(key)
        rng = np.random.default_rng(2026 + epoch)
        subjects = sorted(groups)
        if training:
            rng.shuffle(subjects)
            for sessions in groups.values():
                rng.shuffle(sessions)
        order = []
        while any(groups.values()):
            for subject in subjects:
                if groups[subject]:
                    order.append(groups[subject].pop(0))
        self.chunks: list[tuple[int, int, np.ndarray, tuple[str, str], bool, int]] = []
        for key in order:
            rows = dataset.session_groups[key]["_row_id"].to_numpy()
            for first in range(0, len(rows), dataset.geometry.supervised_steps):
                stop = min(first + dataset.geometry.supervised_steps, len(rows))
                mask = np.zeros(dataset.geometry.supervised_steps, dtype=np.float32)
                mask[-(stop - first):] = 1
                self.chunks.append((int(rows[stop - 1]), epoch, mask, key, stop == len(rows), stop - first))
        self.start_chunk = start_chunk
        if not 0 <= start_chunk <= len(self.chunks):
            raise RuntimeError("Resume chunk cursor is outside the registered session traversal")
        self.identity = digest([(item[0], item[3], item[4], item[5]) for item in self.chunks])

    def __iter__(self):
        for row, epoch, mask, *_ in self.chunks[self.start_chunk:]:
            yield row, epoch, mask

    def __len__(self):
        return len(self.chunks) - self.start_chunk


def single_item_collate(items):
    return {key: value[None] if isinstance(value, torch.Tensor) else value
            for key, value in items[0].items()}


def session_loader(dataset: JointSessionDataset, sampler: SessionChunkSampler,
                   config: dict[str, Any]):
    workers = int(config["training"]["num_workers"])
    return DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=workers,
                      prefetch_factor=int(config["training"]["loader_prefetch_factor"]),
                      persistent_workers=False, collate_fn=single_item_collate,
                      generator=torch.Generator().manual_seed(2026 + sampler.epoch))
