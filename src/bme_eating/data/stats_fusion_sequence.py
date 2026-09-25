from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from bme_eating.data.deep_dataset import Normalization, _sample_grid
from bme_eating.data.session import SessionWindowReader
from bme_eating.features.signal import ppg_quality_features


@dataclass(frozen=True)
class SequenceGeometry:
    supervised_steps: int = 256
    short_receptive_field_steps: int = 127
    long_receptive_field_tokens: int = 127
    long_pool_factor: int = 5
    step_seconds: int = 3

    @property
    def history_steps(self) -> int:
        short_history = self.short_receptive_field_steps - 1
        long_history = self.long_receptive_field_tokens * self.long_pool_factor
        return max(short_history, long_history)

    @property
    def total_steps(self) -> int:
        return self.history_steps + self.supervised_steps


class ClipMixtureSampler(Sampler[tuple[int, int, float]]):
    def __init__(
        self,
        anchors: pd.DataFrame,
        *,
        samples_per_epoch: int,
        mixture: dict[str, float],
        seed: int,
    ) -> None:
        required = {"state_target", "start_target", "end_target", "state_loss_mask"}
        missing = required - set(anchors.columns)
        if missing:
            raise ValueError(f"Clip sampler anchors are missing columns: {sorted(missing)}")
        if set(mixture) != {"uniform", "event", "boundary"}:
            raise ValueError("Clip mixture must define uniform, event, and boundary")
        weights = np.asarray([mixture[name] for name in ("uniform", "event", "boundary")])
        if (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Clip sampling mixture must be non-negative and sum to one")
        eligible = anchors["state_loss_mask"].fillna(0.0).to_numpy(dtype=float) > 0
        indices = np.arange(len(anchors), dtype=np.int64)
        pools = {
            "uniform": indices[eligible],
            "event": indices[eligible & (anchors["state_target"].to_numpy(dtype=float) > 0)],
            "boundary": indices[
                eligible
                & (
                    (anchors["start_target"].to_numpy(dtype=float) > 0)
                    | (anchors["end_target"].to_numpy(dtype=float) > 0)
                )
            ],
        }
        if not len(pools["uniform"]):
            raise ValueError("Clip sampler received no eligible anchors")
        for name in ("event", "boundary"):
            if not len(pools[name]):
                pools[name] = pools["uniform"]
        probabilities = np.zeros(len(anchors), dtype=np.float64)
        for weight, name in zip(weights, ("uniform", "event", "boundary")):
            probabilities[pools[name]] += float(weight) / len(pools[name])
        natural = 1.0 / len(pools["uniform"])
        self.importance = np.divide(
            natural,
            probabilities,
            out=np.zeros_like(probabilities),
            where=probabilities > 0,
        )
        self.pools = pools
        self.weights = weights
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self) -> Iterator[tuple[int, int, float]]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        names = np.asarray(("uniform", "event", "boundary"), dtype=object)
        for _ in range(self.samples_per_epoch):
            category = str(rng.choice(names, p=self.weights))
            index = int(rng.choice(self.pools[category]))
            yield index, self.epoch, float(self.importance[index])


class StatsFusionSequenceDataset(Dataset[dict[str, torch.Tensor | str | int]]):
    def __init__(
        self,
        anchors: pd.DataFrame,
        segments: pd.DataFrame,
        events: pd.DataFrame,
        normalization: Normalization,
        *,
        statistics_columns: Sequence[str],
        geometry: SequenceGeometry | None = None,
        training: bool = False,
        ppg_modality_dropout: float = 0.0,
        seed: int = 2026,
    ) -> None:
        required = {
            "subject_key",
            "session_id",
            "timestamp_ms",
            "state_target",
            "state_loss_mask",
            *statistics_columns,
        }
        missing = required - set(anchors.columns)
        if missing:
            raise ValueError(f"Sequence anchors are missing columns: {sorted(missing)}")
        self.anchors = anchors.reset_index(drop=True).copy()
        self.anchors["_row_id"] = np.arange(len(self.anchors), dtype=np.int64)
        self.segments = segments
        valid_duration = (
            events["valid_duration"].fillna(False).astype(bool)
            if "valid_duration" in events
            else pd.Series(True, index=events.index)
        )
        self.events = events[valid_duration].copy()
        self.normalization = normalization
        self.statistics_columns = tuple(str(value) for value in statistics_columns)
        if len(self.statistics_columns) != 24:
            raise ValueError("Sequence dataset requires 12 scaled values and 12 missing flags")
        self.geometry = geometry or SequenceGeometry()
        self.training = bool(training)
        self.ppg_modality_dropout = float(ppg_modality_dropout)
        if not 0.0 <= self.ppg_modality_dropout <= 1.0:
            raise ValueError("PPG modality dropout must be in [0, 1]")
        self.seed = int(seed)
        self.reader = SessionWindowReader(segments, cache_size=8)
        self.session_groups = {
            (str(subject), str(session)): group.sort_values("timestamp_ms").reset_index(drop=True)
            for (subject, session), group in self.anchors.groupby(
                ["subject_key", "session_id"], sort=False
            )
        }
        self.row_location: dict[int, tuple[str, str]] = {}
        for key, group in self.session_groups.items():
            for row_id in group["_row_id"].to_numpy(dtype=np.int64):
                self.row_location[int(row_id)] = key

    def __len__(self) -> int:
        return len(self.anchors)

    @staticmethod
    def _nearest_indices(source: np.ndarray, target: np.ndarray, tolerance_ms: int) -> np.ndarray:
        if not len(source):
            return np.full(len(target), -1, dtype=np.int64)
        right = np.searchsorted(source, target, side="left")
        right = np.clip(right, 0, len(source) - 1)
        left = np.clip(right - 1, 0, len(source) - 1)
        choose_left = np.abs(source[left] - target) <= np.abs(source[right] - target)
        selected = np.where(choose_left, left, right)
        selected[np.abs(source[selected] - target) > tolerance_ms] = -1
        return selected.astype(np.int64)

    def _aligned_anchor_arrays(
        self, group: pd.DataFrame, timestamps: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        source_time = group["timestamp_ms"].to_numpy(dtype=np.int64)
        indices = self._nearest_indices(
            source_time, timestamps, self.geometry.step_seconds * 500
        )
        valid = indices >= 0
        arrays: dict[str, np.ndarray] = {}
        fields = (
            "state_target",
            "state_loss_mask",
            "start_loss_mask",
            "end_loss_mask",
        )
        for field in fields:
            default = 1.0 if field.endswith("loss_mask") else 0.0
            source = group.get(field, pd.Series(default, index=group.index)).to_numpy(dtype=np.float32)
            output = np.full(len(timestamps), default, dtype=np.float32)
            output[valid] = source[indices[valid]]
            arrays[field] = output
        statistics = np.zeros((len(timestamps), len(self.statistics_columns)), dtype=np.float32)
        source_statistics = group.loc[:, self.statistics_columns].to_numpy(dtype=np.float32)
        statistics[valid] = source_statistics[indices[valid]]
        arrays["statistics"] = statistics
        return valid, arrays

    def _motion_blocks(
        self, payload: dict[str, np.ndarray], first_timestamp_ms: int, steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        seconds = steps * self.geometry.step_seconds
        start_ms = first_timestamp_ms - self.geometry.step_seconds * 1000
        values, mask = _sample_grid(
            payload["motion_timestamp_ms"],
            payload["motion_values"],
            payload["motion_mask"].astype(bool),
            start_ms,
            seconds,
            100,
        )
        values = np.clip(
            (values - self.normalization.motion_median) / self.normalization.motion_iqr,
            -10.0,
            10.0,
        )
        values[~mask] = 0.0
        samples = self.geometry.step_seconds * 100
        values = values.reshape(steps, samples, 6)
        mask = mask.reshape(steps, samples, 6)
        blocks = np.concatenate((values, mask.astype(np.float32)), axis=2).transpose(0, 2, 1)
        valid = mask.any(axis=(1, 2)).astype(np.float32)
        return blocks.astype(np.float32), valid

    def _ppg_blocks(
        self,
        payload: dict[str, np.ndarray],
        timestamps: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        factor = self.geometry.long_pool_factor
        end_indices = np.arange(factor - 1, len(timestamps), factor, dtype=np.int64)
        if not len(end_indices):
            raise ValueError("Sequence is too short to form one completed PPG block")
        block_ends = timestamps[end_indices]
        start_ms = int(block_ends[0] - 15_000)
        duration_seconds = int(len(block_ends) * 15)
        raw, mask = _sample_grid(
            payload["ppg_timestamp_ms"],
            payload["ppg_values"],
            payload["ppg_mask"].astype(bool),
            start_ms,
            duration_seconds,
            50,
        )
        raw = raw.reshape(len(block_ends), 750)
        mask = mask.reshape(len(block_ends), 750)
        quality = np.zeros((len(block_ends), 8), dtype=np.float32)
        valid = mask.mean(axis=1).astype(np.float32)
        for index in range(len(block_ends)):
            quality[index], _ = ppg_quality_features(raw[index], mask[index], 50.0)
        normalized = np.clip(
            (raw - self.normalization.ppg_median) / self.normalization.ppg_iqr,
            -10.0,
            10.0,
        )
        normalized[~mask] = 0.0
        blocks = np.stack((normalized, mask.astype(np.float32)), axis=1).astype(np.float32)
        if self.training and rng.random() < self.ppg_modality_dropout:
            blocks.fill(0.0)
            quality.fill(0.0)
            valid.fill(0.0)
        mapping = np.floor_divide(np.arange(len(timestamps)) + 1, factor) - 1
        mapping = np.clip(mapping, -1, len(block_ends) - 1).astype(np.int64)
        return blocks, quality, valid, mapping

    def _transition_targets(
        self, subject_key: str, timestamps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        selected = self.events[self.events["subject_key"].astype(str) == subject_key]
        if len(timestamps):
            selected = selected[
                selected["start_ms"].between(
                    int(timestamps[0]) - 30_000, int(timestamps[-1]), inclusive="both"
                )
                | selected["end_ms"].between(
                    int(timestamps[0]) - 60_000, int(timestamps[-1]), inclusive="both"
                )
            ]
        onset = np.zeros(len(timestamps), dtype=np.float32)
        offset = np.zeros(len(timestamps), dtype=np.float32)
        smooth = np.ones(len(timestamps), dtype=np.float32)
        for event in selected.itertuples(index=False):
            start_delta = (timestamps - int(event.start_ms)) / 1000.0
            end_delta = (timestamps - int(event.end_ms)) / 1000.0
            onset = np.maximum(
                onset,
                np.where(
                    (start_delta >= 0.0) & (start_delta <= 30.0),
                    1.0 - start_delta / 30.0,
                    0.0,
                ),
            )
            offset = np.maximum(
                offset,
                np.where(
                    (end_delta >= 0.0) & (end_delta <= 60.0),
                    1.0 - end_delta / 60.0,
                    0.0,
                ),
            )
            near_boundary = (np.abs(start_delta) <= 30.0) | (np.abs(end_delta) <= 30.0)
            smooth[near_boundary] = 0.0
        return onset.astype(np.float32), offset.astype(np.float32), smooth

    def __getitem__(
        self, index: int | tuple[int, int, float]
    ) -> dict[str, torch.Tensor | str | int]:
        if isinstance(index, tuple):
            row_index, epoch, importance = index
        else:
            row_index, epoch, importance = index, 0, 1.0
        row_index = int(row_index)
        anchor = self.anchors.iloc[row_index]
        subject_key, session_id = self.row_location[row_index]
        group = self.session_groups[(subject_key, session_id)]
        end_timestamp = int(anchor.timestamp_ms)
        step_ms = self.geometry.step_seconds * 1000
        timestamps = end_timestamp - np.arange(
            self.geometry.total_steps - 1, -1, -1, dtype=np.int64
        ) * step_ms
        payload = self.reader.read(
            session_id,
            int(timestamps[0] - 15_000),
            int(timestamps[-1]),
        )
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, row_index, int(epoch)])
        )
        motion, motion_valid = self._motion_blocks(payload, int(timestamps[0]), len(timestamps))
        ppg, ppg_quality, ppg_valid, ppg_mapping = self._ppg_blocks(payload, timestamps, rng)
        aligned_valid, arrays = self._aligned_anchor_arrays(group, timestamps)
        onset, offset, smooth = self._transition_targets(subject_key, timestamps)
        supervision = np.zeros(len(timestamps), dtype=np.float32)
        supervision[-self.geometry.supervised_steps :] = 1.0
        supervision *= aligned_valid.astype(np.float32)
        result: dict[str, torch.Tensor | str | int] = {
            "motion_blocks": torch.from_numpy(motion),
            "motion_valid": torch.from_numpy(motion_valid),
            "ppg_blocks": torch.from_numpy(ppg),
            "ppg_quality": torch.from_numpy(ppg_quality),
            "ppg_valid": torch.from_numpy(ppg_valid),
            "ppg_to_motion_index": torch.from_numpy(ppg_mapping),
            "statistics": torch.from_numpy(arrays["statistics"]),
            "state_target": torch.from_numpy(arrays["state_target"]),
            "state_loss_mask": torch.from_numpy(arrays["state_loss_mask"]),
            "onset_target": torch.from_numpy(onset),
            "offset_target": torch.from_numpy(offset),
            "onset_loss_mask": torch.from_numpy(arrays["start_loss_mask"]),
            "offset_loss_mask": torch.from_numpy(arrays["end_loss_mask"]),
            "smooth_mask": torch.from_numpy(smooth),
            "supervision_mask": torch.from_numpy(supervision),
            "importance_weight": torch.tensor(float(importance), dtype=torch.float32),
            "timestamp_ms": torch.from_numpy(timestamps),
            "subject_key": subject_key,
            "session_id": session_id,
            "end_timestamp_ms": end_timestamp,
        }
        return result
