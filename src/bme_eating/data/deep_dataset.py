from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from bme_eating.features.signal import ppg_quality_features
from bme_eating.data.session import SessionWindowReader


@dataclass(frozen=True)
class Normalization:
    motion_median: np.ndarray
    motion_iqr: np.ndarray
    ppg_median: float
    ppg_iqr: float

    def to_json(self) -> dict[str, object]:
        return {
            "motion_median": self.motion_median.tolist(),
            "motion_iqr": self.motion_iqr.tolist(),
            "ppg_median": self.ppg_median,
            "ppg_iqr": self.ppg_iqr,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Normalization":
        return cls(
            motion_median=np.asarray(payload["motion_median"], dtype=np.float32),
            motion_iqr=np.asarray(payload["motion_iqr"], dtype=np.float32),
            ppg_median=float(payload["ppg_median"]),
            ppg_iqr=float(payload["ppg_iqr"]),
        )


def _load_segment_archive(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    try:
        with np.load(path) as archive:
            payload = {name: archive[name] for name in archive.files}
        required = {
            "motion_timestamp_ms",
            "motion_values",
            "motion_mask",
            "ppg_timestamp_ms",
            "ppg_values",
            "ppg_mask",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"Segment cache is missing fields: {sorted(missing)}")
        if len(payload["motion_timestamp_ms"]) != len(payload["motion_values"]):
            raise ValueError("Motion timestamps and values have different lengths")
        if len(payload["motion_timestamp_ms"]) != len(payload["motion_mask"]):
            raise ValueError("Motion timestamps and mask have different lengths")
        if len(payload["ppg_timestamp_ms"]) != len(payload["ppg_values"]):
            raise ValueError("PPG timestamps and values have different lengths")
        if len(payload["ppg_timestamp_ms"]) != len(payload["ppg_mask"]):
            raise ValueError("PPG timestamps and mask have different lengths")
        if payload["motion_values"].ndim != 2 or payload["motion_values"].shape[1] != 6:
            raise ValueError("Motion values must have shape (n, 6)")
        if np.any(np.diff(payload["motion_timestamp_ms"]) <= 0) or np.any(
            np.diff(payload["ppg_timestamp_ms"]) <= 0
        ):
            raise ValueError("Segment cache timestamps must be strictly increasing")
        return payload
    except Exception as error:
        raise RuntimeError(
            f"Failed to load segment cache {path}: {type(error).__name__}: {error}"
        ) from error


def compute_normalization(
    segments: pd.DataFrame,
    subject_keys: set[str],
    maximum_samples_per_segment: int = 10_000,
) -> Normalization:
    motion_samples: list[np.ndarray] = []
    ppg_samples: list[np.ndarray] = []
    selected = segments[segments["subject_key"].isin(subject_keys)]
    for segment in selected.itertuples(index=False):
        payload = _load_segment_archive(segment.segment_path)
        motion = payload["motion_values"]
        motion_mask = payload["motion_mask"].astype(bool)
        ppg = payload["ppg_values"].reshape(-1)
        ppg_mask = payload["ppg_mask"].astype(bool).reshape(-1)
        motion_stride = max(1, len(motion) // maximum_samples_per_segment)
        motion_subset = motion[::motion_stride].copy()
        motion_subset[~motion_mask[::motion_stride]] = np.nan
        motion_samples.append(motion_subset)
        valid_ppg = ppg[ppg_mask]
        ppg_stride = max(1, len(valid_ppg) // maximum_samples_per_segment)
        ppg_samples.append(valid_ppg[::ppg_stride])
    if not motion_samples or not ppg_samples:
        raise ValueError("No valid training samples available for normalization")
    motion_all = np.concatenate(motion_samples, axis=0)
    ppg_all = np.concatenate(ppg_samples)
    if not np.isfinite(motion_all).any() or len(ppg_all) == 0 or not np.isfinite(ppg_all).any():
        raise ValueError("No finite training sensor samples available for normalization")
    motion_median = np.nanmedian(motion_all, axis=0)
    motion_median = np.nan_to_num(motion_median, nan=0.0, posinf=0.0, neginf=0.0)
    motion_iqr = np.nanpercentile(motion_all, 75, axis=0) - np.nanpercentile(
        motion_all, 25, axis=0
    )
    motion_iqr = np.where(motion_iqr > 1e-6, motion_iqr, 1.0)
    ppg_median = float(np.median(ppg_all))
    ppg_iqr = float(np.percentile(ppg_all, 75) - np.percentile(ppg_all, 25))
    return Normalization(
        motion_median=motion_median.astype(np.float32),
        motion_iqr=motion_iqr.astype(np.float32),
        ppg_median=ppg_median,
        ppg_iqr=max(ppg_iqr, 1.0),
    )


def save_normalization(normalization: Normalization, path: Path) -> None:
    path.write_text(json.dumps(normalization.to_json(), indent=2), encoding="utf-8")


def load_normalization(path: Path) -> Normalization:
    return Normalization.from_json(json.loads(path.read_text(encoding="utf-8")))


def _sample_grid(
    timestamp_ms: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    start_ms: int,
    duration_seconds: int,
    sampling_hz: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = int(duration_seconds * sampling_hz)
    dimensions = values.shape[1]
    output_values = np.zeros((count, dimensions), dtype=np.float32)
    output_mask = np.zeros((count, dimensions), dtype=bool)
    if count == 0 or len(timestamp_ms) == 0:
        return output_values, output_mask
    period_ms = 1000.0 / sampling_hz
    end_ms = start_ms + duration_seconds * 1000
    left = int(np.searchsorted(timestamp_ms, start_ms - period_ms / 2, side="left"))
    right = int(np.searchsorted(timestamp_ms, end_ms - period_ms / 2, side="left"))
    source_time = timestamp_ms[left:right]
    source_values = values[left:right]
    source_mask = mask[left:right]
    target_indices = np.rint((source_time - start_ms) / period_ms).astype(np.int64)
    valid = (target_indices >= 0) & (target_indices < count)
    target_indices = target_indices[valid]
    output_values[target_indices] = source_values[valid]
    output_mask[target_indices] = source_mask[valid]
    output_values[~output_mask] = 0.0
    return output_values, output_mask


def _corrupt_ppg(
    values: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    values = values.copy()
    mask = mask.copy()
    corruption = int(rng.integers(0, 6))
    length = len(values)
    if corruption == 0:
        start = int(rng.integers(0, max(1, length // 2)))
        width = int(rng.integers(max(1, length // 10), max(2, length // 2)))
        mask[start : min(length, start + width)] = False
        values[~mask] = 0.0
    elif corruption == 1:
        scale = max(float(np.std(values[mask])), 1.0)
        values += rng.normal(0.0, scale * rng.uniform(0.5, 2.0), size=length)
    elif corruption == 2:
        lower, upper = np.percentile(values[mask], [10, 90]) if mask.any() else (-1.0, 1.0)
        values = np.clip(values, lower, upper)
    elif corruption == 3:
        spike_count = max(1, length // 100)
        indices = rng.choice(length, size=spike_count, replace=False)
        scale = max(float(np.std(values[mask])), 1.0)
        values[indices] += rng.normal(0.0, 10.0 * scale, size=spike_count)
    elif corruption == 4:
        scale = max(float(np.std(values[mask])), 1.0)
        values += np.linspace(0.0, rng.uniform(-5.0, 5.0) * scale, length)
    else:
        values[:] = 0.0
        mask[:] = False
    return values.astype(np.float32), mask


class DTPDataset(Dataset[dict[str, torch.Tensor | str | int]]):
    def __init__(
        self,
        anchors: pd.DataFrame,
        segments: pd.DataFrame,
        normalization: Normalization,
        future_context_seconds: int = 0,
        training: bool = False,
        ppg_augmentation_probability: float = 0.0,
        seed: int = 2026,
        motion_block_seconds: int = 3,
        ppg_block_seconds: int = 15,
        motion_bucket_counts: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
        ppg_bucket_counts: Sequence[int] = (1, 2, 4, 8, 16),
    ) -> None:
        self.anchors = anchors.reset_index(drop=True)
        self.session_reader = SessionWindowReader(segments, cache_size=8)
        self.normalization = normalization
        self.future_context_seconds = int(future_context_seconds)
        self.training = training
        self.ppg_augmentation_probability = float(ppg_augmentation_probability)
        self.seed = int(seed)
        self.motion_block_seconds = int(motion_block_seconds)
        self.ppg_block_seconds = int(ppg_block_seconds)
        self.motion_bucket_counts = tuple(int(value) for value in motion_bucket_counts)
        self.ppg_bucket_counts = tuple(int(value) for value in ppg_bucket_counts)
        if self.motion_block_seconds <= 0 or self.ppg_block_seconds <= 0:
            raise ValueError("DTP block durations must be positive")
        if not self.motion_bucket_counts or min(self.motion_bucket_counts) <= 0:
            raise ValueError("Motion bucket counts must be non-empty and positive")
        if not self.ppg_bucket_counts or min(self.ppg_bucket_counts) <= 0:
            raise ValueError("PPG bucket counts must be non-empty and positive")
        if self.future_context_seconds and (
            self.future_context_seconds % self.motion_block_seconds != 0
            or self.future_context_seconds % self.ppg_block_seconds != 0
        ):
            raise ValueError("Future context must be divisible by both DTP block durations")
        self.motion_block_count = sum(self.motion_bucket_counts)
        self.ppg_block_count = sum(self.ppg_bucket_counts)
        self.motion_history_seconds = self.motion_block_seconds * self.motion_block_count
        self.ppg_history_seconds = self.ppg_block_seconds * self.ppg_block_count
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._cache_size = 4

    def __len__(self) -> int:
        return len(self.anchors)

    def _load_segment(self, path: str) -> dict[str, np.ndarray]:
        if path in self._cache:
            payload = self._cache.pop(path)
            self._cache[path] = payload
            return payload
        payload = _load_segment_archive(path)
        self._cache[path] = payload
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return payload

    def _ppg_blocks(
        self,
        timestamp_ms: np.ndarray,
        values: np.ndarray,
        mask: np.ndarray,
        start_ms: int,
        seconds: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw, valid = _sample_grid(timestamp_ms, values, mask, start_ms, seconds, 50)
        if seconds % self.ppg_block_seconds != 0:
            raise ValueError("PPG duration must be divisible by the configured block duration")
        block_count = seconds // self.ppg_block_seconds
        samples_per_block = self.ppg_block_seconds * 50
        raw = raw.reshape(block_count, samples_per_block)
        valid = valid.reshape(block_count, samples_per_block)
        quality_features = np.zeros((block_count, 8), dtype=np.float32)
        quality_target = np.zeros(block_count, dtype=np.float32)
        for block_index in range(block_count):
            if self.training and rng.random() < self.ppg_augmentation_probability:
                raw[block_index], valid[block_index] = _corrupt_ppg(
                    raw[block_index], valid[block_index], rng
                )
            quality_features[block_index], quality_target[block_index] = ppg_quality_features(
                raw[block_index], valid[block_index], 50.0
            )
        normalized = np.clip(
            (raw - self.normalization.ppg_median) / self.normalization.ppg_iqr, -10.0, 10.0
        )
        normalized[~valid] = 0.0
        blocks = np.stack((normalized, valid.astype(np.float32)), axis=1).astype(np.float32)
        return blocks, quality_features, quality_target

    def __getitem__(
        self, index: int | tuple[int, int]
    ) -> dict[str, torch.Tensor | str | int]:
        if isinstance(index, tuple):
            anchor_index, epoch = index
        else:
            anchor_index, epoch = index, 0
        anchor_index = int(anchor_index)
        epoch = int(epoch)
        if anchor_index < 0 or anchor_index >= len(self.anchors) or epoch < 0:
            raise IndexError(f"Invalid DTP dataset index: {(anchor_index, epoch)}")
        anchor = self.anchors.iloc[anchor_index]
        timestamp_ms = int(anchor.timestamp_ms)
        session_id = str(getattr(anchor, "session_id", anchor.segment_id))
        maximum_history_seconds = max(
            self.motion_history_seconds, self.ppg_history_seconds
        )
        payload = self.session_reader.read(
            session_id,
            timestamp_ms - maximum_history_seconds * 1000,
            timestamp_ms + self.future_context_seconds * 1000,
        )
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, anchor_index, epoch])
        )

        motion_start = timestamp_ms - self.motion_history_seconds * 1000
        motion, motion_mask = _sample_grid(
            payload["motion_timestamp_ms"],
            payload["motion_values"],
            payload["motion_mask"].astype(bool),
            motion_start,
            self.motion_history_seconds,
            100,
        )
        motion = np.clip(
            (motion - self.normalization.motion_median) / self.normalization.motion_iqr,
            -10.0,
            10.0,
        )
        motion[~motion_mask] = 0.0
        motion_samples_per_block = self.motion_block_seconds * 100
        motion = motion.reshape(self.motion_block_count, motion_samples_per_block, 6)
        motion_mask = motion_mask.reshape(
            self.motion_block_count, motion_samples_per_block, 6
        )
        motion_blocks = np.concatenate(
            (motion, motion_mask.astype(np.float32)), axis=2
        ).transpose(0, 2, 1)
        motion_valid = motion_mask.any(axis=(1, 2)).astype(np.float32)

        ppg_blocks, ppg_quality, ppg_quality_target = self._ppg_blocks(
            payload["ppg_timestamp_ms"],
            payload["ppg_values"],
            payload["ppg_mask"].astype(bool),
            timestamp_ms - self.ppg_history_seconds * 1000,
            self.ppg_history_seconds,
            rng,
        )
        ppg_valid = (ppg_blocks[:, 1].mean(axis=1) > 0.5).astype(np.float32)

        result: dict[str, torch.Tensor | str | int] = {
            "motion_blocks": torch.from_numpy(motion_blocks),
            "motion_valid": torch.from_numpy(motion_valid),
            "ppg_blocks": torch.from_numpy(ppg_blocks),
            "ppg_quality": torch.from_numpy(ppg_quality),
            "ppg_quality_target": torch.from_numpy(ppg_quality_target),
            "ppg_valid": torch.from_numpy(ppg_valid),
            "state_target": torch.tensor(float(anchor.state_target), dtype=torch.float32),
            "state_loss_mask": torch.tensor(
                float(getattr(anchor, "state_loss_mask", 1.0)), dtype=torch.float32
            ),
            "start_target": torch.tensor(float(anchor.start_target), dtype=torch.float32),
            "end_target": torch.tensor(float(anchor.end_target), dtype=torch.float32),
            "start_loss_mask": torch.tensor(float(anchor.start_loss_mask), dtype=torch.float32),
            "end_loss_mask": torch.tensor(float(anchor.end_loss_mask), dtype=torch.float32),
            "subject_key": str(anchor.subject_key),
            "segment_id": str(anchor.segment_id),
            "session_id": session_id,
            "timestamp_ms": int(timestamp_ms),
        }
        if self.future_context_seconds > 0:
            future_motion, future_motion_mask = _sample_grid(
                payload["motion_timestamp_ms"],
                payload["motion_values"],
                payload["motion_mask"].astype(bool),
                timestamp_ms,
                self.future_context_seconds,
                100,
            )
            future_motion = np.clip(
                (future_motion - self.normalization.motion_median)
                / self.normalization.motion_iqr,
                -10.0,
                10.0,
            )
            future_motion[~future_motion_mask] = 0.0
            motion_count = self.future_context_seconds // self.motion_block_seconds
            motion_samples_per_block = self.motion_block_seconds * 100
            future_motion = future_motion.reshape(
                motion_count, motion_samples_per_block, 6
            )
            future_motion_mask = future_motion_mask.reshape(
                motion_count, motion_samples_per_block, 6
            )
            result["future_motion_blocks"] = torch.from_numpy(
                np.concatenate(
                    (future_motion, future_motion_mask.astype(np.float32)), axis=2
                ).transpose(0, 2, 1)
            )
            result["future_motion_valid"] = torch.from_numpy(
                future_motion_mask.any(axis=(1, 2)).astype(np.float32)
            )
            future_ppg, future_quality, future_quality_target = self._ppg_blocks(
                payload["ppg_timestamp_ms"],
                payload["ppg_values"],
                payload["ppg_mask"].astype(bool),
                timestamp_ms,
                self.future_context_seconds,
                rng,
            )
            result["future_ppg_blocks"] = torch.from_numpy(future_ppg)
            result["future_ppg_quality"] = torch.from_numpy(future_quality)
            result["future_ppg_quality_target"] = torch.from_numpy(future_quality_target)
            result["future_ppg_valid"] = torch.from_numpy(
                (future_ppg[:, 1].mean(axis=1) > 0.5).astype(np.float32)
            )
        return result


class SegmentBalancedBatchSampler(Sampler[list[tuple[int, int]]]):
    def __init__(
        self,
        anchors: pd.DataFrame,
        batch_size: int,
        steps_per_epoch: int,
        positive_fraction: float,
        seed: int,
    ) -> None:
        if batch_size <= 0 or steps_per_epoch <= 0:
            raise ValueError("batch_size and steps_per_epoch must be positive")
        if not 0 < positive_fraction <= 1:
            raise ValueError("positive_fraction must be in (0, 1]")
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.positive_fraction = float(positive_fraction)
        self.seed = int(seed)
        self.epoch = 0
        state_loss_mask = anchors.get(
            "state_loss_mask", pd.Series(1.0, index=anchors.index)
        )
        eligible_rows = anchors[state_loss_mask.fillna(0.0).astype(float) > 0]
        positive_rows = eligible_rows[eligible_rows["state_target"] > 0]
        self.positive_by_event: dict[str, np.ndarray] = {}
        for event_id, group in positive_rows.groupby("event_id", sort=True):
            key = str(event_id) or f"segment:{group.iloc[0].segment_id}"
            self.positive_by_event[key] = group.index.to_numpy(dtype=np.int64)
        negative_rows = eligible_rows[eligible_rows["state_target"] <= 0]
        distance = negative_rows["distance_to_event_seconds"].to_numpy(dtype=np.float64)
        self.near_negative = negative_rows.index.to_numpy(dtype=np.int64)[distance <= 1800.0]
        self.far_negative = negative_rows.index.to_numpy(dtype=np.int64)[distance > 1800.0]
        self.all_negative = negative_rows.index.to_numpy(dtype=np.int64)
        self.positive_events = np.asarray(list(self.positive_by_event), dtype=object)
        if len(self.positive_events) == 0 or len(self.all_negative) == 0:
            raise ValueError(
                "Training anchors must contain both positive and negative rows"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    @staticmethod
    def _sample(
        rng: np.random.Generator,
        values: np.ndarray,
        count: int,
    ) -> list[int]:
        if count <= 0:
            return []
        return rng.choice(values, size=count, replace=len(values) < count).astype(int).tolist()

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        positive_count = max(1, int(round(self.batch_size * self.positive_fraction)))
        positive_count = min(positive_count, self.batch_size)
        negative_count = self.batch_size - positive_count
        for _ in range(self.steps_per_epoch):
            positive = [
                int(rng.choice(self.positive_by_event[str(rng.choice(self.positive_events))]))
                for _ in range(positive_count)
            ]
            near_count = negative_count // 2 if len(self.near_negative) else 0
            far_count = negative_count - near_count if len(self.far_negative) else 0
            negative = self._sample(rng, self.near_negative, near_count)
            negative += self._sample(rng, self.far_negative, far_count)
            negative += self._sample(rng, self.all_negative, negative_count - len(negative))
            batch = positive + negative
            rng.shuffle(batch)
            yield [(index, self.epoch) for index in batch]
