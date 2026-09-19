from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from bme_eating.features.signal import ppg_quality_features


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
        normalization: Normalization,
        future_context_seconds: int = 0,
        training: bool = False,
        ppg_augmentation_probability: float = 0.0,
        seed: int = 2026,
    ) -> None:
        self.anchors = anchors.reset_index(drop=True)
        self.normalization = normalization
        self.future_context_seconds = int(future_context_seconds)
        self.training = training
        self.ppg_augmentation_probability = float(ppg_augmentation_probability)
        self.seed = int(seed)
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
        block_count = seconds // 15
        raw = raw.reshape(block_count, 750)
        valid = valid.reshape(block_count, 750)
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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        anchor = self.anchors.iloc[index]
        payload = self._load_segment(str(anchor.segment_path))
        timestamp_ms = int(anchor.timestamp_ms)
        rng = np.random.default_rng(self.seed + index)

        motion_start = timestamp_ms - 381_000
        motion, motion_mask = _sample_grid(
            payload["motion_timestamp_ms"],
            payload["motion_values"],
            payload["motion_mask"].astype(bool),
            motion_start,
            381,
            100,
        )
        motion = np.clip(
            (motion - self.normalization.motion_median) / self.normalization.motion_iqr,
            -10.0,
            10.0,
        )
        motion[~motion_mask] = 0.0
        motion = motion.reshape(127, 300, 6)
        motion_mask = motion_mask.reshape(127, 300, 6)
        motion_blocks = np.concatenate(
            (motion, motion_mask.astype(np.float32)), axis=2
        ).transpose(0, 2, 1)
        motion_valid = motion_mask.any(axis=(1, 2)).astype(np.float32)

        ppg_blocks, ppg_quality, ppg_quality_target = self._ppg_blocks(
            payload["ppg_timestamp_ms"],
            payload["ppg_values"],
            payload["ppg_mask"].astype(bool),
            timestamp_ms - 465_000,
            465,
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
            "start_target": torch.tensor(float(anchor.start_target), dtype=torch.float32),
            "end_target": torch.tensor(float(anchor.end_target), dtype=torch.float32),
            "start_loss_mask": torch.tensor(float(anchor.start_loss_mask), dtype=torch.float32),
            "end_loss_mask": torch.tensor(float(anchor.end_loss_mask), dtype=torch.float32),
            "subject_key": str(anchor.subject_key),
            "segment_id": str(anchor.segment_id),
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
            motion_count = self.future_context_seconds // 3
            future_motion = future_motion.reshape(motion_count, 300, 6)
            future_motion_mask = future_motion_mask.reshape(motion_count, 300, 6)
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


class SegmentBalancedBatchSampler(Sampler[list[int]]):
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
        self.positive_by_segment: dict[str, np.ndarray] = {}
        self.negative_by_segment: dict[str, np.ndarray] = {}
        for segment_id, group in anchors.groupby("segment_id", sort=False):
            indices = group.index.to_numpy(dtype=np.int64)
            positive = indices[group["state_target"].to_numpy() > 0]
            negative = indices[group["state_target"].to_numpy() <= 0]
            if len(positive):
                self.positive_by_segment[str(segment_id)] = positive
            if len(negative):
                self.negative_by_segment[str(segment_id)] = negative
        self.positive_segments = np.asarray(list(self.positive_by_segment), dtype=object)
        self.negative_segments = np.asarray(list(self.negative_by_segment), dtype=object)
        self.positive_weights = np.asarray(
            [len(self.positive_by_segment[str(key)]) for key in self.positive_segments],
            dtype=np.float64,
        )
        self.negative_weights = np.asarray(
            [len(self.negative_by_segment[str(key)]) for key in self.negative_segments],
            dtype=np.float64,
        )
        if len(self.positive_segments) == 0 or len(self.negative_segments) == 0:
            raise ValueError(
                "Training anchors must contain both positive and negative segments"
            )
        self.positive_weights /= self.positive_weights.sum()
        self.negative_weights /= self.negative_weights.sum()

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
        for _ in range(self.steps_per_epoch):
            use_positive = len(self.positive_segments) > 0 and rng.random() < self.positive_fraction
            if use_positive:
                segment_id = str(
                    rng.choice(self.positive_segments, p=self.positive_weights)
                )
                positive = self._sample(
                    rng, self.positive_by_segment[segment_id], positive_count
                )
                negative_pool = self.negative_by_segment.get(segment_id)
                if negative_pool is None or len(negative_pool) == 0:
                    negative = self._sample(
                        rng,
                        self.positive_by_segment[segment_id],
                        self.batch_size - len(positive),
                    )
                else:
                    negative = self._sample(
                        rng, negative_pool, self.batch_size - len(positive)
                    )
                batch = positive + negative
            else:
                segment_id = str(
                    rng.choice(self.negative_segments, p=self.negative_weights)
                )
                batch = self._sample(
                    rng, self.negative_by_segment[segment_id], self.batch_size
                )
            rng.shuffle(batch)
            yield batch
