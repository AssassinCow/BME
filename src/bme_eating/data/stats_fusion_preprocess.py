from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from bme_eating.data.deep_dataset import Normalization, _sample_grid
from bme_eating.features.baseline import _motion_window_features, _ppg_window_features
from bme_eating.features.signal import ppg_quality_features
from bme_eating.stats_features import STATS_FEATURE_COLUMNS, FoldRobustScaler


def session_right_endpoint_grid(
    first_observation_ms: int, last_observation_ms: int, step_ms: int
) -> np.ndarray:
    if step_ms <= 0:
        raise ValueError("Session anchor step must be positive")
    first = int(first_observation_ms)
    last = int(last_observation_ms)
    start = first + int(step_ms)
    end = first + ((last - first) // int(step_ms)) * int(step_ms)
    return (
        np.arange(start, end + 1, int(step_ms), dtype=np.int64)
        if start <= end
        else np.empty(0, dtype=np.int64)
    )


def causal_completed_block_layout(
    timestamps: np.ndarray,
    *,
    session_origin_ms: int,
    step_ms: int,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.asarray(timestamps, dtype=np.int64).reshape(-1)
    if step_ms <= 0 or factor <= 0:
        raise ValueError("Completed-block step and factor must be positive")
    if len(timestamps) < factor:
        raise ValueError("Sequence is too short to form one completed block")
    if np.any(np.diff(timestamps) != int(step_ms)):
        raise ValueError("Completed-block timestamps must use a uniform high-rate grid")
    if np.any((timestamps - int(session_origin_ms)) % int(step_ms) != 0):
        raise ValueError("Completed-block timestamps are not aligned to the session origin")

    block_count = len(timestamps) // int(factor)
    block_ms = int(step_ms) * int(factor)
    last_ordinal = (int(timestamps[-1]) - int(session_origin_ms)) // block_ms
    ordinals = np.arange(
        last_ordinal - block_count + 1,
        last_ordinal + 1,
        dtype=np.int64,
    )
    block_ends = int(session_origin_ms) + ordinals * block_ms

    candidate_indices = np.searchsorted(timestamps, block_ends, side="left")
    in_bounds = candidate_indices < len(timestamps)
    exact = np.zeros(block_count, dtype=bool)
    exact[in_bounds] = timestamps[candidate_indices[in_bounds]] == block_ends[in_bounds]
    complete = exact & (candidate_indices >= int(factor) - 1)
    block_end_indices = np.where(complete, candidate_indices, -1).astype(np.int64)

    mapping = np.searchsorted(block_ends, timestamps, side="right") - 1
    mapping[(mapping < 0) | (mapping >= block_count)] = -1
    return block_ends, block_end_indices, mapping.astype(np.int64)


def build_motion_blocks(
    *,
    timestamp_ms: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    normalization: Normalization,
    first_timestamp_ms: int,
    steps: int,
    step_seconds: int,
) -> tuple[np.ndarray, np.ndarray]:
    sampled, sampled_mask = _sample_grid(
        timestamp_ms,
        values,
        mask,
        first_timestamp_ms - step_seconds * 1000,
        steps * step_seconds,
        100,
    )
    sampled = np.clip(
        (sampled - normalization.motion_median) / normalization.motion_iqr,
        -10.0,
        10.0,
    )
    sampled[~sampled_mask] = 0.0
    samples = step_seconds * 100
    sampled = sampled.reshape(steps, samples, 6)
    sampled_mask = sampled_mask.reshape(steps, samples, 6)
    blocks = np.concatenate((sampled, sampled_mask.astype(np.float32)), axis=2).transpose(0, 2, 1)
    return blocks.astype(np.float32), sampled_mask.mean(axis=(1, 2)).astype(np.float32)


def build_ppg_blocks(
    *,
    timestamp_ms: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    normalization: Normalization,
    timestamps: np.ndarray,
    session_origin_ms: int,
    step_ms: int,
    factor: int,
    modality_dropout: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    block_ends, block_end_indices, mapping = causal_completed_block_layout(
        timestamps,
        session_origin_ms=session_origin_ms,
        step_ms=step_ms,
        factor=factor,
    )
    block_ms = int(step_ms) * int(factor)
    if block_ms % 1000:
        raise ValueError("PPG completed-block duration must use whole seconds")
    raw, sampled_mask = _sample_grid(
        timestamp_ms,
        np.asarray(values).reshape(-1, 1),
        np.asarray(mask, dtype=bool).reshape(-1, 1),
        int(block_ends[0] - block_ms),
        int(len(block_ends) * block_ms // 1000),
        50,
    )
    samples_per_block = block_ms * 50 // 1000
    raw = raw.reshape(len(block_ends), samples_per_block)
    sampled_mask = sampled_mask.reshape(len(block_ends), samples_per_block)
    quality = np.zeros((len(block_ends), 8), dtype=np.float32)
    valid = sampled_mask.mean(axis=1).astype(np.float32)
    for index in range(len(block_ends)):
        quality[index], _ = ppg_quality_features(raw[index], sampled_mask[index], 50.0)
    normalized = np.clip(
        (raw - normalization.ppg_median) / normalization.ppg_iqr,
        -10.0,
        10.0,
    )
    normalized[~sampled_mask] = 0.0
    blocks = np.stack((normalized, sampled_mask.astype(np.float32)), axis=1).astype(np.float32)
    if modality_dropout:
        blocks.fill(0.0)
        quality.fill(0.0)
        valid.fill(0.0)
    return blocks, quality, valid, mapping, block_end_indices


@dataclass(frozen=True)
class RawSessionInput:
    subject_key: str
    session_id: str
    motion_timestamp_ms: np.ndarray
    motion_values: np.ndarray
    motion_mask: np.ndarray
    ppg_timestamp_ms: np.ndarray
    ppg_values: np.ndarray
    ppg_mask: np.ndarray

    def validated(self) -> RawSessionInput:
        motion_time = np.asarray(self.motion_timestamp_ms, dtype=np.int64).reshape(-1)
        motion_values = np.asarray(self.motion_values, dtype=np.float32)
        motion_mask = np.asarray(self.motion_mask, dtype=bool)
        ppg_time = np.asarray(self.ppg_timestamp_ms, dtype=np.int64).reshape(-1)
        ppg_values = np.asarray(self.ppg_values, dtype=np.float32).reshape(-1)
        ppg_mask = np.asarray(self.ppg_mask, dtype=bool).reshape(-1)
        if not str(self.subject_key) or not str(self.session_id):
            raise ValueError("Raw session subject_key and session_id must be non-empty")
        if motion_values.ndim != 2 or motion_values.shape[1] != 6:
            raise ValueError("Raw motion values must have shape (n, 6)")
        if motion_mask.shape != motion_values.shape or len(motion_time) != len(motion_values):
            raise ValueError("Raw motion timestamps, values, and mask must align")
        if len(ppg_time) != len(ppg_values) or len(ppg_values) != len(ppg_mask):
            raise ValueError("Raw PPG timestamps, values, and mask must align")
        for name, timestamps in (("motion", motion_time), ("PPG", ppg_time)):
            if len(timestamps) and np.any(np.diff(timestamps) <= 0):
                raise ValueError(f"Raw {name} timestamps must be strictly increasing")
        if not len(motion_time) and not len(ppg_time):
            raise ValueError("Raw session contains no observations")
        if np.any(motion_mask & ~np.isfinite(motion_values)):
            raise ValueError("Valid raw motion samples must be finite")
        if np.any(ppg_mask & ~np.isfinite(ppg_values)):
            raise ValueError("Valid raw PPG samples must be finite")
        motion_values = np.where(motion_mask, motion_values, 0.0).astype(np.float32)
        ppg_values = np.where(ppg_mask, ppg_values, 0.0).astype(np.float32)
        return RawSessionInput(
            str(self.subject_key),
            str(self.session_id),
            motion_time,
            motion_values,
            motion_mask,
            ppg_time,
            ppg_values,
            ppg_mask,
        )


class StatsFusionRawSessionPreprocessor:
    def __init__(
        self,
        *,
        normalization: Normalization,
        statistics_scaler: FoldRobustScaler,
        geometry: Any,
    ) -> None:
        self.normalization = normalization
        self.statistics_scaler = statistics_scaler
        self.geometry = geometry

    def anchors(self, session: RawSessionInput) -> np.ndarray:
        observed = [
            timestamps
            for timestamps in (session.motion_timestamp_ms, session.ppg_timestamp_ms)
            if len(timestamps)
        ]
        first = min(int(values[0]) for values in observed)
        last = max(int(values[-1]) for values in observed)
        step_ms = self.geometry.step_seconds * 1000
        return session_right_endpoint_grid(first, last, step_ms)

    def _statistics(self, session: RawSessionInput, timestamps: np.ndarray) -> np.ndarray:
        raw = np.full((len(timestamps), len(STATS_FEATURE_COLUMNS)), np.nan, dtype=np.float64)
        for row, end_ms in enumerate(timestamps):
            start_ms = int(end_ms) - 15_000
            motion_slice = slice(
                int(np.searchsorted(session.motion_timestamp_ms, start_ms, side="left")),
                int(np.searchsorted(session.motion_timestamp_ms, end_ms, side="right")),
            )
            ppg_slice = slice(
                int(np.searchsorted(session.ppg_timestamp_ms, start_ms, side="left")),
                int(np.searchsorted(session.ppg_timestamp_ms, end_ms, side="right")),
            )
            features = _motion_window_features(
                session.motion_values[motion_slice],
                session.motion_mask[motion_slice],
                100.0,
            )
            features.update(
                _ppg_window_features(
                    session.ppg_values[ppg_slice],
                    session.ppg_mask[ppg_slice],
                    50.0,
                )
            )
            for column, name in enumerate(STATS_FEATURE_COLUMNS):
                if name in features:
                    raw[row, column] = float(features[name])
        return self.statistics_scaler.transform_array(raw)

    def _motion_blocks(
        self, session: RawSessionInput, first_timestamp_ms: int, steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return build_motion_blocks(
            timestamp_ms=session.motion_timestamp_ms,
            values=session.motion_values,
            mask=session.motion_mask,
            normalization=self.normalization,
            first_timestamp_ms=first_timestamp_ms,
            steps=steps,
            step_seconds=self.geometry.step_seconds,
        )

    def _ppg_blocks(
        self,
        session: RawSessionInput,
        timestamps: np.ndarray,
        session_origin_ms: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return build_ppg_blocks(
            timestamp_ms=session.ppg_timestamp_ms,
            values=session.ppg_values,
            mask=session.ppg_mask,
            normalization=self.normalization,
            timestamps=timestamps,
            session_origin_ms=session_origin_ms,
            step_ms=self.geometry.step_seconds * 1000,
            factor=self.geometry.long_pool_factor,
        )

    def iter_state_batches(self, raw_session: RawSessionInput) -> Iterator[dict[str, Any]]:
        session = raw_session.validated()
        anchors = self.anchors(session)
        if not len(anchors):
            return
        supervised = self.geometry.supervised_steps
        step_ms = self.geometry.step_seconds * 1000
        session_origin_ms = int(anchors[0]) - step_ms
        endpoints = list(range(min(supervised - 1, len(anchors) - 1), len(anchors), supervised))
        if endpoints[-1] != len(anchors) - 1:
            endpoints.append(len(anchors) - 1)
        for endpoint in endpoints:
            end_timestamp = int(anchors[endpoint])
            timestamps = (
                end_timestamp
                - np.arange(self.geometry.total_steps - 1, -1, -1, dtype=np.int64) * step_ms
            )
            motion, motion_valid = self._motion_blocks(session, int(timestamps[0]), len(timestamps))
            ppg, ppg_quality, ppg_valid, ppg_mapping, block_end_indices = self._ppg_blocks(
                session,
                timestamps,
                session_origin_ms,
            )
            statistics = np.zeros((len(timestamps), 24), dtype=np.float32)
            statistics[:, len(STATS_FEATURE_COLUMNS) :] = 1.0
            observed = (timestamps >= anchors[0]) & (timestamps <= anchors[-1])
            statistics[observed] = self._statistics(session, timestamps[observed])
            supervision = np.zeros(len(timestamps), dtype=np.float32)
            first_supervised = max(0, endpoint - supervised + 1)
            supervised_timestamps = {
                int(value) for value in anchors[first_supervised : endpoint + 1]
            }
            supervision[[int(value) in supervised_timestamps for value in timestamps]] = 1.0
            yield {
                "motion_blocks": torch.from_numpy(motion).unsqueeze(0),
                "motion_valid": torch.from_numpy(motion_valid).unsqueeze(0),
                "motion_present": torch.from_numpy((motion_valid > 0).astype(np.float32)).unsqueeze(
                    0
                ),
                "ppg_blocks": torch.from_numpy(ppg).unsqueeze(0),
                "ppg_quality": torch.from_numpy(ppg_quality).unsqueeze(0),
                "ppg_valid": torch.from_numpy(ppg_valid).unsqueeze(0),
                "ppg_to_motion_index": torch.from_numpy(ppg_mapping).unsqueeze(0),
                "long_block_end_indices": torch.from_numpy(block_end_indices).unsqueeze(0),
                "statistics": torch.from_numpy(statistics).unsqueeze(0),
                "timestamp_ms": torch.from_numpy(timestamps).unsqueeze(0),
                "supervision_mask": torch.from_numpy(supervision).unsqueeze(0),
            }
