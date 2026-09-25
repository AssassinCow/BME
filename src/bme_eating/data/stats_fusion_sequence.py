from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from bme_eating.data.deep_dataset import Normalization
from bme_eating.data.session import SessionWindowReader
from bme_eating.data.stats_fusion_preprocess import (
    build_motion_blocks,
    build_ppg_blocks,
)
from bme_eating.metrics import partition_evaluation_events


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
        long_history = (
            self.long_receptive_field_tokens * self.long_pool_factor
            + self.long_pool_factor
            - 1
        )
        return max(short_history, long_history)

    @property
    def total_steps(self) -> int:
        return self.history_steps + self.supervised_steps


class ClipMixtureSampler(Sampler[tuple[int, int, np.ndarray]]):
    def __init__(
        self,
        anchors: pd.DataFrame,
        *,
        samples_per_epoch: int,
        mixture: dict[str, float],
        seed: int,
        supervised_steps: int = 256,
    ) -> None:
        required = {
            "state_target",
            "start_target",
            "end_target",
            "state_loss_mask",
        }
        missing = required - set(anchors.columns)
        if missing:
            raise ValueError(f"Clip sampler anchors are missing columns: {sorted(missing)}")
        if set(mixture) != {"uniform", "event", "boundary"}:
            raise ValueError("Clip mixture must define uniform, event, and boundary")
        weights = np.asarray([mixture[name] for name in ("uniform", "event", "boundary")])
        if (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Clip sampling mixture must be non-negative and sum to one")
        eligible = anchors["state_loss_mask"].fillna(0.0).to_numpy(dtype=float) > 0
        start_eligible = (
            anchors.get("start_loss_mask", pd.Series(1.0, index=anchors.index))
            .fillna(0.0)
            .to_numpy(dtype=float)
            > 0
        )
        end_eligible = (
            anchors.get("end_loss_mask", pd.Series(1.0, index=anchors.index))
            .fillna(0.0)
            .to_numpy(dtype=float)
            > 0
        )
        indices = np.arange(len(anchors), dtype=np.int64)
        pools = {
            "uniform": indices[eligible],
            "event": indices[eligible & (anchors["state_target"].to_numpy(dtype=float) > 0)],
            "boundary": indices[
                eligible
                & (
                    (
                        (anchors["start_target"].to_numpy(dtype=float) > 0)
                        & start_eligible
                    )
                    | (
                        (anchors["end_target"].to_numpy(dtype=float) > 0)
                        & end_eligible
                    )
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
        self.supervised_steps = int(supervised_steps)
        if self.supervised_steps <= 0:
            raise ValueError("supervised_steps must be positive")
        self.inclusion_importance = np.zeros(len(anchors), dtype=np.float64)
        if {"subject_key", "session_id"}.issubset(anchors.columns):
            grouping = {}
            for key, group in anchors.groupby(["subject_key", "session_id"], sort=False):
                if "timestamp_ms" in group:
                    group = group.sort_values("timestamp_ms", kind="stable")
                grouping[key] = group.index.to_numpy(dtype=np.int64)
        else:
            grouping = {("all", "all"): np.arange(len(anchors), dtype=np.int64)}
        natural_endpoint_probability = np.where(eligible, natural, 0.0)
        for group_indices in grouping.values():
            ordered = np.asarray(group_indices, dtype=np.int64)
            sampled = probabilities[ordered]
            natural_sampled = natural_endpoint_probability[ordered]
            sampled_inclusion = np.convolve(
                sampled[::-1], np.ones(self.supervised_steps), mode="full"
            )[: len(ordered)][::-1]
            natural_inclusion = np.convolve(
                natural_sampled[::-1], np.ones(self.supervised_steps), mode="full"
            )[: len(ordered)][::-1]
            ratio = np.divide(
                natural_inclusion,
                sampled_inclusion,
                out=np.zeros_like(natural_inclusion),
                where=sampled_inclusion > 0,
            )
            self.inclusion_importance[ordered] = ratio
        self._session_positions: dict[int, tuple[np.ndarray, int]] = {}
        for group_indices in grouping.values():
            ordered = np.asarray(group_indices, dtype=np.int64)
            for position, row_index in enumerate(ordered):
                self._session_positions[int(row_index)] = (ordered, position)
        self.pools = pools
        self.weights = weights
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self) -> Iterator[tuple[int, int, np.ndarray]]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        names = np.asarray(("uniform", "event", "boundary"), dtype=object)
        for _ in range(self.samples_per_epoch):
            category = str(rng.choice(names, p=self.weights))
            index = int(rng.choice(self.pools[category]))
            ordered, position = self._session_positions[index]
            first = max(0, position - self.supervised_steps + 1)
            weights = np.zeros(self.supervised_steps, dtype=np.float32)
            selected = ordered[first : position + 1]
            weights[-len(selected) :] = self.inclusion_importance[selected].astype(np.float32)
            yield index, self.epoch, weights


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
        self.geometry = geometry or SequenceGeometry()
        for name, default in (
            ("start_target", 0.0),
            ("end_target", 0.0),
            ("start_loss_mask", 1.0),
            ("end_loss_mask", 1.0),
        ):
            if name not in self.anchors:
                self.anchors[name] = default
        required_event_columns = {"subject_key", "start_ms", "end_ms", "valid_duration"}
        missing_event_columns = required_event_columns - set(events.columns)
        if len(events) and missing_event_columns:
            raise ValueError(
                f"Sequence events are missing columns: {sorted(missing_event_columns)}"
            )
        if len(events) and "evaluable" not in events and "coverage" not in events:
            raise ValueError("Sequence events require evaluable or coverage")
        if not len(events):
            events = events.reindex(
                columns=sorted(set(events.columns) | required_event_columns | {"evaluable"})
            )
        subjects = set(self.anchors["subject_key"].astype(str))
        self.events, self.ignore_events = partition_evaluation_events(events, subjects)
        step_ms = self.geometry.step_seconds * 1000
        for event in self.ignore_events.itertuples(index=False):
            subject = self.anchors["subject_key"].astype(str).eq(str(event.subject_key))
            timestamps = self.anchors["timestamp_ms"].to_numpy(dtype=np.int64)
            state_ignore = subject & (timestamps > int(event.start_ms)) & (
                timestamps - step_ms < int(event.end_ms)
            )
            onset_ignore = subject & (timestamps >= int(event.start_ms)) & (
                timestamps <= int(event.start_ms) + 30_000
            )
            offset_ignore = subject & (timestamps >= int(event.end_ms)) & (
                timestamps <= int(event.end_ms) + 60_000
            )
            self.anchors.loc[state_ignore, "state_loss_mask"] = 0.0
            for target, mask, selected in (
                ("start_target", "start_loss_mask", onset_ignore),
                ("end_target", "end_loss_mask", offset_ignore),
            ):
                if target in self.anchors:
                    self.anchors.loc[selected, target] = 0.0
                if mask not in self.anchors:
                    self.anchors[mask] = 1.0
                self.anchors.loc[selected, mask] = 0.0
        self.normalization = normalization
        self.statistics_columns = tuple(str(value) for value in statistics_columns)
        if len(self.statistics_columns) != 24:
            raise ValueError("Sequence dataset requires 12 scaled values and 12 missing flags")
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
        step_ms = self.geometry.step_seconds * 1000
        self.session_origins = {
            key: int(group["timestamp_ms"].iloc[0]) - step_ms
            for key, group in self.session_groups.items()
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
        indices = self._nearest_indices(source_time, timestamps, self.geometry.step_seconds * 500)
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
            source = group.get(field, pd.Series(default, index=group.index)).to_numpy(
                dtype=np.float32
            )
            output = np.full(len(timestamps), default, dtype=np.float32)
            output[valid] = source[indices[valid]]
            arrays[field] = output
        statistics = np.zeros((len(timestamps), len(self.statistics_columns)), dtype=np.float32)
        statistics[:, len(self.statistics_columns) // 2 :] = 1.0
        source_statistics = group.loc[:, self.statistics_columns].to_numpy(dtype=np.float32)
        statistics[valid] = source_statistics[indices[valid]]
        arrays["statistics"] = statistics
        return valid, arrays

    def _motion_blocks(
        self, payload: dict[str, np.ndarray], first_timestamp_ms: int, steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return build_motion_blocks(
            timestamp_ms=payload["motion_timestamp_ms"],
            values=payload["motion_values"],
            mask=payload["motion_mask"].astype(bool),
            normalization=self.normalization,
            first_timestamp_ms=first_timestamp_ms,
            steps=steps,
            step_seconds=self.geometry.step_seconds,
        )

    def _ppg_blocks(
        self,
        payload: dict[str, np.ndarray],
        timestamps: np.ndarray,
        session_origin_ms: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return build_ppg_blocks(
            timestamp_ms=payload["ppg_timestamp_ms"],
            values=payload["ppg_values"],
            mask=payload["ppg_mask"].astype(bool),
            normalization=self.normalization,
            timestamps=timestamps,
            session_origin_ms=session_origin_ms,
            step_ms=self.geometry.step_seconds * 1000,
            factor=self.geometry.long_pool_factor,
            modality_dropout=(self.training and rng.random() < self.ppg_modality_dropout),
        )

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
        ignored = self.ignore_events[self.ignore_events["subject_key"].astype(str) == subject_key]
        for event in ignored.itertuples(index=False):
            start_delta = (timestamps - int(event.start_ms)) / 1000.0
            end_delta = (timestamps - int(event.end_ms)) / 1000.0
            smooth[(np.abs(start_delta) <= 30.0) | (np.abs(end_delta) <= 30.0)] = 0.0
        return onset.astype(np.float32), offset.astype(np.float32), smooth

    def __getitem__(
        self, index: int | tuple[int, int, np.ndarray]
    ) -> dict[str, torch.Tensor | str | int]:
        if isinstance(index, tuple):
            row_index, epoch, endpoint_importance = index
        else:
            row_index, epoch = index, 0
            endpoint_importance = np.ones(self.geometry.supervised_steps, dtype=np.float32)
        row_index = int(row_index)
        anchor = self.anchors.iloc[row_index]
        subject_key, session_id = self.row_location[row_index]
        group = self.session_groups[(subject_key, session_id)]
        end_timestamp = int(anchor.timestamp_ms)
        step_ms = self.geometry.step_seconds * 1000
        session_origin_ms = self.session_origins[(subject_key, session_id)]
        timestamps = (
            end_timestamp
            - np.arange(self.geometry.total_steps - 1, -1, -1, dtype=np.int64) * step_ms
        )
        payload = self.reader.read(
            session_id,
            int(
                timestamps[0]
                - 2 * self.geometry.long_pool_factor * self.geometry.step_seconds * 1000
            ),
            int(timestamps[-1]),
        )
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, row_index, int(epoch)]))
        motion, motion_valid = self._motion_blocks(payload, int(timestamps[0]), len(timestamps))
        ppg, ppg_quality, ppg_valid, ppg_mapping, block_end_indices = self._ppg_blocks(
            payload,
            timestamps,
            session_origin_ms,
            rng,
        )
        aligned_valid, arrays = self._aligned_anchor_arrays(group, timestamps)
        onset, offset, smooth = self._transition_targets(subject_key, timestamps)
        supervision = np.zeros(len(timestamps), dtype=np.float32)
        supervision[-self.geometry.supervised_steps :] = 1.0
        supervision *= aligned_valid.astype(np.float32)
        importance = np.zeros(len(timestamps), dtype=np.float32)
        importance[-self.geometry.supervised_steps :] = np.asarray(
            endpoint_importance, dtype=np.float32
        )
        result: dict[str, torch.Tensor | str | int] = {
            "motion_blocks": torch.from_numpy(motion),
            "motion_valid": torch.from_numpy(motion_valid),
            "ppg_blocks": torch.from_numpy(ppg),
            "ppg_quality": torch.from_numpy(ppg_quality),
            "ppg_valid": torch.from_numpy(ppg_valid),
            "ppg_to_motion_index": torch.from_numpy(ppg_mapping),
            "long_block_end_indices": torch.from_numpy(block_end_indices),
            "statistics": torch.from_numpy(arrays["statistics"]),
            "state_target": torch.from_numpy(arrays["state_target"]),
            "state_loss_mask": torch.from_numpy(arrays["state_loss_mask"]),
            "onset_target": torch.from_numpy(onset),
            "offset_target": torch.from_numpy(offset),
            "onset_loss_mask": torch.from_numpy(arrays["start_loss_mask"]),
            "offset_loss_mask": torch.from_numpy(arrays["end_loss_mask"]),
            "smooth_mask": torch.from_numpy(smooth),
            "supervision_mask": torch.from_numpy(supervision),
            "importance_weight": torch.from_numpy(importance),
            "motion_present": torch.from_numpy((motion_valid > 0).astype(np.float32)),
            "timestamp_ms": torch.from_numpy(timestamps),
            "subject_key": subject_key,
            "session_id": session_id,
            "end_timestamp_ms": end_timestamp,
        }
        return result
