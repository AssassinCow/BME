from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, Sampler

CATEGORY_ORDER = (
    "positive",
    "near_miss",
    "hard_false_positive",
    "random_background",
)


class ProposalBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        categories: np.ndarray,
        batch_size: int,
        ratios: dict[str, float],
        steps_per_epoch: int,
        seed: int,
    ) -> None:
        self.categories = np.asarray(categories, dtype=str)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size <= 0 or self.steps_per_epoch <= 0:
            raise ValueError("Proposal batch size and steps must be positive")
        if set(ratios) != set(CATEGORY_ORDER):
            raise ValueError("Proposal sampling ratios must define all four categories")
        ratio_values = np.asarray([float(ratios[name]) for name in CATEGORY_ORDER])
        if (ratio_values < 0).any() or not np.isclose(ratio_values.sum(), 1.0):
            raise ValueError("Proposal sampling ratios must be non-negative and sum to one")
        raw = ratio_values * self.batch_size
        counts = np.floor(raw).astype(int)
        remainder = self.batch_size - int(counts.sum())
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remainder]] += 1
        pools = {
            name: np.flatnonzero(self.categories == name) for name in CATEGORY_ORDER
        }
        unknown = sorted(set(self.categories) - set(CATEGORY_ORDER))
        if unknown:
            raise ValueError(f"Unknown proposal sampling categories: {unknown}")
        if not any(len(pool) for pool in pools.values()):
            raise ValueError("Proposal sampler received no rows")
        for index, name in enumerate(CATEGORY_ORDER):
            if len(pools[name]) or counts[index] == 0:
                continue
            missing = int(counts[index])
            counts[index] = 0
            available = [
                offset
                for offset in range(1, len(CATEGORY_ORDER) + 1)
                if len(pools[CATEGORY_ORDER[(index + offset) % len(CATEGORY_ORDER)]])
            ]
            for slot in range(missing):
                counts[(index + available[slot % len(available)]) % len(CATEGORY_ORDER)] += 1
        self.counts = {name: int(counts[index]) for index, name in enumerate(CATEGORY_ORDER)}
        self.pools = pools

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        generator = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch])
        )
        for _ in range(self.steps_per_epoch):
            parts = [
                generator.choice(self.pools[name], size=count, replace=True)
                for name, count in self.counts.items()
                if count
            ]
            batch = np.concatenate(parts)
            generator.shuffle(batch)
            yield batch.astype(int).tolist()


@dataclass(frozen=True)
class ProposalFeatureBatch:
    proposal_ids: np.ndarray
    sequence: np.ndarray
    scalar: np.ndarray
    event_target: np.ndarray | None = None
    iou_target: np.ndarray | None = None


class ProposalTensorDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(self, features: ProposalFeatureBatch) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features.proposal_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item: dict[str, torch.Tensor | str] = {
            "proposal_id": str(self.features.proposal_ids[index]),
            "sequence": torch.from_numpy(self.features.sequence[index]),
            "scalar": torch.from_numpy(self.features.scalar[index]),
        }
        if self.features.event_target is not None:
            item["event_target"] = torch.tensor(
                float(self.features.event_target[index]), dtype=torch.float32
            )
        if self.features.iou_target is not None:
            item["iou_target"] = torch.tensor(
                float(self.features.iou_target[index]), dtype=torch.float32
            )
        return item


class EventVerifier(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        scalar_dim: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        channels = int(config.get("hidden_channels", 64))
        hidden = int(config.get("hidden_dim", 128))
        dropout = float(config.get("dropout", 0.1))
        self.sequence_encoder = nn.Sequential(
            nn.Conv1d(sequence_dim, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
            nn.SiLU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(2 * channels + scalar_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.SiLU(),
        )
        self.event_head = nn.Linear(64, 1)
        self.iou_head = nn.Linear(64, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sequence = self.sequence_encoder(batch["sequence"].transpose(1, 2))
        pooled = torch.cat((sequence.mean(dim=-1), sequence.amax(dim=-1)), dim=-1)
        hidden = self.projection(torch.cat((pooled, batch["scalar"]), dim=-1))
        event_logit = self.event_head(hidden).squeeze(-1)
        iou_logit = self.iou_head(hidden).squeeze(-1)
        return {
            "event_logit": event_logit,
            "iou_logit": iou_logit,
            "predicted_iou": torch.sigmoid(iou_logit),
        }


def verifier_loss(
    output: dict[str, torch.Tensor],
    event_target: torch.Tensor,
    iou_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    event = nn.functional.binary_cross_entropy_with_logits(
        output["event_logit"], event_target
    )
    quality = nn.functional.smooth_l1_loss(output["predicted_iou"], iou_target)
    return event + quality, {"event": event, "quality": quality}


def _sampling_times(
    start_ms: int,
    end_ms: int,
    config: dict[str, Any],
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    specifications = (
        (start_ms - int(config["left_context_seconds"]) * 1000, start_ms, int(config["left_bins"])),
        (start_ms, end_ms, int(config["event_bins"])),
        (end_ms, end_ms + int(config["right_context_seconds"]) * 1000, int(config["right_bins"])),
    )
    for left, right, count in specifications:
        edges = np.linspace(left, right, count + 1)
        pieces.append(0.5 * (edges[:-1] + edges[1:]))
    return np.concatenate(pieces).astype(np.int64)


def _nearest_rows(frame: pd.DataFrame, times: np.ndarray, columns: list[str]) -> np.ndarray:
    timestamp = frame["timestamp_ms"].to_numpy(dtype=np.int64)
    values = frame[columns].to_numpy(dtype=np.float32)
    if len(timestamp) == 0:
        return np.zeros((len(times), len(columns)), dtype=np.float32)
    right = np.searchsorted(timestamp, times, side="left")
    right = np.clip(right, 0, len(timestamp) - 1)
    left = np.clip(right - 1, 0, len(timestamp) - 1)
    choose_left = np.abs(timestamp[left] - times) <= np.abs(timestamp[right] - times)
    indices = np.where(choose_left, left, right)
    sampled = values[indices]
    outside = (times < timestamp[0]) | (times > timestamp[-1])
    sampled[outside] = 0.0
    return np.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)


def _stable_summary(
    feature_frame: pd.DataFrame,
    start_ms: int,
    end_ms: int,
    columns: list[str],
) -> np.ndarray:
    selected = feature_frame[
        feature_frame["timestamp_ms"].between(start_ms, end_ms, inclusive="both")
    ]
    if selected.empty:
        return np.zeros(7 * len(columns), dtype=np.float32)
    values = selected[columns].to_numpy(dtype=np.float64)
    statistics = (
        np.nanmean(values, axis=0),
        np.nanstd(values, axis=0),
        np.nanmin(values, axis=0),
        np.nanmax(values, axis=0),
        np.nanquantile(values, 0.25, axis=0),
        np.nanquantile(values, 0.50, axis=0),
        np.nanquantile(values, 0.75, axis=0),
    )
    return np.nan_to_num(np.concatenate(statistics), nan=0.0).astype(np.float32)


def build_proposal_features(
    proposals: pd.DataFrame,
    windows: pd.DataFrame,
    stable_features: pd.DataFrame,
    stable_columns: list[str],
    config: dict[str, Any],
) -> ProposalFeatureBatch:
    required_windows = {
        "subject_key",
        "session_id",
        "timestamp_ms",
        "state_probability",
        "start_probability",
        "end_probability",
        "ppg_gate_mean",
        "ppg_valid_fraction",
        "motion_valid_fraction",
        "missing_fraction",
    }
    missing = required_windows - set(windows.columns)
    if missing:
        raise ValueError(f"Window predictions are missing verifier fields: {sorted(missing)}")
    missing_stable = set(stable_columns) - set(stable_features.columns)
    if missing_stable:
        raise ValueError(f"Stable feature frame is missing columns: {sorted(missing_stable)}")
    use_embedding = bool(config.get("use_state_embedding", True))
    embedding_columns = sorted(
        column for column in windows.columns if column.startswith("state_embedding_")
    )
    if use_embedding and not embedding_columns:
        raise ValueError("Verifier requested state embeddings but none were provided")
    sequence_columns = [
        "state_probability",
        "start_probability",
        "end_probability",
        "ppg_gate_mean",
        "ppg_valid_fraction",
        "motion_valid_fraction",
        "missing_fraction",
        *(embedding_columns if use_embedding else []),
    ]
    window_groups = {
        (str(subject), str(session)): group.sort_values("timestamp_ms")
        for (subject, session), group in windows.groupby(
            ["subject_key", "session_id"], sort=False
        )
    }
    feature_groups = {
        (str(subject), str(session)): group.sort_values("timestamp_ms")
        for (subject, session), group in stable_features.groupby(
            ["subject_key", "session_id"], sort=False
        )
    }
    sequences: list[np.ndarray] = []
    scalars: list[np.ndarray] = []
    identifiers: list[str] = []
    for proposal in proposals.itertuples(index=False):
        key = (str(proposal.subject_key), str(proposal.session_id))
        if key not in window_groups or key not in feature_groups:
            raise ValueError(f"Proposal has no aligned window features: {proposal.proposal_id}")
        start = int(proposal.coarse_start_ms)
        end = int(proposal.coarse_end_ms)
        times = _sampling_times(start, end, config)
        sequence = _nearest_rows(window_groups[key], times, sequence_columns)
        local = window_groups[key]
        inside = local[local["timestamp_ms"].between(start, end, inclusive="both")]
        state = inside["state_probability"].to_numpy(dtype=np.float64)
        start_hint = inside["start_probability"].to_numpy(dtype=np.float64)
        end_hint = inside["end_probability"].to_numpy(dtype=np.float64)
        missing_value = inside["missing_fraction"].mean() if len(inside) else 1.0
        source_mask = int(proposal.source_mask)
        scalar = np.asarray(
            [
                math_log_duration(end - start),
                float(proposal.generator_score),
                float(state.mean()) if len(state) else 0.0,
                float(state.max()) if len(state) else 0.0,
                float(start_hint.max()) if len(start_hint) else 0.0,
                float(end_hint.max()) if len(end_hint) else 0.0,
                float(missing_value),
                float(bool(source_mask & 1)),
                float(bool(source_mask & 2)),
                float(bool(source_mask & 4)),
                float(bool(source_mask & 8)),
            ],
            dtype=np.float32,
        )
        stable = _stable_summary(feature_groups[key], start, end, stable_columns)
        identifiers.append(str(proposal.proposal_id))
        sequences.append(sequence)
        scalars.append(np.concatenate((scalar, stable)).astype(np.float32))
    sequence_shape = (
        int(config["left_bins"]) + int(config["event_bins"]) + int(config["right_bins"]),
        len(sequence_columns),
    )
    sequence_array = (
        np.stack(sequences).astype(np.float32)
        if sequences
        else np.empty((0, *sequence_shape), dtype=np.float32)
    )
    scalar_dim = 11 + 7 * len(stable_columns)
    scalar_array = (
        np.stack(scalars).astype(np.float32)
        if scalars
        else np.empty((0, scalar_dim), dtype=np.float32)
    )
    event_target = (
        proposals["is_positive"].to_numpy(dtype=np.float32)
        if "is_positive" in proposals
        else None
    )
    iou_target = (
        proposals["max_iou"].to_numpy(dtype=np.float32) if "max_iou" in proposals else None
    )
    return ProposalFeatureBatch(
        proposal_ids=np.asarray(identifiers, dtype=object),
        sequence=sequence_array,
        scalar=scalar_array,
        event_target=event_target,
        iou_target=iou_target,
    )


def math_log_duration(duration_ms: int) -> float:
    return float(np.log1p(max(0, duration_ms) / 1000.0))
