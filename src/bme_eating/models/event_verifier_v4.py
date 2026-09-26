from __future__ import annotations

from collections.abc import Iterator
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


@dataclass(frozen=True)
class ProposalFeatureBatchV4:
    proposal_ids: np.ndarray
    sequence: np.ndarray
    sequence_mask: np.ndarray
    scalar: np.ndarray
    event_target: np.ndarray | None = None
    iou_target: np.ndarray | None = None
    sample_weight: np.ndarray | None = None


class ProposalDatasetV4(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(self, features: ProposalFeatureBatchV4) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features.proposal_ids)

    def __getitem__(self, index: int | tuple[int, float]) -> dict[str, torch.Tensor | str]:
        if isinstance(index, tuple):
            row_index, sampling_probability = index
        else:
            row_index, sampling_probability = index, 1.0
        index = int(row_index)
        output: dict[str, torch.Tensor | str] = {
            "proposal_id": str(self.features.proposal_ids[index]),
            "sequence": torch.from_numpy(self.features.sequence[index]),
            "sequence_mask": torch.from_numpy(self.features.sequence_mask[index]),
            "scalar": torch.from_numpy(self.features.scalar[index]),
        }
        for name in ("event_target", "iou_target", "sample_weight"):
            values = getattr(self.features, name)
            if values is not None:
                output[name] = torch.tensor(float(values[index]), dtype=torch.float32)
        if self.features.sample_weight is not None:
            probability = max(float(sampling_probability), np.finfo(np.float32).tiny)
            output["sampling_probability"] = torch.tensor(probability, dtype=torch.float32)
            output["importance_weight"] = torch.tensor(
                float(self.features.sample_weight[index]) / probability,
                dtype=torch.float32,
            )
        return output


class HardNegativeBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        categories: np.ndarray,
        *,
        batch_size: int,
        ratios: dict[str, float],
        steps_per_epoch: int,
        seed: int,
        target_weights: np.ndarray | None = None,
    ) -> None:
        self.categories = np.asarray(categories, dtype=str)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        if set(ratios) != set(CATEGORY_ORDER):
            raise ValueError("Verifier ratios must define all four categories")
        weights = np.asarray([ratios[name] for name in CATEGORY_ORDER], dtype=np.float64)
        if (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Verifier ratios must be non-negative and sum to one")
        raw = weights * self.batch_size
        counts = np.floor(raw).astype(int)
        for index in np.argsort(-(raw - counts), kind="stable")[: self.batch_size - counts.sum()]:
            counts[index] += 1
        pools = {name: np.flatnonzero(self.categories == name) for name in CATEGORY_ORDER}
        if not any(len(value) for value in pools.values()):
            raise ValueError("Verifier sampler received no proposals")
        redistribution = ("near_miss", "hard_false_positive", "random_background")
        for index, name in enumerate(CATEGORY_ORDER):
            if len(pools[name]) or counts[index] == 0:
                continue
            missing = int(counts[index])
            counts[index] = 0
            available = [value for value in redistribution if len(pools[value]) and value != name]
            if not available:
                available = [value for value in CATEGORY_ORDER if len(pools[value])]
            counts[CATEGORY_ORDER.index(available[0])] += missing
        self.pools = pools
        self.counts = {name: int(counts[index]) for index, name in enumerate(CATEGORY_ORDER)}
        target = (
            np.ones(len(self.categories), dtype=np.float64)
            if target_weights is None
            else np.asarray(target_weights, dtype=np.float64)
        )
        if (
            target.shape != self.categories.shape
            or not np.isfinite(target).all()
            or np.any(target < 0)
        ):
            raise ValueError("Verifier target weights must be finite, non-negative, and aligned")
        self.within_category_probability: dict[str, np.ndarray] = {}
        for name, pool in pools.items():
            if not len(pool):
                self.within_category_probability[name] = np.empty(0, dtype=np.float64)
                continue
            selected = target[pool]
            total = selected.sum()
            self.within_category_probability[name] = (
                selected / total if total > 0 else np.full(len(pool), 1.0 / len(pool))
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        for _ in range(self.steps_per_epoch):
            parts = [
                np.column_stack(
                    (
                        selected := rng.choice(
                            self.pools[name],
                            size=count,
                            replace=True,
                            p=self.within_category_probability[name],
                        ),
                        np.asarray(
                            [
                                (count / self.batch_size)
                                * self.within_category_probability[name][
                                    np.searchsorted(self.pools[name], value)
                                ]
                                for value in selected
                            ]
                        ),
                    )
                )
                for name, count in self.counts.items()
                if count
            ]
            batch = np.concatenate(parts)
            rng.shuffle(batch)
            yield [(int(row[0]), float(row[1])) for row in batch]


class DepthwiseVerifierBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(1).to(values.dtype)
        masked = values * weights
        update = self.pointwise(self.depthwise(masked)).transpose(1, 2)
        update = self.dropout(torch.nn.functional.silu(self.norm(update))).transpose(1, 2)
        return (masked + update * weights) * weights


class EventVerifierV4(nn.Module):
    def __init__(self, sequence_dim: int, scalar_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        channels = int(config.get("hidden_channels", 64))
        hidden = int(config.get("hidden_dim", 128))
        dropout = float(config.get("dropout", 0.1))
        self.input_projection = nn.Conv1d(sequence_dim, channels, kernel_size=1, bias=False)
        self.blocks = nn.ModuleList(
            [
                DepthwiseVerifierBlock(channels, dropout),
                DepthwiseVerifierBlock(channels, dropout),
            ]
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
        mask = batch["sequence_mask"].bool()
        values = batch["sequence"] * mask.unsqueeze(-1).to(batch["sequence"].dtype)
        sequence = self.input_projection(values.transpose(1, 2))
        for block in self.blocks:
            sequence = block(sequence, mask)
        weights = mask.unsqueeze(1).to(sequence.dtype)
        mean = (sequence * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
        maximum = sequence.masked_fill(~mask.unsqueeze(1), -torch.inf).amax(dim=-1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        hidden = self.projection(torch.cat((mean, maximum, batch["scalar"]), dim=-1))
        event_logit = self.event_head(hidden).squeeze(-1)
        iou_logit = self.iou_head(hidden).squeeze(-1)
        return {
            "event_logit": event_logit,
            "iou_logit": iou_logit,
            "predicted_iou": torch.sigmoid(iou_logit),
        }


def verifier_loss_v4(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], iou_weight: float = 0.5
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weight = batch.get(
        "importance_weight",
        batch.get("sample_weight", torch.ones_like(batch["event_target"])),
    )
    event_element = nn.functional.binary_cross_entropy_with_logits(
        output["event_logit"], batch["event_target"], reduction="none"
    )
    iou_element = nn.functional.smooth_l1_loss(
        output["predicted_iou"], batch["iou_target"], reduction="none"
    )
    denominator = weight.sum().clamp_min(1.0)
    event = (event_element * weight).sum() / denominator
    quality = (iou_element * weight).sum() / denominator
    return event + float(iou_weight) * quality, {"event": event, "quality": quality}


def classify_proposals(frame: pd.DataFrame) -> np.ndarray:
    required = {"max_iou", "generator_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Proposal labels are missing columns: {sorted(missing)}")
    iou = frame["max_iou"].to_numpy(dtype=np.float64)
    score = frame["generator_score"].to_numpy(dtype=np.float64)
    hard_threshold = float(np.quantile(score[iou < 0.10], 0.75)) if np.any(iou < 0.10) else np.inf
    historical = frame.get("historical_hard_false_positive", False)
    historical = np.asarray(historical, dtype=bool)
    category = np.full(len(frame), "random_background", dtype=object)
    category[(iou >= 0.10) & (iou <= 0.25)] = "near_miss"
    category[(iou < 0.10) & ((score >= hard_threshold) | historical)] = "hard_false_positive"
    category[iou > 0.25] = "positive"
    return category.astype(str)


def normalized_proposal_weights(frame: pd.DataFrame) -> np.ndarray:
    required = {"subject_key", "proposal_id", "max_iou"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Proposal weights are missing columns: {sorted(missing)}")
    positive = frame["max_iou"].to_numpy(dtype=np.float64) > 0.25
    matched = frame.get("matched_event_id", pd.Series("", index=frame.index)).astype(str)
    family = frame.get("proposal_family_id", frame["proposal_id"]).astype(str)
    group = np.where(positive, "event:" + matched, "family:" + family)
    work = pd.DataFrame(
        {
            "subject_key": frame["subject_key"].astype(str).to_numpy(),
            "group": group,
            "row": np.arange(len(frame), dtype=np.int64),
        }
    )
    weights = np.zeros(len(frame), dtype=np.float64)
    for _, subject in work.groupby("subject_key", sort=False):
        subject_weight = np.zeros(len(subject), dtype=np.float64)
        for _, grouped in subject.groupby("group", sort=False):
            positions = subject.index.get_indexer(grouped.index)
            subject_weight[positions] = 1.0 / len(grouped)
        subject_weight /= subject_weight.sum()
        weights[subject["row"].to_numpy(dtype=np.int64)] = subject_weight
    weights *= len(work["subject_key"].unique()) / weights.sum()
    return weights.astype(np.float32)


def _bin_specs(start_ms: int, end_ms: int, config: dict[str, Any]):
    specifications = (
        (
            start_ms - int(config["left_context_seconds"]) * 1000,
            start_ms,
            int(config["left_bins"]),
            0,
        ),
        (start_ms, end_ms, int(config["event_bins"]), 1),
        (
            end_ms,
            end_ms + int(config["right_context_seconds"]) * 1000,
            int(config["right_bins"]),
            2,
        ),
    )
    total = sum(value[2] for value in specifications)
    position = 0
    for left, right, count, region in specifications:
        edges = np.linspace(left, right, count + 1)
        for index in range(count):
            yield int(edges[index]), int(edges[index + 1]), region, position / max(total - 1, 1)
            position += 1


def build_proposal_features_v4(
    proposals: pd.DataFrame,
    windows: pd.DataFrame,
    statistics_columns: list[str],
    config: dict[str, Any],
) -> ProposalFeatureBatchV4:
    base_columns = [
        "state_probability",
        "onset_probability",
        "offset_probability",
        "ppg_gate",
        "missing_fraction",
        *statistics_columns,
    ]
    required = {"subject_key", "session_id", "timestamp_ms", *base_columns}
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"Verifier windows are missing columns: {sorted(missing)}")
    groups = {
        (str(subject), str(session)): group.sort_values("timestamp_ms")
        for (subject, session), group in windows.groupby(["subject_key", "session_id"], sort=False)
    }
    sequences: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    scalars: list[np.ndarray] = []
    for proposal in proposals.itertuples(index=False):
        group = groups.get((str(proposal.subject_key), str(proposal.session_id)))
        if group is None:
            raise ValueError(f"No verifier windows for proposal {proposal.proposal_id}")
        timestamps = group["timestamp_ms"].to_numpy(dtype=np.int64)
        values = group[base_columns].to_numpy(dtype=np.float64)
        bins: list[np.ndarray] = []
        valid_bins: list[bool] = []
        for left, right, region, relative in _bin_specs(
            int(proposal.coarse_start_ms), int(proposal.coarse_end_ms), config
        ):
            selected = (timestamps > left) & (timestamps <= right)
            selected_values = values[selected]
            valid_bins.append(bool(len(selected_values)))
            if len(selected_values):
                finite = np.isfinite(selected_values)
                cleaned = np.where(finite, selected_values, np.nan)
                with np.errstate(all="ignore"):
                    mean = np.nanmean(cleaned, axis=0)
                    maximum = np.nanmax(cleaned, axis=0)
                    standard_deviation = np.nanstd(cleaned, axis=0)
                last = np.zeros(len(base_columns), dtype=np.float64)
                for column in range(len(base_columns)):
                    indices = np.flatnonzero(finite[:, column])
                    if len(indices):
                        last[column] = selected_values[indices[-1], column]
                count = finite.sum(axis=0).astype(np.float64)
            else:
                mean = maximum = standard_deviation = last = count = np.zeros(
                    len(base_columns), dtype=np.float64
                )
            region_one_hot = np.eye(3, dtype=np.float64)[region]
            metadata = np.asarray(
                [
                    *region_one_hot,
                    relative,
                    (right - left) / 1000.0,
                    float(not len(selected_values)),
                ]
            )
            bins.append(
                np.nan_to_num(
                    np.concatenate((mean, maximum, standard_deviation, last, count, metadata)),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).astype(np.float32)
            )
        duration_seconds = (int(proposal.coarse_end_ms) - int(proposal.coarse_start_ms)) / 1000.0
        in_event = (timestamps > int(proposal.coarse_start_ms)) & (
            timestamps <= int(proposal.coarse_end_ms)
        )
        state_mean = float(np.nanmean(values[in_event, 0])) if in_event.any() else 0.0
        source_mask = int(proposal.source_mask)
        scalars.append(
            np.asarray(
                [
                    np.log1p(max(duration_seconds, 0.0)),
                    float(proposal.generator_score),
                    state_mean,
                    *((source_mask & (1 << bit)) > 0 for bit in range(4)),
                    float(np.mean(valid_bins)),
                ],
                dtype=np.float32,
            )
        )
        sequences.append(np.stack(bins))
        masks.append(np.asarray(valid_bins, dtype=bool))
    event_target = (
        (proposals["max_iou"].to_numpy(dtype=np.float32) > 0.25).astype(np.float32)
        if "max_iou" in proposals
        else None
    )
    iou_target = proposals["max_iou"].to_numpy(dtype=np.float32) if "max_iou" in proposals else None
    weights = normalized_proposal_weights(proposals) if "max_iou" in proposals else None
    bin_count = int(config["left_bins"]) + int(config["event_bins"]) + int(config["right_bins"])
    feature_dim = 5 * len(base_columns) + 6
    return ProposalFeatureBatchV4(
        proposal_ids=proposals["proposal_id"].astype(str).to_numpy(),
        sequence=np.stack(sequences)
        if sequences
        else np.empty((0, bin_count, feature_dim), np.float32),
        sequence_mask=np.stack(masks) if masks else np.empty((0, bin_count), bool),
        scalar=np.stack(scalars) if scalars else np.empty((0, 8), np.float32),
        event_target=event_target,
        iou_target=iou_target,
        sample_weight=weights,
    )
