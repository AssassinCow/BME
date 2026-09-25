from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass(frozen=True)
class BoundaryRange:
    start_seconds: int
    end_seconds: int
    clipped_fraction: float


def select_boundary_range(
    start_residual_seconds: np.ndarray,
    end_residual_seconds: np.ndarray,
    *,
    quantile: float = 0.99,
    rounding_seconds: int = 15,
    minimum_seconds: int = 60,
    maximum_seconds: int = 300,
) -> BoundaryRange:
    start = np.abs(np.asarray(start_residual_seconds, dtype=np.float64))
    end = np.abs(np.asarray(end_residual_seconds, dtype=np.float64))
    start = start[np.isfinite(start)]
    end = end[np.isfinite(end)]
    if not len(start) or not len(end):
        raise ValueError("Boundary range selection requires finite start and end residuals")

    def choose(values: np.ndarray) -> int:
        raw = int(np.ceil(np.quantile(values, quantile) / rounding_seconds) * rounding_seconds)
        return int(np.clip(raw, minimum_seconds, maximum_seconds))

    start_range = choose(start)
    end_range = choose(end)
    clipped = (np.count_nonzero(start > start_range) + np.count_nonzero(end > end_range)) / (
        len(start) + len(end)
    )
    return BoundaryRange(start_range, end_range, float(clipped))


def truncated_gaussian_target(
    offsets_seconds: np.ndarray, target_seconds: float, sigma_seconds: float
) -> np.ndarray:
    offsets = np.asarray(offsets_seconds, dtype=np.float64)
    if sigma_seconds <= 0:
        raise ValueError("Boundary Gaussian sigma must be positive")
    density = np.exp(-0.5 * ((offsets - float(target_seconds)) / sigma_seconds) ** 2)
    total = float(density.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Boundary target cannot be normalized")
    return (density / total).astype(np.float32)


def normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    values = probabilities.clamp_min(1e-8)
    entropy = -(values * values.log()).sum(dim=-1)
    return entropy / np.log(values.shape[-1])


class EndpointNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if hidden_dim % 8 == 0 else 1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8 if hidden_dim % 8 == 0 else 1, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.network(values.transpose(1, 2)).squeeze(1)
        return logits.masked_fill(~mask.bool(), -torch.inf)


class EndpointRefiner(nn.Module):
    def __init__(self, input_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(config.get("hidden_dim", 64))
        dropout = float(config.get("dropout", 0.1))
        self.start_network = EndpointNetwork(input_dim, hidden, dropout)
        self.end_network = EndpointNetwork(input_dim, hidden, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "start_logit": self.start_network(batch["start_sequence"], batch["start_mask"]),
            "end_logit": self.end_network(batch["end_sequence"], batch["end_mask"]),
        }


def endpoint_loss(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weight = batch.get("sample_weight", torch.ones(output["start_logit"].shape[0], device=output["start_logit"].device))
    start = -(batch["start_target"] * torch.log_softmax(output["start_logit"], dim=-1)).sum(dim=-1)
    end = -(batch["end_target"] * torch.log_softmax(output["end_logit"], dim=-1)).sum(dim=-1)
    denominator = weight.sum().clamp_min(1.0)
    start_loss = (start * weight).sum() / denominator
    end_loss = (end * weight).sum() / denominator
    return 0.5 * (start_loss + end_loss), {"start": start_loss, "end": end_loss}


def augment_boundary_training_proposals(
    positive_proposals: pd.DataFrame,
    *,
    maximum_jitters_per_event: int = 4,
    jitter_seconds: int = 60,
) -> pd.DataFrame:
    required = {
        "proposal_id",
        "subject_key",
        "session_id",
        "coarse_start_ms",
        "coarse_end_ms",
        "matched_event_id",
        "truth_start_ms",
        "truth_end_ms",
    }
    missing = required - set(positive_proposals.columns)
    if missing:
        raise ValueError(f"Boundary proposals are missing columns: {sorted(missing)}")
    ordering = ["matched_event_id"]
    ascending = [True]
    for column in ("final_score", "generator_score", "max_iou"):
        if column in positive_proposals:
            ordering.append(column)
            ascending.append(False)
    ordering.append("proposal_id")
    ascending.append(True)
    representatives = (
        positive_proposals.sort_values(ordering, ascending=ascending, kind="stable")
        .drop_duplicates("matched_event_id", keep="first")
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    shifts = ((-jitter_seconds, 0), (jitter_seconds, 0), (0, -jitter_seconds), (0, jitter_seconds))[
        :maximum_jitters_per_event
    ]
    for proposal in representatives.itertuples(index=False):
        base = proposal._asdict()
        variants = [(0, 0), *shifts]
        valid: list[dict[str, Any]] = []
        for index, (start_shift, end_shift) in enumerate(variants):
            row = dict(base)
            row["coarse_start_ms"] = int(proposal.coarse_start_ms) + start_shift * 1000
            row["coarse_end_ms"] = int(proposal.coarse_end_ms) + end_shift * 1000
            if row["coarse_start_ms"] >= row["coarse_end_ms"]:
                continue
            row["boundary_sample_id"] = f"{proposal.proposal_id}:j{index}"
            valid.append(row)
        weight = 1.0 / len(valid)
        for row in valid:
            row["sample_weight"] = weight
            rows.append(row)
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class EndpointFeatureBatch:
    sample_ids: np.ndarray
    start_sequence: np.ndarray
    end_sequence: np.ndarray
    start_mask: np.ndarray
    end_mask: np.ndarray
    start_target: np.ndarray | None
    end_target: np.ndarray | None
    sample_weight: np.ndarray | None
    start_offsets_seconds: np.ndarray
    end_offsets_seconds: np.ndarray


def _nearest_features(
    frame: pd.DataFrame,
    target_ms: np.ndarray,
    columns: list[str],
    maximum_observed_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    timestamps = frame["timestamp_ms"].to_numpy(dtype=np.int64)
    values = frame[columns].to_numpy(dtype=np.float64)
    if not len(timestamps):
        return np.zeros((len(target_ms), len(columns)), np.float32), np.zeros(len(target_ms), bool)
    right = np.searchsorted(timestamps, target_ms, side="left")
    right = np.clip(right, 0, max(len(timestamps) - 1, 0))
    left = np.clip(right - 1, 0, max(len(timestamps) - 1, 0))
    choose_left = np.abs(timestamps[left] - target_ms) <= np.abs(timestamps[right] - target_ms)
    indices = np.where(choose_left, left, right)
    valid = (np.abs(timestamps[indices] - target_ms) <= 1500) & (target_ms <= maximum_observed_ms)
    output = np.zeros((len(target_ms), len(columns)), dtype=np.float64)
    output[valid] = values[indices[valid]]
    finite = np.isfinite(output).all(axis=1)
    valid &= finite
    output[~valid] = 0.0
    return output.astype(np.float32), valid


def build_endpoint_features(
    proposals: pd.DataFrame,
    windows: pd.DataFrame,
    statistics_columns: list[str],
    boundary_range: BoundaryRange,
    config: dict[str, Any],
) -> EndpointFeatureBatch:
    columns = [
        "state_probability",
        "state_probability_derivative",
        "onset_probability",
        "offset_probability",
        "ppg_gate",
        "missing_fraction",
        *statistics_columns,
    ]
    missing = {"subject_key", "session_id", "timestamp_ms", *columns} - set(windows.columns)
    if missing:
        raise ValueError(f"Boundary windows are missing columns: {sorted(missing)}")
    groups = {
        (str(subject), str(session)): group.sort_values("timestamp_ms")
        for (subject, session), group in windows.groupby(["subject_key", "session_id"], sort=False)
    }
    bin_seconds = int(config.get("bin_seconds", 3))
    sigma = float(config.get("gaussian_sigma_seconds", 9))
    start_offsets = np.arange(
        -boundary_range.start_seconds,
        boundary_range.start_seconds + bin_seconds,
        bin_seconds,
        dtype=np.float32,
    )
    end_offsets = np.arange(
        -boundary_range.end_seconds,
        boundary_range.end_seconds + bin_seconds,
        bin_seconds,
        dtype=np.float32,
    )
    start_sequences: list[np.ndarray] = []
    end_sequences: list[np.ndarray] = []
    start_masks: list[np.ndarray] = []
    end_masks: list[np.ndarray] = []
    start_targets: list[np.ndarray] = []
    end_targets: list[np.ndarray] = []
    for proposal in proposals.itertuples(index=False):
        group = groups[(str(proposal.subject_key), str(proposal.session_id))]
        observation_end = min(
            int(group["timestamp_ms"].max()), int(proposal.coarse_end_ms) + 60_000
        )
        start_grid = int(proposal.coarse_start_ms) + (start_offsets * 1000).astype(np.int64)
        end_grid = int(proposal.coarse_end_ms) + (end_offsets * 1000).astype(np.int64)
        start_values, start_valid = _nearest_features(group, start_grid, columns, observation_end)
        end_values, end_valid = _nearest_features(group, end_grid, columns, observation_end)
        if not start_valid.any():
            start_valid[int(np.argmin(np.abs(start_offsets)))] = True
        if not end_valid.any():
            end_valid[int(np.argmin(np.abs(end_offsets)))] = True
        start_relative = (start_offsets / max(boundary_range.start_seconds, 1))[:, None]
        end_relative = (end_offsets / max(boundary_range.end_seconds, 1))[:, None]
        start_sequences.append(np.concatenate((start_values, start_relative), axis=1))
        end_sequences.append(np.concatenate((end_values, end_relative), axis=1))
        start_masks.append(start_valid)
        end_masks.append(end_valid)
        if hasattr(proposal, "truth_start_ms"):
            start_target = (int(proposal.truth_start_ms) - int(proposal.coarse_start_ms)) / 1000.0
            end_target = (int(proposal.truth_end_ms) - int(proposal.coarse_end_ms)) / 1000.0
            start_targets.append(truncated_gaussian_target(start_offsets, start_target, sigma))
            end_targets.append(truncated_gaussian_target(end_offsets, end_target, sigma))
    sample_ids = proposals.get("boundary_sample_id", proposals["proposal_id"]).astype(str).to_numpy()
    feature_dim = len(columns) + 1
    if not start_sequences:
        return EndpointFeatureBatch(
            sample_ids=sample_ids,
            start_sequence=np.empty((0, len(start_offsets), feature_dim), dtype=np.float32),
            end_sequence=np.empty((0, len(end_offsets), feature_dim), dtype=np.float32),
            start_mask=np.empty((0, len(start_offsets)), dtype=bool),
            end_mask=np.empty((0, len(end_offsets)), dtype=bool),
            start_target=None,
            end_target=None,
            sample_weight=None,
            start_offsets_seconds=start_offsets,
            end_offsets_seconds=end_offsets,
        )
    return EndpointFeatureBatch(
        sample_ids=sample_ids,
        start_sequence=np.stack(start_sequences).astype(np.float32),
        end_sequence=np.stack(end_sequences).astype(np.float32),
        start_mask=np.stack(start_masks),
        end_mask=np.stack(end_masks),
        start_target=np.stack(start_targets) if start_targets else None,
        end_target=np.stack(end_targets) if end_targets else None,
        sample_weight=(
            proposals["sample_weight"].to_numpy(dtype=np.float32)
            if "sample_weight" in proposals
            else None
        ),
        start_offsets_seconds=start_offsets,
        end_offsets_seconds=end_offsets,
    )


def local_soft_argmax(
    logits: torch.Tensor, offsets: torch.Tensor, radius_bins: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    probability = torch.softmax(logits, dim=-1)
    mode = probability.argmax(dim=-1)
    index = torch.arange(logits.shape[-1], device=logits.device).unsqueeze(0)
    local = (index - mode.unsqueeze(-1)).abs() <= int(radius_bins)
    local_probability = probability * local
    local_probability /= local_probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    prediction = (local_probability * offsets.unsqueeze(0)).sum(dim=-1)
    return prediction, normalized_entropy(probability)


def apply_boundary_refinement(
    accepted: pd.DataFrame,
    start_offset_seconds: np.ndarray,
    end_offset_seconds: np.ndarray,
    start_entropy: np.ndarray,
    end_entropy: np.ndarray,
    *,
    entropy_threshold: float,
    safety_gap_seconds: int,
) -> pd.DataFrame:
    if not (
        len(accepted)
        == len(start_offset_seconds)
        == len(end_offset_seconds)
        == len(start_entropy)
        == len(end_entropy)
    ):
        raise ValueError("Boundary predictions do not align with accepted proposals")
    output = accepted.copy().reset_index(drop=True)
    start_use = np.asarray(start_entropy) <= entropy_threshold
    end_use = np.asarray(end_entropy) <= entropy_threshold
    output["refined_start_ms"] = output["coarse_start_ms"].to_numpy(dtype=np.int64) + np.where(
        start_use, np.rint(np.asarray(start_offset_seconds) * 1000), 0
    ).astype(np.int64)
    output["refined_end_ms"] = output["coarse_end_ms"].to_numpy(dtype=np.int64) + np.where(
        end_use, np.rint(np.asarray(end_offset_seconds) * 1000), 0
    ).astype(np.int64)
    output["start_entropy"] = np.asarray(start_entropy, dtype=float)
    output["end_entropy"] = np.asarray(end_entropy, dtype=float)
    output["boundary_fallback"] = ~(start_use & end_use)
    safety_gap_ms = int(safety_gap_seconds) * 1000
    for indices in output.groupby(["subject_key", "session_id"], sort=False).groups.values():
        ordered = sorted(
            indices,
            key=lambda index: (
                int(output.at[index, "coarse_start_ms"]),
                int(output.at[index, "coarse_end_ms"]),
                str(output.at[index, "proposal_id"]),
            ),
        )
        for index in ordered:
            coarse_start = int(output.at[index, "coarse_start_ms"])
            coarse_end = int(output.at[index, "coarse_end_ms"])
            start = int(output.at[index, "refined_start_ms"])
            end = int(output.at[index, "refined_end_ms"])
            if start >= end:
                start, end = coarse_start, coarse_end
                output.at[index, "boundary_fallback"] = True
            output.at[index, "refined_start_ms"] = start
            output.at[index, "refined_end_ms"] = end
        for previous_index, index in pairwise(ordered):
            previous_start = int(output.at[previous_index, "refined_start_ms"])
            previous_end = int(output.at[previous_index, "refined_end_ms"])
            start = int(output.at[index, "refined_start_ms"])
            end = int(output.at[index, "refined_end_ms"])
            if previous_end + safety_gap_ms <= start:
                continue
            earliest_split = previous_start + 1
            latest_split = end - safety_gap_ms - 1
            if earliest_split > latest_split:
                raise RuntimeError(
                    "Accepted neighboring events cannot satisfy the boundary safety gap"
                )
            split = int(np.clip((previous_end + start - safety_gap_ms) // 2, earliest_split, latest_split))
            output.at[previous_index, "refined_end_ms"] = split
            output.at[index, "refined_start_ms"] = split + safety_gap_ms
            output.at[previous_index, "boundary_fallback"] = True
            output.at[index, "boundary_fallback"] = True
    if len(output) != len(accepted) or set(output["proposal_id"]) != set(accepted["proposal_id"]):
        raise RuntimeError("Boundary refinement changed proposal identity or count")
    if (output["refined_start_ms"] >= output["refined_end_ms"]).any():
        raise RuntimeError("Boundary refinement produced a non-positive event")
    for indices in output.groupby(["subject_key", "session_id"], sort=False).groups.values():
        ordered = output.loc[list(indices)].sort_values("refined_start_ms")
        if len(ordered) > 1:
            gaps = ordered["refined_start_ms"].to_numpy()[1:] - ordered[
                "refined_end_ms"
            ].to_numpy()[:-1]
            if np.any(gaps < safety_gap_ms):
                raise RuntimeError("Boundary refinement violated the neighboring-event gap")
    return output
