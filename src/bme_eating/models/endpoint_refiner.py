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
    offsets_seconds: np.ndarray,
    target_seconds: float,
    sigma_seconds: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    offsets = np.asarray(offsets_seconds, dtype=np.float64)
    if sigma_seconds <= 0:
        raise ValueError("Boundary Gaussian sigma must be positive")
    density = np.exp(-0.5 * ((offsets - float(target_seconds)) / sigma_seconds) ** 2)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != offsets.shape:
            raise ValueError("Boundary target mask must match the offset grid")
        density = np.where(valid, density, 0.0)
    total = float(density.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Boundary target cannot be normalized")
    return (density / total).astype(np.float32)


def normalized_entropy(
    probabilities: torch.Tensor, valid_mask: torch.Tensor | None = None
) -> torch.Tensor:
    if valid_mask is None:
        valid_mask = torch.ones_like(probabilities, dtype=torch.bool)
    valid = valid_mask.bool()
    values = torch.where(valid, probabilities, torch.zeros_like(probabilities))
    values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    safe = values.clamp_min(1e-8)
    entropy = -(torch.where(valid, values * safe.log(), torch.zeros_like(values))).sum(dim=-1)
    valid_count = valid.sum(dim=-1)
    denominator = valid_count.to(values.dtype).log().clamp_min(1e-8)
    normalized = entropy / denominator
    return torch.where(valid_count >= 2, normalized, torch.ones_like(normalized))


class EndpointNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_convolution = nn.Conv1d(
            input_dim, hidden_dim, kernel_size=3, padding=1, bias=False
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
            bias=False,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_convolution = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.bool()
        weights = valid.unsqueeze(1).to(values.dtype)
        hidden = self.input_convolution(
            (values * valid.unsqueeze(-1).to(values.dtype)).transpose(1, 2)
        )
        hidden = self.dropout(
            torch.nn.functional.silu(self.input_norm(hidden.transpose(1, 2)))
        ).transpose(1, 2)
        hidden *= weights
        hidden = self.pointwise(self.depthwise(hidden)).transpose(1, 2)
        hidden = torch.nn.functional.silu(self.output_norm(hidden)).transpose(1, 2)
        hidden *= weights
        logits = self.output_convolution(hidden).squeeze(1)
        return logits.masked_fill(~valid, -torch.inf)


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
    sample_weight = batch.get(
        "sample_weight",
        torch.ones(output["start_logit"].shape[0], device=output["start_logit"].device),
    )

    def endpoint_term(name: str) -> torch.Tensor:
        logits = output[f"{name}_logit"]
        target = batch[f"{name}_target"].to(logits.dtype)
        mask = batch[f"{name}_mask"].bool()
        endpoint_weight = batch.get(f"{name}_weight", torch.ones_like(sample_weight))
        active = endpoint_weight > 0
        target_sum = target.sum(dim=-1)
        if active.any() and not torch.allclose(
            target_sum[active], torch.ones_like(target_sum[active]), atol=1e-5, rtol=1e-5
        ):
            raise RuntimeError(f"{name} boundary target must sum to one for active endpoints")
        invalid_mass = torch.where(~mask, target, torch.zeros_like(target)).sum(dim=-1)
        if torch.any(invalid_mass > 1e-7):
            raise RuntimeError(f"{name} boundary target assigns mass to invalid bins")
        safe_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        safe_logits = torch.where(
            mask.any(dim=-1, keepdim=True), safe_logits, torch.zeros_like(safe_logits)
        )
        log_probability = torch.log_softmax(safe_logits, dim=-1)
        element = -torch.where(target > 0, target * log_probability, torch.zeros_like(target)).sum(
            dim=-1
        )
        weight = sample_weight * endpoint_weight
        loss = (element * weight).sum() / weight.sum().clamp_min(1.0)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} boundary loss is not finite")
        return loss

    start_loss = endpoint_term("start")
    end_loss = endpoint_term("end")
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
    start_weight: np.ndarray | None
    end_weight: np.ndarray | None
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
    start_weights: list[float] = []
    end_weights: list[float] = []
    kept_indices: list[int] = []
    for proposal_index, proposal in enumerate(proposals.itertuples(index=False)):
        group = groups[(str(proposal.subject_key), str(proposal.session_id))]
        observation_end = min(
            int(group["timestamp_ms"].max()), int(proposal.coarse_end_ms) + 60_000
        )
        start_grid = int(proposal.coarse_start_ms) + (start_offsets * 1000).astype(np.int64)
        end_grid = int(proposal.coarse_end_ms) + (end_offsets * 1000).astype(np.int64)
        start_values, start_valid = _nearest_features(group, start_grid, columns, observation_end)
        end_values, end_valid = _nearest_features(group, end_grid, columns, observation_end)
        start_relative = (start_offsets / max(boundary_range.start_seconds, 1))[:, None]
        end_relative = (end_offsets / max(boundary_range.end_seconds, 1))[:, None]
        start_weight = 1.0
        end_weight = 1.0
        start_distribution: np.ndarray | None = None
        end_distribution: np.ndarray | None = None
        if hasattr(proposal, "truth_start_ms"):
            start_target = (int(proposal.truth_start_ms) - int(proposal.coarse_start_ms)) / 1000.0
            end_target = (int(proposal.truth_end_ms) - int(proposal.coarse_end_ms)) / 1000.0
            start_weight = float(
                start_valid.any()
                and np.min(np.abs(start_offsets[start_valid] - start_target)) <= bin_seconds
            )
            end_weight = float(
                end_valid.any()
                and np.min(np.abs(end_offsets[end_valid] - end_target)) <= bin_seconds
            )
            if not start_weight and not end_weight:
                continue
            start_distribution = (
                truncated_gaussian_target(start_offsets, start_target, sigma, start_valid)
                if start_weight
                else np.zeros_like(start_offsets, dtype=np.float32)
            )
            end_distribution = (
                truncated_gaussian_target(end_offsets, end_target, sigma, end_valid)
                if end_weight
                else np.zeros_like(end_offsets, dtype=np.float32)
            )
        start_sequences.append(np.concatenate((start_values, start_relative), axis=1))
        end_sequences.append(np.concatenate((end_values, end_relative), axis=1))
        start_masks.append(start_valid)
        end_masks.append(end_valid)
        kept_indices.append(proposal_index)
        if start_distribution is not None and end_distribution is not None:
            start_targets.append(start_distribution)
            end_targets.append(end_distribution)
            start_weights.append(start_weight)
            end_weights.append(end_weight)
    source_ids = (
        proposals.get("boundary_sample_id", proposals["proposal_id"]).astype(str).to_numpy()
    )
    sample_ids = source_ids[np.asarray(kept_indices, dtype=np.int64)]
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
            start_weight=None,
            end_weight=None,
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
            proposals.iloc[kept_indices]["sample_weight"].to_numpy(dtype=np.float32)
            if "sample_weight" in proposals
            else None
        ),
        start_weight=np.asarray(start_weights, dtype=np.float32) if start_targets else None,
        end_weight=np.asarray(end_weights, dtype=np.float32) if end_targets else None,
        start_offsets_seconds=start_offsets,
        end_offsets_seconds=end_offsets,
    )


def local_soft_argmax(
    logits: torch.Tensor,
    offsets: torch.Tensor,
    radius_bins: int = 2,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = torch.isfinite(logits) if valid_mask is None else valid_mask.bool()
    safe_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    has_valid = valid.any(dim=-1, keepdim=True)
    safe_logits = torch.where(has_valid, safe_logits, torch.zeros_like(safe_logits))
    probability = torch.softmax(safe_logits, dim=-1) * valid.to(logits.dtype)
    probability /= probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    mode = probability.argmax(dim=-1)
    index = torch.arange(logits.shape[-1], device=logits.device).unsqueeze(0)
    local = (index - mode.unsqueeze(-1)).abs() <= int(radius_bins)
    local_probability = probability * local
    local_probability /= local_probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    prediction = (local_probability * offsets.unsqueeze(0)).sum(dim=-1)
    valid_count = valid.sum(dim=-1)
    prediction = torch.where(valid_count >= 2, prediction, torch.zeros_like(prediction))
    return prediction, normalized_entropy(probability, valid)


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
    output["start_fallback"] = ~start_use
    output["end_fallback"] = ~end_use
    output["boundary_fallback"] = output["start_fallback"] | output["end_fallback"]
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
                output.at[index, "start_fallback"] = True
                output.at[index, "end_fallback"] = True
                output.at[index, "boundary_fallback"] = True
            output.at[index, "refined_start_ms"] = start
            output.at[index, "refined_end_ms"] = end
        for previous_index, index in pairwise(ordered):
            previous_end = int(output.at[previous_index, "refined_end_ms"])
            start = int(output.at[index, "refined_start_ms"])
            if previous_end + safety_gap_ms <= start:
                continue
            output.at[previous_index, "refined_end_ms"] = int(
                output.at[previous_index, "coarse_end_ms"]
            )
            output.at[index, "refined_start_ms"] = int(output.at[index, "coarse_start_ms"])
            output.at[previous_index, "boundary_fallback"] = True
            output.at[index, "boundary_fallback"] = True
            output.at[previous_index, "end_fallback"] = True
            output.at[index, "start_fallback"] = True
    if len(output) != len(accepted) or set(output["proposal_id"]) != set(accepted["proposal_id"]):
        raise RuntimeError("Boundary refinement changed proposal identity or count")
    if (output["refined_start_ms"] >= output["refined_end_ms"]).any():
        raise RuntimeError("Boundary refinement produced a non-positive event")
    for indices in output.groupby(["subject_key", "session_id"], sort=False).groups.values():
        ordered = output.loc[list(indices)].sort_values("refined_start_ms")
        if len(ordered) > 1:
            gaps = (
                ordered["refined_start_ms"].to_numpy()[1:]
                - ordered["refined_end_ms"].to_numpy()[:-1]
            )
            if np.any(gaps < safety_gap_ms):
                raise RuntimeError("Boundary refinement violated the neighboring-event gap")
    return output
