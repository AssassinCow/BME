from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


def offset_bins(range_seconds: int, bin_seconds: int) -> np.ndarray:
    if range_seconds <= 0 or bin_seconds <= 0 or range_seconds % bin_seconds:
        raise ValueError("Boundary range must be a positive multiple of bin size")
    return np.arange(-range_seconds, range_seconds + bin_seconds, bin_seconds, dtype=np.float32)


def soft_offset_target(offset_seconds: float, bins: np.ndarray) -> np.ndarray:
    target = np.zeros(len(bins), dtype=np.float32)
    clipped = float(np.clip(offset_seconds, bins[0], bins[-1]))
    right = int(np.searchsorted(bins, clipped, side="left"))
    if right == 0:
        target[0] = 1.0
        return target
    if right >= len(bins):
        target[-1] = 1.0
        return target
    left = right - 1
    width = float(bins[right] - bins[left])
    right_weight = (clipped - float(bins[left])) / width
    target[left] = 1.0 - right_weight
    target[right] = right_weight
    return target


def normalized_entropy(probability: torch.Tensor) -> torch.Tensor:
    entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=-1)
    return entropy / np.log(probability.shape[-1])


class BoundaryRefiner(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        scalar_dim: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        hidden = int(config.get("hidden_dim", 128))
        dropout = float(config.get("dropout", 0.1))
        self.start_bins = offset_bins(
            int(config["start_range_seconds"]), int(config["coarse_bin_seconds"])
        )
        self.end_bins = offset_bins(
            int(config["end_range_seconds"]), int(config["coarse_bin_seconds"])
        )
        self.fine_range_seconds = float(config["fine_range_seconds"])
        self.sequence_encoder = nn.Sequential(
            nn.Conv1d(sequence_dim, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.shared = nn.Sequential(
            nn.Linear(128 + scalar_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.start_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.SiLU(), nn.Linear(64, len(self.start_bins))
        )
        self.end_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.SiLU(), nn.Linear(64, len(self.end_bins))
        )
        self.start_fine = nn.Sequential(nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 1))
        self.end_fine = nn.Sequential(nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sequence = self.sequence_encoder(batch["sequence"].transpose(1, 2))
        pooled = torch.cat((sequence.mean(dim=-1), sequence.amax(dim=-1)), dim=-1)
        hidden = self.shared(torch.cat((pooled, batch["scalar"]), dim=-1))
        return {
            "start_distribution_logit": self.start_head(hidden),
            "end_distribution_logit": self.end_head(hidden),
            "start_fine_seconds": torch.tanh(self.start_fine(hidden).squeeze(-1))
            * self.fine_range_seconds,
            "end_fine_seconds": torch.tanh(self.end_fine(hidden).squeeze(-1))
            * self.fine_range_seconds,
        }


def boundary_loss(
    output: dict[str, torch.Tensor],
    start_distribution_target: torch.Tensor,
    end_distribution_target: torch.Tensor,
    start_fine_target: torch.Tensor,
    end_fine_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    start_log_probability = nn.functional.log_softmax(
        output["start_distribution_logit"], dim=-1
    )
    end_log_probability = nn.functional.log_softmax(
        output["end_distribution_logit"], dim=-1
    )
    start_coarse = -(start_distribution_target * start_log_probability).sum(dim=-1).mean()
    end_coarse = -(end_distribution_target * end_log_probability).sum(dim=-1).mean()
    start_fine = nn.functional.smooth_l1_loss(
        output["start_fine_seconds"], start_fine_target
    )
    end_fine = nn.functional.smooth_l1_loss(output["end_fine_seconds"], end_fine_target)
    coarse = 0.5 * (start_coarse + end_coarse)
    fine = 0.5 * (start_fine + end_fine)
    return coarse + 0.1 * fine, {"coarse": coarse, "fine": fine}


def build_boundary_targets(
    proposals: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    event_lookup = events.set_index("event_id")
    start_bins = offset_bins(
        int(config["start_range_seconds"]), int(config["coarse_bin_seconds"])
    )
    end_bins = offset_bins(
        int(config["end_range_seconds"]), int(config["coarse_bin_seconds"])
    )
    start_distribution: list[np.ndarray] = []
    end_distribution: list[np.ndarray] = []
    start_fine: list[float] = []
    end_fine: list[float] = []
    coarse_bin = float(config["coarse_bin_seconds"])
    fine_range = float(config["fine_range_seconds"])
    for proposal in proposals.itertuples(index=False):
        event_id = str(proposal.matched_event_id)
        if event_id not in event_lookup.index:
            raise ValueError(f"Matched event is unavailable for boundary training: {event_id}")
        event = event_lookup.loc[event_id]
        start_offset = (int(event.start_ms) - int(proposal.coarse_start_ms)) / 1000
        end_offset = (int(event.end_ms) - int(proposal.coarse_end_ms)) / 1000
        start_distribution.append(soft_offset_target(start_offset, start_bins))
        end_distribution.append(soft_offset_target(end_offset, end_bins))
        start_center = float(start_bins[np.argmin(np.abs(start_bins - start_offset))])
        end_center = float(end_bins[np.argmin(np.abs(end_bins - end_offset))])
        start_fine.append(float(np.clip(start_offset - start_center, -fine_range, fine_range)))
        end_fine.append(float(np.clip(end_offset - end_center, -fine_range, fine_range)))
        if coarse_bin <= 0:
            raise ValueError("Boundary coarse bin must be positive")
    return {
        "start_distribution": np.stack(start_distribution).astype(np.float32),
        "end_distribution": np.stack(end_distribution).astype(np.float32),
        "start_fine": np.asarray(start_fine, dtype=np.float32),
        "end_fine": np.asarray(end_fine, dtype=np.float32),
    }


def decode_boundaries(
    proposals: pd.DataFrame,
    output: dict[str, torch.Tensor],
    config: dict[str, Any],
    entropy_threshold: float,
    observation_end_by_session: dict[tuple[str, str], int],
) -> pd.DataFrame:
    start_bins = torch.as_tensor(
        offset_bins(int(config["start_range_seconds"]), int(config["coarse_bin_seconds"])),
        device=output["start_distribution_logit"].device,
    )
    end_bins = torch.as_tensor(
        offset_bins(int(config["end_range_seconds"]), int(config["coarse_bin_seconds"])),
        device=output["end_distribution_logit"].device,
    )
    start_probability = torch.softmax(output["start_distribution_logit"], dim=-1)
    end_logits = output["end_distribution_logit"].clone()
    available_values: list[float] = []
    for index, proposal in enumerate(proposals.itertuples(index=False)):
        observed_end = observation_end_by_session[
            (str(proposal.subject_key), str(proposal.session_id))
        ]
        available_seconds = (observed_end - int(proposal.coarse_end_ms)) / 1000
        available_values.append(float(available_seconds))
        allowed = end_bins <= available_seconds
        if not bool(allowed.any()):
            raise RuntimeError(
                "Coarse proposal end exceeds the observed timeline beyond the refinement range"
            )
        else:
            end_logits[index, ~allowed] = -torch.inf
    end_probability = torch.softmax(end_logits, dim=-1)
    start_entropy = normalized_entropy(start_probability)
    end_entropy = normalized_entropy(end_probability)
    start_offset = (start_probability * start_bins).sum(dim=-1) + output[
        "start_fine_seconds"
    ]
    end_offset = (end_probability * end_bins).sum(dim=-1) + output["end_fine_seconds"]
    start_offset = torch.where(
        start_entropy <= entropy_threshold, start_offset, torch.zeros_like(start_offset)
    )
    end_offset = torch.where(
        end_entropy <= entropy_threshold, end_offset, torch.zeros_like(end_offset)
    )
    available = torch.as_tensor(
        available_values, dtype=end_offset.dtype, device=end_offset.device
    )
    end_offset = torch.minimum(end_offset, available)
    result = proposals.copy()
    result["refined_start_ms"] = (
        result["coarse_start_ms"].to_numpy(dtype=np.int64)
        + np.rint(start_offset.detach().cpu().numpy() * 1000).astype(np.int64)
    )
    result["refined_end_ms"] = (
        result["coarse_end_ms"].to_numpy(dtype=np.int64)
        + np.rint(end_offset.detach().cpu().numpy() * 1000).astype(np.int64)
    )
    result["start_entropy"] = start_entropy.detach().cpu().numpy().astype(np.float32)
    result["end_entropy"] = end_entropy.detach().cpu().numpy().astype(np.float32)
    result["boundary_fallback"] = np.logical_or(
        result["start_entropy"] > entropy_threshold,
        result["end_entropy"] > entropy_threshold,
    )
    invalid = result["refined_start_ms"] >= result["refined_end_ms"]
    result.loc[invalid, "refined_start_ms"] = result.loc[invalid, "coarse_start_ms"]
    result.loc[invalid, "refined_end_ms"] = result.loc[invalid, "coarse_end_ms"]
    result.loc[invalid, "boundary_fallback"] = True
    return enforce_boundary_order(result, int(config["safety_gap_seconds"]) * 1000)


def enforce_boundary_order(events: pd.DataFrame, safety_gap_ms: int) -> pd.DataFrame:
    if safety_gap_ms < 0:
        raise ValueError("Boundary safety gap cannot be negative")
    output = events.sort_values(
        ["subject_key", "session_id", "refined_start_ms", "refined_end_ms"]
    ).copy()
    for _, group in output.groupby(["subject_key", "session_id"], sort=False):
        indices = list(group.index)
        for left_index, right_index in pairwise(indices):
            left_end = int(output.at[left_index, "refined_end_ms"])
            right_start = int(output.at[right_index, "refined_start_ms"])
            if right_start >= left_end + safety_gap_ms:
                continue
            left_start = int(output.at[left_index, "refined_start_ms"])
            right_end = int(output.at[right_index, "refined_end_ms"])
            if right_end - left_start <= safety_gap_ms + 2:
                raise RuntimeError("Accepted events cannot satisfy the boundary safety gap")
            new_left_end = (left_end + right_start - safety_gap_ms) // 2
            new_left_end = int(
                np.clip(new_left_end, left_start + 1, right_end - safety_gap_ms - 1)
            )
            new_right_start = new_left_end + safety_gap_ms
            output.at[left_index, "refined_end_ms"] = new_left_end
            output.at[right_index, "refined_start_ms"] = new_right_start
    if (output["refined_start_ms"] >= output["refined_end_ms"]).any():
        raise RuntimeError("Boundary ordering could not preserve positive event durations")
    for _, group in output.groupby(["subject_key", "session_id"], sort=False):
        ordered = group.sort_values("refined_start_ms")
        previous_end = ordered["refined_end_ms"].to_numpy(dtype=np.int64)[:-1]
        next_start = ordered["refined_start_ms"].to_numpy(dtype=np.int64)[1:]
        if np.any(next_start - previous_end < safety_gap_ms):
            raise RuntimeError("Boundary ordering did not preserve the configured safety gap")
    return output.sort_index()
