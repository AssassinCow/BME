from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class StateOutput:
    state_logit: torch.Tensor
    start_logit: torch.Tensor
    end_logit: torch.Tensor
    state_embedding: torch.Tensor
    ppg_gate: torch.Tensor
    missing_fraction: torch.Tensor


@dataclass(frozen=True)
class Event:
    subject_key: str
    start_ms: int
    end_ms: int
    score: float = 1.0
    event_id: str | None = None
    hand_relation: str = "unknown"

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass(frozen=True)
class SensorSeries:
    timestamp_ms: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class SegmentReference:
    segment_id: str
    subject_key: str
    path: Path
    start_ms: int
    end_ms: int
    wear_hand: str


@dataclass
class ResampledSegment:
    motion_timestamp_ms: np.ndarray
    motion_values: np.ndarray
    motion_mask: np.ndarray
    ppg_timestamp_ms: np.ndarray
    ppg_values: np.ndarray
    ppg_mask: np.ndarray


@dataclass(frozen=True)
class ProposalRecord:
    proposal_id: str
    subject_key: str
    session_id: str
    coarse_start_ms: int
    coarse_end_ms: int
    source_mask: int
    generator_score: float
    rank_within_session: int
    split_role: str


@dataclass(frozen=True)
class ProposalScore:
    proposal_id: str
    event_logit: float
    predicted_iou: float
    calibrated_event_probability: float
    calibrated_iou: float
    final_score: float
    accepted: bool
    refined_start_ms: int
    refined_end_ms: int
    start_entropy: float
    end_entropy: float
    boundary_fallback: bool

