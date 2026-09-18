from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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

