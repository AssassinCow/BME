from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TruncatedLogNormalDurationPrior:
    log_mean: float
    log_standard_deviation: float
    minimum_seconds: float
    maximum_seconds: float

    @classmethod
    def fit(
        cls,
        durations_seconds: np.ndarray,
        *,
        lower_quantile: float = 0.005,
        upper_quantile: float = 0.995,
        minimum_floor_seconds: float = 15.0,
        maximum_ceiling_seconds: float = 14_400.0,
    ) -> TruncatedLogNormalDurationPrior:
        durations = np.asarray(durations_seconds, dtype=np.float64)
        durations = durations[np.isfinite(durations) & (durations > 0)]
        if not len(durations):
            raise ValueError("Duration prior requires positive training durations")
        minimum = max(float(minimum_floor_seconds), float(np.quantile(durations, lower_quantile)))
        maximum = min(float(maximum_ceiling_seconds), float(np.quantile(durations, upper_quantile)))
        maximum = max(maximum, minimum)
        selected = durations[(durations >= minimum) & (durations <= maximum)]
        if not len(selected):
            selected = durations
        logged = np.log(selected)
        return cls(
            log_mean=float(logged.mean()),
            log_standard_deviation=max(float(logged.std()), 0.05),
            minimum_seconds=minimum,
            maximum_seconds=maximum,
        )

    def log_probability(self, duration_seconds: np.ndarray) -> np.ndarray:
        duration = np.asarray(duration_seconds, dtype=np.float64)
        safe = np.maximum(duration, 1e-6)
        z = (np.log(safe) - self.log_mean) / self.log_standard_deviation
        value = -0.5 * z**2 - np.log(safe * self.log_standard_deviation)
        valid = (duration >= self.minimum_seconds) & (duration <= self.maximum_seconds)
        return np.where(valid, value, -np.inf)

    def to_json(self) -> dict[str, float]:
        return {
            "log_mean": self.log_mean,
            "log_standard_deviation": self.log_standard_deviation,
            "minimum_seconds": self.minimum_seconds,
            "maximum_seconds": self.maximum_seconds,
        }

    @classmethod
    def from_json(cls, payload: dict[str, float]) -> TruncatedLogNormalDurationPrior:
        return cls(**{key: float(value) for key, value in payload.items()})


class FixedLagSemiMarkovDecoder:
    def __init__(
        self,
        prior: TruncatedLogNormalDurationPrior,
        *,
        grid_seconds: int = 15,
        fixed_lag_seconds: int = 60,
        duration_weight: float = 1.0,
    ) -> None:
        self.prior = prior
        self.grid_seconds = int(grid_seconds)
        self.fixed_lag_seconds = int(fixed_lag_seconds)
        self.duration_weight = float(duration_weight)
        if self.grid_seconds <= 0 or self.fixed_lag_seconds < 0:
            raise ValueError("Decoder grid and lag must be non-negative")
        if self.fixed_lag_seconds > 60:
            raise ValueError("Decoder future latency exceeds the 60-second limit")

    def _state_at(
        self,
        endpoint: int,
        target: int,
        previous: np.ndarray,
        labels: np.ndarray,
    ) -> bool:
        cursor = int(endpoint)
        while cursor > 0:
            start = int(previous[cursor])
            if start <= target < cursor:
                return bool(labels[cursor])
            cursor = start
        return False

    def decode_states(self, probabilities: np.ndarray) -> np.ndarray:
        probability = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
        steps = len(probability)
        if not steps:
            return np.zeros(0, dtype=bool)
        minimum_steps = max(1, int(np.ceil(self.prior.minimum_seconds / self.grid_seconds)))
        maximum_steps = max(
            minimum_steps, int(np.floor(self.prior.maximum_seconds / self.grid_seconds))
        )
        log_eating = np.log(probability)
        eating_prefix = np.concatenate(([0.0], np.cumsum(log_eating)))
        dp = np.full(steps + 1, -np.inf, dtype=np.float64)
        previous = np.zeros(steps + 1, dtype=np.int64)
        labels = np.zeros(steps + 1, dtype=bool)
        dp[0] = 0.0
        finalized = np.zeros(steps, dtype=bool)
        lag_steps = int(np.ceil(self.fixed_lag_seconds / self.grid_seconds))
        for endpoint in range(1, steps + 1):
            best_score = dp[endpoint - 1] + np.log1p(-probability[endpoint - 1])
            best_previous = endpoint - 1
            best_label = False
            largest = min(maximum_steps, endpoint)
            if largest >= minimum_steps:
                durations = np.arange(minimum_steps, largest + 1, dtype=np.int64)
                starts = endpoint - durations
                segment_score = eating_prefix[endpoint] - eating_prefix[starts]
                duration_seconds = durations.astype(np.float64) * self.grid_seconds
                candidates = (
                    dp[starts]
                    + segment_score
                    + self.duration_weight * self.prior.log_probability(duration_seconds)
                )
                winner = int(np.argmax(candidates))
                if candidates[winner] > best_score:
                    best_score = float(candidates[winner])
                    best_previous = int(starts[winner])
                    best_label = True
            dp[endpoint] = best_score
            previous[endpoint] = best_previous
            labels[endpoint] = best_label
            commit = endpoint - 1 - lag_steps
            if commit >= 0:
                finalized[commit] = self._state_at(endpoint, commit, previous, labels)
        for target in range(max(0, steps - lag_steps), steps):
            finalized[target] = self._state_at(steps, target, previous, labels)
        return finalized

    def decode_events(
        self, timestamps_ms: np.ndarray, probabilities: np.ndarray
    ) -> list[tuple[int, int, float]]:
        timestamps = np.asarray(timestamps_ms, dtype=np.int64)
        probability = np.asarray(probabilities, dtype=np.float64)
        if len(timestamps) != len(probability):
            raise ValueError("Decoder timestamps and probabilities must align")
        states = self.decode_states(probability)
        padded = np.concatenate(([False], states, [False])).astype(np.int8)
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        output: list[tuple[int, int, float]] = []
        for start, end in zip(starts, ends):
            start_ms = int(timestamps[start])
            end_ms = int(timestamps[end - 1] + self.grid_seconds * 1000)
            output.append((start_ms, end_ms, float(probability[start:end].mean())))
        return output
