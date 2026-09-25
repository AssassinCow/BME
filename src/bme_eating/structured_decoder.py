from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def right_endpoint_run_to_interval(
    timestamps_ms: np.ndarray, start: int, end: int, step_ms: int
) -> tuple[int, int]:
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    if not 0 <= int(start) < int(end) <= len(timestamps):
        raise ValueError("State run indices are outside the timestamp grid")
    if int(step_ms) <= 0:
        raise ValueError("Right-endpoint interval step must be positive")
    return int(timestamps[int(start)] - int(step_ms)), int(timestamps[int(end) - 1])


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
        state: int,
        previous_endpoint: np.ndarray,
        previous_state: np.ndarray,
    ) -> bool:
        cursor = int(endpoint)
        cursor_state = int(state)
        while cursor > 0:
            start = int(previous_endpoint[cursor_state, cursor])
            if start <= target < cursor:
                return bool(cursor_state)
            cursor_state = int(previous_state[cursor_state, cursor])
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
        background = 0
        eating = 1
        scores = np.full((2, steps + 1), -np.inf, dtype=np.float64)
        previous_endpoint = np.zeros((2, steps + 1), dtype=np.int64)
        previous_state = np.zeros((2, steps + 1), dtype=np.int8)
        scores[background, 0] = 0.0
        finalized = np.zeros(steps, dtype=bool)
        lag_steps = int(np.ceil(self.fixed_lag_seconds / self.grid_seconds))
        for endpoint in range(1, steps + 1):
            background_source = int(np.argmax(scores[:, endpoint - 1]))
            scores[background, endpoint] = (
                scores[background_source, endpoint - 1]
                + np.log1p(-probability[endpoint - 1])
            )
            previous_endpoint[background, endpoint] = endpoint - 1
            previous_state[background, endpoint] = background_source
            largest = min(maximum_steps, endpoint)
            if largest >= minimum_steps:
                durations = np.arange(minimum_steps, largest + 1, dtype=np.int64)
                starts = endpoint - durations
                segment_score = eating_prefix[endpoint] - eating_prefix[starts]
                duration_seconds = durations.astype(np.float64) * self.grid_seconds
                candidates = (
                    scores[background, starts]
                    + segment_score
                    + self.duration_weight * self.prior.log_probability(duration_seconds)
                )
                winner = int(np.argmax(candidates))
                if np.isfinite(candidates[winner]):
                    scores[eating, endpoint] = float(candidates[winner])
                    previous_endpoint[eating, endpoint] = int(starts[winner])
                    previous_state[eating, endpoint] = background
            commit = endpoint - 1 - lag_steps
            if commit >= 0:
                terminal_state = int(np.argmax(scores[:, endpoint]))
                finalized[commit] = self._state_at(
                    endpoint,
                    commit,
                    terminal_state,
                    previous_endpoint,
                    previous_state,
                )
        terminal_state = int(np.argmax(scores[:, steps]))
        for target in range(max(0, steps - lag_steps), steps):
            finalized[target] = self._state_at(
                steps,
                target,
                terminal_state,
                previous_endpoint,
                previous_state,
            )
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
            start_ms, end_ms = right_endpoint_run_to_interval(
                timestamps, int(start), int(end), self.grid_seconds * 1000
            )
            output.append((start_ms, end_ms, float(probability[start:end].mean())))
        return output
