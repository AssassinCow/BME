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

    def _duration_steps(self) -> tuple[int, int, np.ndarray]:
        minimum_steps = max(1, int(np.ceil(self.prior.minimum_seconds / self.grid_seconds)))
        maximum_steps = int(np.floor(self.prior.maximum_seconds / self.grid_seconds))
        if maximum_steps < minimum_steps:
            raise ValueError("Duration prior contains no legal structured-grid duration")
        durations = np.arange(maximum_steps + 1, dtype=np.float64) * self.grid_seconds
        log_probability = self.prior.log_probability(durations)
        return minimum_steps, maximum_steps, log_probability

    def _decode_window(
        self,
        probabilities: np.ndarray,
        *,
        finalize: bool,
    ) -> np.ndarray:
        probability = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
        minimum_steps, maximum_steps, duration_log_probability = self._duration_steps()
        state_count = maximum_steps + 1
        scores = np.full(state_count, -np.inf, dtype=np.float64)
        scores[0] = 0.0
        backpointers = np.full((len(probability), state_count), -1, dtype=np.int32)
        legal_exits = np.arange(minimum_steps, maximum_steps + 1, dtype=np.int64)
        for time_index, value in enumerate(probability):
            log_eating = float(np.log(value))
            log_background = float(np.log1p(-value))
            updated = np.full_like(scores, -np.inf)

            background_candidates = [float(scores[0] + log_background)]
            background_sources = [0]
            if len(legal_exits):
                exit_scores = (
                    scores[legal_exits]
                    + log_background
                    + self.duration_weight * duration_log_probability[legal_exits]
                )
                best_exit = int(np.argmax(exit_scores))
                background_candidates.append(float(exit_scores[best_exit]))
                background_sources.append(int(legal_exits[best_exit]))
            best_background = int(np.argmax(background_candidates))
            updated[0] = background_candidates[best_background]
            backpointers[time_index, 0] = background_sources[best_background]

            updated[1] = scores[0] + log_eating
            backpointers[time_index, 1] = 0
            if maximum_steps > 1:
                updated[2:] = scores[1:-1] + log_eating
                backpointers[time_index, 2:] = np.arange(1, maximum_steps, dtype=np.int32)
            scores = updated

        terminal_scores = scores.copy()
        if len(probability):
            eating_states = np.arange(1, maximum_steps + 1, dtype=np.int64)
            if finalize:
                terminal_scores[1:minimum_steps] = -np.inf
            adjusted_duration = np.maximum(eating_states, minimum_steps)
            terminal_scores[eating_states] += (
                self.duration_weight * duration_log_probability[adjusted_duration]
            )
        terminal = int(np.argmax(terminal_scores))
        if (
            not finalize
            and np.isfinite(terminal_scores[0])
            and np.isclose(
                terminal_scores[0],
                terminal_scores[terminal],
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            terminal = 0
        if not np.isfinite(terminal_scores[terminal]):
            raise RuntimeError("Fixed-lag decoder found no legal terminal path")
        path = np.empty(len(probability), dtype=np.int32)
        state = terminal
        for time_index in range(len(probability) - 1, -1, -1):
            path[time_index] = state
            state = int(backpointers[time_index, state])
            if state < 0:
                raise RuntimeError("Fixed-lag decoder produced a broken backtrace")
        return path

    @staticmethod
    def _drop_incomplete_runs(
        states: np.ndarray, minimum_steps: int, maximum_steps: int
    ) -> np.ndarray:
        output = np.asarray(states, dtype=bool).copy()
        padded = np.concatenate(([False], output, [False])).astype(np.int8)
        transitions = np.diff(padded)
        for start, end in zip(
            np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
        ):
            duration = int(end - start)
            if duration < minimum_steps:
                output[start:end] = False
            elif duration > maximum_steps:
                raise RuntimeError("Fixed-lag decoder exceeded its maximum eating duration")
        return output

    def decode_states(self, probabilities: np.ndarray) -> np.ndarray:
        probability = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
        if not len(probability):
            return np.zeros(0, dtype=bool)
        minimum_steps, maximum_steps, _ = self._duration_steps()
        lag_steps = int(np.ceil(self.fixed_lag_seconds / self.grid_seconds))
        window: list[float] = []
        expanded_states: list[int] = []
        for value in probability:
            window.append(float(value))
            if len(window) > lag_steps + maximum_steps:
                path = self._decode_window(np.asarray(window), finalize=False)
                eligible = len(window) - lag_steps
                commit_count = eligible
                if commit_count and path[commit_count - 1] > 0:
                    run_start = commit_count - 1
                    while run_start > 0 and path[run_start - 1] > 0:
                        run_start -= 1
                    commit_count = run_start
                if commit_count:
                    if path[commit_count - 1] != 0:
                        raise RuntimeError("Fixed-lag decoder attempted to split an eating segment")
                    expanded_states.extend(int(state) for state in path[:commit_count])
                    del window[:commit_count]
        if window:
            path = self._decode_window(np.asarray(window), finalize=True)
            expanded_states.extend(int(value) for value in path)
        finalized = np.asarray(expanded_states, dtype=np.int32) > 0
        if len(finalized) != len(probability):
            raise RuntimeError("Fixed-lag decoder did not emit one state per grid point")
        return self._drop_incomplete_runs(finalized, minimum_steps, maximum_steps)

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
            duration_seconds = (end_ms - start_ms) / 1000.0
            if not self.prior.minimum_seconds <= duration_seconds <= self.prior.maximum_seconds:
                raise RuntimeError("Semi-Markov event violated its duration prior bounds")
            output.append((start_ms, end_ms, float(probability[start:end].mean())))
        return output
