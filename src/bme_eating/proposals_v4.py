from __future__ import annotations

import hashlib
from enum import IntFlag
from typing import Any

import numpy as np
import pandas as pd

from bme_eating.structured_decoder import (
    FixedLagSemiMarkovDecoder,
    right_endpoint_run_to_interval,
)


class ProposalSource(IntFlag):
    HYSTERESIS = 1
    SEMI_MARKOV = 2
    TRANSITION = 4
    JITTER = 8


ALLOWED_SOURCE_MASK = int(
    ProposalSource.HYSTERESIS
    | ProposalSource.SEMI_MARKOV
    | ProposalSource.TRANSITION
    | ProposalSource.JITTER
)


def interval_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    intersection = max(0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    return intersection / union if union > 0 else 0.0


def causal_ema(
    probabilities: np.ndarray, half_life_seconds: float, step_seconds: float
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if half_life_seconds < 0 or step_seconds <= 0:
        raise ValueError("EMA half-life must be non-negative and step size positive")
    if half_life_seconds == 0:
        return values.copy()
    alpha = 1.0 - np.exp(-np.log(2.0) * step_seconds / half_life_seconds)
    output = np.empty_like(values)
    if not len(values):
        return output
    output[0] = values[0]
    for index in range(1, len(values)):
        output[index] = alpha * values[index] + (1.0 - alpha) * output[index - 1]
    return output


def _hysteresis(
    timestamps: np.ndarray,
    probabilities: np.ndarray,
    *,
    high: float,
    low: float,
    step_ms: int,
) -> list[tuple[int, int, float, int]]:
    if not 0 <= low <= high <= 1:
        raise ValueError("Hysteresis thresholds must satisfy 0 <= low <= high <= 1")
    events: list[tuple[int, int, float, int]] = []
    active = False
    start = 0
    for index, probability in enumerate(probabilities):
        if not active and probability >= high:
            active = True
            start = index
        elif active and probability < low:
            start_ms, end_ms = right_endpoint_run_to_interval(timestamps, start, index, step_ms)
            events.append(
                (
                    start_ms,
                    end_ms,
                    float(np.max(probabilities[start:index])),
                    int(ProposalSource.HYSTERESIS),
                )
            )
            active = False
    if active:
        start_ms, end_ms = right_endpoint_run_to_interval(
            timestamps, start, len(timestamps), step_ms
        )
        events.append(
            (
                start_ms,
                end_ms,
                float(np.max(probabilities[start:])),
                int(ProposalSource.HYSTERESIS),
            )
        )
    return events


def _merge_gaps(
    events: list[tuple[int, int, float, int]], maximum_gap_ms: int
) -> list[tuple[int, int, float, int]]:
    output: list[tuple[int, int, float, int]] = []
    for event in sorted(events):
        if output and event[0] - output[-1][1] <= maximum_gap_ms:
            previous = output.pop()
            output.append(
                (
                    previous[0],
                    max(previous[1], event[1]),
                    max(previous[2], event[2]),
                    previous[3] | event[3],
                )
            )
        else:
            output.append(event)
    return output


def _transition_candidates(
    timestamps: np.ndarray,
    onset: np.ndarray,
    offset: np.ndarray,
    *,
    threshold: float,
    minimum_ms: int,
    maximum_ms: int,
) -> list[tuple[int, int, float, int]]:
    onset_peaks = np.flatnonzero(
        (onset >= threshold) & (onset >= np.r_[0.0, onset[:-1]]) & (onset >= np.r_[onset[1:], 0.0])
    )
    offset_peaks = np.flatnonzero(
        (offset >= threshold)
        & (offset >= np.r_[0.0, offset[:-1]])
        & (offset >= np.r_[offset[1:], 0.0])
    )
    output: list[tuple[int, int, float, int]] = []
    for start_index in onset_peaks:
        duration = timestamps[offset_peaks] - timestamps[start_index]
        eligible = offset_peaks[(duration >= 0) & (duration <= maximum_ms)]
        if not len(eligible):
            continue
        end_index = int(eligible[0])
        start_ms = int(timestamps[start_index])
        end_ms = max(int(timestamps[end_index]), start_ms + minimum_ms)
        if end_ms > int(timestamps[-1]) or end_ms - start_ms > maximum_ms:
            continue
        output.append(
            (
                start_ms,
                end_ms,
                float(np.sqrt(onset[start_index] * offset[end_index])),
                int(ProposalSource.TRANSITION),
            )
        )
    return output


def _deduplicate(
    events: list[tuple[int, int, float, int, str]], threshold: float
) -> list[tuple[int, int, float, int, str]]:
    kept: list[tuple[int, int, float, int, str]] = []
    for event in sorted(events, key=lambda value: (-value[2], value[0], value[1], value[3])):
        match = next(
            (
                index
                for index, existing in enumerate(kept)
                if interval_iou(event[0], event[1], existing[0], existing[1]) > threshold
            ),
            None,
        )
        if match is None:
            kept.append(event)
        else:
            existing = kept[match]
            kept[match] = (
                existing[0],
                existing[1],
                max(existing[2], event[2]),
                existing[3] | event[3],
                existing[4],
            )
    return sorted(kept, key=lambda value: (value[0], value[1], -value[2]))


def _jitter(
    events: list[tuple[int, int, float, int, str]],
    jitter_seconds: list[int],
    maximum_variants: int,
    minimum_ms: int,
    maximum_ms: int,
    *,
    observation_start_ms: int,
    observation_end_ms: int,
) -> list[tuple[int, int, float, int, str]]:
    if observation_end_ms <= observation_start_ms:
        raise ValueError("Proposal jitter requires a positive observation interval")
    shifts = sorted(
        ((left, right) for left in jitter_seconds for right in jitter_seconds),
        key=lambda value: (abs(value[0]) + abs(value[1]), abs(value[0] - value[1]), value),
    )[:maximum_variants]
    output: list[tuple[int, int, float, int, str]] = []
    for start, end, score, source, family_id in events:
        for left, right in shifts:
            candidate_start = max(int(observation_start_ms), start + left * 1000)
            candidate_end = min(int(observation_end_ms), end + right * 1000)
            duration = candidate_end - candidate_start
            if minimum_ms <= duration <= maximum_ms:
                mask = source | (int(ProposalSource.JITTER) if left or right else 0)
                penalty = np.exp(-0.01 * (abs(left) + abs(right)))
                output.append(
                    (candidate_start, candidate_end, float(score * penalty), mask, family_id)
                )
    return output


def _proposal_id(subject: str, session: str, start: int, end: int, source: int) -> str:
    payload = f"v4|{subject}|{session}|{start}|{end}|{source}".encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def decoder_valid_mask(windows: pd.DataFrame) -> np.ndarray:
    if windows.empty:
        return np.zeros(0, dtype=bool)
    if "decode_valid" in windows:
        return windows["decode_valid"].fillna(False).to_numpy(dtype=bool)
    validity_columns = [
        column
        for column in ("acc_valid_fraction", "gyro_valid_fraction", "ppg_valid_fraction")
        if column in windows
    ]
    if validity_columns:
        validity = np.nan_to_num(windows[validity_columns].to_numpy(dtype=np.float64), nan=0.0)
        return validity.max(axis=1) > 0.0
    if "active_modality_missing_fraction" in windows:
        missing = windows["active_modality_missing_fraction"].to_numpy(dtype=np.float64)
        return np.isfinite(missing) & (missing < 1.0 - 1e-6)
    return np.ones(len(windows), dtype=bool)


def configured_decoder_valid_mask(
    windows: pd.DataFrame,
    *,
    use_motion: bool,
    use_ppg: bool,
) -> np.ndarray:
    masks: list[np.ndarray] = []
    if use_motion:
        motion_columns = [
            column for column in ("acc_valid_fraction", "gyro_valid_fraction") if column in windows
        ]
        if motion_columns:
            motion = np.nan_to_num(windows[motion_columns].to_numpy(dtype=np.float64), nan=0.0)
            masks.append(motion.max(axis=1) > 0.0)
        elif "motion_valid_fraction" in windows:
            masks.append(windows["motion_valid_fraction"].fillna(0.0).to_numpy(dtype=float) > 0.0)
    if use_ppg and "ppg_valid_fraction" in windows:
        masks.append(windows["ppg_valid_fraction"].fillna(0.0).to_numpy(dtype=float) > 0.0)
    if not masks:
        return decoder_valid_mask(windows)
    return np.logical_or.reduce(masks)


def _valid_timeline_runs(windows: pd.DataFrame) -> tuple[list[pd.DataFrame], int]:
    ordered = windows.sort_values("timestamp_ms", kind="stable").reset_index(drop=True)
    if ordered.empty:
        return [], 3000
    timestamps = ordered["timestamp_ms"].to_numpy(dtype=np.int64)
    positive_differences = np.diff(timestamps)
    positive_differences = positive_differences[positive_differences > 0]
    step_ms = int(np.median(positive_differences)) if len(positive_differences) else 3000
    valid = decoder_valid_mask(ordered)
    positions = np.flatnonzero(valid)
    if not len(positions):
        return [], step_ms
    split = (
        np.flatnonzero((np.diff(positions) != 1) | (np.diff(timestamps[positions]) != step_ms)) + 1
    )
    runs = [
        ordered.iloc[indices].reset_index(drop=True)
        for indices in np.split(positions, split)
        if len(indices)
    ]
    return runs, step_ms


def observed_hours_v4(windows: pd.DataFrame) -> float:
    milliseconds = 0
    for _, group in windows.groupby(["subject_key", "session_id"], sort=False):
        _, step_ms = _valid_timeline_runs(group)
        milliseconds += int(decoder_valid_mask(group).sum()) * step_ms
    return milliseconds / 3_600_000.0


def hysteresis_fragment_diagnostics(
    windows: pd.DataFrame, config: dict[str, Any]
) -> dict[str, float]:
    count = 0
    observed_hours = 0.0
    for _, group in windows.groupby(["subject_key", "session_id"], sort=False):
        runs, step_ms = _valid_timeline_runs(group)
        observed_hours += sum(len(run) for run in runs) * step_ms / 3_600_000.0
        for run in runs:
            timestamps = run["timestamp_ms"].to_numpy(dtype=np.int64)
            if len(timestamps) < 2:
                continue
            probabilities = run["state_probability"].to_numpy(dtype=np.float64)
            smoothed = causal_ema(
                probabilities,
                float(config["ema_half_life_seconds"]),
                step_ms / 1000.0,
            )
            fragments = _merge_gaps(
                _hysteresis(
                    timestamps,
                    smoothed,
                    high=float(config["high_threshold"]),
                    low=float(config["low_threshold"]),
                    step_ms=step_ms,
                ),
                int(config["gap_merge_seconds"]) * 1000,
            )
            count += len(fragments)
    return {
        "state_fragment_count": float(count),
        "state_fragments_per_hour": float(count / max(observed_hours, 1e-9)),
    }


def generate_event_candidates_v4(
    windows: pd.DataFrame,
    decoder: FixedLagSemiMarkovDecoder,
    config: dict[str, Any],
    *,
    split_role: str,
) -> pd.DataFrame:
    required = {
        "subject_key",
        "session_id",
        "timestamp_ms",
        "state_probability",
        "onset_probability",
        "offset_probability",
    }
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"V4 window predictions are missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    observed_hours: dict[tuple[str, str], float] = {}
    for (subject, session), group in windows.groupby(["subject_key", "session_id"], sort=False):
        session_key = (str(subject), str(session))
        runs, step_ms = _valid_timeline_runs(group)
        observed_hours[session_key] = sum(len(run) for run in runs) * step_ms / 3_600_000.0
        minimum_ms = round(float(config.get("candidate_minimum_seconds", 3)) * 1000)
        maximum_ms = round(float(config.get("candidate_maximum_seconds", 14_400)) * 1000)
        if minimum_ms <= 0 or maximum_ms < minimum_ms:
            raise ValueError("Candidate duration limits are invalid")
        for run in runs:
            timestamps = run["timestamp_ms"].to_numpy(dtype=np.int64)
            state = run["state_probability"].to_numpy(dtype=np.float64)
            onset = run["onset_probability"].to_numpy(dtype=np.float64)
            offset = run["offset_probability"].to_numpy(dtype=np.float64)
            if len(timestamps) < 2:
                continue
            smoothed = causal_ema(
                state,
                float(config["ema_half_life_seconds"]),
                step_ms / 1000.0,
            )
            seeds = _hysteresis(
                timestamps,
                smoothed,
                high=float(config["high_threshold"]),
                low=float(config["low_threshold"]),
                step_ms=step_ms,
            )
            seeds = _merge_gaps(seeds, int(config["gap_merge_seconds"]) * 1000)
            grid_ms = int(config["grid_seconds"]) * 1000
            grid_start = int(-(-int(timestamps[0]) // grid_ms) * grid_ms)
            grid_end = int(timestamps[-1] // grid_ms * grid_ms)
            grid = np.arange(grid_start, grid_end + 1, grid_ms, dtype=np.int64)
            if len(grid) and bool(config.get("use_semi_markov", True)):
                if grid[0] < timestamps[0] or grid[-1] > timestamps[-1]:
                    raise RuntimeError("Semi-Markov grid extends beyond observed timestamps")
                sampled = np.interp(grid, timestamps, smoothed)
                seeds.extend(
                    (
                        start,
                        end,
                        score,
                        int(ProposalSource.SEMI_MARKOV),
                    )
                    for start, end, score in decoder.decode_events(grid, sampled)
                )
            if bool(config.get("use_transition_candidates", True)):
                seeds.extend(
                    _transition_candidates(
                        timestamps,
                        onset,
                        offset,
                        threshold=float(config["transition_threshold"]),
                        minimum_ms=minimum_ms,
                        maximum_ms=maximum_ms,
                    )
                )
            seeds = [event for event in seeds if minimum_ms <= event[1] - event[0] <= maximum_ms]
            family_seeds = [
                (
                    *event,
                    _proposal_id(
                        str(subject),
                        str(session),
                        int(event[0]),
                        int(event[1]),
                        int(event[3]),
                    ),
                )
                for event in seeds
            ]
            variants = _jitter(
                family_seeds,
                [int(value) for value in config["jitter_seconds"]],
                int(config["maximum_variants_per_event"]),
                minimum_ms,
                maximum_ms,
                observation_start_ms=int(timestamps[0] - step_ms),
                observation_end_ms=int(timestamps[-1]),
            )
            deduplicated = _deduplicate(variants, float(config["deduplication_iou"]))
            for start, end, score, source, family_id in deduplicated:
                if source & ~ALLOWED_SOURCE_MASK:
                    raise RuntimeError("A non-deep proposal source entered the v4 pipeline")
                rows.append(
                    {
                        "proposal_id": _proposal_id(str(subject), str(session), start, end, source),
                        "proposal_family_id": family_id,
                        "subject_key": str(subject),
                        "session_id": str(session),
                        "coarse_start_ms": int(start),
                        "coarse_end_ms": int(end),
                        "source_mask": int(source),
                        "generator_score": float(score),
                        "rank_within_session": 0,
                        "split_role": split_role,
                    }
                )
    columns = [
        "proposal_id",
        "proposal_family_id",
        "subject_key",
        "session_id",
        "coarse_start_ms",
        "coarse_end_ms",
        "source_mask",
        "generator_score",
        "rank_within_session",
        "split_role",
    ]
    unbudgeted = pd.DataFrame(rows, columns=columns)
    if unbudgeted.empty:
        return unbudgeted
    selected_frames: list[pd.DataFrame] = []
    maximum_per_hour = int(config["maximum_candidates_per_hour"])
    for (subject, session), group in unbudgeted.groupby(["subject_key", "session_id"], sort=False):
        session_key = (str(subject), str(session))
        budget = max(1, int(np.ceil(observed_hours[session_key] * maximum_per_hour)))
        ordering = ["generator_score", "coarse_start_ms", "coarse_end_ms", "proposal_id"]
        ascending = [False, True, True, True]
        ranked_group = group.sort_values(ordering, ascending=ascending, kind="stable")
        family_representatives = ranked_group.drop_duplicates("proposal_family_id", keep="first")
        preferred: list[int] = []
        for source in (
            ProposalSource.HYSTERESIS,
            ProposalSource.SEMI_MARKOV,
            ProposalSource.TRANSITION,
        ):
            eligible = family_representatives[
                (family_representatives["source_mask"].astype(int) & int(source)) > 0
            ]
            if len(eligible):
                preferred.append(int(eligible.index[0]))
        preferred = (
            unbudgeted.loc[list(dict.fromkeys(preferred))]
            .sort_values(ordering, ascending=ascending, kind="stable")
            .index.tolist()
        )
        family_ranked = family_representatives.index.tolist()
        variant_ranked = ranked_group.index.tolist()
        selected: list[int] = []
        for index in [*preferred, *family_ranked, *variant_ranked]:
            if index not in selected:
                selected.append(index)
            if len(selected) >= budget:
                break
        selected_frames.append(unbudgeted.loc[selected].copy())
    output = pd.concat(selected_frames, ignore_index=True)
    output["rank_within_session"] = (
        output.sort_values(
            ["generator_score", "coarse_start_ms", "coarse_end_ms", "proposal_id"],
            ascending=[False, True, True, True],
            kind="stable",
        )
        .groupby(["subject_key", "session_id"], sort=False)
        .cumcount()
        .add(1)
        .reindex(output.index)
        .astype(int)
    )
    return (
        output[columns]
        .sort_values(["subject_key", "session_id", "rank_within_session"], kind="stable")
        .reset_index(drop=True)
    )
