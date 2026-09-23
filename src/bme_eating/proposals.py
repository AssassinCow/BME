from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

SOURCE_STATE = 1
SOURCE_HINT = 2
SOURCE_JITTER = 4
SOURCE_XGBOOST = 8

PROPOSAL_COLUMNS = (
    "proposal_id",
    "subject_key",
    "session_id",
    "coarse_start_ms",
    "coarse_end_ms",
    "source_mask",
    "generator_score",
    "rank_within_session",
    "split_role",
)


def interval_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    intersection = max(0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    return intersection / union if union > 0 else 0.0


def duration_bounds_from_training_events(
    events: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[float, float]:
    selected = events[events["end_ms"] > events["start_ms"]].copy()
    if "evaluable" in selected:
        selected = selected[selected["evaluable"].astype(bool)]
    if selected.empty:
        raise ValueError("Duration bounds require at least one positive-duration training event")
    duration = (selected["end_ms"] - selected["start_ms"]).to_numpy(dtype=np.float64) / 1000
    minimum = max(
        float(config["minimum_duration_floor_seconds"]),
        float(np.quantile(duration, float(config["duration_lower_quantile"]))),
    )
    maximum = min(
        float(config["maximum_duration_ceiling_seconds"]),
        float(np.quantile(duration, float(config["duration_upper_quantile"]))),
    )
    if not 0 < minimum < maximum:
        raise ValueError("Training-derived proposal duration bounds are invalid")
    return minimum, maximum


def _hysteresis_candidates(
    frame: pd.DataFrame,
    high: float,
    low: float,
) -> list[dict[str, object]]:
    if not 0 <= low < high <= 1:
        raise ValueError("Hysteresis thresholds must satisfy 0 <= low < high <= 1")
    timestamps = frame["timestamp_ms"].to_numpy(dtype=np.int64)
    probability = frame["state_probability"].to_numpy(dtype=np.float64)
    if len(timestamps) == 0:
        return []
    step = int(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 3000
    candidates: list[dict[str, object]] = []
    active = False
    start = 0
    for index, value in enumerate(probability):
        if not active and value >= high:
            active = True
            start = index
        elif active and value < low:
            end = max(start, index - 1)
            candidates.append(
                {
                    "coarse_start_ms": int(timestamps[start]),
                    "coarse_end_ms": int(timestamps[end] + step),
                    "source_mask": SOURCE_STATE,
                    "generator_score": float(probability[start : end + 1].mean()),
                }
            )
            active = False
    if active:
        candidates.append(
            {
                "coarse_start_ms": int(timestamps[start]),
                "coarse_end_ms": int(timestamps[-1] + step),
                "source_mask": SOURCE_STATE,
                "generator_score": float(probability[start:].mean()),
            }
        )
    return candidates


def _hint_candidates(
    frame: pd.DataFrame,
    threshold: float,
    minimum_seconds: float,
    maximum_seconds: float,
) -> list[dict[str, object]]:
    timestamps = frame["timestamp_ms"].to_numpy(dtype=np.int64)
    start_probability = frame["start_probability"].to_numpy(dtype=np.float64)
    end_probability = frame["end_probability"].to_numpy(dtype=np.float64)
    starts, _ = find_peaks(start_probability, height=threshold)
    ends, _ = find_peaks(end_probability, height=threshold)
    candidates: list[dict[str, object]] = []
    minimum_ms = int(minimum_seconds * 1000)
    maximum_ms = int(maximum_seconds * 1000)
    for start_index in starts:
        valid_ends = [
            int(end_index)
            for end_index in ends
            if minimum_ms
            <= int(timestamps[end_index] - timestamps[start_index])
            <= maximum_ms
        ]
        valid_ends.sort(key=lambda value: end_probability[value], reverse=True)
        for end_index in valid_ends[:3]:
            score = math.sqrt(
                max(0.0, start_probability[start_index] * end_probability[end_index])
            )
            candidates.append(
                {
                    "coarse_start_ms": int(timestamps[start_index]),
                    "coarse_end_ms": int(timestamps[end_index]),
                    "source_mask": SOURCE_HINT,
                    "generator_score": float(score),
                }
            )
    return candidates


def _jitter_candidates(
    seeds: Iterable[dict[str, object]],
    offsets_seconds: list[int],
    maximum_variants: int,
    minimum_seconds: float,
    maximum_seconds: float,
) -> list[dict[str, object]]:
    combinations = sorted(
        ((left, right) for left in offsets_seconds for right in offsets_seconds),
        key=lambda pair: (abs(pair[0]) + abs(pair[1]), abs(pair[0] - pair[1]), pair),
    )
    output: list[dict[str, object]] = []
    for seed in seeds:
        kept = 0
        for left, right in combinations:
            start = int(seed["coarse_start_ms"]) + left * 1000
            end = int(seed["coarse_end_ms"]) + right * 1000
            duration = (end - start) / 1000
            if not minimum_seconds <= duration <= maximum_seconds:
                continue
            output.append(
                {
                    "coarse_start_ms": start,
                    "coarse_end_ms": end,
                    "source_mask": int(seed["source_mask"]) | SOURCE_JITTER,
                    "generator_score": float(seed["generator_score"])
                    * math.exp(-0.01 * (abs(left) + abs(right))),
                }
            )
            kept += 1
            if kept >= maximum_variants:
                break
    return output


def _deduplicate(
    candidates: list[dict[str, object]], threshold: float
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for candidate in sorted(
        candidates,
        key=lambda row: (
            -float(row["generator_score"]),
            int(row["coarse_start_ms"]),
            int(row["coarse_end_ms"]),
        ),
    ):
        duplicate = None
        for existing in selected:
            if interval_iou(
                int(candidate["coarse_start_ms"]),
                int(candidate["coarse_end_ms"]),
                int(existing["coarse_start_ms"]),
                int(existing["coarse_end_ms"]),
            ) > threshold:
                duplicate = existing
                break
        if duplicate is None:
            selected.append(dict(candidate))
        else:
            duplicate["source_mask"] = int(duplicate["source_mask"]) | int(
                candidate["source_mask"]
            )
    return selected


def _proposal_id(subject: str, session: str, start: int, end: int, source_mask: int) -> str:
    payload = f"{subject}|{session}|{start}|{end}|{source_mask}".encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def _apply_candidate_budget(
    candidates: list[dict[str, object]], budget: int
) -> list[dict[str, object]]:
    if budget <= 0:
        raise ValueError("Candidate budget must be positive")
    required: list[dict[str, object]] = []
    for source in (SOURCE_STATE, SOURCE_HINT, SOURCE_XGBOOST):
        candidate = next(
            (row for row in candidates if int(row["source_mask"]) & source), None
        )
        if candidate is not None and candidate not in required:
            required.append(candidate)
    effective_budget = max(budget, len(required))
    selected = list(required)
    for candidate in candidates:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) >= effective_budget:
            break
    return sorted(
        selected[:effective_budget],
        key=lambda row: (
            -float(row["generator_score"]),
            int(row["coarse_start_ms"]),
            int(row["coarse_end_ms"]),
        ),
    )


def generate_event_candidates(
    predictions: pd.DataFrame,
    config: dict[str, Any],
    duration_bounds: tuple[float, float],
    *,
    split_role: str,
    xgb_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {
        "subject_key",
        "session_id",
        "timestamp_ms",
        "state_probability",
        "start_probability",
        "end_probability",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Window predictions are missing columns: {sorted(missing)}")
    minimum_seconds, maximum_seconds = duration_bounds
    xgb_lookup: dict[tuple[str, str], list[dict[str, object]]] = {}
    if xgb_events is not None and not xgb_events.empty:
        xgb_required = {"subject_key", "session_id", "start_ms", "end_ms", "score"}
        xgb_missing = xgb_required - set(xgb_events.columns)
        if xgb_missing:
            raise ValueError(f"XGBoost events are missing columns: {sorted(xgb_missing)}")
        for key, group in xgb_events.groupby(["subject_key", "session_id"], sort=False):
            xgb_lookup[(str(key[0]), str(key[1]))] = [
                {
                    "coarse_start_ms": int(row.start_ms),
                    "coarse_end_ms": int(row.end_ms),
                    "source_mask": SOURCE_XGBOOST,
                    "generator_score": float(row.score),
                }
                for row in group.itertuples(index=False)
            ]

    rows: list[dict[str, object]] = []
    for (subject, session), frame in predictions.groupby(
        ["subject_key", "session_id"], sort=True
    ):
        frame = frame.sort_values("timestamp_ms")
        seed_candidates = _hysteresis_candidates(
            frame, float(config["high_threshold"]), float(config["low_threshold"])
        )
        seed_candidates.extend(
            _hint_candidates(
                frame,
                float(config["hint_threshold"]),
                minimum_seconds,
                maximum_seconds,
            )
        )
        seed_candidates.extend(xgb_lookup.get((str(subject), str(session)), []))
        timestamp_values = frame["timestamp_ms"].to_numpy(dtype=np.int64)
        step = (
            int(np.median(np.diff(timestamp_values)))
            if len(timestamp_values) > 1
            else 3000
        )
        observed_start = int(timestamp_values[0])
        observed_end = int(timestamp_values[-1] + step)
        valid_seeds = [
            row
            for row in seed_candidates
            if minimum_seconds
            <= (int(row["coarse_end_ms"]) - int(row["coarse_start_ms"])) / 1000
            <= maximum_seconds
            and int(row["coarse_start_ms"]) >= observed_start
            and int(row["coarse_end_ms"]) <= observed_end
        ]
        candidates = valid_seeds + _jitter_candidates(
            valid_seeds,
            [int(value) for value in config["jitter_seconds"]],
            int(config["maximum_variants_per_event"]),
            minimum_seconds,
            maximum_seconds,
        )
        candidates = [
            row
            for row in candidates
            if int(row["coarse_start_ms"]) >= observed_start
            and int(row["coarse_end_ms"]) <= observed_end
        ]
        candidates = _deduplicate(candidates, float(config["deduplication_iou"]))
        observed_ms = max(
            1,
            int(frame["timestamp_ms"].max()) - int(frame["timestamp_ms"].min()),
        )
        budget = max(
            1,
            math.ceil(
                observed_ms / 3_600_000 * float(config["maximum_candidates_per_hour"])
            ),
        )
        candidates = _apply_candidate_budget(candidates, budget)
        for rank, candidate in enumerate(candidates):
            start = int(candidate["coarse_start_ms"])
            end = int(candidate["coarse_end_ms"])
            source_mask = int(candidate["source_mask"])
            rows.append(
                {
                    "proposal_id": _proposal_id(
                        str(subject), str(session), start, end, source_mask
                    ),
                    "subject_key": str(subject),
                    "session_id": str(session),
                    "coarse_start_ms": start,
                    "coarse_end_ms": end,
                    "source_mask": source_mask,
                    "generator_score": float(candidate["generator_score"]),
                    "rank_within_session": rank,
                    "split_role": split_role,
                }
            )
    return pd.DataFrame(rows, columns=PROPOSAL_COLUMNS)


def label_event_candidates(
    proposals: pd.DataFrame,
    events: pd.DataFrame,
    iou_threshold: float,
) -> pd.DataFrame:
    output = proposals.copy()
    if output.empty:
        output["max_iou"] = pd.Series(dtype=np.float32)
        output["matched_event_id"] = pd.Series(dtype=object)
        output["is_positive"] = pd.Series(dtype=bool)
        output["negative_type"] = pd.Series(dtype=object)
        return output
    labels: list[tuple[float, str | None]] = []
    grouped = {
        str(subject): group
        for subject, group in events.groupby("subject_key", sort=False)
    }
    for proposal in output.itertuples(index=False):
        best_iou = 0.0
        best_id: str | None = None
        for event in grouped.get(str(proposal.subject_key), pd.DataFrame()).itertuples(
            index=False
        ):
            value = interval_iou(
                int(proposal.coarse_start_ms),
                int(proposal.coarse_end_ms),
                int(event.start_ms),
                int(event.end_ms),
            )
            if value > best_iou:
                best_iou = value
                best_id = str(getattr(event, "event_id", "")) or None
        labels.append((best_iou, best_id))
    output["max_iou"] = np.asarray([value for value, _ in labels], dtype=np.float32)
    output["matched_event_id"] = [event_id for _, event_id in labels]
    output["is_positive"] = output["max_iou"] > float(iou_threshold)
    output["negative_type"] = np.select(
        [
            output["is_positive"],
            output["max_iou"].between(0.10, iou_threshold, inclusive="both"),
            output["generator_score"] >= output["generator_score"].quantile(0.75),
        ],
        ["positive", "near_miss", "hard_false_positive"],
        default="random_background",
    )
    return output


def exclude_ignored_candidates(
    proposals: pd.DataFrame,
    ignored_events: pd.DataFrame,
) -> pd.DataFrame:
    if proposals.empty or ignored_events.empty:
        return proposals.copy()
    grouped = {
        str(subject): group for subject, group in ignored_events.groupby("subject_key")
    }
    keep: list[bool] = []
    for proposal in proposals.itertuples(index=False):
        ignored = grouped.get(str(proposal.subject_key), pd.DataFrame())
        overlaps = any(
            min(int(proposal.coarse_end_ms), int(event.end_ms))
            > max(int(proposal.coarse_start_ms), int(event.start_ms))
            for event in ignored.itertuples(index=False)
        )
        keep.append(bool(getattr(proposal, "is_positive", False)) or not overlaps)
    return proposals.loc[np.asarray(keep, dtype=bool)].reset_index(drop=True)
