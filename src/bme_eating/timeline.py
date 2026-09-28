from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np
import pandas as pd

TIMELINE_KEYS = ("subject_key", "session_id", "timestamp_ms")


def tail_aligned_chunk_endpoints(total_rows: int, chunk_size: int) -> list[int]:
    total = int(total_rows)
    size = int(chunk_size)
    if total < 0:
        raise ValueError("Timeline row count must be non-negative")
    if size <= 0:
        raise ValueError("Timeline chunk size must be positive")
    if total == 0:
        return []
    last = total - 1
    first = last % size
    return list(range(first, total, size))


def claim_new_timeline_rows(
    frame: pd.DataFrame,
    previous_chunks: MutableMapping[tuple[str, str], pd.DataFrame],
    *,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    missing = set(TIMELINE_KEYS) - set(frame.columns)
    if missing:
        raise ValueError(f"Timeline is missing alignment keys: {sorted(missing)}")
    claimed: list[pd.DataFrame] = []
    for (subject_key, session_id), group in frame.groupby(
        ["subject_key", "session_id"], sort=False
    ):
        ordered = group.sort_values("timestamp_ms", kind="stable")
        if ordered["timestamp_ms"].duplicated().any():
            raise RuntimeError("A single inference chunk contains duplicate anchor timestamps")
        key = (str(subject_key), str(session_id))
        previous = previous_chunks.get(key)
        previous_maximum: int | None = None
        if previous is not None:
            previous_maximum = int(previous["timestamp_ms"].max())
            overlap = ordered.loc[ordered["timestamp_ms"] <= previous_maximum]
            if not overlap.empty:
                previous_overlap = previous[
                    previous["timestamp_ms"].isin(overlap["timestamp_ms"])
                ]
                if len(previous_overlap) != len(overlap):
                    raise RuntimeError("Inference chunks are not monotonically aligned")
                deduplicate_consistent_timeline(
                    pd.concat((previous_overlap, overlap), ignore_index=True),
                    tolerance=tolerance,
                )
        previous_chunks[key] = ordered.copy()
        new_rows = (
            ordered
            if previous_maximum is None
            else ordered.loc[ordered["timestamp_ms"] > previous_maximum]
        )
        if new_rows.empty:
            continue
        claimed.append(new_rows)
    if not claimed:
        return frame.iloc[0:0].copy()
    return pd.concat(claimed, ignore_index=True)


def deduplicate_consistent_timeline(
    frame: pd.DataFrame,
    *,
    tolerance: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = set(TIMELINE_KEYS) - set(frame.columns)
    if missing:
        raise ValueError(f"Timeline is missing alignment keys: {sorted(missing)}")
    ordered = frame.sort_values(list(TIMELINE_KEYS), kind="stable").reset_index(drop=True)
    duplicate_mask = ordered.duplicated(list(TIMELINE_KEYS), keep=False)
    conflicts: list[dict[str, Any]] = []
    duplicate_groups = 0
    if duplicate_mask.any():
        for key, group in ordered.loc[duplicate_mask].groupby(list(TIMELINE_KEYS), sort=False):
            duplicate_groups += 1
            conflict_columns: list[str] = []
            for column in ordered.columns:
                if column in TIMELINE_KEYS:
                    continue
                values = group[column]
                if pd.api.types.is_numeric_dtype(values):
                    numeric = values.to_numpy(dtype=np.float64)
                    if not np.isfinite(numeric).all() or (
                        len(numeric) and float(numeric.max() - numeric.min()) > tolerance
                    ):
                        conflict_columns.append(column)
                elif values.astype(str).nunique(dropna=False) > 1:
                    conflict_columns.append(column)
            if conflict_columns:
                conflicts.append(
                    {
                        "subject_key": str(key[0]),
                        "session_id": str(key[1]),
                        "timestamp_ms": int(key[2]),
                        "columns": conflict_columns,
                    }
                )
    if conflicts:
        raise RuntimeError(
            f"Overlapping inference chunks disagree at duplicate anchors: {conflicts[:5]}"
        )
    deduplicated = ordered.drop_duplicates(list(TIMELINE_KEYS), keep="first").reset_index(drop=True)
    return deduplicated, {
        "input_rows": len(frame),
        "output_rows": len(deduplicated),
        "duplicate_groups": int(duplicate_groups),
        "maximum_tolerance": float(tolerance),
        "conflicts": 0,
    }
