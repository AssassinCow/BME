from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TIMELINE_KEYS = ("subject_key", "session_id", "timestamp_ms")


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
