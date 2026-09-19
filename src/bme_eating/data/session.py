from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd


class SessionWindowReader:
    def __init__(self, segments: pd.DataFrame, cache_size: int = 8) -> None:
        required = {"session_id", "segment_id", "segment_path", "start_ms", "end_ms"}
        missing = required - set(segments.columns)
        if missing:
            raise ValueError(f"Session segments are missing columns: {sorted(missing)}")
        self.segments = segments.sort_values(
            ["session_id", "start_ms", "end_ms", "segment_id"]
        ).reset_index(drop=True)
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def _load(self, path: str) -> dict[str, np.ndarray]:
        if path in self._cache:
            payload = self._cache.pop(path)
            self._cache[path] = payload
            return payload
        from bme_eating.data.deep_dataset import _load_segment_archive

        payload = _load_segment_archive(path)
        self._cache[path] = payload
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return payload

    @staticmethod
    def _combine(
        payloads: list[dict[str, np.ndarray]],
        prefix: str,
        start_ms: int,
        end_ms: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        time_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        mask_parts: list[np.ndarray] = []
        for payload in payloads:
            timestamps = payload[f"{prefix}_timestamp_ms"]
            left = int(np.searchsorted(timestamps, start_ms, side="left"))
            right = int(np.searchsorted(timestamps, end_ms, side="right"))
            if right <= left:
                continue
            time_parts.append(timestamps[left:right])
            value_parts.append(payload[f"{prefix}_values"][left:right])
            mask_parts.append(payload[f"{prefix}_mask"][left:right])
        dimensions = 6 if prefix == "motion" else 1
        if not time_parts:
            return (
                np.empty(0, dtype=np.int64),
                np.empty((0, dimensions), dtype=np.float32),
                np.empty((0, dimensions), dtype=bool),
            )
        timestamps = np.concatenate(time_parts).astype(np.int64)
        values = np.concatenate(value_parts).astype(np.float32)
        masks = np.concatenate(mask_parts).astype(bool)
        order = np.argsort(timestamps, kind="stable")
        timestamps = timestamps[order]
        values = values[order]
        masks = masks[order]
        keep = np.concatenate(([True], np.diff(timestamps) > 0))
        return timestamps[keep], values[keep], masks[keep]

    def read(self, session_id: str, start_ms: int, end_ms: int) -> dict[str, np.ndarray]:
        if end_ms < start_ms:
            raise ValueError("Session window end must not precede start")
        selected = self.segments[
            (self.segments["session_id"] == str(session_id))
            & (self.segments["end_ms"] >= start_ms)
            & (self.segments["start_ms"] <= end_ms)
        ]
        payloads = [self._load(str(row.segment_path)) for row in selected.itertuples(index=False)]
        motion_time, motion_values, motion_mask = self._combine(
            payloads, "motion", start_ms, end_ms
        )
        ppg_time, ppg_values, ppg_mask = self._combine(payloads, "ppg", start_ms, end_ms)
        return {
            "motion_timestamp_ms": motion_time,
            "motion_values": motion_values,
            "motion_mask": motion_mask,
            "ppg_timestamp_ms": ppg_time,
            "ppg_values": ppg_values,
            "ppg_mask": ppg_mask,
        }
