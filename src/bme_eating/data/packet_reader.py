from __future__ import annotations

import csv
import json
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bme_eating.constants import ACC_COLUMNS, GYRO_COLUMNS, TIME_COLUMNS
from bme_eating.types import SensorSeries


@dataclass(frozen=True)
class ParsedAttachment:
    acc: SensorSeries
    gyro: SensorSeries
    ppg: SensorSeries
    source_name: str
    info: dict[str, object]


class _PacketExpander:
    def __init__(self, dimensions: int, timestamp_anchor: str) -> None:
        if timestamp_anchor not in {"start", "end"}:
            raise ValueError("packet timestamp anchor must be 'start' or 'end'")
        self.dimensions = dimensions
        self.timestamp_anchor = timestamp_anchor
        self.current_timestamp: int | None = None
        self.current_values: list[np.ndarray] = []
        self.packet_deltas: list[int] = []
        self.timestamps: list[np.ndarray] = []
        self.values: list[np.ndarray] = []

    def add(self, timestamp: int, values: np.ndarray) -> None:
        if timestamp <= 0:
            return
        values = np.asarray(values, dtype=np.float32).reshape(-1, self.dimensions)
        if self.current_timestamp is None:
            self.current_timestamp = timestamp
        if timestamp != self.current_timestamp:
            next_timestamp = timestamp if timestamp > self.current_timestamp else None
            self._flush(next_timestamp)
            self.current_timestamp = timestamp
        self.current_values.append(values)

    def _fallback_delta(self, sample_count: int) -> float:
        if self.packet_deltas:
            return float(statistics.median(self.packet_deltas[-200:]))
        return float(max(sample_count, 1))

    def _flush(self, next_timestamp: int | None) -> None:
        if self.current_timestamp is None or not self.current_values:
            self.current_values = []
            return
        packet = np.concatenate(self.current_values, axis=0)
        count = len(packet)
        if next_timestamp is not None:
            interval = next_timestamp - self.current_timestamp
            if interval > 0:
                self.packet_deltas.append(interval)
            else:
                interval = self._fallback_delta(count)
        else:
            interval = self._fallback_delta(count)
        offsets = np.arange(count, dtype=np.float64) * (float(interval) / max(count, 1))
        if self.timestamp_anchor == "start":
            timestamps = self.current_timestamp + offsets
        else:
            timestamps = self.current_timestamp - float(interval) + offsets
        self.timestamps.append(np.rint(timestamps).astype(np.int64))
        self.values.append(packet)
        self.current_values = []

    def finish(self) -> SensorSeries:
        self._flush(None)
        if not self.values:
            return SensorSeries(
                timestamp_ms=np.empty(0, dtype=np.int64),
                values=np.empty((0, self.dimensions), dtype=np.float32),
            )
        return SensorSeries(
            timestamp_ms=np.concatenate(self.timestamps),
            values=np.concatenate(self.values, axis=0),
        )


def _float_or_zero(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_or_zero(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def parse_sensor_zip(
    zip_path: str | Path,
    ppg_samples_per_row: int = 20,
    timestamp_anchor: str = "start",
) -> ParsedAttachment:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        text_entries = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if len(text_entries) != 1:
            raise ValueError(f"Expected one sensor text file in {zip_path}, found {text_entries}")
        info_entries = [name for name in archive.namelist() if name.lower().endswith("info.json")]
        info: dict[str, object] = {}
        if info_entries:
            with archive.open(info_entries[0]) as handle:
                info = json.load(handle)

        acc_expander = _PacketExpander(3, timestamp_anchor)
        gyro_expander = _PacketExpander(3, timestamp_anchor)
        ppg_expander = _PacketExpander(1, timestamp_anchor)
        with archive.open(text_entries[0]) as raw_handle:
            import io

            with io.TextIOWrapper(raw_handle, encoding="utf-8-sig", newline="") as text_handle:
                reader = csv.reader(text_handle, delimiter="\t")
                header = next(reader)
                index = {name.strip(): position for position, name in enumerate(header)}
                required = set(TIME_COLUMNS) | set(ACC_COLUMNS) | set(GYRO_COLUMNS)
                missing = required - set(index)
                if missing:
                    raise ValueError(f"Missing required columns in {zip_path}: {sorted(missing)}")
                ppg_columns = [f"PPG{number}" for number in range(1, ppg_samples_per_row + 1)]
                missing_ppg = [name for name in ppg_columns if name not in index]
                if missing_ppg:
                    raise ValueError(f"Missing configured PPG columns: {missing_ppg}")

                for row in reader:
                    if len(row) != len(header):
                        continue
                    acc_expander.add(
                        _int_or_zero(row[index["ACC_TIME"]]),
                        np.asarray([_float_or_zero(row[index[name]]) for name in ACC_COLUMNS]),
                    )
                    gyro_expander.add(
                        _int_or_zero(row[index["GYRO_TIME"]]),
                        np.asarray([_float_or_zero(row[index[name]]) for name in GYRO_COLUMNS]),
                    )
                    ppg_timestamp = _int_or_zero(row[index["PPG_TIME"]])
                    if ppg_timestamp > 0:
                        ppg_values = np.asarray(
                            [_float_or_zero(row[index[name]]) for name in ppg_columns],
                            dtype=np.float32,
                        ).reshape(-1, 1)
                        ppg_expander.add(ppg_timestamp, ppg_values)

    return ParsedAttachment(
        acc=acc_expander.finish(),
        gyro=gyro_expander.finish(),
        ppg=ppg_expander.finish(),
        source_name=text_entries[0],
        info=info,
    )


def inspect_ppg_layout(zip_path: str | Path, maximum_rows: int = 100_000) -> dict[str, object]:
    zip_path = Path(zip_path)
    nonzero_counts = np.zeros(44, dtype=np.int64)
    ppg_rows = 0
    with zipfile.ZipFile(zip_path) as archive:
        text_name = next(name for name in archive.namelist() if name.lower().endswith(".txt"))
        import io

        with archive.open(text_name) as raw_handle:
            with io.TextIOWrapper(raw_handle, encoding="utf-8-sig", newline="") as text_handle:
                reader = csv.reader(text_handle, delimiter="\t")
                header = next(reader)
                index = {name.strip(): position for position, name in enumerate(header)}
                for row_number, row in enumerate(reader):
                    if row_number >= maximum_rows:
                        break
                    if _int_or_zero(row[index["PPG_TIME"]]) <= 0:
                        continue
                    ppg_rows += 1
                    for slot in range(44):
                        value = _float_or_zero(row[index[f"PPG{slot + 1}"]])
                        nonzero_counts[slot] += int(value != 0.0)
    return {
        "zip_name": zip_path.name,
        "rows_scanned": min(maximum_rows, row_number + 1 if "row_number" in locals() else 0),
        "ppg_rows": ppg_rows,
        "nonzero_fraction_by_slot": (
            nonzero_counts / max(ppg_rows, 1)
        ).round(6).tolist(),
    }

