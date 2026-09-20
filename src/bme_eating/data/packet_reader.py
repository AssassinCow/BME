from __future__ import annotations

import csv
import io
import json
import math
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bme_eating.constants import ACC_COLUMNS, GYRO_COLUMNS, TIME_COLUMNS
from bme_eating.types import SensorSeries

SENSOR_COLUMNS = (
    list(TIME_COLUMNS)
    + [f"PPG{number}" for number in range(1, 45)]
    + list(ACC_COLUMNS)
    + list(GYRO_COLUMNS)
)
_SENSOR_HEADER_BYTES = "\t".join(SENSOR_COLUMNS).encode("utf-8")
_UTF8_BOM = b"\xef\xbb\xbf"


@dataclass(frozen=True)
class ParsedAttachment:
    acc: SensorSeries
    gyro: SensorSeries
    ppg: SensorSeries
    source_name: str
    info: dict[str, object]
    parser_status: str = "documented_text"
    text_offset_bytes: int = 0
    left_censored: bool = False


class UnsupportedSensorFormatError(ValueError):
    """Raised when a sensor member is not the documented tab-separated text format."""

    def __init__(self, zip_path: Path, member_name: str, reason: str) -> None:
        self.zip_path = zip_path
        self.member_name = member_name
        self.reason = reason
        super().__init__(f"Unsupported sensor format in {zip_path}!{member_name}: {reason}")

    def __reduce__(self):
        return type(self), (self.zip_path, self.member_name, self.reason)


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
        parsed = float(value)
        return parsed if math.isfinite(parsed) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _int_or_zero(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _parse_sensor_value(value: str, row_number: int, column: str) -> float:
    try:
        parsed = float(value.strip())
    except (AttributeError, ValueError) as error:
        raise ValueError(f"row {row_number} has invalid {column}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"row {row_number} has non-finite {column}")
    return parsed


def _parse_timestamp(value: str, row_number: int, column: str) -> int:
    text = value.strip()
    if not text:
        return 0
    try:
        parsed = float(text)
    except ValueError as error:
        raise ValueError(f"row {row_number} has invalid {column}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"row {row_number} has non-finite {column}")
    return int(parsed)


def _find_sensor_header_offsets(
    handle: zipfile.ZipExtFile, stop_after: int | None = 2
) -> list[int]:
    offsets: set[int] = set()
    overlap = len(_SENSOR_HEADER_BYTES) + 2
    tail = b""
    consumed = 0
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        buffer = tail + chunk
        base = consumed - len(tail)
        start = 0
        while True:
            position = buffer.find(_SENSOR_HEADER_BYTES, start)
            if position < 0:
                break
            after = position + len(_SENSOR_HEADER_BYTES)
            if after < len(buffer) and buffer[after : after + 1] in {b"\r", b"\n"}:
                offsets.add(base + position)
            start = position + 1
        consumed += len(chunk)
        tail = buffer[-overlap:]
        if stop_after is not None and len(offsets) >= stop_after:
            break
    return sorted(offsets)


def _locate_sensor_text(
    archive: zipfile.ZipFile,
    zip_path: Path,
    member_name: str,
    required_columns: set[str],
) -> tuple[str, int, list[str]]:
    with archive.open(member_name) as handle:
        prefix = handle.read(len(_UTF8_BOM) + len(_SENSOR_HEADER_BYTES) + 2)
        if prefix.startswith(_UTF8_BOM + _SENSOR_HEADER_BYTES):
            offset = len(_UTF8_BOM)
            status = "documented_text"
        elif prefix.startswith(_SENSOR_HEADER_BYTES):
            offset = 0
            status = "documented_text"
        else:
            handle.seek(0)
            offsets = _find_sensor_header_offsets(handle)
            if len(offsets) != 1:
                reason = "standard sensor header was not found"
                if len(offsets) > 1:
                    reason = "standard sensor header occurs more than once"
                raise UnsupportedSensorFormatError(zip_path, member_name, reason)
            offset = offsets[0]
            status = "recovered_text_suffix"
    columns = SENSOR_COLUMNS.copy()
    missing = required_columns - set(columns)
    if missing:
        raise UnsupportedSensorFormatError(
            zip_path,
            member_name,
            f"standard sensor header is missing required columns: {sorted(missing)}",
        )
    return status, offset, columns


def _sensor_rows(
    archive: zipfile.ZipFile,
    zip_path: Path,
    member_name: str,
    text_offset: int,
):
    raw_handle = archive.open(member_name)
    try:
        raw_handle.seek(text_offset)
        text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8", newline="")
        reader = csv.reader(text_handle, delimiter="\t")
        header = next(reader)
        if [value.strip() for value in header] != SENSOR_COLUMNS:
            raise UnsupportedSensorFormatError(
                zip_path, member_name, "sensor header does not match the 53-column schema"
            )
        yield header, reader
    except (UnicodeDecodeError, csv.Error) as error:
        raise UnsupportedSensorFormatError(
            zip_path, member_name, "sensor text suffix is not valid UTF-8 TSV"
        ) from error
    finally:
        raw_handle.close()


def parse_sensor_zip(
    zip_path: str | Path,
    ppg_samples_per_row: int = 20,
    timestamp_anchor: str = "start",
) -> ParsedAttachment:
    zip_path = Path(zip_path)
    if not 1 <= ppg_samples_per_row <= 44:
        raise ValueError("ppg_samples_per_row must be between 1 and 44")
    with zipfile.ZipFile(zip_path) as archive:
        text_entries = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if len(text_entries) != 1:
            raise ValueError(f"Expected one sensor text file in {zip_path}, found {text_entries}")
        info_entries = [name for name in archive.namelist() if name.lower().endswith("info.json")]
        info: dict[str, object] = {}
        if info_entries:
            with archive.open(info_entries[0]) as handle:
                info = json.load(handle)

        parser_status, text_offset, _ = _locate_sensor_text(
            archive,
            zip_path,
            text_entries[0],
            set(TIME_COLUMNS) | set(ACC_COLUMNS) | set(GYRO_COLUMNS),
        )

        acc_expander = _PacketExpander(3, timestamp_anchor)
        gyro_expander = _PacketExpander(3, timestamp_anchor)
        ppg_expander = _PacketExpander(1, timestamp_anchor)
        for header, reader in _sensor_rows(
            archive, zip_path, text_entries[0], text_offset
        ):
                index = {name.strip(): position for position, name in enumerate(header)}
                required = set(TIME_COLUMNS) | set(ACC_COLUMNS) | set(GYRO_COLUMNS)
                missing = required - set(index)
                if missing:
                    raise ValueError(f"Missing required columns in {zip_path}: {sorted(missing)}")
                ppg_columns = [f"PPG{number}" for number in range(1, 45)]
                missing_ppg = [name for name in ppg_columns if name not in index]
                if missing_ppg:
                    raise ValueError(f"Missing configured PPG columns: {missing_ppg}")

                for row_number, row in enumerate(reader, start=2):
                    if [value.strip() for value in row] == SENSOR_COLUMNS:
                        raise UnsupportedSensorFormatError(
                            zip_path, text_entries[0], "standard sensor header occurs more than once"
                        )
                    if len(row) != len(header):
                        raise UnsupportedSensorFormatError(
                            zip_path,
                            text_entries[0],
                            f"row {row_number} has {len(row)} columns; expected {len(header)}",
                        )
                    acc_expander.add(
                        _parse_timestamp(row[index["ACC_TIME"]], row_number, "ACC_TIME"),
                        np.asarray(
                            [
                                _parse_sensor_value(row[index[name]], row_number, name)
                                for name in ACC_COLUMNS
                            ]
                        ),
                    )
                    gyro_expander.add(
                        _parse_timestamp(row[index["GYRO_TIME"]], row_number, "GYRO_TIME"),
                        np.asarray(
                            [
                                _parse_sensor_value(row[index[name]], row_number, name)
                                for name in GYRO_COLUMNS
                            ]
                        ),
                    )
                    ppg_timestamp = _parse_timestamp(
                        row[index["PPG_TIME"]], row_number, "PPG_TIME"
                    )
                    if ppg_timestamp > 0:
                        all_ppg_values = np.asarray(
                            [
                                _parse_sensor_value(row[index[name]], row_number, name)
                                for name in ppg_columns
                            ],
                            dtype=np.float32,
                        )
                        ppg_values = all_ppg_values[:ppg_samples_per_row].reshape(-1, 1)
                        ppg_expander.add(ppg_timestamp, ppg_values)

    return ParsedAttachment(
        acc=acc_expander.finish(),
        gyro=gyro_expander.finish(),
        ppg=ppg_expander.finish(),
        source_name=text_entries[0],
        info=info,
        parser_status=parser_status,
        text_offset_bytes=text_offset if parser_status == "recovered_text_suffix" else 0,
        left_censored=parser_status == "recovered_text_suffix",
    )


class _PacketCollector:
    def __init__(
        self,
        dimensions: int,
        zip_path: Path,
        member_name: str,
        modality: str,
    ) -> None:
        self.dimensions = dimensions
        self.zip_path = zip_path
        self.member_name = member_name
        self.modality = modality
        self.current_timestamp: int | None = None
        self.current_values: list[np.ndarray] = []
        self.packets: dict[int, np.ndarray] = {}

    def add(self, timestamp: int, values: np.ndarray) -> None:
        if timestamp <= 0:
            return
        values = np.asarray(values, dtype=np.float32).reshape(-1, self.dimensions)
        if self.current_timestamp is None:
            self.current_timestamp = timestamp
        elif timestamp < self.current_timestamp:
            raise UnsupportedSensorFormatError(
                self.zip_path,
                self.member_name,
                f"multisection {self.modality} packet timestamps are nonmonotonic",
            )
        elif timestamp != self.current_timestamp:
            self._flush()
            self.current_timestamp = timestamp
        self.current_values.append(values)

    def _flush(self) -> None:
        if self.current_timestamp is None or not self.current_values:
            self.current_values = []
            return
        packet = np.concatenate(self.current_values, axis=0)
        previous = self.packets.get(self.current_timestamp)
        if previous is not None and not np.array_equal(previous, packet):
            raise UnsupportedSensorFormatError(
                self.zip_path,
                self.member_name,
                f"multisection {self.modality} samples conflict at a packet timestamp",
            )
        self.packets[self.current_timestamp] = packet
        self.current_timestamp = None
        self.current_values = []

    def finish(self) -> dict[int, np.ndarray]:
        self._flush()
        return self.packets


def _merge_exact_packets(
    packet_maps: list[dict[int, np.ndarray]],
    zip_path: Path,
    member_name: str,
    modality: str,
    dimensions: int,
    timestamp_anchor: str,
) -> SensorSeries:
    merged: dict[int, np.ndarray] = {}
    for packets in packet_maps:
        for timestamp, values in packets.items():
            previous = merged.get(timestamp)
            if previous is not None and not np.array_equal(previous, values):
                raise UnsupportedSensorFormatError(
                    zip_path,
                    member_name,
                    f"multisection {modality} samples conflict at a packet timestamp",
                )
            merged[timestamp] = values
    expander = _PacketExpander(dimensions, timestamp_anchor)
    for timestamp in sorted(merged):
        expander.add(timestamp, merged[timestamp])
    return expander.finish()


def parse_multisection_sensor_zip(
    zip_path: str | Path,
    ppg_samples_per_row: int = 20,
    timestamp_anchor: str = "start",
) -> ParsedAttachment:
    """Strictly parse repeated-header text sections and deduplicate exact samples."""
    zip_path = Path(zip_path)
    if not 1 <= ppg_samples_per_row <= 44:
        raise ValueError("ppg_samples_per_row must be between 1 and 44")
    with zipfile.ZipFile(zip_path) as archive:
        text_entries = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if len(text_entries) != 1:
            raise ValueError(f"Expected one sensor text file in {zip_path}, found {text_entries}")
        member_name = text_entries[0]
        info_entries = [name for name in archive.namelist() if name.lower().endswith("info.json")]
        info: dict[str, object] = {}
        if info_entries:
            with archive.open(info_entries[0]) as handle:
                info = json.load(handle)
        with archive.open(member_name) as handle:
            header_offsets = _find_sensor_header_offsets(handle, stop_after=None)
        if len(header_offsets) < 2:
            raise UnsupportedSensorFormatError(
                zip_path, member_name, "multisection recovery requires repeated standard headers"
            )

        raw_handle = archive.open(member_name)
        section_packets: list[
            tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]
        ] = []
        try:
            raw_handle.seek(header_offsets[0])
            text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8", errors="strict", newline="")
            reader = csv.reader(text_handle, delimiter="\t")
            collectors: tuple[_PacketCollector, _PacketCollector, _PacketCollector] | None = None
            section_count = 0
            for row_number, row in enumerate(reader, start=1):
                if [value.strip() for value in row] == SENSOR_COLUMNS:
                    if collectors is not None:
                        section_packets.append(
                            tuple(collector.finish() for collector in collectors)
                        )
                    collectors = (
                        _PacketCollector(3, zip_path, member_name, "acc"),
                        _PacketCollector(3, zip_path, member_name, "gyro"),
                        _PacketCollector(1, zip_path, member_name, "ppg"),
                    )
                    section_count += 1
                    continue
                if collectors is None:
                    raise UnsupportedSensorFormatError(
                        zip_path, member_name, "text suffix does not start with the standard header"
                    )
                if len(row) != len(SENSOR_COLUMNS):
                    raise UnsupportedSensorFormatError(
                        zip_path,
                        member_name,
                        f"row {row_number} has {len(row)} columns; expected {len(SENSOR_COLUMNS)}",
                    )
                acc_collector, gyro_collector, ppg_collector = collectors
                acc_collector.add(
                    _parse_timestamp(row[0], row_number, "ACC_TIME"),
                    np.asarray(
                        [_parse_sensor_value(row[index], row_number, SENSOR_COLUMNS[index])
                         for index in range(47, 50)],
                        dtype=np.float32,
                    ),
                )
                gyro_collector.add(
                    _parse_timestamp(row[2], row_number, "GYRO_TIME"),
                    np.asarray(
                        [_parse_sensor_value(row[index], row_number, SENSOR_COLUMNS[index])
                         for index in range(50, 53)],
                        dtype=np.float32,
                    ),
                )
                ppg_timestamp = _parse_timestamp(row[1], row_number, "PPG_TIME")
                if ppg_timestamp > 0:
                    ppg_collector.add(
                        ppg_timestamp,
                        np.asarray(
                            [_parse_sensor_value(row[index], row_number, SENSOR_COLUMNS[index])
                             for index in range(3, 3 + ppg_samples_per_row)],
                            dtype=np.float32,
                        ).reshape(-1, 1),
                    )
            if collectors is not None:
                section_packets.append(
                    tuple(collector.finish() for collector in collectors)
                )
        except (UnicodeDecodeError, csv.Error) as error:
            raise UnsupportedSensorFormatError(
                zip_path, member_name, "multisection text is not valid UTF-8 TSV"
            ) from error
        finally:
            raw_handle.close()
        if section_count != len(header_offsets) or len(section_packets) != len(header_offsets):
            raise UnsupportedSensorFormatError(
                zip_path, member_name, "not every repeated-header section was parsed"
            )

    return ParsedAttachment(
        acc=_merge_exact_packets(
            [item[0] for item in section_packets],
            zip_path,
            member_name,
            "acc",
            3,
            timestamp_anchor,
        ),
        gyro=_merge_exact_packets(
            [item[1] for item in section_packets],
            zip_path,
            member_name,
            "gyro",
            3,
            timestamp_anchor,
        ),
        ppg=_merge_exact_packets(
            [item[2] for item in section_packets],
            zip_path,
            member_name,
            "ppg",
            1,
            timestamp_anchor,
        ),
        source_name=member_name,
        info=info,
        parser_status="recovered_multisection_deduplicated",
        text_offset_bytes=header_offsets[0],
        left_censored=header_offsets[0] > 0,
    )


def inspect_ppg_layout(zip_path: str | Path, maximum_rows: int = 100_000) -> dict[str, object]:
    zip_path = Path(zip_path)
    nonzero_counts = np.zeros(44, dtype=np.int64)
    ppg_rows = 0
    with zipfile.ZipFile(zip_path) as archive:
        text_name = next(name for name in archive.namelist() if name.lower().endswith(".txt"))
        try:
            parser_status, text_offset, _ = _locate_sensor_text(
                archive,
                zip_path,
                text_name,
                set(TIME_COLUMNS)
                | set(ACC_COLUMNS)
                | set(GYRO_COLUMNS)
                | {f"PPG{number}" for number in range(1, 45)},
            )
        except UnsupportedSensorFormatError as error:
            status = (
                "repeated_header"
                if error.reason == "standard sensor header occurs more than once"
                else "unsupported_binary"
            )
            return {
                "zip_name": zip_path.name,
                "status": status,
                "text_member": text_name,
                "error": error.reason,
                "rows_scanned": 0,
                "ppg_rows": 0,
                "nonzero_fraction_by_slot": [0.0] * 44,
            }

        for header, reader in _sensor_rows(archive, zip_path, text_name, text_offset):
                index = {name.strip(): position for position, name in enumerate(header)}
                for row_number, row in enumerate(reader):
                    if row_number >= maximum_rows:
                        break
                    if [value.strip() for value in row] == SENSOR_COLUMNS:
                        return {
                            "zip_name": zip_path.name,
                            "status": "repeated_header",
                            "text_member": text_name,
                            "error": "standard sensor header occurs more than once",
                            "text_offset_bytes": text_offset,
                            "rows_scanned": row_number + 1,
                            "ppg_rows": ppg_rows,
                            "nonzero_fraction_by_slot": (
                                nonzero_counts / max(ppg_rows, 1)
                            ).round(6).tolist(),
                        }
                    if len(row) != len(header):
                        raise UnsupportedSensorFormatError(
                            zip_path,
                            text_name,
                            f"row {row_number + 2} has {len(row)} columns; expected {len(header)}",
                        )
                    if _int_or_zero(row[index["PPG_TIME"]]) <= 0:
                        continue
                    ppg_rows += 1
                    for slot in range(44):
                        value = _float_or_zero(row[index[f"PPG{slot + 1}"]])
                        nonzero_counts[slot] += int(value != 0.0)
    return {
        "zip_name": zip_path.name,
        "status": parser_status,
        "text_offset_bytes": text_offset if parser_status == "recovered_text_suffix" else 0,
        "rows_scanned": min(maximum_rows, row_number + 1 if "row_number" in locals() else 0),
        "ppg_rows": ppg_rows,
        "nonzero_fraction_by_slot": (
            nonzero_counts / max(ppg_rows, 1)
        ).round(6).tolist(),
    }

