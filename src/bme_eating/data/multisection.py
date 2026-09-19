from __future__ import annotations

import csv
import hashlib
import io
import math
import zipfile
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bme_eating.data.packet_reader import SENSOR_COLUMNS, _SENSOR_HEADER_BYTES


@dataclass(frozen=True)
class PacketFingerprint:
    sample_count: int
    digest: str


@dataclass
class PacketAccumulator:
    packets: dict[int, PacketFingerprint] = field(default_factory=dict)
    current_timestamp: int | None = None
    current_sample_count: int = 0
    current_digest: Any = None
    regressions: int = 0
    internal_identical_packets: int = 0
    internal_conflicting_packets: int = 0

    def add(self, timestamp: int, values: np.ndarray) -> None:
        if timestamp <= 0:
            return
        if self.current_timestamp is None:
            self._start(timestamp)
        elif timestamp != self.current_timestamp:
            previous_timestamp = self.current_timestamp
            self.flush()
            if timestamp < previous_timestamp:
                self.regressions += 1
            self._start(timestamp)
        self.current_digest.update(np.asarray(values, dtype="<f4").tobytes())
        self.current_sample_count += int(np.asarray(values).size)

    def _start(self, timestamp: int) -> None:
        self.current_timestamp = int(timestamp)
        self.current_sample_count = 0
        self.current_digest = hashlib.sha256()

    def flush(self) -> None:
        if self.current_timestamp is None:
            return
        fingerprint = PacketFingerprint(
            sample_count=self.current_sample_count,
            digest=self.current_digest.hexdigest(),
        )
        previous = self.packets.get(self.current_timestamp)
        if previous is None:
            self.packets[self.current_timestamp] = fingerprint
        elif previous == fingerprint:
            self.internal_identical_packets += 1
        else:
            self.internal_conflicting_packets += 1
        self.current_timestamp = None
        self.current_sample_count = 0
        self.current_digest = None


@dataclass
class SectionPackets:
    modalities: dict[str, PacketAccumulator] = field(
        default_factory=lambda: {
            "acc": PacketAccumulator(),
            "ppg": PacketAccumulator(),
            "gyro": PacketAccumulator(),
        }
    )

    def finish(self) -> None:
        for accumulator in self.modalities.values():
            accumulator.flush()


def _find_all_header_offsets(handle: zipfile.ZipExtFile) -> list[int]:
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
    return sorted(offsets)


def _finite_value(value: str, row_number: int, column: str) -> float:
    text = value.strip()
    if not text:
        return 0.0
    try:
        parsed = float(text)
    except ValueError as error:
        raise ValueError(f"row {row_number} has invalid {column}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"row {row_number} has non-finite {column}")
    return parsed


def _timestamp(value: str, row_number: int, column: str) -> int:
    return int(_finite_value(value, row_number, column))


def _parse_sections(
    archive: zipfile.ZipFile,
    member_name: str,
    header_offsets: list[int],
    ppg_samples_per_row: int,
) -> list[SectionPackets]:
    if len(header_offsets) < 2:
        raise ValueError("multisection audit requires at least two standard headers")
    sections: list[SectionPackets] = []
    raw_handle = archive.open(member_name)
    try:
        raw_handle.seek(header_offsets[0])
        with io.TextIOWrapper(
            raw_handle, encoding="utf-8", errors="strict", newline=""
        ) as text_handle:
            reader = csv.reader(text_handle, delimiter="\t")
            current: SectionPackets | None = None
            for row_number, row in enumerate(reader, start=1):
                stripped = [value.strip() for value in row]
                if stripped == SENSOR_COLUMNS:
                    if current is not None:
                        current.finish()
                        sections.append(current)
                    current = SectionPackets()
                    continue
                if current is None:
                    raise ValueError("text suffix does not start with the standard sensor header")
                if len(row) != len(SENSOR_COLUMNS):
                    raise ValueError(
                        f"row {row_number} has {len(row)} columns; "
                        f"expected {len(SENSOR_COLUMNS)}"
                    )
                timestamps = [
                    _timestamp(row[index], row_number, SENSOR_COLUMNS[index])
                    for index in range(3)
                ]
                values = np.asarray(
                    [
                        _finite_value(row[index], row_number, SENSOR_COLUMNS[index])
                        for index in range(3, len(SENSOR_COLUMNS))
                    ],
                    dtype=np.float32,
                )
                if not np.isfinite(values).all():
                    raise ValueError(f"row {row_number} overflows float32")
                current.modalities["acc"].add(timestamps[0], values[44:47])
                current.modalities["ppg"].add(
                    timestamps[1], values[:ppg_samples_per_row]
                )
                current.modalities["gyro"].add(timestamps[2], values[47:50])
            if current is not None:
                current.finish()
                sections.append(current)
    except UnicodeDecodeError as error:
        raise ValueError("text sections are not strict UTF-8") from error
    finally:
        raw_handle.close()
    if len(sections) != len(header_offsets):
        raise ValueError(
            f"located {len(header_offsets)} headers but parsed {len(sections)} sections"
        )
    return sections


def _compare_packet_maps(
    left: dict[int, PacketFingerprint], right: dict[int, PacketFingerprint]
) -> dict[str, int | float | str]:
    if not left and not right:
        return {
            "relationship": "both_missing",
            "overlap_duration_seconds": 0.0,
            "shared_packets": 0,
            "identical_packets": 0,
            "conflicting_packets": 0,
            "left_only_packets": 0,
            "right_only_packets": 0,
        }
    if not left or not right:
        return {
            "relationship": "one_missing",
            "overlap_duration_seconds": 0.0,
            "shared_packets": 0,
            "identical_packets": 0,
            "conflicting_packets": 0,
            "left_only_packets": 0,
            "right_only_packets": 0,
        }
    overlap_start = max(min(left), min(right))
    overlap_end = min(max(left), max(right))
    if overlap_end < overlap_start:
        return {
            "relationship": "disjoint",
            "overlap_duration_seconds": 0.0,
            "shared_packets": 0,
            "identical_packets": 0,
            "conflicting_packets": 0,
            "left_only_packets": 0,
            "right_only_packets": 0,
        }
    left_overlap = {timestamp for timestamp in left if overlap_start <= timestamp <= overlap_end}
    right_overlap = {timestamp for timestamp in right if overlap_start <= timestamp <= overlap_end}
    shared = left_overlap & right_overlap
    identical = sum(left[timestamp] == right[timestamp] for timestamp in shared)
    conflicts = len(shared) - identical
    relationship = "exact_duplicate_overlap"
    if conflicts:
        relationship = "conflicting_overlap"
    elif left_overlap != right_overlap:
        relationship = "partial_overlap"
    return {
        "relationship": relationship,
        "overlap_duration_seconds": (overlap_end - overlap_start) / 1000.0,
        "shared_packets": len(shared),
        "identical_packets": identical,
        "conflicting_packets": conflicts,
        "left_only_packets": len(left_overlap - right_overlap),
        "right_only_packets": len(right_overlap - left_overlap),
    }


def audit_multisection_zip(
    zip_path: str | Path,
    source_zip_sha256: str,
    ppg_samples_per_row: int,
) -> dict[str, Any]:
    zip_path = Path(zip_path)
    if not 1 <= ppg_samples_per_row <= 44:
        raise ValueError("ppg_samples_per_row must be between 1 and 44")
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if len(members) != 1:
            raise ValueError(f"expected one sensor text member, found {len(members)}")
        member_name = members[0]
        with archive.open(member_name) as handle:
            header_offsets = _find_all_header_offsets(handle)
        sections = _parse_sections(
            archive, member_name, header_offsets, ppg_samples_per_row
        )

    section_summaries: list[dict[str, Any]] = []
    total_regressions = 0
    total_internal_conflicts = 0
    for section_index, section in enumerate(sections):
        modalities: dict[str, dict[str, int | float]] = {}
        for modality, accumulator in section.modalities.items():
            timestamps = accumulator.packets.keys()
            duration = (max(timestamps) - min(timestamps)) / 1000.0 if timestamps else 0.0
            modalities[modality] = {
                "packets": len(accumulator.packets),
                "duration_seconds": duration,
                "timestamp_regressions": accumulator.regressions,
                "internal_identical_packets": accumulator.internal_identical_packets,
                "internal_conflicting_packets": accumulator.internal_conflicting_packets,
            }
            total_regressions += accumulator.regressions
            total_internal_conflicts += accumulator.internal_conflicting_packets
        section_summaries.append(
            {"section_index": section_index, "modalities": modalities}
        )

    pair_summaries: list[dict[str, Any]] = []
    relationship_counts: dict[str, int] = {}
    total_conflicting_packets = 0
    for left_index, right_index in combinations(range(len(sections)), 2):
        for modality in ("acc", "ppg", "gyro"):
            comparison = _compare_packet_maps(
                sections[left_index].modalities[modality].packets,
                sections[right_index].modalities[modality].packets,
            )
            relationship = str(comparison["relationship"])
            relationship_counts[relationship] = relationship_counts.get(relationship, 0) + 1
            total_conflicting_packets += int(comparison["conflicting_packets"])
            pair_summaries.append(
                {
                    "left_section": left_index,
                    "right_section": right_index,
                    "modality": modality,
                    **comparison,
                }
            )

    if total_internal_conflicts or total_conflicting_packets:
        classification = "conflicting_overlap"
    elif total_regressions:
        classification = "nonmonotonic_section"
    elif relationship_counts.get("partial_overlap", 0):
        classification = "partial_overlap"
    elif set(relationship_counts) <= {"exact_duplicate_overlap", "both_missing"}:
        classification = "exact_duplicate_overlap"
    elif set(relationship_counts) <= {"disjoint", "both_missing"}:
        classification = "disjoint_sections"
    else:
        classification = "mixed_relationships"
    return {
        "source_zip_sha256": str(source_zip_sha256),
        "classification": classification,
        "header_count": len(header_offsets),
        "binary_prefix_bytes": header_offsets[0],
        "left_censored": header_offsets[0] > 0,
        "timestamp_regressions": total_regressions,
        "internal_conflicting_packets": total_internal_conflicts,
        "cross_section_conflicting_packets": total_conflicting_packets,
        "relationship_counts": relationship_counts,
        "sections": section_summaries,
        "section_pairs": pair_summaries,
    }


def audit_repeated_header_attachments(
    records: pd.DataFrame,
    schema_audit: dict[str, Any],
    ppg_samples_per_row: int,
    show_progress: bool = False,
) -> dict[str, Any]:
    layouts = schema_audit.get("layouts", [])
    repeated_names = {
        str(layout.get("zip_name", ""))
        for layout in layouts
        if layout.get("status") == "unsupported_binary"
        and layout.get("error") == "standard sensor header occurs more than once"
    }
    if not repeated_names:
        raise ValueError("schema audit contains no repeated-header attachments")
    required = {"zip_path", "zip_sha256"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"records index is missing columns: {sorted(missing)}")
    indexed: dict[str, list[pd.Series]] = {}
    for _, record in records.iterrows():
        indexed.setdefault(Path(str(record["zip_path"])).name, []).append(record)
    selected: list[pd.Series] = []
    for name in repeated_names:
        matches = indexed.get(name, [])
        if len(matches) != 1:
            raise ValueError(
                "each repeated-header schema entry must map to exactly one indexed attachment"
            )
        selected.append(matches[0])
    selected.sort(key=lambda row: str(row["zip_sha256"]))

    results: list[dict[str, Any]] = []
    iterator: Any = enumerate(selected, start=1)
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(
            iterator,
            total=len(selected),
            desc="Auditing repeated-header attachments",
            unit="attachment",
        )
    for attachment_index, record in iterator:
        try:
            result = audit_multisection_zip(
                record["zip_path"],
                str(record["zip_sha256"]),
                ppg_samples_per_row,
            )
        except (OSError, ValueError, zipfile.BadZipFile, csv.Error) as error:
            message = str(error) if isinstance(error, ValueError) else "attachment could not be read"
            result = {
                "source_zip_sha256": str(record["zip_sha256"]),
                "classification": "invalid",
                "error_type": type(error).__name__,
                "error": message,
            }
        result["attachment_index"] = attachment_index
        results.append(result)

    classification_counts: dict[str, int] = {}
    for result in results:
        classification = str(result["classification"])
        classification_counts[classification] = classification_counts.get(classification, 0) + 1
    return {
        "version": 1,
        "attachments_audited": len(results),
        "classification_counts": dict(sorted(classification_counts.items())),
        "automatic_recovery_performed": False,
        "results": results,
    }
