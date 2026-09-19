from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from bme_eating.privacy import load_or_create_subject_salt, normalize_subject_id, subject_key


def _parse_hand(value: object) -> str:
    text = "" if value is None else str(value)
    if "左" in text or text.strip().lower() == "left":
        return "left"
    if "右" in text or text.strip().lower() == "right":
        return "right"
    return "unknown"


def _load_download_state(data_root: Path) -> dict[str, Any]:
    state_path = data_root / "manifests" / "download_state.json"
    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict) or not isinstance(state.get("attachments", []), list):
        raise ValueError(f"Invalid download state format: {state_path}")
    return state


def _resolve_attachment_path(data_root: Path, relative_path: str | Path) -> Path:
    candidate = data_root / Path(relative_path)
    if candidate.exists():
        return candidate
    attachment_root = data_root / "raw" / "sensor_attachments"
    filename = Path(relative_path).name
    matches = list(attachment_root.rglob(filename)) if attachment_root.exists() else []
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Downloaded attachment is missing: {candidate}; "
            f"no fallback match for {filename} under {attachment_root}"
        )
    raise RuntimeError(
        f"Downloaded attachment is ambiguous: {filename}; "
        f"found {len(matches)} matches under {attachment_root}"
    )


def build_secure_indices(
    data_root: Path,
    output_root: Path,
    invalid_subjects: set[str],
    subject_pattern: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = _load_download_state(data_root)
    sensor_metadata = pd.read_csv(data_root / "derived" / "valid_metadata" / "sensor.csv")
    meal_metadata = pd.read_csv(data_root / "derived" / "valid_metadata" / "meal.csv")
    salt = load_or_create_subject_salt(output_root)
    formal_pattern = re.compile(subject_pattern)

    sensor_metadata["normalized_subject"] = sensor_metadata["externalid"].map(
        normalize_subject_id
    )
    meal_metadata["normalized_subject"] = meal_metadata["externalid"].map(normalize_subject_id)

    valid_subjects = {
        value
        for value in set(sensor_metadata["normalized_subject"]) & set(meal_metadata["normalized_subject"])
        if formal_pattern.fullmatch(value) and value not in invalid_subjects
    }

    attachment_rows = []
    for item in state["attachments"]:
        normalized = normalize_subject_id(item.get("externalid"))
        if normalized not in valid_subjects:
            continue
        if item.get("status") != "downloaded" or item.get("invalid_externalid"):
            continue
        relative_path = Path(item["relative_path"])
        absolute_path = _resolve_attachment_path(data_root, relative_path)
        attachment_rows.append(
            {
                "subject_key": subject_key(normalized, salt),
                "uniqueid": str(item.get("uniqueid", "")),
                "object_path": str(item.get("object_path", "")),
                "zip_path": str(absolute_path),
                "zip_sha256": str(item.get("sha256", "")),
                "zip_size_bytes": int(item.get("size_bytes") or absolute_path.stat().st_size),
            }
        )
    record_columns = [
        "subject_key",
        "uniqueid",
        "object_path",
        "zip_path",
        "zip_sha256",
        "zip_size_bytes",
    ]
    records = pd.DataFrame(attachment_rows, columns=record_columns).drop_duplicates("zip_path")

    meals = meal_metadata[meal_metadata["normalized_subject"].isin(valid_subjects)].copy()
    meals["subject_key"] = meals["normalized_subject"].map(
        lambda value: subject_key(value, salt)
    )
    meals["start_ms"] = pd.to_numeric(meals["beforeTime"], errors="coerce")
    meals["end_ms"] = pd.to_numeric(meals["afterTime"], errors="coerce")
    meals["dietary_hand"] = meals["dietaryHand"].map(_parse_hand)
    meals["wear_hand"] = meals["wearHand"].map(_parse_hand)
    meals["hand_relation"] = meals.apply(
        lambda row: (
            "same"
            if row["dietary_hand"] == row["wear_hand"] != "unknown"
            else "different"
            if "unknown" not in (row["dietary_hand"], row["wear_hand"])
            else "unknown"
        ),
        axis=1,
    )
    meals["event_id"] = meals["uniqueid"].astype(str)
    events = meals[
        [
            "event_id",
            "subject_key",
            "start_ms",
            "end_ms",
            "dietary_hand",
            "wear_hand",
            "hand_relation",
        ]
    ].copy()
    events["valid_duration"] = (
        events["start_ms"].notna()
        & events["end_ms"].notna()
        & (events["end_ms"] > events["start_ms"])
    )

    secure_dir = output_root / "indices"
    secure_dir.mkdir(parents=True, exist_ok=True)
    records.to_parquet(secure_dir / "records.parquet", index=False)
    events.to_parquet(secure_dir / "events.parquet", index=False)
    return records, events

