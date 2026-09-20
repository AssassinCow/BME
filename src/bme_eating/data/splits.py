from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def create_subject_folds(
    events: pd.DataFrame,
    number_of_folds: int,
    seed: int,
    output_path: Path,
    subject_keys: set[str] | None = None,
) -> dict[str, int]:
    if number_of_folds <= 0:
        raise ValueError("number_of_folds must be positive")
    required = {"subject_key", "event_id", "valid_duration", "hand_relation"}
    missing = required - set(events.columns)
    if missing and len(events):
        raise ValueError(f"Events are missing columns: {sorted(missing)}")
    all_subjects = {
        str(value) for value in (subject_keys or set(events.get("subject_key", [])))
    }
    if events.empty and not all_subjects:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")
        output_path.with_name("subject_folds.manifest.json").write_text(
            json.dumps(
                {
                    "version": 2,
                    "number_of_folds": int(number_of_folds),
                    "seed": int(seed),
                    "subjects": 0,
                    "assignments_sha256": hashlib.sha256(b"{}").hexdigest(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {}
    if missing:
        raise ValueError(f"Events are missing columns: {sorted(missing)}")
    coverage = events.get("evaluable", events.get("coverage", "full") == "full")
    valid = events[(events["valid_duration"]) & coverage]
    summary = (
        valid.assign(
            same=(valid["hand_relation"] == "same").astype(int),
            different=(valid["hand_relation"] == "different").astype(int),
        )
        .groupby("subject_key", as_index=False)
        .agg(events=("event_id", "count"), same=("same", "sum"), different=("different", "sum"))
    )
    summary = pd.DataFrame({"subject_key": sorted(all_subjects)}).merge(
        summary, on="subject_key", how="left"
    )
    for column in ("events", "same", "different"):
        summary[column] = summary[column].fillna(0).astype(int)
    rng = np.random.default_rng(seed)
    summary["tie_breaker"] = rng.random(len(summary))
    summary = summary.sort_values(
        ["events", "same", "different", "tie_breaker"], ascending=False
    )
    totals = np.zeros((number_of_folds, 4), dtype=np.float64)
    assignments: dict[str, int] = {}
    targets = np.asarray(
        [len(summary), summary["events"].sum(), summary["same"].sum(), summary["different"].sum()]
    ) / number_of_folds
    for row in summary.itertuples(index=False):
        contribution = np.asarray([1.0, row.events, row.same, row.different])
        scores = []
        for fold in range(number_of_folds):
            proposed = totals.copy()
            proposed[fold] += contribution
            normalized = (proposed - targets[None, :]) / np.maximum(targets[None, :], 1.0)
            scores.append(float(np.square(normalized).sum()))
        chosen = int(np.argmin(scores))
        totals[chosen] += contribution
        assignments[str(row.subject_key)] = chosen
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(assignments, indent=2, sort_keys=True), encoding="utf-8")
    fingerprint = hashlib.sha256(
        json.dumps(assignments, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "version": 2,
        "number_of_folds": int(number_of_folds),
        "seed": int(seed),
        "subjects": len(assignments),
        "assignments_sha256": fingerprint,
    }
    output_path.with_name("subject_folds.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return assignments


def load_subject_folds(path: str | Path) -> dict[str, int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return {str(key): int(value) for key, value in json.load(handle).items()}

