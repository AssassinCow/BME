from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def create_subject_folds(
    events: pd.DataFrame,
    number_of_folds: int,
    seed: int,
    output_path: Path,
) -> dict[str, int]:
    valid = events[(events["valid_duration"]) & (events.get("coverage", "full") == "full")]
    summary = (
        valid.assign(
            same=(valid["hand_relation"] == "same").astype(int),
            different=(valid["hand_relation"] == "different").astype(int),
        )
        .groupby("subject_key", as_index=False)
        .agg(events=("event_id", "count"), same=("same", "sum"), different=("different", "sum"))
    )
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
    return assignments


def load_subject_folds(path: str | Path) -> dict[str, int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return {str(key): int(value) for key, value in json.load(handle).items()}

