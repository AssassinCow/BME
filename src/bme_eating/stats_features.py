from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STATS_FEATURE_COLUMNS = (
    "local_acc_y_mean",
    "local_acc_z_zero_crossing_rate",
    "local_gyro_z_mad",
    "local_acc_mag_std",
    "local_gyro_z_zero_crossing_rate",
    "local_acc_mag_iqr",
    "local_acc_mag_jerk_rms",
    "local_gyro_y_mad",
    "local_acc_y_median",
    "local_ppg_valid_fraction",
    "local_gyro_mag_jerk_p95",
    "local_acc_x_median",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FoldRobustScaler:
    columns: tuple[str, ...]
    median: np.ndarray
    iqr: np.ndarray
    training_subjects: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        columns: Sequence[str] = STATS_FEATURE_COLUMNS,
        *,
        training_subjects: set[str],
    ) -> FoldRobustScaler:
        columns = tuple(str(column) for column in columns)
        required = {"subject_key", *columns}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Statistics scaler input is missing columns: {sorted(missing)}")
        if not training_subjects:
            raise ValueError("Statistics scaler requires at least one training subject")
        selected = frame[frame["subject_key"].astype(str).isin(training_subjects)]
        observed_subjects = set(selected["subject_key"].astype(str))
        if observed_subjects != set(training_subjects):
            absent = sorted(set(training_subjects) - observed_subjects)
            raise ValueError(f"Statistics scaler subjects have no rows: {absent}")
        values = selected.loc[:, columns].to_numpy(dtype=np.float64)
        values[~np.isfinite(values)] = np.nan
        with np.errstate(all="ignore"):
            median = np.nanmedian(values, axis=0)
            lower = np.nanquantile(values, 0.25, axis=0)
            upper = np.nanquantile(values, 0.75, axis=0)
        median = np.nan_to_num(median, nan=0.0, posinf=0.0, neginf=0.0)
        iqr = np.nan_to_num(upper - lower, nan=0.0, posinf=0.0, neginf=0.0)
        iqr = np.maximum(iqr, 1e-6)
        return cls(
            columns=columns,
            median=median.astype(np.float64),
            iqr=iqr.astype(np.float64),
            training_subjects=tuple(sorted(str(value) for value in training_subjects)),
        )

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.shape[-1] != len(self.columns):
            raise ValueError(
                f"Expected {len(self.columns)} statistics, received shape {values.shape}"
            )
        missing = ~np.isfinite(values)
        imputed = np.where(missing, self.median, values)
        scaled = np.clip((imputed - self.median) / self.iqr, -8.0, 8.0)
        return np.concatenate((scaled, missing.astype(np.float64)), axis=-1).astype(np.float32)

    def transform_frame(self, frame: pd.DataFrame, prefix: str = "stat_") -> pd.DataFrame:
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Statistics frame is missing columns: {sorted(missing)}")
        transformed = self.transform_array(frame.loc[:, self.columns].to_numpy(dtype=np.float64))
        output = frame.copy()
        names = [*(f"{prefix}{name}" for name in self.columns)]
        names.extend(f"{prefix}{name}_missing" for name in self.columns)
        output.loc[:, names] = transformed
        return output

    def assert_unseen(self, subject_keys: set[str]) -> None:
        overlap = set(self.training_subjects) & {str(value) for value in subject_keys}
        if overlap:
            raise RuntimeError(f"Statistics scaler leakage detected for subjects: {sorted(overlap)}")

    def to_json(self) -> dict[str, Any]:
        payload = {
            "columns": list(self.columns),
            "median": self.median.tolist(),
            "iqr": self.iqr.tolist(),
            "training_subjects": list(self.training_subjects),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> FoldRobustScaler:
        content = {key: value for key, value in payload.items() if key != "sha256"}
        canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if payload.get("sha256") != expected:
            raise RuntimeError("Statistics scaler hash mismatch")
        return cls(
            columns=tuple(str(value) for value in content["columns"]),
            median=np.asarray(content["median"], dtype=np.float64),
            iqr=np.asarray(content["iqr"], dtype=np.float64),
            training_subjects=tuple(str(value) for value in content["training_subjects"]),
        )


def audit_feature_provenance(
    *,
    project_root: Path,
    input_root: Path,
    source_paths: Sequence[str],
    selection_note_path: str | None,
    assumed_used_all_outer_folds: bool,
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    for relative in source_paths:
        source = (input_root / relative).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Feature provenance source is missing: {source}")
        sources.append(
            {
                "scope": "input_artifact",
                "relative_path": source.relative_to(input_root).as_posix(),
                "sha256": _sha256(source),
            }
        )
    if selection_note_path:
        note = (project_root / selection_note_path).resolve()
        if not note.is_file():
            raise FileNotFoundError(f"Feature selection note is missing: {note}")
        sources.append(
            {
                "scope": "repository",
                "relative_path": note.relative_to(project_root).as_posix(),
                "sha256": _sha256(note),
            }
        )
    conclusion = (
        "development_stress_only"
        if assumed_used_all_outer_folds
        else "nested_outer_fold_eligible"
    )
    return {
        "schema_version": 1,
        "feature_columns": list(STATS_FEATURE_COLUMNS),
        "feature_order_sha256": hashlib.sha256(
            "\n".join(STATS_FEATURE_COLUMNS).encode("utf-8")
        ).hexdigest(),
        "sources": sources,
        "used_all_outer_folds": bool(assumed_used_all_outer_folds),
        "evidence_classification": conclusion,
        "independent_evidence_requirement": (
            "official_hidden_test_or_new_subjects"
            if assumed_used_all_outer_folds
            else "nested_outer_fold"
        ),
    }
