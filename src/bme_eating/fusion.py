from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bme_eating.metrics import evaluate_events
from bme_eating.postprocess import probabilities_to_events

ALIGNMENT_KEYS = ["subject_key", "session_id", "timestamp_ms"]
PUBLIC_PREDICTION_COLUMNS = [
    "subject_key",
    "segment_id",
    "session_id",
    "timestamp_ms",
    "state_probability",
    "start_probability",
    "end_probability",
]
AUXILIARY_PREDICTION_COLUMNS = [
    "ppg_gate_mean",
    "ppg_gate_recent",
    "ppg_valid_fraction",
    "motion_valid_fraction",
]
FROZEN_BASELINE_FILES = (
    "model.json",
    "metadata.json",
    "validation_predictions.parquet",
    "test_predictions.parquet",
    "selected_postprocess.json",
    "test_metrics.json",
    "run_manifest.json",
)
FROZEN_INPUT_HASHES = (
    "quality_report",
    "quality_expectations",
    "subject_folds",
    "subject_folds_manifest",
    "events",
    "anchors",
    "segments",
)
FUSION_EXPERIMENT_NAME = "baseline_dtp_fusion"
FUSION_RUN_NAME_PATTERN = re.compile(
    rf"{FUSION_EXPERIMENT_NAME}_[A-Za-z0-9][A-Za-z0-9_-]{{0,63}}"
)


class FusionGateError(RuntimeError):
    def __init__(self, message: str, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def validate_fusion_run_name(configured_name: str, requested_name: str | None) -> str:
    if configured_name != FUSION_EXPERIMENT_NAME:
        raise ValueError(f"Fusion experiment name must remain {FUSION_EXPERIMENT_NAME}")
    if requested_name is None:
        return configured_name
    run_name = str(requested_name).strip()
    if not FUSION_RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError(
            "--run-name must start with 'baseline_dtp_fusion_' and contain only "
            "ASCII letters, digits, underscores, or hyphens"
        )
    return run_name


def prepare_fusion_run_root(
    experiments_root: Path,
    run_name: str,
    *,
    fresh: bool,
    named_run: bool,
) -> Path:
    experiments_root.mkdir(parents=True, exist_ok=True)
    run_root = experiments_root / run_name
    if fresh:
        try:
            run_root.mkdir()
        except FileExistsError as error:
            raise RuntimeError(
                f"Fresh fusion run already exists and will not be reused: {run_name}"
            ) from error
    elif named_run and not run_root.is_dir():
        raise FileNotFoundError(
            f"Named fusion run does not exist; start fold 0 with --fresh: {run_name}"
        )
    else:
        run_root.mkdir(exist_ok=True)
    return run_root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path.name}")
    return payload


def validate_frozen_baseline_fold(
    output_root: Path,
    baseline_name: str,
    fold: int,
    source_commit: str,
    *,
    require_clean: bool = False,
) -> dict[str, Any]:
    fold_dir = output_root / "experiments" / baseline_name / f"fold_{fold}"
    missing = [name for name in FROZEN_BASELINE_FILES if not (fold_dir / name).is_file()]
    if missing:
        suffix = (
            "; after committing the reviewed code, run "
            "python scripts/backfill_baseline_manifests.py --source-commit 3ca55bb"
            if missing == ["run_manifest.json"]
            else ""
        )
        raise FileNotFoundError(
            f"Frozen baseline fold {fold} is incomplete; missing: {missing}{suffix}"
        )
    manifest = _read_json(fold_dir / "run_manifest.json")
    git_record = manifest.get("git", {})
    if require_clean:
        if manifest.get("backfilled") or "artifact_provenance" in manifest:
            raise RuntimeError(
                f"Baseline fold {fold} uses a backfilled provenance record; "
                "a clean retrained baseline is required"
            )
        if git_record.get("dirty") is not False:
            raise RuntimeError(f"Baseline fold {fold} was produced from a dirty working tree")
        if str(git_record.get("commit", "")) != source_commit:
            raise RuntimeError(
                f"Baseline fold {fold} Git commit is "
                f"{git_record.get('commit')!r}, expected {source_commit!r}"
            )
    declared_commit = str(
        manifest.get("artifact_provenance", {}).get(
            "claimed_source_commit", git_record.get("commit", "")
        )
    )
    if declared_commit != source_commit:
        raise RuntimeError(
            f"Frozen baseline fold {fold} declared source commit is "
            f"{declared_commit!r}, expected {source_commit!r}"
        )
    artifact_hashes = manifest.get("artifact_hashes", {})
    required_artifacts = [name for name in FROZEN_BASELINE_FILES if name != "run_manifest.json"]
    missing_hashes = [name for name in required_artifacts if name not in artifact_hashes]
    if missing_hashes:
        raise RuntimeError(
            "Frozen baseline manifest does not contain artifact hashes for "
            f"{missing_hashes}; rerun scripts/backfill_baseline_manifests.py (no retraining)"
        )
    for name in required_artifacts:
        actual = sha256_file(fold_dir / name)
        if actual != artifact_hashes[name]:
            raise RuntimeError(f"Frozen baseline artifact hash changed: fold_{fold}/{name}")

    tracked_paths = {
        "quality_report": output_root / "indices" / "quality_report.json",
        "quality_expectations": output_root / "indices" / "quality_expectations.json",
        "subject_folds": output_root / "indices" / "subject_folds.json",
        "subject_folds_manifest": output_root / "indices" / "subject_folds.manifest.json",
        "events": output_root / "indices" / "events.parquet",
        "anchors": output_root / "indices" / "anchors.parquet",
        "segments": output_root / "indices" / "segments.parquet",
    }
    input_hashes = manifest.get("hashes", {})
    for name in FROZEN_INPUT_HASHES:
        path = tracked_paths[name]
        if not path.is_file() or name not in input_hashes:
            raise RuntimeError(f"Frozen baseline manifest/input is missing: {name}")
        if sha256_file(path) != input_hashes[name]:
            raise RuntimeError(f"Frozen data or split fingerprint changed: {name}")
    return {
        "directory": fold_dir,
        "manifest": manifest,
        "artifact_hashes": {name: artifact_hashes[name] for name in required_artifacts},
        "input_hashes": {name: input_hashes[name] for name in FROZEN_INPUT_HASHES},
    }


def validate_clean_baseline_experiment(
    output_root: Path,
    baseline_name: str,
    source_commit: str,
    *,
    number_of_folds: int = 5,
) -> dict[int, dict[str, Any]]:
    if number_of_folds <= 0:
        raise ValueError("number_of_folds must be positive")
    folds = {
        fold: validate_frozen_baseline_fold(
            output_root,
            baseline_name,
            fold,
            source_commit,
            require_clean=True,
        )
        for fold in range(number_of_folds)
    }
    config_hashes = {
        str(info["manifest"].get("resolved_config_sha256", ""))
        for info in folds.values()
    }
    if "" in config_hashes or len(config_hashes) != 1:
        raise RuntimeError("Clean baseline folds do not share one resolved configuration hash")
    input_fingerprints = {
        json.dumps(info["input_hashes"], sort_keys=True, separators=(",", ":"))
        for info in folds.values()
    }
    if len(input_fingerprints) != 1:
        raise RuntimeError("Clean baseline folds do not share one data and split fingerprint")
    return folds


def _validate_prediction_frame(frame: pd.DataFrame, name: str) -> None:
    required = set(PUBLIC_PREDICTION_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} predictions are missing columns: {sorted(missing)}")
    if frame.duplicated(ALIGNMENT_KEYS).any():
        raise ValueError(f"{name} predictions contain duplicate alignment keys")
    numeric = frame[
        ["timestamp_ms", "state_probability", "start_probability", "end_probability"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} predictions contain NaN or infinite values")
    if ((numeric[:, 1:] < 0.0) | (numeric[:, 1:] > 1.0)).any():
        raise ValueError(f"{name} probabilities must be in [0, 1]")
    auxiliary = [column for column in AUXILIARY_PREDICTION_COLUMNS if column in frame]
    if auxiliary:
        values = frame[auxiliary].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} quality diagnostics contain NaN or infinite values")
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"{name} quality diagnostics must be in [0, 1]")


def align_prediction_frames(
    baseline: pd.DataFrame, dtp: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_prediction_frame(baseline, "baseline")
    _validate_prediction_frame(dtp, "DTP")
    baseline_sorted = baseline.sort_values(ALIGNMENT_KEYS).reset_index(drop=True)
    dtp_sorted = dtp.sort_values(ALIGNMENT_KEYS).reset_index(drop=True)
    baseline_keys = pd.MultiIndex.from_frame(baseline_sorted[ALIGNMENT_KEYS])
    dtp_keys = pd.MultiIndex.from_frame(dtp_sorted[ALIGNMENT_KEYS])
    if not baseline_keys.equals(dtp_keys):
        baseline_subject_time = pd.MultiIndex.from_frame(
            baseline_sorted[["subject_key", "timestamp_ms"]]
        )
        dtp_subject_time = pd.MultiIndex.from_frame(dtp_sorted[["subject_key", "timestamp_ms"]])
        if baseline_subject_time.equals(dtp_subject_time):
            raise ValueError("Prediction session_id values disagree on an otherwise equal timeline")
        missing_from_dtp = len(baseline_keys.difference(dtp_keys))
        missing_from_baseline = len(dtp_keys.difference(baseline_keys))
        raise ValueError(
            "Baseline and DTP timelines are not one-to-one: "
            f"missing_from_dtp={missing_from_dtp}, "
            f"missing_from_baseline={missing_from_baseline}"
        )
    return baseline_sorted, dtp_sorted


def average_prediction_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("At least one prediction frame is required")
    reference = frames[0]
    aligned_frames = [align_prediction_frames(reference, frame)[1] for frame in frames]
    auxiliary = [
        column
        for column in AUXILIARY_PREDICTION_COLUMNS
        if all(column in frame.columns for frame in aligned_frames)
    ]
    output = aligned_frames[0][[*PUBLIC_PREDICTION_COLUMNS, *auxiliary]].copy()
    for column in (
        "state_probability",
        "start_probability",
        "end_probability",
        *auxiliary,
    ):
        values = np.stack(
            [frame[column].to_numpy(dtype=np.float64) for frame in aligned_frames], axis=0
        )
        output[column] = values.mean(axis=0).astype(np.float32)
    return output


def assemble_crossfit_predictions(
    validation_frames: dict[int, pd.DataFrame],
    test_frames: dict[int, pd.DataFrame],
    partition_subjects: dict[int, set[str]],
    test_subjects: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    partitions = set(partition_subjects)
    if partitions != set(validation_frames) or partitions != set(test_frames):
        raise ValueError("Cross-fit prediction partitions are incomplete")
    seen_validation_subjects: set[str] = set()
    oof_parts: list[pd.DataFrame] = []
    for partition in sorted(partitions):
        validation = validation_frames[partition].copy()
        actual_subjects = set(validation["subject_key"].astype(str).unique())
        expected_subjects = {str(value) for value in partition_subjects[partition]}
        if actual_subjects != expected_subjects:
            raise RuntimeError(
                f"Cross-fit partition {partition} validation subjects do not match its holdout"
            )
        if seen_validation_subjects & actual_subjects:
            raise RuntimeError("Cross-fit validation subjects appear in more than one partition")
        if actual_subjects & test_subjects:
            raise RuntimeError("Cross-fit OOF predictions contain outer-fold test subjects")
        seen_validation_subjects.update(actual_subjects)
        validation["calibration_fold"] = partition
        oof_parts.append(validation)
        actual_test_subjects = set(test_frames[partition]["subject_key"].astype(str).unique())
        if actual_test_subjects != test_subjects:
            raise RuntimeError(
                f"Cross-fit partition {partition} test predictions do not cover the outer fold"
            )
    expected_oof_subjects = {
        str(value) for subjects in partition_subjects.values() for value in subjects
    }
    if seen_validation_subjects != expected_oof_subjects:
        raise RuntimeError("Cross-fit OOF predictions do not cover all outer-training subjects")
    oof = pd.concat(oof_parts, ignore_index=True).sort_values(ALIGNMENT_KEYS).reset_index(drop=True)
    _validate_prediction_frame(oof, "cross-fit OOF")
    test_ensemble = average_prediction_frames([test_frames[key] for key in sorted(partitions)])
    return oof, test_ensemble


def fuse_prediction_frames(
    baseline: pd.DataFrame,
    dtp: pd.DataFrame,
    *,
    alpha: float,
    beta: float,
    residual_clip: float = 2.0,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    if not all(np.isfinite(value) for value in (alpha, beta, residual_clip, epsilon)):
        raise ValueError("Fusion parameters must be finite")
    if alpha < 0 or residual_clip <= 0 or not 0 < epsilon < 0.5:
        raise ValueError("Expected alpha >= 0, residual_clip > 0, and 0 < epsilon < 0.5")
    baseline_sorted, dtp_sorted = align_prediction_frames(baseline, dtp)
    output = baseline_sorted[PUBLIC_PREDICTION_COLUMNS].copy()
    if alpha == 0.0 and beta == 0.0:
        return output
    base_probability = baseline_sorted["state_probability"].to_numpy(dtype=np.float64)
    dtp_probability = dtp_sorted["state_probability"].to_numpy(dtype=np.float64)
    base_clipped = np.clip(base_probability, epsilon, 1.0 - epsilon)
    dtp_clipped = np.clip(dtp_probability, epsilon, 1.0 - epsilon)
    base_logit = np.log(base_clipped) - np.log1p(-base_clipped)
    dtp_logit = np.log(dtp_clipped) - np.log1p(-dtp_clipped)
    residual = np.clip(dtp_logit - base_logit, -residual_clip, residual_clip)
    fused_logit = base_logit + beta + alpha * residual
    fused = np.empty_like(fused_logit)
    positive = fused_logit >= 0
    fused[positive] = 1.0 / (1.0 + np.exp(-fused_logit[positive]))
    exp_values = np.exp(fused_logit[~positive])
    fused[~positive] = exp_values / (1.0 + exp_values)
    if not np.isfinite(fused).all():
        raise RuntimeError("Fusion generated non-finite probabilities")
    output["state_probability"] = fused.astype(np.float32)
    # Boundary localization is intentionally frozen to the baseline.
    output["start_probability"] = baseline_sorted["start_probability"].to_numpy()
    output["end_probability"] = baseline_sorted["end_probability"].to_numpy()
    return output


def _event_parameters(postprocess: dict[str, Any]) -> dict[str, Any]:
    detector_mode = str(postprocess.get("detector_mode", "hysteresis_v1"))
    if detector_mode == "dual_ema":
        names = (
            "fast_ema_half_life_seconds",
            "slow_ema_half_life_seconds",
            "fast_high_threshold",
            "slow_high_threshold",
            "exit_threshold_ratio",
            "off_duration_seconds",
            "minimum_event_seconds",
            "merge_gap_seconds",
            "boundary_lookback_seconds",
        )
    elif detector_mode == "hysteresis_v1":
        names = (
            "ema_half_life_seconds",
            "high_threshold",
            "low_threshold",
            "minimum_event_seconds",
            "merge_gap_seconds",
            "boundary_lookback_seconds",
        )
    else:
        raise ValueError(f"Unsupported frozen detector mode: {detector_mode}")
    missing = [name for name in names if name not in postprocess]
    if missing:
        raise ValueError(f"Frozen postprocess parameters are incomplete: {missing}")
    return {"detector_mode": detector_mode, **{name: float(postprocess[name]) for name in names}}


def _f1_from_metrics(metrics: dict[str, float]) -> float:
    return float(metrics["f1"])


def _finite_or(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def evaluate_fusion_predictions(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    ignore: pd.DataFrame,
    postprocess: dict[str, Any],
) -> tuple[dict[str, float], pd.DataFrame]:
    events = probabilities_to_events(predictions, **_event_parameters(postprocess))
    iou_threshold = float(postprocess["iou_threshold"])
    matching_method = str(postprocess.get("matching_method", "max_cardinality_iou"))
    metrics, matches = evaluate_events(
        truth,
        events,
        iou_threshold=iou_threshold,
        method=matching_method,
        ignore=ignore,
    )
    strict, _ = evaluate_events(
        truth,
        events,
        iou_threshold=iou_threshold,
        method=matching_method,
    )
    exposure_hours = 0.0
    for _, group in predictions.groupby(["subject_key", "session_id"], sort=False):
        exposure_hours += max(
            0.0,
            (float(group["timestamp_ms"].max()) - float(group["timestamp_ms"].min()) + 3000.0)
            / 3_600_000.0,
        )
    relation = matches.get("hand_relation", pd.Series("unknown", index=matches.index, dtype=object))
    truth_relation = truth.get(
        "hand_relation", pd.Series("unknown", index=truth.index, dtype=object)
    )

    def relation_sensitivity(name: str) -> float:
        count = int((truth_relation == name).sum())
        return float((relation == name).sum() / count) if count else float("nan")

    boundary_values = [
        float(metrics[name])
        for name in ("start_mae_seconds", "end_mae_seconds")
        if math.isfinite(float(metrics[name]))
    ]
    summary = {
        **{name: float(value) for name, value in metrics.items()},
        "strict_no_ignore_f1": _f1_from_metrics(strict),
        "different_sensitivity": relation_sensitivity("different"),
        "same_sensitivity": relation_sensitivity("same"),
        "false_positives_per_observed_hour": (
            float(metrics["false_positive"]) / exposure_hours if exposure_hours else 0.0
        ),
        "boundary_mae_seconds": (
            float(np.mean(boundary_values)) if boundary_values else float("inf")
        ),
        "predicted_events": float(len(events)),
    }
    return summary, events


def _canonical_events(events: pd.DataFrame) -> pd.DataFrame:
    columns = ["subject_key", "session_id", "start_ms", "end_ms", "score"]
    return events[columns].sort_values(columns).reset_index(drop=True)


def _selection_rank(row: dict[str, Any], epoch: int) -> tuple[float, ...]:
    return (
        _finite_or(row.get("f1"), -math.inf),
        _finite_or(row.get("different_sensitivity"), -math.inf),
        -_finite_or(row.get("false_positives_per_observed_hour"), math.inf),
        -_finite_or(row.get("boundary_mae_seconds"), math.inf),
        -float(epoch),
        -float(row["alpha"]),
        -abs(float(row["beta"])),
        -float(row["beta"]),
    )


def complementarity_diagnostics(
    baseline: pd.DataFrame,
    dtp: pd.DataFrame,
    fused: pd.DataFrame,
    residual_clip: float,
    epsilon: float,
) -> dict[str, float | None]:
    baseline_sorted, dtp_sorted = align_prediction_frames(baseline, dtp)
    base = baseline_sorted["state_probability"].to_numpy(dtype=np.float64)
    deep = dtp_sorted["state_probability"].to_numpy(dtype=np.float64)
    base_logit = np.log(np.clip(base, epsilon, 1 - epsilon)) - np.log1p(
        -np.clip(base, epsilon, 1 - epsilon)
    )
    deep_logit = np.log(np.clip(deep, epsilon, 1 - epsilon)) - np.log1p(
        -np.clip(deep, epsilon, 1 - epsilon)
    )
    raw_residual = deep_logit - base_logit
    correlation = (
        float(np.corrcoef(base, deep)[0, 1])
        if len(base) > 1 and np.std(base) > 0 and np.std(deep) > 0
        else float("nan")
    )
    values: dict[str, float] = {
        "state_probability_correlation": correlation,
        "mean_absolute_logit_residual": float(np.mean(np.abs(raw_residual))),
        "residual_clipped_fraction": float(np.mean(np.abs(raw_residual) > residual_clip)),
        "mean_absolute_state_change": float(
            np.mean(np.abs(fused["state_probability"].to_numpy(dtype=np.float64) - base))
        ),
    }
    return {key: value if math.isfinite(value) else None for key, value in values.items()}


@dataclass
class FusionValidator:
    baseline: pd.DataFrame
    truth: pd.DataFrame
    ignore: pd.DataFrame
    postprocess: dict[str, Any]
    fusion_config: dict[str, Any]
    output_dir: Path
    forbidden_subjects: set[str]

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        baseline_subjects = set(self.baseline["subject_key"].astype(str).unique())
        if baseline_subjects & self.forbidden_subjects:
            raise ValueError("Baseline OOF validation data contains outer-fold subjects")
        self.baseline_metrics, self.baseline_events = evaluate_fusion_predictions(
            self.baseline, self.truth, self.ignore, self.postprocess
        )
        self._trials: list[dict[str, Any]] = []
        trial_path = self.output_dir / "fusion_trials.csv"
        if trial_path.exists():
            self._trials = pd.read_csv(trial_path).to_dict("records")

    @property
    def signature(self) -> str:
        payload = {
            "alpha_candidates": self.fusion_config["alpha_candidates"],
            "beta_candidates": self.fusion_config["beta_candidates"],
            "residual_clip": self.fusion_config["residual_clip"],
            "probability_epsilon": self.fusion_config["probability_epsilon"],
            "postprocess": self.postprocess,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def __call__(self, dtp: pd.DataFrame, epoch: int) -> dict[str, Any]:
        dtp_subjects = set(dtp["subject_key"].astype(str).unique())
        if dtp_subjects & self.forbidden_subjects:
            raise ValueError("DTP validation data contains outer-fold subjects")
        baseline, aligned_dtp = align_prediction_frames(self.baseline, dtp)
        rows: list[dict[str, Any]] = []
        candidate_frames: dict[tuple[float, float], pd.DataFrame] = {}
        candidate_events: dict[tuple[float, float], pd.DataFrame] = {}
        for alpha_value in self.fusion_config["alpha_candidates"]:
            for beta_value in self.fusion_config["beta_candidates"]:
                alpha = float(alpha_value)
                beta = float(beta_value)
                fused = fuse_prediction_frames(
                    baseline,
                    aligned_dtp,
                    alpha=alpha,
                    beta=beta,
                    residual_clip=float(self.fusion_config["residual_clip"]),
                    epsilon=float(self.fusion_config["probability_epsilon"]),
                )
                metrics, events = evaluate_fusion_predictions(
                    fused, self.truth, self.ignore, self.postprocess
                )
                row = {"epoch": epoch, "alpha": alpha, "beta": beta, **metrics}
                rows.append(row)
                candidate_frames[(alpha, beta)] = fused
                candidate_events[(alpha, beta)] = events
        reference = next(row for row in rows if row["alpha"] == 0.0 and row["beta"] == 0.0)
        reference_frame = candidate_frames[(0.0, 0.0)]
        if not reference_frame[PUBLIC_PREDICTION_COLUMNS].equals(
            baseline[PUBLIC_PREDICTION_COLUMNS]
        ):
            raise RuntimeError("alpha=0,beta=0 did not reproduce baseline point predictions")
        if not _canonical_events(candidate_events[(0.0, 0.0)]).equals(
            _canonical_events(self.baseline_events)
        ):
            raise RuntimeError("alpha=0,beta=0 did not reproduce baseline events")
        for name in (
            "f1",
            "strict_no_ignore_f1",
            "different_sensitivity",
            "false_positives_per_observed_hour",
            "start_mae_seconds",
            "end_mae_seconds",
        ):
            left = _finite_or(reference.get(name), math.nan)
            right = _finite_or(self.baseline_metrics.get(name), math.nan)
            if not (math.isnan(left) and math.isnan(right)) and not math.isclose(
                left, right, rel_tol=0.0, abs_tol=0.0
            ):
                raise RuntimeError(f"alpha=0,beta=0 did not reproduce baseline metric {name}")
        beta_only = max(
            (row for row in rows if float(row["alpha"]) == 0.0),
            key=lambda row: _selection_rank(row, epoch),
        )
        dtp_candidates = [row for row in rows if float(row["alpha"]) > 0.0]
        if not dtp_candidates:
            raise ValueError("Fusion search must contain at least one alpha > 0 candidate")
        selected = max(dtp_candidates, key=lambda row: _selection_rank(row, epoch))
        selected_frame = candidate_frames[(float(selected["alpha"]), float(selected["beta"]))]
        self._trials.extend(rows)
        trial_frame = (
            pd.DataFrame(self._trials)
            .drop_duplicates(["epoch", "alpha", "beta"], keep="last")
            .sort_values(["epoch", "alpha", "beta"])
            .reset_index(drop=True)
        )
        self._trials = trial_frame.to_dict("records")
        trial_frame.to_csv(self.output_dir / "fusion_trials.csv", index=False)
        return {
            "epoch": epoch,
            "alpha": float(selected["alpha"]),
            "beta": float(selected["beta"]),
            "metrics": {
                key: value
                for key, value in selected.items()
                if key not in {"epoch", "alpha", "beta"}
            },
            "baseline_metrics": self.baseline_metrics,
            "beta_only_beta": float(beta_only["beta"]),
            "beta_only_metrics": {
                key: value
                for key, value in beta_only.items()
                if key not in {"epoch", "alpha", "beta"}
            },
            "rank": list(_selection_rank(selected, epoch)),
            "diagnostics": complementarity_diagnostics(
                baseline,
                aligned_dtp,
                selected_frame,
                float(self.fusion_config["residual_clip"]),
                float(self.fusion_config["probability_epsilon"]),
            ),
        }


def evaluate_internal_gate(
    selection: dict[str, Any], gate_config: dict[str, Any]
) -> dict[str, Any]:
    candidate = selection["metrics"]
    baseline = selection["baseline_metrics"]
    beta_only = selection["beta_only_metrics"]
    checks = {
        "alpha_is_positive": float(selection["alpha"]) > 0.0,
        "f1_improvement": (
            float(candidate["f1"]) - float(baseline["f1"])
            >= float(gate_config["minimum_f1_improvement"])
        ),
        "f1_improvement_over_beta_only": (
            float(candidate["f1"]) - float(beta_only["f1"])
            >= float(gate_config["minimum_f1_improvement_over_beta_only"])
        ),
        "different_sensitivity_improvement": (
            _finite_or(candidate.get("different_sensitivity"), -math.inf)
            - _finite_or(baseline.get("different_sensitivity"), math.inf)
            >= float(gate_config["minimum_different_sensitivity_improvement"])
        ),
        "strict_no_ignore_not_lower": (
            float(candidate["strict_no_ignore_f1"]) >= float(baseline["strict_no_ignore_f1"])
        ),
        "different_sensitivity_not_lower_than_beta_only": (
            _finite_or(candidate.get("different_sensitivity"), -math.inf)
            >= _finite_or(beta_only.get("different_sensitivity"), math.inf)
        ),
        "strict_no_ignore_not_lower_than_beta_only": (
            float(candidate["strict_no_ignore_f1"])
            >= float(beta_only["strict_no_ignore_f1"])
        ),
        "fp_per_hour_not_higher_than_beta_only": (
            float(candidate["false_positives_per_observed_hour"])
            <= float(beta_only["false_positives_per_observed_hour"])
        ),
        "fp_per_hour_within_ratio": (
            float(candidate["false_positives_per_observed_hour"])
            <= float(baseline["false_positives_per_observed_hour"])
            * float(gate_config["maximum_fp_per_hour_ratio"])
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def metrics_summary_from_suite(payload: dict[str, Any]) -> dict[str, float]:
    method = str(payload["primary_method"])
    primary = payload[method]
    return {
        "f1": float(primary["f1"]),
        "sensitivity": float(primary["sensitivity"]),
        "different_sensitivity": _finite_or(
            payload.get("hand_relation", {}).get("different", {}).get("sensitivity"),
            float("nan"),
        ),
        "same_sensitivity": _finite_or(
            payload.get("hand_relation", {}).get("same", {}).get("sensitivity"),
            float("nan"),
        ),
        "strict_no_ignore_f1": float(payload["strict_no_ignore"]["f1"]),
        "false_positives_per_observed_hour": float(primary["false_positives_per_observed_hour"]),
        "start_mae_seconds": float(primary["start_mae_seconds"]),
        "end_mae_seconds": float(primary["end_mae_seconds"]),
    }


def evaluate_fold0_gate(
    candidate: dict[str, float],
    baseline: dict[str, float],
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    boundary_ratio = float(gate_config["maximum_boundary_mae_ratio"])
    checks = {
        "f1_improvement": (
            candidate["f1"] - baseline["f1"] >= float(gate_config["minimum_f1_improvement"])
        ),
        "strict_no_ignore_not_lower": (
            candidate["strict_no_ignore_f1"] >= baseline["strict_no_ignore_f1"]
        ),
        "different_sensitivity_not_lower": (
            candidate["different_sensitivity"] >= baseline["different_sensitivity"]
        ),
        "fp_per_hour_within_ratio": (
            candidate["false_positives_per_observed_hour"]
            <= baseline["false_positives_per_observed_hour"]
            * float(gate_config["maximum_fp_per_hour_ratio"])
        ),
        "start_mae_within_ratio": (
            candidate["start_mae_seconds"] <= baseline["start_mae_seconds"] * boundary_ratio
        ),
        "end_mae_within_ratio": (
            candidate["end_mae_seconds"] <= baseline["end_mae_seconds"] * boundary_ratio
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value
