from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from bme_eating.fusion import (
    ALIGNMENT_KEYS,
    PUBLIC_PREDICTION_COLUMNS,
    align_prediction_frames,
    evaluate_fusion_predictions,
    json_safe,
)
from bme_eating.metrics import evaluate_events, partition_evaluation_events
from bme_eating.postprocess import causal_ema, tune_dual_ema_parameters

TARGET_KEYS = ["subject_key", "session_id", "timestamp_ms"]
FUSION_PARAMETER_NAMES = (
    "alpha_positive",
    "alpha_negative",
    "baseline_support_min",
    "baseline_support_max",
    "dtp_on_threshold",
    "dtp_off_threshold",
    "persistence_seconds",
    "use_quality_weighting",
)


@dataclass(frozen=True)
class PlattCalibrator:
    slope: float
    intercept: float
    regularization: float
    training_rows: int
    training_subjects: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "method": "positive_slope_platt",
            "slope": self.slope,
            "intercept": self.intercept,
            "regularization": self.regularization,
            "training_rows": self.training_rows,
            "training_subjects": self.training_subjects,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlattCalibrator:
        if payload.get("method") != "positive_slope_platt":
            raise ValueError("Unsupported DTP calibration method")
        return cls(
            slope=float(payload["slope"]),
            intercept=float(payload["intercept"]),
            regularization=float(payload.get("regularization", 0.0)),
            training_rows=int(payload.get("training_rows", 0)),
            training_subjects=int(payload.get("training_subjects", 0)),
        )


def _logit(probability: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    values = np.asarray(logit, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def align_calibration_targets(
    predictions: pd.DataFrame,
    anchors: pd.DataFrame,
) -> pd.DataFrame:
    required = set(TARGET_KEYS) | {"state_target", "state_loss_mask"}
    missing = sorted(required - set(anchors.columns))
    if missing:
        raise ValueError(f"Calibration anchors are missing columns: {missing}")
    if anchors.duplicated(TARGET_KEYS).any():
        raise ValueError("Calibration anchors contain duplicate timeline keys")
    prediction_keys = predictions[TARGET_KEYS].copy()
    prediction_keys["_prediction_row"] = np.arange(len(prediction_keys), dtype=np.int64)
    target_columns = [*TARGET_KEYS, "state_target", "state_loss_mask"]
    aligned = prediction_keys.merge(
        anchors[target_columns],
        on=TARGET_KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_prediction_row")
    if aligned[["state_target", "state_loss_mask"]].isna().any().any():
        raise ValueError("Calibration predictions do not exactly match anchor targets")
    return aligned.reset_index(drop=True)


def fit_platt_calibrator(
    dtp_predictions: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    epsilon: float = 1e-6,
    regularization: float = 1e-4,
    maximum_slope: float = 20.0,
) -> PlattCalibrator:
    if not 0.0 < epsilon < 0.5:
        raise ValueError("Calibration epsilon must be in (0, 0.5)")
    if not math.isfinite(regularization) or regularization < 0:
        raise ValueError("Calibration regularization must be finite and non-negative")
    if not math.isfinite(maximum_slope) or maximum_slope <= 0:
        raise ValueError("Calibration maximum_slope must be positive and finite")
    aligned = align_calibration_targets(dtp_predictions, anchors)
    loss_mask = aligned["state_loss_mask"].to_numpy(dtype=np.float64)
    if not np.isfinite(loss_mask).all() or (loss_mask < 0.0).any():
        raise ValueError("DTP calibration state_loss_mask must be finite and non-negative")
    mask = loss_mask > 0
    if not mask.any():
        raise ValueError("DTP calibration has no evaluable rows")
    target = aligned.loc[mask, "state_target"].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or ((target < 0.0) | (target > 1.0)).any():
        raise ValueError("DTP calibration targets must be finite values in [0, 1]")
    if float(target.max()) == float(target.min()):
        raise ValueError("DTP calibration requires non-constant state targets")
    probability = dtp_predictions["state_probability"].to_numpy(dtype=np.float64)[mask]
    feature = _logit(probability, epsilon)
    subjects = aligned.loc[mask, "subject_key"].astype(str)
    eligible_mask = loss_mask[mask]
    subject_mask_totals = pd.Series(eligible_mask, index=subjects.index).groupby(subjects).sum()
    weights = np.asarray(
        [
            weight / float(subject_mask_totals[subject])
            for subject, weight in zip(subjects, eligible_mask, strict=True)
        ],
        dtype=np.float64,
    )
    weights *= len(weights) / weights.sum()

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        slope, intercept = map(float, parameters)
        logits = slope * feature + intercept
        predicted = _sigmoid(logits)
        loss = np.logaddexp(0.0, logits) - target * logits
        residual = (predicted - target) * weights
        value = float(np.average(loss, weights=weights))
        value += regularization * (slope - 1.0) ** 2
        gradient = np.asarray(
            [
                residual.dot(feature) / weights.sum()
                + 2.0 * regularization * (slope - 1.0),
                residual.sum() / weights.sum(),
            ],
            dtype=np.float64,
        )
        return value, gradient

    result = minimize(
        objective,
        x0=np.asarray([1.0, 0.0], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=((1e-6, maximum_slope), (-20.0, 20.0)),
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"DTP Platt calibration failed: {result.message}")
    return PlattCalibrator(
        slope=float(result.x[0]),
        intercept=float(result.x[1]),
        regularization=float(regularization),
        training_rows=int(mask.sum()),
        training_subjects=int(subjects.nunique()),
    )


def apply_platt_calibrator(
    probability: np.ndarray,
    calibrator: PlattCalibrator,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    if calibrator.slope <= 0 or not all(
        math.isfinite(value) for value in (calibrator.slope, calibrator.intercept)
    ):
        raise ValueError("Platt calibrator must have a positive finite slope")
    return _sigmoid(calibrator.slope * _logit(probability, epsilon) + calibrator.intercept)


def _session_ema(
    predictions: pd.DataFrame,
    probability: np.ndarray,
    half_life_seconds: float,
) -> np.ndarray:
    output = np.empty(len(predictions), dtype=np.float64)
    indexed = predictions.reset_index(drop=True)
    for positions in indexed.groupby(["subject_key", "session_id"], sort=False).indices.values():
        positions = np.asarray(positions, dtype=np.int64)
        timestamps = indexed.loc[positions, "timestamp_ms"].to_numpy(dtype=np.int64)
        differences = np.diff(timestamps)
        positive = differences[differences > 0]
        step_seconds = float(np.median(positive) / 1000.0) if len(positive) else 3.0
        output[positions] = causal_ema(
            probability[positions],
            step_seconds=step_seconds,
            half_life_seconds=half_life_seconds,
        )
    return output


def _persistent_mask(
    predictions: pd.DataFrame,
    condition: np.ndarray,
    persistence_seconds: float,
) -> np.ndarray:
    if persistence_seconds <= 0 or not math.isfinite(persistence_seconds):
        raise ValueError("persistence_seconds must be positive and finite")
    output = np.zeros(len(predictions), dtype=bool)
    indexed = predictions.reset_index(drop=True)
    for positions in indexed.groupby(["subject_key", "session_id"], sort=False).indices.values():
        positions = np.asarray(positions, dtype=np.int64)
        timestamps = indexed.loc[positions, "timestamp_ms"].to_numpy(dtype=np.int64)
        differences = np.diff(timestamps)
        positive = differences[differences > 0]
        step_ms = int(np.median(positive)) if len(positive) else 3000
        run_start: int | None = None
        previous_timestamp: int | None = None
        for position, timestamp in zip(positions, timestamps, strict=True):
            if (
                not bool(condition[position])
                or previous_timestamp is not None
                and timestamp - previous_timestamp > max(step_ms * 2, 6000)
            ):
                run_start = None
            if bool(condition[position]):
                if run_start is None:
                    run_start = int(timestamp)
                duration_ms = int(timestamp) - run_start + step_ms
                output[position] = duration_ms >= persistence_seconds * 1000.0
            previous_timestamp = int(timestamp)
    return output


def _quality_multiplier(dtp: pd.DataFrame) -> np.ndarray:
    names = {
        "ppg_gate_mean",
        "ppg_valid_fraction",
        "motion_valid_fraction",
    }
    missing = sorted(names - set(dtp.columns))
    if missing:
        raise ValueError(
            "Quality-weighted fusion requires DTP diagnostic columns: " + ", ".join(missing)
        )
    ppg_reliability = (
        dtp["ppg_gate_mean"].to_numpy(dtype=np.float64)
        * dtp["ppg_valid_fraction"].to_numpy(dtype=np.float64)
    )
    motion_reliability = dtp["motion_valid_fraction"].to_numpy(dtype=np.float64)
    quality = np.maximum(motion_reliability, ppg_reliability)
    if not np.isfinite(quality).all():
        raise ValueError("DTP quality diagnostics contain non-finite values")
    return np.clip(quality, 0.0, 1.0)


def _candidate_identifier(parameters: dict[str, Any]) -> str:
    payload = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def gated_fusion_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    required = {
        "alpha_positive_candidates",
        "alpha_negative_candidates",
        "baseline_support_min_candidates",
        "baseline_support_max",
        "dtp_on_threshold_candidates",
        "dtp_off_threshold_candidates",
        "persistence_seconds_candidates",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Gated fusion configuration is missing: {missing}")
    quality_candidates = [bool(value) for value in config.get("quality_weighting_candidates", [False])]
    candidates: list[dict[str, Any]] = []
    identity = {
        "alpha_positive": 0.0,
        "alpha_negative": 0.0,
        "baseline_support_min": float(config["baseline_support_min_candidates"][0]),
        "baseline_support_max": float(config["baseline_support_max"]),
        "dtp_on_threshold": float(config["dtp_on_threshold_candidates"][0]),
        "dtp_off_threshold": float(config["dtp_off_threshold_candidates"][0]),
        "persistence_seconds": float(config["persistence_seconds_candidates"][0]),
        "use_quality_weighting": False,
    }
    identity["candidate_id"] = "baseline_identity"
    candidates.append(identity)
    values = product(
        config["alpha_positive_candidates"],
        config["alpha_negative_candidates"],
        config["baseline_support_min_candidates"],
        config["dtp_on_threshold_candidates"],
        config["dtp_off_threshold_candidates"],
        config["persistence_seconds_candidates"],
        quality_candidates,
    )
    for alpha_positive, alpha_negative, support_min, on_threshold, off_threshold, persistence, quality in values:
        parameters = {
            "alpha_positive": float(alpha_positive),
            "alpha_negative": float(alpha_negative),
            "baseline_support_min": float(support_min),
            "baseline_support_max": float(config["baseline_support_max"]),
            "dtp_on_threshold": float(on_threshold),
            "dtp_off_threshold": float(off_threshold),
            "persistence_seconds": float(persistence),
            "use_quality_weighting": bool(quality),
        }
        if parameters["dtp_off_threshold"] >= parameters["dtp_on_threshold"]:
            raise ValueError("dtp_off_threshold must be lower than dtp_on_threshold")
        parameters["candidate_id"] = _candidate_identifier(parameters)
        candidates.append(parameters)
    return candidates


def fuse_gated_prediction_frames(
    baseline: pd.DataFrame,
    dtp: pd.DataFrame,
    calibrator: PlattCalibrator,
    parameters: dict[str, Any],
    *,
    residual_clip: float = 2.0,
    epsilon: float = 1e-6,
    gate_ema_half_life_seconds: float = 12.0,
    event_gate: pd.DataFrame | None = None,
) -> pd.DataFrame:
    baseline_sorted, dtp_sorted = align_prediction_frames(baseline, dtp)
    output = baseline_sorted[PUBLIC_PREDICTION_COLUMNS].copy()
    alpha_positive = float(parameters["alpha_positive"])
    alpha_negative = float(parameters["alpha_negative"])
    if alpha_positive == 0.0 and alpha_negative == 0.0:
        return output
    if alpha_positive < 0 or alpha_negative < 0 or residual_clip <= 0:
        raise ValueError("Fusion weights must be non-negative and residual_clip must be positive")
    base_probability = baseline_sorted["state_probability"].to_numpy(dtype=np.float64)
    calibrated_dtp = apply_platt_calibrator(
        dtp_sorted["state_probability"].to_numpy(dtype=np.float64), calibrator, epsilon=epsilon
    )
    smoothed_dtp = _session_ema(dtp_sorted, calibrated_dtp, gate_ema_half_life_seconds)
    support = np.logical_and(
        base_probability >= float(parameters["baseline_support_min"]),
        base_probability <= float(parameters["baseline_support_max"]),
    )
    positive_gate = support & _persistent_mask(
        dtp_sorted,
        smoothed_dtp >= float(parameters["dtp_on_threshold"]),
        float(parameters["persistence_seconds"]),
    )
    negative_gate = _persistent_mask(
        dtp_sorted,
        (smoothed_dtp <= float(parameters["dtp_off_threshold"]))
        & (base_probability > smoothed_dtp),
        float(parameters["persistence_seconds"]),
    )
    quality = (
        _quality_multiplier(dtp_sorted)
        if bool(parameters.get("use_quality_weighting", False))
        else np.ones(len(dtp_sorted), dtype=np.float64)
    )
    base_logit = _logit(base_probability, epsilon)
    dtp_logit = _logit(calibrated_dtp, epsilon)
    residual = np.clip(dtp_logit - base_logit, -residual_clip, residual_clip)
    fused_logit = base_logit.copy()
    event_multiplier = np.ones(len(dtp_sorted), dtype=np.float64)
    if event_gate is not None:
        if event_gate.duplicated(TARGET_KEYS).any() or len(event_gate) != len(dtp_sorted):
            raise ValueError("DTP event gate must align one-to-one with the prediction timeline")
        aligned_gate = dtp_sorted[TARGET_KEYS].merge(
            event_gate[[*TARGET_KEYS, "event_gate"]],
            on=TARGET_KEYS,
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if aligned_gate.event_gate.isna().any():
            raise ValueError("DTP event gate is missing prediction timestamps")
        event_multiplier = aligned_gate.event_gate.to_numpy(dtype=np.float64)
        if (
            not np.isfinite(event_multiplier).all()
            or ((event_multiplier < 0) | (event_multiplier > 1)).any()
        ):
            raise ValueError("DTP event gate values must lie in [0, 1]")
    fused_logit += (
        alpha_positive * quality * positive_gate * event_multiplier * np.maximum(residual, 0.0)
    )
    fused_logit += alpha_negative * quality * negative_gate * np.minimum(residual, 0.0)
    output["state_probability"] = _sigmoid(fused_logit).astype(np.float32)
    return output


def _selection_rank(metrics: dict[str, Any], complexity: int) -> tuple[float, ...]:
    def finite(name: str, fallback: float) -> float:
        value = float(metrics.get(name, fallback))
        return value if math.isfinite(value) else fallback

    return (
        finite("f1", -math.inf),
        finite("strict_no_ignore_f1", -math.inf),
        finite("different_sensitivity", -math.inf),
        -finite("false_positives_per_observed_hour", math.inf),
        -finite("boundary_mae_seconds", math.inf),
        -float(complexity),
    )


def _complete_postprocess(
    parameters: dict[str, Any], baseline_postprocess: dict[str, Any]
) -> dict[str, Any]:
    return {
        **parameters,
        "iou_threshold": float(baseline_postprocess["iou_threshold"]),
        "matching_method": str(
            baseline_postprocess.get("matching_method", "max_cardinality_iou")
        ),
    }


def _scope_subjects(frame: pd.DataFrame) -> set[str]:
    return set(frame["subject_key"].astype(str).unique())


def _scope_record(frame: pd.DataFrame) -> dict[str, Any]:
    subjects = sorted(_scope_subjects(frame))
    return {
        "rows": len(frame),
        "subjects": subjects,
        "subject_count": len(subjects),
        "session_count": int(frame[["subject_key", "session_id"]].drop_duplicates().shape[0]),
        "timestamp_start_ms": int(frame["timestamp_ms"].min()),
        "timestamp_end_ms": int(frame["timestamp_ms"].max()),
    }


def _tune_scope(
    baseline: pd.DataFrame,
    dtp: pd.DataFrame,
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    baseline_postprocess: dict[str, Any],
    fusion_config: dict[str, Any],
    output_dir: Path,
    scope: str,
    workers: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    baseline, dtp = align_prediction_frames(baseline, dtp)
    subjects = _scope_subjects(baseline)
    truth, ignore = partition_evaluation_events(events, subjects)
    calibrator_config = fusion_config["calibration"]
    calibrator = fit_platt_calibrator(
        dtp,
        anchors[anchors["subject_key"].astype(str).isin(subjects)],
        epsilon=float(fusion_config["probability_epsilon"]),
        regularization=float(calibrator_config.get("regularization", 1e-4)),
        maximum_slope=float(calibrator_config.get("maximum_slope", 20.0)),
    )
    candidates = gated_fusion_candidates(fusion_config["gated_residual"])
    epsilon = float(fusion_config["probability_epsilon"])
    residual_clip = float(fusion_config["residual_clip"])
    gate_half_life = float(
        fusion_config["gated_residual"]["gate_ema_half_life_seconds"]
    )
    base_probability = baseline["state_probability"].to_numpy(dtype=np.float64)
    calibrated_dtp = apply_platt_calibrator(
        dtp["state_probability"].to_numpy(dtype=np.float64), calibrator, epsilon=epsilon
    )
    smoothed_dtp = _session_ema(dtp, calibrated_dtp, gate_half_life)
    base_logit = _logit(base_probability, epsilon)
    residual = np.clip(
        _logit(calibrated_dtp, epsilon) - base_logit,
        -residual_clip,
        residual_clip,
    )
    quality = (
        _quality_multiplier(dtp)
        if any(bool(candidate["use_quality_weighting"]) for candidate in candidates)
        else np.ones(len(dtp), dtype=np.float64)
    )
    positive_masks = {
        (float(threshold), float(persistence)): _persistent_mask(
            dtp, smoothed_dtp >= float(threshold), float(persistence)
        )
        for threshold in fusion_config["gated_residual"]["dtp_on_threshold_candidates"]
        for persistence in fusion_config["gated_residual"][
            "persistence_seconds_candidates"
        ]
    }
    negative_masks = {
        (float(threshold), float(persistence)): _persistent_mask(
            dtp,
            (smoothed_dtp <= float(threshold)) & (base_probability > smoothed_dtp),
            float(persistence),
        )
        for threshold in fusion_config["gated_residual"]["dtp_off_threshold_candidates"]
        for persistence in fusion_config["gated_residual"][
            "persistence_seconds_candidates"
        ]
    }

    def build_candidate(parameters: dict[str, Any]) -> pd.DataFrame:
        output = baseline[PUBLIC_PREDICTION_COLUMNS].copy()
        alpha_positive = float(parameters["alpha_positive"])
        alpha_negative = float(parameters["alpha_negative"])
        if alpha_positive == 0.0 and alpha_negative == 0.0:
            return output
        support = np.logical_and(
            base_probability >= float(parameters["baseline_support_min"]),
            base_probability <= float(parameters["baseline_support_max"]),
        )
        persistence = float(parameters["persistence_seconds"])
        positive_gate = support & positive_masks[
            (float(parameters["dtp_on_threshold"]), persistence)
        ]
        negative_gate = negative_masks[
            (float(parameters["dtp_off_threshold"]), persistence)
        ]
        candidate_quality = (
            quality
            if bool(parameters.get("use_quality_weighting", False))
            else np.ones(len(dtp), dtype=np.float64)
        )
        fused_logit = base_logit.copy()
        fused_logit += (
            alpha_positive
            * candidate_quality
            * positive_gate
            * np.maximum(residual, 0.0)
        )
        fused_logit += (
            alpha_negative
            * candidate_quality
            * negative_gate
            * np.minimum(residual, 0.0)
        )
        output["state_probability"] = _sigmoid(fused_logit).astype(np.float32)
        return output

    rows: list[dict[str, Any]] = []
    for parameters in candidates:
        fused = build_candidate(parameters)
        if "calibration_fold" in dtp:
            fused["calibration_fold"] = dtp["calibration_fold"].to_numpy()
        metrics, _ = evaluate_fusion_predictions(fused, truth, ignore, baseline_postprocess)
        rows.append(
            {
                "scope": scope,
                "candidate_id": parameters["candidate_id"],
                **{name: parameters[name] for name in FUSION_PARAMETER_NAMES},
                **metrics,
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: _selection_rank(row, int(row["candidate_id"] != "baseline_identity")),
        reverse=True,
    )
    top_count = max(1, int(fusion_config["meta_selection"].get("top_fusion_candidates", 3)))
    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    identity_parameters = candidate_by_id["baseline_identity"]
    identity_metrics, _ = evaluate_fusion_predictions(
        baseline, truth, ignore, baseline_postprocess
    )
    finalists: list[dict[str, Any]] = [
        {
            "parameters": identity_parameters,
            "postprocess": dict(baseline_postprocess),
            "metrics": identity_metrics,
            "postprocess_source": "frozen_hysteresis_control",
        }
    ]
    postprocess_root = output_dir / "postprocess_trials" / scope
    postprocess_root.mkdir(parents=True, exist_ok=True)
    fusion_finalists = [
        row for row in ranked if row["candidate_id"] != "baseline_identity"
    ][:top_count]
    for row in fusion_finalists:
        parameters = candidate_by_id[str(row["candidate_id"])]
        fused = build_candidate(parameters)
        if "calibration_fold" in dtp:
            fused["calibration_fold"] = dtp["calibration_fold"].to_numpy()
        baseline_metrics, _ = evaluate_fusion_predictions(
            fused, truth, ignore, baseline_postprocess
        )
        finalists.append(
            {
                "parameters": parameters,
                "postprocess": dict(baseline_postprocess),
                "metrics": baseline_metrics,
                "postprocess_source": "frozen_hysteresis_control",
            }
        )
        dual_search = dict(fusion_config["postprocess_search"])
        dual_search["workers"] = int(workers)
        best_dual, dual_trials = tune_dual_ema_parameters(
            fused,
            truth,
            dual_search,
            float(baseline_postprocess["iou_threshold"]),
            checkpoint_path=postprocess_root / f"{parameters['candidate_id']}.checkpoint.jsonl",
            show_progress=bool(fusion_config["meta_selection"].get("show_progress", True)),
            ignore=ignore,
            matching_method=str(
                baseline_postprocess.get("matching_method", "max_cardinality_iou")
            ),
            workers=workers,
        )
        dual_trials.to_csv(
            postprocess_root / f"{parameters['candidate_id']}.csv", index=False
        )
        dual_postprocess = _complete_postprocess(best_dual, baseline_postprocess)
        dual_metrics, _ = evaluate_fusion_predictions(fused, truth, ignore, dual_postprocess)
        finalists.append(
            {
                "parameters": parameters,
                "postprocess": dual_postprocess,
                "metrics": dual_metrics,
                "postprocess_source": "dual_ema_search",
            }
        )
    selected = max(
        finalists,
        key=lambda item: _selection_rank(
            item["metrics"],
            int(item["parameters"]["candidate_id"] != "baseline_identity")
            + int(item["postprocess_source"] == "dual_ema_search"),
        ),
    )
    return (
        {
            "scope": scope,
            "calibrator": calibrator.as_dict(),
            "parameters": selected["parameters"],
            "postprocess": selected["postprocess"],
            "postprocess_source": selected["postprocess_source"],
            "training_metrics": selected["metrics"],
            "training_baseline_metrics": identity_metrics,
        },
        pd.DataFrame(rows),
    )


def summarize_event_predictions(
    predictions: pd.DataFrame,
    predicted_events: pd.DataFrame,
    truth: pd.DataFrame,
    ignore: pd.DataFrame,
    *,
    iou_threshold: float,
    matching_method: str,
    observed_hours: float | None = None,
) -> dict[str, float]:
    metrics, matches = evaluate_events(
        truth,
        predicted_events,
        iou_threshold=iou_threshold,
        method=matching_method,
        ignore=ignore,
    )
    strict, _ = evaluate_events(
        truth,
        predicted_events,
        iou_threshold=iou_threshold,
        method=matching_method,
    )
    exposure_hours = observed_hours
    if exposure_hours is None:
        exposure_hours = 0.0
        for _, group in predictions.groupby(["subject_key", "session_id"], sort=False):
            exposure_hours += max(
                0.0,
                (float(group["timestamp_ms"].max()) - float(group["timestamp_ms"].min()) + 3000.0)
                / 3_600_000.0,
            )
    truth_relation = truth.get(
        "hand_relation", pd.Series("unknown", index=truth.index, dtype=object)
    )
    match_relation = matches.get(
        "hand_relation", pd.Series("unknown", index=matches.index, dtype=object)
    )

    def relation_sensitivity(name: str) -> float:
        count = int((truth_relation == name).sum())
        return float((match_relation == name).sum() / count) if count else float("nan")

    boundary_values = [
        float(metrics[name])
        for name in ("start_mae_seconds", "end_mae_seconds")
        if math.isfinite(float(metrics[name]))
    ]
    return {
        **{name: float(value) for name, value in metrics.items()},
        "strict_no_ignore_f1": float(strict["f1"]),
        "different_sensitivity": relation_sensitivity("different"),
        "same_sensitivity": relation_sensitivity("same"),
        "false_positives_per_observed_hour": (
            float(metrics["false_positive"]) / exposure_hours if exposure_hours else 0.0
        ),
        "boundary_mae_seconds": (
            float(np.mean(boundary_values)) if boundary_values else float("inf")
        ),
        "predicted_events": float(len(predicted_events)),
    }


def evaluate_v4_gate(
    candidate: dict[str, float],
    baseline: dict[str, float],
    gate_config: dict[str, Any],
    partition_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    fp_limit = (
        float(baseline["false_positives_per_observed_hour"])
        * float(gate_config["maximum_fp_per_hour_ratio"])
    )
    boundary_ratio = float(gate_config["maximum_boundary_mae_ratio"])
    checks = {
        "f1_improvement": float(candidate["f1"]) - float(baseline["f1"])
        >= float(gate_config["minimum_f1_improvement"]),
        "different_sensitivity_improvement": (
            float(candidate["different_sensitivity"])
            - float(baseline["different_sensitivity"])
            >= float(gate_config["minimum_different_sensitivity_improvement"])
        ),
        "strict_no_ignore_not_lower": float(candidate["strict_no_ignore_f1"])
        >= float(baseline["strict_no_ignore_f1"]),
        "fp_per_hour_within_ratio": float(candidate["false_positives_per_observed_hour"])
        <= fp_limit,
        "start_mae_within_ratio": float(candidate["start_mae_seconds"])
        <= float(baseline["start_mae_seconds"]) * boundary_ratio,
        "end_mae_within_ratio": float(candidate["end_mae_seconds"])
        <= float(baseline["end_mae_seconds"]) * boundary_ratio,
    }
    if partition_rows is not None:
        maximum_f1_drop = float(gate_config["maximum_partition_f1_drop"])
        maximum_strict_drop = float(gate_config["maximum_partition_strict_f1_drop"])
        checks["partition_f1_stability"] = all(
            float(row["candidate_metrics"]["f1"])
            >= float(row["baseline_metrics"]["f1"]) - maximum_f1_drop
            for row in partition_rows
        )
        checks["partition_strict_f1_stability"] = all(
            float(row["candidate_metrics"]["strict_no_ignore_f1"])
            >= float(row["baseline_metrics"]["strict_no_ignore_f1"])
            - maximum_strict_drop
            for row in partition_rows
        )
    return {"checks": checks, "passed": all(checks.values())}


def _metric_components(
    predictions: pd.DataFrame,
    predicted_events: pd.DataFrame,
    truth: pd.DataFrame,
    ignore: pd.DataFrame,
    subject: str,
    iou_threshold: float,
    matching_method: str,
) -> dict[str, float]:
    subject_predictions = predictions[predictions["subject_key"].astype(str) == subject]
    subject_events = predicted_events[predicted_events["subject_key"].astype(str) == subject]
    subject_truth = truth[truth["subject_key"].astype(str) == subject]
    subject_ignore = ignore[ignore["subject_key"].astype(str) == subject]
    metrics, matches = evaluate_events(
        subject_truth,
        subject_events,
        iou_threshold=iou_threshold,
        method=matching_method,
        ignore=subject_ignore,
    )
    strict, _ = evaluate_events(
        subject_truth,
        subject_events,
        iou_threshold=iou_threshold,
        method=matching_method,
    )
    exposure = 0.0
    for _, group in subject_predictions.groupby("session_id", sort=False):
        exposure += max(
            0.0,
            (float(group["timestamp_ms"].max()) - float(group["timestamp_ms"].min()) + 3000.0)
            / 3_600_000.0,
        )
    relation = matches.get(
        "hand_relation", pd.Series("unknown", index=matches.index, dtype=object)
    )
    truth_relation = subject_truth.get(
        "hand_relation", pd.Series("unknown", index=subject_truth.index, dtype=object)
    )
    true_positive = float(metrics["true_positive"])
    start_mae = float(metrics["start_mae_seconds"])
    end_mae = float(metrics["end_mae_seconds"])
    if true_positive > 0 and not all(math.isfinite(value) for value in (start_mae, end_mae)):
        raise RuntimeError("Matched subject events require finite boundary MAE values")
    return {
        "tp": true_positive,
        "fp": float(metrics["false_positive"]),
        "fn": float(metrics["false_negative"]),
        "strict_tp": float(strict["true_positive"]),
        "strict_fp": float(strict["false_positive"]),
        "strict_fn": float(strict["false_negative"]),
        "start_sum": start_mae * true_positive if true_positive > 0 else 0.0,
        "end_sum": end_mae * true_positive if true_positive > 0 else 0.0,
        "different_tp": float((relation == "different").sum()),
        "different_truth": float((truth_relation == "different").sum()),
        "exposure_hours": exposure,
    }


def paired_subject_bootstrap(
    candidate_predictions: pd.DataFrame,
    candidate_events: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    baseline_events: pd.DataFrame,
    truth: pd.DataFrame,
    ignore: pd.DataFrame,
    *,
    iou_threshold: float,
    matching_method: str,
    replicates: int = 1000,
    seed: int = 2026,
) -> dict[str, dict[str, float]]:
    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive")
    subjects = sorted(set(baseline_predictions["subject_key"].astype(str).unique()))
    if set(candidate_predictions["subject_key"].astype(str).unique()) != set(subjects):
        raise ValueError("Candidate and baseline bootstrap subjects differ")
    if not subjects:
        raise ValueError("Paired subject bootstrap requires at least one subject")
    candidate_components = [
        _metric_components(
            candidate_predictions,
            candidate_events,
            truth,
            ignore,
            subject,
            iou_threshold,
            matching_method,
        )
        for subject in subjects
    ]
    baseline_components = [
        _metric_components(
            baseline_predictions,
            baseline_events,
            truth,
            ignore,
            subject,
            iou_threshold,
            matching_method,
        )
        for subject in subjects
    ]

    def aggregate(items: list[dict[str, float]], indices: np.ndarray) -> dict[str, float]:
        totals = {name: sum(items[index][name] for index in indices) for name in items[0]}
        f1_denominator = 2 * totals["tp"] + totals["fp"] + totals["fn"]
        strict_denominator = (
            2 * totals["strict_tp"] + totals["strict_fp"] + totals["strict_fn"]
        )
        return {
            "f1": 2 * totals["tp"] / f1_denominator if f1_denominator else 0.0,
            "strict_no_ignore_f1": (
                2 * totals["strict_tp"] / strict_denominator if strict_denominator else 0.0
            ),
            "different_sensitivity": (
                totals["different_tp"] / totals["different_truth"]
                if totals["different_truth"]
                else float("nan")
            ),
            "false_positives_per_observed_hour": (
                totals["fp"] / totals["exposure_hours"] if totals["exposure_hours"] else 0.0
            ),
            "start_mae_seconds": (
                totals["start_sum"] / totals["tp"] if totals["tp"] else float("nan")
            ),
            "end_mae_seconds": (
                totals["end_sum"] / totals["tp"] if totals["tp"] else float("nan")
            ),
        }

    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {}
    for _ in range(replicates):
        indices = rng.integers(0, len(subjects), size=len(subjects))
        candidate = aggregate(candidate_components, indices)
        baseline = aggregate(baseline_components, indices)
        for name in candidate:
            delta = candidate[name] - baseline[name]
            if math.isfinite(delta):
                deltas.setdefault(name, []).append(float(delta))
    return {
        name: {
            "lower_2p5": float(np.percentile(values, 2.5)),
            "median": float(np.percentile(values, 50.0)),
            "upper_97p5": float(np.percentile(values, 97.5)),
        }
        for name, values in deltas.items()
        if values
    }


def run_meta_crossfit_selection(
    baseline_oof: pd.DataFrame,
    dtp_oof: pd.DataFrame,
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    baseline_postprocess: dict[str, Any],
    fusion_config: dict[str, Any],
    output_dir: Path,
    *,
    workers: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_oof, dtp_oof = align_prediction_frames(baseline_oof, dtp_oof)
    if "calibration_fold" not in dtp_oof:
        raise ValueError("DTP OOF predictions are missing calibration_fold")
    fold_by_subject = (
        dtp_oof[["subject_key", "calibration_fold"]]
        .drop_duplicates()
        .assign(subject_key=lambda frame: frame["subject_key"].astype(str))
    )
    if fold_by_subject["subject_key"].duplicated().any():
        raise ValueError("A DTP OOF subject appears in more than one calibration fold")
    folds = sorted(int(value) for value in fold_by_subject["calibration_fold"].unique())
    expected_partitions = int(fusion_config.get("crossfit_partitions", 3))
    if folds != list(range(expected_partitions)):
        raise ValueError(f"Expected calibration folds 0..{expected_partitions - 1}, found {folds}")
    all_subjects = _scope_subjects(baseline_oof)
    truth, ignore = partition_evaluation_events(events, all_subjects)
    meta_prediction_parts: list[pd.DataFrame] = []
    meta_event_parts: list[pd.DataFrame] = []
    partition_rows: list[dict[str, Any]] = []
    all_trials: list[pd.DataFrame] = []
    for heldout_fold in folds:
        validation_mask = dtp_oof["calibration_fold"].to_numpy(dtype=int) == heldout_fold
        training_mask = ~validation_mask
        train_baseline = baseline_oof.loc[training_mask].reset_index(drop=True)
        train_dtp = dtp_oof.loc[training_mask].reset_index(drop=True)
        validation_baseline = baseline_oof.loc[validation_mask].reset_index(drop=True)
        validation_dtp = dtp_oof.loc[validation_mask].reset_index(drop=True)
        training_subjects = _scope_subjects(train_baseline)
        validation_subjects = _scope_subjects(validation_baseline)
        if training_subjects & validation_subjects:
            raise RuntimeError("Meta calibration subjects leak into the held-out meta fold")
        selection, trials = _tune_scope(
            train_baseline,
            train_dtp,
            anchors,
            events,
            baseline_postprocess,
            fusion_config,
            output_dir,
            f"meta_holdout_{heldout_fold}",
            workers,
        )
        trials["meta_validation_fold"] = heldout_fold
        all_trials.append(trials)
        calibrator = PlattCalibrator.from_dict(selection["calibrator"])
        validation_fused = fuse_gated_prediction_frames(
            validation_baseline,
            validation_dtp,
            calibrator,
            selection["parameters"],
            residual_clip=float(fusion_config["residual_clip"]),
            epsilon=float(fusion_config["probability_epsilon"]),
            gate_ema_half_life_seconds=float(
                fusion_config["gated_residual"]["gate_ema_half_life_seconds"]
            ),
        )
        validation_fused["calibration_fold"] = heldout_fold
        validation_truth, validation_ignore = partition_evaluation_events(
            events, validation_subjects
        )
        candidate_metrics, candidate_events = evaluate_fusion_predictions(
            validation_fused,
            validation_truth,
            validation_ignore,
            selection["postprocess"],
        )
        baseline_metrics, _ = evaluate_fusion_predictions(
            validation_baseline,
            validation_truth,
            validation_ignore,
            baseline_postprocess,
        )
        meta_prediction_parts.append(validation_fused)
        meta_event_parts.append(candidate_events)
        partition_rows.append(
            {
                "heldout_calibration_fold": heldout_fold,
                "training_calibration_folds": [fold for fold in folds if fold != heldout_fold],
                "training_subjects": len(training_subjects),
                "validation_subjects": len(validation_subjects),
                "training_scope": _scope_record(train_baseline),
                "validation_scope": _scope_record(validation_baseline),
                "calibrator": selection["calibrator"],
                "parameters": selection["parameters"],
                "postprocess": selection["postprocess"],
                "postprocess_source": selection["postprocess_source"],
                "candidate_metrics": candidate_metrics,
                "baseline_metrics": baseline_metrics,
            }
        )
    meta_predictions = (
        pd.concat(meta_prediction_parts, ignore_index=True)
        .sort_values(ALIGNMENT_KEYS)
        .reset_index(drop=True)
    )
    meta_events = (
        pd.concat(meta_event_parts, ignore_index=True)
        .sort_values(["subject_key", "session_id", "start_ms", "end_ms"])
        .reset_index(drop=True)
    )
    if len(meta_predictions) != len(baseline_oof):
        raise RuntimeError("Meta OOF predictions do not cover the complete baseline timeline")
    iou_threshold = float(baseline_postprocess["iou_threshold"])
    matching_method = str(
        baseline_postprocess.get("matching_method", "max_cardinality_iou")
    )
    meta_metrics = summarize_event_predictions(
        meta_predictions,
        meta_events,
        truth,
        ignore,
        iou_threshold=iou_threshold,
        matching_method=matching_method,
    )
    baseline_metrics, baseline_events = evaluate_fusion_predictions(
        baseline_oof, truth, ignore, baseline_postprocess
    )
    gate = evaluate_v4_gate(
        meta_metrics,
        baseline_metrics,
        fusion_config["promotion_gate"],
        partition_rows,
    )
    bootstrap = paired_subject_bootstrap(
        meta_predictions,
        meta_events,
        baseline_oof,
        baseline_events,
        truth,
        ignore,
        iou_threshold=iou_threshold,
        matching_method=matching_method,
        replicates=int(fusion_config["meta_selection"].get("bootstrap_replicates", 1000)),
        seed=int(fusion_config["meta_selection"].get("bootstrap_seed", 2026)),
    )
    final_selection, final_trials = _tune_scope(
        baseline_oof,
        dtp_oof,
        anchors,
        events,
        baseline_postprocess,
        fusion_config,
        output_dir,
        "complete_oof_final",
        workers,
    )
    final_trials["meta_validation_fold"] = -1
    all_trials.append(final_trials)
    trials = pd.concat(all_trials, ignore_index=True)
    record = {
        "version": 4,
        "protocol_version": 4,
        "selection_scope": "nested_meta_crossfit_oof",
        "crossfit_partitions": expected_partitions,
        "meta_folds": partition_rows,
        "meta_oof_metrics": meta_metrics,
        "meta_oof_baseline_metrics": baseline_metrics,
        "meta_oof_gate": gate,
        "paired_subject_bootstrap": bootstrap,
        "final_calibrator": final_selection["calibrator"],
        "final_parameters": final_selection["parameters"],
        "final_postprocess": final_selection["postprocess"],
        "final_postprocess_source": final_selection["postprocess_source"],
        "final_complete_oof_metrics": final_selection["training_metrics"],
        "outer_fold_gate": {"passed": None, "checks": {}, "diagnostic_only": True},
    }
    return json_safe(record), meta_predictions, meta_events, trials


def apply_v4_selection(
    baseline: pd.DataFrame,
    dtp: pd.DataFrame,
    selection: dict[str, Any],
    fusion_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if int(selection.get("protocol_version", 0)) != 4:
        raise ValueError("Selected fusion record is not protocol v4")
    calibrator = PlattCalibrator.from_dict(selection["final_calibrator"])
    fused = fuse_gated_prediction_frames(
        baseline,
        dtp,
        calibrator,
        selection["final_parameters"],
        residual_clip=float(fusion_config["residual_clip"]),
        epsilon=float(fusion_config["probability_epsilon"]),
        gate_ema_half_life_seconds=float(
            fusion_config["gated_residual"]["gate_ema_half_life_seconds"]
        ),
    )
    return fused, dict(selection["final_postprocess"])


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)
