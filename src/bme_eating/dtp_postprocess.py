from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bme_eating.config import load_config, resolve_roots
from bme_eating.data.splits import load_subject_folds
from bme_eating.fusion import json_safe, sha256_file, validate_frozen_baseline_fold
from bme_eating.fusion_v4 import (
    PlattCalibrator,
    apply_platt_calibrator,
    fit_platt_calibrator,
    paired_subject_bootstrap,
    summarize_event_predictions,
)
from bme_eating.metrics import evaluate_events, partition_evaluation_events
from bme_eating.postprocess import causal_ema, probabilities_to_events
from bme_eating.reproducibility import require_clean_git_worktree

EVENT_COLUMNS = ["subject_key", "session_id", "start_ms", "end_ms", "score"]
CORE_COLUMNS = [
    *EVENT_COLUMNS,
    "q80_probability",
    "peak_probability",
    "support_fraction",
    "block_start_ms",
]
KEYS = ["subject_key", "session_id", "timestamp_ms"]
RUN_PATTERN = re.compile(r"dtp_postprocess_[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


@dataclass(frozen=True)
class Decoder:
    space: str
    fast_half_life: float
    slow_half_life: float
    start_mode: str
    fast_quantile: float
    slow_quantile: float
    persistence: float
    exit_ratio: float = 0.5
    off_duration: float = 60.0
    score_quantile: float | None = None
    merge_gap: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def weighted_quantile(values: np.ndarray, subjects: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    subjects = np.asarray(subjects).astype(str)
    if not len(values) or len(values) != len(subjects) or not np.isfinite(values).all():
        raise ValueError("A threshold requires finite training values and subjects")
    if not 0 <= quantile <= 1:
        raise ValueError("Quantile must be in [0, 1]")
    counts = pd.Series(subjects).value_counts()
    weights = np.asarray([1.0 / counts[subject] for subject in subjects])
    order = np.argsort(values, kind="stable")
    cutoff = quantile * weights.sum()
    position = min(
        len(order) - 1, int(np.searchsorted(np.cumsum(weights[order]), cutoff, side="left"))
    )
    return float(values[order[position]])


def _blocks(predictions: pd.DataFrame, step_seconds: float):
    if predictions.duplicated(KEYS).any():
        raise ValueError("Duplicate DTP timeline keys")
    for (subject, session), group in predictions.groupby(["subject_key", "session_id"], sort=True):
        ordered = group.sort_values("timestamp_ms", kind="stable")
        times = ordered["timestamp_ms"].to_numpy(dtype=np.int64)
        splits = np.flatnonzero(np.diff(times) > int(2 * step_seconds * 1000)) + 1
        for block in np.split(np.arange(len(ordered)), splits):
            if len(block):
                yield str(subject), str(session), ordered.iloc[block]


def _smoothed(predictions: pd.DataFrame, half_life: float, step_seconds: float) -> pd.Series:
    result = pd.Series(index=predictions.index, dtype=np.float64)
    for _, _, block in _blocks(predictions, step_seconds):
        values = block["state_probability"].to_numpy(dtype=np.float64)
        result.loc[block.index] = causal_ema(
            np.concatenate(([0.0], values)), step_seconds, half_life
        )[1:]
    return result


def prepare_probabilities(
    predictions: pd.DataFrame, space: str, calibrator: PlattCalibrator | None
) -> pd.DataFrame:
    result = predictions.copy()
    if space == "positive_slope_platt":
        if calibrator is None:
            raise ValueError("Calibrated decoding requires a train-only calibrator")
        result["state_probability"] = apply_platt_calibrator(
            result["state_probability"].to_numpy(), calibrator
        )
    elif space != "raw_control":
        raise ValueError(f"Unknown probability space: {space}")
    if not np.isfinite(result["state_probability"].to_numpy(dtype=float)).all():
        raise ValueError("DTP probabilities must be finite")
    return result


def fit_background_thresholds(
    predictions: pd.DataFrame,
    anchors: pd.DataFrame,
    decoder: Decoder,
    step_seconds: float,
) -> dict[str, float]:
    labels = anchors[[*KEYS, "state_target", "state_loss_mask"]]
    if labels.duplicated(KEYS).any():
        raise ValueError("Duplicate anchor timeline keys")
    smoothed = predictions[[*KEYS]].copy()
    smoothed["fast"] = _smoothed(predictions, decoder.fast_half_life, step_seconds)
    smoothed["slow"] = _smoothed(predictions, decoder.slow_half_life, step_seconds)
    joined = smoothed.merge(labels, on=KEYS, how="left", validate="one_to_one")
    if joined[["state_target", "state_loss_mask"]].isna().any().any():
        raise ValueError("Missing train-only anchor labels for background threshold")
    background = joined[(joined.state_loss_mask > 0) & (joined.state_target <= 0)]
    if background.empty:
        raise ValueError("No eligible negative training windows")
    subjects = background["subject_key"].to_numpy()
    return {
        "fast": weighted_quantile(background.fast.to_numpy(), subjects, decoder.fast_quantile),
        "slow": weighted_quantile(background.slow.to_numpy(), subjects, decoder.slow_quantile),
    }


def generate_core_events(
    predictions: pd.DataFrame,
    decoder: Decoder,
    thresholds: dict[str, float],
    *,
    step_seconds: float = 3.0,
    minimum_seconds: float = 30.0,
) -> pd.DataFrame:
    if decoder.start_mode not in {
        "legacy_or",
        "fast_persistent",
        "fast_persistent_with_slow_support",
    }:
        raise ValueError("Unsupported DTP start mode")
    rows: list[dict[str, Any]] = []
    duration_rows = max(1, math.ceil(decoder.persistence / step_seconds))
    off_rows = max(1, math.ceil(decoder.off_duration / step_seconds))
    for subject, session, block in _blocks(predictions, step_seconds):
        timestamps = block.timestamp_ms.to_numpy(dtype=np.int64)
        state = block.state_probability.to_numpy(dtype=float)
        fast = causal_ema(np.r_[0.0, state], step_seconds, decoder.fast_half_life)[1:]
        slow = causal_ema(np.r_[0.0, state], step_seconds, decoder.slow_half_life)[1:]
        fast_above = fast >= thresholds["fast"]
        active = False
        start = 0
        run = 0
        below = 0

        def append_core(
            start_index: int,
            end_index: int,
            timestamps=timestamps,
            state=state,
            subject=subject,
            session=session,
        ) -> None:
            if (
                end_index <= start_index
                or timestamps[end_index] - timestamps[start_index] < minimum_seconds * 1000
            ):
                return
            inside = state[start_index : end_index + 1]
            rows.append(
                {
                    "subject_key": subject,
                    "session_id": session,
                    "start_ms": int(timestamps[start_index]),
                    "end_ms": int(timestamps[end_index]),
                    "score": float(inside.mean()),
                    "q80_probability": float(np.quantile(inside, 0.8)),
                    "peak_probability": float(inside.max()),
                    "support_fraction": float((inside >= thresholds["fast"]).mean()),
                    "block_start_ms": int(timestamps[0]),
                }
            )

        for index in range(len(timestamps)):
            run = run + 1 if fast_above[index] else 0
            if decoder.start_mode == "legacy_or":
                recent = fast_above[max(0, index - 2) : index + 1]
                starts = np.count_nonzero(recent) >= 2 or slow[index] >= thresholds["slow"]
            else:
                starts = run >= duration_rows and (
                    decoder.start_mode == "fast_persistent" or slow[index] >= thresholds["slow"]
                )
            if not active and starts:
                start = max(0, index - run + 1) if run >= duration_rows else index
                active = True
                below = 0
            if not active:
                continue
            below = (
                below + 1
                if fast[index] < thresholds["fast"] * decoder.exit_ratio
                and slow[index] < thresholds["slow"] * decoder.exit_ratio
                else 0
            )
            if below >= off_rows:
                append_core(start, max(start, index - below))
                active = False
                below = 0
        if active:
            append_core(start, len(timestamps) - 1)
    return pd.DataFrame(rows, columns=CORE_COLUMNS)


def fit_score_threshold(core: pd.DataFrame, quantile: float | None) -> float | None:
    if quantile is None:
        return None
    if core.empty:
        return math.inf
    return weighted_quantile(core.score.to_numpy(), core.subject_key.to_numpy(), float(quantile))


def filter_and_merge(core: pd.DataFrame, threshold: float | None, merge_gap: float) -> pd.DataFrame:
    if merge_gap < 0:
        raise ValueError("Merge gap must be non-negative")
    accepted = core if threshold is None else core[core.score >= threshold]
    if accepted.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    rows: list[dict[str, Any]] = []
    grouping = ["subject_key", "session_id"]
    if "block_start_ms" in accepted.columns:
        grouping.append("block_start_ms")
    for group_key, group in accepted.groupby(grouping, sort=True):
        subject, session = group_key[:2]
        previous: dict[str, Any] | None = None
        mass = 0.0
        duration = 0.0
        for event in group.sort_values("start_ms").itertuples(index=False):
            event_duration = max(1, int(event.end_ms) - int(event.start_ms))
            if (
                previous is not None
                and merge_gap > 0
                and int(event.start_ms) - previous["end_ms"] <= merge_gap * 1000
            ):
                previous["end_ms"] = max(previous["end_ms"], int(event.end_ms))
                mass += float(event.score) * event_duration
                duration += event_duration
                previous["score"] = mass / duration
            else:
                if previous is not None:
                    rows.append(previous)
                previous = {
                    "subject_key": subject,
                    "session_id": session,
                    "start_ms": int(event.start_ms),
                    "end_ms": int(event.end_ms),
                    "score": float(event.score),
                }
                mass = float(event.score) * event_duration
                duration = float(event_duration)
        if previous is not None:
            rows.append(previous)
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def event_gate(
    predictions: pd.DataFrame, events: pd.DataFrame, training_scores: pd.DataFrame
) -> pd.DataFrame:
    gate = predictions[KEYS].copy()
    gate["event_gate"] = 0.0
    if len(events) and not len(training_scores):
        raise ValueError("Event gate needs train-only event score reference")
    values = training_scores.score.to_numpy(dtype=float)
    subjects = training_scores.subject_key.astype(str).to_numpy()
    counts = pd.Series(subjects).value_counts()
    weights = np.asarray([1.0 / counts[subject] for subject in subjects])
    for event in events.itertuples(index=False):
        mask = (
            (gate.subject_key == event.subject_key)
            & (gate.session_id == event.session_id)
            & gate.timestamp_ms.between(event.start_ms, event.end_ms)
        )
        percentile = float(weights[values <= event.score].sum() / weights.sum())
        gate.loc[mask, "event_gate"] = np.maximum(gate.loc[mask, "event_gate"], percentile)
    return gate


def _metrics(
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    truth: pd.DataFrame,
    ignore: pd.DataFrame,
    iou: float,
    method: str,
    *,
    observed_hours: float | None = None,
) -> dict[str, float]:
    return summarize_event_predictions(
        predictions,
        events,
        truth,
        ignore,
        iou_threshold=iou,
        matching_method=method,
        observed_hours=observed_hours,
    )


def _observed_hours(predictions: pd.DataFrame) -> float:
    bounds = predictions.groupby(["subject_key", "session_id"], sort=False).timestamp_ms.agg(
        ["min", "max"]
    )
    return float(((bounds["max"] - bounds["min"] + 3000).clip(lower=0) / 3_600_000).sum())


def _rank(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row.get("minimum_fold_f1", row["f1"])),
        float(row.get("mean_fold_f1", row["f1"])),
        float(row["strict_no_ignore_f1"]),
        -float(row["false_positives_per_observed_hour"]),
    )


def module_gate(
    child: dict[str, float],
    parent: dict[str, float],
    child_folds: list[dict[str, float]],
    parent_folds: list[dict[str, float]],
    config: dict[str, Any],
) -> dict[str, Any]:
    gain = child["f1"] - parent["f1"]
    fp_base = parent["false_positives_per_observed_hour"]
    checks = {
        "f1": gain >= config["module_minimum_f1_improvement"],
        "fp_or_large_f1": (
            child["false_positives_per_observed_hour"]
            <= fp_base * (1 - config["module_minimum_fp_reduction"])
            or gain >= config["module_minimum_large_f1_improvement"]
        ),
        "different_sensitivity": (
            child["different_sensitivity"]
            >= parent["different_sensitivity"] - config["module_maximum_different_sensitivity_drop"]
        ),
        "strict_f1": (
            child["strict_no_ignore_f1"]
            >= parent["strict_no_ignore_f1"] - config["module_maximum_strict_f1_drop"]
        ),
        "partition_f1": all(
            newer["f1"] >= older["f1"] - config["module_maximum_partition_f1_drop"]
            for newer, older in zip(child_folds, parent_folds, strict=True)
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "delta_f1": gain}


def _sha_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(json_safe(payload), sort_keys=True).encode()).hexdigest()


def _git_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    )
    return {"commit": head, "dirty": dirty}


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _score_series(predictions: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    labels = anchors[[*KEYS, "state_target", "state_loss_mask"]]
    joined = predictions[KEYS + ["state_probability"]].merge(
        labels, on=KEYS, how="left", validate="one_to_one"
    )
    if joined[["state_target", "state_loss_mask"]].isna().any().any():
        raise ValueError("Window calibration has missing or duplicate target keys")
    return joined[joined.state_loss_mask > 0]


def _calibration_losses(
    frame: pd.DataFrame, space: str, calibrator: PlattCalibrator | None
) -> dict[str, float]:
    probabilities = prepare_probabilities(frame, space, calibrator).state_probability.to_numpy(
        dtype=float
    )
    targets = frame.state_target.to_numpy(dtype=float)
    subjects = frame.subject_key.astype(str).to_numpy()
    weights = pd.Series(subjects).map(pd.Series(subjects).value_counts()).to_numpy(dtype=float)
    weights = 1.0 / weights
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return {
        "brier": float(np.average((targets - probabilities) ** 2, weights=weights)),
        "log_loss": float(
            np.average(
                -targets * np.log(probabilities) - (1 - targets) * np.log1p(-probabilities),
                weights=weights,
            )
        ),
    }


def _base_decoders(config: dict[str, Any]) -> list[Decoder]:
    return [
        Decoder(
            str(space),
            float(fast),
            float(slow),
            str(mode),
            float(fast_q),
            float(slow_q),
            float(persistence),
        )
        for space, fast, slow, mode, fast_q, slow_q, persistence in product(
            config["probability_spaces"],
            config["fast_ema_half_life_seconds"],
            config["slow_ema_half_life_seconds"],
            config["start_modes"],
            config["fast_background_quantiles"],
            config["slow_background_quantiles"],
            config["persistence_seconds"],
        )
    ]


def _decoder_events(
    predictions: pd.DataFrame,
    decoder: Decoder,
    thresholds: dict[str, float],
    score: float | None,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    core = generate_core_events(
        predictions,
        decoder,
        thresholds,
        step_seconds=float(config["output_step_seconds"]),
        minimum_seconds=float(config["minimum_event_seconds"]),
    )
    return core, filter_and_merge(core, score, decoder.merge_gap)


def _search_scope(
    predictions: pd.DataFrame,
    anchors: pd.DataFrame,
    truth: pd.DataFrame,
    ignore: pd.DataFrame,
    config: dict[str, Any],
    iou: float,
    method: str,
    xgb_reference: dict[str, float],
    allowed_spaces: set[str],
    checkpoint: Path,
    scope: str,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    calibrator = (
        fit_platt_calibrator(predictions, anchors)
        if "positive_slope_platt" in allowed_spaces
        else None
    )
    prepared = {
        space: prepare_probabilities(predictions, space, calibrator) for space in allowed_spaces
    }
    observed_hours = _observed_hours(predictions)
    fold_scopes: dict[int, tuple[set[str], pd.DataFrame, pd.DataFrame, float]] = {}
    for partition in sorted(predictions.calibration_fold.unique()):
        subjects = set(
            predictions.loc[predictions.calibration_fold == partition, "subject_key"].astype(str)
        )
        partition_predictions = predictions[predictions.subject_key.astype(str).isin(subjects)]
        fold_scopes[int(partition)] = (
            subjects,
            truth[truth.subject_key.astype(str).isin(subjects)],
            ignore[ignore.subject_key.astype(str).isin(subjects)],
            _observed_hours(partition_predictions),
        )
    cached_rows: dict[str, dict[str, Any]] = {}
    if checkpoint.is_file():
        contents = checkpoint.read_text(encoding="utf-8")
        lines = contents.splitlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise RuntimeError("Search checkpoint is corrupt before its last line") from None
                checkpoint.write_text(
                    "\n".join(lines[:index]) + ("\n" if index else ""), encoding="utf-8"
                )
                break
            if row["scope"] == scope:
                cached_rows[row["key"]] = row
        else:
            if contents and not contents.endswith("\n"):
                with checkpoint.open("a", encoding="utf-8") as handle:
                    handle.write("\n")

    def evaluate(
        decoder: Decoder,
        thresholds: dict[str, float],
        core: pd.DataFrame,
        threshold: float | None,
        *,
        fold_detail: bool = False,
    ) -> dict[str, Any]:
        key = _sha_payload({"scope": scope, "decoder": decoder.as_dict()})
        if key in cached_rows and (not fold_detail or "minimum_fold_f1" in cached_rows[key]):
            return cached_rows[key]
        predicted = filter_and_merge(core, threshold, decoder.merge_gap)
        metrics = _metrics(
            prepared[decoder.space],
            predicted,
            truth,
            ignore,
            iou,
            method,
            observed_hours=observed_hours,
        )
        row = {
            "scope": scope,
            "key": key,
            "decoder": decoder.as_dict(),
            "thresholds": thresholds,
            "score_threshold": threshold,
            **metrics,
        }
        if fold_detail:
            fold_scores = []
            for subjects, fold_truth, fold_ignore, fold_hours in fold_scopes.values():
                fold_metrics = _metrics(
                    prepared[decoder.space][
                        prepared[decoder.space].subject_key.astype(str).isin(subjects)
                    ],
                    predicted[predicted.subject_key.astype(str).isin(subjects)],
                    fold_truth,
                    fold_ignore,
                    iou,
                    method,
                    observed_hours=fold_hours,
                )
                fold_scores.append(fold_metrics["f1"])
            row["minimum_fold_f1"] = float(min(fold_scores))
            row["mean_fold_f1"] = float(np.mean(fold_scores))
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
        cached_rows[key] = row
        return row

    score_quantiles = config["score_quantiles"]
    base_decoders = [
        decoder for decoder in _base_decoders(config) if decoder.space in allowed_spaces
    ]
    threshold_cache: dict[tuple[Any, ...], dict[str, float]] = {}

    def thresholds_for(decoder: Decoder) -> dict[str, float]:
        key = (
            decoder.space,
            decoder.fast_half_life,
            decoder.slow_half_life,
            decoder.fast_quantile,
            decoder.slow_quantile,
        )
        if key not in threshold_cache:
            threshold_cache[key] = fit_background_thresholds(
                prepared[decoder.space], anchors, decoder, config["output_step_seconds"]
            )
        return threshold_cache[key]

    stage1: list[tuple[tuple[float, ...], Decoder]] = []
    for index, base in enumerate(base_decoders, start=1):
        frame = prepared[base.space]
        thresholds = thresholds_for(base)
        core = generate_core_events(
            frame,
            base,
            thresholds,
            step_seconds=config["output_step_seconds"],
            minimum_seconds=config["minimum_event_seconds"],
        )
        best: dict[str, Any] | None = None
        for quantile, merge_gap in product(score_quantiles, (0, 60)):
            decoder = Decoder(
                **{**base.as_dict(), "score_quantile": quantile, "merge_gap": merge_gap}
            )
            row = evaluate(decoder, thresholds, core, fit_score_threshold(core, quantile))
            if best is None or _rank(row) > _rank(best):
                best = row
        if best is not None:
            stage1.append((_rank(best), base))
        if index % 16 == 0 or index == len(base_decoders):
            print(f"{scope} stage 1: {index}/{len(base_decoders)} generators", flush=True)
    stage1.sort(key=lambda pair: pair[0], reverse=True)
    finalists: list[dict[str, Any]] = []
    for finalist_index, (_, base) in enumerate(stage1[: int(config["stage1_keep"])], start=1):
        frame = prepared[base.space]
        thresholds = thresholds_for(base)
        for ratio, off in product(config["exit_ratios"], config["off_duration_seconds"]):
            adjusted = Decoder(
                **{**base.as_dict(), "exit_ratio": float(ratio), "off_duration": float(off)}
            )
            core = generate_core_events(
                frame,
                adjusted,
                thresholds,
                step_seconds=config["output_step_seconds"],
                minimum_seconds=config["minimum_event_seconds"],
            )
            for quantile, gap in product(score_quantiles, config["merge_gap_seconds"]):
                decoder = Decoder(
                    **{**adjusted.as_dict(), "score_quantile": quantile, "merge_gap": float(gap)}
                )
                finalists.append(
                    evaluate(
                        decoder,
                        thresholds,
                        core,
                        fit_score_threshold(core, quantile),
                        fold_detail=True,
                    )
                )
        print(f"{scope} stage 2: {finalist_index}/{config['stage1_keep']} generators", flush=True)
    finalists.sort(key=_rank, reverse=True)
    top = finalists[: int(config["final_keep"])]
    for candidate in finalists:
        if (
            candidate["false_positives_per_observed_hour"]
            > xgb_reference["false_positives_per_observed_hour"] * config["maximum_xgb_fp_ratio"]
        ):
            continue
        if candidate["strict_no_ignore_f1"] < xgb_reference["strict_no_ignore_f1"]:
            continue
        sensitivity = candidate["different_sensitivity"]
        if (
            sensitivity is None
            or not math.isfinite(float(sensitivity))
            or sensitivity
            < xgb_reference["different_sensitivity"] + config["minimum_different_sensitivity_gain"]
        ):
            continue
        if candidate not in top:
            top.append(candidate)
        break
    return top, pd.DataFrame(cached_rows.values())


def _apply_decoder(
    predictions: pd.DataFrame,
    anchors: pd.DataFrame,
    decoder: Decoder,
    config: dict[str, Any],
    calibrator: PlattCalibrator | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prepared = prepare_probabilities(predictions, decoder.space, calibrator)
    thresholds = fit_background_thresholds(
        prepared, anchors, decoder, config["output_step_seconds"]
    )
    core = generate_core_events(
        prepared,
        decoder,
        thresholds,
        step_seconds=config["output_step_seconds"],
        minimum_seconds=config["minimum_event_seconds"],
    )
    threshold = fit_score_threshold(core, decoder.score_quantile)
    return (
        prepared,
        core,
        filter_and_merge(core, threshold, decoder.merge_gap),
        {
            "thresholds": thresholds,
            "score_threshold": threshold,
            "score_reference": core[["subject_key", "score"]].to_dict("records"),
            "calibrator": calibrator.as_dict() if calibrator else None,
        },
    )


def _decode_validation(
    predictions: pd.DataFrame,
    decoder: Decoder,
    fitted: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calibrator = PlattCalibrator.from_dict(fitted["calibrator"]) if fitted["calibrator"] else None
    prepared = prepare_probabilities(predictions, decoder.space, calibrator)
    core, events = _decoder_events(
        prepared, decoder, fitted["thresholds"], fitted["score_threshold"], config
    )
    return prepared, core, events


def _legacy_control(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    return probabilities_to_events(
        predictions,
        detector_mode="dual_ema",
        fast_ema_half_life_seconds=12,
        slow_ema_half_life_seconds=36,
        fast_high_threshold=0.55,
        slow_high_threshold=0.30,
        exit_threshold_ratio=0.25,
        off_duration_seconds=60,
        minimum_event_seconds=30,
        merge_gap_seconds=60,
        boundary_lookback_seconds=60,
    )


def _scope_metrics(
    predictions: pd.DataFrame,
    candidate: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, Any],
    iou: float,
    method: str,
) -> dict[str, float]:
    subjects = set(predictions.subject_key.astype(str))
    truth, ignore = partition_evaluation_events(events, subjects)
    return _metrics(predictions, candidate, truth, ignore, iou, method)


def _event_diagnostics(
    predictions: pd.DataFrame,
    predicted: pd.DataFrame,
    truth: pd.DataFrame,
    ignore: pd.DataFrame,
    iou: float,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    matches = evaluate_events(truth, predicted, iou, method, ignore=ignore)[1]
    matched_truth = set(
        zip(
            matches.get("subject_key", []),
            matches.get("truth_start_ms", []),
            matches.get("truth_end_ms", []),
        )
    )
    matched_prediction = set(
        zip(
            matches.get("subject_key", []),
            matches.get("prediction_start_ms", []),
            matches.get("prediction_end_ms", []),
        )
    )
    failures = [
        {
            "type": "false_negative",
            "subject_key": row.subject_key,
            "start_ms": row.start_ms,
            "end_ms": row.end_ms,
        }
        for row in truth.itertuples(index=False)
        if (row.subject_key, row.start_ms, row.end_ms) not in matched_truth
    ] + [
        {
            "type": "unmatched_prediction",
            "subject_key": row.subject_key,
            "start_ms": row.start_ms,
            "end_ms": row.end_ms,
        }
        for row in predicted.itertuples(index=False)
        if (row.subject_key, row.start_ms, row.end_ms) not in matched_prediction
    ]
    by_subject = pd.DataFrame(
        [
            {
                "subject_key": subject,
                **_metrics(
                    predictions[predictions.subject_key == subject],
                    predicted[predicted.subject_key == subject],
                    truth[truth.subject_key == subject],
                    ignore[ignore.subject_key == subject],
                    iou,
                    method,
                ),
            }
            for subject in sorted(set(predictions.subject_key.astype(str)))
        ]
    )
    by_hand = {
        relation: {
            "truth_events": int((truth.hand_relation == relation).sum()),
            "matched_events": int(
                (matches.get("hand_relation", pd.Series(dtype=str)) == relation).sum()
            ),
        }
        for relation in ("same", "different", "unknown")
    }
    for counts in by_hand.values():
        counts["sensitivity"] = (
            counts["matched_events"] / counts["truth_events"]
            if counts["truth_events"]
            else None
        )
    return (
        pd.DataFrame(
            failures, columns=["type", "subject_key", "start_ms", "end_ms"]
        ),
        by_subject,
        by_hand,
    )


def _ablation(
    train: pd.DataFrame,
    anchors: pd.DataFrame,
    decoder: Decoder,
    config: dict[str, Any],
    fitted_calibrator: PlattCalibrator | None,
    truth: pd.DataFrame,
    ignore: pd.DataFrame,
    iou: float,
    method: str,
) -> tuple[Decoder, list[dict[str, Any]]]:
    current = Decoder(**decoder.as_dict())
    history: list[dict[str, Any]] = []
    changes = (
        ("start_mode", "legacy_or"),
        ("score_quantile", None),
        ("merge_gap", 0.0),
    )

    def evaluate(option: Decoder) -> tuple[dict[str, float], list[dict[str, float]]]:
        prepared, _, detected, _ = _apply_decoder(train, anchors, option, config, fitted_calibrator)
        fold_metrics = []
        for partition in sorted(train.calibration_fold.unique()):
            subjects = set(
                train.loc[train.calibration_fold == partition, "subject_key"].astype(str)
            )
            fold_metrics.append(
                _metrics(
                    prepared[prepared.subject_key.astype(str).isin(subjects)],
                    detected[detected.subject_key.astype(str).isin(subjects)],
                    truth[truth.subject_key.astype(str).isin(subjects)],
                    ignore[ignore.subject_key.astype(str).isin(subjects)],
                    iou,
                    method,
                )
            )
        return _metrics(prepared, detected, truth, ignore, iou, method), fold_metrics

    for module, parent_value in changes:
        if getattr(current, module) == parent_value:
            continue
        parent = Decoder(**{**current.as_dict(), module: parent_value})
        parent_metrics, parent_folds = evaluate(parent)
        child_metrics, child_folds = evaluate(current)
        gate = module_gate(child_metrics, parent_metrics, child_folds, parent_folds, config)
        history.append(
            {
                "module": module,
                "parent": parent.as_dict(),
                "child": current.as_dict(),
                "parent_metrics": parent_metrics,
                "child_metrics": child_metrics,
                "gate": gate,
            }
        )
        if not gate["passed"]:
            current = parent
    return current, history


def _choose_working_points(
    trials: list[dict[str, Any]], xgb: dict[str, float], config: dict[str, Any]
) -> dict[str, dict[str, Any] | None]:
    if not trials:
        raise RuntimeError("No train-only DTP candidates survived the search")
    best_mean = max(float(row.get("mean_fold_f1", row["f1"])) for row in trials)
    robust = max(
        (row for row in trials if row.get("mean_fold_f1", row["f1"]) >= best_mean - 0.01),
        key=lambda row: (
            row.get("minimum_fold_f1", row["f1"]),
            -row["false_positives_per_observed_hour"],
            row["strict_no_ignore_f1"],
            -int(row["decoder"]["start_mode"] != "legacy_or")
            - int(row["decoder"]["score_quantile"] is not None)
            - int(row["decoder"]["merge_gap"] > 0),
        ),
    )
    feasible = [
        row
        for row in trials
        if row["false_positives_per_observed_hour"]
        <= xgb["false_positives_per_observed_hour"] * config["maximum_xgb_fp_ratio"]
        and row["strict_no_ignore_f1"] >= xgb["strict_no_ignore_f1"]
        and row["different_sensitivity"] is not None
        and row["different_sensitivity"]
        >= xgb["different_sensitivity"] + config["minimum_different_sensitivity_gain"]
    ]
    return {
        "robust_f1": robust,
        "low_false_positive": max(feasible, key=_rank) if feasible else None,
    }


def _point_checks(
    name: str,
    metrics: dict[str, float],
    legacy: dict[str, float],
    xgb: dict[str, float],
    config: dict[str, Any],
) -> dict[str, bool]:
    checks = {
        "f1_over_legacy": metrics["f1"] >= legacy["f1"] + config["minimum_f1_improvement"]
    }
    if name == "low_false_positive":
        checks.update(
            {
                "fp_hour": metrics["false_positives_per_observed_hour"]
                <= xgb["false_positives_per_observed_hour"] * config["maximum_xgb_fp_ratio"],
                "strict_f1": metrics["strict_no_ignore_f1"] >= xgb["strict_no_ignore_f1"],
                "different_sensitivity": metrics["different_sensitivity"]
                >= xgb["different_sensitivity"] + config["minimum_different_sensitivity_gain"],
            }
        )
    return checks


def correct_boundaries(
    predictions: pd.DataFrame, events: pd.DataFrame, lookback_seconds: float = 60.0
) -> pd.DataFrame:
    corrected = events.copy()
    for subject, session, ordered in _blocks(predictions, 3.0):
        times = ordered.timestamp_ms.to_numpy(dtype=np.int64)
        for index in corrected.index[
            (corrected.subject_key == subject)
            & (corrected.session_id == session)
            & (corrected.start_ms >= times[0])
            & (corrected.end_ms <= times[-1])
        ]:
            start = int(corrected.at[index, "start_ms"])
            end = int(corrected.at[index, "end_ms"])
            start_mask = (times >= start - lookback_seconds * 1000) & (
                times <= min(end - 1, start + lookback_seconds * 1000)
            )
            end_mask = (times >= max(start + 1, end - lookback_seconds * 1000)) & (
                times <= end + lookback_seconds * 1000
            )
            if not start_mask.any() or not end_mask.any():
                continue
            start_probability = ordered.start_probability.to_numpy(dtype=float)
            end_probability = ordered.end_probability.to_numpy(dtype=float)
            candidate_start = int(times[start_mask][np.argmax(start_probability[start_mask])])
            candidate_end = int(times[end_mask][np.argmax(end_probability[end_mask])])
            if candidate_start < candidate_end:
                corrected.at[index, "start_ms"] = candidate_start
                corrected.at[index, "end_ms"] = candidate_end
    return corrected


def _boundary_gate(
    predictions: pd.DataFrame,
    predicted: pd.DataFrame,
    events: pd.DataFrame,
    folds: pd.DataFrame,
    iou: float,
    method: str,
) -> tuple[bool, dict[str, Any], pd.DataFrame]:
    adjusted = correct_boundaries(predictions, predicted)
    subjects = set(predictions.subject_key.astype(str))
    truth, ignore = partition_evaluation_events(events, subjects)
    old = _metrics(predictions, predicted, truth, ignore, iou, method)
    new = _metrics(predictions, adjusted, truth, ignore, iou, method)
    checks = {
        "start_mae": math.isfinite(new["start_mae_seconds"])
        and new["start_mae_seconds"] <= old["start_mae_seconds"] * 0.95,
        "end_mae": math.isfinite(new["end_mae_seconds"])
        and new["end_mae_seconds"] <= old["end_mae_seconds"] * 0.95,
        "f1": new["f1"] >= old["f1"] - 0.005,
        "strict_f1": new["strict_no_ignore_f1"] >= old["strict_no_ignore_f1"] - 0.005,
    }
    for fold in sorted(folds.calibration_fold.unique()):
        members = set(folds.loc[folds.calibration_fold == fold, "subject_key"].astype(str))
        fold_predictions = predictions[predictions.subject_key.astype(str).isin(members)]
        fold_truth, fold_ignore = partition_evaluation_events(events, members)
        before = _metrics(
            fold_predictions,
            predicted[predicted.subject_key.astype(str).isin(members)],
            fold_truth,
            fold_ignore,
            iou,
            method,
        )
        after = _metrics(
            fold_predictions,
            adjusted[adjusted.subject_key.astype(str).isin(members)],
            fold_truth,
            fold_ignore,
            iou,
            method,
        )
        checks[f"fold_{fold}_f1"] = after["f1"] >= before["f1"] - 0.005
        checks[f"fold_{fold}_strict"] = (
            after["strict_no_ignore_f1"] >= before["strict_no_ignore_f1"] - 0.005
        )
        checks[f"fold_{fold}_start_mae"] = (
            math.isfinite(after["start_mae_seconds"])
            and after["start_mae_seconds"] <= before["start_mae_seconds"] * 0.95
        )
        checks[f"fold_{fold}_end_mae"] = (
            math.isfinite(after["end_mae_seconds"])
            and after["end_mae_seconds"] <= before["end_mae_seconds"] * 0.95
        )
    return all(checks.values()), {"checks": checks, "before": old, "after": new}, adjusted


def _inner_calibration_gate(
    train: pd.DataFrame,
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, Any],
    iou: float,
    method: str,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    reference = Decoder("raw_control", 12, 36, "fast_persistent", 0.975, 0.95, 6)
    for partition in sorted(train.calibration_fold.unique()):
        fitting = train[train.calibration_fold != partition]
        validation = train[train.calibration_fold == partition]
        fitting_subjects = set(fitting.subject_key.astype(str))
        validation_subjects = set(validation.subject_key.astype(str))
        if fitting_subjects & validation_subjects:
            raise RuntimeError("Calibration gate overlaps its internal validation subjects")
        fitting_anchors = anchors[anchors.subject_key.astype(str).isin(fitting_subjects)]
        validation_anchors = anchors[anchors.subject_key.astype(str).isin(validation_subjects)]
        try:
            calibrator = fit_platt_calibrator(fitting, fitting_anchors)
            windows = _score_series(validation, validation_anchors)
            losses = {
                space: _calibration_losses(windows, space, calibrator)
                for space in ("raw_control", "positive_slope_platt")
            }
            _, _, _, raw_fit = _apply_decoder(fitting, fitting_anchors, reference, config, None)
            _, _, raw_events = _decode_validation(validation, reference, raw_fit, config)
            calibrated = Decoder(**{**reference.as_dict(), "space": "positive_slope_platt"})
            _, _, _, cal_fit = _apply_decoder(
                fitting, fitting_anchors, calibrated, config, calibrator
            )
            _, _, cal_events = _decode_validation(validation, calibrated, cal_fit, config)
            truth, ignore = partition_evaluation_events(events, validation_subjects)
            scores = {
                "raw": _metrics(validation, raw_events, truth, ignore, iou, method),
                "calibrated": _metrics(validation, cal_events, truth, ignore, iou, method),
            }
        except (ValueError, RuntimeError) as error:
            return {"passed": False, "reason": str(error), "inner_folds": comparisons}
        comparisons.append(
            {
                "partition": int(partition),
                "fit_subjects": sorted(fitting_subjects),
                "validation_subjects": sorted(validation_subjects),
                "losses": losses,
                "event_control": scores,
            }
        )
    brier_raw = np.mean([item["losses"]["raw_control"]["brier"] for item in comparisons])
    brier_cal = np.mean([item["losses"]["positive_slope_platt"]["brier"] for item in comparisons])
    log_raw = np.mean([item["losses"]["raw_control"]["log_loss"] for item in comparisons])
    log_cal = np.mean([item["losses"]["positive_slope_platt"]["log_loss"] for item in comparisons])
    checks = {
        "brier_improves_one_percent": brier_cal <= 0.99 * brier_raw,
        "log_loss_not_worse": log_cal <= log_raw,
        "each_inner_fold_f1_stable": all(
            item["event_control"]["calibrated"]["f1"] >= item["event_control"]["raw"]["f1"] - 0.01
            for item in comparisons
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "inner_folds": comparisons}


def _crossfit(
    oof: pd.DataFrame,
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    baseline: pd.DataFrame,
    baseline_postprocess: dict[str, Any],
    config: dict[str, Any],
    checkpoint: Path,
    workers: int = 1,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], pd.DataFrame]:
    from bme_eating.fusion import evaluate_fusion_predictions

    folds = oof[["subject_key", "calibration_fold"]].drop_duplicates()
    if folds.subject_key.duplicated().any() or sorted(folds.calibration_fold.unique()) != [0, 1, 2]:
        raise ValueError("DTP OOF requires exactly three subject-disjoint calibration folds")
    iou = float(baseline_postprocess["iou_threshold"])
    method = str(baseline_postprocess.get("matching_method", "max_cardinality_iou"))
    scopes: list[dict[str, Any]] = []
    for heldout in range(3):
        train = oof[oof.calibration_fold != heldout].copy()
        validation = oof[oof.calibration_fold == heldout].copy()
        train_subjects = set(train.subject_key.astype(str))
        heldout_subjects = set(validation.subject_key.astype(str))
        if not train_subjects or not heldout_subjects or train_subjects & heldout_subjects:
            raise RuntimeError("Meta train and validation subjects overlap or are empty")
        train_anchors = anchors[anchors.subject_key.astype(str).isin(train_subjects)]
        train_truth, train_ignore = partition_evaluation_events(events, train_subjects)
        validation_truth, validation_ignore = partition_evaluation_events(events, heldout_subjects)
        calibration_gate = _inner_calibration_gate(
            train, train_anchors, events, config, iou, method
        )
        calibrator = (
            fit_platt_calibrator(train, train_anchors) if calibration_gate["passed"] else None
        )
        xgb_train = baseline[baseline.subject_key.astype(str).isin(train_subjects)]
        xgb_metrics, _ = evaluate_fusion_predictions(
            xgb_train, train_truth, train_ignore, baseline_postprocess
        )
        xgb_validation = baseline[baseline.subject_key.astype(str).isin(heldout_subjects)]
        xgb_validation_metrics, _ = evaluate_fusion_predictions(
            xgb_validation, validation_truth, validation_ignore, baseline_postprocess
        )
        scopes.append(
            {
                "fold": heldout,
                "train": train,
                "validation": validation,
                "train_anchors": train_anchors,
                "truth": train_truth,
                "ignore": train_ignore,
                "validation_truth": validation_truth,
                "validation_ignore": validation_ignore,
                "train_subjects": sorted(train_subjects),
                "heldout_subjects": sorted(heldout_subjects),
                "calibrator": calibrator,
                "calibration_gate": calibration_gate,
                "xgb_train_metrics": xgb_metrics,
                "xgb_validation_metrics": xgb_validation_metrics,
            }
        )

    selected: dict[str, list[dict[str, Any]]] = {"robust_f1": [], "low_false_positive": []}
    trials: list[pd.DataFrame] = []

    def search(scope: dict[str, Any]) -> tuple[list[dict[str, Any]], pd.DataFrame]:
        return _search_scope(
            scope["train"],
            scope["train_anchors"],
            scope["truth"],
            scope["ignore"],
            config,
            iou,
            method,
            scope["xgb_train_metrics"],
            {"raw_control", "positive_slope_platt"}
            if scope["calibration_gate"]["passed"]
            else {"raw_control"},
            checkpoint.with_name(f"search.heldout_{scope['fold']}.checkpoint.jsonl"),
            f"heldout_{scope['fold']}",
        )

    with ThreadPoolExecutor(max_workers=min(3, max(1, workers))) as executor:
        search_results = list(executor.map(search, scopes))
    for scope, (options, table) in zip(scopes, search_results, strict=True):
        trials.append(table)
        choices = _choose_working_points(options, scope["xgb_train_metrics"], config)
        scope["choices"] = choices
        for name, candidate in choices.items():
            if candidate is None:
                continue
            decoder = Decoder(**candidate["decoder"])
            calibrator = scope["calibrator"] if decoder.space == "positive_slope_platt" else None
            accepted_decoder, ablations = _ablation(
                scope["train"],
                scope["train_anchors"],
                decoder,
                config,
                calibrator,
                scope["truth"],
                scope["ignore"],
                iou,
                method,
            )
            _, _, _, fitted = _apply_decoder(
                scope["train"], scope["train_anchors"], accepted_decoder, config, calibrator
            )
            transformed, _, predicted = _decode_validation(
                scope["validation"], accepted_decoder, fitted, config
            )
            accepted_metrics = _metrics(
                transformed,
                predicted,
                scope["validation_truth"],
                scope["validation_ignore"],
                iou,
                method,
            )
            selected[name].append(
                {
                    "heldout_fold": scope["fold"],
                    "train_subjects": scope["train_subjects"],
                    "validation_subjects": scope["heldout_subjects"],
                    "decoder": accepted_decoder.as_dict(),
                    "fitted": fitted,
                    "metrics": accepted_metrics,
                    "ablations": ablations,
                    "xgb_metrics": scope["xgb_validation_metrics"],
                }
            )
    subjects = set(oof.subject_key.astype(str))
    truth, ignore = partition_evaluation_events(events, subjects)
    legacy = _legacy_control(oof, {"fusion": {}})
    legacy_metrics = _metrics(oof, legacy, truth, ignore, iou, method)
    xgb_metrics, xgb_events = evaluate_fusion_predictions(
        baseline, truth, ignore, baseline_postprocess
    )
    results: dict[str, Any] = {}
    outputs: dict[str, pd.DataFrame] = {}

    def decode_folds(
        decoders: list[Decoder], *, boundary_head: bool = False,
    ) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
        decoded_rows: list[dict[str, Any]] = []
        event_parts: list[pd.DataFrame] = []
        gate_parts: list[pd.DataFrame] = []
        for scope, decoder in zip(scopes, decoders, strict=True):
            calibrator = scope["calibrator"] if decoder.space == "positive_slope_platt" else None
            _, _, _, fitted = _apply_decoder(
                scope["train"], scope["train_anchors"], decoder, config, calibrator
            )
            transformed, _, predicted = _decode_validation(
                scope["validation"], decoder, fitted, config
            )
            if boundary_head:
                predicted = correct_boundaries(scope["validation"], predicted)
            decoded_rows.append(
                {
                    "decoder": decoder.as_dict(),
                    "fitted": fitted,
                    "metrics": _metrics(
                        transformed,
                        predicted,
                        scope["validation_truth"],
                        scope["validation_ignore"],
                        iou,
                        method,
                    ),
                }
            )
            event_parts.append(predicted)
            gate_parts.append(
                event_gate(scope["validation"], predicted, pd.DataFrame(fitted["score_reference"]))
            )
        return (
            decoded_rows,
            pd.concat(event_parts, ignore_index=True),
            pd.concat(gate_parts, ignore_index=True),
        )

    for name, rows in selected.items():
        if len(rows) != 3:
            results[name] = {
                "available": False,
                "reason": "No feasible setting for each meta-train split",
            }
            continue
        decoders = [Decoder(**row["decoder"]) for row in rows]
        global_ablations: list[dict[str, Any]] = []
        for module, parent_value in (
            ("start_mode", "legacy_or"),
            ("score_quantile", None),
            ("merge_gap", 0.0),
        ):
            if all(getattr(decoder, module) == parent_value for decoder in decoders):
                continue
            parents = [
                Decoder(**{**decoder.as_dict(), module: parent_value}) for decoder in decoders
            ]
            child_rows, child_events, _ = decode_folds(decoders)
            parent_rows, parent_events, _ = decode_folds(parents)
            child_metrics = _metrics(oof, child_events, truth, ignore, iou, method)
            parent_metrics = _metrics(oof, parent_events, truth, ignore, iou, method)
            decision = module_gate(
                child_metrics,
                parent_metrics,
                [row["metrics"] for row in child_rows],
                [row["metrics"] for row in parent_rows],
                config,
            )
            global_ablations.append(
                {
                    "module": module,
                    "gate": decision,
                    "parent_metrics": parent_metrics,
                    "child_metrics": child_metrics,
                }
            )
            if not decision["passed"]:
                decoders = parents
        decoded_rows, predicted, gate = decode_folds(decoders)
        for row, decoded in zip(rows, decoded_rows, strict=True):
            row.update(decoded)
        metrics = _metrics(oof, predicted, truth, ignore, iou, method)
        checks = _point_checks(name, metrics, legacy_metrics, xgb_metrics, config)
        bootstrap = paired_subject_bootstrap(
            oof,
            predicted,
            oof,
            legacy,
            truth,
            ignore,
            iou_threshold=iou,
            matching_method=method,
            replicates=int(config["bootstrap_replicates"]),
            seed=int(config["bootstrap_seed"]),
        )
        results[name] = {
            "available": True,
            "passed": all(checks.values()),
            "checks": checks,
            "metrics": metrics,
            "folds": rows,
            "global_ablations": global_ablations,
            "bootstrap_against_legacy": bootstrap,
        }
        outputs[f"{name}_events"] = predicted
        outputs[f"{name}_gate"] = gate
    for name, point in results.items():
        if not point.get("passed"):
            continue
        split_rows = point["folds"]
        distinct = {
            _sha_payload(row["decoder"]): Decoder(**row["decoder"]) for row in split_rows
        }
        deployment_trials = []
        for decoder in distinct.values():
            decoded, predicted, gate = decode_folds([decoder] * len(scopes))
            metrics = _metrics(oof, predicted, truth, ignore, iou, method)
            checks = _point_checks(name, metrics, legacy_metrics, xgb_metrics, config)
            checks["per_fold_stability"] = all(
                candidate["metrics"]["f1"]
                >= original["metrics"]["f1"] - config["module_maximum_partition_f1_drop"]
                for candidate, original in zip(decoded, split_rows, strict=True)
            )
            deployment_trials.append(
                {
                    "decoder": decoder.as_dict(),
                    "metrics": metrics,
                    "fold_metrics": [row["metrics"] for row in decoded],
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
        point["deployment_trials"] = deployment_trials
        passing = [trial for trial in deployment_trials if trial["passed"]]
        if not passing:
            point["passed"] = False
            point["deployment_failure"] = "No single decoder passed heldout meta checks"
            continue
        winner = max(
            passing,
            key=lambda trial: (
                min(fold["f1"] for fold in trial["fold_metrics"]),
                np.mean([fold["f1"] for fold in trial["fold_metrics"]]),
                trial["metrics"]["strict_no_ignore_f1"],
                -trial["metrics"]["false_positives_per_observed_hour"],
            ),
        )
        decoder = Decoder(**winner["decoder"])
        decoded, predicted, gate = decode_folds([decoder] * len(scopes))
        point["selection_meta_metrics"] = point["metrics"]
        point["selection_folds"] = split_rows
        point["folds"] = [
            {**original, **candidate}
            for original, candidate in zip(split_rows, decoded, strict=True)
        ]
        point["decoder"] = decoder.as_dict()
        point["metrics"] = winner["metrics"]
        point["checks"] = winner["checks"]
        point["bootstrap_against_legacy"] = paired_subject_bootstrap(
            oof,
            predicted,
            oof,
            legacy,
            truth,
            ignore,
            iou_threshold=iou,
            matching_method=method,
            replicates=int(config["bootstrap_replicates"]),
            seed=int(config["bootstrap_seed"]),
        )
        outputs[f"{name}_events"] = predicted
        outputs[f"{name}_gate"] = gate
    primary = next(
        (name for name in ("low_false_positive", "robust_f1") if results[name].get("passed")), None
    )
    if primary:
        passed, boundary_record, _ = _boundary_gate(
            oof,
            outputs[f"{primary}_events"],
            events,
            oof[["subject_key", "calibration_fold"]].drop_duplicates(),
            iou,
            method,
        )
        adjusted_metrics = boundary_record["after"]
        if passed:
            additional = _point_checks(
                primary, adjusted_metrics, legacy_metrics, xgb_metrics, config
            )
            boundary_record["promotion_checks"] = additional
            passed = all(additional.values())
        results[primary]["boundary_ablation"] = {"passed": passed, **boundary_record}
        results[primary]["use_boundary_head"] = passed
        if passed:
            adjusted_rows, adjusted_events, adjusted_gate = decode_folds(
                [Decoder(**results[primary]["decoder"])] * len(scopes), boundary_head=True
            )
            for row, decoded in zip(results[primary]["folds"], adjusted_rows, strict=True):
                row.update(decoded)
            outputs[f"{primary}_events"] = adjusted_events
            outputs[f"{primary}_gate"] = adjusted_gate
            results[primary]["metrics"] = _metrics(
                oof, adjusted_events, truth, ignore, iou, method
            )
            results[primary]["bootstrap_against_legacy"] = paired_subject_bootstrap(
                oof,
                adjusted_events,
                oof,
                legacy,
                truth,
                ignore,
                iou_threshold=iou,
                matching_method=method,
                replicates=int(config["bootstrap_replicates"]),
                seed=int(config["bootstrap_seed"]),
            )
    record = {
        "protocol_version": 2,
        "selection_scope": "subject_disjoint_meta_crossfit_within_outer_train",
        "calibration_gate": [
            {"heldout_fold": scope["fold"], **scope["calibration_gate"]} for scope in scopes
        ],
        "legacy_metrics": legacy_metrics,
        "xgb_metrics": xgb_metrics,
        "working_points": results,
        "primary_working_point": primary,
        "meta_gate_passed": primary is not None,
        "limitation": "Cross-fold outer sets reuse subjects across fold-0 selection and fold-1 evaluation; fold 1 is not an independent new-subject confirmation.",
    }
    outputs["legacy_events"] = legacy
    outputs["xgb_events"] = xgb_events
    return json_safe(record), outputs, pd.concat(trials, ignore_index=True)


def _run_name(value: str) -> str:
    if not RUN_PATTERN.fullmatch(value):
        raise ValueError(
            "Run name must begin with dtp_postprocess_ and contain ASCII letters, digits, _ or -"
        )
    return value


def _source(output_root: Path, source_run: str, fold: int) -> tuple[Path, dict[str, str]]:
    from bme_eating.cli import _validate_v4_prediction_source

    return _validate_v4_prediction_source(output_root, source_run, fold)


def _baseline(output_root: Path, config: dict[str, Any], fold: int) -> tuple[Path, dict[str, Any]]:
    information = validate_frozen_baseline_fold(
        output_root,
        str(config["experiment"]["baseline_name"]),
        fold,
        str(config["experiment"]["baseline_source_commit"]),
    )
    return Path(information["directory"]), information


def _manifest(output_dir: Path, config: dict[str, Any], provenance: dict[str, Any]) -> None:
    outputs = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    }
    _save_json(
        output_dir / "run_manifest.json",
        {
            "protocol_version": 2,
            "experiment": {
                "name": output_dir.parent.name,
                "fold": int(output_dir.name.removeprefix("fold_")),
            },
            "git": _git_identity(),
            "config_sha256": sha256_file(Path(config["_config_path"])),
            "provenance": provenance,
            "artifact_hashes": outputs,
            "random_seeds": {
                "split": int(config["data"]["split_seed"]),
                "bootstrap": int(config["dtp_postprocess"]["bootstrap_seed"]),
            },
        },
    )


def _validate_selection(directory: Path, run_name: str, fold: int) -> dict[str, Any]:
    manifest_path = directory / "run_manifest.json"
    selection_path = directory / "selected_dtp_postprocess.json"
    if not manifest_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError("Frozen DTP postprocess selection and manifest are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["experiment"] != {"name": run_name, "fold": fold}:
        raise RuntimeError("Selection manifest identity changed")
    if manifest["artifact_hashes"].get(selection_path.name) != sha256_file(selection_path):
        raise RuntimeError("Frozen DTP selection hash changed")
    if manifest["git"]["dirty"] or manifest["git"]["commit"] != _git_identity()["commit"]:
        raise RuntimeError("DTP selection was not produced from the current clean Git commit")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("protocol_version") != 2 or selection.get("selection_run") != run_name:
        raise RuntimeError("DTP selection protocol or run identity changed")
    return selection


def _training_inputs(
    output_root: Path, source_dir: Path, baseline_dir: Path, fold: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from bme_eating.fusion import align_prediction_frames

    oof = pd.read_parquet(source_dir / "dtp_oof_predictions.parquet")
    baseline = pd.read_parquet(baseline_dir / "validation_predictions.parquet")
    baseline, oof = align_prediction_frames(baseline, oof)
    if "calibration_fold" not in oof:
        raise ValueError("DTP OOF has no calibration_fold")
    subject_folds = load_subject_folds(output_root / "indices" / "subject_folds.json")
    actual = set(oof.subject_key.astype(str))
    expected = {str(subject) for subject, partition in subject_folds.items() if partition != fold}
    if actual != expected:
        raise ValueError("DTP outer-train subjects disagree with frozen fold map")
    filters = [("subject_key", "in", sorted(actual))]
    anchors = pd.read_parquet(output_root / "indices" / "anchors.parquet", filters=filters)
    events = pd.read_parquet(output_root / "indices" / "events.parquet", filters=filters)
    if set(anchors.subject_key.astype(str)) != actual:
        raise ValueError("Outer-train anchor subjects do not cover the prediction subjects")
    return oof, baseline, anchors, events


def tune_dtp_postprocess(args: Any) -> Path:
    if int(args.workers) < 1:
        raise ValueError("--workers must be positive")
    config = load_config(args.config)
    settings = config["dtp_postprocess"]
    if int(settings["protocol_version"]) != 2:
        raise ValueError("Expected pure DTP postprocessing protocol 2")
    require_clean_git_worktree()
    _, output_root = resolve_roots(config)
    run_name = _run_name(args.run_name)
    fold = int(args.fold)
    if fold != 0:
        raise ValueError("DTP v2 selection is restricted to development fold 0")
    source_dir, source_hashes = _source(output_root, args.source_run, fold)
    baseline_dir, baseline_info = _baseline(output_root, config, fold)
    oof, baseline, anchors, events = _training_inputs(output_root, source_dir, baseline_dir, fold)
    baseline_postprocess = json.loads(
        (baseline_dir / "selected_postprocess.json").read_text(encoding="utf-8")
    )
    signature = _sha_payload(
        {
            "settings": settings,
            "git": _git_identity()["commit"],
            "code": sha256_file(Path(__file__)),
            "config": sha256_file(Path(config["_config_path"])),
            "source": source_hashes,
            "baseline": baseline_info["artifact_hashes"],
            "anchors": baseline_info["input_hashes"]["anchors"],
            "events_and_ignore": baseline_info["input_hashes"]["events"],
            "fold_map": baseline_info["input_hashes"]["subject_folds"],
        }
    )
    output_dir = output_root / "experiments" / run_name / f"fold_{fold}"
    if (output_dir / "selected_dtp_postprocess.json").exists():
        raise FileExistsError("This DTP postprocess selection is already frozen")
    signature_path = output_dir / "search_signature.json"
    if signature_path.exists():
        saved = json.loads(signature_path.read_text(encoding="utf-8"))
        if saved.get("signature") != signature:
            raise RuntimeError(
                "Cannot resume search after prediction, label, config or code changes"
            )
    else:
        if output_dir.exists():
            raise FileExistsError("Existing directory has no trusted search signature")
        output_dir.mkdir(parents=True)
        _save_json(signature_path, {"signature": signature})
    print("[1/3] Fitting train-only calibration and searching three meta splits...", flush=True)
    selection, outputs, trials = _crossfit(
        oof,
        anchors,
        events,
        baseline,
        baseline_postprocess,
        settings,
        output_dir / "search.checkpoint.jsonl",
        workers=int(args.workers),
    )
    selected_name = selection["primary_working_point"]
    if selected_name:
        decoder = Decoder(**selection["working_points"][selected_name]["decoder"])
        calibrator = (
            fit_platt_calibrator(oof, anchors) if decoder.space == "positive_slope_platt" else None
        )
        _, _, _, fitted = _apply_decoder(oof, anchors, decoder, settings, calibrator)
        selection["final_decoder"] = decoder.as_dict()
        selection["final_fit"] = fitted
    selection.update(
        {
            "selection_run": run_name,
            "fold": fold,
            "source_run": args.source_run,
            "source_hashes": source_hashes,
            "baseline_hashes": baseline_info["artifact_hashes"],
            "input_hashes": baseline_info["input_hashes"],
            "search_signature": signature,
            "development_only": fold == 0,
            "worker_request": int(args.workers),
        }
    )
    print("[2/3] Writing meta-OOF evidence and frozen selection...", flush=True)
    for name, frame in outputs.items():
        frame.to_parquet(output_dir / f"meta_{name}.parquet", index=False)
    if selected_name:
        outputs[f"{selected_name}_events"].to_csv(output_dir / "meta_oof_events.csv", index=False)
        outputs[f"{selected_name}_gate"].to_parquet(
            output_dir / "dtp_event_gate.parquet", index=False
        )
        subjects = set(oof.subject_key.astype(str))
        truth, ignore = partition_evaluation_events(events, subjects)
        failures, per_subject, by_hand = _event_diagnostics(
            oof,
            outputs[f"{selected_name}_events"],
            truth,
            ignore,
            float(baseline_postprocess["iou_threshold"]),
            str(baseline_postprocess.get("matching_method", "max_cardinality_iou")),
        )
        failures.to_csv(output_dir / "meta_oof_failure_cases.csv", index=False)
        per_subject.to_csv(output_dir / "meta_oof_per_subject_metrics.csv", index=False)
        _save_json(output_dir / "meta_oof_hand_relation_metrics.json", by_hand)
    _save_json(
        output_dir / "meta_oof_metrics.json",
        {
            "legacy": selection["legacy_metrics"],
            "xgboost": selection["xgb_metrics"],
            "working_points": {
                name: {
                    "metrics": value.get("metrics"),
                    "checks": value.get("checks"),
                    "bootstrap": value.get("bootstrap_against_legacy"),
                }
                for name, value in selection["working_points"].items()
            },
        },
    )
    pd.DataFrame(
        [
            {
                "working_point": name,
                "heldout_fold": fold["heldout_fold"],
                "module": ablation["module"],
                "passed": ablation["gate"]["passed"],
                "delta_f1": ablation["gate"]["delta_f1"],
            }
            for name, value in selection["working_points"].items()
            for fold in value.get("folds", [])
            for ablation in fold["ablations"]
        ]
        + [
            {
                "working_point": name,
                "heldout_fold": "meta_oof",
                "module": ablation["module"],
                "passed": ablation["gate"]["passed"],
                "delta_f1": ablation["gate"]["delta_f1"],
            }
            for name, value in selection["working_points"].items()
            for ablation in value.get("global_ablations", [])
        ],
        columns=["working_point", "heldout_fold", "module", "passed", "delta_f1"],
    ).to_csv(output_dir / "module_ablations.csv", index=False)
    trials.to_csv(output_dir / "search_trials.csv", index=False)
    _save_json(output_dir / "selected_dtp_postprocess.json", selection)
    _manifest(
        output_dir,
        config,
        {
            "source_run": args.source_run,
            "source_hashes": source_hashes,
            "baseline_hashes": baseline_info["artifact_hashes"],
            "input_hashes": baseline_info["input_hashes"],
            "search_signature": signature,
        },
    )
    print("[3/3] Search complete; no outer predictions or labels were loaded.", flush=True)
    print(json.dumps({"primary": selected_name, "directory": str(output_dir)}, ensure_ascii=False))
    return output_dir


def evaluate_dtp_postprocess(args: Any) -> Path:
    config = load_config(args.config)
    require_clean_git_worktree()
    _, output_root = resolve_roots(config)
    fold = int(args.fold)
    if fold == 0:
        raise ValueError("Fold 0 is development-only; outer DTP evaluation starts at fold 1")
    result_name = _run_name(args.run_name)
    selection_name = _run_name(args.selection_run)
    selection_fold = 0
    selection_dir = output_root / "experiments" / selection_name / f"fold_{selection_fold}"
    selection = _validate_selection(selection_dir, selection_name, selection_fold)
    if not selection["meta_gate_passed"] or not selection["primary_working_point"]:
        raise RuntimeError("Meta-OOF gate failed; outer evaluation is forbidden")
    output_dir = output_root / "experiments" / result_name / f"fold_{fold}"
    if output_dir.exists():
        raise FileExistsError(
            "Outer DTP evaluation is already present; no overwrites or point switching"
        )
    for existing in (output_root / "experiments").glob(
        f"dtp_postprocess_*/fold_{fold}/applied_selection.json"
    ):
        raise RuntimeError(f"Outer fold {fold} was already evaluated at {existing}")
    if fold >= 2:
        confirmation = output_root / "experiments" / result_name / "fold_1" / "outer_gate.json"
        if (
            not confirmation.is_file()
            or not json.loads(confirmation.read_text(encoding="utf-8"))["passed"]
        ):
            raise RuntimeError("Fold 1 cross-fold diagnostic failed; folds 2–4 are blocked")
    source_dir, source_hashes = _source(output_root, args.source_run, fold)
    baseline_dir, baseline_info = _baseline(output_root, config, fold)
    if baseline_info["input_hashes"] != selection["input_hashes"]:
        raise RuntimeError("Source and selection use different data or subject folds")
    oof, _, anchors, _ = _training_inputs(output_root, source_dir, baseline_dir, fold)
    decoder = Decoder(**selection["final_decoder"])
    calibrator = (
        fit_platt_calibrator(oof, anchors) if decoder.space == "positive_slope_platt" else None
    )
    _, _, _, fitted = _apply_decoder(oof, anchors, decoder, config["dtp_postprocess"], calibrator)
    test = pd.read_parquet(source_dir / "dtp_test_predictions.parquet")
    expected_subjects = {
        str(subject)
        for subject, partition in load_subject_folds(
            output_root / "indices" / "subject_folds.json"
        ).items()
        if partition == fold
    }
    if set(test.subject_key.astype(str)) != expected_subjects or set(
        test.subject_key.astype(str)
    ) & set(oof.subject_key.astype(str)):
        raise ValueError("Outer test predictions violate subject isolation")
    prepared, core, predicted = _decode_validation(test, decoder, fitted, config["dtp_postprocess"])
    boundary_used = bool(
        selection["working_points"][selection["primary_working_point"]].get("use_boundary_head")
    )
    if boundary_used:
        predicted = correct_boundaries(test, predicted)
    gate = event_gate(test, predicted, pd.DataFrame(fitted["score_reference"]))
    outer_truth = pd.read_parquet(
        output_root / "indices" / "events.parquet",
        filters=[("subject_key", "in", sorted(expected_subjects))],
    )
    truth, ignore = partition_evaluation_events(outer_truth, expected_subjects)
    baseline_test = pd.read_parquet(baseline_dir / "test_predictions.parquet")
    postprocess = json.loads(
        (baseline_dir / "selected_postprocess.json").read_text(encoding="utf-8")
    )
    from bme_eating.fusion import evaluate_fusion_predictions

    xgb, _ = evaluate_fusion_predictions(baseline_test, truth, ignore, postprocess)
    metrics = _metrics(
        prepared,
        predicted,
        truth,
        ignore,
        postprocess["iou_threshold"],
        postprocess["matching_method"],
    )
    failures, per_subject, by_hand = _event_diagnostics(
        prepared,
        predicted,
        truth,
        ignore,
        postprocess["iou_threshold"],
        postprocess["matching_method"],
    )
    checks = {
        "f1": metrics["f1"] >= xgb["f1"],
        "strict": metrics["strict_no_ignore_f1"] >= xgb["strict_no_ignore_f1"],
        "fp_hour": metrics["false_positives_per_observed_hour"]
        <= xgb["false_positives_per_observed_hour"]
        * config["dtp_postprocess"]["maximum_xgb_fp_ratio"],
    }
    outer_gate = {
        "passed": all(checks.values()),
        "checks": checks,
        "independent_confirmation": False,
        "limitation": selection["limitation"],
    }
    output_dir.mkdir(parents=True)
    predicted.to_csv(output_dir / "test_events.csv", index=False)
    failures.to_csv(output_dir / "test_failure_cases.csv", index=False)
    core.to_csv(output_dir / "core_events.csv", index=False)
    gate.to_parquet(output_dir / "dtp_event_gate.parquet", index=False)
    per_subject.to_csv(output_dir / "per_subject_metrics.csv", index=False)
    _save_json(output_dir / "hand_relation_metrics.json", by_hand)
    _save_json(output_dir / "test_metrics.json", {"candidate": metrics, "xgboost": xgb})
    _save_json(output_dir / "outer_gate.json", outer_gate)
    _save_json(
        output_dir / "applied_selection.json",
        {
            "selection_run": selection_name,
            "source_run": args.source_run,
            "selection_sha256": sha256_file(selection_dir / "selected_dtp_postprocess.json"),
            "primary_working_point": selection["primary_working_point"],
            "decoder": decoder.as_dict(),
            "outer_train_fit": fitted,
            "boundary_head": boundary_used,
        },
    )
    _manifest(
        output_dir,
        config,
        {
            "source_run": args.source_run,
            "source_hashes": source_hashes,
            "baseline_hashes": baseline_info["artifact_hashes"],
            "input_hashes": baseline_info["input_hashes"],
            "selection_run": selection_name,
            "selection_sha256": sha256_file(selection_dir / "selected_dtp_postprocess.json"),
        },
    )
    print(
        json.dumps(
            {"gate": outer_gate, "metrics": metrics, "directory": str(output_dir)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return output_dir
