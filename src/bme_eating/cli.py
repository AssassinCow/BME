from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xgboost
from tqdm import tqdm

from bme_eating.config import load_config, resolve_roots
from bme_eating.data.labels import build_anchor_index, classify_event_coverage
from bme_eating.data.manifest import build_secure_indices
from bme_eating.data.packet_reader import UnsupportedSensorFormatError, inspect_ppg_layout
from bme_eating.data.preprocess import preprocess_attachment, write_preprocess_summary
from bme_eating.data.splits import create_subject_folds, load_subject_folds
from bme_eating.features.baseline import build_segment_features
from bme_eating.metrics import evaluate_events
from bme_eating.models.dtp_sqf import DTPSQF
from bme_eating.models.xgb_baseline import predict_xgboost, train_xgboost_fold
from bme_eating.postprocess import probabilities_to_events, tune_postprocess_parameters
from bme_eating.training.dtp_trainer import train_dtp_fold

_RECOVERABLE_INPUT_ERRORS = (
    OSError,
    ValueError,
    KeyError,
    EOFError,
    IndexError,
    StopIteration,
    TypeError,
    RuntimeError,
)


def _invalid_subjects(config: dict[str, Any]) -> set[str]:
    return {str(value).upper() for value in config["data"]["invalid_subjects"]}


def _indices(config: dict[str, Any], data_root: Path, output_root: Path):
    index_dir = output_root / "indices"
    records_path = index_dir / "records.parquet"
    events_path = index_dir / "events.parquet"
    if records_path.exists() and events_path.exists():
        return pd.read_parquet(records_path), pd.read_parquet(events_path)
    return build_secure_indices(
        data_root,
        output_root,
        _invalid_subjects(config),
        str(config["data"]["formal_subject_pattern"]),
    )


def _audit_checkpoint_path(output_root: Path) -> Path:
    return output_root / "indices" / "schema_audit.checkpoint.json"


def _selected_zip_fingerprints(selected: pd.DataFrame) -> list[dict[str, object]]:
    fingerprints: list[dict[str, object]] = []
    for record in selected.itertuples(index=False):
        fingerprints.append(
            {
                "zip_path": str(record.zip_path),
                "zip_sha256": str(getattr(record, "zip_sha256", "")),
                "zip_size_bytes": int(getattr(record, "zip_size_bytes", 0) or 0),
            }
        )
    return fingerprints


def _load_audit_checkpoint(
    checkpoint_path: Path,
    schema_zips: str,
    maximum_rows: int,
    selected_fingerprints: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    if not checkpoint_path.exists():
        return {}
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    try:
        checkpoint_rows = int(checkpoint.get("maximum_rows", -1))
    except (TypeError, ValueError):
        return {}
    if (
        checkpoint.get("schema_zips") != schema_zips
        or checkpoint_rows != maximum_rows
        or checkpoint.get("selected_fingerprints") != selected_fingerprints
    ):
        return {}
    completed = checkpoint.get("completed", {})
    return completed if isinstance(completed, dict) else {}


def _write_audit_checkpoint(
    checkpoint_path: Path,
    schema_zips: str,
    maximum_rows: int,
    selected_fingerprints: list[dict[str, object]],
    completed: dict[str, dict[str, object]],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "schema_zips": schema_zips,
        "maximum_rows": maximum_rows,
        "selected_fingerprints": selected_fingerprints,
        "completed": completed,
    }
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(checkpoint_path)


def _audit_layout_issue(zip_path: str, error: Exception) -> dict[str, object]:
    return {
        "zip_name": Path(zip_path).name,
        "status": "error",
        "zip_path": zip_path,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "rows_scanned": 0,
        "ppg_rows": 0,
        "nonzero_fraction_by_slot": [0.0] * 44,
    }


def command_environment(_: argparse.Namespace) -> None:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "xgboost": xgboost.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory
        if torch.cuda.is_available()
        else 0,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run this project on the RTX 4080 computer.")


def command_audit(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    data_root, output_root = resolve_roots(config)
    records, events = build_secure_indices(
        data_root,
        output_root,
        _invalid_subjects(config),
        str(config["data"]["formal_subject_pattern"]),
    )
    if args.schema_zips == "all":
        selected = records
    else:
        count = int(args.schema_zips)
        selected = (
            records.sort_values("zip_path").groupby("subject_key", as_index=False).head(1).head(count)
        )
    maximum_rows = int(args.maximum_rows)
    selected_fingerprints = _selected_zip_fingerprints(selected)
    checkpoint_path = _audit_checkpoint_path(output_root)
    completed = {} if getattr(args, "no_resume", False) else _load_audit_checkpoint(
        checkpoint_path,
        str(args.schema_zips),
        maximum_rows,
        selected_fingerprints,
    )
    for fingerprint in tqdm(selected_fingerprints, desc="Inspecting PPG layout"):
        zip_path = str(fingerprint["zip_path"])
        if zip_path in completed:
            continue
        try:
            completed[zip_path] = inspect_ppg_layout(zip_path, maximum_rows=maximum_rows)
        except _RECOVERABLE_INPUT_ERRORS as error:
            completed[zip_path] = _audit_layout_issue(zip_path, error)
        _write_audit_checkpoint(
            checkpoint_path,
            str(args.schema_zips),
            maximum_rows,
            selected_fingerprints,
            completed,
        )
    layouts = [completed[str(fingerprint["zip_path"])] for fingerprint in selected_fingerprints]
    active_slots = 0
    for layout in layouts:
        fractions = layout["nonzero_fraction_by_slot"]
        for index, value in enumerate(fractions):
            if value > 0:
                active_slots = max(active_slots, index + 1)
    layout_status_counts: dict[str, int] = {}
    for layout in layouts:
        status = str(layout.get("status", "ok"))
        layout_status_counts[status] = layout_status_counts.get(status, 0) + 1
    audit = {
        "records": len(records),
        "subjects": int(records["subject_key"].nunique()),
        "events": len(events),
        "positive_duration_events": int(events["valid_duration"].sum()),
        "nonpositive_duration_events": int((~events["valid_duration"]).sum()),
        "configured_ppg_samples_per_row": int(config["data"]["ppg_samples_per_row"]),
        "maximum_observed_nonzero_ppg_slot": active_slots,
        "layout_files_inspected": len(layouts),
        "layout_status_counts": layout_status_counts,
        "layouts": layouts,
    }
    audit_path = output_root / "indices" / "schema_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint_path.unlink(missing_ok=True)
    print(audit_path)
    if active_slots > int(config["data"]["ppg_samples_per_row"]):
        raise SystemExit(
            "Observed nonzero PPG slots exceed ppg_samples_per_row; update config before preprocessing."
        )


def _preprocess_job(
    record: dict[str, object],
    segment_dir: str,
    data_config: dict[str, object],
    compressed: bool,
    overwrite: bool,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    try:
        rows = preprocess_attachment(
            record,
            Path(segment_dir),
            data_config,
            compressed=compressed,
            overwrite=overwrite,
        )
    except UnsupportedSensorFormatError as error:
        return [], {
            "status": "unsupported_binary",
            "zip_path": str(record["zip_path"]),
            "zip_sha256": str(record.get("zip_sha256", "")),
            "subject_key": str(record.get("subject_key", "")),
            "member_name": error.member_name,
            "error_type": type(error).__name__,
            "error": error.reason,
            "traceback": traceback.format_exc(),
        }
    except _RECOVERABLE_INPUT_ERRORS as error:
        return [], {
            "status": "preprocess_error",
            "zip_path": str(record.get("zip_path", "")),
            "zip_sha256": str(record.get("zip_sha256", "")),
            "subject_key": str(record.get("subject_key", "")),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    except Exception as error:  # noqa: BLE001 - worker boundary must serialize all failures
        return [], {
            "status": "preprocess_unexpected_error",
            "zip_path": str(record.get("zip_path", "")),
            "zip_sha256": str(record.get("zip_sha256", "")),
            "subject_key": str(record.get("subject_key", "")),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    return rows, None


def command_preprocess(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    data_root, output_root = resolve_roots(config)
    records, events = _indices(config, data_root, output_root)
    segment_dir = output_root / "segments"
    workers = int(args.workers or config["preprocess"]["workers"])
    all_rows: list[dict[str, object]] = []
    preprocess_issues: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _preprocess_job,
                record._asdict(),
                str(segment_dir),
                dict(config["data"]),
                bool(config["preprocess"]["compression"]),
                bool(args.overwrite or config["preprocess"]["overwrite"]),
            ): record._asdict()
            for record in records.itertuples(index=False)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            try:
                rows, issue = future.result()
            except _RECOVERABLE_INPUT_ERRORS as error:
                record = futures[future]
                rows, issue = [], {
                    "status": "preprocess_worker_error",
                    "zip_path": str(record.get("zip_path", "")),
                    "zip_sha256": str(record.get("zip_sha256", "")),
                    "subject_key": str(record.get("subject_key", "")),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            except Exception as error:  # noqa: BLE001 - preserve worker diagnostics
                record = futures[future]
                rows, issue = [], {
                    "status": "preprocess_worker_unexpected_error",
                    "zip_path": str(record.get("zip_path", "")),
                    "zip_sha256": str(record.get("zip_sha256", "")),
                    "subject_key": str(record.get("subject_key", "")),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            all_rows.extend(rows)
            if issue is not None:
                preprocess_issues.append(issue)
    issues_path = output_root / "indices" / "preprocess_issues.json"
    issues_path.write_text(
        json.dumps(preprocess_issues, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fatal_issues = [
        issue for issue in preprocess_issues if issue.get("status") != "unsupported_binary"
    ]
    if fatal_issues:
        raise RuntimeError(
            f"Preprocessing failed for {len(fatal_issues)} attachment(s); see {issues_path}"
        )
    if not all_rows:
        raise RuntimeError(f"Preprocessing produced no segments; see {issues_path}")
    segment_index_path = output_root / "indices" / "segments.parquet"
    write_preprocess_summary(all_rows, segment_index_path)
    segments = pd.DataFrame(all_rows)
    events = classify_event_coverage(events, segments)
    events.to_parquet(output_root / "indices" / "events.parquet", index=False)
    anchors = build_anchor_index(
        segments,
        events,
        int(config["data"]["output_step_seconds"]),
        output_root / "indices" / "anchors.parquet",
    )
    create_subject_folds(
        events,
        int(config["data"]["subject_folds"]),
        int(config["data"]["split_seed"]),
        output_root / "indices" / "subject_folds.json",
        subject_keys=set(segments["subject_key"].astype(str)),
    )
    print(
        json.dumps(
            {
                "segments": len(segments),
                "anchors": len(anchors),
                "preprocess_issues": len(preprocess_issues),
                "events_by_coverage": events["coverage"].value_counts().to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _feature_job(
    segment_path: str,
    anchors: pd.DataFrame,
    feature_config: dict[str, Any],
    cache_path: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    try:
        if cache_path is not None and Path(cache_path).exists():
            try:
                return pd.read_parquet(cache_path), None
            except _RECOVERABLE_INPUT_ERRORS:
                Path(cache_path).unlink(missing_ok=True)
        frame = build_segment_features(
            segment_path,
            anchors,
            int(feature_config["window_seconds"]),
            bool(feature_config["include_dyadic"]),
            [int(value) for value in feature_config["motion_bucket_seconds"]],
            [int(value) for value in feature_config["ppg_bucket_seconds"]],
        )
        if cache_path is not None:
            target = Path(cache_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = target.with_name(target.name + ".tmp")
            frame.to_parquet(temporary_path, index=False)
            temporary_path.replace(target)
    except _RECOVERABLE_INPUT_ERRORS as error:
        return pd.DataFrame(), {
            "status": "feature_extraction_error",
            "segment_path": segment_path,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    except Exception as error:  # noqa: BLE001 - worker boundary must serialize all failures
        return pd.DataFrame(), {
            "status": "feature_extraction_unexpected_error",
            "segment_path": segment_path,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    return frame, None


def _feature_cache_path(
    cache_dir: Path,
    segment_path: str,
    anchors: pd.DataFrame,
    feature_config: dict[str, Any],
) -> Path:
    path = Path(segment_path)
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(feature_config, sort_keys=True, separators=(",", ":")).encode()
    )
    digest.update(str(path.resolve()).encode())
    digest.update(f"{stat.st_size}|{stat.st_mtime_ns}".encode())
    digest.update(pd.util.hash_pandas_object(anchors, index=True).to_numpy().tobytes())
    return cache_dir / f"{path.stem}-{digest.hexdigest()[:16]}.parquet"


def command_build_features(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, output_root = resolve_roots(config)
    anchors = pd.read_parquet(output_root / "indices" / "anchors.parquet")
    grouped = list(anchors.groupby("segment_id", sort=False))
    workers = int(args.workers or config["features"]["workers"])
    feature_name = "baseline_dyadic" if config["features"]["include_dyadic"] else "baseline"
    cache_dir = output_root / "features" / ".cache" / feature_name
    jobs = [
        (
            str(group.iloc[0].segment_path),
            group,
            _feature_cache_path(
                cache_dir,
                str(group.iloc[0].segment_path),
                group,
                dict(config["features"]),
            ),
        )
        for _, group in grouped
    ]
    cached_count = sum(cache_path.exists() for _, _, cache_path in jobs)
    print(f"Feature cache: {cached_count}/{len(jobs)} segments reusable", flush=True)
    frames: list[pd.DataFrame] = []
    feature_issues: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _feature_job,
                segment_path,
                group,
                dict(config["features"]),
                str(cache_path),
            ): segment_path
            for segment_path, group, cache_path in jobs
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
            try:
                frame, issue = future.result()
            except _RECOVERABLE_INPUT_ERRORS as error:
                frame, issue = pd.DataFrame(), {
                    "status": "feature_worker_error",
                    "segment_path": futures[future],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            except Exception as error:  # noqa: BLE001 - preserve worker diagnostics
                frame, issue = pd.DataFrame(), {
                    "status": "feature_worker_unexpected_error",
                    "segment_path": futures[future],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            if issue is not None:
                feature_issues.append(issue)
            elif len(frame):
                frames.append(frame)
    issues_path = output_root / "indices" / "feature_issues.json"
    issues_path.write_text(
        json.dumps(feature_issues, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if feature_issues:
        raise RuntimeError(
            f"Feature extraction failed for {len(feature_issues)} segment(s); see {issues_path}"
        )
    if not frames:
        raise RuntimeError("Feature extraction produced no rows")
    features = pd.concat(frames, ignore_index=True).sort_values(
        ["subject_key", "segment_id", "timestamp_ms"]
    ).reset_index(drop=True)
    path = output_root / "features" / f"{feature_name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    features.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)
    print(path)


def _evaluate_prediction_file(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    postprocess_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, object]]:
    events = probabilities_to_events(
        predictions,
        ema_half_life_seconds=float(postprocess_config["ema_half_life_seconds"]),
        high_threshold=float(postprocess_config["high_threshold"]),
        low_threshold=float(postprocess_config["low_threshold"]),
        minimum_event_seconds=float(postprocess_config["minimum_event_seconds"]),
        merge_gap_seconds=float(postprocess_config["merge_gap_seconds"]),
        boundary_lookback_seconds=float(postprocess_config["boundary_lookback_seconds"]),
    )
    output: dict[str, object] = {}
    for method in ("hungarian", "greedy"):
        metrics, matches = evaluate_events(
            truth,
            events,
            iou_threshold=float(postprocess_config["iou_threshold"]),
            method=method,
        )
        output[method] = metrics
        if method == "hungarian" and len(matches):
            relation_summary: dict[str, object] = {}
            for relation in ("same", "different", "unknown"):
                truth_count = int((truth["hand_relation"] == relation).sum())
                matched_relation = matches[matches["hand_relation"] == relation]
                relation_summary[relation] = {
                    "truth_events": truth_count,
                    "matched_events": len(matched_relation),
                    "sensitivity": len(matched_relation) / truth_count if truth_count else None,
                    "start_mae_seconds": float(
                        matched_relation["start_absolute_error_ms"].mean() / 1000.0
                    )
                    if len(matched_relation)
                    else None,
                    "end_mae_seconds": float(
                        matched_relation["end_absolute_error_ms"].mean() / 1000.0
                    )
                    if len(matched_relation)
                    else None,
                }
            output["hand_relation"] = relation_summary
    return events, output


def _tune_and_save_postprocess(
    validation_predictions: pd.DataFrame,
    validation_truth: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, float]:
    best, trials = tune_postprocess_parameters(
        validation_predictions,
        validation_truth,
        config["postprocess_search"],
        float(config["postprocess"]["iou_threshold"]),
        checkpoint_path=output_dir / "postprocess_search.checkpoint.jsonl",
    )
    best["iou_threshold"] = float(config["postprocess"]["iou_threshold"])
    (output_dir / "selected_postprocess.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trials.to_csv(output_dir / "postprocess_trials.csv", index=False)
    return best


def command_train_xgb(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, output_root = resolve_roots(config)
    feature_name = "baseline_dyadic" if config["features"]["include_dyadic"] else "baseline"
    print("[1/5] Loading features and subject folds...", flush=True)
    features = pd.read_parquet(output_root / "features" / f"{feature_name}.parquet")
    subject_folds = load_subject_folds(output_root / "indices" / "subject_folds.json")
    experiment_dir = output_root / "experiments" / feature_name / f"fold_{args.fold}"
    print("[2/5] Tuning and fitting XGBoost...", flush=True)
    model, columns, validation, test = train_xgboost_fold(
        features,
        subject_folds,
        int(args.fold),
        config["xgboost"],
        experiment_dir,
        resume_search=not bool(getattr(args, "no_resume", False)),
    )
    print("[3/5] Predicting validation and test windows...", flush=True)
    validation_predictions = predict_xgboost(model, validation, columns)
    validation_predictions.to_parquet(
        experiment_dir / "validation_predictions.parquet", index=False
    )
    predictions = predict_xgboost(model, test, columns)
    predictions.to_parquet(experiment_dir / "test_predictions.parquet", index=False)
    events = pd.read_parquet(output_root / "indices" / "events.parquet")
    validation_subjects = set(validation["subject_key"].unique())
    validation_truth = events[
        events["subject_key"].isin(validation_subjects)
        & events["valid_duration"]
        & (events["coverage"] == "full")
    ]
    print("[4/5] Tuning event postprocessing on CPU...", flush=True)
    selected_postprocess = _tune_and_save_postprocess(
        validation_predictions, validation_truth, config, experiment_dir
    )
    test_subjects = set(test["subject_key"].unique())
    truth = events[
        events["subject_key"].isin(test_subjects)
        & events["valid_duration"]
        & (events["coverage"] == "full")
    ]
    print("[5/5] Evaluating the held-out fold...", flush=True)
    predicted_events, metrics = _evaluate_prediction_file(
        predictions, truth, selected_postprocess
    )
    predicted_events.to_csv(experiment_dir / "test_events.csv", index=False)
    (experiment_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(experiment_dir)


def command_train_dtp(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, output_root = resolve_roots(config)
    print("[1/4] Loading anchors, segments, events, and subject folds...", flush=True)
    anchors = pd.read_parquet(output_root / "indices" / "anchors.parquet")
    segments = pd.read_parquet(output_root / "indices" / "segments.parquet")
    events = pd.read_parquet(output_root / "indices" / "events.parquet")
    subject_folds = load_subject_folds(output_root / "indices" / "subject_folds.json")
    future = int(config["model"].get("future_context_seconds", 0))
    experiment_name = "dtp_sqf_causal" if future == 0 else f"dtp_sqf_future{future}"
    experiment_dir = output_root / "experiments" / experiment_name / f"fold_{args.fold}"
    print("[2/4] Training DTP-SQF on CUDA...", flush=True)
    checkpoint = train_dtp_fold(
        anchors,
        segments,
        events,
        subject_folds,
        int(args.fold),
        config["model"],
        config["training"],
        config["loss"],
        config["postprocess"],
        experiment_dir,
        Path(args.resume) if args.resume else None,
    )
    predictions = pd.read_parquet(experiment_dir / "test_predictions.parquet")
    validation_predictions = pd.read_parquet(
        experiment_dir / "best_validation_predictions.parquet"
    )
    validation_subjects = set(validation_predictions["subject_key"].unique())
    validation_truth = events[
        events["subject_key"].isin(validation_subjects)
        & events["valid_duration"]
        & (events["coverage"] == "full")
    ]
    print("[3/4] Tuning event postprocessing on CPU...", flush=True)
    selected_postprocess = _tune_and_save_postprocess(
        validation_predictions, validation_truth, config, experiment_dir
    )
    test_subjects = {
        subject for subject, fold in subject_folds.items() if fold == int(args.fold)
    }
    truth = events[
        events["subject_key"].isin(test_subjects)
        & events["valid_duration"]
        & (events["coverage"] == "full")
    ]
    print("[4/4] Evaluating the held-out fold...", flush=True)
    predicted_events, metrics = _evaluate_prediction_file(
        predictions, truth, selected_postprocess
    )
    predicted_events.to_csv(experiment_dir / "test_events.csv", index=False)
    (experiment_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(checkpoint)


def command_evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, output_root = resolve_roots(config)
    predictions = pd.read_parquet(args.predictions)
    events = pd.read_parquet(output_root / "indices" / "events.parquet")
    subjects = set(predictions["subject_key"].unique())
    truth = events[
        events["subject_key"].isin(subjects)
        & events["valid_duration"]
        & (events["coverage"] == "full")
    ]
    predicted_events, metrics = _evaluate_prediction_file(
        predictions, truth, config["postprocess"]
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    predicted_events.to_csv(output_dir / "events.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def command_smoke_model(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run this command on the RTX 4080 computer.")
    device = torch.device("cuda")
    future = int(config["model"].get("future_context_seconds", 0))
    batch_size = int(args.batch_size)
    batch: dict[str, torch.Tensor] = {
        "motion_blocks": torch.randn(batch_size, 127, 12, 300, device=device),
        "motion_valid": torch.ones(batch_size, 127, device=device),
        "ppg_blocks": torch.randn(batch_size, 31, 2, 750, device=device),
        "ppg_quality": torch.rand(batch_size, 31, 8, device=device),
        "ppg_valid": torch.ones(batch_size, 31, device=device),
    }
    if future:
        batch.update(
            {
                "future_motion_blocks": torch.randn(
                    batch_size, future // 3, 12, 300, device=device
                ),
                "future_motion_valid": torch.ones(batch_size, future // 3, device=device),
                "future_ppg_blocks": torch.randn(
                    batch_size, future // 15, 2, 750, device=device
                ),
                "future_ppg_quality": torch.rand(
                    batch_size, future // 15, 8, device=device
                ),
                "future_ppg_valid": torch.ones(batch_size, future // 15, device=device),
            }
        )
    model = DTPSQF(config["model"]).to(device)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(batch)
    print({key: tuple(value.shape) for key, value in output.items()})
    print({"peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()})

