from __future__ import annotations

import argparse
import json
import platform
import sys
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
from bme_eating.data.packet_reader import inspect_ppg_layout
from bme_eating.data.preprocess import preprocess_attachment, write_preprocess_summary
from bme_eating.data.splits import create_subject_folds, load_subject_folds
from bme_eating.features.baseline import build_segment_features
from bme_eating.metrics import evaluate_events
from bme_eating.models.dtp_sqf import DTPSQF
from bme_eating.models.xgb_baseline import predict_xgboost, train_xgboost_fold
from bme_eating.postprocess import probabilities_to_events, tune_postprocess_parameters
from bme_eating.training.dtp_trainer import train_dtp_fold


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
    layouts = [
        inspect_ppg_layout(path, maximum_rows=int(args.maximum_rows))
        for path in tqdm(selected["zip_path"], desc="Inspecting PPG layout")
    ]
    active_slots = 0
    for layout in layouts:
        fractions = layout["nonzero_fraction_by_slot"]
        for index, value in enumerate(fractions):
            if value > 0:
                active_slots = max(active_slots, index + 1)
    audit = {
        "records": int(len(records)),
        "subjects": int(records["subject_key"].nunique()),
        "events": int(len(events)),
        "positive_duration_events": int(events["valid_duration"].sum()),
        "nonpositive_duration_events": int((~events["valid_duration"]).sum()),
        "configured_ppg_samples_per_row": int(config["data"]["ppg_samples_per_row"]),
        "maximum_observed_nonzero_ppg_slot": active_slots,
        "layout_files_inspected": len(layouts),
        "layouts": layouts,
    }
    audit_path = output_root / "indices" / "schema_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
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
) -> list[dict[str, object]]:
    return preprocess_attachment(
        record,
        Path(segment_dir),
        data_config,
        compressed=compressed,
        overwrite=overwrite,
    )


def command_preprocess(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    data_root, output_root = resolve_roots(config)
    records, events = _indices(config, data_root, output_root)
    segment_dir = output_root / "segments"
    workers = int(args.workers or config["preprocess"]["workers"])
    all_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _preprocess_job,
                record._asdict(),
                str(segment_dir),
                dict(config["data"]),
                bool(config["preprocess"]["compression"]),
                bool(args.overwrite or config["preprocess"]["overwrite"]),
            )
            for record in records.itertuples(index=False)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            all_rows.extend(future.result())
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
    )
    print(
        json.dumps(
            {
                "segments": len(segments),
                "anchors": len(anchors),
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
) -> pd.DataFrame:
    return build_segment_features(
        segment_path,
        anchors,
        int(feature_config["window_seconds"]),
        bool(feature_config["include_dyadic"]),
        [int(value) for value in feature_config["motion_bucket_seconds"]],
        [int(value) for value in feature_config["ppg_bucket_seconds"]],
    )


def command_build_features(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, output_root = resolve_roots(config)
    anchors = pd.read_parquet(output_root / "indices" / "anchors.parquet")
    grouped = list(anchors.groupby("segment_id", sort=False))
    workers = int(args.workers or config["features"]["workers"])
    frames: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _feature_job,
                str(group.iloc[0].segment_path),
                group,
                dict(config["features"]),
            )
            for _, group in grouped
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
            frames.append(future.result())
    features = pd.concat(frames, ignore_index=True)
    feature_name = "baseline_dyadic" if config["features"]["include_dyadic"] else "baseline"
    path = output_root / "features" / f"{feature_name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False)
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
                    "matched_events": int(len(matched_relation)),
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
    features = pd.read_parquet(output_root / "features" / f"{feature_name}.parquet")
    subject_folds = load_subject_folds(output_root / "indices" / "subject_folds.json")
    experiment_dir = output_root / "experiments" / feature_name / f"fold_{args.fold}"
    model, columns, validation, test = train_xgboost_fold(
        features,
        subject_folds,
        int(args.fold),
        config["xgboost"],
        experiment_dir,
    )
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
    selected_postprocess = _tune_and_save_postprocess(
        validation_predictions, validation_truth, config, experiment_dir
    )
    test_subjects = set(test["subject_key"].unique())
    truth = events[
        events["subject_key"].isin(test_subjects)
        & events["valid_duration"]
        & (events["coverage"] == "full")
    ]
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
    anchors = pd.read_parquet(output_root / "indices" / "anchors.parquet")
    segments = pd.read_parquet(output_root / "indices" / "segments.parquet")
    events = pd.read_parquet(output_root / "indices" / "events.parquet")
    subject_folds = load_subject_folds(output_root / "indices" / "subject_folds.json")
    future = int(config["model"].get("future_context_seconds", 0))
    experiment_name = "dtp_sqf_causal" if future == 0 else f"dtp_sqf_future{future}"
    experiment_dir = output_root / "experiments" / experiment_name / f"fold_{args.fold}"
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

