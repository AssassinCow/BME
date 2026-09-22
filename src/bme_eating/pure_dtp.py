from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from bme_eating.config import load_config, resolve_roots
from bme_eating.data.splits import load_subject_folds
from bme_eating.fusion import align_prediction_frames, sha256_file, validate_frozen_baseline_fold
from bme_eating.metrics import partition_evaluation_events
from bme_eating.postprocess import tune_dual_ema_parameters


def validate_dtp_splits(
    oof: pd.DataFrame,
    test: pd.DataFrame,
    baseline_oof: pd.DataFrame,
    baseline_test: pd.DataFrame,
    subject_folds: dict[str, int],
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_oof, oof = align_prediction_frames(baseline_oof, oof)
    baseline_test, test = align_prediction_frames(baseline_test, test)
    expected_test = {str(subject) for subject, number in subject_folds.items() if number == fold}
    expected_oof = {str(subject) for subject, number in subject_folds.items() if number != fold}
    actual_test = set(test["subject_key"].astype(str))
    actual_oof = set(oof["subject_key"].astype(str))
    if actual_test != expected_test or actual_oof != expected_oof:
        raise ValueError(
            "DTP outer-train OOF / holdout subjects disagree with frozen subject folds"
        )
    if actual_test & actual_oof:
        raise ValueError("DTP OOF includes held-out subjects")
    if "calibration_fold" not in oof or oof["calibration_fold"].isna().any():
        raise ValueError("DTP OOF is missing inner crossfit partition assignments")
    assignments = oof[["subject_key", "calibration_fold"]].drop_duplicates()
    if (
        assignments["subject_key"].duplicated().any()
        or assignments["calibration_fold"].nunique() != 3
    ):
        raise ValueError("Each OOF subject must belong to exactly one of three inner partitions")
    if "calibration_fold" in test:
        raise ValueError("DTP holdout must be the ensemble, not an inner-partition prediction")
    return oof, test


def window_diagnostics(predictions: pd.DataFrame, anchors: pd.DataFrame) -> dict[str, Any]:
    keys = ["subject_key", "session_id", "timestamp_ms"]
    labels = anchors[[*keys, "state_target", "state_loss_mask"]]
    if labels.duplicated(keys).any():
        raise ValueError("Anchor labels contain duplicate alignment keys")
    aligned = predictions[[*keys, "state_probability"]].merge(
        labels, on=keys, how="left", validate="one_to_one"
    )
    if aligned["state_target"].isna().any() or aligned["state_loss_mask"].isna().any():
        raise ValueError("DTP predictions have timestamps without anchor labels")
    eligible = aligned["state_loss_mask"].to_numpy(dtype=float) > 0
    targets = aligned["state_target"].to_numpy(dtype=float)[eligible] > 0
    probabilities = aligned["state_probability"].to_numpy(dtype=float)[eligible]
    return {
        "total_windows": len(predictions),
        "eligible_windows": len(targets),
        "positive_windows": int(targets.sum()),
        "positive_fraction": float(targets.mean()) if len(targets) else None,
        "auprc": float(average_precision_score(targets, probabilities)) if targets.any() else None,
        "auroc": float(roc_auc_score(targets, probabilities))
        if len(np.unique(targets)) == 2
        else None,
        "brier": float(brier_score_loss(targets, probabilities)) if len(targets) else None,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_pure_dtp(args: argparse.Namespace) -> Path:
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if not re.fullmatch(r"pure_dtp_[A-Za-z0-9_-]{1,64}", args.run_name):
        raise ValueError(
            "--run-name must start with pure_dtp_ and contain only ASCII letters, digits, _ or -"
        )
    from bme_eating.cli import _evaluate_prediction_file, _validate_v4_prediction_source

    config = load_config(args.config)
    _, output_root = resolve_roots(config)
    source_dir, source_hashes = _validate_v4_prediction_source(
        output_root, args.source_run, args.fold
    )
    baseline_name = str(config["experiment"]["baseline_name"])
    baseline_info = validate_frozen_baseline_fold(
        output_root,
        baseline_name,
        args.fold,
        str(config["experiment"]["baseline_source_commit"]),
    )
    baseline_dir = Path(baseline_info["directory"])
    output_dir = output_root / "experiments" / args.run_name / f"fold_{args.fold}"
    if output_dir.exists():
        raise FileExistsError(f"Diagnostic output already exists: {output_dir}")

    print("[1/4] Verifying frozen DTP predictions, split, and data fingerprints...", flush=True)
    subject_folds = load_subject_folds(output_root / "indices" / "subject_folds.json")
    oof, test = validate_dtp_splits(
        pd.read_parquet(source_dir / "dtp_oof_predictions.parquet"),
        pd.read_parquet(source_dir / "dtp_test_predictions.parquet"),
        pd.read_parquet(baseline_dir / "validation_predictions.parquet"),
        pd.read_parquet(baseline_dir / "test_predictions.parquet"),
        subject_folds,
        args.fold,
    )
    events = pd.read_parquet(output_root / "indices" / "events.parquet")
    oof_truth, oof_ignore = partition_evaluation_events(events, set(oof["subject_key"]))
    test_truth, test_ignore = partition_evaluation_events(events, set(test["subject_key"]))
    anchors = pd.read_parquet(
        output_root / "indices" / "anchors.parquet",
        columns=["subject_key", "session_id", "timestamp_ms", "state_target", "state_loss_mask"],
    )
    output_dir.mkdir(parents=True)
    grid = config["fusion"]["postprocess_search"]
    iou = float(config["postprocess"]["iou_threshold"])
    method = str(config["postprocess"].get("matching_method", "max_cardinality_iou"))

    print("[2/4] Selecting DTP-only dual EMA on outer-train crossfit OOF...", flush=True)
    selected, trials = tune_dual_ema_parameters(
        oof,
        oof_truth,
        grid,
        iou,
        ignore=oof_ignore,
        matching_method=method,
        workers=args.workers,
        checkpoint_path=output_dir / "search.checkpoint.jsonl",
    )
    selected.update({"iou_threshold": iou, "matching_method": method})
    _write_json(output_dir / "selected_postprocess.json", selected)
    trials.to_csv(output_dir / "postprocess_trials.csv", index=False)

    print("[3/4] Evaluating frozen setting once on holdout...", flush=True)
    oof_events, oof_metrics, _ = _evaluate_prediction_file(oof, oof_truth, selected, oof_ignore)
    test_events, test_metrics, failures = _evaluate_prediction_file(
        test, test_truth, selected, test_ignore
    )
    oof_events.to_csv(output_dir / "oof_events.csv", index=False)
    test_events.to_csv(output_dir / "test_events.csv", index=False)
    failures.to_csv(output_dir / "test_failure_cases.csv", index=False)
    baseline_metrics = json.loads((baseline_dir / "test_metrics.json").read_text(encoding="utf-8"))
    summary = {
        "scope": "pure_dtp_diagnostic_only_not_a_confirmation_fold",
        "fold": args.fold,
        "source_run": args.source_run,
        "prediction_model": "three_crossfit_DTP_checkpoints_averaged_on_holdout",
        "selection": "dual_ema_selected_only_on_outer_train_DTP_OOF",
        "oof_event_metrics_in_sample_postprocess_selection": oof_metrics,
        "holdout_event_metrics": test_metrics,
        "baseline_holdout_event_metrics": baseline_metrics,
        "oof_window": window_diagnostics(oof, anchors),
        "holdout_window": window_diagnostics(test, anchors),
        "limitations": [
            "Fold 0 has been observed in earlier fusion work; this is a development diagnostic, not independent confirmation.",
            "OOF event metrics are optimistic because the postprocessor was selected on those OOF labels.",
            "The holdout was not used to select checkpoints or postprocessing; do not retune after reading it.",
            "Event matching uses the repository's max-cardinality IoU>0.25 implementation, not an official evaluation API.",
        ],
    }
    _write_json(output_dir / "diagnostics.json", _json_safe(summary))

    print("[4/4] Recording immutable source fingerprints and output hashes...", flush=True)
    project_root = Path(__file__).resolve().parents[2]
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    artifact_names = (
        "selected_postprocess.json",
        "postprocess_trials.csv",
        "search.checkpoint.jsonl",
        "oof_events.csv",
        "test_events.csv",
        "test_failure_cases.csv",
        "diagnostics.json",
    )
    manifest = {
        "kind": "pure_dtp_holdout_diagnostic",
        "git": {"commit": git_head, "dirty": dirty},
        "config_sha256": sha256_file(Path(config["_config_path"])),
        "random_seeds": {
            "training": config["training"]["random_seed"],
            "split": config["data"]["split_seed"],
        },
        "fold": args.fold,
        "source_run": args.source_run,
        "source_manifest_sha256": sha256_file(source_dir / "run_manifest.json"),
        "source_prediction_hashes": source_hashes,
        "baseline_artifact_hashes": baseline_info["artifact_hashes"],
        "input_hashes": baseline_info["input_hashes"],
        "subject_counts": {
            "oof": int(oof["subject_key"].nunique()),
            "holdout": int(test["subject_key"].nunique()),
        },
        "search_grid": grid,
        "artifacts": {name: sha256_file(output_dir / name) for name in artifact_names},
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    primary = test_metrics[method]
    print(
        json.dumps(
            {
                "f1": primary["f1"],
                "tp": primary["true_positive"],
                "fp": primary["false_positive"],
                "fn": primary["false_negative"],
                "baseline_f1": baseline_metrics[method]["f1"],
                "holdout_auprc": summary["holdout_window"]["auprc"],
                "output": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return output_dir
