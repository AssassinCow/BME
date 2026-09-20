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
from tqdm import tqdm

from bme_eating.config import load_config, resolve_roots
from bme_eating.data.labels import build_anchor_index, classify_event_coverage
from bme_eating.data.manifest import build_secure_indices
from bme_eating.data.multisection import (
    audit_repeated_header_attachments,
    validate_multisection_preprocess_policy,
    write_quarantine_manifest,
)
from bme_eating.data.packet_reader import UnsupportedSensorFormatError, inspect_ppg_layout
from bme_eating.data.preprocess import (
    assign_virtual_sessions,
    preprocess_attachment,
    write_preprocess_summary,
)
from bme_eating.data.quality import (
    build_quality_report,
    validate_configured_expectations,
    validate_quality_gate,
    validate_quality_invariants,
)
from bme_eating.data.splits import create_subject_folds, load_subject_folds
from bme_eating.features.baseline import build_segment_features
from bme_eating.metrics import evaluate_events, partition_evaluation_events
from bme_eating.models.dtp_sqf import DTPSQF
from bme_eating.postprocess import probabilities_to_events, tune_postprocess_parameters

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

_FEATURE_CACHE_SCHEMA_VERSION = "v2.1-nonoverlap-terminal-validity"


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
        subject_aliases=dict(config["data"].get("subject_aliases", {})),
        privacy_root=Path(config.get("_output_base_path", output_root)),
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


def _audit_layout_job(zip_path: str, maximum_rows: int) -> tuple[str, dict[str, object]]:
    try:
        layout = inspect_ppg_layout(zip_path, maximum_rows=maximum_rows)
    except _RECOVERABLE_INPUT_ERRORS as error:
        layout = _audit_layout_issue(zip_path, error)
    except Exception as error:  # noqa: BLE001 - worker boundary must preserve diagnostics
        layout = _audit_layout_issue(zip_path, error)
    return zip_path, layout


def _inspect_selected_layouts(
    selected_fingerprints: list[dict[str, object]],
    maximum_rows: int,
    workers: int,
    checkpoint_path: Path,
    schema_zips: str,
    completed: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    pending = [
        str(fingerprint["zip_path"])
        for fingerprint in selected_fingerprints
        if str(fingerprint["zip_path"]) not in completed
    ]
    progress = tqdm(
        total=len(selected_fingerprints),
        initial=len(selected_fingerprints) - len(pending),
        desc=f"Inspecting PPG layout ({workers} workers)",
    )

    def save_result(zip_path: str, layout: dict[str, object]) -> None:
        completed[zip_path] = layout
        _write_audit_checkpoint(
            checkpoint_path,
            schema_zips,
            maximum_rows,
            selected_fingerprints,
            completed,
        )
        progress.update(1)

    try:
        if workers == 1:
            for zip_path in pending:
                result_path, layout = _audit_layout_job(zip_path, maximum_rows)
                save_result(result_path, layout)
        elif pending:
            with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
                futures = {
                    executor.submit(_audit_layout_job, zip_path, maximum_rows): zip_path
                    for zip_path in pending
                }
                for future in as_completed(futures):
                    zip_path = futures[future]
                    try:
                        result_path, layout = future.result()
                    except Exception as error:
                        raise RuntimeError(
                            f"Schema audit worker failed while processing {zip_path}"
                        ) from error
                    save_result(result_path, layout)
    finally:
        progress.close()
    return [
        completed[str(fingerprint["zip_path"])]
        for fingerprint in selected_fingerprints
    ]


def command_environment(_: argparse.Namespace) -> None:
    try:
        import xgboost
    except ImportError as error:
        raise SystemExit("XGBoost is unavailable in the active environment.") from error
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
    records, events = _indices(config, data_root, output_root)
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
    requested_workers = getattr(args, "workers", None)
    workers = int(
        config.get("audit", {}).get("workers", 1)
        if requested_workers is None
        else requested_workers
    )
    layouts = _inspect_selected_layouts(
        selected_fingerprints,
        maximum_rows,
        workers,
        checkpoint_path,
        str(args.schema_zips),
        completed,
    )
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
    allowed_statuses = {"documented_text", "recovered_text_suffix", "repeated_header"}
    bad_statuses = {
        status: count
        for status, count in layout_status_counts.items()
        if status not in allowed_statuses and count
    }
    if args.schema_zips == "all" and len(layouts) != len(records):
        raise SystemExit("Full schema audit did not inspect every indexed attachment.")
    if bad_statuses:
        raise SystemExit(
            "Schema audit found unsupported or invalid attachments: "
            + json.dumps(bad_statuses, sort_keys=True)
        )
    if active_slots > int(config["data"]["ppg_samples_per_row"]):
        raise SystemExit(
            "Observed nonzero PPG slots exceed ppg_samples_per_row; update config before preprocessing."
        )


def command_audit_multisection(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, output_root = resolve_roots(config)
    index_dir = output_root / "indices"
    records_path = index_dir / "records.parquet"
    schema_audit_path = index_dir / "schema_audit.json"
    if not records_path.exists() or not schema_audit_path.exists():
        raise SystemExit("Run the full schema audit before the multisection audit.")
    records = pd.read_parquet(records_path)
    schema_audit = json.loads(schema_audit_path.read_text(encoding="utf-8"))
    report = audit_repeated_header_attachments(
        records,
        schema_audit,
        int(config["data"]["ppg_samples_per_row"]),
        show_progress=True,
        workers=int(getattr(args, "workers", None) or config["preprocess"]["workers"]),
    )
    output_path = index_dir / "multisection_audit.json"
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(output_path)
    summary = {
        "attachments_audited": report["attachments_audited"],
        "classification_counts": report["classification_counts"],
        "automatic_recovery_performed": False,
    }
    print(output_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    blocking = {
        key: value
        for key, value in report["classification_counts"].items()
        if key != "exact_duplicate_overlap" and int(value) > 0
    }
    if blocking:
        raise SystemExit(
            "Multisection audit found attachments that are not safe for exact deduplication: "
            + json.dumps(blocking, sort_keys=True)
        )


def _preprocess_job(
    record: dict[str, object],
    segment_dir: str,
    data_config: dict[str, object],
    compressed: bool,
    overwrite: bool,
    parser_mode: str = "standard",
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    try:
        rows = preprocess_attachment(
            record,
            Path(segment_dir),
            data_config,
            compressed=compressed,
            overwrite=overwrite,
            parser_mode=parser_mode,
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
    index_dir = output_root / "indices"
    schema_audit_path = index_dir / "schema_audit.json"
    multisection_audit_path = index_dir / "multisection_audit.json"
    if not schema_audit_path.exists() or not multisection_audit_path.exists():
        raise RuntimeError(
            "Full schema and multisection audits are required before preprocessing"
        )
    schema_audit = json.loads(schema_audit_path.read_text(encoding="utf-8"))
    multisection_audit = json.loads(
        multisection_audit_path.read_text(encoding="utf-8")
    )
    exact_hashes, quarantined = validate_multisection_preprocess_policy(
        records,
        schema_audit,
        multisection_audit,
        expected_exact=int(
            config["quality_gates"]["expected_recovered_multisection_deduplicated"]
        ),
        expected_quarantined=int(
            config["quality_gates"]["expected_quarantined_multisection"]
        ),
        expected_documented=int(config["quality_gates"]["expected_documented_text"]),
        expected_text_suffix=int(
            config["quality_gates"]["expected_recovered_text_suffix"]
        ),
    )
    write_quarantine_manifest(index_dir, quarantined)
    quarantined_hashes = set(quarantined)
    segment_dir = output_root / "segments"
    for digest in quarantined_hashes:
        token = digest[:20]
        for stale_path in segment_dir.glob(f"{token}_s*.npz"):
            stale_path.unlink(missing_ok=True)
        for stale_path in segment_dir.glob(f"{token}_s*.npz.tmp"):
            stale_path.unlink(missing_ok=True)
    selected_records = records[
        ~records["zip_sha256"].astype(str).str.lower().isin(quarantined_hashes)
    ]
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
                (
                    "exact_multisection"
                    if str(record.zip_sha256).lower() in exact_hashes
                    else "standard"
                ),
            ): record._asdict()
            for record in selected_records.itertuples(index=False)
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
    if preprocess_issues:
        raise RuntimeError(
            f"Preprocessing failed for {len(preprocess_issues)} attachment(s); see {issues_path}"
        )
    if not all_rows:
        raise RuntimeError(f"Preprocessing produced no segments; see {issues_path}")
    segments = assign_virtual_sessions(
        pd.DataFrame(all_rows),
        int(float(config["data"]["session_join_gap_seconds"]) * 1000),
    )
    segment_index_path = output_root / "indices" / "segments.parquet"
    write_preprocess_summary(segments, segment_index_path)
    events = classify_event_coverage(
        events,
        segments,
        int(config["data"]["output_step_seconds"]),
    )
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
    quality_report = build_quality_report(output_root)
    validate_configured_expectations(quality_report, config["quality_gates"])
    validate_quality_invariants(quality_report)
    print(
        json.dumps(
            {
                "segments": len(segments),
                "anchors": len(anchors),
                "preprocess_issues": len(preprocess_issues),
                "quarantined_attachments": len(quarantined_hashes),
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
    context_segments: pd.DataFrame | None = None,
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
            context_segments=context_segments,
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
    context_segments: pd.DataFrame | None = None,
) -> Path:
    path = Path(segment_path)
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(_FEATURE_CACHE_SCHEMA_VERSION.encode())
    digest.update(
        json.dumps(feature_config, sort_keys=True, separators=(",", ":")).encode()
    )
    digest.update(path.name.encode())
    digest.update(f"{stat.st_size}|{stat.st_mtime_ns}".encode())
    if context_segments is not None:
        for row in context_segments.sort_values("segment_id").itertuples(index=False):
            context_path = Path(str(row.segment_path))
            context_stat = context_path.stat()
            digest.update(str(row.segment_id).encode())
            digest.update(f"{context_stat.st_size}|{context_stat.st_mtime_ns}".encode())
    digest.update(pd.util.hash_pandas_object(anchors, index=True).to_numpy().tobytes())
    return cache_dir / f"{path.stem}-{digest.hexdigest()[:16]}.parquet"


def command_build_features(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, output_root = resolve_roots(config)
    validate_quality_gate(output_root)
    anchors = pd.read_parquet(output_root / "indices" / "anchors.parquet")
    segments = pd.read_parquet(output_root / "indices" / "segments.parquet")
    grouped = list(anchors.groupby("segment_id", sort=False))
    workers = int(args.workers or config["features"]["workers"])
    feature_name = "baseline_dyadic" if config["features"]["include_dyadic"] else "baseline"
    cache_dir = output_root / "features" / ".cache" / feature_name
    jobs = []
    for _, group in grouped:
        session_id = str(group.iloc[0].get("session_id", group.iloc[0].segment_id))
        context = segments[segments["session_id"].astype(str) == session_id].copy()
        segment_path = str(group.iloc[0].segment_path)
        cache_path = _feature_cache_path(
            cache_dir,
            segment_path,
            group,
            dict(config["features"]),
            context,
        )
        jobs.append((segment_path, group, cache_path, context))
    cached_count = sum(cache_path.exists() for _, _, cache_path, _ in jobs)
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
                context,
            ): segment_path
            for segment_path, group, cache_path, context in jobs
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
    ignore: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
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
    primary_method = str(
        postprocess_config.get("matching_method", "max_cardinality_iou")
    )
    methods = list(dict.fromkeys((primary_method, "hungarian_iou_legacy", "greedy")))
    primary_matches = pd.DataFrame()
    for method in methods:
        metrics, matches = evaluate_events(
            truth,
            events,
            iou_threshold=float(postprocess_config["iou_threshold"]),
            method=method,
            ignore=ignore,
        )
        exposure_hours = 0.0
        if len(predictions):
            for _, group in predictions.groupby(["subject_key", "session_id"]):
                exposure_hours += max(
                    0.0,
                    (float(group["timestamp_ms"].max())
                    - float(group["timestamp_ms"].min())
                    + 3000.0)
                    / 3_600_000.0,
                )
        metrics["false_positives_per_observed_hour"] = (
            float(metrics["false_positive"]) / exposure_hours if exposure_hours else 0.0
        )
        output[method] = metrics
        if method == primary_method:
            primary_matches = matches
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
            coverage_summary: dict[str, object] = {}
            truth_with_band = truth.copy()
            ratios = truth_with_band.get(
                "coverage_ratio", pd.Series(1.0, index=truth_with_band.index)
            ).fillna(0.0)
            truth_with_band["coverage_band"] = pd.cut(
                ratios,
                bins=[-np.inf, 0.9, 0.99, np.inf],
                labels=["le_0p90", "0p90_to_0p99", "gt_0p99"],
                include_lowest=True,
            ).astype(str)
            matched_keys = set(
                zip(
                    matches.get("subject_key", []),
                    matches.get("truth_start_ms", []),
                    matches.get("truth_end_ms", []),
                )
            )
            for band, band_truth in truth_with_band.groupby("coverage_band", sort=True):
                hits = sum(
                    (str(row.subject_key), int(row.start_ms), int(row.end_ms))
                    in matched_keys
                    for row in band_truth.itertuples(index=False)
                )
                coverage_summary[str(band)] = {
                    "truth_events": int(len(band_truth)),
                    "matched_events": int(hits),
                    "sensitivity": hits / len(band_truth) if len(band_truth) else None,
                }
            output["coverage"] = coverage_summary
    strict_metrics, _ = evaluate_events(
        truth,
        events,
        iou_threshold=float(postprocess_config["iou_threshold"]),
        method=primary_method,
    )
    output["strict_no_ignore"] = strict_metrics
    output["primary_method"] = primary_method
    by_subject: dict[str, dict[str, float]] = {}
    for subject_key in sorted(
        set(truth.get("subject_key", [])) | set(events.get("subject_key", []))
    ):
        subject_truth = truth[truth["subject_key"] == subject_key]
        subject_events = events[events["subject_key"] == subject_key]
        subject_ignore = (
            ignore[ignore["subject_key"] == subject_key]
            if ignore is not None
            else None
        )
        subject_metrics, _ = evaluate_events(
            subject_truth,
            subject_events,
            iou_threshold=float(postprocess_config["iou_threshold"]),
            method=primary_method,
            ignore=subject_ignore,
        )
        by_subject[str(subject_key)] = subject_metrics
    output["by_subject"] = by_subject
    output["evaluation_counts"] = {
        "evaluable_truth": int(len(truth)),
        "ignored_truth": int(len(ignore)) if ignore is not None else 0,
        "predicted_events": int(len(events)),
    }
    matched_truth = set(
        zip(
            primary_matches.get("subject_key", []),
            primary_matches.get("truth_start_ms", []),
            primary_matches.get("truth_end_ms", []),
        )
    )
    matched_prediction = set(
        zip(
            primary_matches.get("subject_key", []),
            primary_matches.get("prediction_start_ms", []),
            primary_matches.get("prediction_end_ms", []),
        )
    )
    failures: list[dict[str, object]] = []
    for row in truth.itertuples(index=False):
        key = (str(row.subject_key), int(row.start_ms), int(row.end_ms))
        if key not in matched_truth:
            failures.append(
                {
                    "failure_type": "false_negative",
                    "subject_key": str(row.subject_key),
                    "start_ms": int(row.start_ms),
                    "end_ms": int(row.end_ms),
                    "score": np.nan,
                    "hand_relation": str(getattr(row, "hand_relation", "unknown")),
                    "coverage_ratio": float(getattr(row, "coverage_ratio", np.nan)),
                }
            )
    ignore_frame = (
        ignore if ignore is not None else pd.DataFrame(columns=["subject_key", "start_ms", "end_ms"])
    )
    for row in events.itertuples(index=False):
        key = (str(row.subject_key), int(row.start_ms), int(row.end_ms))
        if key in matched_prediction:
            continue
        subject_ignore = ignore_frame[ignore_frame["subject_key"] == row.subject_key]
        ignored = False
        if len(subject_ignore):
            ignored = bool(
                (
                    np.minimum(int(row.end_ms), subject_ignore["end_ms"].to_numpy())
                    > np.maximum(int(row.start_ms), subject_ignore["start_ms"].to_numpy())
                ).any()
            )
        if not ignored:
            failures.append(
                {
                    "failure_type": "false_positive",
                    "subject_key": str(row.subject_key),
                    "start_ms": int(row.start_ms),
                    "end_ms": int(row.end_ms),
                    "score": float(row.score),
                    "hand_relation": "background",
                    "coverage_ratio": np.nan,
                }
            )
    failure_columns = [
        "failure_type",
        "subject_key",
        "start_ms",
        "end_ms",
        "score",
        "hand_relation",
        "coverage_ratio",
    ]
    return events, output, pd.DataFrame(failures, columns=failure_columns)


def _tune_and_save_postprocess(
    validation_predictions: pd.DataFrame,
    validation_truth: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    ignore: pd.DataFrame | None = None,
) -> dict[str, float]:
    best, trials = tune_postprocess_parameters(
        validation_predictions,
        validation_truth,
        config["postprocess_search"],
        float(config["postprocess"]["iou_threshold"]),
        checkpoint_path=output_dir / "postprocess_search.checkpoint.jsonl",
        ignore=ignore,
        matching_method=str(
            config["postprocess"].get("matching_method", "max_cardinality_iou")
        ),
        workers=int(config["postprocess_search"].get("workers", 1)),
    )
    best["iou_threshold"] = float(config["postprocess"]["iou_threshold"])
    best["matching_method"] = str(
        config["postprocess"].get("matching_method", "max_cardinality_iou")
    )
    high_values = sorted(float(value) for value in trials["high_threshold"].unique())
    threshold_at_boundary = len(high_values) < 3 or float(best["high_threshold"]) in {
        high_values[0],
        high_values[-1],
    }
    best["threshold_at_search_boundary"] = threshold_at_boundary
    (output_dir / "selected_postprocess.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trials.to_csv(output_dir / "postprocess_trials.csv", index=False)
    if (
        bool(config["postprocess_search"].get("require_interior_threshold", True))
        and threshold_at_boundary
    ):
        raise RuntimeError(
            "Best high threshold is on the search boundary; expand the validation-only grid"
        )
    return best


def command_train_xgb(args: argparse.Namespace) -> None:
    from bme_eating.models.xgb_baseline import predict_xgboost, train_xgboost_fold

    config = load_config(args.config)
    _, output_root = resolve_roots(config)
    validate_quality_gate(output_root)
    feature_name = "baseline_dyadic" if config["features"]["include_dyadic"] else "baseline"
    print("[1/5] Loading features and subject folds...", flush=True)
    features = pd.read_parquet(output_root / "features" / f"{feature_name}.parquet")
    subject_folds = load_subject_folds(output_root / "indices" / "subject_folds.json")
    experiment_dir = output_root / "experiments" / feature_name / f"fold_{args.fold}"
    print("[2/5] Tuning and fitting XGBoost...", flush=True)
    model, columns, validation_predictions, test = train_xgboost_fold(
        features,
        subject_folds,
        int(args.fold),
        config["xgboost"],
        experiment_dir,
        resume_search=not bool(getattr(args, "no_resume", False)),
    )
    print("[3/5] Predicting validation and test windows...", flush=True)
    validation_predictions.to_parquet(
        experiment_dir / "validation_predictions.parquet", index=False
    )
    predictions = predict_xgboost(model, test, columns)
    predictions.to_parquet(experiment_dir / "test_predictions.parquet", index=False)
    events = pd.read_parquet(output_root / "indices" / "events.parquet")
    validation_subjects = set(validation_predictions["subject_key"].unique())
    validation_truth, validation_ignore = partition_evaluation_events(
        events, validation_subjects
    )
    print(
        "[4/5] Tuning event postprocessing on CPU "
        f"({int(config['postprocess_search'].get('workers', 1))} workers)...",
        flush=True,
    )
    selected_postprocess = _tune_and_save_postprocess(
        validation_predictions,
        validation_truth,
        config,
        experiment_dir,
        validation_ignore,
    )
    test_subjects = set(test["subject_key"].unique())
    truth, test_ignore = partition_evaluation_events(events, test_subjects)
    print("[5/5] Evaluating the held-out fold...", flush=True)
    predicted_events, metrics, failures = _evaluate_prediction_file(
        predictions, truth, selected_postprocess, test_ignore
    )
    predicted_events.to_csv(experiment_dir / "test_events.csv", index=False)
    failures.to_csv(experiment_dir / "test_failure_cases.csv", index=False)
    (experiment_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if float(metrics[str(metrics["primary_method"])]["sensitivity"]) <= 0:
        raise RuntimeError("Held-out fold has zero event recall; model upgrade is blocked")
    print(experiment_dir)


def command_train_dtp(args: argparse.Namespace) -> None:
    from bme_eating.training.dtp_trainer import train_dtp_fold

    config = load_config(args.config)
    _, output_root = resolve_roots(config)
    validate_quality_gate(output_root)
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
        {**config["postprocess"], "search": config["postprocess_search"]},
        experiment_dir,
        Path(args.resume) if args.resume else None,
    )
    predictions = pd.read_parquet(experiment_dir / "test_predictions.parquet")
    validation_predictions = pd.read_parquet(
        experiment_dir / "best_validation_predictions.parquet"
    )
    validation_subjects = set(validation_predictions["subject_key"].unique())
    validation_truth, validation_ignore = partition_evaluation_events(
        events, validation_subjects
    )
    print(
        "[3/4] Tuning event postprocessing on CPU "
        f"({int(config['postprocess_search'].get('workers', 1))} workers)...",
        flush=True,
    )
    selected_postprocess = _tune_and_save_postprocess(
        validation_predictions,
        validation_truth,
        config,
        experiment_dir,
        validation_ignore,
    )
    test_subjects = {
        subject for subject, fold in subject_folds.items() if fold == int(args.fold)
    }
    truth, test_ignore = partition_evaluation_events(events, test_subjects)
    print("[4/4] Evaluating the held-out fold...", flush=True)
    predicted_events, metrics, failures = _evaluate_prediction_file(
        predictions, truth, selected_postprocess, test_ignore
    )
    predicted_events.to_csv(experiment_dir / "test_events.csv", index=False)
    failures.to_csv(experiment_dir / "test_failure_cases.csv", index=False)
    (experiment_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if float(metrics[str(metrics["primary_method"])]["sensitivity"]) <= 0:
        raise RuntimeError("Held-out fold has zero event recall; model upgrade is blocked")
    print(checkpoint)


def command_evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, output_root = resolve_roots(config)
    predictions = pd.read_parquet(args.predictions)
    events = pd.read_parquet(output_root / "indices" / "events.parquet")
    subjects = set(predictions["subject_key"].unique())
    truth, ignore = partition_evaluation_events(events, subjects)
    predicted_events, metrics, failures = _evaluate_prediction_file(
        predictions, truth, config["postprocess"], ignore
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    predicted_events.to_csv(output_dir / "events.csv", index=False)
    failures.to_csv(output_dir / "failure_cases.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def command_smoke_model(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run this command on the RTX 4080 computer.")
    device = torch.device("cuda")
    model = DTPSQF(config["model"]).to(device)
    future = model.future_context_seconds
    batch_size = int(args.batch_size)
    motion_block_count = sum(model.motion_bucket_counts)
    ppg_block_count = sum(model.ppg_bucket_counts)
    motion_samples = model.motion_block_seconds * 100
    ppg_samples = model.ppg_block_seconds * 50
    batch: dict[str, torch.Tensor] = {
        "motion_blocks": torch.randn(
            batch_size, motion_block_count, 12, motion_samples, device=device
        ),
        "motion_valid": torch.ones(batch_size, motion_block_count, device=device),
        "ppg_blocks": torch.randn(
            batch_size, ppg_block_count, 2, ppg_samples, device=device
        ),
        "ppg_quality": torch.rand(batch_size, ppg_block_count, 8, device=device),
        "ppg_valid": torch.ones(batch_size, ppg_block_count, device=device),
    }
    if future:
        batch.update(
            {
                "future_motion_blocks": torch.randn(
                    batch_size,
                    future // model.motion_block_seconds,
                    12,
                    motion_samples,
                    device=device,
                ),
                "future_motion_valid": torch.ones(
                    batch_size, future // model.motion_block_seconds, device=device
                ),
                "future_ppg_blocks": torch.randn(
                    batch_size,
                    future // model.ppg_block_seconds,
                    2,
                    ppg_samples,
                    device=device,
                ),
                "future_ppg_quality": torch.rand(
                    batch_size, future // model.ppg_block_seconds, 8, device=device
                ),
                "future_ppg_valid": torch.ones(
                    batch_size, future // model.ppg_block_seconds, device=device
                ),
            }
        )
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(batch)
    print({key: tuple(value.shape) for key, value in output.items()})
    print({"peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()})

