from __future__ import annotations

import argparse
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
    validate_quality_invariants,
)
from bme_eating.data.splits import create_subject_folds

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
    return [completed[str(fingerprint["zip_path"])] for fingerprint in selected_fingerprints]


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
            records.sort_values("zip_path")
            .groupby("subject_key", as_index=False)
            .head(1)
            .head(count)
        )
    maximum_rows = int(args.maximum_rows)
    selected_fingerprints = _selected_zip_fingerprints(selected)
    checkpoint_path = _audit_checkpoint_path(output_root)
    completed = (
        {}
        if getattr(args, "no_resume", False)
        else _load_audit_checkpoint(
            checkpoint_path,
            str(args.schema_zips),
            maximum_rows,
            selected_fingerprints,
        )
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
    temporary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
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
        raise RuntimeError("Full schema and multisection audits are required before preprocessing")
    schema_audit = json.loads(schema_audit_path.read_text(encoding="utf-8"))
    multisection_audit = json.loads(multisection_audit_path.read_text(encoding="utf-8"))
    exact_hashes, quarantined = validate_multisection_preprocess_policy(
        records,
        schema_audit,
        multisection_audit,
        expected_exact=int(config["quality_gates"]["expected_recovered_multisection_deduplicated"]),
        expected_quarantined=int(config["quality_gates"]["expected_quarantined_multisection"]),
        expected_documented=int(config["quality_gates"]["expected_documented_text"]),
        expected_text_suffix=int(config["quality_gates"]["expected_recovered_text_suffix"]),
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
                rows, issue = (
                    [],
                    {
                        "status": "preprocess_worker_error",
                        "zip_path": str(record.get("zip_path", "")),
                        "zip_sha256": str(record.get("zip_sha256", "")),
                        "subject_key": str(record.get("subject_key", "")),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
            except Exception as error:  # noqa: BLE001 - preserve worker diagnostics
                record = futures[future]
                rows, issue = (
                    [],
                    {
                        "status": "preprocess_worker_unexpected_error",
                        "zip_path": str(record.get("zip_path", "")),
                        "zip_sha256": str(record.get("zip_sha256", "")),
                        "subject_key": str(record.get("subject_key", "")),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
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
