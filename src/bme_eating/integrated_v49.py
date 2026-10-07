from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bme_eating.hierarchical_artifacts import (
    HierarchicalRun,
    sha256_file,
    validate_run_name,
    write_json_atomic,
)
from bme_eating.proposals_v4 import interval_iou, validate_proposal_lineage


def is_v49(config: dict[str, Any]) -> bool:
    return config.get("decoder", {}).get("candidate_protocol") == "v4.9"


def write_failure_report(root: Path, stage: str, error: BaseException) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    archive = root / "failure_evidence" / f"{stage}_{stamp}"
    evidence: dict[str, str] = {}
    evidence_error: str | None = None
    try:
        for source in sorted(root.rglob("*")):
            if not source.is_file() or "failure_evidence" in source.relative_to(root).parts:
                continue
            relative = source.relative_to(root)
            if any(part in {"model_bundle", "model_bundle.tmp"} for part in relative.parts):
                continue
            try:
                evidence[relative.as_posix()] = sha256_file(source)
                if source.suffix in {".json", ".yaml", ".csv"} or (
                    source.suffix == ".parquet"
                    and ("diagnostics" in relative.parts or "scores" in source.name)
                ):
                    target = archive / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            except MemoryError as hash_error:
                evidence_error = f"{type(hash_error).__name__}: {hash_error}"
                break
    except MemoryError as scan_error:
        evidence_error = f"{type(scan_error).__name__}: {scan_error}"
    report = {
        "protocol": "v49_failure_report_v1",
        "stage": stage,
        "passed": False,
        "deployable": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "timestamp_utc": stamp,
        "evidence_archive": archive.relative_to(root).as_posix(),
        "artifact_sha256": evidence,
    }
    if evidence_error is not None:
        report["evidence_error"] = evidence_error
        report["evidence_complete"] = False
    else:
        report["evidence_complete"] = True
    if stage in {"final_training", "export", "integrated_execution"}:
        quarantined = []
        for name in ("model_bundle", "model_bundle.tmp", "model_bundle.zip"):
            artifact = root / name
            if artifact.exists():
                destination = archive / "quarantined" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(artifact), str(destination))
                quarantined.append(destination.relative_to(root).as_posix())
        report["quarantined_bundles"] = quarantined
    write_json_atomic(archive / "failure_report.json", report)
    write_json_atomic(root / "failure_report.json", report)
    return root / "failure_report.json"


def run_cli_with_failure_report(main: Callable[[], None], stage: str) -> None:
    try:
        main()
    except (Exception, SystemExit, KeyboardInterrupt) as error:
        if isinstance(error, SystemExit) and error.code in {None, 0}:
            raise
        arguments = sys.argv[1:]
        if "--config" in arguments and "--run-name" in arguments:
            from bme_eating.config import load_config, resolve_artifact_roots

            config = load_config(arguments[arguments.index("--config") + 1])
            if is_v49(config):
                run_name = arguments[arguments.index("--run-name") + 1]
                validate_run_name(run_name)
                if isinstance(error, FileExistsError) or (
                    config.get("v49", {}).get("protocol") == "integrated_repair_v2"
                    and not run_name.startswith(config["v49"]["run_name_prefix"])
                ):
                    raise
                _, _, output_root = resolve_artifact_roots(config)
                branch = "final" if stage in {"final_training", "export"} else "experiments"
                failure_root = output_root / branch / run_name
                write_failure_report(failure_root, stage, error)
                if stage == "integrated_execution":
                    final_root = output_root / "final" / run_name
                    if (final_root / "model_bundle").exists():
                        write_failure_report(final_root, stage, error)
                status_path = failure_root / "execution_status.json"
                if status_path.is_file():
                    try:
                        status = json.loads(status_path.read_text(encoding="utf-8"))
                        status.update({
                            "status": "PAUSED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        })
                        write_json_atomic(status_path, status)
                    except (OSError, ValueError):
                        pass
        raise


def candidate_coverage(proposals: pd.DataFrame, truth: pd.DataFrame, windows: pd.DataFrame | None = None) -> dict[str, Any]:
    covered_ids: list[str] = []
    counts = {"covered": 0, "same_covered": 0, "different_covered": 0}
    by_source: dict[str, int] = {}
    strata: dict[str, dict[str, dict[str, int]]] = {}
    windows_by_session = {} if windows is None else {
        (str(subject), str(session)): group.sort_values("timestamp_ms")
        for (subject, session), group in windows.groupby(["subject_key", "session_id"], sort=False)
    }
    for event in truth.itertuples(index=False):
        candidates = proposals.loc[
            proposals["subject_key"].astype(str).eq(str(event.subject_key))
            & proposals["session_id"].astype(str).eq(str(event.session_id))
        ]
        matches = [
            row for row in candidates.itertuples(index=False)
            if interval_iou(
                int(row.coarse_start_ms), int(row.coarse_end_ms),
                int(event.start_ms), int(event.end_ms),
            ) > 0.25
        ]
        relation = str(getattr(event, "hand_relation", "unknown"))
        duration = (int(event.end_ms) - int(event.start_ms)) / 1000
        duration_bin = "short_le_30s" if duration <= 30 else ("medium_le_300s" if duration <= 300 else "long_gt_300s")
        gyro_bin = "unknown"
        session_windows = windows_by_session.get((str(event.subject_key), str(event.session_id)))
        if session_windows is not None and "gyro_valid_fraction" in session_windows:
            timestamps = session_windows["timestamp_ms"].to_numpy()
            event_windows = session_windows.iloc[np.searchsorted(timestamps, int(event.start_ms), side="right"):np.searchsorted(timestamps, int(event.end_ms), side="right")]
            if len(event_windows):
                gyro_bin = "observed" if event_windows["gyro_valid_fraction"].mean() >= 0.5 else "missing"
        for name, value in (("hand_relation", relation), ("duration", duration_bin),
                            ("wear_hand", str(getattr(event, "wear_hand", "unknown"))),
                            ("gyro_missingness", gyro_bin)):
            bucket = strata.setdefault(name, {}).setdefault(value, {"truth_count": 0, "covered": 0})
            bucket["truth_count"] += 1
            bucket["covered"] += int(bool(matches))
        if not matches:
            continue
        counts["covered"] += 1
        if relation in {"same", "different"}:
            counts[f"{relation}_covered"] += 1
        covered_ids.append(f"{event.subject_key}:{event.session_id}:{event.event_id}")
        for bit in range(7):
            if any(int(row.source_mask) & (1 << bit) for row in matches):
                by_source[str(1 << bit)] = by_source.get(str(1 << bit), 0) + 1
    return {
        **counts, "truth_count": len(truth), "covered_event_ids": covered_ids,
        "candidate_recall": counts["covered"] / max(len(truth), 1),
        "source_wise_covered": by_source, "strata": strata,
        "candidate_count": len(proposals),
        "independent_family_count": proposals.get("proposal_family_id", pd.Series(dtype=str)).nunique(),
        "gyro_missingness_candidates": proposals.get("gyro_missingness_bin", pd.Series(dtype=str)).value_counts().to_dict(),
        "formal_outer_holdout": True,
    }


def validate_v49_gate_manifest(
    output_root: Path, run_name: str, identity: dict[str, Any],
) -> dict[str, Any]:
    root = output_root / "experiments" / run_name
    configured_name = str(identity.get("v49_gate_manifest", "v49_gate_manifest.json"))
    path = root / configured_name
    if not path.is_file():
        raise RuntimeError(f"v4.9 final training requires {configured_name}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != ("v49_gate_manifest_v2" if configured_name == "v49_gate_manifest_v2.json" else "v49_gate_manifest_v1")
        or manifest.get("candidate_run") != run_name
        or manifest.get("folds") != list(range(5))
        or manifest.get("passed") is not True
        or manifest.get("crossfold_gate", {}).get("passed") is not True
    ):
        raise RuntimeError("v4.9 protocol lock is incomplete or failed")
    sensor_only = manifest.get("sensor_only_reference", {})
    if sensor_only != manifest["crossfold_gate"].get("sensor_only_reference"):
        raise RuntimeError("v4.9 protocol lock sensor-only comparison status changed")
    if sensor_only.get("status") == "not_available":
        if (
            manifest.get("s0_run") is not None
            or sensor_only.get("comparison_performed") is not False
            or sensor_only.get("reason") != "explicit_skip_s0"
            or "sensor_only_coverage_retained" in manifest["crossfold_gate"].get("checks", {})
        ):
            raise RuntimeError("v4.9 unavailable S0 must remain explicitly unassessed")
    elif sensor_only.get("status") == "verified":
        if (
            not manifest.get("s0_run")
            or sensor_only.get("run_name") != manifest["s0_run"]
            or sensor_only.get("comparison_performed") is not True
            or manifest["crossfold_gate"].get("checks", {}).get("sensor_only_coverage_retained") is not True
        ):
            raise RuntimeError("v4.9 S0 comparison lacks verified qualification")
    else:
        raise RuntimeError("v4.9 protocol lock lacks an explicit S0 comparison policy")
    identity_keys = ["resolved_config_sha256", "input_hashes", "git"]
    if configured_name == "v49_gate_manifest_v2.json":
        identity_keys.extend(("execution_source_identity", "execution_environment", "v49_gate_manifest"))
    for key in identity_keys:
        if key not in identity:
            raise RuntimeError(f"v4.9 protocol lock lacks identity: {key}")
        if manifest.get("identity", {}).get(key) != identity.get(key):
            raise RuntimeError(f"v4.9 protocol lock identity changed: {key}")
    evidence = manifest.get("evidence_sha256", {})
    if not evidence or any(f"fold_{fold}/run_manifest.json" not in evidence for fold in range(5)):
        raise RuntimeError("v4.9 protocol lock lacks all fold manifests")
    for relative, digest in evidence.items():
        artifact = (root / relative).resolve()
        if not artifact.is_relative_to(root.resolve()) or not artifact.is_file():
            raise RuntimeError(f"v4.9 protocol lock evidence is missing: {relative}")
        if sha256_file(artifact) != digest:
            raise RuntimeError(f"v4.9 protocol lock evidence changed: {relative}")
    references = manifest.get("reference_evidence_sha256", {})
    if not references:
        raise RuntimeError("v4.9 protocol lock lacks verified reference evidence")
    for relative, digest in references.items():
        artifact = (output_root / relative).resolve()
        if not artifact.is_relative_to(output_root.resolve()) or not artifact.is_file():
            raise RuntimeError(f"v4.9 reference evidence is missing: {relative}")
        if sha256_file(artifact) != digest:
            raise RuntimeError(f"v4.9 reference evidence changed: {relative}")
    return manifest


def _evaluation_cohort_identity(fold_root: Path) -> dict[str, str]:
    specifications = {
        "truth": ("evaluation/truth_events.parquet", [
            "subject_key", "session_id", "event_id", "start_ms", "end_ms", "hand_relation",
        ]),
        "ignore": ("evaluation/ignore_events.parquet", ["subject_key", "session_id", "start_ms", "end_ms"]),
        "timeline": ("outer/window_predictions.parquet", ["subject_key", "session_id", "timestamp_ms"]),
    }
    identity = {}
    for name, (relative, columns) in specifications.items():
        frame = pd.read_parquet(fold_root / relative, columns=columns)
        ordered = frame.sort_values(columns, kind="stable").reset_index(drop=True)
        identity[name] = hashlib.sha256(ordered.to_json(orient="split", index=False).encode()).hexdigest()
    return identity


def write_v49_raw_imu_gate(final_root: Path, config: dict[str, Any], deep_passed: bool) -> Path:
    report = json.loads((final_root / "deep_crossfit.json").read_text(encoding="utf-8"))
    lineages = report.get("lineage", [])
    checks = {
        "deep_promotion": bool(deep_passed),
        "raw_branch": config["verifier"].get("use_raw_imu_branch") is True,
        "five_folds": sorted(row.get("outer_fold", -1) for row in lineages) == list(range(5)),
        "three_seed_profile": all(row.get("seeds") == [2026, 2027, 2028] for row in lineages),
        "fully_excluded_state": all(row.get("upstream_lineage", {}).get("upstream_isolation") == "fully_excluded_nested_state_oof_v1" for row in lineages),
        "causal_normalized_raw": all(
            row.get("raw_imu", {}).get("maximum_future_seconds", 61) <= 60
            and row.get("raw_imu", {}).get("normalization") == "training_fold_robust_v1"
            and row.get("raw_imu", {}).get("shape", [0])[1:] == [3, 12, 300]
            and row.get("raw_imu", {}).get("normalization_training_subjects") == row.get("training_subjects")
            for row in lineages
        ),
    }
    path = final_root / "v49_raw_imu_gate.json"
    write_json_atomic(path, {
        "protocol": "v49_raw_imu_nested_crossfit_gate_v2", "checks": checks,
        "passed": all(checks.values()), "raw_imu_verifier_protocol": config["verifier"].get("raw_imu_protocol"),
        "evidence_sha256": {name: sha256_file(final_root / name) for name in (
            "deep_crossfit.json", "deep_crossfit_scores.parquet", "resolved_config.yaml", "v49_deep_gate.json",
        )},
    })
    return path


def validate_v49_raw_imu_gate(final_root: Path) -> dict[str, Any]:
    path = final_root / "v49_raw_imu_gate.json"
    if not path.is_file():
        raise RuntimeError("v4.9 raw IMU gate evidence is missing")
    gate = json.loads(path.read_text(encoding="utf-8"))
    required = {"deep_crossfit.json", "deep_crossfit_scores.parquet", "resolved_config.yaml", "v49_deep_gate.json"}
    if (gate.get("protocol") != "v49_raw_imu_nested_crossfit_gate_v2" or gate.get("passed") is not True
            or not gate.get("checks") or not all(gate["checks"].values())
            or set(gate.get("evidence_sha256", {})) != required):
        raise RuntimeError("v4.9 raw IMU promotion failed; bundle export is disabled")
    for name, digest in gate["evidence_sha256"].items():
        if not (final_root / name).is_file() or sha256_file(final_root / name) != digest:
            raise RuntimeError("v4.9 raw IMU promotion evidence changed")
    return gate


def validate_sensor_only_reference(
    output_root: Path, s0_run: str, *, expected_input_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    validate_run_name(s0_run)
    evidence: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    reference_inputs: dict[str, str] | None = None
    for fold in range(5):
        root = output_root / "experiments" / s0_run / f"fold_{fold}"
        manifest_path = root / "run_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Verified sensor-only baseline is missing for fold {fold}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = yaml.safe_load((root / "resolved_config.yaml").read_text(encoding="utf-8"))
        if config.get("experiment", {}).get("ablation_id") not in {"R3-S0", "S0"}:
            raise RuntimeError("S0 reference is not identified as sensor-only")
        if bool(config.get("model", {}).get("use_statistics", True)):
            raise RuntimeError("S0 reference unexpectedly enables the statistics branch")
        if manifest.get("stage") != "EVALUATED" or manifest.get("outer_fold") != fold:
            raise RuntimeError(f"S0 fold {fold} is incomplete or has changed split ownership")
        inputs = manifest.get("input_hashes", {})
        if not inputs or (reference_inputs is not None and inputs != reference_inputs):
            raise RuntimeError("S0 folds lack consistent input identity")
        if expected_input_hashes is not None and inputs != expected_input_hashes:
            raise RuntimeError("S0 reference uses a different data or subject split identity")
        reference_inputs = inputs
        HierarchicalRun(root, manifest_path, manifest).verify_artifacts()
        for artifact in [manifest_path, *(root / name for name in manifest["artifact_hashes"])]:
            evidence[artifact.relative_to(output_root).as_posix()] = sha256_file(artifact)
        rows.append({
            "fold": fold, "cohort_sha256": _evaluation_cohort_identity(root),
            "coverage": candidate_coverage(
                pd.read_parquet(root / "outer/proposals.parquet"),
                pd.read_parquet(root / "evaluation/truth_events.parquet"),
            ),
        })
    return {"input_hashes": reference_inputs, "folds": rows, "evidence_sha256": evidence}


def evaluate_v49_outer_preflight(
    output_root: Path, run_name: str, s0_run: str | None, config: dict[str, Any],
    *, skip_s0: bool = False,
) -> dict[str, Any]:
    from bme_eating.hierarchical_v4_artifacts import _saved_resume_config_hash, resume_config_hash

    validate_run_name(run_name)
    if skip_s0 == bool(s0_run):
        raise ValueError("Specify a verified --s0-run or an explicit --skip-s0")
    if s0_run is not None:
        validate_run_name(s0_run)
    if run_name == s0_run:
        raise RuntimeError("S0 reference must be a separately verified sensor-only run")
    root = output_root / "experiments" / run_name
    rows: list[dict[str, Any]] = []
    evidence: dict[str, str] = {}
    sensor_only = None if skip_s0 else validate_sensor_only_reference(output_root, s0_run)
    reference_evidence = {} if sensor_only is None else dict(sensor_only["evidence_sha256"])
    sensor_only_status = (
        {"status": "not_available", "comparison_performed": False, "reason": "explicit_skip_s0",
         "qualification_reference": config["v49"]["frozen_reference_run"]}
        if skip_s0 else {"status": "verified", "comparison_performed": True, "run_name": s0_run}
    )
    execution_protocol_path = root / "execution_protocol.json"
    if execution_protocol_path.is_file():
        expected_protocol = {"run_name": run_name, "s0_run": s0_run, "skip_s0": skip_s0}
        if json.loads(execution_protocol_path.read_text(encoding="utf-8")) != expected_protocol:
            raise RuntimeError("v4.9 S0 comparison policy differs from the original execution")
        evidence["execution_protocol.json"] = sha256_file(execution_protocol_path)
    from bme_eating.v49_resume import validate_resume_migration

    migration = validate_resume_migration(root, config)
    if migration is not None:
        evidence["resume_migration.json"] = sha256_file(root / "resume_migration.json")
        evidence.update(migration["backup_sha256"])
    identity: dict[str, Any] | None = None
    for fold in range(5):
        fold_root = root / f"fold_{fold}"
        path = fold_root / "run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("stage") != "EVALUATED" or manifest.get("code_version") != "v4.9":
            raise RuntimeError(f"v4.9 fold {fold} must be fully evaluated")
        if _saved_resume_config_hash(fold_root, manifest) != resume_config_hash(config):
            raise RuntimeError(f"v4.9 fold {fold} config identity changed")
        HierarchicalRun(fold_root, path, manifest).verify_artifacts()
        current = {
            "resolved_config_sha256": resume_config_hash(config),
            "input_hashes": manifest["input_hashes"], "git": manifest["git"],
        }
        if config.get("v49", {}).get("protocol") == "integrated_repair_v2":
            from bme_eating.v4_protocol import (
                execution_environment_identity,
                execution_source_identity,
            )

            project = Path(__file__).resolve().parents[2]
            current.update({
                "execution_source_identity": execution_source_identity(project),
                "execution_environment": execution_environment_identity(),
                "v49_gate_manifest": config["v49"]["gate_manifest"],
            })
            for key in ("execution_source_identity", "execution_environment"):
                if manifest.get(key) != current[key]:
                    raise RuntimeError(f"v4.9 fold {fold} {key} changed")
        if migration is not None and (
            current["git"] != migration["active_git"]
            or current["input_hashes"] != migration["input_hashes"]
        ):
            raise RuntimeError("v4.9 outer fold differs from its audited continuation")
        if identity is not None and current != identity:
            raise RuntimeError("v4.9 outer folds have inconsistent data or source identities")
        identity = current
        cohort = _evaluation_cohort_identity(fold_root)
        if sensor_only is not None:
            if sensor_only["input_hashes"] != current["input_hashes"]:
                raise RuntimeError("S0 reference uses a different data or subject split identity")
            if cohort != sensor_only["folds"][fold]["cohort_sha256"]:
                raise RuntimeError(f"S0 fold {fold} truth, ignore or observation cohort differs")
        for artifact in [path, *(fold_root / name for name in manifest["artifact_hashes"])]:
            evidence[artifact.relative_to(root).as_posix()] = sha256_file(artifact)
        epoch_path = fold_root / "selection" / "state_epochs.json"
        epochs = json.loads(epoch_path.read_text(encoding="utf-8"))
        selector_paths = list((fold_root / "crossfit").rglob("selector_seed_*.json"))
        if not selector_paths:
            raise RuntimeError(f"v4.9 fold {fold} has no State qualification evidence")
        for selector_path in selector_paths:
            selector = json.loads(selector_path.read_text(encoding="utf-8"))
            if selector.get("promotion_eligible") is not True:
                raise RuntimeError(f"v4.9 fold {fold} has an unqualified State epoch")
            evidence[selector_path.relative_to(root).as_posix()] = sha256_file(selector_path)
        coverage = candidate_coverage(
            pd.read_parquet(fold_root / "outer" / "proposals.parquet"),
            pd.read_parquet(fold_root / "evaluation" / "truth_events.parquet"),
            pd.read_parquet(fold_root / "outer" / "window_predictions.parquet"),
        )
        validate_proposal_lineage(pd.read_parquet(fold_root / "outer" / "proposals.parquet"))
        s0_coverage = None if sensor_only is None else sensor_only["folds"][fold]["coverage"]
        rows.append({"fold": fold, "coverage": coverage, "sensor_only_coverage": s0_coverage,
                     "state_epochs": epochs, "cohort_sha256": cohort,
                     "training_source_git": manifest.get("training_source_git", manifest["git"]),
                     "source_identity_history": manifest.get("source_identity_history", [])})
    frozen = config["promotion_gate"]["v48_frozen_baseline"]
    totals = {key: sum(row["coverage"][key] for row in rows)
              for key in ("covered", "same_covered", "different_covered", "truth_count")}
    checks = {
        "all_outer_folds": len(rows) == 5,
        "candidate_coverage": totals["covered"] >= int(config["promotion_gate"]["candidate_coverage_minimum"]),
        "same_coverage_retained": totals["same_covered"] >= int(frozen["same_candidate_covered"]),
        "different_coverage_retained": totals["different_covered"] >= int(frozen["different_candidate_covered"]),
    }
    if config.get("v49", {}).get("protocol") == "integrated_repair_v2":
        checks["fold_candidate_recall_floor"] = all(
            row["coverage"]["candidate_recall"] >= float(config["promotion_gate"]["fold_candidate_recall_minimum"])
            for row in rows
        )
    if sensor_only is not None:
        checks["sensor_only_coverage_retained"] = all(
            row["coverage"][key] >= row["sensor_only_coverage"][key]
            for row in rows for key in ("same_covered", "different_covered")
        )
    reference_run = str(config["v49"]["frozen_reference_run"])
    validate_run_name(reference_run)
    frozen_root = output_root / "final" / reference_run
    frozen_manifest = json.loads((frozen_root / "final_manifest.json").read_text(encoding="utf-8"))
    if frozen_manifest.get("resume_identity", {}).get("input_hashes") != identity["input_hashes"]:
        raise RuntimeError("Frozen Deep reference uses a different data or subject split identity")
    for fold in range(5):
        frozen_fold = output_root / "experiments" / reference_run / f"fold_{fold}"
        if _evaluation_cohort_identity(frozen_fold) != rows[fold]["cohort_sha256"]:
            raise RuntimeError(f"Frozen Deep fold {fold} evaluation cohort differs")
        for relative in ("evaluation/truth_events.parquet", "evaluation/ignore_events.parquet"):
            path = frozen_fold / relative
            reference_evidence[path.relative_to(output_root).as_posix()] = sha256_file(path)
    for name in ("final_manifest.json", "deep_crossfit.json", "deep_crossfit_scores.parquet", "selected_pipeline.json"):
        reference_evidence[(frozen_root / name).relative_to(output_root).as_posix()] = sha256_file(frozen_root / name)
    return {"mode": "full", "folds": rows, "checks": checks, "passed": all(checks.values()),
            "sensor_only_reference": sensor_only_status,
            "coverage": totals, "identity": identity, "evidence_sha256": evidence,
            "reference_evidence_sha256": reference_evidence,
            "deferred_checks": ["deep_f1", "deep_fp_per_hour", "paired_frozen_bootstrap",
                                "matching_ranking", "stress_folds", "endpoint_refiner"]}
