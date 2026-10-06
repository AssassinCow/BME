from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    for source in sorted(root.rglob("*")):
        if not source.is_file() or "failure_evidence" in source.relative_to(root).parts:
            continue
        relative = source.relative_to(root)
        if any(part in {"model_bundle", "model_bundle.tmp"} for part in relative.parts):
            continue
        evidence[relative.as_posix()] = sha256_file(source)
        if source.suffix in {".json", ".yaml", ".csv"} or (
            source.suffix == ".parquet" and ("diagnostics" in relative.parts or "scores" in source.name)
        ):
            target = archive / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
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
    write_json_atomic(archive / "failure_report.json", report)
    write_json_atomic(root / "failure_report.json", report)
    return root / "failure_report.json"


def run_cli_with_failure_report(main: Callable[[], None], stage: str) -> None:
    try:
        main()
    except (Exception, SystemExit) as error:
        if isinstance(error, SystemExit) and error.code in {None, 0}:
            raise
        arguments = sys.argv[1:]
        if "--config" in arguments and "--run-name" in arguments:
            from bme_eating.config import load_config, resolve_artifact_roots

            config = load_config(arguments[arguments.index("--config") + 1])
            if is_v49(config):
                run_name = arguments[arguments.index("--run-name") + 1]
                validate_run_name(run_name)
                _, _, output_root = resolve_artifact_roots(config)
                branch = "final" if stage in {"final_training", "export"} else "experiments"
                write_failure_report(output_root / branch / run_name, stage, error)
        raise


def candidate_coverage(proposals: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    covered_ids: list[str] = []
    counts = {"covered": 0, "same_covered": 0, "different_covered": 0}
    by_source: dict[str, int] = {}
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
        if not matches:
            continue
        counts["covered"] += 1
        relation = str(getattr(event, "hand_relation", "unknown"))
        if relation in {"same", "different"}:
            counts[f"{relation}_covered"] += 1
        covered_ids.append(f"{event.subject_key}:{event.session_id}:{event.event_id}")
        for bit in range(7):
            if any(int(row.source_mask) & (1 << bit) for row in matches):
                by_source[str(1 << bit)] = by_source.get(str(1 << bit), 0) + 1
    return {
        **counts, "truth_count": len(truth), "covered_event_ids": covered_ids,
        "source_wise_covered": by_source, "formal_outer_holdout": True,
    }


def validate_v49_gate_manifest(
    output_root: Path, run_name: str, identity: dict[str, Any],
) -> dict[str, Any]:
    root = output_root / "experiments" / run_name
    path = root / "v49_gate_manifest.json"
    if not path.is_file():
        raise RuntimeError("v4.9 final training requires v49_gate_manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != "v49_gate_manifest_v1"
        or manifest.get("candidate_run") != run_name
        or manifest.get("folds") != list(range(5))
        or manifest.get("passed") is not True
        or manifest.get("crossfold_gate", {}).get("passed") is not True
    ):
        raise RuntimeError("v4.9 protocol lock is incomplete or failed")
    for key in ("resolved_config_sha256", "input_hashes", "git"):
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
    output_root: Path, run_name: str, s0_run: str, config: dict[str, Any],
) -> dict[str, Any]:
    from bme_eating.hierarchical_v4_artifacts import _saved_resume_config_hash, resume_config_hash

    validate_run_name(run_name)
    validate_run_name(s0_run)
    if run_name == s0_run:
        raise RuntimeError("S0 reference must be a separately verified sensor-only run")
    root = output_root / "experiments" / run_name
    rows: list[dict[str, Any]] = []
    evidence: dict[str, str] = {}
    sensor_only = validate_sensor_only_reference(output_root, s0_run)
    reference_evidence = dict(sensor_only["evidence_sha256"])
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
        if identity is not None and current != identity:
            raise RuntimeError("v4.9 outer folds have inconsistent data or source identities")
        identity = current
        if sensor_only["input_hashes"] != current["input_hashes"]:
            raise RuntimeError("S0 reference uses a different data or subject split identity")
        if _evaluation_cohort_identity(fold_root) != sensor_only["folds"][fold]["cohort_sha256"]:
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
        )
        validate_proposal_lineage(pd.read_parquet(fold_root / "outer" / "proposals.parquet"))
        s0_coverage = sensor_only["folds"][fold]["coverage"]
        rows.append({"fold": fold, "coverage": coverage, "sensor_only_coverage": s0_coverage,
                     "state_epochs": epochs})
    frozen = config["promotion_gate"]["v48_frozen_baseline"]
    totals = {key: sum(row["coverage"][key] for row in rows)
              for key in ("covered", "same_covered", "different_covered", "truth_count")}
    checks = {
        "all_outer_folds": len(rows) == 5,
        "candidate_coverage": totals["covered"] >= int(config["promotion_gate"]["candidate_coverage_minimum"]),
        "same_coverage_retained": totals["same_covered"] >= int(frozen["same_candidate_covered"]),
        "different_coverage_retained": totals["different_covered"] >= int(frozen["different_candidate_covered"]),
        "sensor_only_coverage_retained": all(
            row["coverage"][key] >= row["sensor_only_coverage"][key]
            for row in rows for key in ("same_covered", "different_covered")
        ),
    }
    reference_run = str(config["v49"]["frozen_reference_run"])
    validate_run_name(reference_run)
    frozen_root = output_root / "final" / reference_run
    frozen_manifest = json.loads((frozen_root / "final_manifest.json").read_text(encoding="utf-8"))
    if frozen_manifest.get("resume_identity", {}).get("input_hashes") != identity["input_hashes"]:
        raise RuntimeError("Frozen Deep reference uses a different data or subject split identity")
    for fold in range(5):
        frozen_fold = output_root / "experiments" / reference_run / f"fold_{fold}"
        if _evaluation_cohort_identity(frozen_fold) != sensor_only["folds"][fold]["cohort_sha256"]:
            raise RuntimeError(f"Frozen Deep fold {fold} evaluation cohort differs")
        for relative in ("evaluation/truth_events.parquet", "evaluation/ignore_events.parquet"):
            path = frozen_fold / relative
            reference_evidence[path.relative_to(output_root).as_posix()] = sha256_file(path)
    for name in ("final_manifest.json", "deep_crossfit.json", "deep_crossfit_scores.parquet", "selected_pipeline.json"):
        reference_evidence[(frozen_root / name).relative_to(output_root).as_posix()] = sha256_file(frozen_root / name)
    return {"mode": "full", "folds": rows, "checks": checks, "passed": all(checks.values()),
            "coverage": totals, "identity": identity, "evidence_sha256": evidence,
            "reference_evidence_sha256": reference_evidence,
            "deferred_checks": ["deep_f1", "deep_fp_per_hour", "paired_frozen_bootstrap",
                                "matching_ranking", "stress_folds", "endpoint_refiner"]}
