from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from bme_eating.fusion import ALIGNMENT_KEYS, FROZEN_INPUT_HASHES, json_safe

V4_CONFIRMATION_FOLDS = (1, 2, 3, 4)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _verify_artifact(fold_dir: Path, manifest: dict[str, Any], name: str) -> str:
    path = fold_dir / name
    expected = manifest.get("artifact_hashes", {}).get(name)
    if not path.is_file() or not expected:
        raise RuntimeError(f"Frozen fold is missing a manifested artifact: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"Frozen artifact changed after manifesting: {path}")
    return actual


def _prediction_scope(path: Path) -> tuple[pd.DataFrame, float, set[str]]:
    frame = pd.read_parquet(path, columns=ALIGNMENT_KEYS)
    if frame.duplicated(ALIGNMENT_KEYS).any():
        raise RuntimeError(f"Prediction timeline contains duplicate keys: {path}")
    frame["subject_key"] = frame["subject_key"].astype(str)
    frame = frame.sort_values(ALIGNMENT_KEYS).reset_index(drop=True)
    exposure_hours = 0.0
    for _, group in frame.groupby(["subject_key", "session_id"], sort=False):
        exposure_hours += max(
            0.0,
            (
                float(group["timestamp_ms"].max())
                - float(group["timestamp_ms"].min())
                + 3000.0
            )
            / 3_600_000.0,
        )
    return frame, exposure_hours, set(frame["subject_key"].unique())


def _metrics_row(payload: dict[str, Any], exposure_hours: float) -> dict[str, Any]:
    method = str(payload["primary_method"])
    primary = payload[method]
    strict = payload["strict_no_ignore"]
    for name, metrics in (("primary", primary), ("strict_no_ignore", strict)):
        true_positive = float(metrics["true_positive"])
        false_positive = float(metrics["false_positive"])
        false_negative = float(metrics["false_negative"])
        denominator = 2.0 * true_positive + false_positive + false_negative
        expected_f1 = 2.0 * true_positive / denominator if denominator else 0.0
        if not math.isclose(
            float(metrics["f1"]), expected_f1, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise RuntimeError(f"{name} F1 is inconsistent with its event counts")
    subject_metrics = payload.get("by_subject", {})
    for count_name in ("true_positive", "false_positive", "false_negative"):
        if subject_metrics and not math.isclose(
            sum(float(metrics.get(count_name) or 0) for metrics in subject_metrics.values()),
            float(primary[count_name]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(f"Per-subject {count_name} does not match the fold total")
    reported_fp_rate = primary.get("false_positives_per_observed_hour")
    calculated_fp_rate = (
        float(primary["false_positive"]) / exposure_hours if exposure_hours else 0.0
    )
    if reported_fp_rate is not None and not math.isclose(
        float(reported_fp_rate), calculated_fp_rate, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise RuntimeError("Reported false positives per hour do not match the timeline")
    different = payload.get("hand_relation", {}).get("different", {})
    same = payload.get("hand_relation", {}).get("same", {})
    return {
        "primary_method": method,
        "true_positive": float(primary["true_positive"]),
        "false_positive": float(primary["false_positive"]),
        "false_negative": float(primary["false_negative"]),
        "f1": float(primary["f1"]),
        "start_mae_seconds": float(primary["start_mae_seconds"]),
        "end_mae_seconds": float(primary["end_mae_seconds"]),
        "strict_true_positive": float(strict["true_positive"]),
        "strict_false_positive": float(strict["false_positive"]),
        "strict_false_negative": float(strict["false_negative"]),
        "different_truth": float(different.get("truth_events") or 0),
        "different_matches": float(different.get("matched_events") or 0),
        "same_truth": float(same.get("truth_events") or 0),
        "same_matches": float(same.get("matched_events") or 0),
        "exposure_hours": float(exposure_hours),
        "subjects": sorted(str(subject) for subject in subject_metrics),
    }


def _validate_group_provenance(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    commits = {row["git_commit"] for row in rows}
    config_hashes = {row["config_hash"] for row in rows}
    fingerprint_values = {
        json.dumps(row["input_hashes"], sort_keys=True, separators=(",", ":"))
        for row in rows
    }
    if len(commits) != 1:
        raise RuntimeError(f"{name} folds mix Git commits: {sorted(commits)}")
    if len(config_hashes) != 1:
        raise RuntimeError(f"{name} folds mix resolved configurations")
    if len(fingerprint_values) != 1:
        raise RuntimeError(f"{name} folds mix data or split fingerprints")
    subjects = [subject for row in rows for subject in row["subjects"]]
    if len(subjects) != len(set(subjects)):
        raise RuntimeError(f"A {name} subject appears in more than one confirmation fold")
    return {
        "git_commit": next(iter(commits)),
        "resolved_config_sha256": next(iter(config_hashes)),
        "input_hashes": rows[0]["input_hashes"],
    }


def load_v4_promotion_evidence(
    baseline_root: Path,
    candidate_root: Path,
    folds: tuple[int, ...] = V4_CONFIRMATION_FOLDS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    baseline_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {"folds": {}}
    for fold in folds:
        baseline_dir = baseline_root / f"fold_{fold}"
        candidate_dir = candidate_root / f"fold_{fold}"
        baseline_manifest_path = baseline_dir / "run_manifest.json"
        candidate_manifest_path = candidate_dir / "run_manifest.json"
        if not baseline_manifest_path.is_file() or not candidate_manifest_path.is_file():
            raise FileNotFoundError(f"Fold {fold} is missing a run manifest")
        baseline_manifest = _read_object(baseline_manifest_path)
        candidate_manifest = _read_object(candidate_manifest_path)
        for root, manifest, label in (
            (baseline_root, baseline_manifest, "baseline"),
            (candidate_root, candidate_manifest, "candidate"),
        ):
            expected_identity = {"name": root.name, "fold": fold}
            if manifest.get("experiment") != expected_identity:
                raise RuntimeError(f"{label} fold {fold} manifest identity is inconsistent")
            if manifest.get("git", {}).get("dirty") is not False:
                raise RuntimeError(f"{label} fold {fold} was produced from a dirty worktree")

        baseline_hashes = {
            name: _verify_artifact(baseline_dir, baseline_manifest, name)
            for name in ("test_metrics.json", "test_predictions.parquet")
        }
        candidate_hashes = {
            name: _verify_artifact(candidate_dir, candidate_manifest, name)
            for name in (
                "test_metrics.json",
                "test_predictions.parquet",
                "selected_fusion.json",
            )
        }
        baseline_timeline, baseline_exposure, baseline_subjects = _prediction_scope(
            baseline_dir / "test_predictions.parquet"
        )
        candidate_timeline, candidate_exposure, candidate_subjects = _prediction_scope(
            candidate_dir / "test_predictions.parquet"
        )
        if not baseline_timeline.equals(candidate_timeline):
            raise RuntimeError(f"Fold {fold} baseline and candidate timelines differ")
        if not math.isclose(baseline_exposure, candidate_exposure, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Fold {fold} baseline and candidate exposure differs")
        baseline_metrics = _read_object(baseline_dir / "test_metrics.json")
        candidate_metrics = _read_object(candidate_dir / "test_metrics.json")
        if set(map(str, baseline_metrics.get("by_subject", {}))) != baseline_subjects:
            raise RuntimeError(f"Baseline fold {fold} subject metrics do not cover its timeline")
        if set(map(str, candidate_metrics.get("by_subject", {}))) != candidate_subjects:
            raise RuntimeError(f"Candidate fold {fold} subject metrics do not cover its timeline")
        selection = _read_object(candidate_dir / "selected_fusion.json")
        if (
            int(selection.get("protocol_version", 0)) != 4
            or selection.get("run_name") != candidate_root.name
            or int(selection.get("fold", -1)) != fold
        ):
            raise RuntimeError(f"Candidate fold {fold} is not a matching protocol-v4 selection")
        if selection.get("meta_oof_gate", {}).get("passed") is not True:
            raise RuntimeError(f"Candidate fold {fold} lacks a passing meta-OOF gate")
        outer_passed = selection.get("outer_fold_gate", {}).get("passed")
        if not isinstance(outer_passed, bool):
            raise TypeError(f"Candidate fold {fold} has not completed outer evaluation")

        baseline_row = {
            "fold": fold,
            **_metrics_row(baseline_metrics, baseline_exposure),
            "git_commit": str(baseline_manifest.get("git", {}).get("commit", "")),
            "config_hash": str(baseline_manifest.get("resolved_config_sha256", "")),
            "input_hashes": {
                name: baseline_manifest.get("hashes", {}).get(name)
                for name in FROZEN_INPUT_HASHES
            },
        }
        candidate_row = {
            "fold": fold,
            **_metrics_row(candidate_metrics, candidate_exposure),
            "git_commit": str(candidate_manifest.get("git", {}).get("commit", "")),
            "config_hash": str(candidate_manifest.get("resolved_config_sha256", "")),
            "input_hashes": {
                name: candidate_manifest.get("hashes", {}).get(name)
                for name in FROZEN_INPUT_HASHES
            },
            "outer_gate_passed": outer_passed,
        }
        if None in baseline_row["input_hashes"].values() or None in candidate_row[
            "input_hashes"
        ].values():
            raise RuntimeError(f"Fold {fold} manifest lacks a required input fingerprint")
        baseline_rows.append(baseline_row)
        candidate_rows.append(candidate_row)
        evidence["folds"][str(fold)] = {
            "baseline": {
                "run_manifest.json": _sha256(baseline_manifest_path),
                **baseline_hashes,
            },
            "candidate": {
                "run_manifest.json": _sha256(candidate_manifest_path),
                **candidate_hashes,
            },
        }

    baseline_provenance = _validate_group_provenance(baseline_rows, "baseline")
    candidate_provenance = _validate_group_provenance(candidate_rows, "candidate")
    if baseline_provenance["input_hashes"] != candidate_provenance["input_hashes"]:
        raise RuntimeError("Baseline and candidate data or split fingerprints differ")
    if any(
        baseline_rows[index]["subjects"] != candidate_rows[index]["subjects"]
        for index in range(len(folds))
    ):
        raise RuntimeError("Baseline and candidate per-fold subject coverage differs")
    evidence["baseline_provenance"] = baseline_provenance
    evidence["candidate_provenance"] = candidate_provenance
    return baseline_rows, candidate_rows, evidence


def _f1(true_positive: float, false_positive: float, false_negative: float) -> float:
    denominator = 2.0 * true_positive + false_positive + false_negative
    return 2.0 * true_positive / denominator if denominator else 0.0


def aggregate_v4_folds(rows: list[dict[str, Any]]) -> dict[str, float]:
    true_positive = sum(float(row["true_positive"]) for row in rows)
    false_positive = sum(float(row["false_positive"]) for row in rows)
    false_negative = sum(float(row["false_negative"]) for row in rows)
    strict_true_positive = sum(float(row["strict_true_positive"]) for row in rows)
    strict_false_positive = sum(float(row["strict_false_positive"]) for row in rows)
    strict_false_negative = sum(float(row["strict_false_negative"]) for row in rows)
    exposure_hours = sum(float(row["exposure_hours"]) for row in rows)
    different_truth = sum(float(row["different_truth"]) for row in rows)
    different_matches = sum(float(row["different_matches"]) for row in rows)
    same_truth = sum(float(row["same_truth"]) for row in rows)
    same_matches = sum(float(row["same_matches"]) for row in rows)
    for name in ("start_mae_seconds", "end_mae_seconds"):
        if any(
            float(row["true_positive"]) > 0 and not math.isfinite(float(row[name]))
            for row in rows
        ):
            raise RuntimeError(f"Cannot aggregate non-finite {name} with matched events")
    start_sum = sum(
        float(row["start_mae_seconds"]) * float(row["true_positive"])
        for row in rows
        if float(row["true_positive"]) > 0
    )
    end_sum = sum(
        float(row["end_mae_seconds"]) * float(row["true_positive"])
        for row in rows
        if float(row["true_positive"]) > 0
    )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "f1": _f1(true_positive, false_positive, false_negative),
        "strict_no_ignore_f1": _f1(
            strict_true_positive, strict_false_positive, strict_false_negative
        ),
        "different_sensitivity": (
            different_matches / different_truth if different_truth else float("nan")
        ),
        "same_sensitivity": same_matches / same_truth if same_truth else float("nan"),
        "false_positives_per_observed_hour": (
            false_positive / exposure_hours if exposure_hours else 0.0
        ),
        "start_mae_seconds": start_sum / true_positive if true_positive else float("nan"),
        "end_mae_seconds": end_sum / true_positive if true_positive else float("nan"),
        "exposure_hours": exposure_hours,
        "subjects": float(sum(len(row["subjects"]) for row in rows)),
    }


def _ratio(candidate: float, baseline: float) -> float:
    if baseline == 0.0:
        return 1.0 if candidate == 0.0 else float("inf")
    return candidate / baseline


def evaluate_v4_final_promotion(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    if [row["fold"] for row in baseline_rows] != list(V4_CONFIRMATION_FOLDS):
        raise ValueError("Baseline promotion evidence must contain folds 1-4 in order")
    if [row["fold"] for row in candidate_rows] != list(V4_CONFIRMATION_FOLDS):
        raise ValueError("Candidate promotion evidence must contain folds 1-4 in order")
    baseline = aggregate_v4_folds(baseline_rows)
    candidate = aggregate_v4_folds(candidate_rows)
    fold_deltas = [
        {
            "fold": int(candidate_row["fold"]),
            "baseline_f1": float(baseline_row["f1"]),
            "candidate_f1": float(candidate_row["f1"]),
            "f1_delta": float(candidate_row["f1"]) - float(baseline_row["f1"]),
            "outer_gate_passed": bool(candidate_row["outer_gate_passed"]),
        }
        for baseline_row, candidate_row in zip(
            baseline_rows, candidate_rows, strict=True
        )
    ]
    nondegraded = sum(row["f1_delta"] >= 0.0 for row in fold_deltas)
    checks = {
        "fold_1_confirmation_passed": fold_deltas[0]["outer_gate_passed"],
        "aggregate_f1_improvement": (
            candidate["f1"] - baseline["f1"]
            >= float(gate_config["minimum_f1_improvement"])
        ),
        "fp_per_hour_within_ratio": (
            _ratio(
                candidate["false_positives_per_observed_hour"],
                baseline["false_positives_per_observed_hour"],
            )
            <= float(gate_config["maximum_fp_per_hour_ratio"])
        ),
        "start_mae_within_ratio": (
            _ratio(candidate["start_mae_seconds"], baseline["start_mae_seconds"])
            <= float(gate_config["maximum_boundary_mae_ratio"])
        ),
        "end_mae_within_ratio": (
            _ratio(candidate["end_mae_seconds"], baseline["end_mae_seconds"])
            <= float(gate_config["maximum_boundary_mae_ratio"])
        ),
        "minimum_nondegraded_folds": nondegraded
        >= int(gate_config["minimum_nondegraded_folds"]),
        "maximum_fold_f1_drop": min(row["f1_delta"] for row in fold_deltas)
        >= -float(gate_config["maximum_fold_f1_drop"]),
    }
    return json_safe(
        {
            "version": 1,
            "protocol_version": 4,
            "folds": list(V4_CONFIRMATION_FOLDS),
            "baseline": baseline,
            "candidate": candidate,
            "deltas": {
                "f1": candidate["f1"] - baseline["f1"],
                "strict_no_ignore_f1": (
                    candidate["strict_no_ignore_f1"] - baseline["strict_no_ignore_f1"]
                ),
                "different_sensitivity": (
                    candidate["different_sensitivity"] - baseline["different_sensitivity"]
                ),
                "false_positives_per_observed_hour": (
                    candidate["false_positives_per_observed_hour"]
                    - baseline["false_positives_per_observed_hour"]
                ),
                "start_mae_seconds": (
                    candidate["start_mae_seconds"] - baseline["start_mae_seconds"]
                ),
                "end_mae_seconds": (
                    candidate["end_mae_seconds"] - baseline["end_mae_seconds"]
                ),
            },
            "ratios": {
                "false_positives_per_observed_hour": _ratio(
                    candidate["false_positives_per_observed_hour"],
                    baseline["false_positives_per_observed_hour"],
                ),
                "start_mae_seconds": _ratio(
                    candidate["start_mae_seconds"], baseline["start_mae_seconds"]
                ),
                "end_mae_seconds": _ratio(
                    candidate["end_mae_seconds"], baseline["end_mae_seconds"]
                ),
            },
            "fold_deltas": fold_deltas,
            "nondegraded_folds": nondegraded,
            "checks": checks,
            "promote": all(checks.values()),
        }
    )


def write_promotion_report(path: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    digest = _sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return path, sidecar
