from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence_file(path: Path, output_base: Path) -> dict[str, str]:
    return {
        "relative_path": path.resolve().relative_to(output_base.resolve()).as_posix(),
        "sha256": sha256_file(path),
    }


def verify_gate_evidence(output_base: Path, report: dict[str, Any]) -> None:
    evidence: list[dict[str, str]] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if set(value) == {"relative_path", "sha256"}:
                evidence.append(value)
            else:
                for item in value.values():
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(report.get("evidence_sha256", {}))
    collect(report.get("folds", []))
    if not evidence:
        raise RuntimeError("Gate report contains no hash-locked evidence")
    for item in evidence:
        path = output_base / item["relative_path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Gate evidence changed after selection: {item['relative_path']}")


def _candidate_metrics(output_root: Path, run_name: str, fold: int = 0) -> dict[str, Any]:
    root = output_root / "experiments" / run_name / f"fold_{fold}"
    manifest = _read_json(root / "run_manifest.json")
    allowed = {
        "PROPOSALS_COMPLETE",
        "VERIFIER_COMPLETE",
        "BOUNDARY_COMPLETE",
        "SELECTED",
        "EVALUATED",
    }
    if manifest.get("stage") not in allowed:
        raise RuntimeError(f"{run_name} fold {fold} has not completed proposal generation")
    return _read_json(root / "decoder" / "candidate_metrics.json")


def _validate_ppg_source_run(
    output_root: Path,
    run_name: str,
    *,
    expected_ablation: str,
    expected_use_ppg: bool,
) -> Path:
    config_path = output_root / "experiments" / run_name / "fold_0" / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    experiment = config.get("experiment", {})
    if experiment.get("protocol_version") != "statsfusion-r2":
        raise RuntimeError(f"{expected_ablation} source run must use statsfusion-r2")
    if str(experiment.get("ablation_id", "")) != expected_ablation:
        raise RuntimeError(f"{expected_ablation} source run has the wrong ablation_id")
    if bool(config.get("model", {}).get("use_ppg", True)) != expected_use_ppg:
        raise RuntimeError(f"{expected_ablation} source run has the wrong model.use_ppg value")
    return config_path


def evaluate_ppg_promotion(
    output_root: Path,
    *,
    s2_run: str,
    s3_run: str,
    gate: dict[str, Any],
) -> dict[str, Any]:
    if s2_run == s3_run:
        raise RuntimeError("S2 and S3 promotion evidence must come from distinct runs")
    source_configs = {
        "S2": _validate_ppg_source_run(
            output_root,
            s2_run,
            expected_ablation="S2",
            expected_use_ppg=False,
        ),
        "S3": _validate_ppg_source_run(
            output_root,
            s3_run,
            expected_ablation="S3",
            expected_use_ppg=True,
        ),
    }
    s2 = _candidate_metrics(output_root, s2_run)
    s3 = _candidate_metrics(output_root, s3_run)
    evidence = {
        name: {
            "resolved_config": _evidence_file(source_configs[name], output_root.parent),
            "run_manifest": _evidence_file(
                output_root / "experiments" / run / "fold_0" / "run_manifest.json",
                output_root.parent,
            ),
            "candidate_metrics": _evidence_file(
                output_root / "experiments" / run / "fold_0" / "decoder" / "candidate_metrics.json",
                output_root.parent,
            ),
        }
        for name, run in (("S2", s2_run), ("S3", s3_run))
    }
    f1_delta = float(s3["state_only_f1"]) - float(s2["state_only_f1"])
    recall_delta = float(s3["candidate_recall"]) - float(s2["candidate_recall"])
    different_delta = float(s3["different_sensitivity"]) - float(s2["different_sensitivity"])
    checks = {
        "quality_route": bool(
            f1_delta >= float(gate.get("minimum_ppg_f1_improvement", 0.005))
            or (
                recall_delta
                >= float(gate.get("minimum_ppg_candidate_recall_improvement", 0.010))
                and f1_delta >= -float(gate.get("maximum_ppg_f1_drop", 0.005))
            )
        ),
        "fp_per_hour": float(s3["state_only_fp_per_hour"])
        <= float(s2["state_only_fp_per_hour"])
        * float(gate.get("maximum_fp_per_hour_ratio", 1.05)),
        "different_sensitivity": different_delta
        >= -float(gate.get("maximum_ppg_different_sensitivity_drop", 0.02)),
    }
    return {
        "schema_version": 1,
        "protocol_version": "statsfusion-r2",
        "source_runs": {"S2": s2_run, "S3": s3_run},
        "evidence_sha256": evidence,
        "f1_delta_vs_S2": f1_delta,
        "candidate_recall_delta_vs_S2": recall_delta,
        "different_sensitivity_delta_vs_S2": different_delta,
        "checks": checks,
        "passed": all(checks.values()),
        "resolved_use_ppg": all(checks.values()),
    }


def evaluate_fold0_ablations(
    output_root: Path,
    *,
    s0_run: str,
    s1_run: str,
    s2_run: str,
    s3_run: str,
    s4_run: str,
    ppg_only_run: str,
    gate: dict[str, Any],
) -> dict[str, Any]:
    runs = {
        "S0": s0_run,
        "S1": s1_run,
        "S2": s2_run,
        "S3": s3_run,
        "S4": s4_run,
        "PPG_ONLY": ppg_only_run,
    }
    metrics = {name: _candidate_metrics(output_root, run) for name, run in runs.items()}
    evidence_sha256 = {}
    for name, run in runs.items():
        fold_root = output_root / "experiments" / run / "fold_0"
        evidence_sha256[name] = {
            "run_manifest": _evidence_file(fold_root / "run_manifest.json", output_root.parent),
            "candidate_metrics": _evidence_file(
                fold_root / "decoder" / "candidate_metrics.json", output_root.parent
            ),
        }

    def delta(left: str, right: str, metric: str) -> float:
        return float(metrics[left][metric]) - float(metrics[right][metric])

    statistics_checks = {
        "f1_gain": delta("S1", "S0", "state_only_f1")
        >= float(gate["minimum_statistics_f1_improvement"]),
        "fp_per_hour": float(metrics["S1"]["state_only_fp_per_hour"])
        <= float(metrics["S0"]["state_only_fp_per_hour"])
        * float(gate["maximum_fp_per_hour_ratio"]),
        "different_sensitivity": float(metrics["S1"]["different_sensitivity"])
        >= float(metrics["S0"]["different_sensitivity"])
        - float(gate.get("maximum_statistics_different_sensitivity_drop", 0.02)),
    }
    recall_gain = delta("S2", "S1", "candidate_recall")
    f1_gain = delta("S2", "S1", "state_only_f1")
    fragment_reduction = 1.0 - float(metrics["S2"]["state_fragment_count"]) / max(
        float(metrics["S1"]["state_fragment_count"]), 1.0
    )
    long_context_checks = {
        "recall_route": recall_gain >= float(gate["minimum_long_context_recall_improvement"]),
        "f1_fragment_route": f1_gain >= float(gate["minimum_long_context_f1_improvement"])
        and fragment_reduction >= float(gate["minimum_fragment_reduction"]),
    }
    ppg_evidence = {
        "f1_delta_vs_S2": delta("S3", "S2", "state_only_f1"),
        "candidate_recall_delta_vs_S2": delta("S3", "S2", "candidate_recall"),
        "different_sensitivity_delta_vs_S2": delta("S3", "S2", "different_sensitivity"),
        "motion_only_f1": float(metrics["S0"]["state_only_f1"]),
        "ppg_only_f1": float(metrics["PPG_ONLY"]["state_only_f1"]),
        "motion_ppg_f1": float(metrics["S3"]["state_only_f1"]),
    }
    ppg_checks = {
        "quality_route": (
            ppg_evidence["f1_delta_vs_S2"] >= float(gate.get("minimum_ppg_f1_improvement", 0.005))
            or (
                ppg_evidence["candidate_recall_delta_vs_S2"]
                >= float(gate.get("minimum_ppg_candidate_recall_improvement", 0.010))
                and ppg_evidence["f1_delta_vs_S2"] >= -float(gate.get("maximum_ppg_f1_drop", 0.005))
            )
        ),
        "fp_per_hour": float(metrics["S3"]["state_only_fp_per_hour"])
        <= float(metrics["S2"]["state_only_fp_per_hour"])
        * float(gate["maximum_fp_per_hour_ratio"]),
        "different_sensitivity": ppg_evidence["different_sensitivity_delta_vs_S2"]
        >= -float(gate.get("maximum_ppg_different_sensitivity_drop", 0.02)),
    }
    ppg_passed = all(ppg_checks.values())
    s4_config = yaml.safe_load(
        (output_root / "experiments" / s4_run / "fold_0" / "resolved_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    if bool(s4_config["model"].get("use_ppg", True)) != ppg_passed:
        expected = "enabled" if ppg_passed else "disabled"
        raise RuntimeError(f"S4 must be rebuilt from the promoted state path with PPG {expected}")
    candidate_checks = {
        "state_calibration": bool(metrics["S4"]["state_calibration_gate_passed"]),
        "minimum_recall": float(metrics["S4"]["candidate_recall"])
        >= float(gate["minimum_candidate_recall"]),
        "same_side_not_worse_than_sensor_only": float(metrics["S4"]["same_candidate_recall"])
        >= float(metrics["S0"]["same_candidate_recall"]) - 0.03,
        "different_side_not_worse_than_sensor_only": float(
            metrics["S4"]["different_candidate_recall"]
        )
        >= float(metrics["S0"]["different_candidate_recall"]) - 0.03,
    }
    report = {
        "runs": runs,
        "evidence_sha256": evidence_sha256,
        "metrics": metrics,
        "statistics_branch": {
            "checks": statistics_checks,
            "passed": all(statistics_checks.values()),
        },
        "long_context": {
            "recall_gain": recall_gain,
            "f1_gain": f1_gain,
            "fragment_reduction": fragment_reduction,
            "checks": long_context_checks,
            "passed": any(long_context_checks.values()),
        },
        "ppg_ablation": {
            **ppg_evidence,
            "checks": ppg_checks,
            "passed": ppg_passed,
            "s4_use_ppg": bool(s4_config["model"].get("use_ppg", True)),
        },
        "candidate_gate": {
            "checks": candidate_checks,
            "passed": all(candidate_checks.values()),
        },
    }
    report["passed"] = bool(
        report["statistics_branch"]["passed"]
        and report["long_context"]["passed"]
        and report["candidate_gate"]["passed"]
    )
    path = output_root / "experiments" / s4_run / "ablation" / "fold_0_report.json"
    write_json_atomic(path, report)
    return report


def _evaluation_metrics(
    root: Path, run_name: str, fold: int, *, state_only: bool = False
) -> dict[str, float]:
    payload = _read_json(
        root / "experiments" / run_name / f"fold_{fold}" / "evaluation" / "metrics.json"
    )
    primary = (
        payload["state_only"]
        if state_only
        else payload.get(
            "max_cardinality_iou",
            payload.get("selected", payload.get("refined", payload)),
        )
    )
    hand = primary if state_only else payload.get("hand", primary)
    return {
        "f1": float(primary["f1"]),
        "false_positive": float(primary["false_positive"]),
        "false_negative": float(primary["false_negative"]),
        "true_positive": float(primary["true_positive"]),
        "fp_per_hour": float(
            primary.get(
                "fp_per_hour",
                primary.get(
                    "false_positives_per_observed_hour",
                    payload.get("fp_per_hour", 0.0),
                ),
            )
        ),
        "different_sensitivity": float(
            hand.get("different_sensitivity", primary.get("different_sensitivity", np.nan))
        ),
    }


def _per_subject(root: Path, run_name: str, fold: int, *, state_only: bool = False) -> pd.DataFrame:
    filename = "state_only_per_subject_metrics.csv" if state_only else "per_subject_metrics.csv"
    path = root / "experiments" / run_name / f"fold_{fold}" / "evaluation" / filename
    frame = pd.read_csv(path)
    required = {"subject_key", "true_positive", "false_positive", "false_negative"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Per-subject gate evidence is missing columns: {sorted(missing)}")
    return frame


def _pooled(rows: pd.DataFrame) -> dict[str, float]:
    true_positive = float(rows["true_positive"].sum())
    false_positive = float(rows["false_positive"].sum())
    false_negative = float(rows["false_negative"].sum())
    denominator = 2 * true_positive + false_positive + false_negative
    hours = float(rows.get("observed_hours", pd.Series(0.0, index=rows.index)).sum())
    return {
        "f1": 2 * true_positive / denominator if denominator else 0.0,
        "fp_per_hour": false_positive / hours if hours else 0.0,
    }


def _bootstrap_probability(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    replicates: int = 2000,
    seed: int = 2026,
) -> float:
    merged = candidate.merge(
        baseline,
        on="subject_key",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    if len(merged) != len(candidate) or len(merged) != len(baseline):
        raise ValueError("Candidate and baseline bootstrap subjects differ")
    rng = np.random.default_rng(seed)
    wins = 0

    def f1(frame: pd.DataFrame, suffix: str) -> float:
        true_positive = float(frame[f"true_positive_{suffix}"].sum())
        false_positive = float(frame[f"false_positive_{suffix}"].sum())
        false_negative = float(frame[f"false_negative_{suffix}"].sum())
        denominator = 2 * true_positive + false_positive + false_negative
        return 2 * true_positive / denominator if denominator else 0.0

    for _ in range(replicates):
        indices = rng.integers(0, len(merged), size=len(merged))
        sampled = merged.iloc[indices]
        wins += int(f1(sampled, "candidate") > f1(sampled, "baseline"))
    return wins / replicates


def _stronger_baseline(
    v4_root: Path,
    fold: int,
    *,
    s0_run: str,
    v3_root: Path | None,
    v3_run: str | None,
) -> tuple[str, dict[str, float], pd.DataFrame]:
    choices = []
    s0_path = v4_root / "experiments" / s0_run / f"fold_{fold}" / "evaluation" / "metrics.json"
    if s0_path.is_file():
        choices.append(
            (
                f"v4:{s0_run}:state_only",
                _evaluation_metrics(v4_root, s0_run, fold, state_only=True),
                _per_subject(v4_root, s0_run, fold, state_only=True),
            )
        )
    if v3_root is not None and v3_run is not None:
        choices.append(
            (
                f"v3:{v3_run}",
                _evaluation_metrics(v3_root, v3_run, fold),
                _per_subject(v3_root, v3_run, fold),
            )
        )
    if not choices:
        raise FileNotFoundError(f"Fold {fold} has neither S0 state-only nor v3 baseline evidence")
    return max(choices, key=lambda value: value[1]["f1"])


def evaluate_crossfold_gate(
    output_root: Path,
    *,
    candidate_run: str,
    s0_run: str,
    folds: tuple[int, ...],
    gate: dict[str, Any],
    mode: str,
    v3_root: Path | None = None,
    v3_run: str | None = None,
) -> dict[str, Any]:
    if mode not in {"development", "stress"}:
        raise ValueError("Crossfold gate mode must be development or stress")
    fold_rows: list[dict[str, Any]] = []
    candidate_subjects: list[pd.DataFrame] = []
    baseline_subjects: list[pd.DataFrame] = []
    for fold in folds:
        candidate = _evaluation_metrics(output_root, candidate_run, fold)
        candidate_per_subject = _per_subject(output_root, candidate_run, fold)
        baseline_name, baseline, baseline_per_subject = _stronger_baseline(
            output_root,
            fold,
            s0_run=s0_run,
            v3_root=v3_root,
            v3_run=v3_run,
        )
        candidate_directory = (
            output_root / "experiments" / candidate_run / f"fold_{fold}" / "evaluation"
        )
        if baseline_name.startswith("v4:"):
            baseline_directory = (
                output_root / "experiments" / s0_run / f"fold_{fold}" / "evaluation"
            )
            baseline_subject_name = "state_only_per_subject_metrics.csv"
        else:
            if v3_root is None or v3_run is None:
                raise RuntimeError("Selected v3 baseline has no configured root")
            baseline_directory = v3_root / "experiments" / v3_run / f"fold_{fold}" / "evaluation"
            baseline_subject_name = "per_subject_metrics.csv"
        fold_rows.append(
            {
                "fold": fold,
                "baseline": baseline_name,
                "candidate": candidate,
                "baseline_metrics": baseline,
                "delta_f1": candidate["f1"] - baseline["f1"],
                "fp_ratio": candidate["fp_per_hour"] / max(baseline["fp_per_hour"], 1e-9),
                "different_sensitivity_delta": candidate["different_sensitivity"]
                - baseline["different_sensitivity"],
                "evidence_sha256": {
                    "candidate_metrics": _evidence_file(
                        candidate_directory / "metrics.json", output_root.parent
                    ),
                    "candidate_per_subject": _evidence_file(
                        candidate_directory / "per_subject_metrics.csv",
                        output_root.parent,
                    ),
                    "baseline_metrics": _evidence_file(
                        baseline_directory / "metrics.json", output_root.parent
                    ),
                    "baseline_per_subject": _evidence_file(
                        baseline_directory / baseline_subject_name,
                        output_root.parent,
                    ),
                },
            }
        )
        candidate_subjects.append(candidate_per_subject.assign(fold=fold))
        baseline_subjects.append(baseline_per_subject.assign(fold=fold))
    candidate_frame = pd.concat(candidate_subjects, ignore_index=True)
    baseline_frame = pd.concat(baseline_subjects, ignore_index=True)
    candidate_pooled = _pooled(candidate_frame)
    baseline_pooled = _pooled(baseline_frame)
    if mode == "development":
        probability = _bootstrap_probability(candidate_frame, baseline_frame)
        checks = {
            "both_folds_improve": all(row["delta_f1"] > 0 for row in fold_rows),
            "mean_f1_gain": float(np.mean([row["delta_f1"] for row in fold_rows]))
            >= float(gate["minimum_mean_f1_improvement"]),
            "fp_per_hour": candidate_pooled["fp_per_hour"]
            <= baseline_pooled["fp_per_hour"] * float(gate["maximum_fp_per_hour_ratio"]),
            "different_sensitivity": all(
                row["different_sensitivity_delta"]
                >= -float(gate["maximum_different_sensitivity_drop"])
                for row in fold_rows
            ),
            "paired_bootstrap": probability
            >= float(gate["minimum_positive_bootstrap_probability"]),
        }
    else:
        probability = None
        checks = {
            "two_of_three_non_degrading": sum(row["delta_f1"] >= 0 for row in fold_rows) >= 2,
            "pooled_f1": candidate_pooled["f1"] > baseline_pooled["f1"],
            "maximum_single_fold_drop": min(row["delta_f1"] for row in fold_rows)
            >= -float(gate.get("maximum_stress_fold_f1_drop", 0.03)),
            "fp_per_hour": candidate_pooled["fp_per_hour"]
            <= baseline_pooled["fp_per_hour"] * float(gate["maximum_fp_per_hour_ratio"]),
            "different_sensitivity": all(
                row["different_sensitivity_delta"]
                >= -float(gate.get("maximum_stress_different_sensitivity_drop", 0.05))
                for row in fold_rows
            ),
        }
    report = {
        "mode": mode,
        "candidate_run": candidate_run,
        "s0_run": s0_run,
        "v3_run": v3_run,
        "folds": fold_rows,
        "candidate_pooled": candidate_pooled,
        "baseline_pooled": baseline_pooled,
        "paired_bootstrap_probability_delta_f1_positive": probability,
        "checks": checks,
        "passed": all(checks.values()),
    }
    filename = "development_gate.json" if mode == "development" else "stress_gate.json"
    write_json_atomic(output_root / "experiments" / candidate_run / filename, report)
    return report
