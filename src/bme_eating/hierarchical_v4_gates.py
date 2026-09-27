from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.v4_protocol import PROTOCOL_VERSION


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

    collect(report)
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
    expected_ablations: set[str],
    expected_use_ppg: bool,
) -> tuple[Path, dict[str, Any]]:
    config_path = output_root / "experiments" / run_name / "fold_0" / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    experiment = config.get("experiment", {})
    if experiment.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"PPG source run must use {PROTOCOL_VERSION}")
    ablation = str(experiment.get("ablation_id", ""))
    if ablation not in expected_ablations:
        raise RuntimeError(
            f"PPG source run has the wrong ablation_id {ablation!r}; "
            f"expected one of {sorted(expected_ablations)}"
        )
    if bool(config.get("model", {}).get("use_ppg", True)) != expected_use_ppg:
        raise RuntimeError("PPG source run has the wrong model.use_ppg value")
    return config_path, config


def _ppg_comparison_identity(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    normalized.pop("_config_path", None)
    experiment = normalized.get("experiment", {})
    for key in ("name", "ablation_id", "variant", "ppg_promotion", "motion_parent"):
        experiment.pop(key, None)
    normalized.setdefault("model", {})["use_ppg"] = False
    return normalized


def evaluate_ppg_promotion(
    output_root: Path,
    *,
    s2_run: str,
    s3_run: str,
    gate: dict[str, Any],
) -> dict[str, Any]:
    if s2_run == s3_run:
        raise RuntimeError("Motion and PPG promotion evidence must come from distinct runs")
    motion_path, motion_config = _validate_ppg_source_run(
            output_root,
            s2_run,
            expected_ablations={"R3-S2", "R3-D1", "R3-D2a", "R3-D2b", "R3-D2c"},
            expected_use_ppg=False,
        )
    ppg_path, ppg_config = _validate_ppg_source_run(
            output_root,
            s3_run,
            expected_ablations={"R3-P1"},
            expected_use_ppg=True,
        )
    if _ppg_comparison_identity(motion_config) != _ppg_comparison_identity(ppg_config):
        raise RuntimeError("Motion and PPG source runs differ in fields other than PPG enablement")
    source_configs = {"MOTION": motion_path, "PPG": ppg_path}
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
        for name, run in (("MOTION", s2_run), ("PPG", s3_run))
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
        "protocol_version": PROTOCOL_VERSION,
        "source_runs": {"MOTION": s2_run, "PPG": s3_run},
        "evidence_sha256": evidence,
        "f1_delta_vs_motion": f1_delta,
        "candidate_recall_delta_vs_motion": recall_delta,
        "different_sensitivity_delta_vs_motion": different_delta,
        "checks": checks,
        "passed": all(checks.values()),
        "resolved_use_ppg": all(checks.values()),
    }


def _fold0_run_evidence(output_root: Path, run_name: str) -> dict[str, dict[str, str]]:
    root = output_root / "experiments" / run_name / "fold_0"
    evidence = {
        "resolved_config": _evidence_file(root / "resolved_config.yaml", output_root.parent),
        "run_manifest": _evidence_file(root / "run_manifest.json", output_root.parent),
        "candidate_metrics": _evidence_file(
            root / "decoder" / "candidate_metrics.json", output_root.parent
        ),
    }
    for name, relative in (
        ("domain_metrics", "diagnostics/domain_metrics.json"),
        ("state_only_per_subject", "decoder/state_only_per_subject_metrics.csv"),
    ):
        path = root / relative
        if path.is_file():
            evidence[name] = _evidence_file(path, output_root.parent)
    return evidence


def _fold0_run_config(output_root: Path, run_name: str) -> dict[str, Any]:
    path = output_root / "experiments" / run_name / "fold_0" / "resolved_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if config.get("experiment", {}).get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"Fold-0 promotion evidence must use {PROTOCOL_VERSION}")
    return config


def _metric_delta(candidate: dict[str, Any], baseline: dict[str, Any], name: str) -> float:
    return float(candidate[name]) - float(baseline[name])


def evaluate_state_promotion(
    output_root: Path,
    *,
    baseline_run: str,
    candidate_run: str,
    gate_type: str,
    gate: dict[str, Any],
) -> dict[str, Any]:
    if baseline_run == candidate_run:
        raise ValueError("Promotion baseline and candidate runs must differ")
    if gate_type == "ppg":
        return evaluate_ppg_promotion(
            output_root,
            s2_run=baseline_run,
            s3_run=candidate_run,
            gate=gate,
        )
    if gate_type not in {"statistics", "long_context", "domain", "semi_markov"}:
        raise ValueError(f"Unsupported StatsFusion-r3 promotion gate: {gate_type}")
    baseline_config = _fold0_run_config(output_root, baseline_run)
    candidate_config = _fold0_run_config(output_root, candidate_run)
    baseline = _candidate_metrics(output_root, baseline_run)
    candidate = _candidate_metrics(output_root, candidate_run)
    evidence = {
        "baseline": _fold0_run_evidence(output_root, baseline_run),
        "candidate": _fold0_run_evidence(output_root, candidate_run),
    }
    f1_delta = _metric_delta(candidate, baseline, "state_only_f1")
    recall_delta = _metric_delta(candidate, baseline, "candidate_recall")
    different_delta = _metric_delta(candidate, baseline, "different_sensitivity")
    same_delta = _metric_delta(candidate, baseline, "same_sensitivity")
    fragment_reduction = 1.0 - float(candidate["state_fragment_count"]) / max(
        float(baseline["state_fragment_count"]), 1.0
    )
    common_checks = {
        "fp_per_hour": float(candidate["state_only_fp_per_hour"])
        <= float(baseline["state_only_fp_per_hour"])
        * float(gate.get("maximum_fp_per_hour_ratio", 1.05))
    }
    diagnostics: dict[str, Any] = {
        "f1_delta": f1_delta,
        "candidate_recall_delta": recall_delta,
        "different_sensitivity_delta": different_delta,
        "same_sensitivity_delta": same_delta,
        "fragment_reduction": fragment_reduction,
    }
    if gate_type == "statistics":
        checks = {
            **common_checks,
            "f1_gain": f1_delta >= float(gate["minimum_statistics_f1_improvement"]),
            "different_sensitivity": different_delta
            >= -float(gate.get("maximum_statistics_different_sensitivity_drop", 0.02)),
        }
    elif gate_type == "long_context":
        checks = {
            "recall_or_f1_fragment": bool(
                recall_delta >= float(gate["minimum_long_context_recall_improvement"])
                or (
                    f1_delta >= float(gate["minimum_long_context_f1_improvement"])
                    and fragment_reduction >= float(gate["minimum_fragment_reduction"])
                )
            )
        }
    elif gate_type == "semi_markov":
        checks = {
            **common_checks,
            "recall_or_fragment": bool(
                recall_delta >= float(gate["minimum_semi_markov_recall_improvement"])
                or fragment_reduction >= float(gate["minimum_fragment_reduction"])
            ),
            "f1_non_degradation": f1_delta
            >= -float(gate.get("maximum_ppg_f1_drop", 0.005)),
        }
        if bool(baseline_config.get("decoder", {}).get("use_semi_markov", False)):
            raise RuntimeError("Semi-Markov baseline must have decoder.use_semi_markov=false")
        if not bool(candidate_config.get("decoder", {}).get("use_semi_markov", False)):
            raise RuntimeError("Semi-Markov candidate must have decoder.use_semi_markov=true")
    else:
        baseline_domain_path = (
            output_root / "experiments" / baseline_run / "fold_0" / "diagnostics" / "domain_metrics.json"
        )
        candidate_domain_path = (
            output_root / "experiments" / candidate_run / "fold_0" / "diagnostics" / "domain_metrics.json"
        )
        baseline_subject_path = (
            output_root
            / "experiments"
            / baseline_run
            / "fold_0"
            / "decoder"
            / "state_only_per_subject_metrics.csv"
        )
        candidate_subject_path = (
            output_root
            / "experiments"
            / candidate_run
            / "fold_0"
            / "decoder"
            / "state_only_per_subject_metrics.csv"
        )
        baseline_domain = _read_json(baseline_domain_path)
        candidate_domain = _read_json(candidate_domain_path)
        probability = _bootstrap_probability(
            pd.read_csv(candidate_subject_path),
            pd.read_csv(baseline_subject_path),
            replicates=1000,
            seed=2026,
        )
        gyro_checks: dict[str, bool] = {}
        for stratum in ("missing", "complete"):
            baseline_stratum = baseline_domain["gyro_strata"][stratum]
            candidate_stratum = candidate_domain["gyro_strata"][stratum]
            truth_count = min(
                int(baseline_stratum["truth_count"]), int(candidate_stratum["truth_count"])
            )
            if truth_count >= 5:
                gyro_checks[stratum] = float(candidate_stratum["state_only_recall"]) >= float(
                    baseline_stratum["state_only_recall"]
                ) - float(gate["maximum_gyro_stratum_recall_drop"])
        diagnostics["paired_bootstrap_probability_delta_f1_positive"] = probability
        diagnostics["gyro_checks"] = gyro_checks
        checks = {
            **common_checks,
            "overall_or_different_hand_gain": bool(
                f1_delta >= float(gate["minimum_domain_f1_improvement"])
                or different_delta >= float(gate["minimum_different_hand_recall_improvement"])
            ),
            "same_hand": same_delta >= -float(gate["maximum_same_hand_recall_drop"]),
            "gyro_strata": all(gyro_checks.values()),
            "paired_bootstrap": probability
            >= float(gate["minimum_positive_bootstrap_probability"]),
        }
    return {
        "schema_version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "gate_type": gate_type,
        "source_runs": {"baseline": baseline_run, "candidate": candidate_run},
        "source_ablations": {
            "baseline": baseline_config.get("experiment", {}).get("ablation_id"),
            "candidate": candidate_config.get("experiment", {}).get("ablation_id"),
        },
        "evidence_sha256": evidence,
        "diagnostics": diagnostics,
        "checks": checks,
        "passed": all(checks.values()),
    }


def evaluate_fold0_ablations(
    output_root: Path,
    *,
    selected_run: str,
    comparisons: list[tuple[str, str, str]],
    gate: dict[str, Any],
    sensor_only_run: str,
) -> dict[str, Any]:
    selected = _candidate_metrics(output_root, selected_run)
    sensor = _candidate_metrics(output_root, sensor_only_run)
    module_reports: dict[str, dict[str, Any]] = {}
    for gate_type, baseline, candidate in comparisons:
        key = f"{gate_type}:{candidate}"
        if key in module_reports:
            raise ValueError(f"Duplicate fold-0 comparison: {key}")
        module_reports[key] = evaluate_state_promotion(
            output_root,
            baseline_run=baseline,
            candidate_run=candidate,
            gate_type=gate_type,
            gate=gate,
        )
    candidate_checks = {
        "state_calibration": bool(selected["state_calibration_gate_passed"]),
        "minimum_recall": float(selected["candidate_recall"])
        >= float(gate["minimum_candidate_recall"]),
        "same_side_not_worse_than_sensor_only": float(selected["same_candidate_recall"])
        >= float(sensor["same_candidate_recall"]) - 0.03,
        "different_side_not_worse_than_sensor_only": float(
            selected["different_candidate_recall"]
        )
        >= float(sensor["different_candidate_recall"]) - 0.03,
    }
    report = {
        "schema_version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "selected_run": selected_run,
        "sensor_only_run": sensor_only_run,
        "comparisons": module_reports,
        "evidence_sha256": {
            "selected": _fold0_run_evidence(output_root, selected_run),
            "sensor_only": _fold0_run_evidence(output_root, sensor_only_run),
        },
        "candidate_gate": {
            "checks": candidate_checks,
            "passed": all(candidate_checks.values()),
        },
    }
    report["passed"] = bool(
        all(value["passed"] for value in module_reports.values())
        and report["candidate_gate"]["passed"]
    )
    path = output_root / "experiments" / selected_run / "ablation" / "fold_0_report.json"
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
