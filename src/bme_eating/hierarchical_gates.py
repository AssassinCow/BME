from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic


def _selected_oof(output_root: Path, run_name: str, fold: int) -> dict[str, Any]:
    path = (
        output_root
        / "experiments"
        / run_name
        / f"fold_{fold}"
        / "selection"
        / "selected_pipeline.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing hierarchical selection: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def select_fold0_mode(
    output_root: Path,
    verifier_only_run: str,
    state_xgb_run: str,
    no_embedding_run: str | None = None,
) -> dict[str, Any]:
    verifier = _selected_oof(output_root, verifier_only_run, 0)
    state_xgb = _selected_oof(output_root, state_xgb_run, 0)
    f1_gain = float(state_xgb["refined_f1"]) - float(verifier["refined_f1"])
    fp_ratio = float(state_xgb["refined_false_positives_per_observed_hour"]) / max(
        float(verifier["refined_false_positives_per_observed_hour"]), 1e-12
    )
    different_drop = float(verifier["refined_different_sensitivity"]) - float(
        state_xgb["refined_different_sensitivity"]
    )
    state_xgb_passed = f1_gain >= 0.010 and fp_ratio <= 1.05 and different_drop <= 0.02
    selected_run = state_xgb_run if state_xgb_passed else verifier_only_run
    result: dict[str, Any] = {
        "scope": "fold_0_outer_train_oof_only",
        "verifier_only_run": verifier_only_run,
        "state_and_verifier_run": state_xgb_run,
        "f1_gain": f1_gain,
        "fp_per_hour_ratio": fp_ratio,
        "different_sensitivity_drop": different_drop,
        "state_and_verifier_gate_passed": state_xgb_passed,
        "selected_run": selected_run,
        "selected_xgb_mode": "state_and_verifier" if state_xgb_passed else "verifier_only",
    }
    if no_embedding_run is not None:
        embedded = _selected_oof(output_root, selected_run, 0)
        no_embedding = _selected_oof(output_root, no_embedding_run, 0)
        if no_embedding.get("xgb_mode") != embedded.get("xgb_mode"):
            raise RuntimeError(
                "Embedding ablation must use the same XGBoost mode as the selected run"
            )
        embedding_gain = float(embedded["refined_f1"]) - float(
            no_embedding["refined_f1"]
        )
        result.update(
            {
                "no_embedding_run": no_embedding_run,
                "embedding_f1_gain": embedding_gain,
                "retain_state_embedding": embedding_gain >= 0.005,
            }
        )
    return result


def _candidate_fold(output_root: Path, run_name: str, fold: int) -> dict[str, Any]:
    root = output_root / "experiments" / run_name / f"fold_{fold}"
    metrics = json.loads((root / "evaluation" / "metrics.json").read_text(encoding="utf-8"))[
        "refined"
    ]
    ablations = json.loads(
        (root / "evaluation" / "ablation_metrics.json").read_text(encoding="utf-8")
    )
    per_subject = pd.read_csv(root / "evaluation" / "per_subject_metrics.csv")
    ablation_subject = pd.read_csv(
        root / "evaluation" / "ablation_per_subject_metrics.csv"
    )
    selection = _selected_oof(output_root, run_name, fold)
    windows = pd.read_parquet(root / "outer" / "window_predictions.parquet")
    duration_ms = windows.groupby(["subject_key", "session_id"])["timestamp_ms"].agg(
        lambda values: max(0, int(values.max()) - int(values.min()))
    )
    return {
        "metrics": metrics,
        "a2": ablations["A2_state_only"],
        "per_subject": per_subject,
        "a2_per_subject": ablation_subject[
            ablation_subject["ablation"] == "A2_state_only"
        ],
        "selection": selection,
        "observed_hours": max(float(duration_ms.sum()) / 3_600_000, 1e-9),
    }


def _baseline_fold(input_root: Path, fold: int) -> dict[str, Any]:
    path = input_root / "experiments" / "baseline" / f"fold_{fold}" / "test_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "metrics": payload[str(payload["primary_method"])],
        "different_sensitivity": payload["hand_relation"]["different"]["sensitivity"],
        "per_subject": pd.DataFrame.from_dict(payload["by_subject"], orient="index")
        .rename_axis("subject_key")
        .reset_index(),
    }


def _f1_from_counts(true_positive: float, false_positive: float, false_negative: float) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def _paired_bootstrap_probability(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    iterations: int = 2000,
) -> float:
    paired = candidate[["subject_key", "f1"]].merge(
        baseline[["subject_key", "f1"]],
        on="subject_key",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError("Paired bootstrap has no shared subjects")
    differences = (
        paired["f1_candidate"].to_numpy(dtype=np.float64)
        - paired["f1_baseline"].to_numpy(dtype=np.float64)
    )
    generator = np.random.default_rng(2026)
    draws = generator.integers(0, len(differences), size=(iterations, len(differences)))
    return float((differences[draws].mean(axis=1) > 0).mean())


def evaluate_development_gate(
    input_root: Path,
    output_root: Path,
    run_name: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    candidate_subjects: list[pd.DataFrame] = []
    baseline_subjects: list[pd.DataFrame] = []
    for fold in (0, 1):
        candidate = _candidate_fold(output_root, run_name, fold)
        a0 = _baseline_fold(input_root, fold)
        a2 = candidate["a2"]
        use_a2 = float(a2["f1"]) >= float(a0["metrics"]["f1"])
        strong_metrics = a2 if use_a2 else a0["metrics"]
        strong_different = (
            float(a2["different_sensitivity"])
            if use_a2
            else float(a0["different_sensitivity"])
        )
        strong_subjects = candidate["a2_per_subject"] if use_a2 else a0["per_subject"]
        metrics = candidate["metrics"]
        rows.append(
            {
                "fold": fold,
                "strong_baseline": "A2" if use_a2 else "A0",
                "f1_delta": float(metrics["f1"]) - float(strong_metrics["f1"]),
                "fp_ratio": float(metrics["false_positives_per_observed_hour"])
                / max(float(strong_metrics["false_positives_per_observed_hour"]), 1e-12),
                "different_drop": strong_different
                - float(metrics["different_sensitivity"]),
                "seed_direction_passed": bool(
                    candidate["selection"]["verifier_seed_direction_passed"]
                )
                and bool(candidate["selection"]["boundary_seed_direction_passed"]),
            }
        )
        candidate_subjects.append(candidate["per_subject"][["subject_key", "f1"]])
        baseline_subjects.append(strong_subjects[["subject_key", "f1"]])
    frame = pd.DataFrame(rows)
    bootstrap = _paired_bootstrap_probability(
        pd.concat(candidate_subjects, ignore_index=True),
        pd.concat(baseline_subjects, ignore_index=True),
    )
    passed = bool(
        (frame["f1_delta"] > 0).all()
        and frame["f1_delta"].mean() >= 0.015
        and (frame["fp_ratio"] <= 1.05).all()
        and (frame["different_drop"] <= 0.03).all()
        and bootstrap >= 0.80
        and frame["seed_direction_passed"].all()
    )
    return {
        "phase": "folds_0_1_development",
        "run_name": run_name,
        "folds": rows,
        "mean_f1_delta": float(frame["f1_delta"].mean()),
        "paired_bootstrap_probability_delta_f1_positive": bootstrap,
        "passed": passed,
    }


def evaluate_stress_gate(
    input_root: Path,
    output_root: Path,
    run_name: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    candidate_counts = np.zeros(3, dtype=np.float64)
    baseline_counts = np.zeros(3, dtype=np.float64)
    candidate_fp = 0.0
    baseline_fp = 0.0
    observed_hours = 0.0
    for fold in (2, 3, 4):
        candidate = _candidate_fold(output_root, run_name, fold)
        baseline = _baseline_fold(input_root, fold)
        metrics = candidate["metrics"]
        baseline_metrics = baseline["metrics"]
        delta = float(metrics["f1"]) - float(baseline_metrics["f1"])
        different_drop = float(baseline["different_sensitivity"]) - float(
            metrics["different_sensitivity"]
        )
        rows.append(
            {
                "fold": fold,
                "f1_delta": delta,
                "different_drop": different_drop,
                "non_degraded": delta >= 0,
            }
        )
        candidate_counts += [
            metrics["true_positive"],
            metrics["false_positive"],
            metrics["false_negative"],
        ]
        baseline_counts += [
            baseline_metrics["true_positive"],
            baseline_metrics["false_positive"],
            baseline_metrics["false_negative"],
        ]
        candidate_fp += float(metrics["false_positive"])
        baseline_fp += float(baseline_metrics["false_positive"])
        observed_hours += float(candidate["observed_hours"])
    candidate_f1 = _f1_from_counts(*candidate_counts)
    baseline_f1 = _f1_from_counts(*baseline_counts)
    candidate_fph = candidate_fp / observed_hours
    baseline_fph = baseline_fp / observed_hours
    frame = pd.DataFrame(rows)
    passed = bool(
        int(frame["non_degraded"].sum()) >= 2
        and candidate_f1 > baseline_f1
        and float(frame["f1_delta"].min()) >= -0.03
        and candidate_fph <= baseline_fph * 1.05
        and float(frame["different_drop"].max()) <= 0.05
    )
    return {
        "phase": "folds_2_4_frozen_stress",
        "run_name": run_name,
        "folds": rows,
        "pooled_candidate_f1": candidate_f1,
        "pooled_baseline_f1": baseline_f1,
        "pooled_candidate_fp_per_hour": candidate_fph,
        "pooled_baseline_fp_per_hour": baseline_fph,
        "passed": passed,
        "independence_claim": "frozen_stress_only_not_external_validation",
    }


def write_freeze_manifest(
    output_root: Path,
    run_name: str,
    development_decision: dict[str, Any],
) -> Path:
    if not bool(development_decision.get("passed", False)):
        raise RuntimeError("Cannot freeze a hierarchical protocol that failed development gates")
    if development_decision.get("phase") != "folds_0_1_development":
        raise RuntimeError("Freeze decision is not a folds 0-1 development gate result")
    if development_decision.get("run_name") != run_name:
        raise RuntimeError("Freeze decision belongs to a different hierarchical run")
    experiment_root = output_root / "experiments" / run_name
    manifests = [experiment_root / f"fold_{fold}" / "run_manifest.json" for fold in (0, 1)]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    config_hashes = {payload["resolved_config_sha256"] for payload in payloads}
    commits = {payload["git"]["commit"] for payload in payloads}
    if len(config_hashes) != 1 or len(commits) != 1:
        raise RuntimeError("Development folds do not share one config and Git commit")
    search_contract = {
        "calibration": json.loads(
            (experiment_root / "fold_0" / "selection" / "selected_pipeline.json").read_text(
                encoding="utf-8"
            )
        )["protocol_version"],
        "development_decision": development_decision,
    }
    search_hash = hashlib.sha256(
        json.dumps(search_contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = experiment_root / "freeze_manifest.json"
    if path.exists():
        raise FileExistsError("freeze_manifest.json already exists and is immutable")
    write_json_atomic(
        path,
        {
            "version": 3,
            "run_name": run_name,
            "resolved_config_sha256": config_hashes.pop(),
            "git_commit": commits.pop(),
            "development_manifest_hashes": {
                str(fold): sha256_file(manifests[fold]) for fold in (0, 1)
            },
            "search_contract_sha256": search_hash,
            "development_gate": development_decision,
        },
    )
    return path
