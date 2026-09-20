from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_folds(root: Path) -> list[dict[str, Any]]:
    rows = []
    for fold in range(5):
        path = root / f"fold_{fold}" / "test_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing fold metrics: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = str(payload.get("primary_method", "max_cardinality_iou"))
        manifest_path = root / f"fold_{fold}" / "run_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        rows.append(
            {
                "fold": fold,
                "metrics": payload[method],
                "different": payload.get("hand_relation", {}).get("different", {}),
                "same": payload.get("hand_relation", {}).get("same", {}),
                "strict": payload.get("strict_no_ignore", {}),
                "by_subject": payload.get("by_subject", {}),
                "fingerprints": {
                    key: value
                    for key, value in manifest.get("hashes", {}).items()
                    if key
                    in {
                        "quality_report",
                        "quality_expectations",
                        "subject_folds",
                        "subject_folds_manifest",
                        "events",
                    }
                },
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    true_positive = sum(float(row["metrics"]["true_positive"]) for row in rows)
    false_positive = sum(float(row["metrics"]["false_positive"]) for row in rows)
    false_negative = sum(float(row["metrics"]["false_negative"]) for row in rows)
    precision = true_positive / max(true_positive + false_positive, 1.0)
    sensitivity = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    different_truth = sum(float(row["different"].get("truth_events") or 0) for row in rows)
    different_hits = sum(float(row["different"].get("matched_events") or 0) for row in rows)
    same_truth = sum(float(row["same"].get("truth_events") or 0) for row in rows)
    same_hits = sum(float(row["same"].get("matched_events") or 0) for row in rows)
    strict_true_positive = sum(float(row["strict"].get("true_positive") or 0) for row in rows)
    strict_false_positive = sum(float(row["strict"].get("false_positive") or 0) for row in rows)
    strict_false_negative = sum(float(row["strict"].get("false_negative") or 0) for row in rows)
    strict_precision = strict_true_positive / max(
        strict_true_positive + strict_false_positive, 1.0
    )
    strict_sensitivity = strict_true_positive / max(
        strict_true_positive + strict_false_negative, 1.0
    )
    strict_f1 = 2 * strict_precision * strict_sensitivity / max(
        strict_precision + strict_sensitivity, 1e-12
    )
    matched = max(true_positive, 1.0)
    return {
        "f1": f1,
        "precision": precision,
        "sensitivity": sensitivity,
        "different_sensitivity": different_hits / max(different_truth, 1.0),
        "same_sensitivity": same_hits / max(same_truth, 1.0),
        "strict_no_ignore_f1": strict_f1,
        "weighted_start_mae_seconds": sum(
            float(row["metrics"].get("start_mae_seconds") or 0)
            * float(row["metrics"]["true_positive"])
            for row in rows
        )
        / matched,
        "weighted_end_mae_seconds": sum(
            float(row["metrics"].get("end_mae_seconds") or 0)
            * float(row["metrics"]["true_positive"])
            for row in rows
        )
        / matched,
        "mean_false_positives_per_observed_hour": sum(
            float(row["metrics"]["false_positives_per_observed_hour"]) for row in rows
        )
        / len(rows),
    }


def _f1_from_counts(true_positive: float, false_positive: float, false_negative: float) -> float:
    precision = true_positive / max(true_positive + false_positive, 1.0)
    sensitivity = true_positive / max(true_positive + false_negative, 1.0)
    return 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)


def _subject_counts(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float, float]]:
    output: dict[str, tuple[float, float, float]] = {}
    for row in rows:
        for subject, metrics in row.get("by_subject", {}).items():
            output[str(subject)] = (
                float(metrics.get("true_positive") or 0),
                float(metrics.get("false_positive") or 0),
                float(metrics.get("false_negative") or 0),
            )
    return output


def paired_bootstrap_probability(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    iterations: int = 2000,
    seed: int = 2026,
) -> float:
    baseline = _subject_counts(baseline_rows)
    candidate = _subject_counts(candidate_rows)
    subjects = sorted(set(baseline) & set(candidate))
    if not subjects or set(baseline) != set(candidate):
        return float("nan")
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(iterations):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        baseline_counts = np.asarray([baseline[str(subject)] for subject in sampled]).sum(axis=0)
        candidate_counts = np.asarray([candidate[str(subject)] for subject in sampled]).sum(axis=0)
        baseline_f1 = _f1_from_counts(*baseline_counts)
        candidate_f1 = _f1_from_counts(*candidate_counts)
        wins += candidate_f1 > baseline_f1
    return wins / iterations


def _fingerprints_unchanged(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> bool:
    baseline = [row.get("fingerprints", {}) for row in baseline_rows]
    candidate = [row.get("fingerprints", {}) for row in candidate_rows]
    return bool(baseline[0]) and all(item == baseline[0] for item in baseline + candidate)


def evaluate_promotion(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(baseline_rows) != 5 or len(candidate_rows) != 5:
        raise ValueError("Promotion requires exactly five baseline and five candidate folds")
    baseline = _aggregate(baseline_rows)
    candidate = _aggregate(candidate_rows)
    nondegraded_folds = sum(
        float(candidate_rows[index]["metrics"]["f1"])
        >= float(baseline_rows[index]["metrics"]["f1"]) - 0.01
        for index in range(5)
    )
    bootstrap_probability = paired_bootstrap_probability(baseline_rows, candidate_rows)
    checks = {
        "all_baseline_folds_have_recall": all(
            float(row["metrics"]["sensitivity"]) > 0 for row in baseline_rows
        ),
        "all_candidate_folds_have_recall": all(
            float(row["metrics"]["sensitivity"]) > 0 for row in candidate_rows
        ),
        "micro_f1_at_least_0p490": candidate["f1"] >= 0.490,
        "at_least_three_folds_within_0p01": nondegraded_folds >= 3,
        "different_sensitivity_at_least_0p184": candidate["different_sensitivity"] >= 0.184,
        "same_sensitivity_at_least_0p623": candidate["same_sensitivity"] >= 0.623,
        "strict_no_ignore_f1_at_least_0p434": candidate["strict_no_ignore_f1"] >= 0.434,
        "false_positives_per_hour_at_most_0p02844": (
            candidate["mean_false_positives_per_observed_hour"] <= 0.02844
        ),
        "weighted_start_mae_at_most_256_seconds": (
            candidate["weighted_start_mae_seconds"] <= 256.0
        ),
        "weighted_end_mae_at_most_102_seconds": (
            candidate["weighted_end_mae_seconds"] <= 102.0
        ),
        "paired_bootstrap_probability_at_least_0p80": bootstrap_probability >= 0.80,
        "data_and_fold_fingerprints_unchanged": _fingerprints_unchanged(
            baseline_rows, candidate_rows
        ),
    }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "nondegraded_folds": nondegraded_folds,
        "paired_bootstrap_iterations": 2000,
        "paired_bootstrap_probability": bootstrap_probability,
        "checks": checks,
        "promote": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the five-fold model promotion gate.")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline experiment root.")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate experiment root.")
    parser.add_argument("--output", type=Path, required=True, help="JSON decision output.")
    args = parser.parse_args()
    baseline_rows = _load_folds(args.baseline)
    candidate_rows = _load_folds(args.candidate)
    output = evaluate_promotion(baseline_rows, candidate_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if not output["promote"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
