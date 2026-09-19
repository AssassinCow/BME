from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_folds(root: Path) -> list[dict[str, Any]]:
    rows = []
    for fold in range(5):
        path = root / f"fold_{fold}" / "test_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing fold metrics: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = str(payload.get("primary_method", "max_cardinality_iou"))
        rows.append(
            {
                "fold": fold,
                "metrics": payload[method],
                "different": payload.get("hand_relation", {}).get("different", {}),
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
    return {
        "f1": f1,
        "precision": precision,
        "sensitivity": sensitivity,
        "different_sensitivity": different_hits / max(different_truth, 1.0),
        "mean_false_positives_per_observed_hour": sum(
            float(row["metrics"]["false_positives_per_observed_hour"]) for row in rows
        )
        / len(rows),
    }


def evaluate_promotion(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(baseline_rows) != 5 or len(candidate_rows) != 5:
        raise ValueError("Promotion requires exactly five baseline and five candidate folds")
    baseline = _aggregate(baseline_rows)
    candidate = _aggregate(candidate_rows)
    nondegraded_folds = sum(
        float(candidate_rows[index]["metrics"]["f1"])
        >= float(baseline_rows[index]["metrics"]["f1"])
        for index in range(5)
    )
    fp_limit = baseline["mean_false_positives_per_observed_hour"] * 1.2
    checks = {
        "all_baseline_folds_have_recall": all(
            float(row["metrics"]["sensitivity"]) > 0 for row in baseline_rows
        ),
        "all_candidate_folds_have_recall": all(
            float(row["metrics"]["sensitivity"]) > 0 for row in candidate_rows
        ),
        "aggregate_f1_gain_at_least_0p02": candidate["f1"] - baseline["f1"] >= 0.02,
        "at_least_three_folds_nondegraded": nondegraded_folds >= 3,
        "different_sensitivity_drop_at_most_0p02": (
            candidate["different_sensitivity"] >= baseline["different_sensitivity"] - 0.02
        ),
        "false_positives_per_hour_at_most_1p2x": (
            candidate["mean_false_positives_per_observed_hour"] <= fp_limit
        ),
    }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "nondegraded_folds": nondegraded_folds,
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
