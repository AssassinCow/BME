from __future__ import annotations

import json

import pandas as pd
import pytest

from bme_eating.hierarchical_v4_gates import (
    evaluate_crossfold_gate,
    evaluate_fold0_ablations,
    verify_gate_evidence,
)


def _json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate(root, run: str, metrics: dict[str, float]) -> None:
    fold = root / "experiments" / run / "fold_0"
    _json(fold / "run_manifest.json", {"stage": "PROPOSALS_COMPLETE"})
    _json(fold / "decoder" / "candidate_metrics.json", metrics)


def _state_metrics(
    *, f1: float, fp: float, recall: float, same: float, different: float, fragments: float
) -> dict[str, float]:
    return {
        "state_only_f1": f1,
        "state_only_fp_per_hour": fp,
        "candidate_recall": recall,
        "same_candidate_recall": same,
        "different_candidate_recall": different,
        "same_sensitivity": same,
        "different_sensitivity": different,
        "state_fragment_count": fragments,
        "state_calibration_gate_passed": True,
    }


def test_fold0_ablation_gate_enforces_statistics_long_context_and_candidate_rules(
    tmp_path,
) -> None:
    root = tmp_path / "v4"
    _candidate(
        root,
        "s0",
        _state_metrics(f1=0.50, fp=10, recall=0.82, same=0.84, different=0.60, fragments=100),
    )
    _candidate(
        root,
        "s1",
        _state_metrics(f1=0.52, fp=10, recall=0.84, same=0.85, different=0.59, fragments=95),
    )
    _candidate(
        root,
        "s2",
        _state_metrics(f1=0.52, fp=10, recall=0.87, same=0.86, different=0.60, fragments=90),
    )
    _candidate(
        root,
        "s3",
        _state_metrics(f1=0.54, fp=9, recall=0.88, same=0.86, different=0.60, fragments=85),
    )
    _candidate(
        root,
        "s4",
        _state_metrics(f1=0.55, fp=9, recall=0.90, same=0.86, different=0.59, fragments=85),
    )
    _candidate(
        root,
        "ppg",
        _state_metrics(f1=0.30, fp=12, recall=0.60, same=0.60, different=0.60, fragments=120),
    )
    report = evaluate_fold0_ablations(
        root,
        s0_run="s0",
        s1_run="s1",
        s2_run="s2",
        s3_run="s3",
        s4_run="s4",
        ppg_only_run="ppg",
        gate={
            "minimum_statistics_f1_improvement": 0.01,
            "maximum_fp_per_hour_ratio": 1.05,
            "maximum_statistics_different_sensitivity_drop": 0.02,
            "minimum_long_context_recall_improvement": 0.02,
            "minimum_long_context_f1_improvement": 0.01,
            "minimum_fragment_reduction": 0.20,
            "minimum_candidate_recall": 0.83,
        },
    )
    assert report["passed"]
    assert report["statistics_branch"]["passed"]
    assert report["long_context"]["passed"]
    verify_gate_evidence(root.parent, report)
    changed = root / "experiments" / "s0" / "fold_0" / "decoder" / "candidate_metrics.json"
    changed.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Gate evidence changed"):
        verify_gate_evidence(root.parent, report)


def _evaluated_fold(root, run: str, fold: int, *, tp: int, fp: int, fn: int) -> None:
    f1 = 2 * tp / (2 * tp + fp + fn)
    directory = root / "experiments" / run / f"fold_{fold}" / "evaluation"
    _json(
        directory / "metrics.json",
        {
            "max_cardinality_iou": {
                "f1": f1,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
            },
            "hand": {"different_sensitivity": tp / (tp + fn)},
            "fp_per_hour": fp / 10,
            "state_only": {
                "f1": f1,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "different_sensitivity": tp / (tp + fn),
                "fp_per_hour": fp / 10,
            },
        },
    )
    per_subject = pd.DataFrame(
        {
            "subject_key": [f"subject-{fold}"],
            "true_positive": [tp],
            "false_positive": [fp],
            "false_negative": [fn],
            "observed_hours": [10.0],
        }
    )
    per_subject.to_csv(directory / "per_subject_metrics.csv", index=False)
    per_subject.to_csv(directory / "state_only_per_subject_metrics.csv", index=False)


def test_development_gate_uses_subject_paired_evidence(tmp_path) -> None:
    root = tmp_path / "v4"
    for fold in (0, 1):
        _evaluated_fold(root, "candidate", fold, tp=9, fp=1, fn=1)
        _evaluated_fold(root, "baseline", fold, tp=7, fp=2, fn=3)
    report = evaluate_crossfold_gate(
        root,
        candidate_run="candidate",
        s0_run="baseline",
        folds=(0, 1),
        gate={
            "minimum_mean_f1_improvement": 0.02,
            "maximum_fp_per_hour_ratio": 1.05,
            "maximum_different_sensitivity_drop": 0.03,
            "minimum_positive_bootstrap_probability": 0.80,
        },
        mode="development",
    )
    assert report["passed"]
    assert report["paired_bootstrap_probability_delta_f1_positive"] == 1.0
