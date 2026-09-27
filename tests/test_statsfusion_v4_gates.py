from __future__ import annotations

import json

import pandas as pd
import pytest

from bme_eating.hierarchical_v4_artifacts import _validate_m1_promotion, _validate_p1_parent
from bme_eating.hierarchical_v4_gates import (
    evaluate_crossfold_gate,
    evaluate_fold0_ablations,
    evaluate_ppg_promotion,
    evaluate_state_promotion,
    verify_gate_evidence,
)


def _json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate(
    root,
    run: str,
    metrics: dict[str, float],
    *,
    ablation_id: str | None = None,
) -> None:
    fold = root / "experiments" / run / "fold_0"
    _json(fold / "run_manifest.json", {"stage": "PROPOSALS_COMPLETE"})
    _json(fold / "decoder" / "candidate_metrics.json", metrics)
    resolved_ablation = ablation_id or f"R3-{run.upper()}"
    use_ppg = resolved_ablation == "R3-P1"
    (fold / "resolved_config.yaml").write_text(
        "experiment:\n"
        "  protocol_version: statsfusion-r3\n"
        f"  ablation_id: {resolved_ablation}\n"
        "model:\n"
        f"  use_ppg: {str(use_ppg).lower()}\n"
        "decoder:\n"
        "  use_semi_markov: false\n"
        "promotion_gate:\n"
        "  maximum_fp_per_hour_ratio: 1.05\n",
        encoding="utf-8",
    )


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


def test_fold0_ablation_gate_enforces_selected_r3_path_and_candidate_rules(
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
        ablation_id="R3-S2",
    )
    report = evaluate_fold0_ablations(
        root,
        selected_run="s2",
        sensor_only_run="s0",
        comparisons=[("statistics", "s0", "s1"), ("long_context", "s1", "s2")],
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
    assert report["comparisons"]["statistics:s1"]["passed"]
    assert report["comparisons"]["long_context:s2"]["passed"]
    verify_gate_evidence(root.parent, report)
    changed = root / "experiments" / "s0" / "fold_0" / "decoder" / "candidate_metrics.json"
    changed.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Gate evidence changed"):
        verify_gate_evidence(root.parent, report)


def test_ppg_promotion_keeps_motion_path_when_ppg_fails(tmp_path) -> None:
    root = tmp_path / "v4"
    _candidate(
        root,
        "s2",
        _state_metrics(f1=0.52, fp=10, recall=0.87, same=0.86, different=0.60, fragments=90),
        ablation_id="R3-S2",
    )
    _candidate(
        root,
        "s3",
        _state_metrics(f1=0.50, fp=12, recall=0.87, same=0.86, different=0.55, fragments=90),
        ablation_id="R3-P1",
    )
    report = evaluate_ppg_promotion(
        root,
        s2_run="s2",
        s3_run="s3",
        gate={"maximum_fp_per_hour_ratio": 1.05},
    )
    assert not report["passed"]
    assert report["resolved_use_ppg"] is False
    verify_gate_evidence(root.parent, report)


def test_p1_rejects_unresolved_motion_parent(tmp_path) -> None:
    config = {
        "experiment": {"protocol_version": "statsfusion-r3", "ablation_id": "R3-P1"},
        "model": {"use_ppg": True},
    }
    with pytest.raises(TypeError, match="prepare_hierarchical_v4_p1.py"):
        _validate_p1_parent(config, tmp_path / "v4")


def test_m1_requires_hash_locked_state_promotion_evidence(tmp_path) -> None:
    root = tmp_path / "v4"
    _candidate(
        root,
        "s2",
        _state_metrics(f1=0.52, fp=10, recall=0.87, same=0.86, different=0.60, fragments=90),
        ablation_id="R3-S2",
    )
    _candidate(
        root,
        "s3",
        _state_metrics(f1=0.50, fp=12, recall=0.87, same=0.86, different=0.55, fragments=90),
        ablation_id="R3-P1",
    )
    gate = {"maximum_fp_per_hour_ratio": 1.05}
    config = {
        "experiment": {"protocol_version": "statsfusion-r3", "ablation_id": "R3-M1"},
        "model": {"use_ppg": False},
        "decoder": {"use_semi_markov": True},
        "promotion_gate": gate,
    }
    with pytest.raises(RuntimeError, match="prepare_hierarchical_v4_m1.py"):
        _validate_m1_promotion(config, root)

    decision = evaluate_ppg_promotion(
        root,
        s2_run="s2",
        s3_run="s3",
        gate=gate,
    )
    config["experiment"]["state_promotion"] = decision
    _validate_m1_promotion(config, root)

    config["model"]["use_ppg"] = True
    with pytest.raises(RuntimeError, match="differs from its locked"):
        _validate_m1_promotion(config, root)
    config["model"]["use_ppg"] = False

    evidence = root / "experiments" / "s2" / "fold_0" / "decoder" / "candidate_metrics.json"
    evidence.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Gate evidence changed"):
        _validate_m1_promotion(config, root)


def test_m1_recomputes_decision_and_validates_source_runs(tmp_path) -> None:
    root = tmp_path / "v4"
    metrics = _state_metrics(
        f1=0.52,
        fp=10,
        recall=0.87,
        same=0.86,
        different=0.60,
        fragments=90,
    )
    _candidate(root, "s2", metrics, ablation_id="R3-S2")
    _candidate(root, "s3", metrics, ablation_id="R3-P1")
    gate = {"maximum_fp_per_hour_ratio": 1.05}
    decision = evaluate_ppg_promotion(root, s2_run="s2", s3_run="s3", gate=gate)
    decision["resolved_use_ppg"] = not decision["resolved_use_ppg"]
    config = {
        "experiment": {
            "protocol_version": "statsfusion-r3",
            "ablation_id": "R3-M1",
            "state_promotion": decision,
        },
        "model": {"use_ppg": decision["resolved_use_ppg"]},
        "decoder": {"use_semi_markov": True},
        "promotion_gate": gate,
    }
    with pytest.raises(RuntimeError, match="does not match its locked evidence"):
        _validate_m1_promotion(config, root)

    _candidate(root, "not_s2", metrics, ablation_id="R3-S1")
    with pytest.raises(RuntimeError, match="wrong ablation_id"):
        evaluate_ppg_promotion(root, s2_run="not_s2", s3_run="s3", gate=gate)


def test_domain_promotion_checks_subject_bootstrap_and_gyro_strata(tmp_path) -> None:
    root = tmp_path / "v4"
    baseline = _state_metrics(
        f1=0.50, fp=10, recall=0.84, same=0.80, different=0.50, fragments=100
    )
    candidate = _state_metrics(
        f1=0.51, fp=10, recall=0.85, same=0.79, different=0.54, fragments=95
    )
    _candidate(root, "s2", baseline, ablation_id="R3-S2")
    _candidate(root, "d1", candidate, ablation_id="R3-D1")
    for run, missing, complete, tp in (
        ("s2", 0.60, 0.80, 6),
        ("d1", 0.59, 0.78, 8),
    ):
        fold = root / "experiments" / run / "fold_0"
        _json(
            fold / "diagnostics" / "domain_metrics.json",
            {
                "gyro_strata": {
                    "missing": {"truth_count": 5, "state_only_recall": missing},
                    "complete": {"truth_count": 5, "state_only_recall": complete},
                }
            },
        )
        pd.DataFrame(
            {
                "subject_key": ["a", "b"],
                "true_positive": [tp, tp],
                "false_positive": [1, 1],
                "false_negative": [10 - tp, 10 - tp],
                "observed_hours": [1.0, 1.0],
            }
        ).to_csv(fold / "decoder" / "state_only_per_subject_metrics.csv", index=False)
    report = evaluate_state_promotion(
        root,
        baseline_run="s2",
        candidate_run="d1",
        gate_type="domain",
        gate={
            "minimum_domain_f1_improvement": 0.01,
            "minimum_different_hand_recall_improvement": 0.03,
            "maximum_fp_per_hour_ratio": 1.05,
            "maximum_same_hand_recall_drop": 0.02,
            "maximum_gyro_stratum_recall_drop": 0.03,
            "minimum_positive_bootstrap_probability": 0.80,
        },
    )
    assert report["passed"]
    assert report["checks"]["gyro_strata"]
    verify_gate_evidence(root.parent, report)


def test_semi_markov_promotion_requires_recall_or_fragment_gain_without_f1_regression(
    tmp_path,
) -> None:
    root = tmp_path / "v4"
    _candidate(
        root,
        "state",
        _state_metrics(f1=0.60, fp=10, recall=0.85, same=0.8, different=0.7, fragments=100),
        ablation_id="R3-P1",
    )
    _candidate(
        root,
        "m1",
        _state_metrics(f1=0.596, fp=10, recall=0.87, same=0.8, different=0.7, fragments=95),
        ablation_id="R3-M1",
    )
    config_path = root / "experiments" / "m1" / "fold_0" / "resolved_config.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "use_semi_markov: false", "use_semi_markov: true"
        ),
        encoding="utf-8",
    )
    report = evaluate_state_promotion(
        root,
        baseline_run="state",
        candidate_run="m1",
        gate_type="semi_markov",
        gate={
            "minimum_semi_markov_recall_improvement": 0.01,
            "minimum_fragment_reduction": 0.20,
            "maximum_ppg_f1_drop": 0.005,
            "maximum_fp_per_hour_ratio": 1.05,
        },
    )
    assert report["passed"]


def _evaluated_fold(
    root, run: str, fold: int, *, tp: int, fp: int, fn: int, observed_hours: float = 10.0
) -> None:
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
            "fp_per_hour": fp / observed_hours,
            "state_only": {
                "f1": f1,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "different_sensitivity": tp / (tp + fn),
                "fp_per_hour": fp / observed_hours,
            },
        },
    )
    per_subject = pd.DataFrame(
        {
            "subject_key": [f"subject-{fold}"],
            "true_positive": [tp],
            "false_positive": [fp],
            "false_negative": [fn],
            "observed_hours": [observed_hours],
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


def test_crossfold_gate_uses_total_false_positives_over_total_hours(tmp_path) -> None:
    root = tmp_path / "v4"
    _evaluated_fold(root, "candidate", 0, tp=10, fp=10, fn=0, observed_hours=1.0)
    _evaluated_fold(root, "candidate", 1, tp=10, fp=0, fn=0, observed_hours=100.0)
    _evaluated_fold(root, "baseline", 0, tp=10, fp=1, fn=0, observed_hours=1.0)
    _evaluated_fold(root, "baseline", 1, tp=10, fp=100, fn=0, observed_hours=100.0)
    report = evaluate_crossfold_gate(
        root,
        candidate_run="candidate",
        s0_run="baseline",
        folds=(0, 1),
        gate={
            "maximum_fp_per_hour_ratio": 1.05,
            "maximum_stress_fold_f1_drop": 1.0,
            "maximum_stress_different_sensitivity_drop": 1.0,
        },
        mode="stress",
    )
    assert report["candidate_pooled"]["fp_per_hour"] == pytest.approx(10 / 101)
    assert report["baseline_pooled"]["fp_per_hour"] == pytest.approx(1.0)
    assert report["checks"]["fp_per_hour"]
