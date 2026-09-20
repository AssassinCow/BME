import json

import pandas as pd

from bme_eating.reporting import generate_experiment_report


def _write_fold(root, fold, f1, sensitivity, different_hits):
    fold_dir = root / f"fold_{fold}"
    fold_dir.mkdir(parents=True)
    true_positive = 4
    false_positive = 1
    false_negative = 4
    payload = {
        "primary_method": "max_cardinality_iou",
        "max_cardinality_iou": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "ignored_predictions": 0,
            "precision": 0.8,
            "sensitivity": sensitivity,
            "f1": f1,
            "start_mae_seconds": 30.0,
            "end_mae_seconds": 45.0,
            "false_positives_per_observed_hour": 0.05,
        },
        "hand_relation": {
            "same": {"truth_events": 4, "matched_events": 3, "sensitivity": 0.75},
            "different": {
                "truth_events": 4,
                "matched_events": different_hits,
                "sensitivity": different_hits / 4,
            },
        },
        "by_subject": {
            f"subject-{fold}": {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": 0.8,
                "sensitivity": sensitivity,
                "f1": f1,
            }
        },
        "evaluation_counts": {
            "evaluable_truth": 8,
            "ignored_truth": 1,
            "predicted_events": 5,
        },
    }
    (fold_dir / "test_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    (fold_dir / "metadata.json").write_text(
        json.dumps({"validation_auprc": 0.4, "final_estimators": 10}),
        encoding="utf-8",
    )
    (fold_dir / "selected_postprocess.json").write_text(
        json.dumps({"high_threshold": 0.5, "low_threshold": 0.1}),
        encoding="utf-8",
    )


def test_result_report_generates_tables_and_dashboard(tmp_path):
    experiment_root = tmp_path / "experiments" / "baseline"
    for fold in range(5):
        _write_fold(experiment_root, fold, 0.6, 0.5, 1)
    paths = generate_experiment_report(
        {"baseline": experiment_root}, tmp_path / "report"
    )

    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    aggregate = pd.read_csv(paths["aggregate_csv"]).iloc[0]
    folds = pd.read_csv(paths["fold_csv"])

    assert summary["provisional"] is False
    assert aggregate.folds_complete == 5
    assert aggregate.micro_f1 == 0.6153846153846154
    assert aggregate.different_sensitivity == 0.25
    assert len(folds) == 5
    assert "进食检测实验结果仪表板" in paths["html"].read_text(encoding="utf-8")


def test_result_report_marks_missing_folds_as_provisional(tmp_path):
    experiment_root = tmp_path / "experiments" / "baseline"
    _write_fold(experiment_root, 0, 0.6, 0.5, 1)

    paths = generate_experiment_report(
        {"baseline": experiment_root}, tmp_path / "report"
    )
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    folds = pd.read_csv(paths["fold_csv"])
    dashboard = paths["html"].read_text(encoding="utf-8")

    assert summary["provisional"] is True
    assert "仅完成 1/5 折" in summary["warnings"][0]
    assert folds.status.tolist().count("missing") == 4
    assert dashboard.count("style='width:0.0%'") == 4
