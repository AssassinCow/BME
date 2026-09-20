from scripts.compare_models import evaluate_promotion


def _rows(true_positive: int, false_positive: int, false_negative: int, fp_hour: float):
    precision = true_positive / (true_positive + false_positive)
    sensitivity = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * sensitivity / (precision + sensitivity)
    return [
        {
            "fold": fold,
            "metrics": {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "sensitivity": sensitivity,
                "f1": f1,
                "false_positives_per_observed_hour": fp_hour,
                "start_mae_seconds": 100.0,
                "end_mae_seconds": 50.0,
            },
            "different": {"truth_events": 10, "matched_events": 8},
            "same": {"truth_events": 10, "matched_events": 8},
            "strict": {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
            },
            "by_subject": {
                f"subject-{fold}": {
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                }
            },
            "fingerprints": {"subject_folds": "same", "events": "same"},
        }
        for fold in range(5)
    ]


def test_model_promotion_accepts_candidate_meeting_all_gates():
    baseline = _rows(45, 35, 55, 0.02)
    candidate = _rows(76, 24, 24, 0.025)
    decision = evaluate_promotion(baseline, candidate)
    assert decision["promote"]


def test_model_promotion_rejects_any_zero_recall_fold():
    baseline = _rows(45, 35, 55, 0.02)
    candidate = _rows(76, 24, 24, 0.025)
    candidate[2]["metrics"].update(
        {"true_positive": 0, "false_negative": 100, "sensitivity": 0.0, "f1": 0.0}
    )
    decision = evaluate_promotion(baseline, candidate)
    assert not decision["promote"]
    assert not decision["checks"]["all_candidate_folds_have_recall"]
