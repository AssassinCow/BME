import hashlib
import json

import pandas as pd
import pytest

from bme_eating.fusion import FROZEN_INPUT_HASHES
from bme_eating.fusion_v4_promotion import (
    evaluate_v4_final_promotion,
    load_v4_promotion_evidence,
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(subject, true_positive, false_positive, false_negative):
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / denominator
    return {
        "primary_method": "max_cardinality_iou",
        "max_cardinality_iou": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "f1": f1,
            "start_mae_seconds": 100.0,
            "end_mae_seconds": 50.0,
        },
        "strict_no_ignore": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "f1": f1,
        },
        "hand_relation": {
            "different": {"truth_events": 5, "matched_events": true_positive / 2},
            "same": {"truth_events": 5, "matched_events": true_positive / 2},
        },
        "by_subject": {
            subject: {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "f1": f1,
            }
        },
    }


def _write_fold(root, fold, *, candidate):
    fold_dir = root / f"fold_{fold}"
    fold_dir.mkdir(parents=True)
    subject = f"subject-{fold}"
    predictions = pd.DataFrame(
        {
            "subject_key": [subject, subject],
            "session_id": [f"session-{fold}", f"session-{fold}"],
            "timestamp_ms": [0, 3000],
        }
    )
    predictions.to_parquet(fold_dir / "test_predictions.parquet", index=False)
    metrics = _metrics(subject, 7 if candidate else 5, 3, 3 if candidate else 5)
    (fold_dir / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    artifacts = {
        name: _digest(fold_dir / name)
        for name in ("test_predictions.parquet", "test_metrics.json")
    }
    if candidate:
        selection = {
            "protocol_version": 4,
            "run_name": root.name,
            "fold": fold,
            "meta_oof_gate": {"passed": True},
            "outer_fold_gate": {"passed": True, "diagnostic_only": False},
        }
        (fold_dir / "selected_fusion.json").write_text(
            json.dumps(selection), encoding="utf-8"
        )
        artifacts["selected_fusion.json"] = _digest(fold_dir / "selected_fusion.json")
    manifest = {
        "experiment": {"name": root.name, "fold": fold},
        "git": {"commit": "candidate" if candidate else "baseline", "dirty": False},
        "resolved_config_sha256": "c" * 64 if candidate else "b" * 64,
        "hashes": {name: "a" * 64 for name in FROZEN_INPUT_HASHES},
        "artifact_hashes": artifacts,
    }
    (fold_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_runs(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "baseline_dtp_fusion_v4"
    for fold in range(1, 5):
        _write_fold(baseline, fold, candidate=False)
        _write_fold(candidate, fold, candidate=True)
    return baseline, candidate


def _gate():
    return {
        "minimum_f1_improvement": 0.02,
        "maximum_fp_per_hour_ratio": 1.2,
        "maximum_boundary_mae_ratio": 1.1,
        "minimum_nondegraded_folds": 3,
        "maximum_fold_f1_drop": 0.03,
    }


def test_final_v4_promotion_accepts_four_fold_improvement(tmp_path):
    baseline, candidate = _write_runs(tmp_path)
    baseline_rows, candidate_rows, evidence = load_v4_promotion_evidence(
        baseline, candidate
    )

    decision = evaluate_v4_final_promotion(baseline_rows, candidate_rows, _gate())

    assert decision["promote"]
    assert decision["nondegraded_folds"] == 4
    assert set(evidence["folds"]) == {"1", "2", "3", "4"}


def test_final_v4_promotion_rejects_large_single_fold_drop(tmp_path):
    baseline, candidate = _write_runs(tmp_path)
    baseline_rows, candidate_rows, _ = load_v4_promotion_evidence(baseline, candidate)
    candidate_rows[2]["f1"] = baseline_rows[2]["f1"] - 0.031

    decision = evaluate_v4_final_promotion(baseline_rows, candidate_rows, _gate())

    assert not decision["promote"]
    assert not decision["checks"]["maximum_fold_f1_drop"]


def test_final_v4_promotion_rejects_mutated_frozen_artifact(tmp_path):
    baseline, candidate = _write_runs(tmp_path)
    path = candidate / "fold_2" / "test_metrics.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed after manifesting"):
        load_v4_promotion_evidence(baseline, candidate)
