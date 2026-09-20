import hashlib
import json

import pytest

from scripts.compare_models import _load_folds, evaluate_promotion


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


def test_model_promotion_rejects_failed_internal_fusion_gate():
    baseline = _rows(45, 35, 55, 0.02)
    candidate = _rows(76, 24, 24, 0.025)
    for row in candidate:
        row["provenance"] = {"selection_run_name": "baseline_dtp_fusion_clean_test"}
        row["internal_fusion_gate_passed"] = True
    candidate[3]["internal_fusion_gate_passed"] = False

    decision = evaluate_promotion(baseline, candidate)

    assert not decision["promote"]
    assert not decision["checks"]["all_internal_fusion_gates_passed"]


def _write_experiment(root, *, dirty=False):
    indices = root.parent.parent / "indices"
    indices.mkdir(parents=True, exist_ok=True)
    input_hashes = {}
    for name, filename in {
        "quality_report": "quality_report.json",
        "quality_expectations": "quality_expectations.json",
        "subject_folds": "subject_folds.json",
        "subject_folds_manifest": "subject_folds.manifest.json",
        "events": "events.parquet",
        "anchors": "anchors.parquet",
        "segments": "segments.parquet",
    }.items():
        input_path = indices / filename
        input_path.write_bytes(name.encode())
        input_hashes[name] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    for fold in range(5):
        fold_dir = root / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        metrics = {
            "primary_method": "max_cardinality_iou",
            "max_cardinality_iou": _rows(5, 1, 2, 0.01)[fold]["metrics"],
            "hand_relation": {
                "different": {"truth_events": 1, "matched_events": 1},
                "same": {"truth_events": 1, "matched_events": 1},
            },
            "strict_no_ignore": {
                "true_positive": 5,
                "false_positive": 1,
                "false_negative": 2,
            },
            "by_subject": {
                f"subject-{fold}": {
                    "true_positive": 5,
                    "false_positive": 1,
                    "false_negative": 2,
                }
            },
        }
        metrics_path = fold_dir / "test_metrics.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        manifest = {
            "version": 2,
            "experiment": {"name": root.name, "fold": fold},
            "git": {"commit": "abc1234", "dirty": dirty},
            "resolved_config_sha256": "config-hash",
            "hashes": input_hashes,
            "artifact_hashes": {
                "test_metrics.json": hashlib.sha256(metrics_path.read_bytes()).hexdigest()
            },
        }
        (fold_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_model_comparison_rejects_dirty_or_mutated_evidence(tmp_path):
    dirty_root = tmp_path / "dirty" / "experiments" / "dirty_candidate"
    _write_experiment(dirty_root, dirty=True)
    with pytest.raises(RuntimeError, match="dirty working tree"):
        _load_folds(dirty_root)

    clean_root = tmp_path / "clean" / "experiments" / "clean_candidate"
    _write_experiment(clean_root)
    metrics_path = clean_root / "fold_2" / "test_metrics.json"
    metrics_path.write_text(metrics_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after manifesting"):
        _load_folds(clean_root)


def test_model_comparison_requires_declared_crossfit_artifacts(tmp_path):
    root = tmp_path / "clean" / "experiments" / "baseline_dtp_fusion_clean_test"
    _write_experiment(root)
    fold_dir = root / "fold_0"
    selection_path = fold_dir / "selected_fusion.json"
    selection_path.write_text(
        json.dumps(
            {
                "version": 3,
                "run_name": root.name,
                "crossfit_partitions": 3,
                "internal_gate": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    manifest_path = fold_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["selected_fusion.json"] = hashlib.sha256(
        selection_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="lacks cross-fit artifacts"):
        _load_folds(root)


def test_model_comparison_rejects_mixed_commits_and_duplicate_subjects(tmp_path):
    mixed_root = tmp_path / "mixed" / "experiments" / "candidate"
    _write_experiment(mixed_root)
    manifest_path = mixed_root / "fold_4" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["git"]["commit"] = "def5678"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="mix Git commits"):
        _load_folds(mixed_root)

    duplicate_root = tmp_path / "duplicate" / "experiments" / "candidate"
    _write_experiment(duplicate_root)
    metrics_path = duplicate_root / "fold_1" / "test_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["by_subject"] = {"subject-0": metrics["by_subject"].pop("subject-1")}
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    manifest_path = duplicate_root / "fold_1" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["test_metrics.json"] = hashlib.sha256(
        metrics_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="more than one fold"):
        _load_folds(duplicate_root)
