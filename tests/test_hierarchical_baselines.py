from __future__ import annotations

import json

import pandas as pd
import pytest

import bme_eating.training.hierarchical_trainer as trainer
from bme_eating.hierarchical_artifacts import sha256_file


def _config(tmp_path) -> dict[str, object]:
    return {
        "_input_artifact_root": str(tmp_path),
        "postprocess": {"matching_method": "max_cardinality_iou"},
        "historical_baselines": {
            "hard_dyadic_experiment": "historical-dtp",
            "hard_dyadic_metrics_file": "diagnostics.json",
        },
    }


def _patch_xgboost_metrics(monkeypatch, metrics: dict[str, float]) -> None:
    monkeypatch.setattr(
        trainer,
        "_load_xgb_candidate_events",
        lambda *_arguments: (pd.DataFrame([{"subject_key": "train"}]), None),
    )
    monkeypatch.setattr(
        trainer,
        "_metrics_with_diagnostics",
        lambda *_arguments: (metrics, pd.DataFrame()),
    )


def _write_historical_diagnostic(tmp_path, expected_hash: str | None = None) -> None:
    source_root = tmp_path / "experiments" / "historical-dtp" / "fold_0"
    source_root.mkdir(parents=True)
    metrics_path = source_root / "diagnostics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "oof_event_metrics_in_sample_postprocess_selection": {
                    "max_cardinality_iou": {
                        "f1": 0.42,
                        "false_positives_per_observed_hour": 0.18,
                    },
                    "hand_relation": {
                        "same": {"sensitivity": 0.7},
                        "different": {"sensitivity": 0.5},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "artifacts": {
            metrics_path.name: expected_hash or sha256_file(metrics_path),
        }
    }
    (source_root / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _registered_rows(config, fold: int) -> list[dict[str, object]]:
    return trainer._registered_baseline_ablations(
        config,
        fold,
        {"train"},
        {"test"},
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )


def test_registered_a0_writes_prefixed_oof_metrics(tmp_path, monkeypatch) -> None:
    _patch_xgboost_metrics(
        monkeypatch,
        {"f1": 0.55, "false_positives_per_observed_hour": 0.2},
    )

    row = _registered_rows(_config(tmp_path), fold=1)[0]

    assert row == {
        "ablation": "A0_frozen_xgboost",
        "eligible": True,
        "evidence_scope": "outer_train_subject_oof_frozen_v2",
        "refined_f1": 0.55,
        "refined_false_positives_per_observed_hour": 0.2,
    }


def test_registered_a1_rejects_diagnostic_hash_mismatch(tmp_path, monkeypatch) -> None:
    _patch_xgboost_metrics(monkeypatch, {"f1": 0.55})
    _write_historical_diagnostic(tmp_path, expected_hash="0" * 64)

    with pytest.raises(RuntimeError, match="diagnostic hash mismatch"):
        _registered_rows(_config(tmp_path), fold=0)


@pytest.mark.parametrize("fold", [1, 2, 3, 4])
def test_registered_a1_is_fold0_history_only(tmp_path, monkeypatch, fold) -> None:
    _patch_xgboost_metrics(monkeypatch, {"f1": 0.55})

    row = _registered_rows(_config(tmp_path), fold=fold)[1]

    assert row["eligible"] is False
    assert row["evidence_scope"] == "registered_fold_0_history_only"
    assert "only for fold 0" in str(row["reason"])


def test_registered_a1_is_labeled_historical_not_current_ablation(
    tmp_path, monkeypatch
) -> None:
    _patch_xgboost_metrics(monkeypatch, {"f1": 0.55})
    _write_historical_diagnostic(tmp_path)

    row = _registered_rows(_config(tmp_path), fold=0)[1]

    assert row["eligible"] is True
    assert row["evidence_scope"] == "historical_fold_0_oof_postprocess_diagnostic"
    assert "independent" not in str(row["evidence_scope"])
    assert row["refined_f1"] == 0.42
    assert row["refined_same_sensitivity"] == 0.7
    assert row["refined_different_sensitivity"] == 0.5
