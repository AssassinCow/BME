from __future__ import annotations

import json

import pytest

from bme_eating.hierarchical_gates import select_fold0_mode, write_freeze_manifest


def _selection(root, run_name: str, values: dict[str, float]) -> None:
    path = root / "experiments" / run_name / "fold_0" / "selection"
    path.mkdir(parents=True)
    (path / "selected_pipeline.json").write_text(
        json.dumps(values), encoding="utf-8"
    )


def test_state_xgb_mode_requires_all_three_fold0_oof_gates(tmp_path) -> None:
    verifier = {
        "refined_f1": 0.50,
        "refined_false_positives_per_observed_hour": 0.10,
        "refined_different_sensitivity": 0.40,
        "xgb_mode": "verifier_only",
    }
    state_xgb = {
        "refined_f1": 0.512,
        "refined_false_positives_per_observed_hour": 0.104,
        "refined_different_sensitivity": 0.39,
        "xgb_mode": "state_and_verifier",
    }
    _selection(tmp_path, "verifier", verifier)
    _selection(tmp_path, "state", state_xgb)
    result = select_fold0_mode(tmp_path, "verifier", "state")
    assert result["state_and_verifier_gate_passed"]
    assert result["selected_run"] == "state"

    state_xgb["refined_false_positives_per_observed_hour"] = 0.106
    _selection(tmp_path, "state_bad", state_xgb)
    rejected = select_fold0_mode(tmp_path, "verifier", "state_bad")
    assert not rejected["state_and_verifier_gate_passed"]
    assert rejected["selected_run"] == "verifier"


def test_embedding_ablation_must_preserve_selected_xgb_mode(tmp_path) -> None:
    verifier = {
        "refined_f1": 0.50,
        "refined_false_positives_per_observed_hour": 0.10,
        "refined_different_sensitivity": 0.40,
        "xgb_mode": "verifier_only",
    }
    state_xgb = {
        "refined_f1": 0.52,
        "refined_false_positives_per_observed_hour": 0.10,
        "refined_different_sensitivity": 0.40,
        "xgb_mode": "state_and_verifier",
    }
    no_embedding = {**verifier, "refined_f1": 0.49}
    _selection(tmp_path, "verifier", verifier)
    _selection(tmp_path, "state", state_xgb)
    _selection(tmp_path, "no_embedding", no_embedding)

    with pytest.raises(RuntimeError, match="same XGBoost mode"):
        select_fold0_mode(
            tmp_path,
            "verifier",
            "state",
            no_embedding_run="no_embedding",
        )


def test_freeze_rejects_decision_from_another_run(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="different hierarchical run"):
        write_freeze_manifest(
            tmp_path,
            "selected-run",
            {
                "phase": "folds_0_1_development",
                "run_name": "other-run",
                "passed": True,
            },
        )
