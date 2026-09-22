from argparse import Namespace

import pandas as pd
import pytest

from bme_eating.dtp_postprocess import event_gate
from bme_eating.fusion_event_ablation import _rescue_variants, run_ablation


def test_nested_rescue_enumerates_predeclared_quantiles_independently_of_component_choice():
    config = {
        "event_rescue": {
            "quantile_candidates": [0.8, 0.85, 0.9],
            "alpha_candidates": [0.5, 1.0],
        }
    }
    variants = list(_rescue_variants(config))
    assert variants[0]["name"] == "original_v4"
    assert {(row["quantile"], row["alpha"]) for row in variants[1:]} == {
        (quantile, alpha) for quantile in (0.8, 0.85, 0.9) for alpha in (0.5, 1.0)
    }


def test_event_gate_maps_unsorted_timestamps_with_subject_equal_score_weights():
    predictions = pd.DataFrame(
        [("one", "day", 6000), ("one", "day", 0), ("one", "day", 3000)],
        columns=["subject_key", "session_id", "timestamp_ms"],
        index=[2, 2, 9],
    )
    events = pd.DataFrame(
        [("one", "day", 3000, 6000, 0.7)],
        columns=["subject_key", "session_id", "start_ms", "end_ms", "score"],
    )
    reference = pd.DataFrame(
        [("one", 0.3), ("one", 0.7), ("two", 0.9)],
        columns=["subject_key", "score"],
    )
    mapped = event_gate(predictions, events, reference)
    assert mapped.event_gate.tolist() == pytest.approx([0.5, 0, 0.5])


def test_ablation_rejects_outer_fold_before_loading_any_prediction(monkeypatch):
    import bme_eating.fusion_event_ablation as module

    monkeypatch.setattr(module, "require_clean_git_worktree", lambda: None)
    monkeypatch.setattr(module, "resolve_roots", lambda config: pytest.fail("outer read"))
    with pytest.raises(ValueError, match="development fold 0"):
        run_ablation(
            Namespace(config="configs/dtp_fusion_event_ablation.yaml", fold=1),
        )
