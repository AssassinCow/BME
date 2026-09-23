from __future__ import annotations

import json

import pandas as pd

from bme_eating.training.hierarchical_trainer import load_hierarchical_inputs


def _write_inputs(root) -> dict[str, object]:
    (root / "indices").mkdir(parents=True)
    (root / "features").mkdir()
    anchors = pd.DataFrame(
        {
            "subject_key": ["train", "outer"],
            "session_id": ["a", "b"],
            "timestamp_ms": [0, 0],
        }
    )
    anchors.to_parquet(root / "indices" / "anchors.parquet", index=False)
    pd.DataFrame(
        {"subject_key": ["train", "outer"], "segment_id": ["a", "b"]}
    ).to_parquet(root / "indices" / "segments.parquet", index=False)
    pd.DataFrame(
        {
            "subject_key": ["train", "outer"],
            "event_id": ["train-event", "outer-event"],
            "start_ms": [0, 0],
            "end_ms": [1000, 1000],
        }
    ).to_parquet(root / "indices" / "events.parquet", index=False)
    (root / "indices" / "subject_folds.json").write_text(
        json.dumps({"train": 1, "outer": 0}), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "subject_key": ["train", "outer"],
            "session_id": ["a", "b"],
            "timestamp_ms": [0, 0],
            "stable": [1.0, 2.0],
        }
    ).to_parquet(root / "features" / "baseline.parquet", index=False)
    return {
        "data": {"subject_folds": 2},
        "features": {"artifact_name": "baseline"},
        "hierarchical": {"stable_feature_columns": ["stable"]},
    }


def test_training_input_loader_physically_excludes_outer_events(tmp_path) -> None:
    config = _write_inputs(tmp_path)

    inputs = load_hierarchical_inputs(
        config, tmp_path, fold=0, event_role="outer_train"
    )

    assert set(inputs.events["subject_key"]) == {"train"}


def test_evaluation_input_loader_reads_only_outer_events(tmp_path) -> None:
    config = _write_inputs(tmp_path)

    inputs = load_hierarchical_inputs(
        config, tmp_path, fold=0, event_role="outer_test"
    )

    assert set(inputs.events["subject_key"]) == {"outer"}
