import pandas as pd

from bme_eating.data.splits import create_subject_folds


def test_subjects_are_assigned_to_one_fold(tmp_path):
    rows = []
    for subject_index in range(10):
        for event_index in range(3):
            rows.append(
                {
                    "subject_key": f"s{subject_index}",
                    "event_id": f"e{subject_index}_{event_index}",
                    "valid_duration": True,
                    "coverage": "full",
                    "hand_relation": "same" if event_index % 2 else "different",
                }
            )
    folds = create_subject_folds(pd.DataFrame(rows), 5, 2026, tmp_path / "folds.json")
    assert set(folds) == {f"s{index}" for index in range(10)}
    assert all(0 <= fold < 5 for fold in folds.values())


def test_background_subjects_without_full_events_are_assigned(tmp_path):
    rows = [
        {
            "subject_key": "s0",
            "event_id": "e0",
            "valid_duration": True,
            "coverage": "full",
            "hand_relation": "same",
        }
    ]
    folds = create_subject_folds(
        pd.DataFrame(rows),
        2,
        2026,
        tmp_path / "folds.json",
        subject_keys={"s0", "background_only"},
    )
    assert set(folds) == {"s0", "background_only"}

