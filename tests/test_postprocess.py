import pandas as pd

from bme_eating.postprocess import probabilities_to_events


def test_hysteresis_creates_one_event():
    frame = pd.DataFrame(
        {
            "subject_key": ["s"] * 8,
            "segment_id": ["x"] * 8,
            "timestamp_ms": [3000 * index for index in range(1, 9)],
            "state_probability": [0.0, 0.1, 0.8, 0.9, 0.8, 0.2, 0.1, 0.0],
            "start_probability": [0.0, 0.1, 0.9, 0.2, 0.1, 0.0, 0.0, 0.0],
            "end_probability": [0.0, 0.0, 0.0, 0.1, 0.2, 0.9, 0.1, 0.0],
        }
    )
    events = probabilities_to_events(
        frame,
        ema_half_life_seconds=0.1,
        high_threshold=0.6,
        low_threshold=0.3,
        minimum_event_seconds=3,
        merge_gap_seconds=0,
        boundary_lookback_seconds=6,
    )
    assert len(events) == 1
    assert events.iloc[0].end_ms > events.iloc[0].start_ms

