from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from helpers import candidate, truth_fixture

from bme_eating.config import load_config
from bme_eating.evaluation import NonnegativePlatt, commit_candidates, metrics_with_exposure
from bme_eating.journal import CheckedJournal
from bme_eating.operating_points import CachedSearch
from bme_eating.proposal_stream import make_stream


@pytest.fixture
def config():
    value = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v414_full_data.yaml")
    value["training"]["device"] = "cpu"
    torch.set_num_threads(2)
    return value


@pytest.mark.parametrize("onset", [0.0, .29, .8])
def test_learned_onset_and_fallback_prefix(config, onset):
    def replay(length):
        stream = make_stream("subject", "session", "model", config["decoder"])
        output = []
        for index in range(1, length + 1):
            output.extend(stream.observe(index * 3000, {"state": .9 if 8 <= index <= 25 else 0},
                                         1, onset_probability=onset if index == 9 else 0))
        return output
    short, long = replay(70), replay(100)
    assert short and [item.to_dict() for item in short] == [item.to_dict() for item in long[:len(short)]]
    assert len(short) <= config["decoder"]["maximum_versions_per_parent"]
    assert len({(item.start_ms, item.end_ms) for item in short}) == len(short)
    assert any(item.source == "learned_onset" for item in short) == (onset >= .30)
    if onset >= .30:
        assert {item.start_ms for item in short if item.source == "learned_onset"} == {27_000}


def test_checked_journal_truncates_uncommitted_tail_and_rejects_corruption(tmp_path):
    path = tmp_path / "samples.jsonl"
    journal = CheckedJournal(path)
    journal.append([{"chunk": 1}])
    offset = journal.offset
    journal.append([{"chunk": 2}])
    resumed = CheckedJournal(path, offset)
    assert list(resumed.records()) == [{"chunk": 1}]
    record = json.loads(path.read_text())
    record["payload"]["chunk"] = 9
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(RuntimeError, match="checksum"):
        CheckedJournal(path, path.stat().st_size)


def test_cached_matching_and_stream_nms_identical_to_reference():
    rng = np.random.default_rng(2026)
    truth = truth_fixture()
    ignore = truth.iloc[:0]
    pool = [candidate(start=int(rng.integers(0, 70_000)), end=90_000,
                      identity=str(index), parent=str(index // 3)) for index in range(24)]
    for item in pool:
        item.event_logit = float(rng.normal())
        item.predicted_quality = float(rng.random())
    windows = pd.DataFrame({"valid": np.ones(50)})
    calibrator = NonnegativePlatt(1, -.1, True)
    cached = CachedSearch(pool, truth, ignore, windows, calibrator)
    for acceptance in (.01, .25, .5, .75, .99):
        for nms in (.3, .5, .7):
            prediction = commit_candidates(copy.deepcopy(pool), calibrator, acceptance, nms)
            expected, _ = metrics_with_exposure(truth, prediction, ignore, windows)
            actual = cached.evaluate(acceptance, nms)
            for name, value in expected.items():
                if value is None:
                    assert actual[name] is None
                else:
                    assert actual[name] == pytest.approx(value, nan_ok=True, abs=1e-12)


def test_no_onset_ablation_keeps_legacy_compatibility_pool(config):
    decoder = {**config["decoder"], "ablation": "no_onset"}
    stream = make_stream("subject", "session", "model", decoder)
    emitted = []
    for index in range(1, 100):
        emitted.extend(stream.observe(index * 3000, {"state": .9 if index <= 40 else 0}, 1,
                                      onset_probability=.9 if index == 3 else 0))
    assert len(emitted) > 3
    assert not any(item.source == "learned_onset" for item in emitted)


