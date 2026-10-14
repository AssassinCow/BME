from __future__ import annotations

import copy
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch
from helpers import candidate, feature_fixture, truth_fixture

from bme_eating.config import load_config
from bme_eating.proposal_stream import make_stream
from bme_eating.protocol import PROFILE, validate_config
from bme_eating.training import candidate_supervision as supervision


@pytest.fixture
def config():
    torch.set_num_threads(2)
    return load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v414_full_data.yaml")


def candidates(decoder, *, first=12, last=74, peaks=(45000,), steps=180, identity="model"):
    stream = make_stream("subject", "session", identity, copy.deepcopy(decoder))
    output = []
    for position in range(1, steps + 1):
        output.extend(stream.observe(position * 3000,
            {"state": .9 if first <= position <= last else 0}, 1,
            onset_probability=.8 if position * 3000 in peaks else 0))
    output.extend(stream.finish())
    return output


def geometries(pool):
    return {(item.start_ms, item.end_ms) for item in pool}


@pytest.mark.parametrize("peaks", [(45000,), (45000, 63000), (33000,), (33000, 45000)])
def test_all_original_fallback_geometries_survive_learning(config, peaks):
    baseline = candidates(config["decoder"], peaks=())
    actual = candidates(config["decoder"], peaks=peaks)
    assert geometries(baseline) <= geometries(actual)
    assert len(geometries(actual)) == len(actual) <= 4
    assert sum(item.source == "learned_onset" for item in actual) <= 1
    if (45000,) == peaks:
        assert len(actual) == 4
        assert (63000, 210000) in geometries(actual)
    extended = candidates(config["decoder"], peaks=peaks, steps=240)
    assert [item.to_dict() for item in actual] == [item.to_dict() for item in extended[:len(actual)]]
    other = candidates(config["decoder"], peaks=peaks, identity="other")
    assert geometries(other) == geometries(actual)


def test_all_decoder_fallback_sets_are_preserved(config):
    for high, ratio, gap in product(config["search"]["high_thresholds"],
                                  config["search"]["low_ratios"], config["search"]["gap_seconds"]):
        decoder = {**config["decoder"], "high_threshold": high, "low_ratio": ratio,
                   "gap_merge_seconds": gap}
        for first, last in ((2, 15), (8, 25), (12, 74), (38, 120)):
            baseline = candidates(decoder, first=first, last=last, peaks=())
            actual = candidates(decoder, first=first, last=last, peaks=(first * 3000 + 12000,))
            assert geometries(baseline) <= geometries(actual)
            assert len(actual) <= 4


def test_positive_sampling_retains_four_versions(config, monkeypatch):
    versions = [candidate(end=90000 + index * 3000, identity=str(index)) for index in range(4)]
    monkeypatch.setattr(supervision, "score_candidates", lambda *args, **kwargs: [])
    groups, _ = supervision.sample_groups(torch.nn.Linear(1, 1), versions, truth_fixture(),
        config, np.random.default_rng(2026), {}, set(), *feature_fixture(), "cpu")
    assert len(groups) == 1
    assert len(groups[0]["versions"]) == 4


def test_negative_groups_keep_four_versions_with_96_prescoring_cap(config, monkeypatch):
    versions = [candidate(start=120000 + family * 300000,
        end=180000 + family * 300000 + version * 3000,
        identity=f"{family}:{version}", family=str(family))
        for family in range(40) for version in range(4)]
    scored = []

    def score(model, pool, *args, **kwargs):
        scored.extend(pool)
        return [{"event_logit": torch.tensor([.2])} for _ in pool]

    monkeypatch.setattr(supervision, "score_candidates", score)
    groups, report = supervision.sample_groups(torch.nn.Linear(1, 1), versions, truth_fixture(),
        config, np.random.default_rng(2026), {}, set(), *feature_fixture(), "cpu")
    assert len(scored) == report["mining_head_candidates"] == 96
    assert report["mining_families"] == 32
    assert all(len(group["versions"]) == 4 for group in groups)
    assert len({group["family"] for group in groups}) == len(groups)
    assert sum(group["category"] == "random_background" for group in groups) == 1


def test_four_version_protocol_rejects_three_version_and_r3_config(config):
    assert PROFILE.protocol == "statsfusion-joint-event-v9"
    assert PROFILE.repair_revision == "v414-r4"
    validate_config(config)
    for section, key, value in (("training", "versions_per_group", 3),
                               ("training", "mining_versions_per_family", 4),
                               ("decoder", "candidate_version_policy", "coarse_compatibility_learned_v1"),
                               ("experiment", "protocol_version", "statsfusion-joint-event-v8")):
        invalid = copy.deepcopy(config)
        invalid[section][key] = value
        with pytest.raises(ValueError):
            validate_config(invalid)
