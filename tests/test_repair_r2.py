from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from helpers import candidate, truth_fixture

from bme_eating.candidates import FrozenSourceCDF, OnlineBudget, candidate_targets
from bme_eating.config import load_config
from bme_eating.endpoint_peaks import CausalOnsetPeaks
from bme_eating.models.statsfusion_joint_event import CompletedScale
from bme_eating.training import candidate_supervision as supervision
from bme_eating.training.contributions import ContributionBank, PositiveGradientBuffer
from bme_eating.training.natural_replay import merge_replay_candidates, release_candidates


@pytest.fixture
def config():
    torch.set_num_threads(2)
    return load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v414_full_data.yaml")


@pytest.mark.parametrize("factor", [1, 5, 20])
def test_proposal_empty_blocks_are_zero_and_fractional_mean_is_identity(factor):
    torch.manual_seed(2026)
    scale = CompletedScale(factor, 4, 0).eval()
    scale.fractional_mean = True
    features = torch.randn(1, factor * 7, 4)
    weights = torch.ones(1, factor * 7)
    positions = torch.arange(factor * 7)[None]
    full, full_valid = scale(features, weights, positions)
    fractional, fractional_valid = scale(features, weights * .2, positions)
    torch.testing.assert_close(full, fractional, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fractional_valid, full_valid * .2)
    weights[:, factor * 2:factor * 3] = 0
    masked, valid = scale(features, weights, positions)
    assert masked[0, factor * 3 - 1] == 0
    assert valid[0, factor * 3 - 1] == 0
    empty, empty_valid = scale(features, weights * 0, positions)
    assert not empty.any() and not empty_valid.any()
    altered = features.clone()
    altered[:, factor * 2:factor * 3] = 10000
    torch.testing.assert_close(scale(altered, weights, positions)[0], masked, rtol=0, atol=0)
    prefix = factor * 5
    torch.testing.assert_close(scale(features[:, :prefix], weights[:, :prefix], positions[:, :prefix])[0],
                               masked[:, :prefix], rtol=0, atol=0)


@pytest.mark.parametrize("invalid", range(7))
def test_onset_confirmation_rejects_every_invalid_neighbor(invalid):
    detector = CausalOnsetPeaks()
    peaks = [detector.observe((index + 1) * 3000, probability, int(index != invalid))
             for index, probability in enumerate([.1, .2, .1, .9, .1, .2, .1])]
    assert peaks[-1] is None


def test_onset_requires_adjacent_grid_and_does_not_fill_eof():
    probabilities = [.1, .2, .1, .9, .1, .2, .1]
    detector = CausalOnsetPeaks()
    for timestamp, probability in zip([3000, 6000, 10000, 12000, 15000, 18000, 21000], probabilities):
        peak = detector.observe(timestamp, probability, 1)
    assert peak is None
    detector = CausalOnsetPeaks()
    for index, probability in enumerate(probabilities):
        peak = detector.observe((index + 1) * 3000, probability, .4)
        if index < 6:
            assert peak is None
    assert peak == (12000, .9, 21000)


def test_retention_drops_group_whose_only_positive_version_is_owned():
    groups = [{"id": "first", "maximum_iou": 1., "bins": [2], "decision": 1,
               "geometries": [(0, 60), (10, 60)], "qualities": [1., 5 / 6]},
              {"id": "second", "maximum_iou": 5 / 6, "bins": [2], "decision": 2,
               "geometries": [(10, 60), (70, 80)], "qualities": [5 / 6, 0.]}]
    selected = supervision.retained_groups(groups)
    assert [group["id"] for group in selected] == ["first"]
    assert groups[1]["qualities"] == [5 / 6, 0.]


def test_mixed_positive_group_keeps_actual_negative_bce(config, monkeypatch):
    versions = [candidate(identity="positive"), candidate(start=90000, end=120000, identity="negative")]
    logits = [torch.zeros(1, requires_grad=True), torch.zeros(1, requires_grad=True)]
    monkeypatch.setattr(supervision, "score_candidates", lambda *args, **kwargs:
        [{"event_logit": value, "quality_logit": value, "candidate_embedding": torch.zeros(1, 64)}
         for value in logits])
    losses, _ = supervision.group_losses(SimpleNamespace(), [{"versions": versions}], truth_fixture(),
        None, None, "cpu", None, config, {"boundary": 0})
    gradients = [float(torch.autograd.grad(tasks["event"], logit, retain_graph=True)[0])
                 for tasks, logit in zip(losses[0][1]["_versions"], logits)]
    assert gradients == [-.5, .5]


def test_owned_near_miss_leaves_background_category(config, monkeypatch):
    near = candidate(start=10000, end=40000, identity="near")
    zero = candidate(start=90000, end=120000, identity="zero")
    owned = {("subject", "session", "truth"): {
        "old": {"descriptor": {"geometries": [(10000, 40000)]}}}}
    monkeypatch.setattr(supervision, "score_candidates", lambda model, candidates, *args, **kwargs:
        [{"event_logit": torch.zeros(1)} for item in candidates])
    groups, report = supervision.sample_groups(torch.nn.Linear(1, 1), [near, zero], truth_fixture(),
        config, np.random.default_rng(2026), owned, set(), None, None, "cpu")
    assert len(groups) == 1 and groups[0]["category"] == "random_background"
    assert candidate_targets(groups[0]["versions"][0], truth_fixture())[1] == 0
    assert report["categories"]["near_miss"]["actual_groups"] == 0


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("quotas", [{}, {"proposal": .3, "short": .2, "expansion": .2}])
def test_budget_geometry_dedup_is_independent_of_cdf(fallback, quotas):
    mapping = FrozenSourceCDF({"proposal:3": [.1, .8]}, ["fit"], fallback=fallback)
    first, later = candidate(), candidate(identity="later")
    later.decision_time_ms += 3000
    later.max_input_time_ms += 3000
    budget = OnlineBudget(20, quotas)
    budget.observe(3600, 1)
    assert budget.select([first], mapping) == [first]
    assert not budget.select([later], mapping)
    assert budget.spent == 1
    invalid = copy.deepcopy(later)
    invalid.raw_score = float("nan")
    with pytest.raises(RuntimeError, match="Nonfinite"):
        budget.select([invalid], mapping)


def test_replay_first_metadata_and_duplicate_release_resume():
    first = candidate(identity="first")
    first.source = "state"
    first.raw_score = .2
    simultaneous = copy.deepcopy(first)
    simultaneous.proposal_id = "simultaneous"
    simultaneous.source = "jitter"
    simultaneous.raw_score = .7
    later = copy.deepcopy(first)
    later.proposal_id = "later"
    later.source = "backtrack"
    later.decision_time_ms += 3000
    later.max_input_time_ms += 3000
    eof = copy.deepcopy(later)
    eof.proposal_id = "eof"
    eof.source = "start_expansion"
    eof.decision_phase = "eof"
    pool, duplicates = merge_replay_candidates([eof, later, simultaneous, first])
    assert len(pool) == 1
    representative = pool[0]
    assert representative.raw_score == .7
    assert set(representative.generation_parameters["source_flags"]) == {"state", "jitter"}
    assert "source_flags" not in first.generation_parameters
    payload = {"candidates": pool, "duplicates": duplicates}
    state = {"cursor": 0, "released": 0, "duplicate_cursor": 0, "duplicates_released": 0}
    assert not release_candidates(payload, state, first.decision_time_ms - 1)
    assert release_candidates(payload, state, first.decision_time_ms) == pool
    assert len(state["last_duplicates"]) == 1
    resumed = copy.deepcopy(state)
    assert not release_candidates(payload, resumed, later.decision_time_ms)
    assert len(resumed["last_duplicates"]) == 2
    assert not release_candidates(payload, resumed, later.decision_time_ms)
    assert resumed["last_duplicates"] == []
    assert resumed["duplicates_released"] == 3


def test_budget_does_not_merge_eof_source_into_regular_metadata():
    first = candidate()
    eof = copy.deepcopy(first)
    eof.proposal_id = "eof"
    eof.decision_phase = "eof"
    eof.source = "jitter"
    budget = OnlineBudget(20, {})
    budget.observe(3600, 1)
    selected = budget.select([eof, first], FrozenSourceCDF({"proposal:3": [.1, .8]}, ["fit"]))
    assert selected == [first]
    assert "source_flags" not in first.generation_parameters


def test_bounded_history_reserves_background_and_balances_events():
    records = [(event * 12 + position, event, position // 4)
               for event in range(20) for position in range(12)]
    records += [(240 + position, -2, 0) for position in range(12)]
    indices, identities, phases = zip(*records)
    selected = supervision.bounded_history(indices, identities, phases, 128)
    assert len(selected) == 128 and selected == sorted(selected)
    assert len({index for index, _, _ in selected}) == 128
    assert sum(identity == -2 for _, identity, _ in selected) == 12
    assert {identity for _, identity, _ in selected if identity >= 0} == set(range(20))
    natural = records[:96] + records[-12:]
    assert supervision.bounded_history(*zip(*natural), 128) == natural
    with pytest.raises(RuntimeError, match="Conflicting"):
        supervision.bounded_history([1, 1], [0, 1], [0, 0], 128)


def test_negative_only_group_is_removed_from_gradient_and_event_count():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1)
    storage = {}
    buffer = PositiveGradientBuffer(storage)
    def add(identity, geometries, qualities, coefficients):
        group = {"event_key": "event", "descriptor": {"id": identity,
            "maximum_iou": max(qualities), "bins": [2], "decision": 1,
            "geometries": geometries, "qualities": qualities}}
        atoms = [{"geometries": [geometry], "tasks": {
            "classification": model.weight.sum() * coefficient}}
            for geometry, coefficient in zip(geometries, coefficients)]
        return buffer.add(group, {"_atoms": atoms}, model)
    add("old", [(10, 60), (70, 80)], [5 / 6, 0], [1000, 100])
    result = add("new", [(0, 60), (10, 60)], [1, 5 / 6], [2, 4])
    assert result["removed"] == ["old"]
    assert result["dropped"][0]["reason"] == "no_positive_after_dedup"
    bank = ContributionBank({})
    bank.finish_session(storage)
    bank.commit(model)
    assert float(model.weight.grad) == 3
    empty = {}
    group = {"event_key": "none", "descriptor": {"id": "negative", "maximum_iou": .25,
        "bins": [], "decision": 1, "geometries": [(0, 1)], "qualities": [.25]}}
    assert PositiveGradientBuffer(empty).add(group, {}, model)["added"] is None
    assert not empty
