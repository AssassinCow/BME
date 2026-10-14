from __future__ import annotations

import copy
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from helpers import candidate, truth_fixture

from bme_eating.candidates import (
    FrozenSourceCDF,
    NoSourceCandidatesError,
    decision_key,
    geometry_key,
)
from bme_eating.config import load_config
from bme_eating.diagnostics import endpoint_diagnostics
from bme_eating.proposal_stream import make_stream
from bme_eating.streaming import JointChunkDetector
from bme_eating.training.candidate_supervision import (
    ContributionBank,
    PositiveGradientBuffer,
    heatmap_focal_loss,
    historical_contrastive,
    retained_groups,
    sample_groups,
)
from bme_eating.training.endpoint_balance import epoch_balance


@pytest.fixture
def config():
    result = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v414_full_data.yaml")
    result["training"]["device"] = "cpu"
    torch.set_num_threads(2)
    return result


def test_endpoint_epoch_weights_and_rng(config):
    counts = {("first", "positive"): (1, 20), ("second", "background"): (0, 40)}
    dataset = SimpleNamespace(session_groups=counts, endpoint_counts=lambda key, name: counts[key])
    before = torch.get_rng_state().clone()
    balance = epoch_balance(dataset, 3, {"first": .5, "second": 2}, config)
    assert torch.equal(before, torch.get_rng_state())
    assert balance["scales"]["onset"]["peak_scale"] == 5
    assert balance["scales"]["offset"]["background_scale"] == 1
    assert balance == epoch_balance(dataset, 3, {"first": .5, "second": 2}, config)
    dataset.endpoint_counts = lambda key, name: (0, 0)
    empty = epoch_balance(dataset, 3, {"first": .5, "second": 2}, config)
    assert empty["scales"]["onset"]["peak_scale"] == 0
    assert empty["scales"]["onset"]["background_scale"] == 0


@pytest.mark.parametrize("probability", [.10, .247, .35])
def test_endpoint_gradient_independent_of_positive_session_fraction(probability):
    gradients = []
    for sessions in (5, 40):
        value = torch.tensor(float(np.log(probability / (1 - probability))), requires_grad=True)
        positive = heatmap_focal_loss(value.expand(2), torch.tensor([1., 0.]), torch.ones(2),
            torch.tensor(1), torch.tensor(1), peak_scale=sessions)
        background = heatmap_focal_loss(value[None], torch.zeros(1), torch.ones(1),
            torch.tensor(0), torch.tensor(1), peak_scale=sessions)
        loss = (positive + (sessions - 1) * background) / sessions
        loss.backward()
        gradients.append(float(value.grad))
    assert gradients[0] == pytest.approx(gradients[1], abs=1e-6)
    if probability == .247:
        assert abs(gradients[0]) < .001


def test_atomic_positive_trim_preserves_unique_version_and_removes_pair():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1)
    storage = {}
    buffer = PositiveGradientBuffer(storage)
    def add(identity, geometries, qualities, coefficients, ranking):
        group = {"event_key": "event", "descriptor": {"id": identity,
            "maximum_iou": max(qualities), "bins": [min(2, int(np.ceil(value * 4)) - 2)
                for value in qualities], "decision": 10, "geometries": geometries, "qualities": qualities}}
        atoms = [{"geometries": [geometry], "tasks": {"classification": model.weight.sum() * coefficient}}
                 for geometry, coefficient in zip(geometries, coefficients)]
        atoms.append({"geometries": geometries, "tasks": {"quality_ranking": model.weight.sum() * ranking}})
        buffer.add(group, {"_atoms": atoms}, model)
    add("first", [(0, 60), (10, 60)], [1., 5 / 6], [100, 2], 5)
    add("second", [(10, 60), (43, 60)], [5 / 6, 17 / 60], [1000, 4], 7)
    assert storage["event"]["second"]["descriptor"]["geometries"] == [(43, 60)]
    assert "quality_ranking" not in storage["event"]["second"]["task_gradients"]
    bank = ContributionBank({})
    bank.finish_session(storage)
    bank.commit(model)
    assert float(model.weight.grad) == pytest.approx(32.5)
    assert not storage


def test_negative_auxiliary_tasks_reach_commit():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1)
    state = {}
    bank = ContributionBank(state)
    bank.negative({"classification": model.weight.sum() * 8,
                   "quality_ranking": model.weight.sum() * 3,
                   "boundary": model.weight.sum() * 2}, model)
    bank.finish_session({})
    bank.negative({"classification": model.weight.sum() * 10}, model)
    bank.finish_session({})
    bank.commit(model)
    assert float(model.weight.grad) == pytest.approx(14)
    assert state["last_update_contributions"] == {"negative": 2, "boundary": 1, "quality_ranking": 1}
    assert sum(state["last_update_candidate_objective"].values()) == pytest.approx(14)


def test_negative_geometry_collision_and_lock(config, monkeypatch):
    import bme_eating.training.candidate_supervision as supervision
    first = candidate(start=258_000, end=273_000, identity="natural", parent="natural_parent")
    second = copy.deepcopy(first)
    second.proposal_id, second.proposal_family_id, second.source = "random", "random_family", "random_background"
    monkeypatch.setattr(supervision, "score_candidates", lambda model, pool, *args, **kwargs:
        [{"event_logit": torch.tensor([0.])} for item in pool])
    model = torch.nn.Linear(1, 1)
    arguments = (model, [first, second], truth_fixture(), config, random.Random(2026),
                 {}, set(), torch.arange(1), torch.zeros(1, 201), torch.device("cpu"))
    groups, report = sample_groups(*arguments)
    assert len(groups) == 1
    assert report["mining_head_candidates"] == 1
    assert groups[0]["category"] == "random_background"
    groups, _ = sample_groups(*arguments, locked_geometries={geometry_key(first)})
    assert groups == []


def test_fallback_geometry_independent_of_run_identity(config):
    def replay(identity):
        stream = make_stream("subject", "session", identity, config["decoder"])
        output = []
        for index in range(1, 70):
            output.extend(stream.observe(index * 3000, {"state": .9 if 8 <= index <= 25 else 0},
                                         1, onset_probability=0))
        return [(item.start_ms, item.end_ms, item.source) for item in output]
    assert replay("run_a") == replay("run_b")
    assert 0 < len(replay("run_a")) <= 3


def test_constant_onset_diagnostic_has_no_usable_peaks():
    windows = pd.DataFrame({"subject_key": "subject", "session_id": "session",
        "timestamp_ms": np.arange(1, 51) * 3000, "valid": 1.,
        "onset_probability": .4, "offset_probability": 0.})
    truth = truth_fixture()
    result = endpoint_diagnostics(windows, truth, truth.iloc[:0], [])
    assert result["onset"]["endpoint_hits"] == 1
    assert result["onset"]["usable_endpoint_hits"] == 0
    assert result["onset"]["usable_peaks"] == 0


def test_end_of_chunk_decision_is_returned_immediately(config, monkeypatch):
    import bme_eating.candidate_scoring as scoring
    model = torch.nn.Linear(1, 1)
    mapping = FrozenSourceCDF({"state:3": [0, 1]}, ["fit"], fallback=True)
    point = {"setting": "high_0.1_low_0.9_gap_15", "budget": 20, "acceptance": .5,
             "nms": .3, "calibration": {"slope": 1, "intercept": 20, "qualified": True},
             "boundary_enabled": False}
    detector = JointChunkDetector(model, config, point, mapping, "model", {}, "subject", "session")
    item = candidate(start=3000, end=6000)
    item.decision_time_ms = item.max_input_time_ms = 66_000
    detector.stream = SimpleNamespace(observe=lambda timestamp, *args, **kwargs:
        [copy.deepcopy(item)] if timestamp == 66_000 else [], finish=list)
    monkeypatch.setattr(scoring, "score_candidates", lambda *args, **kwargs: [])
    times = torch.arange(1, 23) * 3000
    features = torch.zeros(22, 201)
    features[:, 128] = 1
    output = detector.push_features(times, features, final=False)
    assert len(output) == 1
    assert output[0]["emitted_at_ms"] == 66_000
    assert output[0]["emitted_at_ms"] - output[0]["end_ms"] == 60_000
    assert not detector.push_features(torch.empty(0, dtype=torch.int64), torch.empty(0, 201), final=True)


def test_empty_cdf_has_specific_error_and_invalid_subject_still_fails():
    with pytest.raises(NoSourceCandidatesError):
        FrozenSourceCDF.fit([], {"fit"})
    with pytest.raises(RuntimeError, match="leakage"):
        FrozenSourceCDF.fit([candidate()], {"other"})


def test_historical_background_enables_long_event_contrast(config):
    model = SimpleNamespace(contrastive_projection=torch.nn.Identity())
    current = torch.randn(6, 64, requires_grad=True)
    history = torch.randn(90, 201)
    history[:, 128] = 1
    times = torch.arange(1, 91) * 3000
    truth = pd.DataFrame([{"subject_key": "subject", "session_id": "session", "event_id": "event",
                          "start_ms": 60_000, "end_ms": 600_000}])
    loss, pairs = historical_contrastive(model, current, torch.zeros(6, dtype=torch.long),
        torch.zeros(6, dtype=torch.long), times, history, truth, "subject", "session", config,
        ignore=truth.iloc[:0])
    assert pairs > 0 and float(loss.detach()) > 0
    loss.backward()
    assert current.grad.abs().sum() > 0
    assert not history.requires_grad


def test_retention_does_not_discard_group_with_one_unique_geometry():
    groups = [{"id": "a", "maximum_iou": 1., "bins": [2], "decision": 1,
               "geometries": [(0, 60), (10, 60)], "qualities": [1., 5 / 6]},
              {"id": "b", "maximum_iou": 5 / 6, "bins": [0, 2], "decision": 2,
               "geometries": [(10, 60), (43, 60)], "qualities": [5 / 6, 17 / 60]}]
    selected = retained_groups(groups)
    assert len(selected) == 2
    assert selected[1]["geometries"] == [(43, 60)]
    assert groups[1]["geometries"] == [(10, 60), (43, 60)]


@pytest.mark.parametrize("failure", [KeyboardInterrupt, torch.OutOfMemoryError])
def test_empty_source_validation_resume_counts_once(config, tmp_path, monkeypatch, failure):
    from test_training import test_replay_and_contribution_chunk_resume
    test_replay_and_contribution_chunk_resume(config, tmp_path, monkeypatch, failure,
        empty_source_validation=True)


def test_eof_phase_preserves_prior_commit_and_cached_search_matches():
    from bme_eating.evaluation import NonnegativePlatt, commit_candidates, metrics_with_exposure
    from bme_eating.operating_points import CachedSearch
    first = candidate(start=3000, end=30_000, identity="regular", parent="regular")
    last = candidate(start=6000, end=30_000, identity="eof", parent="eof")
    first.decision_time_ms = first.max_input_time_ms = 60_000
    last.decision_time_ms = last.max_input_time_ms = 60_000
    first.event_probability = first.rank_score = .7
    last.event_probability = last.rank_score = .9
    first.event_logit, last.event_logit = float(np.log(.7 / .3)), float(np.log(.9 / .1))
    last.decision_phase = "eof"
    assert decision_key(first) < decision_key(last)
    calibration = NonnegativePlatt(1, 0, True)
    emitted = commit_candidates([last, first], calibration, .5, .3)
    assert [item.proposal_id for item in emitted] == [first.proposal_id]
    truth = truth_fixture().assign(start_ms=3000, end_ms=30_000)
    windows = pd.DataFrame({"subject_key": "subject", "session_id": "session",
        "timestamp_ms": np.arange(1, 21) * 3000, "valid": 1.})
    reference, _ = metrics_with_exposure(truth, emitted, truth.iloc[:0], windows)
    cached = CachedSearch([last, first], truth, truth.iloc[:0], windows, calibration).evaluate(.5, .3)
    for name in ("f1", "true_positive", "false_positive", "false_negative", "start_mae_seconds"):
        assert cached[name] == reference[name]


def test_checkpoint_protocol_and_atomic_state_rejected(config):
    from bme_eating.protocol import CHECKPOINT_STATE_VERSION, checkpoint_qualified, protocol_for
    payload = {"protocol": protocol_for(config), "state_version": CHECKPOINT_STATE_VERSION, "epoch_complete": True,
        "promotion_eligible": True, "epoch": 8, "clipping_fraction": 0, "identity": "model",
        "training_subjects": ["fit"], "model": {"parameter": torch.tensor(1.)}}
    assert checkpoint_qualified(payload, "model", {"fit"}, 8)
    assert not checkpoint_qualified({**payload, "protocol": "statsfusion-joint-event-v5"}, "model", {"fit"}, 8)
    assert not checkpoint_qualified({**payload, "state_version": 1}, "model", {"fit"}, 8)


def test_legal_zero_loss_counts_and_nonfinite_atomic_gradient_rejected():
    from bme_eating.training.contributions import atomic_gradients
    model = torch.nn.Linear(1, 1, bias=False)
    bank = ContributionBank({})
    bank.negative({"quality_ranking": model.weight.sum() * 0}, model)
    bank.finish_session({})
    bank.commit(model)
    assert bank.state["last_update_contributions"] == {"quality_ranking": 1}
    class InvalidBackward(torch.autograd.Function):
        @staticmethod
        def forward(context, value):
            return value.sum() * 0 + 1

        @staticmethod
        def backward(context, gradient):
            return gradient.new_full((1, 1), float("nan"))
    with pytest.raises(RuntimeError, match="Nonfinite"):
        atomic_gradients([InvalidBackward.apply(model.weight)], model)
