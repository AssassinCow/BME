from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bme_eating.config import load_config
from bme_eating.joint_evaluation import NonnegativePlatt, commit_candidates, metrics_with_exposure
from bme_eating.joint_protocol import config_identity, initialize_run, validate_config
from bme_eating.joint_stream import FrozenSourceCDF, JointCandidate, OnlineBudget, candidate_targets
from bme_eating.joint_v411_features import ExactROICache, FeatureBlocks
from bme_eating.joint_v411_journal import CheckedJournal
from bme_eating.joint_v411_scoring import boundary_loss, refine, score_candidates
from bme_eating.joint_v411_search import CachedSearch
from bme_eating.joint_v411_stream import make_stream
from bme_eating.joint_v411_training import PositiveGradientBuffer, sample_groups
from bme_eating.models.statsfusion_joint_event_v411 import build_joint_model, indexed_roi


@pytest.fixture
def config():
    value = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v411_learned_boundary.yaml")
    value["training"]["device"] = "cpu"
    torch.set_num_threads(2)
    return value


def candidate(start=30_000, end=90_000, identity="one", parent="parent", family="family"):
    return JointCandidate("subject", "session", start, end, end + 30_000, "proposal", 3, .8,
        30_000, 90_000, family, identity, parent, "parent_hash", "model",
        {"v411": True, "parent_source": "proposal", "parent_scale": 3}, end + 30_000,
        score_formula="calibrated_event_probability_v411")


def feature_fixture(steps=50, gradient=False):
    torch.manual_seed(2026)
    values = torch.randn(steps, 137)
    values[:, 128:] = .8
    return torch.arange(1, steps + 1) * 3000, values.requires_grad_(gradient)


def truth_fixture():
    return pd.DataFrame({"subject_key": ["subject"], "session_id": ["session"],
                         "event_id": ["truth"], "start_ms": [30_000], "end_ms": [90_000],
                         "hand_relation": ["unknown"]})


def test_registered_profile_and_resume_identity(config, tmp_path, monkeypatch):
    validate_config(config)
    monkeypatch.setattr("bme_eating.joint_protocol.runtime_identity", lambda: {"runtime": "fixed"})
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts" / "runner.py").write_text("original")
    run = initialize_run(config, source, tmp_path / "output", "run", {}, fresh=True,
                         skip_s0=True, historical_reference="reference")
    original = config_identity(config)
    changed = copy.deepcopy(config)
    changed["training"]["num_workers"] = 3
    assert config_identity(changed) == original
    assert initialize_run(changed, source, tmp_path / "output", "run", {}, fresh=False,
                          skip_s0=True, historical_reference="reference") == run
    (source / "scripts" / "export.py").write_text("changed")
    with pytest.raises(RuntimeError, match="identity changed"):
        initialize_run(changed, source, tmp_path / "output", "run", {}, fresh=False,
                       skip_s0=True, historical_reference="reference")


def test_v411_version_validation_reports_registered_version(config):
    invalid = copy.deepcopy(config)
    invalid["experiment"]["code_version"] = "v4.10"
    with pytest.raises(ValueError, match="Joint version must be v4.11"):
        validate_config(invalid)


@pytest.mark.parametrize("setting,value", [("group_quotas", {"positive": 8}),
    ("candidates_per_chunk", 4), ("versions_per_group", 0), ("sample_composition", {})])
def test_v411_rejects_unregistered_or_unused_sampling_configuration(config, setting, value):
    invalid = copy.deepcopy(config)
    invalid["training"][setting] = value
    with pytest.raises(ValueError, match="group quotas"):
        validate_config(invalid)


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
    assert len(short) <= 3
    assert len({(item.start_ms, item.end_ms) for item in short}) == len(short)
    assert any(item.source == "learned_onset" for item in short) == (onset >= .30)
    if onset >= .30:
        assert {item.start_ms for item in short if item.source == "learned_onset"} == {27_000}


def test_legacy_candidates_keep_probability_quality_scoring(config):
    from bme_eating.joint_stream import StreamingProposals
    stream = StreamingProposals("subject", "session", "model", config["decoder"])
    stream.observe(3000, {"state": .8}, 1)
    pool = stream.finish()
    chosen = commit_candidates(pool, NonnegativePlatt(1, 0, True), 0, .5)
    assert all(item.score_formula == "event_probability_times_quality_v410" for item in pool)
    assert all(item.rank_score == item.event_probability * item.predicted_quality for item in chosen)


@pytest.mark.parametrize("chunk_size", [20, 40])
@pytest.mark.parametrize("empty_final", [False, True])
def test_eof_pool_shares_one_budget_decision(config, monkeypatch, chunk_size, empty_final):
    from bme_eating.joint_pipeline import JointChunkDetector
    model = build_joint_model(config)
    point = {"setting": "high_0.1_gap_15", "budget": 30, "nms": .5, "acceptance": .1,
             "boundary_enabled": False, "calibration": NonnegativePlatt(1, 0, True).to_json()}
    detector = JointChunkDetector(model, config, point,
        FrozenSourceCDF({"proposal:3": [.1, .9]}, ["fit"]), "model", {}, "subject", "session")
    early = candidate(identity="early")
    early.raw_score, early.event_logit, early.predicted_quality = .1, 10, .9
    late = candidate(start=60_000, end=120_000, identity="late", parent="late_parent")
    late.coarse_start_ms, late.coarse_end_ms = late.start_ms, late.end_ms
    late.decision_time_ms = late.max_input_time_ms = 120_000
    late.raw_score, late.event_logit, late.predicted_quality = .9, 10, .9

    class FinalStream:
        def observe(self, timestamp, *arguments, **keywords):
            return [copy.deepcopy(early)] if timestamp == 120_000 else []

        def finish(self):
            return [copy.deepcopy(late)]

    detector.stream = FinalStream()
    monkeypatch.setattr("bme_eating.joint_v411_scoring.score_candidates", lambda *args, **kwargs: [])
    times, features = feature_fixture(steps=40)
    output = []
    for first in range(0, len(times), chunk_size):
        output.extend(detector.push_features(times[first:first + chunk_size],
            features[first:first + chunk_size], final=not empty_final and first + chunk_size == len(times)))
    if empty_final:
        output.extend(detector.push_features(torch.empty(0, dtype=torch.int64), torch.empty(0, 137), final=True))
    assert [item["proposal_id"] for item in output] == ["late"]
    assert detector.budget.spent == 1


def test_missing_short_eof_retains_coarse(config):
    stream = make_stream("subject", "session", "model", config["decoder"])
    stream.observe(3000, {"state": .8}, 1, onset_probability=.8)
    stream.observe(6000, {"state": .8}, 0)
    pool = stream.finish()
    assert any(item.start_ms == item.coarse_start_ms and item.end_ms == item.coarse_end_ms for item in pool)
    assert all(item.decision_time_ms - item.end_ms <= 60_000 for item in pool)


@pytest.mark.parametrize("true_start", [0, 30_000])
def test_boundary_supervision_uses_only_observed_endpoint_support(config, true_start):
    model = build_joint_model(config)
    times, features = feature_fixture(steps=40, gradient=True)
    item = candidate()
    score = score_candidates(model, [item], times, features, "cpu")[0]
    loss = boundary_loss(model, item, (1.0, .5, "event", true_start, 300_000), times,
        features, "cpu", {"start_range_seconds": 60, "end_range_seconds": 30},
        score["candidate_embedding"][0])
    assert bool(loss.detach() > 0) == (true_start == 30_000)
    if true_start == 30_000:
        gradient = torch.autograd.grad(loss, model.boundary_head.head[0].weight)[0]
        assert gradient.abs().sum() > 0


def test_roi_causal_mask_and_current_gradient(config):
    model = build_joint_model(config).eval()
    times, values = feature_fixture(gradient=True)
    first = candidate()
    scores = score_candidates(model, [first], times, values, "cpu")[0]
    assert scores["candidate_embedding"].shape == (1, 64)
    loss = scores["event_logit"].sum() + scores["quality_logit"].sum()
    loss.backward()
    assert values.grad[:, :128].abs().sum() > 0
    assert values.grad[times > first.decision_time_ms].abs().sum() == 0
    changed = values.detach().clone()
    changed[times > first.decision_time_ms] = 1000
    with torch.no_grad():
        repeated = score_candidates(model, [copy.deepcopy(first)], times, changed, "cpu")[0]
    torch.testing.assert_close(scores["event_logit"], repeated["event_logit"], rtol=0, atol=0)


def test_batch128_matches_batch1_and_metadata_not_reused(config):
    model = build_joint_model(config).eval()
    times, values = feature_fixture()
    pool = [candidate(identity=str(index)) for index in range(128)]
    with torch.no_grad():
        cache = ExactROICache("model_session")
        individual = score_candidates(model, copy.deepcopy(pool), times, values, "cpu", batch_size=1)
        batched = score_candidates(model, pool, times, values, "cpu", batch_size=128, cache=cache)
        error = max(float((left[name] - right[name]).abs().max()) for left, right in zip(individual, batched)
                    for name in ("event_logit", "quality_logit"))
        assert error <= 1e-6
        assert cache.hits == 127
        altered = copy.deepcopy(pool[0])
        altered.raw_score = .1
        other = score_candidates(model, [altered], times, values, "cpu", cache=cache)[0]
        assert not torch.equal(batched[0]["event_logit"], other["event_logit"])


def test_positive_gradient_buffer_normalizes_actual_groups():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(2)
    state = {}
    buffer = PositiveGradientBuffer(state)
    for value in (1., 3.):
        buffer.add("event_one", model(torch.tensor([[value]])).sum(), model)
    buffer.add("event_two", model(torch.tensor([[5.]])).sum(), model)
    buffer.finish(model, 4)
    torch.testing.assert_close(model.weight.grad, torch.tensor([[(2 + 5) / 8]]))
    assert not state


def test_group_limits_strict_targets_and_unique_geometries(config):
    truth = truth_fixture()
    pool = [candidate(start=30_000 + index * 3000, identity=str(index)) for index in range(3)]
    counts = {}
    for iteration in range(4):
        groups, report = sample_groups(pool, truth, config, np.random.default_rng(2026), counts)
        assert report["groups"] == int(iteration < 3)
        if groups:
            assert len(groups[0]["versions"]) == 3
    item = candidate(start=30_000, end=270_000)
    assert candidate_targets(item, truth)[0] == 0


def test_group_versions_are_attributed_to_their_own_positive_event(config):
    truth = pd.DataFrame({"subject_key": ["subject", "subject"], "session_id": ["session", "session"],
        "event_id": ["first", "second"], "start_ms": [30_000, 150_000], "end_ms": [90_000, 210_000]})
    pool = [candidate(identity="first"), candidate(start=150_000, end=210_000, identity="second")]
    counts = {}
    groups, report = sample_groups(pool, truth, config, np.random.default_rng(2026), counts)
    assert report["independent_positive_events"] == 2
    assert set(counts) == {("subject", "session", "first"), ("subject", "session", "second")}
    assert all(candidate_targets(item, truth)[2] == group["event_key"][2]
               for group in groups for item in group["versions"])


def test_committed_events_do_not_share_mutable_metadata_with_inputs_or_outputs():
    from bme_eating.joint_stream import ImmutableCommitter
    item = candidate()
    item.event_probability = item.rank_score = .9
    item.generation_parameters["source_flags"] = ["proposal"]
    committer = ImmutableCommitter(.5, .5)
    emitted = committer.submit([item])
    item.generation_parameters["source_flags"].append("changed_input")
    emitted[0].generation_parameters["source_flags"].append("changed_output")
    emitted[0].start_ms = 0
    assert committer.committed[0].start_ms == 30_000
    assert committer.committed[0].generation_parameters["source_flags"] == ["proposal"]
    assert committer.diagnostics[0].generation_parameters["source_flags"] == ["proposal"]


def test_probability_acceptance_and_parent_quality_selection():
    first, second = candidate(identity="first"), candidate(start=33_000, identity="second")
    first.event_logit, first.predicted_quality = 2., .1
    second.event_logit, second.predicted_quality = 1., .9
    output = commit_candidates([first, second], NonnegativePlatt(1, 0, True), .7, .5)
    assert [item.proposal_id for item in output] == ["second"]
    assert output[0].rank_score == output[0].event_probability
    later = candidate(start=150_000, end=180_000, identity="later", parent="adjacent", family="family")
    later.event_logit, later.predicted_quality = 2., .9
    assert len(commit_candidates([first, later], NonnegativePlatt(1, 0, True), .7, .5)) == 2


def test_unique_geometry_budget_and_parent_cdf():
    pool = [candidate(identity="first"), candidate(identity="duplicate", parent="different")]
    mapping = FrozenSourceCDF.fit(pool, {"subject"})
    learned = copy.deepcopy(pool[0])
    learned.source = "learned_onset"
    assert mapping.transform(learned) == mapping.transform(pool[0])
    tracker = OnlineBudget(20, {})
    tracker.observe(3600, 1)
    assert len(tracker.select(pool, mapping)) == 1
    assert tracker.spent == 1


def test_boundary_training_without60_and_start_only_end_unchanged(config):
    model = build_joint_model(config).eval()
    times, values = feature_fixture(gradient=True)
    item = candidate()
    score = score_candidates(model, [item], times, values, "cpu")[0]
    boundary = {"qualified": True, "mode": "start_only", "start_range_seconds": 60,
                "end_range_seconds": 30, "sigma_seconds": 9}
    loss = boundary_loss(model, item, candidate_targets(item, truth_fixture()), times, values,
                         "cpu", boundary, score["candidate_embedding"][0])
    loss.backward()
    assert model.boundary_head.head[0].weight.grad.abs().sum() > 0
    assert values.grad[:, 64:128].abs().sum() > 0
    with torch.no_grad():
        result = refine(model, item, times, values.detach(), "cpu", boundary, 1,
                        embedding=score["candidate_embedding"][0].detach())
    assert result.end_ms == item.end_ms


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


def test_blocked_history_keeps_current_graph_and_detaches13022second_history():
    historical = torch.randn(4341, 137)
    current = torch.randn(256, 137, requires_grad=True)
    blocks = FeatureBlocks([historical, current])
    selected = blocks[4300:4400, :64]
    selected.sum().backward()
    assert current.grad[:59, :64].sum() == 59 * 64
    assert not historical.requires_grad
    assert blocks.shape == (4597, 137)


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


def test_roi_has_five_regions_with40_tokens():
    times, values = feature_fixture()
    roi = indexed_roi(values[:, :64], values[:, 64:128], times, values[:, 128],
                      30_000, 90_000, 120_000)
    assert roi.shape == (40, 258)


def test_start_only_gate_and_zero_failure_confidence_interval(config):
    from bme_eating.training.statsfusion_joint_trainer import boundary_pair_gate
    coarse, refined, truth = [], [], []
    for index in range(60):
        item = candidate(start=0, end=30_000, identity=str(index), parent=str(index))
        item.subject_key = str(index)
        item.event_logit, item.predicted_quality = 2, .9
        version = copy.deepcopy(item)
        version.start_ms = 6000
        version.refined = True
        version.unrefined_start_ms, version.unrefined_end_ms = 0, 30_000
        version.boundary_valid_start_bins = 3
        coarse.append(item)
        refined.append(version)
        truth.append({"subject_key": str(index), "session_id": "session", "event_id": str(index),
                      "start_ms": 6000, "end_ms": 30_000})
    truth = pd.DataFrame(truth)
    windows = pd.DataFrame({"valid": np.ones(60)})
    platt = NonnegativePlatt(1, 0, True)
    coarse = commit_candidates(coarse, platt, .5, .5)
    refined = commit_candidates(refined, platt, .5, .5)
    start = boundary_pair_gate(coarse, refined, truth, truth.iloc[:0], windows, config, mode="start_only")
    both = boundary_pair_gate(coarse, refined, truth, truth.iloc[:0], windows, config, mode="both")
    assert start["passed"] and not both["passed"]
    assert start["tp_to_fp_rate"] == 0
    assert start["tp_to_fp_exact_95_interval"][1] > .01
    assert start["tp_to_fp_denominator"] == 60


def test_no_onset_ablation_keeps_legacy_compatibility_pool(config):
    decoder = {**config["decoder"], "ablation": "no_onset"}
    stream = make_stream("subject", "session", "model", decoder)
    emitted = []
    for index in range(1, 100):
        emitted.extend(stream.observe(index * 3000, {"state": .9 if index <= 40 else 0}, 1,
                                      onset_probability=.9 if index == 3 else 0))
    assert len(emitted) > 3
    assert not any(item.source == "learned_onset" for item in emitted)


@pytest.mark.parametrize("failure", [KeyboardInterrupt, torch.OutOfMemoryError])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_chunk_resume_restores_positive_buffers_logs_and_gradients(config, tmp_path, monkeypatch, failure, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config["training"]["device"] = device
    from types import SimpleNamespace

    from bme_eating.joint_validation import synthetic_batch
    from bme_eating.training import statsfusion_joint_trainer as trainer
    config["training"]["gradient_diagnostics_every_chunks"] = 1000
    config["training"]["gradient_clip_norm"] = 1000
    events = pd.concat([truth_fixture().assign(subject_key="fit", session_id=session,
                       start_ms=0, end_ms=6000, evaluable=True, valid_duration=True)
                        for session in ("first", "second")], ignore_index=True)
    groups = {("fit", session): pd.DataFrame({"_row_id": np.arange(index * 8, (index + 1) * 8)})
              for index, session in enumerate(("first", "second"))}
    dataset = SimpleNamespace(events=events, ignore_events=events.iloc[:0], session_groups=groups,
                              geometry=SimpleNamespace(supervised_steps=4))
    monkeypatch.setattr(trainer, "make_joint_dataset", lambda *args, **kwargs: (dataset, {"training_subjects": ["fit"]}))
    monkeypatch.setattr(trainer, "loss_schedule", lambda epoch: {"event": 1, "quality": 1,
                        "boundary": 1, "contrastive": 0, "truth_jitter": 0})
    interrupted = {"enabled": False}
    def loader(dataset, sampler, configuration):
        for position in range(sampler.start_chunk, len(sampler.chunks)):
            if interrupted["enabled"] and position == 1:
                interrupted["enabled"] = False
                raise failure("interrupted")
            row, _, _, key, _, _ = sampler.chunks[position]
            batch = synthetic_batch(configuration, 20)
            batch["timestamp_ms"] += (row % 8 - 19) * 3000
            batch["timeline_position"] += row % 8 - 19
            for name in ("state_target", "onset_target", "offset_target"):
                batch[name] = torch.zeros(1, 20)
            for name in ("state_loss_mask", "onset_loss_mask", "offset_loss_mask", "smooth_mask", "smooth_loss_mask"):
                batch[name] = torch.ones(1, 20)
            batch["subject_key"], batch["session_id"] = key
            yield batch
    def proposal(stream, times, features):
        item = candidate(start=0, end=6000, identity=str(int(times[-1])), parent=str(int(times[-1])))
        item.subject_key, item.session_id = stream.subject, stream.session
        item.model_identity = stream.identity
        item.decision_time_ms = item.max_input_time_ms = int(times[-1])
        return [item]
    monkeypatch.setattr(trainer, "session_loader", loader)
    monkeypatch.setattr(trainer, "generate_chunk", proposal)
    inputs = SimpleNamespace(identity={"data": "test"})
    trainer.train_joint_model(inputs, {"fit"}, config, tmp_path / "continuous", "identity", epochs=1, resume=False)
    interrupted["enabled"] = True
    with pytest.raises(failure):
        trainer.train_joint_model(inputs, {"fit"}, config, tmp_path / "interrupted", "identity", epochs=1, resume=False)
    saved = torch.load(tmp_path / "interrupted" / "resume.pt", weights_only=False)
    assert saved["state"]["positive_gradients"]
    journal = tmp_path / "interrupted" / "sampling.jsonl"
    with journal.open("ab") as handle:
        handle.write(b"uncommitted invalid tail")
    config["training"]["num_workers"] = 10
    config["training"]["loader_prefetch_factor"] = 1
    trainer.train_joint_model(inputs, {"fit"}, config, tmp_path / "interrupted", "identity", epochs=1, resume=True)
    complete = torch.load(tmp_path / "continuous" / "latest.pt", weights_only=True)
    resumed = torch.load(tmp_path / "interrupted" / "latest.pt", weights_only=True)
    for name in complete["model"]:
        torch.testing.assert_close(complete["model"][name], resumed["model"][name], rtol=0, atol=0)
