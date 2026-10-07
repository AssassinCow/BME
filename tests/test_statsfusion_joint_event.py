from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bme_eating.config import load_config
from bme_eating.joint_evaluation import NonnegativePlatt, paired_subject_bootstrap
from bme_eating.joint_protocol import (
    GPUResourcesBusy,
    config_identity,
    require_gate,
    require_idle_gpu,
    source_identity,
    validate_config,
)
from bme_eating.joint_stream import (
    FrozenSourceCDF,
    ImmutableCommitter,
    JointCandidate,
    OnlineBudget,
    StreamingProposals,
    TimelineCache,
    candidate_targets,
    sample_candidates,
)
from bme_eating.joint_validation import validate_joint_implementation
from bme_eating.joint_workflow import verify_oof_lineage
from bme_eating.models.statsfusion_joint_event import CompletedScale, event_contrastive_loss


@pytest.fixture
def config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v410_joint_event.yaml")


def candidate(start=0, end=10000, *, source="proposal", family="family", score=.8):
    return JointCandidate("subject", "session", start, end, end + 30_000, source, 3, score,
        start, end, family, family + source, family, "hash", "model", {}, end + 30_000)


def test_shared_model_forward_backward_and_long_event(config):
    report = validate_joint_implementation(config)
    assert report["shared_event_quality_gradients"]
    assert report["long_event"]["event_duration_seconds"] >= 13022
    assert report["repeat_max_error"] <= 1e-6


def test_completed_scale_only_uses_finished_blocks():
    torch.manual_seed(2026)
    module = CompletedScale(5, 64, 0).eval()
    values = torch.randn(1, 21, 64)
    valid = torch.ones(1, 21)
    positions = torch.arange(21)[None]
    outputs, masks = module(values, valid, positions)
    changed = values.clone()
    changed[:, 20] = 1000
    repeated, _ = module(changed, valid, positions)
    torch.testing.assert_close(outputs, repeated)
    assert not masks[:, :4].any()
    torch.testing.assert_close(outputs[:, 4], outputs[:, 8])


def test_candidate_boundary_change_relabels_strict_iou():
    truth = pd.DataFrame({"subject_key": ["subject"], "session_id": ["session"],
                          "start_ms": [0], "end_ms": [1000], "event_id": ["event"]})
    original = candidate(0, 4000)
    assert candidate_targets(original, truth)[:2] == (0.0, .25)
    original.end_ms = 3999
    assert candidate_targets(original, truth)[0] == 1


def test_frozen_source_mapping_does_not_use_prediction_percentiles():
    mapping = FrozenSourceCDF.fit([candidate(score=.1, family="low"), candidate(score=.9, family="high")], {"subject"})
    assert mapping.transform(candidate(score=.5)) == .5
    with pytest.raises(RuntimeError, match="leakage"):
        FrozenSourceCDF.fit([candidate()], {"other"})


def test_online_budget_cumulative_quotas_and_family_diversity():
    tracker = OnlineBudget(20, {"proposal": .20, "short": .15, "expansion": .15})
    tracker.observe(600, 1)
    assert tracker.allowance == 4
    pool = [candidate(source="proposal", family="first"),
            candidate(source="jitter", family="first"),
            candidate(source="start_expansion", family="second"),
            candidate(source="state", family="third")]
    mapping = FrozenSourceCDF({source: [0, 1] for source in
                              ("proposal", "jitter", "start_expansion", "state")}, ["fit"])
    selected = tracker.select(pool, mapping)
    assert len(selected) == 4
    assert len({item.proposal_family_id for item in selected[:3]}) == 3
    assert tracker.spent == 4
    assert tracker.spent == len(selected)
    assert tracker.counts["short"] == 4


def test_committed_event_cannot_be_replaced():
    commit = ImmutableCommitter(.1, .3)
    original = candidate()
    original.rank_score = .5
    assert commit.submit([original]) == [original]
    later = candidate(0, 11000, family="later")
    later.rank_score = .9
    assert not commit.submit([later])
    assert commit.committed[0].proposal_id == original.proposal_id


def test_stream_prefix_unchanged_after_future_append(config):
    def prefix(length):
        stream = StreamingProposals("subject", "session", "model", config["decoder"])
        emitted = []
        for index in range(1, length + 1):
            probabilities = {"state": .95 if index <= 5 else 0}
            emitted.extend(stream.observe(index * 3000, probabilities, 1))
        return [item.to_dict() for item in emitted]
    short, long = prefix(40), prefix(80)
    assert short
    assert short == long[:len(short)]


def test_cache_identity_and_partial_write_recovery(tmp_path):
    binding = {"subject": "subject", "session": "session", "epoch": 1, "update": 0}
    cache = TimelineCache(tmp_path, binding, 4)
    cache.append(np.asarray([3000, 6000]), np.ones((2, 4)))
    cache.append(np.asarray([9000]), np.zeros((1, 4)))
    recovered = TimelineCache(tmp_path, binding, 4, committed_rows=2)
    times, values = recovered.tensors()
    assert times.tolist() == [3000, 6000]
    assert values.shape == (2, 4) and not values.requires_grad
    del times, values
    with pytest.raises(RuntimeError, match="identity"):
        TimelineCache(tmp_path, {**binding, "update": 1}, 4, 2)


def test_family_sampling_does_not_inflate_negatives(config):
    truth = pd.DataFrame(columns=["subject_key", "session_id", "start_ms", "end_ms"])
    pool = [candidate(source="proposal"), candidate(source="jitter"), candidate(source="state")]
    selected, report = sample_candidates(pool, truth, 8, config["training"]["sample_composition"],
                                         np.random.default_rng(2026))
    assert len(selected) <= 1
    assert report["independent_families"] <= 1
    assert sum(report["desired"].values()) == 8


def test_platt_never_reverses_event_ranking():
    calibrator = NonnegativePlatt.fit(np.asarray([-2, -1, 1, 2]), np.asarray([1, 1, 0, 0]))
    assert calibrator.slope >= 0
    assert np.diff(calibrator.transform([-2, -1, 1, 2])).min() >= 0


def test_resume_only_runtime_fields_can_change(config):
    first = config_identity(config)
    runtime_change = copy.deepcopy(config)
    runtime_change["training"]["num_workers"] = 10
    runtime_change["training"]["loader_prefetch_factor"] = 2
    assert config_identity(runtime_change) == first
    runtime_change["loss"]["event"] = 2
    assert config_identity(runtime_change) != first


def test_source_identity_covers_runner_gate_export_raw(tmp_path):
    for name in ("src/bme_eating/joint_pipeline.py", "src/bme_eating/joint_workflow.py",
                 "scripts/run_statsfusion_joint.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original", encoding="utf-8")
    first = source_identity(tmp_path)
    (tmp_path / "scripts/run_statsfusion_joint.py").write_text("changed", encoding="utf-8")
    assert source_identity(tmp_path) != first


def test_missing_gate_and_oof_fail_closed(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        require_gate(tmp_path)
    with pytest.raises(RuntimeError, match="three"):
        verify_oof_lineage([], {"subject"}, {"outer"})


def test_bootstrap_keeps_pure_fp_subject():
    truth = pd.DataFrame({"subject_key": ["truth"], "session_id": ["session"],
                          "start_ms": [0], "end_ms": [1000]})
    ignore = truth.iloc[:0]
    reference = truth.copy()
    windows = pd.DataFrame({"subject_key": ["truth", "pure_fp"], "session_id": ["session"] * 2,
                            "timestamp_ms": [3000] * 2, "valid": [1, 1]})
    true_prediction = candidate(0, 1000, family="positive")
    true_prediction.subject_key = "truth"
    false_prediction = candidate(family="fp")
    false_prediction.subject_key = "pure_fp"
    report = paired_subject_bootstrap(truth, [true_prediction, false_prediction], reference, ignore, windows, 10)
    assert "pure_fp" in report["cohort"]
    assert report["difference_interval"][0] < 0


def test_contrastive_same_event_other_phases_are_not_negatives():
    values = torch.randn(4, 32, requires_grad=True)
    loss, count = event_contrastive_loss(values, torch.zeros(4, dtype=torch.long),
                                        torch.tensor([0, 0, 1, 1]))
    assert count == 0 and float(loss) == 0


def test_config_independent_joint_profile(config):
    validate_config(config)
    invalid = copy.deepcopy(config)
    invalid["training"]["seeds"] = [2026, 2027, 2028]
    with pytest.raises(ValueError, match="one seed"):
        validate_config(invalid)


def test_gpu_guard_preserves_current_training(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("bme_eating.joint_protocol.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="123, C:/env/python.exe\n"))
    with pytest.raises(GPUResourcesBusy):
        require_idle_gpu("cuda")
    require_idle_gpu("cpu")


def test_boundary_qualification_failure_returns_coarse(config):
    from bme_eating.training.statsfusion_joint_trainer import refine_candidate
    original = candidate()
    fallback = refine_candidate(None, original, torch.empty(0), torch.empty(0, 137),
                                torch.device("cpu"), {"qualified": False}, .65)
    assert fallback.start_ms == original.start_ms and fallback.end_ms == original.end_ms
    assert not fallback.refined
    assert fallback.boundary_fallback_reason == "training_qualification_failed"


def test_source_cdf_counts_one_family_per_source_scale():
    original = candidate(family="one", score=.1)
    variant = copy.deepcopy(original)
    variant.raw_score = .9
    long_scale = candidate(family="long", score=.2)
    long_scale.scale_seconds = 60
    mapping = FrozenSourceCDF.fit([original, variant, long_scale], {"subject"})
    assert mapping.distributions["proposal:3"] == [.9]
    assert mapping.distributions["proposal:60"] == [.2]


def test_missing_reference_cannot_be_assumed_qualified(tmp_path):
    from types import SimpleNamespace

    from bme_eating.joint_workflow import audit_reference
    report = audit_reference(tmp_path, SimpleNamespace(identity={}))
    assert not report["qualified"] and not report["comparison_performed"]


def test_actual_stream_lineage_ids_are_unique_and_traceable(config):
    stream = StreamingProposals("subject", "session", "model", config["decoder"])
    generated = []
    for index in range(1, 50):
        generated.extend(stream.observe(index * 3000, {"proposal_3": .95 if index < 7 else 0}, 1))
    assert generated
    assert len({item.proposal_id for item in generated}) == len(generated)
    parents = {item.proposal_id for item in generated if item.source == "proposal"}
    assert all(item.ancestor_id in parents for item in generated)
    assert all(len(item.parent_hash) == 64 and item.proposal_family_id for item in generated)


def test_shared_roi_history_graph_is_bounded():
    from types import SimpleNamespace

    from bme_eating.models.statsfusion_joint_event import JointCandidateHead
    from bme_eating.training.statsfusion_joint_trainer import score_candidate
    torch.manual_seed(2026)
    model = SimpleNamespace(candidate_head=JointCandidateHead(dropout=0).eval())
    counts = []
    for length in (256, 4341):
        history = torch.zeros(length, 137)
        current = torch.randn(20, 128, requires_grad=True)
        quality = torch.zeros(20, 9)
        quality[:, 0] = 1
        features = torch.cat((history, torch.cat((current, quality), -1)))
        features = features.clone()
        features[:, 128] = 1
        times = torch.arange(1, len(features) + 1) * 3000
        proposal = candidate(0, int(times[-1]) - 30_000)
        scores = score_candidate(model, proposal, times, features, torch.device("cpu"))
        loss = scores["event_logit"].sum() + scores["quality_logit"].sum()
        pending, visited = [loss.grad_fn], set()
        while pending:
            node = pending.pop()
            if node is None or node in visited:
                continue
            visited.add(node)
            pending.extend(parent for parent, _ in node.next_functions)
        counts.append(len(visited))
        loss.backward()
        assert current.grad is not None and current.grad.abs().sum() > 0
        assert not history.requires_grad
    assert abs(counts[0] - counts[1]) <= 5


def test_operating_point_uses_natural_oof_and_retains_fp_subject(config):
    from bme_eating.joint_evaluation import choose_operating_point
    config = copy.deepcopy(config)
    config["search"].update({"budgets": [30], "nms_ious": [.5], "acceptance_min": .1,
                             "acceptance_max": .9, "acceptance_step": .1})
    positive = candidate(0, 9000, family="positive")
    positive.subject_key, positive.event_logit, positive.predicted_quality = "positive", 8, .9
    negative = candidate(0, 9000, family="negative")
    negative.subject_key, negative.event_logit, negative.predicted_quality = "negative", -8, .9
    truth = pd.DataFrame({"subject_key": ["positive"], "session_id": ["session"],
                          "start_ms": [0], "end_ms": [9000], "hand_relation": ["same"]})
    windows = pd.DataFrame([{"subject_key": subject, "session_id": "session", "valid": 1,
                             "timestamp_ms": timestamp} for subject in ("positive", "negative")
                            for timestamp in range(3000, 300_001, 3000)])
    point, records = choose_operating_point({"setting": [positive, negative]}, windows, truth,
        truth.iloc[:0], {"model": FrozenSourceCDF({"proposal": [0, 1]}, ["fit"])}, config)
    assert point["promotion_eligible"] and point["f1"] == 1 and point["false_positive"] == 0
    assert len(records) == 9


def test_boundary_distribution_is_invariant_to_absolute_epoch_time():
    from bme_eating.models.statsfusion_joint_event import (
        distribution_endpoint,
        gaussian_distribution,
    )
    times = torch.arange(-30_000, 30_001, 3000)
    origin = 1_700_000_000_000
    target = gaussian_distribution(times, 0)
    shifted = gaussian_distribution(times + origin, origin)
    torch.testing.assert_close(target, shifted, rtol=0, atol=0)
    logits = target.clamp_min(1e-12).log()
    relative_endpoint, _ = distribution_endpoint(logits, times)
    absolute_endpoint, _ = distribution_endpoint(logits, times + origin)
    assert absolute_endpoint - origin == relative_endpoint == 0


def test_multiscale_depthwise_paths_have_five_step_causal_receptive_field():
    from bme_eating.models.statsfusion_joint_event import CausalSeparableBlock
    module = torch.nn.Sequential(CausalSeparableBlock(4, 0), CausalSeparableBlock(4, 0)).eval()
    features = torch.randn(1, 10, 4, requires_grad=True)
    output = module(features)
    torch.testing.assert_close(output[:, :5], module(features[:, :5]), atol=1e-6, rtol=0)
    gradient = torch.autograd.grad(output[:, -1].sum(), features)[0]
    assert not gradient[:, :5].abs().any()
    assert gradient[:, 5:].abs().sum() > 0
    assert all(layer.depthwise.groups == 4 for layer in module)


def test_random_background_sampling_remains_available_when_proposal_scores_are_high(config):
    from bme_eating.training.statsfusion_joint_trainer import random_background_candidates
    empty = pd.DataFrame(columns=["subject_key", "session_id", "start_ms", "end_ms"])
    times = torch.arange(1, 257) * 3000
    features = torch.zeros(256, 137)
    features[:, 128:130] = 1
    generated = random_background_candidates(empty, empty, "subject", "session", times,
                                             features, "model", np.random.default_rng(2026))
    assert generated
    assert len({item.proposal_family_id for item in generated}) == len(generated)
    chosen, report = sample_candidates(generated, empty, 8,
        config["training"]["sample_composition"], np.random.default_rng(2026))
    assert len(chosen) == 1
    assert report["deficits"]["random_background"] == 0
    assert chosen[0].max_input_time_ms <= chosen[0].decision_time_ms
    assert chosen[0].decision_time_ms - chosen[0].end_ms <= 30_000
    truth = pd.DataFrame([{"subject_key": "subject", "session_id": "session",
                           "start_ms": 0, "end_ms": 768_000}])
    assert not random_background_candidates(truth, empty, "subject", "session", times,
                                             features, "model", np.random.default_rng(2026))
