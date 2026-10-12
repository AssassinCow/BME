from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from test_statsfusion_v411 import candidate, feature_fixture, truth_fixture

from bme_eating.config import load_config
from bme_eating.data.joint_v412_augmentation import (
    JointSessionDatasetV412,
    augmentation_description,
    selected_statistics,
    transform_payload,
)
from bme_eating.features.baseline import _motion_window_features, _ppg_window_features
from bme_eating.joint_protocol import event_score_formula, stages_for, validate_config
from bme_eating.joint_stream import candidate_targets
from bme_eating.joint_v412_training import (
    ReplaceablePositiveGradientBuffer,
    describe_group,
    heatmap_focal_loss,
    learning_rate_fraction,
    sample_groups,
    subject_weights,
)
from bme_eating.models.statsfusion_joint_event_v411 import build_joint_model
from bme_eating.stats_features import STATS_FEATURE_COLUMNS


@pytest.fixture
def config():
    torch.set_num_threads(2)
    value = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v412_integrated_repair.yaml")
    value["training"]["device"] = "cpu"
    return value


def raw_fixture(seconds=40):
    rng = np.random.default_rng(2026)
    motion_times = np.arange(seconds * 100) * 10
    return {"motion_timestamp_ms": motion_times, "motion_values": rng.normal(size=(len(motion_times), 6)).astype(np.float32),
            "motion_mask": np.ones((len(motion_times), 6), bool),
            "ppg_timestamp_ms": np.arange(seconds * 50) * 20,
            "ppg_values": rng.normal(size=(seconds * 50, 1)).astype(np.float32),
            "ppg_mask": np.ones((seconds * 50, 1), bool)}


def test_profile_defaults_and_legal_execution_modes(config):
    validate_config(config)
    assert event_score_formula(config) == "calibrated_event_probability_v412"
    assert "ablation" not in stages_for(config)
    for mode, internal in (("fold0", "oof"), ("fivefold", "single")):
        invalid = copy.deepcopy(config)
        invalid["execution"] = {"mode": mode, "internal_mode": internal}
        with pytest.raises(ValueError, match="fold0"):
            validate_config(invalid)
    config["execution"] = {"mode": "fivefold", "internal_mode": "oof"}
    validate_config(config)


def test_augmentation_determinism_dropout_rotation_and_natural_statistics(config):
    training = config["training"]
    assert augmentation_description(2026, 3, "subject", "session", training) == augmentation_description(
        2026, 3, "subject", "session", training)
    payload = raw_fixture()
    natural = selected_statistics(payload, 30_000)
    motion = (payload["motion_timestamp_ms"] >= 15_000) & (payload["motion_timestamp_ms"] <= 30_000)
    ppg = (payload["ppg_timestamp_ms"] >= 15_000) & (payload["ppg_timestamp_ms"] <= 30_000)
    expected = _motion_window_features(payload["motion_values"][motion], payload["motion_mask"][motion], 100)
    expected.update(_ppg_window_features(payload["ppg_values"][ppg].ravel(), payload["ppg_mask"][ppg].ravel(), 50))
    np.testing.assert_allclose(natural, [expected[name] for name in STATS_FEATURE_COLUMNS], rtol=1e-6)
    description = {"rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                   "gyro_dropout": True, "ppg_dropout": True, "bandwidth_hz": None}
    changed = transform_payload(payload, description, 0)
    np.testing.assert_array_equal(changed["motion_values"][:, 1], payload["motion_values"][:, 0])
    assert not changed["motion_mask"][:, 3:].any()
    assert not changed["ppg_mask"].any()
    statistics = dict(zip(STATS_FEATURE_COLUMNS, selected_statistics(changed, 30_000)))
    assert np.isnan(statistics["local_gyro_z_mad"])
    assert statistics["local_ppg_valid_fraction"] == 0
    assert payload["motion_mask"].all()


@pytest.mark.parametrize("rate", [25.0, 26.316, 50.0])
def test_low_bandwidth_causal_overlap_and_gaps(rate):
    payload = raw_fixture()
    payload["motion_mask"][1800:1900] = False
    description = {"rotation": np.eye(3).tolist(), "gyro_dropout": False,
                   "ppg_dropout": False, "bandwidth_hz": rate}
    full = transform_payload(payload, description, 0)
    cropped = {name: array[200 if name.startswith("ppg") else 400:].copy() for name, array in payload.items()}
    part = transform_payload(cropped, description, 0)
    np.testing.assert_array_equal(full["motion_values"][700:], part["motion_values"][300:])
    np.testing.assert_array_equal(full["ppg_values"][400:], part["ppg_values"][200:])
    assert not full["motion_mask"][1800:1900].any()
    prefix = {name: array[:1000 if name.startswith("ppg") else 2000] for name, array in payload.items()}
    np.testing.assert_array_equal(full["motion_values"][:2000], transform_payload(prefix, description, 0)["motion_values"])


def test_endpoint_alignment_ignores_left_censoring_and_counts_sensor_validity():
    dataset = object.__new__(JointSessionDatasetV412)
    key = ("subject", "session")
    group = pd.DataFrame({"timestamp_ms": np.arange(1, 21) * 3000,
                          "start_loss_mask": 1, "end_loss_mask": 1})
    dataset.session_groups, dataset.session_origins = {key: group}, {key: 0}
    dataset.events = pd.DataFrame({"subject_key": [key[0]], "session_id": [key[1]],
                                  "start_ms": [3100], "end_ms": [33_000]})
    onset, offset, _ = dataset._transition_targets(*key, group.timestamp_ms.to_numpy())
    assert onset[1] == 1 and onset[0] == 0 and offset[10] == 1
    assert 0 <= group.timestamp_ms.iloc[onset.argmax()] - 3100 < 3000
    dataset.description = {"gyro_dropout": True, "ppg_dropout": True}
    payload = raw_fixture(61)
    payload["motion_mask"][:900, :3] = False
    dataset.reader = SimpleNamespace(reader=SimpleNamespace(read=lambda *args, **kwargs: payload))
    positive, background = dataset.endpoint_counts(key, "onset")
    assert positive == 0 and background == 17
    dataset.events.loc[0, "start_ms"] = 0
    assert dataset._transition_targets(*key, group.timestamp_ms.to_numpy())[0].max() == 0


def test_endpoint_peak_gradient_does_not_dilute_with_long_background():
    gradients = []
    for length in (20, 2000):
        logits = torch.zeros(length, requires_grad=True)
        target = torch.zeros(length)
        target[0] = 1
        heatmap_focal_loss(logits, target, torch.ones(length), torch.tensor(1), torch.tensor(length - 1)).backward()
        gradients.append(logits.grad[0])
    torch.testing.assert_close(*gradients, rtol=0, atol=0)


def test_subject_weights_and_scheduler_prefix(config):
    weights = subject_weights([("many", str(index)) for index in range(85)]
                              + [("few", str(index)) for index in range(11)])
    assert .5 <= weights["many"] < weights["few"] <= 2
    assert (85 * weights["many"] + 11 * weights["few"]) / 96 == pytest.approx(1)
    assert learning_rate_fraction(0, config["training"]) == pytest.approx(.1)
    assert learning_rate_fraction(2, config["training"]) == 1
    assert learning_rate_fraction(64, config["training"]) == pytest.approx(.1)
    shorter = {**config["training"], "max_epochs": 16}
    assert learning_rate_fraction(16, shorter) == learning_rate_fraction(16, config["training"])


def test_replaceable_positive_gradients_remove_displaced_contribution():
    model = torch.nn.Linear(1, 1, bias=False)
    state = {}
    buffer = ReplaceablePositiveGradientBuffer(state)
    for index in range(1, 5):
        descriptor = {"id": str(index), "maximum_iou": index / 5, "bins": [0],
                      "decision": 0, "geometries": [(index, index + 10)]}
        buffer.add({"event_key": "event", "descriptor": descriptor}, model(torch.tensor([[float(index)]])).sum(), model)
    assert set(state["event"]) == {"2", "3", "4"}
    assert all(value.device.type == "cpu" for value in buffer.tensors())
    buffer.finish(model, 2)
    assert model.weight.grad.item() == pytest.approx(1.5)
    assert state == {}


def test_iou_band_endpoints_and_strict_positive_labels():
    truth = truth_fixture()
    quarter = candidate(start=30_000, end=270_000)
    assert candidate_targets(quarter, truth)[:2] == (0, .25)
    for end, band in ((150_000, 0), (110_000, 1), (90_000, 2)):
        group = {"family": "family", "event_key": "event", "versions": [candidate(end=end)]}
        assert describe_group(group, truth)["bins"] == [band]


def test_hard_negative_mining_uses_event_head_and_restores_dropout(config, monkeypatch):
    model = torch.nn.Linear(1, 1).train()
    pool = [candidate(start=300_000 + index * 1000, end=360_000 + index * 1000,
                      identity=str(index), family=str(index)) for index in range(40)]
    for index, item in enumerate(pool):
        item.raw_score = 1 - index / 40
    calls = []
    def score(network, candidates, *args, **kwargs):
        assert not network.training and not torch.is_grad_enabled()
        calls.append(len(candidates))
        return [{"event_logit": torch.tensor([float(item.proposal_id)])} for item in candidates]
    monkeypatch.setattr("bme_eating.joint_v412_training.score_candidates", score)
    selected, report = sample_groups(model, pool, truth_fixture(), config, np.random.default_rng(2026),
                                     {}, set(), None, None, "cpu")
    assert model.training and max(calls) <= 96
    assert report["mining_families"] <= 32
    ranking = report["mining_ranking"]
    assert ranking[0]["bce"] >= ranking[-1]["bce"]
    assert [group["family"] for group in selected[:3]] == [group["family"] for group in ranking[:3]]
    assert report["independent_negative_families"] == 4


def test_roi_regions_and_invalid_tokens_do_not_contaminate(config):
    head = build_joint_model(config).candidate_head.eval()
    roi = torch.randn(1, 40, 258)
    roi[..., -2] = 1
    roi[:, 1, -2] = 0
    changed = roi.clone()
    changed[:, 1, :-2] = 1e5
    changed[:, 4:20, :-2] = 1e5
    with torch.no_grad():
        expected = head.encode_regions(roi)
        actual = head.encode_regions(changed)
    torch.testing.assert_close(expected[:, :4], actual[:, :4], rtol=0, atol=0)
    assert not actual[:, 1].any()
    times, features = feature_fixture(50, gradient=True)
    from bme_eating.joint_v411_scoring import score_candidates
    model = build_joint_model(config)
    score = score_candidates(model, [candidate()], times, features, "cpu")[0]
    (score["event_logit"].sum() + score["quality_logit"].sum()).backward()
    assert features.grad[:, :128].abs().sum() > 0


def test_missing_reference_remains_diagnostic(config, tmp_path):
    from bme_eating.joint_single_fold import fold_reference_comparison
    result = fold_reference_comparison(None, config, tmp_path)
    assert not result["comparison_performed"] and not result["qualified"]


@pytest.mark.parametrize("changes", [None, {"epoch_complete": False}, {"clipping_fraction": .21},
    {"identity": ""}, {"model": {"weight": torch.tensor([float("nan")])}},
    {"source_sha256": "stale"}, {"data_sha256": "different"},
    {"preprocessing": {"training_subjects": ["holdout"]}}, {"excluded_subjects": ["fit"]}])
def test_benchmark_requires_qualified_checkpoint_and_exclusions(config, monkeypatch, changes):
    import importlib.util

    from bme_eating.joint_protocol import config_identity, digest, source_identity

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("v412_benchmark_test", root / "scripts/benchmark_statsfusion_v412.py")
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    data = {"fixture": "natural"}
    payload = {"epoch": 8, "epoch_complete": True, "promotion_eligible": True,
        "clipping_fraction": .1, "identity": "model", "training_subjects": ["fit"],
        "excluded_subjects": ["holdout"], "model": {"weight": torch.ones(1)},
        "config_sha256": config_identity(config), "source_sha256": digest(source_identity(root)),
        "data_sha256": digest(data), "preprocessing": {"training_subjects": ["fit"]}}
    if changes is None:
        assert benchmark.require_benchmark_checkpoint(payload, config, {"holdout"}, data) == {"fit"}
    else:
        with pytest.raises(RuntimeError, match="Benchmark"):
            benchmark.require_benchmark_checkpoint({**payload, **changes}, config, {"holdout"}, data)


def test_feature_cache_checks_identity_checksum_and_causal_prefix(config, tmp_path):
    from bme_eating.joint_v412_cache import PredictionFeatureCache, feature_binding
    model = torch.nn.Linear(1, 1)
    inputs = SimpleNamespace(identity={"data": "natural"})
    binding = feature_binding(model, inputs, config, {}, "model")
    cache = PredictionFeatureCache(tmp_path, binding)
    cache.append(torch.tensor([3000, 6000, 9000]), torch.ones(3, 137))
    cache.finish()
    cached = PredictionFeatureCache(tmp_path, binding)
    cached.read(np.array([3000]))
    assert len(cached.prefix()[0]) == 1
    with pytest.raises(RuntimeError, match="trailing"):
        cached.finish()
    cached.read(np.array([6000, 9000]))
    cached.finish()
    with torch.no_grad():
        model.weight.add_(1)
    changed = feature_binding(model, inputs, config, {}, "model")
    assert changed != binding
    with pytest.raises(RuntimeError, match="identity"):
        PredictionFeatureCache(tmp_path, changed)
    with (tmp_path / "features.bin").open("r+b") as handle:
        handle.write(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        PredictionFeatureCache(tmp_path, binding)


def test_boundary_insufficient_evidence_skips_all_refinement_search(config):
    from bme_eating.joint_v411_boundary import lock_boundary_modes
    point, reports, records = lock_boundary_modes({"candidate_recall": .9}, [candidate()], [], [],
        truth_fixture(), pd.DataFrame(), pd.DataFrame(), config)
    assert point["boundary_mode"] == "off" and not point["boundary_enabled"]
    assert reports[0]["search_skipped"] and records == []


def test_endpoint_diagnostics_reports_hit_and_background_rate():
    from bme_eating.joint_v412_diagnostics import endpoint_diagnostics
    windows = pd.DataFrame({"subject_key": "subject", "session_id": "session",
        "timestamp_ms": np.arange(1, 51) * 3000, "valid": 1,
        "onset_probability": [0] * 9 + [.8] + [0] * 19 + [.9] + [0] * 20,
        "offset_probability": [0] * 29 + [.7] + [0] * 20})
    truth = pd.DataFrame({"subject_key": ["subject"], "session_id": ["session"],
                          "start_ms": [30_000], "end_ms": [90_000]})
    report = endpoint_diagnostics(windows, truth, truth.iloc[:0], [])
    assert report["onset"]["endpoint_hits"] == 1
    assert report["onset"]["background_peaks"] == 1
    assert report["offset"]["maximum_probability"] == .7


def test_prediction_cache_skips_raw_loading_and_preserves_scored_candidates(config, tmp_path, monkeypatch):
    from test_statsfusion_joint_io_runtime import STATISTICS_COLUMNS, native_session_frames

    from bme_eating.data.deep_dataset import Normalization
    from bme_eating.data.stats_fusion_sequence import SequenceGeometry
    from bme_eating.joint_stream import FrozenSourceCDF
    from bme_eating.joint_v411_prediction import predict
    from bme_eating.stats_features import FoldRobustScaler

    anchors, segments, events = native_session_frames(tmp_path)
    config["training"]["num_workers"] = 0
    geometry = SequenceGeometry(supervised_steps=4, short_receptive_field_steps=7,
        fused_short_receptive_field_steps=7, long_receptive_field_tokens=3)
    inputs = SimpleNamespace(identity={"fixture": "natural"}, frames=lambda *args, **kwargs:
                             (anchors, None, events, segments))
    def dataset(*args, **kwargs):
        result = JointSessionDatasetV412(anchors, segments, events,
            Normalization(np.zeros(6), np.ones(6), 0., 1.), statistics_columns=STATISTICS_COLUMNS,
            geometry=geometry, training=False, seed=2026)
        result.configure(config, FoldRobustScaler(STATS_FEATURE_COLUMNS, np.zeros(12), np.ones(12),
                                                  ("fit",)), inputs.identity)
        return result, {}
    monkeypatch.setattr("bme_eating.joint_v411_prediction.make_joint_dataset", dataset)
    torch.manual_seed(2026)
    model = build_joint_model(config).eval()
    subjects = {"first", "second"}
    parents = predict(model, inputs, subjects, config, {}, tmp_path / "cached/cdf", "model",
                      {}, source_only=True)
    mapping = FrozenSourceCDF.fit([item for pool in parents["coarse"].values() for item in pool], subjects)
    fresh = predict(model, inputs, subjects, config, {}, tmp_path / "fresh/prediction", "model",
                    {"search_eligible": False}, mapping=mapping)
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed feature cache must bypass raw loading and encoding")
    monkeypatch.setattr("bme_eating.joint_v411_prediction.session_loader", forbidden)
    monkeypatch.setattr(model, "forward", forbidden)
    reused = predict(model, inputs, subjects, config, {}, tmp_path / "cached/prediction", "model",
                     {"search_eligible": False}, mapping=mapping)
    pd.testing.assert_frame_equal(fresh["windows"], reused["windows"])
    assert reused["performance"]["feature_cache_hit_chunks"] == 20
    assert reused["performance"]["cache_write_bytes"] == 0
    assert reused["performance"]["roi_hits"] == fresh["performance"]["roi_hits"]
    assert sum(map(len, fresh["coarse"].values())) > 0
    for setting in fresh["coarse"]:
        assert [item.to_dict() for item in fresh["coarse"][setting]] == [
            item.to_dict() for item in reused["coarse"][setting]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_restore_moves_replaceable_positive_gradients_back_to_cpu(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False).cuda()
    state = {}
    buffer = ReplaceablePositiveGradientBuffer(state)
    descriptor = {"id": "first", "maximum_iou": .5, "bins": [0], "decision": 0,
                  "geometries": [(0, 10)]}
    buffer.add({"event_key": "event", "descriptor": descriptor}, model(torch.ones(1, 1, device="cuda")).sum(), model)
    torch.save(state, tmp_path / "buffer.pt")
    restored = torch.load(tmp_path / "buffer.pt", map_location="cuda", weights_only=True)
    resumed = ReplaceablePositiveGradientBuffer(restored)
    assert all(value.device.type == "cpu" for value in resumed.tensors())
    descriptor = {**descriptor, "id": "better", "maximum_iou": .9, "geometries": [(1, 10)]}
    resumed.add({"event_key": "event", "descriptor": descriptor}, model(torch.full((1, 1), 3., device="cuda")).sum(), model)
    resumed.finish(model, 4)
    assert model.weight.grad.item() == pytest.approx(.5)
