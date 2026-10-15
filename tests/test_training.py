from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from helpers import candidate, truth_fixture

from bme_eating.candidate_scoring import boundary_loss, score_candidates
from bme_eating.candidates import FrozenSourceCDF, OnlineBudget, candidate_targets
from bme_eating.config import load_config
from bme_eating.data.augmentation import augmentation_description
from bme_eating.evaluation import NonnegativePlatt
from bme_eating.models.stats_fusion_state import CausalTCN
from bme_eating.models.statsfusion_joint_event import (
    build_joint_model,
    fractional_mean,
    indexed_roi,
)
from bme_eating.protocol import calibration_discriminates, capabilities, validate_config
from bme_eating.training.candidate_supervision import (
    ContributionBank,
    PositiveGradientBuffer,
    group_losses,
    heatmap_focal_loss,
    sample_groups,
    selection_update,
)
from bme_eating.training.natural_replay import preserve_training_state, release_candidates
from bme_eating.training.statsfusion_joint_trainer import decoder_settings, pack_features
from bme_eating.validation import synthetic_batch


@pytest.fixture
def config():
    torch.set_num_threads(2)
    value = load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v414_full_data.yaml")
    value["training"]["device"] = "cpu"
    return value


def test_registered_identity_complete_search_and_layout(config):
    validate_config(config)
    assert capabilities(config).feature_width == 201
    assert len(decoder_settings(config)) * 3 * 3 * 99 == 16038
    for mode, internal in (("fold0", "oof"), ("fivefold", "single")):
        invalid = copy.deepcopy(config)
        invalid["execution"] = {"mode": mode, "internal_mode": internal}
        with pytest.raises(ValueError):
            validate_config(invalid)


def test_fractional_pooling_empty_and_three_second_identity(config):
    values = torch.ones(3, 4)
    torch.testing.assert_close(fractional_mean(values, torch.tensor([.1, .2, .0])), torch.ones(4))
    assert fractional_mean(values, torch.zeros(3)).eq(0).all()
    model = build_joint_model(config).eval()
    module = model.scales["3"]
    received = []
    handle = module.temporal.register_forward_pre_hook(lambda _, args: received.append(args[0]))
    module(values[None, :, :1].expand(-1, -1, 64), torch.full((1, 3), .2), torch.arange(3)[None])
    handle.remove()
    assert received[0].eq(1).all()


def test_masked_tcn_invalid_warmup_and_missing_bins():
    torch.manual_seed(2026)
    temporal = CausalTCN(8, 8, (1, 2, 4, 8, 16, 32), 0).eval()
    values = torch.randn(1, 160, 8, requires_grad=True)
    valid = torch.zeros(1, 160)
    valid[:, -10:] = 1
    outputs = temporal(values, valid)
    assert outputs[:, :-10].eq(0).all()
    changed = values.detach().clone()
    changed[:, :-10] += 1000
    torch.testing.assert_close(outputs, temporal(changed, valid), atol=0, rtol=0)
    (outputs[:, -10:].square().mean()).backward()
    assert values.grad[:, :-10].eq(0).all()
    norm = torch.nn.utils.clip_grad_norm_(temporal.parameters(), float("inf"))
    assert torch.isfinite(norm) and norm < 100
    torch.testing.assert_close(temporal(values, torch.ones_like(valid)), temporal(values), atol=0, rtol=0)
    temporal.zero_grad(set_to_none=True)
    empty = temporal(values, torch.zeros_like(valid))
    assert empty.eq(0).all()
    empty.sum().backward()
    assert all(parameter.grad is None or parameter.grad.eq(0).all() for parameter in temporal.parameters())


def test_contribution_bank_rejects_disconnected_classifier(config):
    model = build_joint_model(config)
    state = {}
    bank = ContributionBank(state)
    bank.pending("negative", {"encoder.state_head.bias": torch.ones_like(model.encoder.state_head.bias)})
    with pytest.raises(RuntimeError, match="classification gradient disconnected"):
        bank.commit(model)
    assert state["pending_task_sessions"] == {"negative": 1}
    boundary_bank = ContributionBank({})
    boundary_bank.pending("boundary", {"encoder.state_head.bias": torch.ones_like(model.encoder.state_head.bias)})
    with pytest.raises(RuntimeError, match="boundary gradient disconnected"):
        boundary_bank.commit(model)


def test_roi_coverage_and_prefix_mask():
    times = torch.arange(1, 61) * 3000
    semantic, local = torch.ones(60, 64), torch.ones(60, 64)
    valid = torch.full((60,), .2)
    roi = indexed_roi(semantic, local, times, valid, 30_000, 90_000, 90_000)
    assert roi.shape == (40, 259)
    assert roi[roi[:, 256] > 0, :128].eq(1).all()
    assert roi[20:24, 257].eq(0).all()
    assert roi[36:, 257].eq(0).all()
    prefix = indexed_roi(semantic[:30], local[:30], times[:30], valid[:30], 30_000, 90_000, 90_000)
    torch.testing.assert_close(roi, prefix, atol=0, rtol=0)


def test_localization_causality_raw_encoded_once_and_shared_losses(config):
    model = build_joint_model(config).eval()
    batch = synthetic_batch(config, 40)
    batch["motion_blocks"][:, :, 6:] = 1
    counts = {}
    handles = []
    for name in ("acc_encoder", "gyro_encoder", "invariant_motion_encoder", "ppg_encoder"):
        def count_hook(_, __, ___, name=name):
            counts[name] = counts.get(name, 0) + 1
        handles.append(getattr(model.encoder, name).register_forward_hook(count_hook))
    outputs = model(batch)
    for handle in handles:
        handle.remove()
    assert set(counts.values()) == {1}
    assert model.localization_receptive_field_seconds == 45
    assert model.localization_dependency_seconds["ppg_maximum_hold"] == 12
    changed = {name: value.clone() for name, value in batch.items()}
    changed["motion_blocks"][:, 25:, :6] += 100
    changed["statistics"][:, 25:, :12] += 100
    future = model(changed)
    torch.testing.assert_close(outputs["localization_features"][:, :25], future["localization_features"][:, :25], atol=0, rtol=0)
    packed = pack_features(outputs, torch.ones(40, dtype=torch.bool))
    assert packed.shape == (40, 201)
    item = candidate(start=30_000, end=90_000)
    scores = score_candidates(model, [item], batch["timestamp_ms"][0], packed, "cpu")[0]
    endpoint = boundary_loss(model, item, candidate_targets(item, truth_fixture()),
        batch["timestamp_ms"][0], packed, "cpu", {"start_range_seconds": 900}, scores["candidate_embedding"][0])
    for loss in (scores["event_logit"].sum(), scores["quality_logit"].sum(), endpoint):
        gradient = torch.autograd.grad(loss, model.encoder.acc_encoder.parameters().__next__(), retain_graph=True)[0]
        assert gradient.abs().sum() > 0


def test_region_isolation_invalid_bins_and_batch_equivalence(config):
    head = build_joint_model(config).candidate_head.eval()
    roi = torch.randn(1, 40, 259)
    roi[..., 256:258] = 1
    changed = roi.clone()
    changed[:, 4:20, :256] += 100
    torch.testing.assert_close(head.encode_regions(roi)[:, :4], head.encode_regions(changed)[:, :4], atol=0, rtol=0)
    roi[:, 5, 256] = 0
    changed = roi.clone()
    changed[:, 5, :256] += 1000
    torch.testing.assert_close(head.encode_regions(roi), head.encode_regions(changed), atol=0, rtol=0)
    with torch.no_grad():
        single = head(roi, torch.zeros(1, 12))
        batch = head(roi.expand(128, -1, -1), torch.zeros(128, 12))
    for name in single:
        assert (single[name] - batch[name][:1]).abs().max() <= 1e-6


def test_scoring_instrumentation_preserves_results(config):
    model = build_joint_model(config).eval()
    times = torch.arange(1, 21) * 3000
    features = torch.randn(20, 201)
    features[:, 128:137] = .5
    pool = [candidate(start=0, end=30_000)]
    performance = {}
    with torch.no_grad():
        plain = score_candidates(model, copy.deepcopy(pool), times, features, "cpu")
        timed = score_candidates(model, copy.deepcopy(pool), times, features, "cpu",
                                 performance=performance)
    for name in plain[0]:
        torch.testing.assert_close(plain[0][name], timed[0][name], atol=0, rtol=0)
    assert performance["scoring_candidates"] == performance["scoring_head_calls"] == 1
    assert performance["roi_seconds"] >= 0
    assert performance["candidate_head_seconds"] >= 0


def test_endpoint_session_chunk_equivalence_and_background_strength():
    targets = torch.zeros(2000)
    targets[300] = 1
    logits = torch.zeros_like(targets, requires_grad=True)
    peak, background = torch.tensor(1), torch.tensor(1999)
    whole = heatmap_focal_loss(logits, targets, torch.ones_like(targets), peak, background)
    chunks = sum(heatmap_focal_loss(logits[first:first + 256], targets[first:first + 256],
                 torch.ones_like(targets[first:first + 256]), peak, background) for first in range(0, 2000, 256))
    torch.testing.assert_close(whole, chunks)
    assert torch.autograd.grad(whole, logits)[0][300].abs() > 0
    constant = torch.tensor(0., requires_grad=True)
    optimizer = torch.optim.Adam([constant], lr=.05)
    for _ in range(150):
        optimizer.zero_grad()
        heatmap_focal_loss(constant.expand(2), torch.tensor([1., 0.]), torch.ones(2), torch.tensor(1), torch.tensor(1)).backward()
        optimizer.step()
    assert .24 < float(constant.sigmoid().detach()) < .26


def test_task_bank_replacement_and_contributing_session_mean():
    model = torch.nn.Linear(1, 1, bias=False)
    state = {"positive_gradients": {}}
    bank = ContributionBank(state)
    buffer = PositiveGradientBuffer(state["positive_gradients"])
    def group(identity, quality):
        return {"event_key": "event", "descriptor": {"id": identity, "maximum_iou": quality,
            "bins": [2], "decision": 10, "geometries": [(0, 10)]}}
    buffer.add(group("old", .6), {"classification": model.weight.sum() * 100}, model)
    buffer.add(group("new", .9), {"classification": model.weight.sum() * 2}, model)
    assert list(state["positive_gradients"]["event"]) == ["new"]
    bank.negative(model.weight.sum() * 8, model)
    bank.negative(model.weight.sum() * 4, model)
    bank.finish_session(state["positive_gradients"])
    bank.negative(model.weight.sum() * 10, model)
    bank.finish_session({})
    bank.finish_session({})
    bank.commit(model)
    assert float(model.weight.grad) == pytest.approx(5.0)


def test_rng_preservation_release_and_augmentation_schedule(config):
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout()).train()
    rng = torch.get_rng_state().clone()
    with preserve_training_state(model):
        assert not model.training
        torch.rand(200)
        np.random.rand(200)
    assert model.training and torch.equal(rng, torch.get_rng_state())
    first, second = candidate(), candidate(end=120_000, identity="two")
    state = {"cursor": 0, "released": 0}
    payload = {"candidates": [first, second]}
    assert not release_candidates(payload, state, first.decision_time_ms - 1)
    assert release_candidates(payload, state, first.decision_time_ms) == [first]
    training = {**config["training"], "_stabilized": True}
    for epoch, dropout, bandwidth in ((1, .1, 0), (2, .1, 0), (5, .15, .1), (8, .2, .2)):
        description = augmentation_description(2026, epoch, "s", "x", training)
        assert description["schedule"]["gyro_modality_dropout"] == pytest.approx(dropout)
        assert description["schedule"]["low_bandwidth_probability"] == pytest.approx(bandwidth)


def test_budget_first_geometry_and_cdf_fallback():
    mapping = FrozenSourceCDF({"proposal:3": [.1, .8], "state:15": [.3]}, ["fit"], fallback=True)
    first, later = candidate(), candidate(identity="later")
    later.decision_time_ms += 3000
    later.max_input_time_ms += 3000
    budget = OnlineBudget(20, {})
    budget.observe(3600, 1)
    assert budget.select([first], mapping) == [first]
    assert not budget.select([later], mapping)
    assert budget.spent == 1
    unseen = candidate()
    unseen.scale_seconds = 60
    assert mapping.transform(unseen) == 1
    assert mapping.fallback_counts["parent"] == 1
    unseen.generation_parameters["parent_source"] = "unseen"
    assert mapping.transform(unseen) == 1
    assert mapping.fallback_counts["global"] == 1


def test_no_qualified_early_stop_and_constant_calibration(config):
    calibration = NonnegativePlatt.fit(np.ones(10), np.array([0] * 5 + [1] * 5))
    assert not calibration.discriminative
    assert not calibration_discriminates(config, calibration.to_json())
    assert not calibration_discriminates(config, {"slope": 1, "qualified": True})
    state = {"best_epoch": None, "best_f1": -1, "stale_checks": 0}
    for epoch in range(8, 24, 2):
        assert not selection_update(state, epoch, False, .9, config["training"])[1]
    assert selection_update(state, 24, False, .9, config["training"])[1]
    assert state["selection_failure"] == "qualification_deadline"
    state = {"best_epoch": None, "best_f1": -1, "stale_checks": 0}
    assert selection_update(state, 8, True, .5, config["training"])[0]
    for epoch in (10, 12):
        assert not selection_update(state, epoch, False, .8, config["training"])[1]
    assert selection_update(state, 14, False, .8, config["training"])[1]


def test_random_anchor_reserved_and_mining_limit(config):
    model = build_joint_model(config).eval()
    times = torch.arange(1, 61) * 3000
    features = torch.zeros(60, 201)
    features[:, 128:137] = .8
    candidates = [candidate(start=120_000, end=150_000, identity=str(index), family=str(index)) for index in range(40)]
    groups, report = sample_groups(model, candidates, truth_fixture(), config, np.random.default_rng(2026),
                                  {}, set(), times, features, "cpu")
    assert report["random_anchor_reserved"]
    assert report["mining_head_candidates"] <= 96
    assert sum(group["category"] == "random_background" for group in groups) == 1
    assert len({group["family"] for group in groups}) == len(groups)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bf16_mining_then_loss_updates_all_heads(config, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(2026)
    model = build_joint_model(config).to(device).train()
    times = torch.arange(1, 61) * 3000
    features = torch.randn(60, 201, requires_grad=True)
    with torch.no_grad():
        features[:, 128:137] = .8
    candidates = [candidate(identity="positive", family="positive"),
                  candidate(start=120_000, end=150_000, identity="negative", family="negative")]
    parameters = {name: parameter for name, parameter in model.named_parameters()
                  if name.startswith(("candidate_head.event.", "candidate_head.quality.", "boundary_head."))}
    before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        groups, _ = sample_groups(model, candidates, truth_fixture(), config,
            np.random.default_rng(2026), {}, set(), times, features, device)
        groups.sort(key=lambda group: group["category"] == "positive")
        grouped, _ = group_losses(model, groups, truth_fixture(), times, features, device,
            {"start_range_seconds": 900, "end_range_seconds": 30}, config, {"boundary": 1})
        loss = sum(tasks["event"] + tasks["quality"] + tasks["boundary"] for _, tasks in grouped)
    assert model.training
    loss.backward()
    for name, parameter in parameters.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        if name != "boundary_head.head.2.bias":
            assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    for name, parameter in parameters.items():
        if name != "boundary_head.head.2.bias":
            assert (parameter.detach() - before[name]).abs().max() > 0, name


def test_strict_iou_quarter_negative():
    assert candidate_targets(candidate(start=30_000, end=45_000), truth_fixture())[0] == 0


def test_replay_hash_rejects_changed_cache(config, tmp_path):
    from types import SimpleNamespace

    from bme_eating.training.natural_replay import replay_session
    model = build_joint_model(config)
    with pytest.raises((RuntimeError, FileNotFoundError), match="Natural replay|path|file"):
        replay_session(model, SimpleNamespace(identity={}), config, {}, None, None,
            ("subject", "session"), tmp_path, "model", 3, 0,
            stored={"binding": {}, "path": "wrong", "sha256": "wrong"})


@pytest.mark.parametrize("failure", [KeyboardInterrupt, torch.OutOfMemoryError])
def test_replay_and_contribution_chunk_resume(config, tmp_path, monkeypatch, failure, *,
                                            epochs=1, deployment_epoch=None, interruption_epoch=1,
                                            empty_source_validation=False, periodic_baseline=False,
                                            development_mode=False, compatible_resume=False):
    from types import SimpleNamespace

    import pandas as pd

    from bme_eating.data.joint_sessions import SessionChunkSampler
    from bme_eating.training import statsfusion_joint_trainer as trainer

    config["training"].update({"gradient_clip_norm": 1000, "gradient_diagnostics_every_chunks": 1000,
                               "natural_replay_from_epoch": 1})
    if empty_source_validation:
        config["training"].update({"checkpoint_min_epoch": 1, "validation_every_epochs": 1})
    truth = pd.concat([truth_fixture().assign(subject_key="fit", session_id=name,
        start_ms=0, end_ms=6000, evaluable=True, valid_duration=True) for name in ("first", "second")])
    class Dataset:
        def __init__(self):
            self.session_groups = {("fit", name): pd.DataFrame({"_row_id": np.arange(index * 8, (index + 1) * 8)})
                                   for index, name in enumerate(("first", "second"))}
        training = True
        geometry = SimpleNamespace(supervised_steps=4)
        events, ignore_events = truth, truth.iloc[:0]

        def endpoint_counts(self, key, name):
            return 1, 8

        def __getitem__(self, item):
            row, _, _ = item
            batch = synthetic_batch(config, 20)
            batch["timestamp_ms"] += (row % 8 - 19) * 3000
            batch["timeline_position"] += row % 8 - 19
            batch["motion_blocks"][:, :, 6:] = 1
            for name in ("state_target", "onset_target", "offset_target"):
                batch[name] = torch.zeros(1, 20)
            for name in ("state_loss_mask", "onset_loss_mask", "offset_loss_mask", "smooth_mask", "smooth_loss_mask"):
                batch[name] = torch.ones(1, 20)
            batch["augmentation_identity"] = "fixed"
            batch["augmentation_description"] = {"view": "fixed"}
            for name in ("state", "proposal_3", "proposal_15", "proposal_60", "smooth"):
                batch[name + "_session_count"] = torch.tensor(8.)
            for name in ("onset", "offset"):
                batch[name + "_positive_count"] = torch.tensor(1.)
                batch[name + "_background_count"] = torch.tensor(8.)
            return {name: value[0] if isinstance(value, torch.Tensor) and value.ndim > 0 else value
                    for name, value in batch.items()}
    interrupted = {"enabled": False}
    def loader(dataset, sampler, configuration):
        for position in range(sampler.start_chunk, len(sampler.chunks)):
            if interrupted["enabled"] and not empty_source_validation and sampler.epoch == interruption_epoch and position == 1:
                interrupted["enabled"] = False
                raise failure("injected interruption")
            row, epoch, mask, *_ = sampler.chunks[position]
            batch = dataset[(row, epoch, mask)]
            yield {name: value[None] if isinstance(value, torch.Tensor) else value for name, value in batch.items()}
    def generated(stream, times, features):
        item = candidate(start=0, end=6000, identity=str(int(times[-1])), parent=str(int(times[-1])))
        item.subject_key, item.session_id = stream.subject, stream.session
        item.model_identity = stream.identity
        item.decision_time_ms = item.max_input_time_ms = int(times[-1])
        return [item]
    dataset = Dataset()
    assert len(SessionChunkSampler(dataset, 1).chunks) == 4
    monkeypatch.setattr(trainer, "make_joint_dataset", lambda *args, **kwargs: (Dataset(), {"training_subjects": ["fit"]}))
    monkeypatch.setattr(trainer, "session_loader", loader)
    monkeypatch.setattr(trainer, "generate_chunk", generated)
    monkeypatch.setattr(trainer, "loss_schedule", lambda epoch: {"event": 1, "quality": 1,
        "boundary": 0, "contrastive": 0, "truth_jitter": 0})
    neighbor_epochs = {deployment_epoch + offset for offset in (-4, -2, 0, 2, 4)
                       if 0 < deployment_epoch + offset <= epochs} if deployment_epoch is not None else set()
    if periodic_baseline:
        original_save = trainer.save_torch_atomic
        def save_checkpoint(path, payload):
            if "continuous" in path.parts:
                if path.parent.name == "periodic":
                    return
                if path.parent.name == "nearby" and payload["epoch"] not in neighbor_epochs:
                    return
            return original_save(path, payload)
        monkeypatch.setattr(trainer, "save_torch_atomic", save_checkpoint)
    inputs = SimpleNamespace(identity={"data": "fixture"})
    if development_mode:
        inputs.frames = lambda *args, **kwargs: (None, None,
            truth_fixture().assign(subject_key="selector", evaluable=True, valid_duration=True), None)
        monkeypatch.setattr(trainer, "fit_source_mapping", lambda *args, **kwargs: None)
        monkeypatch.setattr(trainer, "predict_joint_model", lambda model, inputs, subjects, config,
            preprocessing, directory, *args, **kwargs: {"coarse": [int(directory.name)], "windows": pd.DataFrame()})
        monkeypatch.setattr(trainer, "choose_operating_point", lambda coarse, *args, **kwargs:
            ({"f1": coarse[0] / 100, "promotion_eligible": True}, []))
    if compatible_resume:
        from bme_eating.protocol import source_identity
        current_source = source_identity(Path(config["_config_path"]).resolve().parents[1])
        training_source = {**current_source, "src/bme_eating/training/natural_replay.py": "0" * 64}
        config["_output_only_resume"] = {"policy": "test_output_only_resume",
            "execution_source": current_source, "training_source": training_source}
    subjects = {"fit"} | ({f"audit_subject_{index}" for index in range(38)} if deployment_epoch is not None else set())
    if empty_source_validation:
        from bme_eating.candidates import NoSourceCandidatesError
        def no_sources(*args, **kwargs):
            raise NoSourceCandidatesError("No legal training parent candidates")
        monkeypatch.setattr(trainer, "fit_source_mapping", no_sources)
        original_write = trainer.write_json_atomic
        def write_status(path, payload):
            if interrupted["enabled"] and "interrupted" in path.parts and path.name == "validation_status.json":
                interrupted["enabled"] = False
                raise failure("injected validation interruption")
            return original_write(path, payload)
        monkeypatch.setattr(trainer, "write_json_atomic", write_status)
    selector = {"selector"} if empty_source_validation or development_mode else None
    def train(directory, resume):
        if empty_source_validation:
            with pytest.raises(RuntimeError, match="selection failed"):
                trainer.train_joint_model(inputs, subjects, config, directory, "identity", epochs=epochs,
                    resume=resume, selector=selector, deployment_epoch=deployment_epoch)
        else:
            trainer.train_joint_model(inputs, subjects, config, directory, "identity", epochs=epochs,
                resume=resume, selector=selector, deployment_epoch=deployment_epoch)
    train(tmp_path / "continuous", False)
    interrupted["enabled"] = True
    with pytest.raises(failure):
        trainer.train_joint_model(inputs, subjects, config, tmp_path / "interrupted", "identity",
                                  epochs=epochs, resume=False, selector=selector, deployment_epoch=deployment_epoch)
    saved = torch.load(tmp_path / "interrupted/resume.pt", weights_only=False)
    if empty_source_validation:
        assert saved["state"]["epoch_complete"]
        assert saved["state"]["next_chunk"] == 4
    else:
        assert saved["state"]["natural_replay"]["cursor"] > 0
        assert saved["state"]["positive_gradients"]
    assert saved["state"]["epoch"] == interruption_epoch
    train(tmp_path / "interrupted", True)
    continuous = torch.load(tmp_path / "continuous/latest.pt", weights_only=True)
    resumed = torch.load(tmp_path / "interrupted/latest.pt", weights_only=True)
    for name in continuous["model"]:
        torch.testing.assert_close(continuous["model"][name], resumed["model"][name], rtol=0, atol=0)
    if deployment_epoch is not None or development_mode:
        continuous_resume = torch.load(tmp_path / "continuous/resume.pt", weights_only=False)
        resumed_resume = torch.load(tmp_path / "interrupted/resume.pt", weights_only=False)
        for parameter, values in continuous_resume["optimizer"]["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(value, resumed_resume["optimizer"]["state"][parameter][name], rtol=0, atol=0)
        assert continuous_resume["python_rng"] == resumed_resume["python_rng"]
        np.testing.assert_equal(continuous_resume["numpy_rng"], resumed_resume["numpy_rng"])
        torch.testing.assert_close(continuous_resume["torch_rng"], resumed_resume["torch_rng"], rtol=0, atol=0)
    if empty_source_validation:
        resumed_state = torch.load(tmp_path / "interrupted/resume.pt", weights_only=False)["state"]
        assert resumed_state["unqualified_checks"] == 1
        assert len(resumed_state["history"]) == 1
        assert resumed_state["history"][0]["selector"]["f1"] is None
    if deployment_epoch is not None:
        assert continuous["epoch"] == resumed["epoch"] == deployment_epoch
        assert resumed_resume["state"]["training_complete"] and resumed_resume["state"]["epoch"] == epochs + 1
        assert [record["epoch"] for record in resumed_resume["state"]["history"]] == list(range(1, epochs + 1))
        for name in continuous_resume["model"]:
            torch.testing.assert_close(continuous_resume["model"][name], resumed_resume["model"][name], rtol=0, atol=0)
        for directory in ("continuous", "interrupted"):
            checkpoints = sorted((tmp_path / directory / "nearby").glob("*.pt"))
            periodic_epochs = set() if periodic_baseline and directory == "continuous" else set(range(4, epochs + 1, 4))
            expected_epochs = sorted(periodic_epochs | neighbor_epochs)
            assert [int(path.stem.split("_")[-1]) for path in checkpoints] == expected_epochs
        for retained_epoch in sorted(neighbor_epochs if periodic_baseline else set(expected_epochs)):
            filename = f"nearby/epoch_{retained_epoch:03d}.pt"
            original_checkpoint = torch.load(tmp_path / "continuous" / filename, weights_only=True)
            resumed_checkpoint = torch.load(tmp_path / "interrupted" / filename, weights_only=True)
            assert original_checkpoint["epoch"] == resumed_checkpoint["epoch"] == retained_epoch
            for name in original_checkpoint["model"]:
                torch.testing.assert_close(original_checkpoint["model"][name],
                                           resumed_checkpoint["model"][name], rtol=0, atol=0)
    else:
        assert not (tmp_path / "continuous/nearby").exists()
        assert not (tmp_path / "interrupted/nearby").exists()
    for directory in ("continuous", "interrupted"):
        checkpoints = sorted((tmp_path / directory / "periodic").glob("*.pt"))
        expected_epochs = list(range(10, epochs + 1, 4)) if development_mode else []
        if periodic_baseline and directory == "continuous":
            expected_epochs = []
        assert [int(path.stem.split("_")[-1]) for path in checkpoints] == expected_epochs
        for path, retained_epoch in zip(checkpoints, expected_epochs):
            payload = torch.load(path, weights_only=True)
            assert payload["epoch"] == retained_epoch and payload["epoch_complete"]
            assert payload["excluded_subjects"] == ["selector"]
    if development_mode:
        for directory in ("continuous", "interrupted"):
            selected = torch.load(tmp_path / directory / "selected.pt", weights_only=True)
            assert selected["epoch"] == epochs
            assert [record["epoch"] for record in resumed_resume["state"]["history"]
                    if "selector" in record] == list(range(8, epochs + 1, 2))
    if compatible_resume:
        assert resumed_resume["state"]["resume_migration"]["policy"] == "test_output_only_resume"


@pytest.mark.parametrize("failure", [KeyboardInterrupt, torch.OutOfMemoryError])
def test_development_periodic_checkpoints_quiet_replay_and_compatible_resume(config, tmp_path, monkeypatch, failure, capsys):
    test_replay_and_contribution_chunk_resume(config, tmp_path, monkeypatch, failure,
        epochs=14, interruption_epoch=10, development_mode=True, periodic_baseline=True,
        compatible_resume=True)
    captured = capsys.readouterr()
    assert "Natural replay" not in captured.out + captured.err


@pytest.mark.parametrize("failure", [KeyboardInterrupt, torch.OutOfMemoryError])
def test_final39_periodic_checkpoints_preserve_resume_and_deployment(config, tmp_path, monkeypatch, failure):
    test_replay_and_contribution_chunk_resume(config, tmp_path, monkeypatch, failure,
        epochs=16, deployment_epoch=12, interruption_epoch=4, periodic_baseline=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("failure", [KeyboardInterrupt, torch.cuda.OutOfMemoryError])
def test_cuda_replay_and_contribution_chunk_resume(config, tmp_path, monkeypatch, failure):
    config["training"]["device"] = "cuda"
    test_replay_and_contribution_chunk_resume(config, tmp_path, monkeypatch, failure)
