from __future__ import annotations

import builtins
import importlib
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from bme_eating.calibration_v4 import PlattCalibration
from bme_eating.data.stats_fusion_sequence import ClipMixtureSampler
from bme_eating.hierarchical_v4_pipeline import HierarchicalEatingDetectorV4
from bme_eating.models.endpoint_refiner import (
    apply_boundary_refinement,
    augment_boundary_training_proposals,
    select_boundary_range,
    truncated_gaussian_target,
)
from bme_eating.models.event_verifier_v4 import (
    HardNegativeBatchSampler,
    build_proposal_features_v4,
    normalized_proposal_weights,
)
from bme_eating.models.stats_fusion_loss import (
    StatsFusionStateLoss,
    one_sided_transition_targets,
)
from bme_eating.models.stats_fusion_state import StatsFusionStateModel
from bme_eating.proposals_v4 import (
    ALLOWED_SOURCE_MASK,
    ProposalSource,
    _jitter,
    generate_event_candidates_v4,
)
from bme_eating.stats_features import STATS_FEATURE_COLUMNS, FoldRobustScaler
from bme_eating.structured_decoder import (
    FixedLagSemiMarkovDecoder,
    TruncatedLogNormalDurationPrior,
)
from bme_eating.training.hierarchical_v4_trainer import (
    UNLABELED_ANCHOR_COLUMNS,
    _candidate_recall,
    _hand_metrics,
    _matching_ranking_reversal,
    load_v4_inputs,
)


def _model_config() -> dict[str, object]:
    return {
        "motion_block_seconds": 3,
        "ppg_block_seconds": 15,
        "motion_dilations": [1, 2, 4, 8, 16, 32],
        "ppg_dilations": [1, 2, 4, 8],
        "statistics_dilations": [1, 2],
        "long_dilations": [1, 2, 4, 8, 16, 32],
        "motion_embedding_dim": 8,
        "ppg_embedding_dim": 8,
        "hidden_dim": 8,
        "statistics_dim": 8,
        "long_pool_factor": 5,
        "dropout": 0.0,
        "gate_initial_bias": -2.0,
        "future_context_seconds": 0,
        "stable_feature_columns": list(STATS_FEATURE_COLUMNS),
    }


def _state_batch(steps: int = 20) -> dict[str, torch.Tensor]:
    ppg_steps = steps // 5
    mapping = torch.div(torch.arange(steps) + 1, 5, rounding_mode="floor") - 1
    return {
        "motion_blocks": torch.randn(1, steps, 12, 300),
        "motion_valid": torch.ones(1, steps),
        "ppg_blocks": torch.randn(1, ppg_steps, 2, 750),
        "ppg_quality": torch.randn(1, ppg_steps, 8),
        "ppg_valid": torch.ones(1, ppg_steps),
        "ppg_to_motion_index": mapping.unsqueeze(0),
        "statistics": torch.randn(1, steps, 24),
    }


def test_stats_scaler_handles_missing_extreme_and_zero_iqr() -> None:
    frame = pd.DataFrame({"subject_key": ["a", "a", "b"]})
    for index, column in enumerate(STATS_FEATURE_COLUMNS):
        frame[column] = [1.0, np.inf if index == 0 else 1.0, 1e9]
    scaler = FoldRobustScaler.fit(frame, training_subjects={"a"})
    transformed = scaler.transform_array(frame[list(STATS_FEATURE_COLUMNS)].to_numpy())
    assert transformed.shape == (3, 24)
    assert np.isfinite(transformed).all()
    assert np.max(np.abs(transformed[:, :12])) <= 8.0
    assert transformed[1, 12] == 1.0
    scaler.assert_unseen({"b"})
    with pytest.raises(RuntimeError, match="leakage"):
        scaler.assert_unseen({"a"})


def test_outer_training_never_loads_outer_anchor_labels(tmp_path, monkeypatch) -> None:
    input_root = tmp_path / "v2"
    (input_root / "indices").mkdir(parents=True)
    (input_root / "features").mkdir()
    anchors = pd.DataFrame(
        {
            "segment_id": ["g0", "g1"],
            "session_id": ["d0", "d1"],
            "segment_path": ["p0", "p1"],
            "subject_key": ["train", "outer"],
            "timestamp_ms": [0, 0],
            "state_target": [1.0, 99.0],
            "state_loss_mask": [1.0, 99.0],
            "start_target": [0.0, 99.0],
            "end_target": [0.0, 99.0],
            "start_loss_mask": [1.0, 99.0],
            "end_loss_mask": [1.0, 99.0],
            "event_id": ["e0", "outer-secret"],
            "hand_relation": ["same", "different"],
            "motion_history_available_seconds": [0.0, 0.0],
            "ppg_history_available_seconds": [0.0, 0.0],
        }
    )
    anchors.to_parquet(input_root / "indices" / "anchors.parquet", index=False)
    pd.DataFrame(
        {
            "subject_key": ["train", "outer"],
            "event_id": ["e0", "e1"],
            "start_ms": [0, 0],
            "end_ms": [1000, 1000],
        }
    ).to_parquet(input_root / "indices" / "events.parquet", index=False)
    pd.DataFrame(
        columns=["session_id", "segment_id", "segment_path", "start_ms", "end_ms"]
    ).to_parquet(input_root / "indices" / "segments.parquet", index=False)
    statistics = anchors[["segment_id", "session_id", "subject_key", "timestamp_ms"]].copy()
    for column in STATS_FEATURE_COLUMNS:
        statistics[column] = 0.0
    statistics.to_parquet(input_root / "features" / "baseline.parquet", index=False)
    (input_root / "indices" / "subject_folds.json").write_text(
        json.dumps({"train": 1, "outer": 0}), encoding="utf-8"
    )
    original = pd.read_parquet
    anchor_reads: list[dict[str, object]] = []

    def tracked(path, *args, **kwargs):
        if str(path).endswith("anchors.parquet"):
            anchor_reads.append(dict(kwargs))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", tracked)
    inputs = load_v4_inputs(
        {"features": {"artifact_name": "baseline"}},
        input_root,
        fold=0,
        event_role="outer_train",
    )
    outer = inputs.anchors[inputs.anchors["subject_key"] == "outer"].iloc[0]
    assert outer.state_target == 0.0
    assert pd.isna(outer.event_id)
    assert any(read.get("columns") == UNLABELED_ANCHOR_COLUMNS for read in anchor_reads)
    assert set(inputs.events["subject_key"]) == {"train"}
    with pytest.raises(RuntimeError, match="Outer-test labels require"):
        load_v4_inputs(
            {"features": {"artifact_name": "baseline"}},
            input_root,
            fold=0,
            event_role="outer_test",
        )
    evaluation_inputs = load_v4_inputs(
        {"features": {"artifact_name": "baseline"}},
        input_root,
        fold=0,
        event_role="outer_test",
        allow_outer_labels=True,
    )
    assert evaluation_inputs.anchors.iloc[0].state_target == 99.0
    assert set(evaluation_inputs.events["subject_key"]) == {"outer"}


def test_statsfusion_receptive_fields_shapes_and_causality() -> None:
    torch.manual_seed(7)
    model = StatsFusionStateModel(_model_config()).eval()
    assert model.short_receptive_field_seconds == 381
    assert model.long_receptive_field_seconds == 1905
    batch = _state_batch()
    with torch.no_grad():
        original = model(batch)
    assert original["state_logit"].shape == (1, 20)
    changed = {key: value.clone() for key, value in batch.items()}
    changed["motion_blocks"][:, 11:] += 100.0
    changed["statistics"][:, 11:] -= 100.0
    changed["ppg_blocks"][:, 2:] += 100.0
    with torch.no_grad():
        future_changed = model(changed)
    torch.testing.assert_close(
        original["state_logit"][:, :11], future_changed["state_logit"][:, :11]
    )


def test_ppg_all_missing_uses_finite_fallback_and_zero_gate() -> None:
    model = StatsFusionStateModel(_model_config()).eval()
    batch = _state_batch()
    batch["ppg_blocks"].zero_()
    batch["ppg_quality"].zero_()
    batch["ppg_valid"].zero_()
    with torch.no_grad():
        output = model(batch)
    assert torch.isfinite(output["state_logit"]).all()
    assert torch.count_nonzero(output["ppg_gate"]) == 0


@pytest.mark.parametrize("disabled", ["motion", "ppg"])
def test_disabled_modality_cannot_leak_values_or_validity(disabled: str) -> None:
    config = _model_config()
    config[f"use_{disabled}"] = False
    model = StatsFusionStateModel(config).eval()
    batch = _state_batch()
    changed = {key: value.clone() for key, value in batch.items()}
    if disabled == "motion":
        changed["motion_blocks"] += 100.0
        changed["motion_valid"].zero_()
    else:
        changed["ppg_blocks"] -= 100.0
        changed["ppg_quality"] += 100.0
        changed["ppg_valid"].zero_()
    with torch.no_grad():
        original = model(batch)["state_logit"]
        modified = model(changed)["state_logit"]
    torch.testing.assert_close(original, modified)


def test_one_sided_targets_and_final_logit_smoothing() -> None:
    timestamps = torch.tensor([-3000, 0, 3000, 30_000, 60_000, 63_000])
    onset, offset = one_sided_transition_targets(
        timestamps, torch.tensor([0]), torch.tensor([60_000])
    )
    assert onset[0] == 0 and onset[1] == 1
    assert offset[4] == 1 and offset[3] == 0
    logits = torch.tensor([[0.0, 2.0, -2.0]], requires_grad=True)
    output = {
        "state_logit": logits,
        "onset_logit": torch.zeros_like(logits, requires_grad=True),
        "offset_logit": torch.zeros_like(logits, requires_grad=True),
    }
    batch = {
        "state_target": torch.zeros_like(logits),
        "onset_target": torch.zeros_like(logits),
        "offset_target": torch.zeros_like(logits),
        "supervision_mask": torch.ones_like(logits),
        "state_loss_mask": torch.ones_like(logits),
        "smooth_mask": torch.ones_like(logits),
    }
    loss, components = StatsFusionStateLoss(
        smooth_weight=1.0, smooth_tau=100.0, boundary_weight=0.1
    )(output, batch)
    loss.backward()
    assert components["smooth"] > 0
    assert logits.grad is not None and torch.count_nonzero(logits.grad) > 0


def test_clip_sampler_has_importance_weights() -> None:
    anchors = pd.DataFrame(
        {
            "state_target": [0.0, 1.0, 0.0, 1.0],
            "start_target": [0.0, 1.0, 0.0, 0.0],
            "end_target": [0.0, 0.0, 0.0, 1.0],
            "state_loss_mask": [1.0] * 4,
        }
    )
    sampler = ClipMixtureSampler(
        anchors,
        samples_per_epoch=20,
        mixture={"uniform": 0.5, "event": 0.25, "boundary": 0.25},
        seed=1,
    )
    samples = list(sampler)
    assert len(samples) == 20
    assert all(np.isfinite(weight) and weight > 0 for _, _, weight in samples)


def test_fixed_lag_decoder_and_candidates_have_only_allowed_sources() -> None:
    prior = TruncatedLogNormalDurationPrior.fit(np.asarray([60.0, 120.0, 180.0]))
    decoder = FixedLagSemiMarkovDecoder(prior, fixed_lag_seconds=60)
    with pytest.raises(ValueError, match="60-second"):
        FixedLagSemiMarkovDecoder(prior, fixed_lag_seconds=75)
    timestamps = np.arange(0, 600_000, 3000, dtype=np.int64)
    state = np.zeros(len(timestamps), dtype=float)
    state[20:80] = 0.9
    windows = pd.DataFrame(
        {
            "subject_key": "s",
            "session_id": "d",
            "timestamp_ms": timestamps,
            "state_probability": state,
            "onset_probability": np.where(np.arange(len(state)) == 20, 0.9, 0.0),
            "offset_probability": np.where(np.arange(len(state)) == 80, 0.9, 0.0),
        }
    )
    config = {
        "ema_half_life_seconds": 12,
        "high_threshold": 0.5,
        "low_threshold": 0.2,
        "gap_merge_seconds": 60,
        "transition_threshold": 0.5,
        "grid_seconds": 15,
        "jitter_seconds": [-15, 0, 15],
        "maximum_variants_per_event": 6,
        "maximum_candidates_per_hour": 20,
        "deduplication_iou": 0.9,
    }
    proposals = generate_event_candidates_v4(windows, decoder, config, split_role="test")
    assert len(proposals)
    assert all((int(mask) & ~ALLOWED_SOURCE_MASK) == 0 for mask in proposals["source_mask"])
    without_structured = generate_event_candidates_v4(
        windows,
        decoder,
        {**config, "use_semi_markov": False},
        split_role="test",
    )
    assert not any(
        int(mask) & int(ProposalSource.SEMI_MARKOV)
        for mask in without_structured["source_mask"]
    )


def test_jitter_variants_preserve_proposal_family() -> None:
    family_id = "coarse-family"
    variants = _jitter(
        [(60_000, 180_000, 0.8, int(ProposalSource.HYSTERESIS), family_id)],
        [-15, 0, 15],
        maximum_variants=6,
        minimum_ms=15_000,
        maximum_ms=600_000,
    )
    assert len(variants) == 6
    assert {variant[4] for variant in variants} == {family_id}
    assert any(variant[3] & int(ProposalSource.JITTER) for variant in variants)


def test_candidate_budget_is_applied_across_fragmented_subject_sessions() -> None:
    prior = TruncatedLogNormalDurationPrior.fit(np.asarray([60.0, 120.0]))
    decoder = FixedLagSemiMarkovDecoder(prior, fixed_lag_seconds=60)
    frames = []
    for session in range(20):
        timestamps = np.arange(20, dtype=np.int64) * 3000 + session * 1_000_000
        frames.append(
            pd.DataFrame(
                {
                    "subject_key": "s",
                    "session_id": f"d{session}",
                    "timestamp_ms": timestamps,
                    "state_probability": 0.9,
                    "onset_probability": 0.0,
                    "offset_probability": 0.0,
                }
            )
        )
    proposals = generate_event_candidates_v4(
        pd.concat(frames, ignore_index=True),
        decoder,
        {
            "use_semi_markov": False,
            "ema_half_life_seconds": 12,
            "high_threshold": 0.5,
            "low_threshold": 0.2,
            "gap_merge_seconds": 0,
            "transition_threshold": 0.5,
            "grid_seconds": 15,
            "jitter_seconds": [0],
            "maximum_variants_per_event": 1,
            "maximum_candidates_per_hour": 20,
            "deduplication_iou": 0.9,
        },
        split_role="test",
    )
    assert len(proposals) <= 7


def test_candidate_recall_counts_each_recalled_truth_event() -> None:
    proposals = pd.DataFrame(
        {
            "subject_key": ["s"],
            "coarse_start_ms": [0],
            "coarse_end_ms": [200_000],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "subject_key": ["s", "s"],
            "start_ms": [0, 100_000],
            "end_ms": [100_000, 200_000],
            "hand_relation": ["same", "different"],
            "evaluable": [True, True],
        }
    )
    metrics = _candidate_recall(proposals, events)
    assert metrics["candidate_recall"] == 1.0
    assert metrics["same_candidate_recall"] == 1.0
    assert metrics["different_candidate_recall"] == 1.0


def test_hand_recall_does_not_reuse_one_prediction_for_two_truth_events() -> None:
    truth = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "subject_key": ["s", "s"],
            "start_ms": [0, 100_000],
            "end_ms": [100_000, 200_000],
            "hand_relation": ["same", "different"],
        }
    )
    prediction = pd.DataFrame(
        {
            "proposal_id": ["p"],
            "subject_key": ["s"],
            "start_ms": [0],
            "end_ms": [200_000],
        }
    )
    metrics = _hand_metrics(truth, prediction)
    assert metrics["same_sensitivity"] + metrics["different_sensitivity"] == 1.0


def test_proposal_pooling_uses_all_rows_and_masks_empty_bins() -> None:
    proposals = pd.DataFrame(
        [
            {
                "proposal_id": "p",
                "subject_key": "s",
                "session_id": "d",
                "coarse_start_ms": 0,
                "coarse_end_ms": 12_000,
                "source_mask": 1,
                "generator_score": 0.8,
            }
        ]
    )
    windows = pd.DataFrame(
        {
            "subject_key": "s",
            "session_id": "d",
            "timestamp_ms": [0, 3000, 6000, 9000],
            "state_probability": [1.0, 3.0, 5.0, 7.0],
            "onset_probability": 0.0,
            "offset_probability": 0.0,
            "ppg_gate": 1.0,
            "missing_fraction": 0.0,
            "stat": [1.0, 2.0, 3.0, 4.0],
        }
    )
    config = {
        "left_context_seconds": 60,
        "right_context_seconds": 60,
        "left_bins": 4,
        "event_bins": 2,
        "right_bins": 4,
    }
    features = build_proposal_features_v4(proposals, windows, ["stat"], config)
    assert features.sequence[0, 4, 0] == 2.0
    assert not features.sequence_mask[0, 0]
    assert features.sequence_mask[0, 4]


def test_verifier_sampling_and_group_weights() -> None:
    categories = np.asarray(
        ["positive", "near_miss", "hard_false_positive", "random_background"]
    )
    sampler = HardNegativeBatchSampler(
        categories,
        batch_size=20,
        ratios={
            "positive": 0.30,
            "near_miss": 0.25,
            "hard_false_positive": 0.30,
            "random_background": 0.15,
        },
        steps_per_epoch=1,
        seed=1,
    )
    sampled = categories[next(iter(sampler))]
    assert {name: int(np.count_nonzero(sampled == name)) for name in set(categories)} == {
        "positive": 6,
        "near_miss": 5,
        "hard_false_positive": 6,
        "random_background": 3,
    }
    frame = pd.DataFrame(
        {
            "subject_key": ["a", "a", "b"],
            "proposal_id": ["p1", "p2", "p3"],
            "proposal_family_id": ["f", "f", "g"],
            "matched_event_id": ["", "", ""],
            "max_iou": [0.0, 0.0, 0.0],
        }
    )
    weights = normalized_proposal_weights(frame)
    family_totals = frame.assign(weight=weights).groupby("proposal_family_id")["weight"].sum()
    assert np.isclose(family_totals["f"], family_totals["g"])


def test_verifier_missing_category_redistributes_in_declared_order() -> None:
    categories = np.asarray(["near_miss", "hard_false_positive", "random_background"])
    sampler = HardNegativeBatchSampler(
        categories,
        batch_size=20,
        ratios={
            "positive": 0.30,
            "near_miss": 0.25,
            "hard_false_positive": 0.30,
            "random_background": 0.15,
        },
        steps_per_epoch=1,
        seed=1,
    )
    assert sampler.counts == {
        "positive": 0,
        "near_miss": 11,
        "hard_false_positive": 6,
        "random_background": 3,
    }


def test_boundary_range_targets_and_identity_constraints() -> None:
    boundary_range = select_boundary_range(
        np.asarray([-40.0, 20.0, 70.0]), np.asarray([30.0, -80.0, 10.0])
    )
    assert 60 <= boundary_range.start_seconds <= 300
    target = truncated_gaussian_target(np.arange(-60, 61, 3), 12.0, 9.0)
    assert np.isclose(target.sum(), 1.0)
    accepted = pd.DataFrame(
        [
            {
                "proposal_id": "p",
                "subject_key": "s",
                "session_id": "d",
                "coarse_start_ms": 1000,
                "coarse_end_ms": 2000,
            }
        ]
    )
    refined = apply_boundary_refinement(
        accepted,
        np.asarray([10.0]),
        np.asarray([-10.0]),
        np.asarray([0.1]),
        np.asarray([0.1]),
        entropy_threshold=0.5,
        safety_gap_seconds=3,
    )
    assert list(refined["proposal_id"]) == ["p"]
    assert refined.iloc[0].refined_start_ms < refined.iloc[0].refined_end_ms


def test_boundary_jitter_is_limited_and_normalized_per_true_event() -> None:
    proposals = pd.DataFrame(
        {
            "proposal_id": ["worse", "best"],
            "subject_key": ["s", "s"],
            "session_id": ["d", "d"],
            "coarse_start_ms": [60_000, 63_000],
            "coarse_end_ms": [180_000, 177_000],
            "matched_event_id": ["event", "event"],
            "truth_start_ms": [65_000, 65_000],
            "truth_end_ms": [175_000, 175_000],
            "max_iou": [0.7, 0.9],
        }
    )
    augmented = augment_boundary_training_proposals(proposals)
    assert len(augmented) == 5
    assert set(augmented["proposal_id"]) == {"best"}
    assert np.isclose(augmented["sample_weight"].sum(), 1.0)


def test_boundary_refinement_enforces_neighbor_safety_gap() -> None:
    accepted = pd.DataFrame(
        {
            "proposal_id": ["left", "right"],
            "subject_key": ["s", "s"],
            "session_id": ["d", "d"],
            "coarse_start_ms": [0, 9_000],
            "coarse_end_ms": [10_000, 20_000],
        }
    )
    refined = apply_boundary_refinement(
        accepted,
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        entropy_threshold=0.5,
        safety_gap_seconds=3,
    ).sort_values("refined_start_ms")
    assert refined.iloc[1].refined_start_ms - refined.iloc[0].refined_end_ms >= 3000
    assert set(refined["proposal_id"]) == {"left", "right"}


def test_v4_pipeline_import_does_not_import_xgboost(monkeypatch) -> None:
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "xgboost" or name.startswith("xgboost."):
            raise AssertionError("v4 attempted to import xgboost")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    sys.modules.pop("bme_eating.hierarchical_v4_pipeline", None)
    importlib.import_module("bme_eating.hierarchical_v4_pipeline")


def test_matching_method_ranking_reversal_is_blocked_only_when_material() -> None:
    assert _matching_ranking_reversal(0.02, -0.03, tolerance=0.005)
    assert not _matching_ranking_reversal(0.002, -0.03, tolerance=0.005)
    assert not _matching_ranking_reversal(0.02, 0.01, tolerance=0.005)


def test_final_state_ensemble_averages_only_logits() -> None:
    class Stub(torch.nn.Module):
        def __init__(self, logit: float, gate: float) -> None:
            super().__init__()
            self.logit = logit
            self.gate = gate

        def forward(self, batch):
            shape = batch["statistics"].shape[:2]
            logit = torch.full(shape, self.logit)
            gate = torch.full(shape, self.gate)
            return {
                "state_logit": logit,
                "onset_logit": logit,
                "offset_logit": logit,
                "ppg_gate": gate,
                "statistics_gate": gate,
                "missing_fraction": gate,
            }

    detector = object.__new__(HierarchicalEatingDetectorV4)
    detector.device = torch.device("cpu")
    detector.state_models = [Stub(2.0, 0.1), Stub(4.0, 0.9)]
    detector.state_calibration = PlattCalibration(1.0, 0.0)
    detector.statistics_columns = [f"stat_{name}" for name in STATS_FEATURE_COLUMNS]
    frame = detector.predict_state_sequence(
        {
            "timestamp_ms": torch.tensor([[0, 3000]]),
            "statistics": torch.zeros(1, 2, 24),
        },
        subject_key="s",
        session_id="d",
    )
    assert np.allclose(frame["state_logit"], 3.0)
    assert np.allclose(frame["ppg_gate"], 0.1)
    assert np.allclose(frame["statistics_gate"], 0.1)
