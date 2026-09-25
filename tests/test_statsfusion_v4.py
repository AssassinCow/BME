from __future__ import annotations

import builtins
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from bme_eating.calibration_v4 import (
    PlattCalibration,
    binary_state_targets,
    state_calibration_metrics,
    subject_crossfit_platt,
)
from bme_eating.data.deep_dataset import Normalization
from bme_eating.data.labels import build_statsfusion_session_anchor_index
from bme_eating.data.stats_fusion_preprocess import (
    RawSessionInput,
    StatsFusionRawSessionPreprocessor,
    build_motion_blocks,
    causal_completed_block_layout,
    session_right_endpoint_grid,
)
from bme_eating.data.stats_fusion_sequence import (
    ClipMixtureSampler,
    SequenceGeometry,
    StatsFusionSequenceDataset,
)
from bme_eating.hierarchical_artifacts import sha256_file
from bme_eating.hierarchical_v4_pipeline import HierarchicalEatingDetectorV4
from bme_eating.metrics import (
    evaluate_events,
    evaluation_event_partition_summary,
    partition_evaluation_events,
)
from bme_eating.models.endpoint_refiner import (
    EndpointNetwork,
    apply_boundary_refinement,
    augment_boundary_training_proposals,
    endpoint_loss,
    local_soft_argmax,
    select_boundary_range,
    truncated_gaussian_target,
)
from bme_eating.models.event_verifier_v4 import (
    EventVerifierV4,
    HardNegativeBatchSampler,
    ProposalFeatureBatchV4,
    build_proposal_features_v4,
    normalized_proposal_weights,
)
from bme_eating.models.stats_fusion_loss import (
    StatsFusionStateLoss,
    one_sided_transition_targets,
)
from bme_eating.models.stats_fusion_state import CausalCompletedBlockPool, StatsFusionStateModel
from bme_eating.proposals import exclude_ignored_candidates, label_event_candidates
from bme_eating.proposals_v4 import (
    ALLOWED_SOURCE_MASK,
    ProposalSource,
    _hysteresis,
    _jitter,
    generate_event_candidates_v4,
)
from bme_eating.stats_features import STATS_FEATURE_COLUMNS, FoldRobustScaler
from bme_eating.structured_decoder import (
    FixedLagSemiMarkovDecoder,
    TruncatedLogNormalDurationPrior,
    right_endpoint_run_to_interval,
)
from bme_eating.training.hierarchical_v4_trainer import (
    UNLABELED_ANCHOR_COLUMNS,
    V4Inputs,
    _add_robust_epoch_metrics,
    _assert_nested_lineage,
    _assert_proposal_feature_alignment,
    _build_seeded_state_model,
    _candidate_recall,
    _evaluable_state_calibration_rows,
    _gap_aware_nms,
    _hand_metrics,
    _mask_ignored_state_rows,
    _matching_ranking_reversal,
    _nested_cache_artifacts,
    _nested_cache_key,
    _prepare_nested_meta_cache,
    _selector_early_stopping_improved,
    _selector_split,
    _truth_event_durations,
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
        "long_block_end_indices": torch.arange(4, steps, 5).unsqueeze(0),
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


def test_soft_state_targets_are_binarized_for_selection_and_calibration() -> None:
    targets = np.array([0.0, 0.005, 0.75, 1.0], dtype=np.float64)
    logits = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
    binary = binary_state_targets(targets)
    assert binary.tolist() == [0, 1, 1, 1]

    calibrator = PlattCalibration.fit(logits, targets)
    calibrated = calibrator.transform(logits)
    metrics = state_calibration_metrics(
        targets,
        logits,
        calibrated,
        low_threshold=0.15,
    )
    assert np.isfinite(calibrated).all()
    assert metrics["prevalence"] == pytest.approx(0.75)
    assert np.isfinite(metrics["brier"])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        binary_state_targets(np.array([0.0, 1.1]))


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


def test_r2_loads_session_statistics_without_segment_id(tmp_path, monkeypatch) -> None:
    import bme_eating.training.hierarchical_v4_trainer as trainer

    input_root = tmp_path / "v2"
    output_root = tmp_path / "v4"
    (input_root / "indices").mkdir(parents=True)
    canonical_root = output_root / "canonical_input"
    canonical_root.mkdir(parents=True)
    anchors = pd.DataFrame(
        {
            "segment_id": ["g0", "g1"],
            "session_id": ["d0", "d1"],
            "segment_path": ["p0", "p1"],
            "subject_key": ["train", "outer"],
            "timestamp_ms": [3_000, 3_000],
            "state_target": [1.0, 0.0],
            "state_loss_mask": [1.0, 1.0],
            "start_target": [0.0, 0.0],
            "end_target": [0.0, 0.0],
            "start_loss_mask": [1.0, 1.0],
            "end_loss_mask": [1.0, 1.0],
            "event_id": ["e0", None],
            "hand_relation": ["same", "different"],
            "motion_history_available_seconds": [0.0, 0.0],
            "ppg_history_available_seconds": [0.0, 0.0],
        }
    )
    anchors_path = canonical_root / "anchors.parquet"
    anchors.to_parquet(anchors_path, index=False)
    statistics = anchors[["subject_key", "session_id", "timestamp_ms"]].copy()
    for column in STATS_FEATURE_COLUMNS:
        statistics[column] = 0.0
    statistics_path = canonical_root / "statistics.parquet"
    statistics.to_parquet(statistics_path, index=False)
    pd.DataFrame(
        columns=["session_id", "segment_id", "segment_path", "start_ms", "end_ms"]
    ).to_parquet(input_root / "indices" / "segments.parquet", index=False)
    pd.DataFrame(
        {
            "subject_key": ["train"],
            "event_id": ["e0"],
            "start_ms": [0],
            "end_ms": [3_000],
        }
    ).to_parquet(input_root / "indices" / "events.parquet", index=False)
    (input_root / "indices" / "subject_folds.json").write_text(
        json.dumps({"train": 1, "outer": 0}), encoding="utf-8"
    )
    monkeypatch.setattr(trainer, "verify_canonical_statsfusion_inputs", lambda *_args: {})
    monkeypatch.setattr(
        trainer,
        "canonical_input_paths",
        lambda _root: {"anchors": anchors_path, "statistics": statistics_path},
    )

    inputs = load_v4_inputs(
        {
            "experiment": {"protocol_version": "statsfusion-r2"},
            "project": {"artifact_schema_version": "v4"},
            "features": {"artifact_name": "baseline"},
        },
        input_root,
        fold=0,
        event_role="outer_train",
    )

    assert "segment_id" not in inputs.statistics.columns
    assert inputs.statistics.columns[:3].tolist() == [
        "subject_key",
        "session_id",
        "timestamp_ms",
    ]


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
    assert all(np.isfinite(weight).all() and np.any(weight > 0) for _, _, weight in samples)


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
        int(mask) & int(ProposalSource.SEMI_MARKOV) for mask in without_structured["source_mask"]
    )


def test_jitter_variants_preserve_proposal_family() -> None:
    family_id = "coarse-family"
    variants = _jitter(
        [(60_000, 180_000, 0.8, int(ProposalSource.HYSTERESIS), family_id)],
        [-15, 0, 15],
        maximum_variants=6,
        minimum_ms=15_000,
        maximum_ms=600_000,
        observation_start_ms=0,
        observation_end_ms=300_000,
    )
    assert len(variants) == 6
    assert {variant[4] for variant in variants} == {family_id}
    assert any(variant[3] & int(ProposalSource.JITTER) for variant in variants)


def test_jitter_is_clipped_to_observed_session_bounds() -> None:
    variants = _jitter(
        [(0, 60_000, 0.8, int(ProposalSource.HYSTERESIS), "family")],
        [-30, 0, 30],
        maximum_variants=9,
        minimum_ms=15_000,
        maximum_ms=120_000,
        observation_start_ms=0,
        observation_end_ms=90_000,
    )
    assert variants
    assert min(value[0] for value in variants) == 0
    assert max(value[1] for value in variants) <= 90_000


def test_candidate_budget_is_independent_per_session() -> None:
    prior = TruncatedLogNormalDurationPrior.fit(np.asarray([60.0, 120.0]))
    decoder = FixedLagSemiMarkovDecoder(prior, fixed_lag_seconds=60)
    frames = []
    for session in range(20):
        timestamps = np.arange(21, dtype=np.int64) * 3000 + session * 1_000_000
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
    assert len(proposals) == 20
    assert proposals.groupby("session_id").size().eq(1).all()


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
    categories = np.asarray(["positive", "near_miss", "hard_false_positive", "random_background"])
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
    sampled_batch = next(iter(sampler))
    sampled = categories[[index for index, _ in sampled_batch]]
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


def test_masked_boundary_loss_and_valid_entropy_are_finite() -> None:
    start_logit = torch.tensor([[1.0, -torch.inf, 0.0]], requires_grad=True)
    end_logit = torch.tensor([[-torch.inf, 2.0, -torch.inf]], requires_grad=True)
    batch = {
        "start_mask": torch.tensor([[True, False, True]]),
        "end_mask": torch.tensor([[False, True, False]]),
        "start_target": torch.tensor([[0.75, 0.0, 0.25]]),
        "end_target": torch.tensor([[0.0, 1.0, 0.0]]),
        "sample_weight": torch.ones(1),
        "start_weight": torch.ones(1),
        "end_weight": torch.ones(1),
    }
    loss, _ = endpoint_loss({"start_logit": start_logit, "end_logit": end_logit}, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(start_logit.grad).all()
    offset, entropy = local_soft_argmax(
        end_logit.detach(), torch.tensor([-3.0, 0.0, 3.0]), valid_mask=batch["end_mask"]
    )
    assert offset.item() == 0.0
    assert entropy.item() == 1.0


def test_masked_temporal_models_ignore_invalid_bin_values() -> None:
    torch.manual_seed(20260925)
    mask = torch.tensor([[False, True, True, True, False]])
    sequence = torch.randn(1, 5, 4)
    changed = sequence.clone()
    changed[:, ~mask[0], :] += 1000.0
    verifier = EventVerifierV4(
        4, 2, {"hidden_channels": 8, "hidden_dim": 16, "dropout": 0.0}
    ).eval()
    scalar = torch.zeros(1, 2)
    with torch.no_grad():
        first = verifier({"sequence": sequence, "sequence_mask": mask, "scalar": scalar})
        second = verifier({"sequence": changed, "sequence_mask": mask, "scalar": scalar})
    assert torch.allclose(first["event_logit"], second["event_logit"], atol=1e-7, rtol=0)
    assert torch.allclose(first["iou_logit"], second["iou_logit"], atol=1e-7, rtol=0)

    endpoint = EndpointNetwork(4, 8, 0.0).eval()
    with torch.no_grad():
        first_endpoint = endpoint(sequence, mask)
        second_endpoint = endpoint(changed, mask)
    assert torch.allclose(
        first_endpoint[mask], second_endpoint[mask], atol=1e-7, rtol=0
    )


def test_all_invalid_endpoint_forces_finite_fallback_decode() -> None:
    network = EndpointNetwork(4, 8, 0.0).eval()
    mask = torch.zeros(1, 5, dtype=torch.bool)
    with torch.no_grad():
        logits = network(torch.randn(1, 5, 4), mask)
    offset, entropy = local_soft_argmax(
        logits,
        torch.arange(-2.0, 3.0),
        valid_mask=mask,
    )
    assert torch.isfinite(offset).all()
    assert torch.isfinite(entropy).all()
    assert offset.item() == 0.0
    assert entropy.item() == 1.0


def test_right_endpoint_state_run_semantics() -> None:
    timestamps = np.asarray([3_000, 6_000, 9_000], dtype=np.int64)
    assert right_endpoint_run_to_interval(timestamps, 0, 1, 3_000) == (0, 3_000)
    assert right_endpoint_run_to_interval(timestamps, 1, 2, 3_000) == (3_000, 6_000)
    assert right_endpoint_run_to_interval(timestamps, 2, 3, 3_000) == (6_000, 9_000)
    assert right_endpoint_run_to_interval(timestamps, 0, 3, 3_000) == (0, 9_000)
    hysteresis = _hysteresis(
        timestamps,
        np.asarray([0.9, 0.9, 0.0]),
        high=0.5,
        low=0.2,
        step_ms=3_000,
    )
    assert hysteresis[0][:2] == (0, 6_000)

    class ForcedDecoder(FixedLagSemiMarkovDecoder):
        def decode_states(self, probabilities):
            return np.asarray([True, False], dtype=bool)

    prior = TruncatedLogNormalDurationPrior.fit(np.asarray([15.0, 30.0, 45.0]))
    decoder = ForcedDecoder(prior, grid_seconds=15, fixed_lag_seconds=60)
    assert decoder.decode_events(
        np.asarray([15_000, 30_000]), np.asarray([0.9, 0.1])
    )[0][:2] == (0, 15_000)
    grid = np.asarray([15_000, 30_000, 45_000], dtype=np.int64)
    assert right_endpoint_run_to_interval(grid, 0, 1, 15_000) == (0, 15_000)
    assert right_endpoint_run_to_interval(grid, 1, 3, 15_000) == (15_000, 45_000)
    assert right_endpoint_run_to_interval(grid, 0, 3, 15_000) == (0, 45_000)


def test_gap_aware_acceptance_is_deterministic_and_precedes_boundary() -> None:
    frame = pd.DataFrame(
        {
            "proposal_id": ["lower", "higher", "separate"],
            "coarse_start_ms": [0, 2_000, 20_000],
            "coarse_end_ms": [10_000, 12_000, 30_000],
            "final_score": [0.8, 0.9, 0.7],
        }
    )
    kept = _gap_aware_nms(frame, 0.95, "final_score", minimum_gap_seconds=3)
    assert kept["proposal_id"].tolist() == ["higher", "separate"]
    tied = frame.assign(final_score=[0.9, 0.9, 0.7]).sample(frac=1.0, random_state=7)
    tied_kept = _gap_aware_nms(tied, 0.95, "final_score", minimum_gap_seconds=3)
    assert tied_kept["proposal_id"].tolist() == ["lower", "separate"]


def test_high_entropy_boundary_preserves_gap_safe_coarse_endpoints() -> None:
    accepted = pd.DataFrame(
        {
            "proposal_id": ["left", "right"],
            "subject_key": ["s", "s"],
            "session_id": ["d", "d"],
            "coarse_start_ms": [0, 103_000],
            "coarse_end_ms": [100_000, 150_000],
        }
    )
    refined = apply_boundary_refinement(
        accepted,
        np.asarray([40.0, -40.0]),
        np.asarray([40.0, -40.0]),
        np.asarray([0.9, 0.9]),
        np.asarray([0.9, 0.9]),
        entropy_threshold=0.5,
        safety_gap_seconds=3,
    )
    assert refined["refined_start_ms"].tolist() == accepted["coarse_start_ms"].tolist()
    assert refined["refined_end_ms"].tolist() == accepted["coarse_end_ms"].tolist()


def test_nested_lineage_rejects_prediction_or_excluded_subjects_in_training() -> None:
    _assert_nested_lineage(
        training_subjects={"train"},
        prediction_subjects={"holdout"},
        globally_excluded_subjects={"holdout"},
    )
    with pytest.raises(RuntimeError, match="prediction subjects"):
        _assert_nested_lineage(
            training_subjects={"train", "holdout"},
            prediction_subjects={"holdout"},
            globally_excluded_subjects=set(),
        )
    with pytest.raises(RuntimeError, match="globally excluded"):
        _assert_nested_lineage(
            training_subjects={"train", "excluded"},
            prediction_subjects={"holdout"},
            globally_excluded_subjects={"excluded"},
        )


def test_nested_single_seed_epoch_cache_dependency_replay(tmp_path) -> None:
    training_subjects = {"train-a", "train-b", "train-c"}
    holdout_subjects = {"meta-holdout"}
    state_seeds = [2026]
    config = {
        "model": {"architecture": "stub"},
        "sequence": {"step_seconds": 3},
        "training": {"random_seed": 2026, "max_epochs": 1},
        "decoder": {"grid_seconds": 15},
        "final_training": {"state_seeds": state_seeds},
    }
    parent_hashes = {"outer_oof_window_logits": "parent-sha256"}
    cache_key = _nested_cache_key(
        config,
        training_subjects,
        holdout_subjects,
        parent_hashes,
    )
    root = tmp_path / "nested" / "meta_0"
    required = {
        name: root / filename
        for name, filename in {
            "train_windows": "train_window_predictions.parquet",
            "train_proposals": "train_proposals_labeled.parquet",
            "holdout_windows": "holdout_window_predictions.parquet",
            "holdout_proposals": "holdout_proposals.parquet",
            "calibration": "state_calibration.json",
            "duration_prior": "duration_prior.json",
        }.items()
    }
    for name, path in required.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    inner_models = []
    for partition, prediction_subject in enumerate(sorted(training_subjects)):
        inner_root = root / "state" / f"inner_{partition}"
        paths = {
            "checkpoint": inner_root / "seed_2026.pt",
            "selector": inner_root / "selector_seed_2026.json",
            "scaler": inner_root / "statistics_scaler.json",
            "normalization": inner_root / "sensor_normalization.json",
        }
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if name != "checkpoint":
                path.write_text(f"{name}-{partition}", encoding="utf-8")
        torch.save(
            {
                "epochs": 1,
                "parent_artifact_sha256": {
                    "statistics_scaler": sha256_file(paths["scaler"]),
                    "sensor_normalization": sha256_file(paths["normalization"]),
                },
            },
            paths["checkpoint"],
        )
        inner_models.append(
            {
                "seed": 2026,
                "selected_epoch": 1,
                "inner_partition": partition,
                "training_subjects": sorted(training_subjects - {prediction_subject}),
                "prediction_subjects": [prediction_subject],
                "globally_excluded_subjects": sorted(holdout_subjects),
                **{
                    f"{name}_path": path.relative_to(tmp_path).as_posix()
                    for name, path in paths.items()
                },
                **{f"{name}_sha256": sha256_file(path) for name, path in paths.items()},
            }
        )
    lineage_path = root / "lineage.json"
    lineage = {
        "protocol_version": "statsfusion-r2",
        "meta_crossfit_protocol": "fully_nested_v1",
        "cache_key": cache_key,
        "training_subjects": sorted(training_subjects),
        "prediction_subjects": sorted(holdout_subjects),
        "globally_excluded_subjects": sorted(holdout_subjects),
        "state_seeds": state_seeds,
        "inner_models": inner_models,
        "parent_artifact_sha256": parent_hashes,
        "artifact_sha256": {name: sha256_file(path) for name, path in required.items()},
    }
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
    artifacts = _nested_cache_artifacts(
        SimpleNamespace(root=tmp_path),
        lineage,
        required,
        lineage_path,
        training_subjects=training_subjects,
        holdout_subjects=holdout_subjects,
        state_seeds=state_seeds,
        cache_key=cache_key,
        parent_artifact_sha256=parent_hashes,
    )
    assert len(artifacts) == len(set(artifacts))
    assert all(model["selected_epoch"] == 1 for model in inner_models)
    corrupted = tmp_path / inner_models[0]["checkpoint_path"]
    corrupted.write_text("corrupted", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inner model artifact changed"):
        _nested_cache_artifacts(
            SimpleNamespace(root=tmp_path),
            lineage,
            required,
            lineage_path,
            training_subjects=training_subjects,
            holdout_subjects=holdout_subjects,
            state_seeds=state_seeds,
            cache_key=cache_key,
            parent_artifact_sha256=parent_hashes,
        )


def test_nested_single_seed_epoch_pipeline_replay(tmp_path, monkeypatch) -> None:
    training_subjects = {"train-a", "train-b", "train-c"}
    holdout_subjects = {"meta-holdout"}
    all_subjects = sorted(training_subjects | holdout_subjects)
    anchor_rows = []
    event_rows = []
    for subject in all_subjects:
        for index, timestamp in enumerate((3_000, 6_000, 9_000, 12_000)):
            anchor_rows.append(
                {
                    "segment_id": f"segment-{subject}",
                    "session_id": f"session-{subject}",
                    "subject_key": subject,
                    "timestamp_ms": timestamp,
                    "state_target": float(index % 2),
                    "state_loss_mask": 1.0,
                }
            )
        event_rows.append(
            {
                "event_id": f"event-{subject}",
                "subject_key": subject,
                "session_id": f"session-{subject}",
                "start_ms": 0,
                "end_ms": 6_000,
                "valid_duration": True,
                "evaluable": True,
            }
        )
    anchors = pd.DataFrame(anchor_rows)
    inputs = V4Inputs(
        anchors=anchors,
        segments=pd.DataFrame({"subject_key": all_subjects}),
        events=pd.DataFrame(event_rows),
        statistics=anchors[["segment_id", "session_id", "subject_key", "timestamp_ms"]],
        subject_folds={subject: 0 for subject in all_subjects},
    )
    config = {
        "model": {"architecture": "stub"},
        "sequence": {"step_seconds": 3},
        "training": {"random_seed": 2026, "selector_fraction": 0.2, "max_epochs": 1},
        "decoder": {
            "duration_lower_quantile": 0.005,
            "duration_upper_quantile": 0.995,
            "minimum_duration_floor_seconds": 3,
            "maximum_duration_ceiling_seconds": 60,
            "grid_seconds": 15,
            "fixed_lag_seconds": 60,
        },
        "hierarchical": {"verifier_crossfit_partitions": 3},
        "final_training": {"state_seeds": [2026]},
    }
    run = SimpleNamespace(root=tmp_path)
    oof_path = tmp_path / "oof" / "window_logits.parquet"
    oof_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "subject_key": ["meta-holdout"] * 4,
            "session_id": ["session-meta-holdout"] * 4,
            "timestamp_ms": [3_000, 6_000, 9_000, 12_000],
            "state_logit": [-2.0, 2.0, -2.0, 2.0],
            "onset_logit": [0.0] * 4,
            "offset_logit": [0.0] * 4,
            "state_target": [0.0, 1.0, 0.0, 1.0],
            "state_loss_mask": [1.0] * 4,
            "stacking_partition": [0] * 4,
        }
    ).to_parquet(oof_path, index=False)

    class StubScaler:
        def to_json(self):
            return {"training_subjects": sorted(training_subjects)}

    class StubStateModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    calls = {"train": 0, "infer": 0}

    def fake_scaler_transform(_inputs, _subjects):
        return StubScaler(), anchors.copy()

    def fake_dataset(rows, *_args, **_kwargs):
        return rows.copy().reset_index(drop=True)

    def fake_train(*_args, **_kwargs):
        calls["train"] += 1

    def fake_infer(_model, dataset, _config, *, stacking_partition, **_kwargs):
        calls["infer"] += 1
        output = dataset[
            ["subject_key", "session_id", "timestamp_ms", "state_target", "state_loss_mask"]
        ].copy()
        output["state_logit"] = np.where(output["state_target"] > 0, 2.0, -2.0)
        output["onset_logit"] = 0.0
        output["offset_logit"] = 0.0
        output["stacking_partition"] = int(stacking_partition)
        return output

    def fake_candidates(windows, _decoder, _config, *, split_role):
        rows = []
        for (subject, session), _ in windows.groupby(["subject_key", "session_id"]):
            rows.append(
                {
                    "proposal_id": f"proposal-{split_role}-{subject}",
                    "proposal_family_id": f"family-{split_role}-{subject}",
                    "subject_key": str(subject),
                    "session_id": str(session),
                    "coarse_start_ms": 0,
                    "coarse_end_ms": 6_000,
                    "source_mask": 1,
                    "generator_score": 0.9,
                    "rank_within_session": 1,
                    "split_role": split_role,
                }
            )
        return pd.DataFrame(rows)

    normalization = Normalization(
        np.zeros(6, dtype=np.float32),
        np.ones(6, dtype=np.float32),
        0.0,
        1.0,
    )
    state_root = tmp_path / "crossfit" / "partition_0" / "state"
    state_root.mkdir(parents=True)
    scaler_path = state_root / "statistics_scaler.json"
    normalization_path = state_root / "sensor_normalization.json"
    scaler_path.write_text(
        json.dumps({"training_subjects": sorted(training_subjects)}), encoding="utf-8"
    )
    normalization_path.write_text("normalization", encoding="utf-8")
    torch.save(
        {
            "model": {},
            "training_subjects": sorted(training_subjects),
            "prediction_subjects": sorted(holdout_subjects),
            "globally_excluded_subjects": sorted(holdout_subjects),
            "parent_artifact_sha256": {
                "statistics_scaler": sha256_file(scaler_path),
                "sensor_normalization": sha256_file(normalization_path),
            },
        },
        state_root / "seed_2026.pt",
    )
    trainer_module = importlib.import_module("bme_eating.training.hierarchical_v4_trainer")
    monkeypatch.setattr(trainer_module, "_fit_scaler_and_transform", fake_scaler_transform)
    monkeypatch.setattr(trainer_module, "compute_normalization", lambda *_args: normalization)
    monkeypatch.setattr(trainer_module, "_make_dataset", fake_dataset)
    monkeypatch.setattr(trainer_module, "_select_epoch", lambda *_args, **_kwargs: (1, {}))
    monkeypatch.setattr(trainer_module, "build_state_model", lambda *_args: StubStateModel())
    monkeypatch.setattr(trainer_module, "_train_state_epochs", fake_train)
    monkeypatch.setattr(trainer_module, "infer_state_windows", fake_infer)
    monkeypatch.setattr(trainer_module, "generate_event_candidates_v4", fake_candidates)
    root, artifacts = _prepare_nested_meta_cache(
        run,
        config,
        inputs,
        meta_partition=0,
        training_subjects=training_subjects,
        holdout_subjects=holdout_subjects,
    )
    assert root == tmp_path / "nested" / "meta_0"
    assert calls == {"train": 3, "infer": 3}
    lineage = json.loads((root / "lineage.json").read_text(encoding="utf-8"))
    assert set(lineage["training_subjects"]) == training_subjects
    assert set(lineage["globally_excluded_subjects"]) == holdout_subjects
    assert all(model["selected_epoch"] == 1 for model in lineage["inner_models"])
    assert len(artifacts) == len(set(artifacts))
    cached_root, cached_artifacts = _prepare_nested_meta_cache(
        run,
        config,
        inputs,
        meta_partition=0,
        training_subjects=training_subjects,
        holdout_subjects=holdout_subjects,
    )
    assert cached_root == root
    assert cached_artifacts == artifacts
    assert calls == {"train": 3, "infer": 3}
    checkpoint_path = state_root / "seed_2026.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["training_subjects"] = sorted(training_subjects | holdout_subjects)
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(RuntimeError, match="prediction subjects"):
        _prepare_nested_meta_cache(
            run,
            config,
            inputs,
            meta_partition=0,
            training_subjects=training_subjects,
            holdout_subjects=holdout_subjects,
        )


def test_boundary_fallback_flags_are_independent() -> None:
    accepted = pd.DataFrame(
        [
            {
                "proposal_id": "p",
                "subject_key": "s",
                "session_id": "d",
                "coarse_start_ms": 0,
                "coarse_end_ms": 30_000,
            }
        ]
    )
    refined = apply_boundary_refinement(
        accepted,
        np.asarray([3.0]),
        np.asarray([3.0]),
        np.asarray([0.9]),
        np.asarray([0.1]),
        entropy_threshold=0.5,
        safety_gap_seconds=3,
    )
    assert bool(refined.iloc[0].start_fallback)
    assert not bool(refined.iloc[0].end_fallback)
    assert bool(refined.iloc[0].boundary_fallback)


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
            "coarse_start_ms": [0, 13_000],
            "coarse_end_ms": [10_000, 20_000],
        }
    )
    refined = apply_boundary_refinement(
        accepted,
        np.asarray([0.0, -5.0]),
        np.asarray([5.0, 0.0]),
        np.zeros(2),
        np.zeros(2),
        entropy_threshold=0.5,
        safety_gap_seconds=3,
    ).sort_values("refined_start_ms")
    assert refined.iloc[1].refined_start_ms - refined.iloc[0].refined_end_ms >= 3000
    assert set(refined["proposal_id"]) == {"left", "right"}
    assert refined.iloc[0].refined_end_ms == refined.iloc[0].coarse_end_ms
    assert refined.iloc[1].refined_start_ms == refined.iloc[1].coarse_start_ms
    assert bool(refined.iloc[0].end_fallback)
    assert bool(refined.iloc[1].start_fallback)


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


def test_clip_sampler_uses_timestep_inclusion_correction() -> None:
    anchors = pd.DataFrame(
        {
            "subject_key": "s",
            "session_id": "d",
            "timestamp_ms": np.arange(12) * 3000,
            "state_target": [0.0] * 6 + [1.0] * 6,
            "start_target": [0.0] * 6 + [1.0] + [0.0] * 5,
            "end_target": [0.0] * 11 + [1.0],
            "state_loss_mask": [1.0] * 12,
        }
    )
    sampler = ClipMixtureSampler(
        anchors,
        samples_per_epoch=10,
        mixture={"uniform": 0.5, "event": 0.25, "boundary": 0.25},
        seed=2,
        supervised_steps=4,
    )
    _, _, weights = next(iter(sampler))
    assert weights.shape == (4,)
    assert np.isfinite(weights).all()


def test_truth_ignore_partition_and_ignore_predictions() -> None:
    events = pd.DataFrame(
        {
            "event_id": [f"t{i}" for i in range(161)]
            + [f"i{i}" for i in range(100)]
            + [f"x{i}" for i in range(6)],
            "subject_key": "s",
            "start_ms": np.arange(267) * 20_000,
            "end_ms": np.arange(267) * 20_000 + 10_000,
            "valid_duration": [True] * 261 + [False] * 6,
            "evaluable": [True] * 161 + [False] * 106,
        }
    )
    assert evaluation_event_partition_summary(events) == {
        "truth": 161,
        "ignore": 100,
        "invalid_duration": 6,
    }
    truth, ignore = partition_evaluation_events(events, {"s"})
    prediction = pd.DataFrame(
        [
            {
                "subject_key": "s",
                "start_ms": int(ignore.iloc[0].start_ms),
                "end_ms": int(ignore.iloc[0].end_ms),
            }
        ]
    )
    metrics, _ = evaluate_events(truth, prediction, ignore=ignore)
    assert metrics["false_positive"] == 0
    assert metrics["ignored_predictions"] == 1


def test_ignore_overlap_is_removed_even_if_truth_label_is_positive() -> None:
    proposal = pd.DataFrame(
        [
            {
                "proposal_id": "p",
                "subject_key": "s",
                "coarse_start_ms": 0,
                "coarse_end_ms": 10_000,
                "generator_score": 1.0,
            }
        ]
    )
    truth = pd.DataFrame([{"event_id": "t", "subject_key": "s", "start_ms": 0, "end_ms": 10_000}])
    ignore = pd.DataFrame(
        [{"event_id": "i", "subject_key": "s", "start_ms": 5_000, "end_ms": 15_000}]
    )
    labeled = label_event_candidates(proposal, truth, 0.25)
    assert labeled.iloc[0].is_positive
    assert exclude_ignored_candidates(labeled, ignore).empty


def test_semi_markov_grid_uses_ceil_without_extrapolation() -> None:
    prior = TruncatedLogNormalDurationPrior.fit(np.asarray([60.0, 120.0]))

    class RecordingDecoder(FixedLagSemiMarkovDecoder):
        seen: np.ndarray | None = None

        def decode_events(self, timestamps_ms, probabilities):
            self.seen = np.asarray(timestamps_ms)
            return []

    decoder = RecordingDecoder(prior, grid_seconds=15, fixed_lag_seconds=60)
    timestamps = np.arange(1_000, 91_001, 3_000, dtype=np.int64)
    windows = pd.DataFrame(
        {
            "subject_key": "s",
            "session_id": "d",
            "timestamp_ms": timestamps,
            "state_probability": 0.01,
            "onset_probability": 0.0,
            "offset_probability": 0.0,
        }
    )
    config = {
        "ema_half_life_seconds": 12,
        "high_threshold": 0.9,
        "low_threshold": 0.8,
        "gap_merge_seconds": 60,
        "transition_threshold": 0.9,
        "grid_seconds": 15,
        "use_semi_markov": True,
        "jitter_seconds": [0],
        "maximum_variants_per_event": 1,
        "maximum_candidates_per_hour": 20,
        "deduplication_iou": 0.9,
    }
    generate_event_candidates_v4(windows, decoder, config, split_role="test")
    assert decoder.seen is not None
    assert decoder.seen[0] == 15_000
    assert decoder.seen[-1] <= timestamps[-1]


def test_long_pool_uses_last_valid_token() -> None:
    pool = CausalCompletedBlockPool(1, 1, 5)
    pool.projection = torch.nn.Identity()
    values = torch.tensor([[[1.0], [2.0], [30.0], [40.0], [50.0]]])
    valid = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
    pooled, ratio = pool(values, valid, torch.tensor([[4]]))
    assert pooled[0, 0, 2].item() == 2.0
    assert ratio[0, 0].item() == pytest.approx(0.4)


def test_session_phased_completed_blocks_are_chunk_invariant_at_903_seconds() -> None:
    torch.manual_seed(2026)
    config = _model_config()
    config.update(
        {
            "motion_dilations": [1, 2],
            "ppg_dilations": [1],
            "statistics_dilations": [1, 2],
            "long_dilations": [1, 2],
        }
    )
    model = StatsFusionStateModel(config).eval()
    steps = 300
    target_timestamp = 903_000

    def batch_for_end(end_timestamp: int) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        timestamps = end_timestamp - np.arange(steps - 1, -1, -1, dtype=np.int64) * 3_000
        block_ends, block_indices, mapping = causal_completed_block_layout(
            timestamps,
            session_origin_ms=0,
            step_ms=3_000,
            factor=5,
        )
        high_index = (timestamps // 3_000).astype(np.float32)
        motion_value = np.sin(high_index / 17.0).astype(np.float32)
        motion = np.broadcast_to(
            motion_value[:, None, None], (steps, 12, 300)
        ).copy()
        statistics = np.stack(
            [np.sin(high_index / (index + 3.0)) for index in range(24)], axis=1
        ).astype(np.float32)
        low_value = np.cos(block_ends.astype(np.float64) / 41_000.0).astype(np.float32)
        ppg = np.broadcast_to(
            low_value[:, None, None], (len(block_ends), 2, 750)
        ).copy()
        return (
            {
                "motion_blocks": torch.from_numpy(motion).unsqueeze(0),
                "motion_valid": torch.ones(1, steps),
                "ppg_blocks": torch.from_numpy(ppg).unsqueeze(0),
                "ppg_quality": torch.zeros(1, len(block_ends), 8),
                "ppg_valid": torch.ones(1, len(block_ends)),
                "ppg_to_motion_index": torch.from_numpy(mapping).unsqueeze(0),
                "long_block_end_indices": torch.from_numpy(block_indices).unsqueeze(0),
                "statistics": torch.from_numpy(statistics).unsqueeze(0),
            },
            block_ends,
        )

    first, first_ends = batch_for_end(target_timestamp)
    second, second_ends = batch_for_end(target_timestamp + 256 * 3_000)
    first_position = int(np.flatnonzero(first["ppg_to_motion_index"][0].numpy() >= 0)[-1])
    second_timestamps = target_timestamp + 256 * 3_000 - np.arange(
        steps - 1, -1, -1, dtype=np.int64
    ) * 3_000
    second_position = int(np.flatnonzero(second_timestamps == target_timestamp)[0])
    first_block_end = first_ends[int(first["ppg_to_motion_index"][0, first_position])]
    second_block_end = second_ends[int(second["ppg_to_motion_index"][0, second_position])]
    assert first_block_end == 900_000
    assert second_block_end == 900_000

    with torch.no_grad():
        first_output = model(first)["state_logit"][0, -1]
        second_output = model(second)["state_logit"][0, second_position]
    torch.testing.assert_close(first_output, second_output, atol=1e-6, rtol=1e-6)


def test_sequence_geometry_reserves_global_block_phase_margin() -> None:
    geometry = SequenceGeometry()
    assert geometry.history_steps == 127 * 5 + 4
    assert geometry.total_steps == 895
    for endpoint_phase in range(5):
        timestamps = (
            endpoint_phase - np.arange(geometry.total_steps - 1, -1, -1)
        ) * 3_000
        _, block_indices, mapping = causal_completed_block_layout(
            timestamps,
            session_origin_ms=0,
            step_ms=3_000,
            factor=5,
        )
        first_supervised = geometry.history_steps
        current_block = mapping[first_supervised]
        complete_history = block_indices[: current_block + 1] >= 0
        assert int(complete_history.sum()) >= geometry.long_receptive_field_tokens


def test_dataset_and_raw_preprocessor_share_completed_block_phase(tmp_path) -> None:
    geometry = SequenceGeometry(
        supervised_steps=8,
        short_receptive_field_steps=3,
        long_receptive_field_tokens=3,
        long_pool_factor=5,
        step_seconds=3,
    )
    motion_time = np.arange(1_000, 61_001, 10, dtype=np.int64)
    ppg_time = np.arange(1_000, 61_001, 20, dtype=np.int64)
    motion_values = np.stack(
        [np.sin(motion_time / (1_000.0 + index * 100.0)) for index in range(6)],
        axis=1,
    ).astype(np.float32)
    ppg_values = np.cos(ppg_time / 2_000.0).astype(np.float32)
    archive = tmp_path / "segment.npz"
    np.savez(
        archive,
        motion_timestamp_ms=motion_time,
        motion_values=motion_values,
        motion_mask=np.ones_like(motion_values, dtype=bool),
        ppg_timestamp_ms=ppg_time,
        ppg_values=ppg_values.reshape(-1, 1),
        ppg_mask=np.ones((len(ppg_time), 1), dtype=bool),
    )
    segments = pd.DataFrame(
        {
            "segment_id": ["g"],
            "session_id": ["d"],
            "segment_path": [str(archive)],
            "subject_key": ["s"],
            "start_ms": [1_000],
            "end_ms": [61_000],
        }
    )
    events = pd.DataFrame(
        columns=["event_id", "subject_key", "start_ms", "end_ms", "valid_duration"]
    )
    anchors = build_statsfusion_session_anchor_index(segments, events)
    statistics_columns = [
        *(f"stat_{name}" for name in STATS_FEATURE_COLUMNS),
        *(f"stat_{name}_missing" for name in STATS_FEATURE_COLUMNS),
    ]
    for column in statistics_columns:
        anchors[column] = 0.0
    normalization = Normalization(np.zeros(6), np.ones(6), 0.0, 1.0)
    dataset = StatsFusionSequenceDataset(
        anchors,
        segments,
        events,
        normalization,
        statistics_columns=statistics_columns,
        geometry=geometry,
    )
    dataset_batch = dataset[len(dataset) - 1]
    scaler = FoldRobustScaler(
        STATS_FEATURE_COLUMNS,
        np.zeros(len(STATS_FEATURE_COLUMNS)),
        np.ones(len(STATS_FEATURE_COLUMNS)),
        ("train",),
    )
    preprocessor = StatsFusionRawSessionPreprocessor(
        normalization=normalization,
        statistics_scaler=scaler,
        geometry=geometry,
    )
    raw_session = RawSessionInput(
        "s",
        "d",
        motion_time,
        motion_values,
        np.ones_like(motion_values, dtype=bool),
        ppg_time,
        ppg_values,
        np.ones(len(ppg_values), dtype=bool),
    )
    raw_batch = list(preprocessor.iter_state_batches(raw_session))[-1]
    torch.testing.assert_close(
        raw_batch["ppg_blocks"][0], dataset_batch["ppg_blocks"], atol=0, rtol=0
    )
    torch.testing.assert_close(
        raw_batch["ppg_to_motion_index"][0],
        dataset_batch["ppg_to_motion_index"],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        raw_batch["long_block_end_indices"][0],
        dataset_batch["long_block_end_indices"],
        atol=0,
        rtol=0,
    )


def test_motion_validity_is_a_true_sampling_ratio() -> None:
    timestamps = np.arange(300, dtype=np.int64) * 10
    mask = np.zeros((300, 6), dtype=bool)
    mask[:150] = True
    normalization = Normalization(np.zeros(6), np.ones(6), 0.0, 1.0)
    _, valid = build_motion_blocks(
        timestamp_ms=timestamps,
        values=np.zeros((300, 6), dtype=np.float32),
        mask=mask,
        normalization=normalization,
        first_timestamp_ms=3_000,
        steps=1,
        step_seconds=3,
    )
    assert valid[0] == pytest.approx(0.5)


def test_raw_session_validation_rejects_nonfinite_valid_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        RawSessionInput(
            "s",
            "d",
            np.asarray([0]),
            np.asarray([[np.nan] * 6], dtype=np.float32),
            np.ones((1, 6), dtype=bool),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=bool),
        ).validated()


def test_raw_session_anchors_preserve_training_session_phase() -> None:
    first_ms = 2_800
    last_ms = 12_780
    timestamps = np.arange(first_ms, last_ms + 1, 10, dtype=np.int64)
    session = RawSessionInput(
        "s",
        "d",
        timestamps,
        np.zeros((len(timestamps), 6), dtype=np.float32),
        np.ones((len(timestamps), 6), dtype=bool),
        np.asarray([], dtype=np.int64),
        np.asarray([], dtype=np.float32),
        np.asarray([], dtype=bool),
    ).validated()
    scaler = FoldRobustScaler(
        STATS_FEATURE_COLUMNS,
        np.zeros(len(STATS_FEATURE_COLUMNS)),
        np.ones(len(STATS_FEATURE_COLUMNS)),
        ("s",),
    )
    preprocessor = StatsFusionRawSessionPreprocessor(
        normalization=Normalization(np.zeros(6), np.ones(6), 0.0, 1.0),
        statistics_scaler=scaler,
        geometry=SequenceGeometry(),
    )
    assert preprocessor.anchors(session).tolist() == [5_800, 8_800, 11_800]


def test_canonical_session_grid_does_not_restart_at_fragment_boundaries(tmp_path) -> None:
    first_ms = 1_000
    segment_rows = []
    motion_parts = []
    ppg_parts = []
    fragments = ((first_ms, 7_000, 7_000), (7_010, 13_000, 12_999))
    for index, (start_ms, raw_end_ms, metadata_end_ms) in enumerate(fragments):
        motion_time = np.arange(start_ms, raw_end_ms + 1, 10, dtype=np.int64)
        ppg_time = np.arange(start_ms, raw_end_ms + 1, 20, dtype=np.int64)
        motion_values = np.column_stack(
            [np.linspace(0.0, 1.0, len(motion_time), dtype=np.float32)] * 6
        )
        ppg_values = np.linspace(0.0, 1.0, len(ppg_time), dtype=np.float32)
        path = tmp_path / f"segment-{index}.npz"
        np.savez(
            path,
            motion_timestamp_ms=motion_time,
            motion_values=motion_values,
            motion_mask=np.ones_like(motion_values, dtype=bool),
            ppg_timestamp_ms=ppg_time,
            ppg_values=ppg_values.reshape(-1, 1),
            ppg_mask=np.ones((len(ppg_time), 1), dtype=bool),
        )
        segment_rows.append(
            {
                "segment_id": f"g{index}",
                "session_id": "d",
                "segment_path": str(path),
                "subject_key": "s",
                "start_ms": start_ms,
                "end_ms": metadata_end_ms,
            }
        )
        motion_parts.append((motion_time, motion_values))
        ppg_parts.append((ppg_time, ppg_values))
    events = pd.DataFrame(
        columns=["event_id", "subject_key", "start_ms", "end_ms", "valid_duration"]
    )
    anchors = build_statsfusion_session_anchor_index(pd.DataFrame(segment_rows), events)
    expected = session_right_endpoint_grid(first_ms, 13_000, 3_000)
    assert anchors["timestamp_ms"].tolist() == expected.tolist()
    assert 10_010 not in set(anchors["timestamp_ms"])
    motion_time = np.concatenate([part[0] for part in motion_parts])
    motion_values = np.concatenate([part[1] for part in motion_parts])
    ppg_time = np.concatenate([part[0] for part in ppg_parts])
    ppg_values = np.concatenate([part[1] for part in ppg_parts])
    session = RawSessionInput(
        "s",
        "d",
        motion_time,
        motion_values,
        np.ones_like(motion_values, dtype=bool),
        ppg_time,
        ppg_values,
        np.ones(len(ppg_values), dtype=bool),
    )
    scaler = FoldRobustScaler(
        STATS_FEATURE_COLUMNS, np.zeros(12), np.ones(12), ("train",)
    )
    preprocessor = StatsFusionRawSessionPreprocessor(
        normalization=Normalization(np.zeros(6), np.ones(6), 0.0, 1.0),
        statistics_scaler=scaler,
        geometry=SequenceGeometry(),
    )
    assert preprocessor.anchors(session).tolist() == expected.tolist()


def test_historical_statistics_padding_marks_all_features_missing() -> None:
    columns = [
        *(f"stat_{name}" for name in STATS_FEATURE_COLUMNS),
        *(f"stat_{name}_missing" for name in STATS_FEATURE_COLUMNS),
    ]
    group = pd.DataFrame(
        {
            "timestamp_ms": [3_000],
            "state_target": [0.0],
            "state_loss_mask": [1.0],
            "start_loss_mask": [1.0],
            "end_loss_mask": [1.0],
            **{name: [0.0] for name in columns},
        }
    )
    dataset = object.__new__(StatsFusionSequenceDataset)
    dataset.geometry = SequenceGeometry()
    dataset.statistics_columns = tuple(columns)
    valid, arrays = dataset._aligned_anchor_arrays(group, np.asarray([0, 3_000]))
    assert valid.tolist() == [False, True]
    assert np.all(arrays["statistics"][0, :12] == 0.0)
    assert np.all(arrays["statistics"][0, 12:] == 1.0)


def test_seeded_selector_model_initialization_is_reproducible() -> None:
    config = {"model": {"architecture": "stats_fusion_state", **_model_config()}}
    first = _build_seeded_state_model(config, 2026).state_dict()
    torch.manual_seed(9999)
    second = _build_seeded_state_model(config, 2026).state_dict()
    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_selector_split_is_subject_stratified_deterministic_and_large_enough() -> None:
    subjects = {f"s{index:02d}" for index in range(20)}
    event_rows = []
    anchor_rows = []
    for index, subject in enumerate(sorted(subjects)):
        for event_index in range(1 + index % 4):
            event_rows.append(
                {
                    "event_id": f"{subject}-e{event_index}",
                    "subject_key": subject,
                    "start_ms": event_index * 120_000,
                    "end_ms": event_index * 120_000 + (30 + index * 3) * 1000,
                    "valid_duration": True,
                    "evaluable": True,
                    "hand_relation": "same" if (index + event_index) % 2 else "different",
                }
            )
        for timestamp_ms in range(3_000, (10 + index) * 3_000, 3_000):
            anchor_rows.append(
                {
                    "subject_key": subject,
                    "session_id": f"session-{subject}",
                    "timestamp_ms": timestamp_ms,
                }
            )
    events = pd.DataFrame(event_rows)
    anchors = pd.DataFrame(anchor_rows)

    first = _selector_split(subjects, 0.35, 2026, events=events, anchors=anchors)
    second = _selector_split(subjects, 0.35, 2026, events=events, anchors=anchors)
    fit, selector, report = first

    assert first == second
    assert len(selector) == 7
    assert len(fit) == 13
    assert not fit & selector
    assert fit | selector == subjects
    assert report["strategy"] == "event_stratified_subject_subset_v1"
    assert report["selector_subject_count"] == 7
    assert report["selector_totals"]["same_event_count"] > 0
    assert report["selector_totals"]["different_event_count"] > 0


def test_selector_epoch_metrics_use_trailing_robust_window() -> None:
    epochs = [
        {
            "candidate_recall": 0.2,
            "calibration_passed": True,
            "event_f1": 0.1,
            "state_fragment_count": 20.0,
            "ece": 0.04,
            "window_auprc": 0.2,
        },
        {
            "candidate_recall": 0.9,
            "calibration_passed": False,
            "event_f1": 0.8,
            "state_fragment_count": 80.0,
            "ece": 0.2,
            "window_auprc": 0.9,
        },
        {
            "candidate_recall": 0.3,
            "calibration_passed": True,
            "event_f1": 0.2,
            "state_fragment_count": 25.0,
            "ece": 0.05,
            "window_auprc": 0.3,
        },
    ]

    _add_robust_epoch_metrics(epochs, 3)

    assert epochs[-1]["robust_candidate_recall"] == pytest.approx(0.3)
    assert epochs[-1]["robust_event_f1"] == pytest.approx(0.2)
    assert epochs[-1]["robust_calibration_passed"] is True


def test_selector_early_stopping_uses_v3_minimum_delta_semantics() -> None:
    best = {
        "robust_candidate_recall": 0.30,
        "robust_event_f1": 0.20,
        "robust_calibration_passed": False,
    }
    insignificant = {
        "robust_candidate_recall": 0.302,
        "robust_event_f1": 0.202,
        "robust_calibration_passed": False,
    }
    improved = {
        "robust_candidate_recall": 0.34,
        "robust_event_f1": 0.20,
        "robust_calibration_passed": False,
    }

    assert not _selector_early_stopping_improved(
        insignificant, best, minimum_recall=0.83, minimum_delta=0.003
    )
    assert _selector_early_stopping_improved(
        improved, best, minimum_recall=0.83, minimum_delta=0.003
    )


def test_semi_markov_cannot_chain_eating_segments_past_maximum_duration() -> None:
    prior = TruncatedLogNormalDurationPrior(
        log_mean=float(np.log(30.0)),
        log_standard_deviation=0.1,
        minimum_seconds=15.0,
        maximum_seconds=30.0,
    )
    decoder = FixedLagSemiMarkovDecoder(prior, grid_seconds=15, fixed_lag_seconds=0)
    for steps in (4, 8):
        timestamps = np.arange(1, steps + 1, dtype=np.int64) * 15_000
        events = decoder.decode_events(timestamps, np.full(steps, 0.99))
        assert events
        assert all(end - start <= 30_000 for start, end, _ in events)


def test_sequence_dataset_accepts_empty_unlabeled_event_frame() -> None:
    anchors = pd.DataFrame(
        {
            "subject_key": ["s"],
            "session_id": ["d"],
            "timestamp_ms": [3_000],
            "state_target": [0.0],
            "state_loss_mask": [1.0],
            **{f"stat_{name}": [0.0] for name in STATS_FEATURE_COLUMNS},
            **{f"stat_{name}_missing": [0.0] for name in STATS_FEATURE_COLUMNS},
        }
    )
    dataset = StatsFusionSequenceDataset(
        anchors,
        pd.DataFrame(columns=["session_id", "segment_id", "segment_path", "start_ms", "end_ms"]),
        pd.DataFrame(),
        Normalization(np.zeros(6), np.ones(6), 0.0, 1.0),
        statistics_columns=[
            *(f"stat_{name}" for name in STATS_FEATURE_COLUMNS),
            *(f"stat_{name}_missing" for name in STATS_FEATURE_COLUMNS),
        ],
    )
    assert dataset.events.empty
    assert dataset.ignore_events.empty
    assert {"subject_key", "start_ms", "end_ms", "valid_duration", "evaluable"}.issubset(
        dataset.events.columns
    )


def test_sequence_dataset_masks_ignore_events_from_state_and_boundary_training() -> None:
    anchors = pd.DataFrame(
        {
            "subject_key": ["s"] * 4,
            "session_id": ["d"] * 4,
            "timestamp_ms": [3_000, 6_000, 9_000, 12_000],
            "state_target": [1.0, 1.0, 1.0, 0.0],
            "state_loss_mask": [1.0] * 4,
            "start_target": [1.0, 0.9, 0.8, 0.7],
            "end_target": [0.0, 0.0, 1.0, 0.9],
            "start_loss_mask": [1.0] * 4,
            "end_loss_mask": [1.0] * 4,
            **{f"stat_{name}": [0.0] * 4 for name in STATS_FEATURE_COLUMNS},
            **{f"stat_{name}_missing": [0.0] * 4 for name in STATS_FEATURE_COLUMNS},
        }
    )
    events = pd.DataFrame(
        {
            "subject_key": ["s", "s"],
            "start_ms": [0, 30_000],
            "end_ms": [9_000, 40_000],
            "valid_duration": [True, True],
            "evaluable": [False, True],
        }
    )
    dataset = StatsFusionSequenceDataset(
        anchors,
        pd.DataFrame(
            columns=["session_id", "segment_id", "segment_path", "start_ms", "end_ms"]
        ),
        events,
        Normalization(np.zeros(6), np.ones(6), 0.0, 1.0),
        statistics_columns=[
            *(f"stat_{name}" for name in STATS_FEATURE_COLUMNS),
            *(f"stat_{name}_missing" for name in STATS_FEATURE_COLUMNS),
        ],
    )
    assert len(dataset.events) == 1
    assert len(dataset.ignore_events) == 1
    assert dataset.anchors["state_loss_mask"].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert dataset.anchors["start_loss_mask"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert dataset.anchors["end_loss_mask"].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_duration_prior_uses_only_evaluable_truth() -> None:
    events = pd.DataFrame(
        {
            "subject_key": ["s", "s", "s"],
            "start_ms": [0, 20_000, 40_000],
            "end_ms": [10_000, 50_000, 35_000],
            "valid_duration": [True, True, False],
            "evaluable": [True, False, True],
        }
    )
    assert _truth_event_durations(events, {"s"}).tolist() == [10.0]


def test_state_platt_fits_only_unmasked_windows_but_scores_full_timeline() -> None:
    frame = pd.DataFrame(
        {
            "subject_key": ["s0"] * 3 + ["s1"] * 3,
            "stacking_partition": [0] * 3 + [1] * 3,
            "state_logit": [-2.0, 2.0, 100.0, -1.5, 1.5, -100.0],
            "state_target": [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            "state_loss_mask": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        }
    )
    calibrated, final = subject_crossfit_platt(frame, fit_mask_column="state_loss_mask")
    eligible = frame["state_loss_mask"] > 0
    expected = PlattCalibration.fit(
        frame.loc[eligible, "state_logit"].to_numpy(),
        frame.loc[eligible, "state_target"].to_numpy(),
    )
    assert np.isfinite(calibrated["state_probability"]).all()
    assert final.coefficient == pytest.approx(expected.coefficient)
    assert final.intercept == pytest.approx(expected.intercept)


def test_outer_calibration_rows_exclude_state_loss_mask() -> None:
    frame = pd.DataFrame(
        {
            "state_target": [0.0, 1.0, 1.0],
            "state_logit": [-1.0, 1.0, 100.0],
            "state_probability": [0.2, 0.8, 1.0],
            "state_loss_mask": [1.0, 1.0, 0.0],
        }
    )
    selected, counts = _evaluable_state_calibration_rows(frame)
    assert selected.index.tolist() == [0, 1]
    assert counts == {
        "calibration_total_windows": 3,
        "calibration_evaluable_windows": 2,
        "calibration_masked_windows": 1,
    }


def test_current_anchor_snapshot_mask_count_when_available() -> None:
    path = Path(__file__).resolve().parents[3] / "outputs" / "v2" / "indices" / "anchors.parquet"
    if not path.is_file():
        pytest.skip("Current v2 anchor snapshot is not available")
    anchors = pd.read_parquet(path, columns=["state_loss_mask"])
    assert int((anchors["state_loss_mask"] <= 0).sum()) == 1117


def test_outer_state_labels_mask_ignore_intervals() -> None:
    frame = pd.DataFrame(
        {
            "subject_key": ["s", "s", "s"],
            "timestamp_ms": [3_000, 6_000, 12_000],
            "state_loss_mask": [1.0, 1.0, 1.0],
        }
    )
    ignore = pd.DataFrame({"subject_key": ["s"], "start_ms": [0], "end_ms": [9_000]})
    masked = _mask_ignored_state_rows(frame, ignore, step_ms=3_000)
    assert masked["state_loss_mask"].tolist() == [0.0, 0.0, 1.0]


def test_pooled_proposal_alignment_rejects_reordered_rows() -> None:
    features = ProposalFeatureBatchV4(
        proposal_ids=np.asarray(["p1", "p2"]),
        sequence=np.zeros((2, 1, 1), dtype=np.float32),
        sequence_mask=np.ones((2, 1), dtype=bool),
        scalar=np.zeros((2, 1), dtype=np.float32),
        event_target=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    frame = pd.DataFrame(
        {"proposal_id": ["p2", "p1"], "is_positive": [False, True]}
    )
    with pytest.raises(RuntimeError, match="not aligned"):
        _assert_proposal_feature_alignment(features, frame, context="test")
