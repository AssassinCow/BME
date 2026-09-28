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
    LogisticScoreCombiner,
    PlattCalibration,
    ProposalCalibrationV4,
    binary_state_targets,
    state_calibration_metrics,
    subject_crossfit_platt,
)
from bme_eating.data.deep_dataset import Normalization, compute_normalization
from bme_eating.data.labels import assign_event_sessions, build_statsfusion_session_anchor_index
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
from bme_eating.features.signal import robust_statistics
from bme_eating.hierarchical_artifacts import sha256_file
from bme_eating.hierarchical_v4_pipeline import (
    HierarchicalEatingDetectorV4,
    _iter_tail_aligned_raw_state_batches,
)
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
    pooled_logistic_features,
)
from bme_eating.models.stats_fusion_loss import (
    StatsFusionStateLoss,
    one_sided_transition_targets,
)
from bme_eating.models.stats_fusion_state import (
    CausalCompletedBlockPool,
    StatsFusionStateModel,
    TimewiseLayerNorm,
)
from bme_eating.proposals import exclude_ignored_candidates, label_event_candidates
from bme_eating.proposals_v4 import (
    ALLOWED_SOURCE_MASK,
    ProposalSource,
    _hysteresis,
    _jitter,
    generate_event_candidates_v4,
    observed_hours_v4,
)
from bme_eating.stats_features import STATS_FEATURE_COLUMNS, FoldRobustScaler
from bme_eating.structured_decoder import (
    FixedLagSemiMarkovDecoder,
    TruncatedLogNormalDurationPrior,
    right_endpoint_run_to_interval,
)
from bme_eating.timeline import (
    claim_new_timeline_rows,
    deduplicate_consistent_timeline,
    tail_aligned_chunk_endpoints,
)
from bme_eating.training.hierarchical_v4_trainer import (
    UNLABELED_ANCHOR_COLUMNS,
    V4Inputs,
    _add_robust_epoch_metrics,
    _apply_state_calibration_to_windows,
    _assert_nested_lineage,
    _assert_proposal_feature_alignment,
    _attach_truth_boundaries,
    _build_isolated_pooled_fold_data,
    _build_seeded_state_model,
    _candidate_domain_metrics,
    _candidate_recall,
    _choose_conservative_state_epoch,
    _evaluable_state_calibration_rows,
    _fit_pooled_logistic_crossfit,
    _gap_aware_nms,
    _hand_metrics,
    _inference_endpoints,
    _load_outer_event_snapshot,
    _mask_ignored_state_rows,
    _matching_ranking_reversal,
    _nested_cache_artifacts,
    _nested_cache_key,
    _prepare_nested_meta_cache,
    _selector_early_stopping_improved,
    _selector_split,
    _state_samples_per_epoch,
    _truth_event_durations,
    _write_outer_event_snapshot,
    infer_state_windows,
    load_v4_inputs,
)


def test_inference_endpoints_tile_tail_without_overlapping_supervision() -> None:
    rows = np.arange(1921, dtype=np.int64)
    dataset = SimpleNamespace(
        geometry=SimpleNamespace(supervised_steps=256),
        session_groups={
            ("subject", "session"): pd.DataFrame({"_row_id": rows})
        },
    )
    endpoints = _inference_endpoints(dataset)
    assert endpoints[-1] == 1920
    owned: list[int] = []
    for endpoint in endpoints:
        owned.extend(range(max(0, endpoint - 255), endpoint + 1))
    assert owned == list(range(1921))


@pytest.mark.parametrize("total_rows", [0, 1, 2, 255, 256, 257, 1921])
def test_tail_aligned_chunk_endpoints_cover_each_row_once(total_rows: int) -> None:
    endpoints = tail_aligned_chunk_endpoints(total_rows, 256)
    owned: list[int] = []
    for endpoint in endpoints:
        owned.extend(range(max(0, endpoint - 255), endpoint + 1))
    assert owned == list(range(total_rows))


def test_outer_truth_ignore_snapshot_rejects_tampering(tmp_path) -> None:
    truth = pd.DataFrame(
        {
            "subject_key": ["s0"],
            "session_id": ["session"],
            "event_id": ["event"],
            "start_ms": [0],
            "end_ms": [10_000],
            "hand_relation": ["same"],
        }
    )
    ignore = truth.assign(event_id="ignore", start_ms=20_000, end_ms=30_000)
    truth_path, _, _ = _write_outer_event_snapshot(
        tmp_path, truth, ignore, expected_subjects={"s0", "no-events"}
    )
    loaded_truth, loaded_ignore, manifest = _load_outer_event_snapshot(tmp_path)
    assert len(loaded_truth) == 1
    assert len(loaded_ignore) == 1
    assert manifest["snapshot_protocol"] == "outer_evaluation_truth_ignore_sha256_v1"
    assert manifest["subjects"] == ["no-events", "s0"]
    assert manifest["subjects_with_events"] == ["s0"]
    truth_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="snapshot changed"):
        _load_outer_event_snapshot(tmp_path)


def test_pooled_fold_upstream_and_model_inputs_exclude_heldout_labels(monkeypatch) -> None:
    import bme_eating.training.hierarchical_v4_trainer as trainer

    logits: list[pd.DataFrame] = []
    windows: list[pd.DataFrame] = []
    truth: list[pd.DataFrame] = []
    ignore: list[pd.DataFrame] = []
    for fold in range(5):
        subject = f"s{fold}"
        session = f"d{fold}"
        timestamp = np.arange(4, dtype=np.int64) * 3_000 + 3_000
        frame = pd.DataFrame(
            {
                "subject_key": subject,
                "session_id": session,
                "timestamp_ms": timestamp,
                "state_logit": [-2.0, -0.5, 0.5, 2.0],
                "state_target": [0.0, 0.25, 0.75, 1.0],
                "state_loss_mask": 1.0,
            }
        )
        logits.append(frame.copy())
        windows.append(frame.copy())
        truth.append(
            pd.DataFrame(
                {
                    "subject_key": [subject],
                    "session_id": [session],
                    "event_id": [f"e{fold}"],
                    "start_ms": [0],
                    "end_ms": [6_000],
                }
            )
        )
        ignore.append(
            pd.DataFrame(columns=["subject_key", "session_id", "start_ms", "end_ms"])
        )

    def select_decoder(*_args, **_kwargs):
        return (
            {
                "grid_seconds": 15,
                "fixed_lag_seconds": 60,
                "semi_markov_duration_weight": 1.0,
            },
            pd.DataFrame({"candidate_recall": [1.0]}),
        )

    def candidates(frame, *_args, **_kwargs):
        subject = str(frame.iloc[0].subject_key)
        session = str(frame.iloc[0].session_id)
        return pd.DataFrame(
            {
                "proposal_id": ["positive", "negative"],
                "proposal_family_id": ["positive", "negative"],
                "subject_key": [subject, subject],
                "session_id": [session, session],
                "coarse_start_ms": [0, 9_000],
                "coarse_end_ms": [6_000, 12_000],
                "generator_score": [0.9, 0.1],
                "source_mask": [1, 1],
            }
        )

    def proposal_features(proposals, *_args, **_kwargs):
        count = len(proposals)
        coordinates = proposals[["coarse_start_ms", "coarse_end_ms"]].to_numpy(
            dtype=np.float32
        )
        sequence = np.repeat(coordinates[:, None, :], 2, axis=1)
        scalar = np.column_stack(
            (
                proposals["generator_score"].to_numpy(dtype=np.float32),
                np.zeros((count, 7), dtype=np.float32),
            )
        )
        return ProposalFeatureBatchV4(
            proposal_ids=proposals["proposal_id"].astype(str).to_numpy(),
            sequence=sequence,
            sequence_mask=np.ones((count, 2), dtype=bool),
            scalar=scalar,
            event_target=proposals["is_positive"].to_numpy(dtype=np.float32),
            iou_target=proposals["max_iou"].to_numpy(dtype=np.float32),
            sample_weight=np.ones(count, dtype=np.float32),
        )

    monkeypatch.setattr(trainer, "_select_decoder_configuration", select_decoder)
    monkeypatch.setattr(trainer, "generate_event_candidates_v4", candidates)
    monkeypatch.setattr(trainer, "build_proposal_features_v4", proposal_features)
    config = {
        "decoder": {
            "duration_lower_quantile": 0.0,
            "duration_upper_quantile": 1.0,
            "minimum_duration_floor_seconds": 3,
            "maximum_duration_ceiling_seconds": 14_400,
        },
        "verifier": {},
    }
    baseline = _build_isolated_pooled_fold_data(logits, windows, truth, ignore, config)
    changed_truth = [frame.copy() for frame in truth]
    changed_truth[0] = changed_truth[0].assign(start_ms=30_000, end_ms=36_000)
    changed = _build_isolated_pooled_fold_data(
        logits, windows, changed_truth, ignore, config
    )
    baseline_fold = baseline[0]
    changed_fold = changed[0]
    assert baseline_fold.upstream_lineage == changed_fold.upstream_lineage
    np.testing.assert_array_equal(
        baseline_fold.proposals.iloc[baseline_fold.training_indices]["proposal_id"],
        changed_fold.proposals.iloc[changed_fold.training_indices]["proposal_id"],
    )
    for name in ("sequence", "sequence_mask", "scalar"):
        np.testing.assert_array_equal(
            getattr(baseline_fold.features, name)[baseline_fold.prediction_indices],
            getattr(changed_fold.features, name)[changed_fold.prediction_indices],
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


def test_timewise_layer_norm_keeps_low_variance_gradients_finite() -> None:
    torch.manual_seed(2026)
    values = (torch.ones(2, 32, 11) + 1e-7 * torch.randn(2, 32, 11)).requires_grad_()
    normalized = TimewiseLayerNorm(32)(values)
    loss = normalized.square().mean()
    loss.backward()
    assert torch.isfinite(normalized).all()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert float(values.grad.norm()) < 1e4


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


def test_empty_robust_statistics_are_missing_not_zero_signal() -> None:
    summary = robust_statistics(np.asarray([np.nan, np.inf, -np.inf]))
    assert summary
    assert all(np.isnan(value) for value in summary.values())


def test_soft_state_targets_are_preserved_for_calibration_and_binarized_only_for_diagnostics() -> (
    None
):
    targets = np.array([0.0, 0.005, 0.75, 1.0], dtype=np.float64)
    logits = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
    binary = binary_state_targets(targets)
    assert binary.tolist() == [0, 0, 1, 1]

    calibrator = PlattCalibration.fit(logits, targets)
    calibrated = calibrator.transform(logits)
    metrics = state_calibration_metrics(
        targets,
        logits,
        calibrated,
        low_threshold=0.15,
    )
    assert np.isfinite(calibrated).all()
    assert metrics["prevalence"] == pytest.approx(targets.mean())
    assert np.isfinite(metrics["brier"])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        binary_state_targets(np.array([0.0, 1.1]))


def test_proposal_calibration_fails_fast_on_empty_or_single_class_oof() -> None:
    with pytest.raises(ValueError, match="non-empty 2D"):
        LogisticScoreCombiner.fit(np.empty((0, 3)), np.empty(0))
    with pytest.raises(ValueError, match="positive and negative"):
        LogisticScoreCombiner.fit(np.ones((2, 3)), np.ones(2))
    frame = pd.DataFrame(
        {
            "event_logit": [0.0, 1.0],
            "iou_logit": [0.0, 1.0],
            "state_score": [0.2, 0.8],
            "is_positive": [1, 1],
            "max_iou": [0.5, 0.8],
        }
    )
    with pytest.raises(ValueError, match="positive and negative"):
        ProposalCalibrationV4.fit(frame)


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


def test_r3_loads_session_statistics_without_segment_id(tmp_path, monkeypatch) -> None:
    import bme_eating.training.hierarchical_v4_trainer as trainer

    input_root = tmp_path / "v2"
    output_root = tmp_path / "v4"
    (input_root / "indices").mkdir(parents=True)
    canonical_root = output_root / "canonical_input_r3_2"
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
    canonical_events = pd.DataFrame(
        {
            "subject_key": ["train"],
            "session_id": ["d0"],
            "event_id": ["e0"],
            "start_ms": [0],
            "end_ms": [3_000],
            "valid_duration": [True],
            "evaluable": [True],
        }
    )
    canonical_events.to_parquet(input_root / "indices" / "events.parquet", index=False)
    canonical_events_path = canonical_root / "events_with_session.parquet"
    canonical_events.to_parquet(canonical_events_path, index=False)
    (input_root / "indices" / "subject_folds.json").write_text(
        json.dumps({"train": 1, "outer": 0}), encoding="utf-8"
    )
    monkeypatch.setattr(trainer, "verify_canonical_statsfusion_inputs", lambda *_args: {})
    monkeypatch.setattr(
        trainer,
        "canonical_input_paths",
        lambda _root: {
            "anchors": anchors_path,
            "statistics": statistics_path,
            "events": canonical_events_path,
        },
    )

    inputs = load_v4_inputs(
        {
            "experiment": {"protocol_version": "statsfusion-r3.2"},
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


def test_session_assignment_keeps_truth_unique_and_expands_ignore_support() -> None:
    segments = pd.DataFrame(
        {
            "subject_key": ["s", "s"],
            "session_id": ["a", "b"],
            "start_ms": [0, 20_000],
            "end_ms": [10_000, 30_000],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["truth", "ignore", "unobserved"],
            "subject_key": ["s", "s", "s"],
            "start_ms": [1_000, 5_000, 100_000],
            "end_ms": [5_000, 25_000, 110_000],
            "valid_duration": [True, True, True],
            "evaluable": [True, False, False],
        }
    )
    assigned = assign_event_sessions(events, segments)
    assert assigned.loc[assigned.event_id == "truth", "session_id"].tolist() == ["a"]
    assert set(assigned.loc[assigned.event_id == "ignore", "session_id"]) == {"a", "b"}
    unobserved = assigned[assigned.event_id == "unobserved"].iloc[0]
    assert str(unobserved.session_id).startswith("__unobserved__:")
    assert evaluation_event_partition_summary(assigned) == {
        "truth": 1,
        "ignore": 2,
        "invalid_duration": 0,
    }


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


def test_s2_missing_fraction_excludes_disabled_ppg() -> None:
    config = _model_config()
    config["use_ppg"] = False
    model = StatsFusionStateModel(config).eval()
    batch = _state_batch()
    changed = {key: value.clone() for key, value in batch.items()}
    changed["ppg_valid"].zero_()
    changed["ppg_quality"].zero_()
    with torch.no_grad():
        original = model(batch)
        modified = model(changed)
    torch.testing.assert_close(
        original["active_modality_missing_fraction"],
        modified["active_modality_missing_fraction"],
        atol=0,
        rtol=0,
    )


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
        smooth_weight=1.0, smooth_beta=0.5, boundary_weight=0.1
    )(output, batch)
    loss.backward()
    assert components["smooth"] > 0
    assert logits.grad is not None and torch.count_nonzero(logits.grad) > 0


def test_huber_smoothing_keeps_nonzero_gradient_for_large_jumps() -> None:
    logits = torch.tensor([[0.0, 10.0]], requires_grad=True)
    zeros = torch.zeros_like(logits, requires_grad=True)
    _, components = StatsFusionStateLoss(smooth_weight=1.0, smooth_beta=0.5, boundary_weight=0.1)(
        {
            "state_logit": logits,
            "onset_logit": zeros,
            "offset_logit": zeros,
        },
        {
            "state_target": torch.zeros_like(logits),
            "onset_target": torch.zeros_like(logits),
            "offset_target": torch.zeros_like(logits),
            "supervision_mask": torch.ones_like(logits),
            "state_loss_mask": torch.ones_like(logits),
            "onset_loss_mask": torch.ones_like(logits),
            "offset_loss_mask": torch.ones_like(logits),
            "smooth_loss_mask": torch.ones_like(logits),
            "smooth_mask": torch.ones_like(logits),
        },
    )
    gradient = torch.autograd.grad(components["smooth"], logits)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) == 2


def test_ignore_region_changes_do_not_affect_any_loss_or_gradient() -> None:
    criterion = StatsFusionStateLoss(smooth_weight=0.5, smooth_beta=0.5, boundary_weight=0.1)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0]])
    batch = {
        "state_target": torch.tensor([[0.0, 1.0, 0.3, 0.7, 1.0, 0.0]]),
        "onset_target": torch.zeros(1, 6),
        "offset_target": torch.zeros(1, 6),
        "supervision_mask": torch.ones(1, 6),
        "state_loss_mask": mask,
        "onset_loss_mask": mask,
        "offset_loss_mask": mask,
        "smooth_loss_mask": mask,
        "smooth_mask": torch.ones(1, 6),
    }

    def evaluate(replacement: float) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        state = torch.tensor(
            [[-2.0, 2.0, replacement, -replacement, 2.0, -2.0]], requires_grad=True
        )
        onset = state.detach().clone().requires_grad_(True)
        offset = state.detach().clone().requires_grad_(True)
        total, components = criterion(
            {"state_logit": state, "onset_logit": onset, "offset_logit": offset}, batch
        )
        gradients = torch.autograd.grad(total, (state, onset, offset))
        packed = torch.stack(gradients)
        return total.detach(), {key: value.detach() for key, value in components.items()}, packed

    first_total, first_components, first_gradient = evaluate(10.0)
    second_total, second_components, second_gradient = evaluate(100.0)
    torch.testing.assert_close(first_total, second_total, atol=0, rtol=0)
    for key in ("state", "onset", "offset", "smooth"):
        torch.testing.assert_close(first_components[key], second_components[key], atol=0, rtol=0)
    torch.testing.assert_close(first_gradient, second_gradient, atol=0, rtol=0)
    assert torch.count_nonzero(first_gradient[:, :, 2:4]) == 0


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


def test_subject_balanced_clip_sampler_equalizes_subject_exposure() -> None:
    anchors = pd.DataFrame(
        {
            "subject_key": ["long"] * 100 + ["short"] * 10,
            "session_id": ["long-session"] * 100 + ["short-session"] * 10,
            "timestamp_ms": np.arange(110) * 3_000,
            "state_target": 0.0,
            "start_target": 0.0,
            "end_target": 0.0,
            "state_loss_mask": 1.0,
        }
    )
    sampler = ClipMixtureSampler(
        anchors,
        samples_per_epoch=20_000,
        mixture={"uniform": 0.5, "event": 0.25, "boundary": 0.25},
        seed=2026,
        supervised_steps=1,
        balance_subjects=True,
    )
    counts = {"long": 0, "short": 0}
    subjects = anchors["subject_key"].to_numpy()
    for index, _, _ in sampler:
        counts[str(subjects[index])] += 1
    assert abs(counts["long"] - counts["short"]) / 20_000 < 0.02


def test_state_epoch_exposure_scales_with_subject_count() -> None:
    dataset = SimpleNamespace(anchors=pd.DataFrame({"subject_key": ["a", "b", "c", "c"]}))
    config = {
        "training": {
            "batch_size": 8,
            "steps_per_epoch": 10,
            "clips_per_subject_per_epoch": 1_000,
        }
    }
    assert _state_samples_per_epoch(dataset, config) == 3_000


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


def test_candidate_budget_keeps_distinct_families_before_extra_variants(monkeypatch) -> None:
    monkeypatch.setattr(
        "bme_eating.proposals_v4._hysteresis",
        lambda *_args, **_kwargs: [
            (0, 30_000, 0.9, int(ProposalSource.HYSTERESIS)),
            (120_000, 150_000, 0.8, int(ProposalSource.HYSTERESIS)),
        ],
    )

    def variants(seeds, *_args, **_kwargs):
        first, second = seeds
        return [
            (first[0], first[1], 0.9, first[3], first[4]),
            (30_000, 60_000, 0.89, first[3] | int(ProposalSource.JITTER), first[4]),
            (second[0], second[1], 0.8, second[3], second[4]),
        ]

    monkeypatch.setattr("bme_eating.proposals_v4._jitter", variants)
    timestamps = np.arange(1, 121, dtype=np.int64) * 3_000
    windows = pd.DataFrame(
        {
            "subject_key": "s",
            "session_id": "d",
            "timestamp_ms": timestamps,
            "state_probability": 0.9,
            "onset_probability": 0.0,
            "offset_probability": 0.0,
        }
    )
    prior = TruncatedLogNormalDurationPrior.fit(np.asarray([30.0, 60.0]))
    proposals = generate_event_candidates_v4(
        windows,
        FixedLagSemiMarkovDecoder(prior, fixed_lag_seconds=60),
        {
            "use_semi_markov": False,
            "use_transition_candidates": False,
            "ema_half_life_seconds": 0,
            "high_threshold": 0.5,
            "low_threshold": 0.2,
            "gap_merge_seconds": 0,
            "transition_threshold": 0.5,
            "grid_seconds": 15,
            "jitter_seconds": [0],
            "maximum_variants_per_event": 3,
            "maximum_candidates_per_hour": 20,
            "deduplication_iou": 0.99,
            "candidate_minimum_seconds": 3,
            "candidate_maximum_seconds": 14_400,
        },
        split_role="test",
    )
    assert len(proposals) == 2
    assert proposals["proposal_family_id"].nunique() == 2


def test_candidates_do_not_cross_fully_unobserved_timeline_gap() -> None:
    prior = TruncatedLogNormalDurationPrior.fit(np.asarray([30.0, 60.0]))
    decoder = FixedLagSemiMarkovDecoder(prior, fixed_lag_seconds=60)
    timestamps = np.arange(40, dtype=np.int64) * 3_000
    decode_valid = np.ones(len(timestamps), dtype=bool)
    decode_valid[10:20] = False
    windows = pd.DataFrame(
        {
            "subject_key": "s",
            "session_id": "d",
            "timestamp_ms": timestamps,
            "state_probability": 0.9,
            "onset_probability": 0.0,
            "offset_probability": 0.0,
            "decode_valid": decode_valid,
        }
    )
    proposals = generate_event_candidates_v4(
        windows,
        decoder,
        {
            "use_semi_markov": True,
            "ema_half_life_seconds": 0,
            "high_threshold": 0.5,
            "low_threshold": 0.2,
            "gap_merge_seconds": 60,
            "transition_threshold": 0.5,
            "grid_seconds": 15,
            "jitter_seconds": [0],
            "maximum_variants_per_event": 1,
            "maximum_candidates_per_hour": 20,
            "deduplication_iou": 0.9,
            "candidate_minimum_seconds": 3,
            "candidate_maximum_seconds": 14_400,
        },
        split_role="test",
    )
    gap_start = int(timestamps[10] - 3_000)
    gap_end = int(timestamps[19])
    assert len(proposals)
    assert observed_hours_v4(windows) == pytest.approx(30 * 3 / 3600)
    assert not (
        (proposals["coarse_start_ms"] < gap_end) & (proposals["coarse_end_ms"] > gap_start)
    ).any()


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


def test_candidate_recall_uses_session_scoped_event_identity() -> None:
    events = pd.DataFrame(
        {
            "subject_key": ["subject", "subject"],
            "session_id": ["morning", "evening"],
            "event_id": ["meal", "meal"],
            "start_ms": [0, 100_000],
            "end_ms": [10_000, 110_000],
            "evaluable": [True, True],
            "hand_relation": ["same", "different"],
        }
    )
    proposals = pd.DataFrame(
        {
            "subject_key": ["subject", "subject"],
            "session_id": ["morning", "evening"],
            "coarse_start_ms": [0, 100_000],
            "coarse_end_ms": [10_000, 110_000],
        }
    )
    metrics = _candidate_recall(proposals, events)
    assert metrics["candidate_recall"] == 1.0
    assert metrics["same_candidate_recall"] == 1.0
    assert metrics["different_candidate_recall"] == 1.0


def test_duration_theoretical_matchability_uses_best_legal_candidate_duration() -> None:
    truth = pd.DataFrame(
        {
            "subject_key": ["subject"] * 3,
            "session_id": ["session"] * 3,
            "event_id": ["short", "ordinary", "over"],
            "start_ms": [0, 10_000, 200_000],
            "end_ms": [2_000, 110_000, 20_200_000],
        }
    )
    windows = pd.DataFrame(
        {
            "subject_key": ["subject"],
            "session_id": ["session"],
            "timestamp_ms": [3_000],
            "gyro_valid_fraction": [1.0],
        }
    )
    empty_proposals = pd.DataFrame(
        columns=["subject_key", "session_id", "coarse_start_ms", "coarse_end_ms"]
    )
    metrics = _candidate_domain_metrics(empty_proposals, truth, windows)
    assert metrics["duration_strata"]["under_15s"]["theoretical_matchable_fraction"] == 1.0
    assert metrics["duration_strata"]["ordinary"]["theoretical_matchable_fraction"] == 1.0
    assert metrics["duration_strata"]["over_4h"]["theoretical_matchable_fraction"] == 1.0


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
            "statistics_gate": 1.0,
            "long_gate": 1.0,
            "gyro_gate": 1.0,
            "invariant_gate": 1.0,
            "missing_fraction": 0.0,
            "acc_valid_fraction": 1.0,
            "gyro_valid_fraction": 1.0,
            "ppg_valid_fraction": 1.0,
            "statistics_missing_fraction": 0.0,
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
    assert features.sequence[0, 3, 0] == 1.0
    assert features.sequence[0, 4, 0] == 4.0
    assert features.sequence[0, 5, 0] == 7.0
    assert features.scalar[0, 2] == 5.0
    assert not features.sequence_mask[0, 0]
    assert features.sequence_mask[0, 4]


def test_proposal_pooling_handles_empty_candidate_frame_without_nan_weights() -> None:
    proposals = pd.DataFrame(
        columns=[
            "proposal_id",
            "subject_key",
            "session_id",
            "coarse_start_ms",
            "coarse_end_ms",
            "source_mask",
            "generator_score",
            "max_iou",
        ]
    )
    windows = pd.DataFrame(
        columns=[
            "subject_key",
            "session_id",
            "timestamp_ms",
            "state_probability",
            "onset_probability",
            "offset_probability",
            "ppg_gate",
            "statistics_gate",
            "long_gate",
            "gyro_gate",
            "invariant_gate",
            "missing_fraction",
            "acc_valid_fraction",
            "gyro_valid_fraction",
            "ppg_valid_fraction",
            "statistics_missing_fraction",
        ]
    )
    features = build_proposal_features_v4(
        proposals,
        windows,
        [],
        {
            "left_context_seconds": 60,
            "right_context_seconds": 60,
            "left_bins": 4,
            "event_bins": 16,
            "right_bins": 4,
        },
    )
    assert features.sequence.shape[0] == 0
    assert features.sample_weight is not None
    assert features.sample_weight.size == 0


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


def test_positive_verifier_weights_keep_same_event_id_separate_across_sessions() -> None:
    frame = pd.DataFrame(
        {
            "subject_key": ["subject"] * 4,
            "session_id": ["morning", "evening", "evening", "evening"],
            "proposal_id": ["m", "e1", "e2", "e3"],
            "matched_event_id": ["meal"] * 4,
            "max_iou": [0.9] * 4,
        }
    )
    weights = normalized_proposal_weights(frame)
    assert weights[0] == pytest.approx(0.5)
    assert weights[1:].sum() == pytest.approx(0.5)


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
    assert torch.allclose(first_endpoint[mask], second_endpoint[mask], atol=1e-7, rtol=0)


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

    prior = TruncatedLogNormalDurationPrior(
        log_mean=float(np.log(30.0)),
        log_standard_deviation=0.5,
        minimum_seconds=15.0,
        maximum_seconds=45.0,
    )
    decoder = ForcedDecoder(prior, grid_seconds=15, fixed_lag_seconds=60)
    assert decoder.decode_events(np.asarray([15_000, 30_000]), np.asarray([0.9, 0.1]))[0][:2] == (
        0,
        15_000,
    )
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
        "protocol_version": "statsfusion-r3.2",
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
            "use_semi_markov": False,
            "ema_half_life_seconds": 12,
            "high_threshold": 0.20,
            "low_threshold": 0.10,
            "gap_merge_seconds": 30,
            "transition_threshold": 0.30,
            "semi_markov_duration_weight": 1.0,
            "candidate_minimum_seconds": 3,
            "candidate_maximum_seconds": 14_400,
            "jitter_seconds": [0],
            "maximum_variants_per_event": 1,
            "maximum_candidates_per_hour": 20,
            "deduplication_iou": 0.9,
        },
        "decoder_search": {
            "high_threshold": [0.20],
            "low_threshold": [0.10],
            "ema_half_life_seconds": [12],
            "gap_merge_seconds": [30],
            "transition_threshold": [0.30],
            "semi_markov_duration_weight": [1.0],
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
    monkeypatch.setattr(
        trainer_module, "compute_normalization", lambda *_args, **_kwargs: normalization
    )
    monkeypatch.setattr(trainer_module, "_make_dataset", fake_dataset)
    monkeypatch.setattr(trainer_module, "_select_epoch", lambda *_args, **_kwargs: (1, {}))
    monkeypatch.setattr(trainer_module, "build_state_model", lambda *_args: StubStateModel())
    monkeypatch.setattr(trainer_module, "_train_state_with_checkpoints", fake_train)
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


def test_boundary_event_identity_is_scoped_by_subject_and_session() -> None:
    proposals = pd.DataFrame(
        {
            "proposal_id": ["morning", "evening"],
            "subject_key": ["subject", "subject"],
            "session_id": ["morning", "evening"],
            "coarse_start_ms": [0, 100_000],
            "coarse_end_ms": [10_000, 110_000],
            "matched_event_id": ["meal", "meal"],
            "max_iou": [0.9, 0.9],
        }
    )
    events = pd.DataFrame(
        {
            "subject_key": ["subject", "subject"],
            "session_id": ["morning", "evening"],
            "event_id": ["meal", "meal"],
            "start_ms": [1_000, 102_000],
            "end_ms": [9_000, 108_000],
        }
    )
    attached = _attach_truth_boundaries(proposals, events)
    assert attached["truth_start_ms"].tolist() == [1_000, 102_000]
    augmented = augment_boundary_training_proposals(attached)
    assert set(augmented["proposal_id"]) == {"morning", "evening"}
    assert (
        augmented.groupby(["subject_key", "session_id", "matched_event_id"])["sample_weight"]
        .sum()
        .eq(1.0)
        .all()
    )


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
    assert refined.iloc[0].refined_start_ms == refined.iloc[0].coarse_start_ms
    assert refined.iloc[1].refined_end_ms == refined.iloc[1].coarse_end_ms
    assert refined[["start_fallback", "end_fallback", "boundary_fallback"]].all().all()


def test_boundary_neighbor_conflict_cannot_create_non_positive_event() -> None:
    accepted = pd.DataFrame(
        {
            "proposal_id": ["left", "right"],
            "subject_key": ["s", "s"],
            "session_id": ["d", "d"],
            "coarse_start_ms": [0, 123_000],
            "coarse_end_ms": [100_000, 200_000],
        }
    )
    refined = apply_boundary_refinement(
        accepted,
        np.asarray([95.0, -120.0]),
        np.asarray([150.0, -75.0]),
        np.zeros(2),
        np.zeros(2),
        entropy_threshold=0.5,
        safety_gap_seconds=3,
    )
    assert refined["refined_start_ms"].tolist() == accepted["coarse_start_ms"].tolist()
    assert refined["refined_end_ms"].tolist() == accepted["coarse_end_ms"].tolist()
    assert (refined["refined_start_ms"] < refined["refined_end_ms"]).all()


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


def test_small_ignore_overlap_does_not_exempt_long_false_prediction() -> None:
    truth = pd.DataFrame(columns=["subject_key", "session_id", "start_ms", "end_ms"])
    prediction = pd.DataFrame(
        {
            "subject_key": ["s"],
            "session_id": ["d"],
            "start_ms": [0],
            "end_ms": [100_000],
        }
    )
    ignore = pd.DataFrame(
        {
            "subject_key": ["s"],
            "session_id": ["d"],
            "start_ms": [50_000],
            "end_ms": [51_000],
        }
    )
    primary, _ = evaluate_events(truth, prediction, ignore=ignore)
    sensitivity, _ = evaluate_events(
        truth,
        prediction,
        ignore=ignore,
        ignore_policy="any_overlap",
        ignore_threshold=0.0,
    )
    assert primary["false_positive"] == 1
    assert sensitivity["ignored_predictions"] == 1


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


def test_long_pool_all_invalid_block_is_strictly_zero() -> None:
    pool = CausalCompletedBlockPool(1, 1, 5)
    values = torch.randn(1, 5, 1)
    pooled, ratio = pool(values, torch.zeros(1, 5), torch.tensor([[4]]))
    torch.testing.assert_close(pooled, torch.zeros_like(pooled), atol=0, rtol=0)
    torch.testing.assert_close(ratio, torch.zeros_like(ratio), atol=0, rtol=0)


def test_active_missing_fraction_keeps_acc_gyro_ppg_statistics_independent() -> None:
    config = _model_config()
    config.update(
        {
            "separate_motion_branches": True,
            "use_invariant_motion_branch": False,
            "use_motion": True,
            "use_ppg": True,
            "use_statistics": True,
            "use_long_context": False,
        }
    )
    model = StatsFusionStateModel(config).eval()
    batch = _state_batch()
    batch["motion_blocks"][:, :, 6:12] = 1.0
    batch["motion_blocks"][:, :, 9:12] = 0.0
    batch["statistics"][:, :, 12:] = 0.0
    with torch.no_grad():
        output = model(batch)
    torch.testing.assert_close(
        output["acc_valid_fraction"], torch.ones_like(output["acc_valid_fraction"])
    )
    torch.testing.assert_close(
        output["gyro_valid_fraction"], torch.zeros_like(output["gyro_valid_fraction"])
    )
    expected = 1.0 - torch.stack(
        (
            output["acc_valid_fraction"],
            output["gyro_valid_fraction"],
            output["ppg_valid_fraction"],
            1.0 - output["statistics_missing_fraction"],
        )
    ).mean(dim=0)
    torch.testing.assert_close(
        output["active_modality_missing_fraction"],
        expected,
    )


def test_pooled_logistic_scores_are_subject_disjoint(monkeypatch) -> None:
    proposal_count = 10
    proposals = pd.DataFrame(
        {
            "proposal_id": [f"p{index}" for index in range(proposal_count)],
            "subject_key": [f"s{index // 2}" for index in range(proposal_count)],
            "outer_fold": np.repeat(np.arange(5), 2),
            "generator_score": 0.5,
            "is_positive": np.tile([0, 1], 5),
        }
    )
    features = ProposalFeatureBatchV4(
        proposal_ids=proposals["proposal_id"].to_numpy(),
        sequence=np.arange(proposal_count, dtype=np.float32).reshape(-1, 1, 1),
        sequence_mask=np.ones((proposal_count, 1), dtype=bool),
        scalar=np.column_stack(
            (np.arange(proposal_count, dtype=np.float32), np.ones(proposal_count))
        ),
        event_target=proposals["is_positive"].to_numpy(dtype=np.float32),
        iou_target=np.zeros(proposal_count, dtype=np.float32),
        sample_weight=np.ones(proposal_count, dtype=np.float32),
    )

    def operating_point(frame, *_args, **_kwargs):
        logistic = "logistic_score" in frame
        return {
            "f1": 0.7 if logistic else 0.5,
            "fp_per_hour": 0.5 if logistic else 1.0,
            "acceptance_threshold": 0.5,
            "nms_iou_threshold": 0.5,
        }

    monkeypatch.setattr(
        "bme_eating.training.hierarchical_v4_trainer._best_verifier_operating_point",
        operating_point,
    )
    scored, _, report, _ = _fit_pooled_logistic_crossfit(
        features,
        proposals,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {
            "verifier": {"logistic_c_values": [0.01, 0.1, 1.0]},
            "promotion_gate": {
                "maximum_verifier_f1_drop": 0.005,
                "minimum_verifier_fp_reduction": 0.10,
            },
        },
    )
    assert np.isfinite(scored["logistic_score"]).all()
    assert report["promotion"]["passed"] is True
    for lineage in report["lineage"]:
        assert set(lineage["training_subjects"]).isdisjoint(lineage["prediction_subjects"])


def test_pooled_logistic_features_ignore_all_masked_bin_values() -> None:
    features = ProposalFeatureBatchV4(
        proposal_ids=np.asarray(["p0", "p1"]),
        sequence=np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ],
            dtype=np.float32,
        ),
        sequence_mask=np.asarray([[True, False, True], [False, False, False]]),
        scalar=np.asarray([[0.25], [0.75]], dtype=np.float32),
    )
    expected = pooled_logistic_features(features)
    mutated = features.sequence.copy()
    mutated[0, 1] = np.asarray([np.nan, np.inf])
    mutated[1] = np.asarray([[np.nan, np.inf], [-np.inf, np.nan], [1e30, -1e30]])
    actual = pooled_logistic_features(
        ProposalFeatureBatchV4(
            proposal_ids=features.proposal_ids,
            sequence=mutated,
            sequence_mask=features.sequence_mask,
            scalar=features.scalar,
        )
    )
    np.testing.assert_array_equal(actual, expected)


def test_pooled_logistic_falls_back_to_state_only_scores(monkeypatch) -> None:
    proposal_count = 10
    proposals = pd.DataFrame(
        {
            "proposal_id": [f"p{index}" for index in range(proposal_count)],
            "subject_key": [f"s{index // 2}" for index in range(proposal_count)],
            "outer_fold": np.repeat(np.arange(5), 2),
            "generator_score": np.linspace(0.1, 0.9, proposal_count),
            "is_positive": np.tile([0, 1], 5),
        }
    )
    features = ProposalFeatureBatchV4(
        proposal_ids=proposals["proposal_id"].to_numpy(),
        sequence=np.arange(proposal_count, dtype=np.float32).reshape(-1, 1, 1),
        sequence_mask=np.ones((proposal_count, 1), dtype=bool),
        scalar=np.column_stack(
            (np.arange(proposal_count, dtype=np.float32), np.ones(proposal_count))
        ),
        event_target=proposals["is_positive"].to_numpy(dtype=np.float32),
        iou_target=np.zeros(proposal_count, dtype=np.float32),
        sample_weight=np.ones(proposal_count, dtype=np.float32),
    )

    def operating_point(frame, *_args, **_kwargs):
        logistic = "logistic_score" in frame
        return {
            "f1": 0.4 if logistic else 0.6,
            "fp_per_hour": 1.0,
            "acceptance_threshold": 0.5,
            "nms_iou_threshold": 0.5,
        }

    monkeypatch.setattr(
        "bme_eating.training.hierarchical_v4_trainer._best_verifier_operating_point",
        operating_point,
    )
    scored, _, report, point = _fit_pooled_logistic_crossfit(
        features,
        proposals,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {
            "verifier": {"logistic_c_values": [0.01, 0.1, 1.0]},
            "promotion_gate": {
                "maximum_verifier_f1_drop": 0.005,
                "minimum_verifier_fp_reduction": 0.10,
            },
        },
    )
    assert report["promotion"]["passed"] is False
    assert scored["final_score"].to_numpy() == pytest.approx(
        proposals["generator_score"].to_numpy()
    )
    assert np.isfinite(scored["logistic_score"]).all()
    assert point["f1"] == pytest.approx(0.6)


def test_pooled_logistic_final_model_matches_deployment_scoring(monkeypatch) -> None:
    proposal_count = 10
    proposals = pd.DataFrame(
        {
            "proposal_id": [f"p{index}" for index in range(proposal_count)],
            "subject_key": [f"s{index // 2}" for index in range(proposal_count)],
            "outer_fold": np.repeat(np.arange(5), 2),
            "generator_score": np.linspace(0.1, 0.9, proposal_count),
            "is_positive": np.tile([0, 1], 5),
        }
    )
    features = ProposalFeatureBatchV4(
        proposal_ids=proposals["proposal_id"].to_numpy(),
        sequence=np.arange(proposal_count * 4, dtype=np.float32).reshape(-1, 2, 2),
        sequence_mask=np.ones((proposal_count, 2), dtype=bool),
        scalar=np.column_stack(
            (
                np.linspace(0.1, 0.9, proposal_count, dtype=np.float32),
                np.ones(proposal_count, dtype=np.float32),
                proposals["generator_score"].to_numpy(dtype=np.float32),
            )
        ),
        event_target=proposals["is_positive"].to_numpy(dtype=np.float32),
        iou_target=np.zeros(proposal_count, dtype=np.float32),
        sample_weight=np.ones(proposal_count, dtype=np.float32),
    )

    def operating_point(frame, *_args, **_kwargs):
        logistic = "logistic_score" in frame
        return {
            "f1": 0.7 if logistic else 0.5,
            "fp_per_hour": 0.5 if logistic else 1.0,
            "acceptance_threshold": 0.5,
            "nms_iou_threshold": 0.5,
        }

    monkeypatch.setattr(
        "bme_eating.training.hierarchical_v4_trainer._best_verifier_operating_point",
        operating_point,
    )
    _, final_model, report, _ = _fit_pooled_logistic_crossfit(
        features,
        proposals,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {
            "verifier": {"logistic_c_values": [0.1]},
            "promotion_gate": {
                "maximum_verifier_f1_drop": 0.005,
                "minimum_verifier_fp_reduction": 0.10,
            },
        },
    )
    assert report["promotion"]["passed"] is True
    monkeypatch.setitem(
        HierarchicalEatingDetectorV4._score_proposals.__wrapped__.__globals__,
        "build_proposal_features_v4",
        lambda *_args, **_kwargs: features,
    )
    detector = object.__new__(HierarchicalEatingDetectorV4)
    detector.selection = {"verifier_kind": "logistic"}
    detector.logistic_verifier = final_model
    detector.statistics_columns = []
    detector.config = {"verifier": {}}
    deployed = detector._score_proposals(proposals, pd.DataFrame())
    expected = final_model.predict(pooled_logistic_features(features))
    assert deployed["final_score"].to_numpy() == pytest.approx(expected)


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
        motion = np.broadcast_to(motion_value[:, None, None], (steps, 12, 300)).copy()
        statistics = np.stack(
            [np.sin(high_index / (index + 3.0)) for index in range(24)], axis=1
        ).astype(np.float32)
        low_value = np.cos(block_ends.astype(np.float64) / 41_000.0).astype(np.float32)
        ppg = np.broadcast_to(low_value[:, None, None], (len(block_ends), 2, 750)).copy()
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
    second_timestamps = (
        target_timestamp + 256 * 3_000 - np.arange(steps - 1, -1, -1, dtype=np.int64) * 3_000
    )
    second_position = int(np.flatnonzero(second_timestamps == target_timestamp)[0])
    first_block_end = first_ends[int(first["ppg_to_motion_index"][0, first_position])]
    second_block_end = second_ends[int(second["ppg_to_motion_index"][0, second_position])]
    assert first_block_end == 900_000
    assert second_block_end == 900_000

    with torch.no_grad():
        first_output = model(first)["state_logit"][0, -1]
        second_output = model(second)["state_logit"][0, second_position]
    torch.testing.assert_close(first_output, second_output, atol=1e-4, rtol=1e-5)


def test_sequence_geometry_reserves_global_block_phase_margin() -> None:
    geometry = SequenceGeometry()
    assert geometry.history_steps == 154 + 127 * 5 + 4
    assert geometry.total_steps == 1049
    for endpoint_phase in range(5):
        timestamps = (endpoint_phase - np.arange(geometry.total_steps - 1, -1, -1)) * 3_000
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


def test_state_inference_tail_chunk_emits_each_anchor_once() -> None:
    anchors = np.arange(0, 18_000, 3_000, dtype=np.int64)

    class Dataset:
        def __init__(self) -> None:
            self.geometry = SequenceGeometry(
                supervised_steps=4,
                short_receptive_field_steps=1,
                fused_short_receptive_field_steps=1,
                long_receptive_field_tokens=1,
                long_pool_factor=1,
                step_seconds=3,
                use_long_context=False,
            )
            self.session_groups = {
                ("subject", "session"): pd.DataFrame(
                    {"_row_id": np.arange(len(anchors), dtype=np.int64)}
                )
            }

        def __len__(self) -> int:
            return len(anchors)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
            end_timestamp = int(anchors[index])
            timestamps = end_timestamp - np.arange(3, -1, -1, dtype=np.int64) * 3_000
            supervision = np.isin(timestamps, anchors).astype(np.float32)
            return {
                "timestamp_ms": torch.from_numpy(timestamps),
                "supervision_mask": torch.from_numpy(supervision),
                "state_target": torch.zeros(4),
                "state_loss_mask": torch.ones(4),
                "statistics": torch.zeros((4, 24)),
                "subject_key": "subject",
                "session_id": "session",
            }

    class PositionModel(torch.nn.Module):
        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            absolute_position = batch["timestamp_ms"].to(torch.float32) / 3_000.0
            names = (
                "state_logit",
                "onset_logit",
                "offset_logit",
                "ppg_gate",
                "statistics_gate",
                "long_gate",
                "missing_fraction",
                "active_modality_missing_fraction",
                "motion_valid_fraction",
                "acc_valid_fraction",
                "gyro_valid_fraction",
                "gyro_gate",
                "invariant_gate",
                "ppg_valid_fraction",
                "statistics_missing_fraction",
            )
            return {name: absolute_position for name in names}

    output = infer_state_windows(
        PositionModel(),
        Dataset(),
        {"training": {"device": "cpu", "inference_batch_size": 2, "num_workers": 0}},
        stacking_partition=0,
    )
    assert output["timestamp_ms"].tolist() == anchors.tolist()
    assert not output.duplicated(["subject_key", "session_id", "timestamp_ms"]).any()


def test_timeline_ownership_compares_overlap_before_claiming_rows() -> None:
    first = pd.DataFrame(
        {
            "subject_key": ["subject"] * 4,
            "session_id": ["session"] * 4,
            "timestamp_ms": [0, 3_000, 6_000, 9_000],
            "state_logit": [0.0, 1.0, 2.0, 3.0],
        }
    )
    tail = pd.DataFrame(
        {
            "subject_key": ["subject"] * 3,
            "session_id": ["session"] * 3,
            "timestamp_ms": [6_000, 9_000, 12_000],
            "state_logit": [20.0, 30.0, 40.0],
        }
    )
    ownership: dict[tuple[str, str], pd.DataFrame] = {}
    claimed_first = claim_new_timeline_rows(first, ownership)
    with pytest.raises(RuntimeError, match="disagree at duplicate anchors"):
        claim_new_timeline_rows(tail, ownership)
    consistent_tail = tail.copy()
    consistent_tail["state_logit"] = [2.0, 3.0, 4.0]
    claimed_tail = claim_new_timeline_rows(consistent_tail, ownership)
    assert claimed_tail["timestamp_ms"].tolist() == [12_000]
    combined = pd.concat((claimed_first, claimed_tail), ignore_index=True)
    deduplicated, diagnostics = deduplicate_consistent_timeline(combined)
    assert deduplicated["timestamp_ms"].tolist() == [0, 3_000, 6_000, 9_000, 12_000]
    assert diagnostics["duplicate_groups"] == 0
    with pytest.raises(RuntimeError, match="disagree at duplicate anchors"):
        deduplicate_consistent_timeline(pd.concat((first, tail), ignore_index=True))


def test_dataset_and_raw_preprocessor_share_completed_block_phase(tmp_path) -> None:
    geometry = SequenceGeometry(
        supervised_steps=8,
        short_receptive_field_steps=3,
        fused_short_receptive_field_steps=3,
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
        subject_key="s",
        session_id="d",
        motion_timestamp_ms=motion_time,
        motion_values=motion_values,
        motion_mask=np.ones_like(motion_values, dtype=bool),
        ppg_timestamp_ms=ppg_time,
        ppg_values=ppg_values,
        ppg_mask=np.ones(len(ppg_values), dtype=bool),
    )
    raw_batch = list(preprocessor.iter_state_batches(raw_session))[-1]
    torch.testing.assert_close(
        raw_batch["motion_invariant_blocks"][0],
        dataset_batch["motion_invariant_blocks"],
        atol=0,
        rtol=0,
    )
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


def test_physical_motion_invariants_survive_signed_permutation_rotation() -> None:
    timestamps = np.arange(300, dtype=np.int64) * 10
    rng = np.random.default_rng(2026)
    values = rng.normal(0.0, 0.05, size=(300, 6)).astype(np.float32)
    mask = np.ones_like(values, dtype=bool)
    normalization = Normalization(np.zeros(6), np.ones(6), 0.0, 1.0)
    rotation = np.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    _, _, baseline = build_motion_blocks(
        timestamp_ms=timestamps,
        values=values,
        mask=mask,
        normalization=normalization,
        first_timestamp_ms=3_000,
        steps=1,
        step_seconds=3,
        return_invariant=True,
    )
    _, _, rotated = build_motion_blocks(
        timestamp_ms=timestamps,
        values=values,
        mask=mask,
        normalization=normalization,
        first_timestamp_ms=3_000,
        steps=1,
        step_seconds=3,
        rotation_matrix=rotation,
        return_invariant=True,
    )
    np.testing.assert_allclose(rotated, baseline, atol=1e-6, rtol=1e-6)


def test_motion_only_normalization_does_not_require_ppg(tmp_path) -> None:
    archive = tmp_path / "motion-only.npz"
    timestamps = np.arange(300, dtype=np.int64) * 10
    values = np.ones((300, 6), dtype=np.float32)
    np.savez(
        archive,
        motion_timestamp_ms=timestamps,
        motion_values=values,
        motion_mask=np.ones_like(values, dtype=bool),
        ppg_timestamp_ms=np.asarray([], dtype=np.int64),
        ppg_values=np.asarray([], dtype=np.float32),
        ppg_mask=np.asarray([], dtype=bool),
    )
    segments = pd.DataFrame({"subject_key": ["s"], "segment_path": [str(archive)]})
    normalization = compute_normalization(segments, {"s"}, require_motion=True, require_ppg=False)
    assert normalization.ppg_median == 0.0
    assert normalization.ppg_iqr == 1.0
    with pytest.raises(ValueError, match="PPG"):
        compute_normalization(segments, {"s"}, require_motion=True, require_ppg=True)


def test_normalization_ignores_nonfinite_valid_samples_and_empty_invariant_channels(
    tmp_path,
) -> None:
    archive = tmp_path / "partial-modalities.npz"
    timestamps = np.arange(300, dtype=np.int64) * 10
    values = np.ones((300, 6), dtype=np.float32)
    values[5, 0] = np.nan
    motion_mask = np.ones_like(values, dtype=bool)
    motion_mask[:, 3:] = False
    ppg_values = np.ones(150, dtype=np.float32)
    ppg_values[7] = np.inf
    np.savez(
        archive,
        motion_timestamp_ms=timestamps,
        motion_values=values,
        motion_mask=motion_mask,
        ppg_timestamp_ms=np.arange(150, dtype=np.int64) * 20,
        ppg_values=ppg_values,
        ppg_mask=np.ones(150, dtype=bool),
    )
    segments = pd.DataFrame({"subject_key": ["s"], "segment_path": [str(archive)]})
    normalization = compute_normalization(segments, {"s"}, require_motion=True, require_ppg=True)
    assert np.isfinite(normalization.motion_median).all()
    assert np.isfinite(normalization.motion_iqr).all()
    assert np.isfinite(normalization.motion_invariant_median).all()
    assert np.isfinite(normalization.motion_invariant_iqr).all()
    assert np.isfinite(normalization.ppg_median)
    assert np.isfinite(normalization.ppg_iqr)


def test_raw_session_validation_rejects_nonfinite_valid_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        RawSessionInput(
            subject_key="s",
            session_id="d",
            motion_timestamp_ms=np.asarray([0]),
            motion_values=np.asarray([[np.nan] * 6], dtype=np.float32),
            motion_mask=np.ones((1, 6), dtype=bool),
            ppg_timestamp_ms=np.asarray([], dtype=np.int64),
            ppg_values=np.asarray([], dtype=np.float32),
            ppg_mask=np.asarray([], dtype=bool),
        ).validated()


def test_raw_session_reports_long_sensor_gaps_without_hiding_them() -> None:
    motion_time = np.concatenate(
        (np.arange(200, dtype=np.int64) * 10, 3_600_000 + np.arange(200) * 10)
    )
    session = RawSessionInput(
        subject_key="s",
        session_id="d",
        motion_timestamp_ms=motion_time,
        motion_values=np.zeros((len(motion_time), 6), dtype=np.float32),
        motion_mask=np.ones((len(motion_time), 6), dtype=bool),
        ppg_timestamp_ms=np.asarray([], dtype=np.int64),
        ppg_values=np.asarray([], dtype=np.float32),
        ppg_mask=np.asarray([], dtype=bool),
    ).validated()
    diagnostics = session.sampling_diagnostics()
    assert diagnostics["motion_gaps_over_3s"] == 1
    assert float(diagnostics["motion_max_gap_ms"]) > 3_000_000


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"schema_version": "statsfusion-raw-v1"}, "schema_version"),
        ({"timestamp_unit": "s"}, "milliseconds"),
        (
            {
                "motion_channel_order": (
                    "gyro_x",
                    "gyro_y",
                    "gyro_z",
                    "acc_x",
                    "acc_y",
                    "acc_z",
                )
            },
            "channel order",
        ),
        ({"unit_contract_id": "unknown"}, "sensor units"),
        ({"resampling_state": "resampled"}, "native, unresampled"),
    ],
)
def test_raw_session_v2_contract_rejects_unknown_semantics(override, message) -> None:
    arguments = {
        "subject_key": "s",
        "session_id": "d",
        "motion_timestamp_ms": np.arange(4, dtype=np.int64) * 10,
        "motion_values": np.zeros((4, 6), dtype=np.float32),
        "motion_mask": np.ones((4, 6), dtype=bool),
        "ppg_timestamp_ms": np.arange(4, dtype=np.int64) * 20,
        "ppg_values": np.zeros(4, dtype=np.float32),
        "ppg_mask": np.ones(4, dtype=bool),
        **override,
    }
    with pytest.raises(ValueError, match=message):
        RawSessionInput(**arguments).validated()


def test_raw_session_v2_contract_rejects_out_of_distribution_sampling_rate() -> None:
    with pytest.raises(ValueError, match="sampling rate"):
        RawSessionInput(
            subject_key="s",
            session_id="d",
            motion_timestamp_ms=np.arange(4, dtype=np.int64) * 100,
            motion_values=np.zeros((4, 6), dtype=np.float32),
            motion_mask=np.ones((4, 6), dtype=bool),
            ppg_timestamp_ms=np.asarray([], dtype=np.int64),
            ppg_values=np.asarray([], dtype=np.float32),
            ppg_mask=np.asarray([], dtype=bool),
        ).validated()


def test_raw_session_anchors_preserve_training_session_phase() -> None:
    first_ms = 2_800
    last_ms = 12_780
    timestamps = np.arange(first_ms, last_ms + 1, 10, dtype=np.int64)
    session = RawSessionInput(
        subject_key="s",
        session_id="d",
        motion_timestamp_ms=timestamps,
        motion_values=np.zeros((len(timestamps), 6), dtype=np.float32),
        motion_mask=np.ones((len(timestamps), 6), dtype=bool),
        ppg_timestamp_ms=np.asarray([], dtype=np.int64),
        ppg_values=np.asarray([], dtype=np.float32),
        ppg_mask=np.asarray([], dtype=bool),
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


def test_raw_session_tail_chunks_supervise_each_anchor_once() -> None:
    motion_time = np.arange(0, 18_001, 10, dtype=np.int64)
    session = RawSessionInput(
        subject_key="s",
        session_id="d",
        motion_timestamp_ms=motion_time,
        motion_values=np.zeros((len(motion_time), 6), dtype=np.float32),
        motion_mask=np.ones((len(motion_time), 6), dtype=bool),
        ppg_timestamp_ms=np.asarray([], dtype=np.int64),
        ppg_values=np.asarray([], dtype=np.float32),
        ppg_mask=np.asarray([], dtype=bool),
    )
    scaler = FoldRobustScaler(
        STATS_FEATURE_COLUMNS,
        np.zeros(len(STATS_FEATURE_COLUMNS)),
        np.ones(len(STATS_FEATURE_COLUMNS)),
        ("train",),
    )
    geometry = SequenceGeometry(
        supervised_steps=4,
        short_receptive_field_steps=1,
        fused_short_receptive_field_steps=1,
        long_receptive_field_tokens=1,
        long_pool_factor=1,
        step_seconds=3,
        use_long_context=False,
    )
    preprocessor = StatsFusionRawSessionPreprocessor(
        normalization=Normalization(np.zeros(6), np.ones(6), 0.0, 1.0),
        statistics_scaler=scaler,
        geometry=geometry,
    )
    supervised_timestamps: list[int] = []
    direct_batches = list(preprocessor.iter_state_batches(session))
    delegated_batches = list(_iter_tail_aligned_raw_state_batches(preprocessor, session))
    assert len(delegated_batches) == len(direct_batches)
    for direct, delegated in zip(direct_batches, delegated_batches, strict=True):
        assert direct.keys() == delegated.keys()
        for key in direct:
            torch.testing.assert_close(direct[key], delegated[key], atol=0, rtol=0)
    for batch in direct_batches:
        mask = batch["supervision_mask"][0].bool().numpy()
        supervised_timestamps.extend(batch["timestamp_ms"][0].numpy()[mask].tolist())
    assert supervised_timestamps == preprocessor.anchors(session).tolist()


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
        subject_key="s",
        session_id="d",
        motion_timestamp_ms=motion_time,
        motion_values=motion_values,
        motion_mask=np.ones_like(motion_values, dtype=bool),
        ppg_timestamp_ms=ppg_time,
        ppg_values=ppg_values,
        ppg_mask=np.ones(len(ppg_values), dtype=bool),
    )
    scaler = FoldRobustScaler(STATS_FEATURE_COLUMNS, np.zeros(12), np.ones(12), ("train",))
    preprocessor = StatsFusionRawSessionPreprocessor(
        normalization=Normalization(np.zeros(6), np.ones(6), 0.0, 1.0),
        statistics_scaler=scaler,
        geometry=SequenceGeometry(),
    )
    assert preprocessor.anchors(session).tolist() == expected.tolist()


def test_canonical_anchor_inside_raw_motion_gap_is_not_supervised(tmp_path) -> None:
    first = np.arange(0, 3_001, 10, dtype=np.int64)
    second = np.arange(60_000, 63_001, 10, dtype=np.int64)
    motion_time = np.concatenate((first, second))
    motion_values = np.ones((len(motion_time), 6), dtype=np.float32)
    archive = tmp_path / "gapped.npz"
    np.savez(
        archive,
        motion_timestamp_ms=motion_time,
        motion_values=motion_values,
        motion_mask=np.ones_like(motion_values, dtype=bool),
        ppg_timestamp_ms=np.asarray([], dtype=np.int64),
        ppg_values=np.asarray([], dtype=np.float32),
        ppg_mask=np.asarray([], dtype=bool),
    )
    segments = pd.DataFrame(
        {
            "segment_id": ["g0"],
            "session_id": ["d"],
            "segment_path": [str(archive)],
            "subject_key": ["s"],
            "start_ms": [0],
            "end_ms": [63_000],
        }
    )
    events = pd.DataFrame(
        columns=[
            "event_id",
            "subject_key",
            "session_id",
            "start_ms",
            "end_ms",
            "valid_duration",
        ]
    )
    anchors = build_statsfusion_session_anchor_index(segments, events)
    gap_anchor = anchors.loc[anchors["timestamp_ms"].eq(30_000)].iloc[0]
    assert gap_anchor.state_loss_mask == 0.0
    assert gap_anchor.censor_mask == 1.0


def test_canonical_targets_do_not_cross_sessions_for_same_subject(tmp_path) -> None:
    segments = pd.DataFrame(
        {
            "segment_id": ["g0", "g1"],
            "session_id": ["d0", "d1"],
            "segment_path": [str(tmp_path / "missing0.npz"), str(tmp_path / "missing1.npz")],
            "subject_key": ["s", "s"],
            "start_ms": [0, 0],
            "end_ms": [12_000, 12_000],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["e0"],
            "subject_key": ["s"],
            "session_id": ["d0"],
            "start_ms": [0],
            "end_ms": [6_000],
            "valid_duration": [True],
            "evaluable": [True],
        }
    )
    anchors = build_statsfusion_session_anchor_index(segments, events)
    assert anchors.loc[anchors["session_id"].eq("d0"), "state_target"].max() == 1.0
    assert anchors.loc[anchors["session_id"].eq("d1"), "state_target"].max() == 0.0


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
            "subject_macro_soft_bce": 0.4,
            "soft_bce_standard_error": 0.02,
            "center_window_auprc": 0.2,
        },
        {
            "candidate_recall": 0.9,
            "calibration_passed": False,
            "event_f1": 0.8,
            "state_fragment_count": 80.0,
            "ece": 0.2,
            "subject_macro_soft_bce": 0.3,
            "soft_bce_standard_error": 0.02,
            "center_window_auprc": 0.9,
        },
        {
            "candidate_recall": 0.3,
            "calibration_passed": True,
            "event_f1": 0.2,
            "state_fragment_count": 25.0,
            "ece": 0.05,
            "subject_macro_soft_bce": 0.35,
            "soft_bce_standard_error": 0.02,
            "center_window_auprc": 0.3,
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


def test_unqualified_state_epoch_selection_prefers_earliest_stable_epoch() -> None:
    metrics = [
        {
            "epoch": epoch,
            "robust_candidate_recall": recall,
            "robust_subject_macro_soft_bce": bce,
            "robust_soft_bce_standard_error": 0.01,
            "robust_state_fragment_count": fragments,
            "robust_ece": 0.01,
        }
        for epoch, recall, bce, fragments in (
            (4, 0.790, 0.120, 20),
            (5, 0.792, 0.115, 18),
            (6, 0.792, 0.110, 16),
        )
    ]
    selected, promotion_eligible = _choose_conservative_state_epoch(
        metrics,
        [],
        minimum_delta=0.003,
    )
    assert promotion_eligible is False
    assert selected["epoch"] == 4


def test_state_epoch_selection_respects_minimum_training_epoch() -> None:
    metrics = [
        {
            "epoch": epoch,
            "robust_candidate_recall": recall,
            "robust_subject_macro_soft_bce": bce,
            "robust_soft_bce_standard_error": 0.01,
            "robust_state_fragment_count": 20 - epoch,
            "robust_ece": 0.01,
        }
        for epoch, recall, bce in (
            (4, 0.792, 0.120),
            (5, 0.791, 0.115),
            (6, 0.790, 0.110),
        )
    ]
    selected, promotion_eligible = _choose_conservative_state_epoch(
        metrics,
        [],
        minimum_delta=0.003,
        minimum_epoch=5,
    )
    assert promotion_eligible is False
    assert selected["epoch"] == 5


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


@pytest.mark.parametrize("lag_seconds", [0, 15, 30, 60])
def test_semi_markov_random_sequences_always_respect_duration_bounds(
    lag_seconds: int,
) -> None:
    prior = TruncatedLogNormalDurationPrior(
        log_mean=float(np.log(45.0)),
        log_standard_deviation=0.5,
        minimum_seconds=15.0,
        maximum_seconds=75.0,
    )
    decoder = FixedLagSemiMarkovDecoder(prior, grid_seconds=15, fixed_lag_seconds=lag_seconds)
    rng = np.random.default_rng(2026 + lag_seconds)
    cases = [
        np.full(40, 0.99),
        np.tile(np.asarray([0.99, 0.01]), 20),
        np.full(40, 0.01),
        *[rng.uniform(0.0, 1.0, size=length) for length in range(1, 50)],
    ]
    for probability in cases:
        timestamps = np.arange(1, len(probability) + 1, dtype=np.int64) * 15_000
        for start, end, _ in decoder.decode_events(timestamps, probability):
            assert 15_000 <= end - start <= 75_000


def test_three_second_candidate_can_match_two_second_truth_strictly_above_iou_threshold() -> None:
    windows = pd.DataFrame(
        {
            "subject_key": ["s", "s"],
            "session_id": ["d", "d"],
            "timestamp_ms": [3_000, 6_000],
            "state_probability": [0.9, 0.0],
            "onset_probability": [0.0, 0.0],
            "offset_probability": [0.0, 0.0],
        }
    )
    prior = TruncatedLogNormalDurationPrior(
        log_mean=float(np.log(30.0)),
        log_standard_deviation=0.5,
        minimum_seconds=15.0,
        maximum_seconds=60.0,
    )
    proposals = generate_event_candidates_v4(
        windows,
        FixedLagSemiMarkovDecoder(prior, grid_seconds=15, fixed_lag_seconds=60),
        {
            "use_semi_markov": False,
            "ema_half_life_seconds": 0,
            "high_threshold": 0.5,
            "low_threshold": 0.2,
            "gap_merge_seconds": 0,
            "transition_threshold": 0.9,
            "candidate_minimum_seconds": 3,
            "candidate_maximum_seconds": 14_400,
            "grid_seconds": 15,
            "jitter_seconds": [0],
            "maximum_variants_per_event": 1,
            "maximum_candidates_per_hour": 20,
            "deduplication_iou": 0.9,
        },
        split_role="test",
    )
    truth = pd.DataFrame(
        {
            "subject_key": ["s"],
            "session_id": ["d"],
            "start_ms": [500],
            "end_ms": [2_500],
        }
    )
    metrics, _ = evaluate_events(
        truth,
        proposals.rename(
            columns={
                "coarse_start_ms": "start_ms",
                "coarse_end_ms": "end_ms",
            }
        ),
    )
    assert metrics["true_positive"] == 1


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
            "session_id": ["d", "d"],
            "start_ms": [0, 30_000],
            "end_ms": [9_000, 40_000],
            "valid_duration": [True, True],
            "evaluable": [False, True],
        }
    )
    dataset = StatsFusionSequenceDataset(
        anchors,
        pd.DataFrame(columns=["session_id", "segment_id", "segment_path", "start_ms", "end_ms"]),
        events,
        Normalization(np.zeros(6), np.ones(6), 0.0, 1.0),
        statistics_columns=[
            *(f"stat_{name}" for name in STATS_FEATURE_COLUMNS),
            *(f"stat_{name}_missing" for name in STATS_FEATURE_COLUMNS),
        ],
    )
    assert len(dataset.events) == 1
    assert len(dataset.ignore_events) == 1
    assert dataset.anchors["state_loss_mask"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert dataset.anchors["start_loss_mask"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert dataset.anchors["end_loss_mask"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert dataset.anchors["smooth_loss_mask"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert dataset.anchors["start_target"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert dataset.anchors["end_target"].tolist() == [0.0, 0.0, 0.0, 0.0]


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


def test_state_calibration_recomputes_deployment_probability_and_derivative() -> None:
    frame = pd.DataFrame(
        {
            "subject_key": ["s", "s", "s"],
            "session_id": ["a", "a", "b"],
            "timestamp_ms": [6_000, 3_000, 3_000],
            "state_logit": [1.0, 0.0, -1.0],
            "state_probability": [0.99, 0.99, 0.99],
        }
    )
    calibrator = PlattCalibration(coefficient=2.0, intercept=-0.5)
    output = _apply_state_calibration_to_windows(frame, calibrator)
    expected = calibrator.transform(np.asarray([0.0, 1.0, -1.0]))
    assert output["timestamp_ms"].tolist() == [3_000, 6_000, 3_000]
    assert output["state_probability"].to_numpy() == pytest.approx(expected)
    assert output["state_probability_derivative"].to_numpy() == pytest.approx(
        [0.0, expected[1] - expected[0], 0.0]
    )


def test_state_only_pipeline_uses_generator_score_without_verifier() -> None:
    detector = object.__new__(HierarchicalEatingDetectorV4)
    detector.selection = {"verifier_kind": "state_only"}
    proposals = pd.DataFrame(
        {
            "proposal_id": ["p0", "p1"],
            "generator_score": [0.25, 0.75],
        }
    )
    scored = detector._score_proposals(proposals, pd.DataFrame())
    assert scored["state_score"].tolist() == [0.25, 0.75]
    assert scored["final_score"].tolist() == [0.25, 0.75]
    assert scored["event_logit"].isna().all()
    assert scored["predicted_iou"].isna().all()


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


def test_current_r32_canonical_manifest_mask_count_matches_anchors_when_available() -> None:
    root = Path(__file__).resolve().parents[3] / "outputs" / "v4" / "canonical_input_r3_2"
    manifest_path = root / "manifest.json"
    anchor_path = root / "anchors.parquet"
    if not manifest_path.is_file() or not anchor_path.is_file():
        pytest.skip("Current r3.2 canonical input is not available")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "canonical_state_loss_masked_anchors" not in manifest.get("counts", {}):
        pytest.skip("Current canonical input predates clarified mask-count diagnostics")
    anchors = pd.read_parquet(anchor_path, columns=["state_loss_mask"])
    actual = int((anchors["state_loss_mask"].fillna(0.0).astype(float) <= 0).sum())
    assert manifest["counts"]["anchors"] == len(anchors)
    assert manifest["counts"]["canonical_state_loss_masked_anchors"] == actual
    assert (
        manifest["counts"]["effective_state_loss_masked_anchors"]
        >= manifest["counts"]["canonical_state_loss_masked_anchors"]
    )


def test_outer_state_labels_mask_ignore_intervals() -> None:
    frame = pd.DataFrame(
        {
            "subject_key": ["s", "s", "s", "s"],
            "session_id": ["d0", "d0", "d0", "d1"],
            "timestamp_ms": [3_000, 6_000, 12_000, 6_000],
            "state_loss_mask": [1.0, 1.0, 1.0, 1.0],
        }
    )
    ignore = pd.DataFrame(
        {
            "subject_key": ["s"],
            "session_id": ["d0"],
            "start_ms": [0],
            "end_ms": [9_000],
        }
    )
    masked = _mask_ignored_state_rows(frame, ignore, step_ms=3_000)
    assert masked["state_loss_mask"].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_pooled_proposal_alignment_rejects_reordered_rows() -> None:
    features = ProposalFeatureBatchV4(
        proposal_ids=np.asarray(["p1", "p2"]),
        sequence=np.zeros((2, 1, 1), dtype=np.float32),
        sequence_mask=np.ones((2, 1), dtype=bool),
        scalar=np.zeros((2, 1), dtype=np.float32),
        event_target=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    frame = pd.DataFrame({"proposal_id": ["p2", "p1"], "is_positive": [False, True]})
    with pytest.raises(RuntimeError, match="not aligned"):
        _assert_proposal_feature_alignment(features, frame, context="test")
