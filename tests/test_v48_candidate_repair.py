from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from bme_eating.config import load_config
from bme_eating.models.endpoint_refiner import boundary_geometric_feasibility
from bme_eating.models.event_verifier_v4 import (
    EventVerifierV4,
    _raw_imu_proposal_snippets,
    build_proposal_features_v4,
)
from bme_eating.models.stats_fusion_loss import StatsFusionStateLoss
from bme_eating.proposals_v4 import (
    ProposalSource,
    _jitter,
    generate_event_candidates_v4,
    interval_iou,
)
from bme_eating.structured_decoder import (
    FixedLagSemiMarkovDecoder,
    TruncatedLogNormalDurationPrior,
)
from bme_eating.training.hierarchical_v4_trainer import (
    _paired_boundary_matches,
    _v48_deep_gate,
)
from bme_eating.training.v48_targets import V48ProposalSequenceDataset
from bme_eating.v4_protocol import validate_r3_config
from scripts.compare_v48_raw_imu import compare


def test_v48_jitter_has_both_thirty_second_directions() -> None:
    variants = _jitter(
        [(100_000, 200_000, 1.0, int(ProposalSource.HYSTERESIS), "family")],
        [-30, -15, -3, 0, 3, 15, 30],
        17,
        3_000,
        14_400_000,
        observation_start_ms=0,
        observation_end_ms=300_000,
        symmetric_v48=True,
    )
    assert len(variants) == 17
    assert {(start, end) for start, end, *_ in variants} >= {
        (70_000, 200_000),
        (130_000, 200_000),
        (100_000, 170_000),
        (100_000, 230_000),
        (85_000, 215_000),
        (115_000, 185_000),
    }
    assert all(0 <= start < end <= 300_000 for start, end, *_ in variants)


def test_v48_jitter_discards_out_of_session_variants() -> None:
    variants = _jitter(
        [(3_000, 9_000, 1.0, int(ProposalSource.HYSTERESIS), "family")],
        [-30, -15, -3, 0, 3, 15, 30],
        17,
        3_000,
        14_400_000,
        observation_start_ms=0,
        observation_end_ms=12_000,
        symmetric_v48=True,
    )
    assert all(0 <= start < end <= 12_000 for start, end, *_ in variants)
    assert len(variants) < 17


def test_v48_short_truth_can_match_three_second_candidate() -> None:
    assert interval_iou(3_000, 5_000, 3_000, 6_000) > 0.25


def test_v48_proposal_loss_ignores_masked_points() -> None:
    criterion = StatsFusionStateLoss(
        smooth_weight=0.0, smooth_beta=0.5, boundary_weight=0.0,
        proposal_weight=0.1,
    )
    proposal = torch.tensor([[0.2, -0.4, 1.0]], requires_grad=True)
    state = torch.zeros_like(proposal, requires_grad=True)
    output = {
        "state_logit": state,
        "onset_logit": state * 0,
        "offset_logit": state * 0,
        "proposal_logit": proposal,
    }
    batch = {
        "supervision_mask": torch.ones(1, 3),
        "state_loss_mask": torch.tensor([[1.0, 0.0, 1.0]]),
        "state_target": torch.zeros(1, 3),
        "onset_target": torch.zeros(1, 3),
        "offset_target": torch.zeros(1, 3),
        "proposal_target": torch.tensor([[0.0, 1.0, 1.0]]),
        "proposal_weight": torch.ones(1, 3),
    }
    loss, components = criterion(output, batch)
    loss.backward()
    assert components["proposal"].isfinite()
    assert proposal.grad is not None
    assert proposal.grad[0, 1] == 0
    assert proposal.grad[0, 0] != 0


def test_v48_proposal_labels_are_session_scoped() -> None:
    anchors = pd.DataFrame({
        "subject_key": ["subject"] * 4,
        "session_id": ["one", "one", "two", "two"],
        "timestamp_ms": [3_000, 6_000, 3_000, 6_000],
        "state_target": [0.0] * 4,
        "state_loss_mask": [1.0] * 4,
        **{f"stat_{index}": [0.0] * 4 for index in range(24)},
    })
    events = pd.DataFrame({
        "subject_key": ["subject"], "session_id": ["one"],
        "start_ms": [2_000], "end_ms": [4_000],
        "valid_duration": [True], "evaluable": [True],
    })
    segments = pd.DataFrame(columns=[
        "subject_key", "session_id", "segment_id", "segment_path", "start_ms", "end_ms"
    ])
    dataset = V48ProposalSequenceDataset(
        anchors, segments, events, object(),
        statistics_columns=[f"stat_{index}" for index in range(24)],
    )
    assert dataset.anchors.loc[dataset.anchors["session_id"].eq("one"), "proposal_target"].any()
    assert not dataset.anchors.loc[dataset.anchors["session_id"].eq("two"), "proposal_target"].any()


def test_v48_candidate_budget_and_proposal_head() -> None:
    timestamps = np.arange(3_000, 3_603_000, 3_000)
    windows = pd.DataFrame({
        "subject_key": "subject",
        "session_id": "session",
        "timestamp_ms": timestamps,
        "state_probability": np.zeros(len(timestamps)) + 0.01,
        "onset_probability": np.zeros(len(timestamps)),
        "offset_probability": np.zeros(len(timestamps)),
        "proposal_logit": np.where((np.arange(len(timestamps)) % 30) == 0, 8.0, -8.0),
    })
    prior = TruncatedLogNormalDurationPrior.fit(np.array([60.0, 120.0, 300.0]))
    decoder = FixedLagSemiMarkovDecoder(prior, grid_seconds=15, fixed_lag_seconds=60)
    config = load_config("configs/hierarchical_v4_v48_proposal_head.yaml")["decoder"]
    proposals = generate_event_candidates_v4(windows, decoder, config, split_role="test")
    assert len(proposals) <= 21
    assert ((proposals["source_mask"] & int(ProposalSource.PROPOSAL_HEAD)) > 0).sum() <= 2
    assert (proposals["coarse_start_ms"] >= 0).all()


def test_v48_synthetic_candidate_to_deep_forward() -> None:
    config = load_config("configs/hierarchical_v4_v48_proposal_head.yaml")
    timestamps = np.arange(3_000, 603_000, 3_000)
    active = (timestamps >= 120_000) & (timestamps <= 210_000)
    windows = pd.DataFrame({
        "subject_key": "subject", "session_id": "session",
        "timestamp_ms": timestamps,
        "state_probability": np.where(active, 0.12, 0.01),
        "onset_probability": np.zeros(len(timestamps)),
        "offset_probability": np.zeros(len(timestamps)),
        "proposal_logit": np.where(timestamps == 180_000, 8.0, -8.0),
    })
    for name in (
        "ppg_gate", "statistics_gate", "long_gate", "gyro_gate", "invariant_gate",
        "missing_fraction", "acc_valid_fraction", "gyro_valid_fraction",
        "ppg_valid_fraction", "statistics_missing_fraction",
        "state_probability_derivative", "stat_example",
        *(f"state_hidden_{index:03d}" for index in range(64)),
    ):
        windows[name] = 0.0
    windows["acc_valid_fraction"] = 1.0
    prior = TruncatedLogNormalDurationPrior.fit(np.array([60.0, 120.0, 300.0]))
    decoder = FixedLagSemiMarkovDecoder(prior, grid_seconds=15, fixed_lag_seconds=60)
    proposals = generate_event_candidates_v4(
        windows, decoder, config["decoder"], split_role="synthetic_v48"
    )
    assert not proposals.empty
    features = build_proposal_features_v4(
        proposals, windows, ["stat_example"], config["verifier"]
    )
    model = EventVerifierV4(
        features.sequence.shape[-1], features.scalar.shape[-1], config["verifier"]
    )
    model.eval()
    with torch.no_grad():
        result = model({
            "sequence": torch.from_numpy(features.sequence),
            "sequence_mask": torch.from_numpy(features.sequence_mask),
            "scalar": torch.from_numpy(features.scalar),
        })
    assert result["event_logit"].shape == (len(proposals),)
    assert torch.isfinite(result["event_logit"]).all()


def test_v48_raw_imu_snippets_are_causal_and_mask_gyro() -> None:
    timestamps = np.arange(0, 120_000, 10, dtype=np.int64)
    values = np.ones((len(timestamps), 6), dtype=np.float32)
    mask = np.ones_like(values, dtype=bool)
    mask[:, 3:] = False
    first = _raw_imu_proposal_snippets(timestamps, values, mask, 30_000, 60_000)
    changed = values.copy()
    changed[timestamps > 90_000] = 99.0
    second = _raw_imu_proposal_snippets(timestamps, changed, mask, 30_000, 60_000)
    assert first.shape == (3, 12, 300)
    assert np.array_equal(first, second)
    assert not first[:, 9:, :].any()
    model = EventVerifierV4(4, 3, {"use_raw_imu_branch": True})
    output = model({
        "sequence": torch.zeros(2, 5, 4),
        "sequence_mask": torch.ones(2, 5, dtype=torch.bool),
        "scalar": torch.zeros(2, 3),
        "raw_imu": torch.from_numpy(np.stack((first, first)).astype(np.float32)),
    })
    assert output["event_logit"].shape == (2,)
    assert torch.isfinite(output["event_logit"]).all()


def test_v48_raw_imu_training_and_runtime_features_match(tmp_path) -> None:
    timestamps = np.arange(0, 120_000, 10, dtype=np.int64)
    values = np.stack([
        np.sin(timestamps / 1000.0 + channel) for channel in range(6)
    ], axis=1).astype(np.float32)
    mask = np.ones_like(values, dtype=bool)
    archive = tmp_path / "segment.npz"
    np.savez(
        archive,
        motion_timestamp_ms=timestamps,
        motion_values=values,
        motion_mask=mask,
        ppg_timestamp_ms=np.empty(0, dtype=np.int64),
        ppg_values=np.empty((0, 1), dtype=np.float32),
        ppg_mask=np.empty((0, 1), dtype=bool),
    )
    segments = pd.DataFrame({
        "subject_key": ["subject"], "session_id": ["session"],
        "segment_id": ["segment"], "segment_path": [str(archive)],
        "start_ms": [0], "end_ms": [119_990],
    })
    proposals = pd.DataFrame({
        "proposal_id": ["proposal"], "subject_key": ["subject"],
        "session_id": ["session"], "coarse_start_ms": [30_000],
        "coarse_end_ms": [60_000], "generator_score": [0.5],
        "source_mask": [int(ProposalSource.HYSTERESIS | ProposalSource.START_EXPANSION
                            | ProposalSource.PROPOSAL_HEAD)],
    })
    statistics_columns = ["stat_example"]
    windows = pd.DataFrame({
        "subject_key": "subject", "session_id": "session",
        "timestamp_ms": np.arange(3_000, 120_000, 3_000),
    })
    for column in (
        "state_probability", "onset_probability", "offset_probability",
        "ppg_gate", "statistics_gate", "long_gate", "gyro_gate", "invariant_gate",
        "missing_fraction", "acc_valid_fraction", "gyro_valid_fraction",
        "ppg_valid_fraction", "statistics_missing_fraction", "stat_example",
    ):
        windows[column] = 0.5
    config = {
        "use_raw_imu_branch": True, "left_context_seconds": 15,
        "right_context_seconds": 15, "left_bins": 2, "event_bins": 4,
        "right_bins": 2, "include_v48_source_flags": True,
    }
    runtime = build_proposal_features_v4(
        proposals, windows, statistics_columns, config,
        raw_session=SimpleNamespace(
            subject_key="subject", session_id="session",
            motion_timestamp_ms=timestamps, motion_values=values, motion_mask=mask,
        ),
    )
    training = build_proposal_features_v4(
        proposals, windows, statistics_columns, config, raw_segments=segments,
    )
    assert runtime.raw_imu is not None and training.raw_imu is not None
    assert np.array_equal(runtime.raw_imu, training.raw_imu)
    assert runtime.scalar.shape == (1, 11)
    assert runtime.scalar[0, 8] == 1.0
    assert runtime.scalar[0, 9] == 1.0


def test_v48_boundary_feasibility_respects_future_limit() -> None:
    positives = pd.DataFrame({
        "subject_key": ["one", "two"],
        "session_id": ["session", "session"],
        "matched_event_id": ["event1", "event2"],
        "max_iou": [0.5, 0.5],
        "coarse_start_ms": [0, 0],
        "coarse_end_ms": [120_000, 120_000],
        "truth_start_ms": [30_000, 30_000],
        "truth_end_ms": [150_000, 210_000],
    })
    report = boundary_geometric_feasibility(positives)
    assert report["independent_events"] == 2
    assert report["both_correctable_count"] == 1
    assert report["both_correctable_fraction"] == 0.5


def test_v48_boundary_mae_pairs_the_same_truth_events() -> None:
    coarse = pd.DataFrame({
        "subject_key": ["subject", "subject"],
        "session_id": ["session", "session"],
        "event_id": ["one", "two"],
        "hand_relation": ["same", "different"],
        "start_absolute_error_ms": [10_000, 1_000_000],
        "end_absolute_error_ms": [10_000, 1_000_000],
    })
    refined = pd.DataFrame({
        "subject_key": ["subject"],
        "session_id": ["session"],
        "event_id": ["one"],
        "start_absolute_error_ms": [20_000],
        "end_absolute_error_ms": [20_000],
    })
    paired = _paired_boundary_matches(coarse, refined)
    assert len(paired) == 1
    assert paired["start_absolute_error_ms_coarse"].mean() == 10_000
    assert paired["start_absolute_error_ms_refined"].mean() == 20_000


def test_v48_deep_gate_blocks_regression_against_frozen_baseline() -> None:
    config = load_config("configs/hierarchical_v4_v48_candidate_repair.yaml")
    diagnostics = {
        "point": {"f1": 0.55, "fp_per_hour": 0.061},
        "hand": {"same_sensitivity": 0.65, "different_sensitivity": 0.46},
        "greedy_f1": 0.55,
    }
    failed = _v48_deep_gate(diagnostics, config)
    assert not failed["passed"]
    assert not failed["checks"]["fp_per_hour"]
    diagnostics["point"]["fp_per_hour"] = 0.055
    assert _v48_deep_gate(diagnostics, config)["passed"]


def test_v48_raw_imu_comparison_requires_single_changed_factor(tmp_path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    for root, name, raw_imu, f1 in (
        (reference, "reference", False, 0.55),
        (candidate, "candidate", True, 0.57),
    ):
        root.mkdir()
        (root / "resolved_config.yaml").write_text(yaml.safe_dump({
            "experiment": {"name": name},
            "decoder": {"candidate_protocol": "v4.8"},
            "verifier": {"use_raw_imu_branch": raw_imu},
            "training": {"learning_rate": 0.00015},
        }), encoding="utf-8")
        (root / "deep_crossfit.json").write_text(json.dumps({
            "deep": {"point": {"f1": f1, "fp_per_hour": 0.05}},
        }), encoding="utf-8")
        pd.DataFrame({
            "proposal_id": ["outer0:one"], "subject_key": ["subject"],
            "session_id": ["session"], "coarse_start_ms": [0],
            "coarse_end_ms": [30_000],
        }).to_parquet(root / "deep_crossfit_scores.parquet")
    assert compare(reference, candidate)["passed"]
    config = yaml.safe_load((candidate / "resolved_config.yaml").read_text(encoding="utf-8"))
    config["training"]["learning_rate"] = 0.001
    (candidate / "resolved_config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="more than the verifier branch"):
        compare(reference, candidate)


@pytest.mark.parametrize("config_name", [
    "hierarchical_v4_v48_candidate_repair.yaml",
    "hierarchical_v4_v48_proposal_head.yaml",
    "hierarchical_v4_v48_proposal_head_no_mixstyle.yaml",
    "hierarchical_v4_v48_proposal_head_no_contrastive.yaml",
    "hierarchical_v4_v48_raw_imu.yaml",
    "hierarchical_v4_v48_proposal_head_raw_imu.yaml",
])
def test_v48_configs_are_registered(config_name: str) -> None:
    validate_r3_config(load_config(f"configs/{config_name}"))
