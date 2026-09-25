from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from bme_eating.calibration_v4 import (
    LogisticScoreCombiner,
    ProposalCalibrationV4,
    binary_state_targets,
    state_calibration_metrics,
    subject_crossfit_platt,
)
from bme_eating.config import feature_artifact_name
from bme_eating.data.deep_dataset import (
    Normalization,
    compute_normalization,
    load_normalization,
    save_normalization,
)
from bme_eating.data.stats_fusion_inputs import (
    canonical_input_paths,
    verify_canonical_statsfusion_inputs,
)
from bme_eating.data.stats_fusion_sequence import (
    ClipMixtureSampler,
    SequenceGeometry,
    StatsFusionSequenceDataset,
)
from bme_eating.hierarchical_artifacts import (
    HierarchicalRun,
    assert_disjoint_subjects,
    sha256_file,
    write_json_atomic,
    write_parquet_atomic,
    write_yaml_atomic,
)
from bme_eating.hierarchical_v4_artifacts import current_v4_identity
from bme_eating.hierarchical_v4_gates import verify_gate_evidence
from bme_eating.metrics import (
    evaluate_events,
    evaluation_event_partition_summary,
    partition_evaluation_events,
    prediction_ignore_mask,
)
from bme_eating.models.endpoint_refiner import (
    EndpointRefiner,
    apply_boundary_refinement,
    augment_boundary_training_proposals,
    build_endpoint_features,
    endpoint_loss,
    local_soft_argmax,
    select_boundary_range,
)
from bme_eating.models.event_verifier_v4 import (
    EventVerifierV4,
    HardNegativeBatchSampler,
    ProposalDatasetV4,
    ProposalFeatureBatchV4,
    build_proposal_features_v4,
    classify_proposals,
    verifier_loss_v4,
)
from bme_eating.models.factory import build_state_model
from bme_eating.models.stats_fusion_loss import StatsFusionStateLoss
from bme_eating.proposals import exclude_ignored_candidates, label_event_candidates
from bme_eating.proposals_v4 import (
    generate_event_candidates_v4,
    hysteresis_fragment_diagnostics,
    interval_iou,
)
from bme_eating.stats_features import STATS_FEATURE_COLUMNS, FoldRobustScaler
from bme_eating.structured_decoder import (
    FixedLagSemiMarkovDecoder,
    TruncatedLogNormalDurationPrior,
)

LEGACY_ALIGNMENT_KEYS = ["segment_id", "session_id", "subject_key", "timestamp_ms"]
SESSION_ALIGNMENT_KEYS = ["subject_key", "session_id", "timestamp_ms"]
UNLABELED_ANCHOR_COLUMNS = [
    "segment_id",
    "session_id",
    "segment_path",
    "subject_key",
    "timestamp_ms",
    "motion_history_available_seconds",
    "ppg_history_available_seconds",
]
OUTER_PLACEHOLDER_COLUMNS = {
    "state_target": 0.0,
    "state_loss_mask": 0.0,
    "start_target": 0.0,
    "end_target": 0.0,
    "start_loss_mask": 0.0,
    "end_loss_mask": 0.0,
}


def _dataframe_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, columns].sort_values(columns, kind="stable").reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(selected, index=False).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update("\n".join(columns).encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def _truth_event_durations(events: pd.DataFrame, subjects: set[str]) -> np.ndarray:
    truth, _ = partition_evaluation_events(events, subjects)
    durations = (
        truth["end_ms"].to_numpy(dtype=np.float64)
        - truth["start_ms"].to_numpy(dtype=np.float64)
    ) / 1000.0
    if not len(durations):
        raise ValueError("Duration prior requires at least one evaluable truth event")
    return durations


def _mask_ignored_state_rows(
    frame: pd.DataFrame,
    ignore: pd.DataFrame,
    *,
    step_ms: int,
) -> pd.DataFrame:
    output = frame.copy()
    if "state_loss_mask" not in output:
        output["state_loss_mask"] = 1.0
    for event in ignore.itertuples(index=False):
        selected = output["subject_key"].astype(str).eq(str(event.subject_key))
        timestamps = output["timestamp_ms"].to_numpy(dtype=np.int64)
        selected &= (timestamps > int(event.start_ms)) & (
            timestamps - int(step_ms) < int(event.end_ms)
        )
        output.loc[selected, "state_loss_mask"] = 0.0
    return output


def _evaluable_state_calibration_rows(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {"state_target", "state_logit", "state_probability", "state_loss_mask"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"State calibration frame is missing columns: {sorted(missing)}")
    mask = frame["state_loss_mask"].to_numpy(dtype=float) > 0
    selected = frame.loc[mask].copy()
    if selected.empty:
        raise RuntimeError("State calibration has no evaluable windows")
    return selected, {
        "calibration_total_windows": len(frame),
        "calibration_evaluable_windows": int(mask.sum()),
        "calibration_masked_windows": int((~mask).sum()),
    }


def _assert_proposal_feature_alignment(
    features: ProposalFeatureBatchV4,
    frame: pd.DataFrame,
    *,
    context: str,
) -> None:
    frame_ids = frame["proposal_id"].astype(str).to_numpy()
    feature_ids = np.asarray(features.proposal_ids, dtype=str)
    if len(np.unique(frame_ids)) != len(frame_ids):
        raise RuntimeError(f"{context} proposal IDs are not globally unique")
    if not np.array_equal(feature_ids, frame_ids):
        raise RuntimeError(f"{context} proposal feature rows are not aligned with score rows")
    if features.event_target is not None and "is_positive" in frame:
        targets = frame["is_positive"].to_numpy(dtype=np.float32)
        if not np.array_equal(features.event_target.astype(np.float32), targets):
            raise RuntimeError(f"{context} proposal labels are not aligned with feature targets")


@dataclass(frozen=True)
class V4Inputs:
    anchors: pd.DataFrame
    segments: pd.DataFrame
    events: pd.DataFrame
    statistics: pd.DataFrame
    subject_folds: dict[str, int]


def load_v4_inputs(
    config: dict[str, Any],
    input_root: Path,
    *,
    fold: int,
    event_role: str,
    allow_outer_labels: bool = False,
) -> V4Inputs:
    if event_role not in {"outer_train", "outer_test", "all"}:
        raise ValueError(f"Unknown v4 event role: {event_role}")
    if event_role in {"outer_test", "all"} and not allow_outer_labels:
        raise RuntimeError(
            "Outer-test labels require the sealed evaluation or post-gate final entrypoint"
        )
    subject_folds = {
        str(key): int(value)
        for key, value in json.loads(
            (input_root / "indices" / "subject_folds.json").read_text(encoding="utf-8")
        ).items()
    }
    formal_r2 = config.get("experiment", {}).get("protocol_version") == "statsfusion-r2"
    if formal_r2:
        output_root = input_root.parent / str(config["project"]["artifact_schema_version"])
        verify_canonical_statsfusion_inputs(input_root, output_root)
        canonical_paths = canonical_input_paths(output_root)
        anchor_path = canonical_paths["anchors"]
    else:
        anchor_path = input_root / "indices" / "anchors.parquet"
    outer_train_subjects = {
        subject for subject, subject_fold in subject_folds.items() if subject_fold != fold
    }
    outer_test_subjects = set(subject_folds) - outer_train_subjects
    if event_role == "all":
        anchors = pd.read_parquet(anchor_path)
    elif event_role == "outer_train":
        labeled = pd.read_parquet(
            anchor_path,
            filters=[("subject_key", "in", sorted(outer_train_subjects))],
        )
        unlabeled = pd.read_parquet(
            anchor_path,
            columns=UNLABELED_ANCHOR_COLUMNS,
            filters=[("subject_key", "in", sorted(outer_test_subjects))],
        )
        for column, value in OUTER_PLACEHOLDER_COLUMNS.items():
            unlabeled[column] = value
        anchors = pd.concat((labeled, unlabeled), ignore_index=True, sort=False)
    else:
        anchors = pd.read_parquet(
            anchor_path,
            filters=[("subject_key", "in", sorted(outer_test_subjects))],
        )
    segments = pd.read_parquet(input_root / "indices" / "segments.parquet")
    if event_role == "all":
        events = pd.read_parquet(input_root / "indices" / "events.parquet")
    else:
        selected = sorted(
            outer_train_subjects if event_role == "outer_train" else outer_test_subjects
        )
        events = pd.read_parquet(
            input_root / "indices" / "events.parquet",
            filters=[("subject_key", "in", selected)],
        )
    feature_path = (
        canonical_paths["statistics"]
        if formal_r2
        else input_root / "features" / f"{feature_artifact_name(config)}.parquet"
    )
    alignment_keys = SESSION_ALIGNMENT_KEYS if formal_r2 else LEGACY_ALIGNMENT_KEYS
    statistics = pd.read_parquet(
        feature_path,
        columns=[*alignment_keys, *STATS_FEATURE_COLUMNS],
    )
    if statistics.duplicated(alignment_keys).any():
        raise RuntimeError("V4 statistics contain duplicate timeline keys")
    return V4Inputs(anchors, segments, events, statistics, subject_folds)


def outer_subject_sets(inputs: V4Inputs, fold: int) -> tuple[set[str], set[str]]:
    subjects = set(inputs.anchors["subject_key"].astype(str).unique())
    outer_test = {subject for subject in subjects if inputs.subject_folds[subject] == fold}
    outer_train = subjects - outer_test
    assert_disjoint_subjects(outer_train=outer_train, outer_test=outer_test)
    if not outer_train or not outer_test:
        raise ValueError("Outer fold must have non-empty train and test subjects")
    return outer_train, outer_test


def stacking_partitions(subjects: set[str], partitions: int, seed: int) -> dict[str, int]:
    if partitions < 2 or len(subjects) < partitions:
        raise ValueError("Insufficient subjects for v4 stacking partitions")
    rng = np.random.default_rng(seed)
    ordered = np.asarray(sorted(subjects), dtype=object)
    rng.shuffle(ordered)
    return {str(subject): index % partitions for index, subject in enumerate(ordered)}


def _selector_subject_summary(
    subjects: set[str],
    events: pd.DataFrame,
    anchors: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    ordered = sorted(str(subject) for subject in subjects)
    summary = pd.DataFrame({"subject_key": ordered})
    candidate_events = events[events["subject_key"].astype(str).isin(subjects)].copy()
    if "valid_duration" in candidate_events:
        candidate_events = candidate_events[candidate_events["valid_duration"].fillna(False)]
    if "evaluable" in candidate_events:
        candidate_events = candidate_events[candidate_events["evaluable"].fillna(False)]
    elif "coverage" in candidate_events:
        candidate_events = candidate_events[candidate_events["coverage"].fillna("").eq("full")]
    candidate_events = candidate_events[
        candidate_events["end_ms"].to_numpy(dtype=np.int64)
        > candidate_events["start_ms"].to_numpy(dtype=np.int64)
    ].copy()
    candidate_events["duration_seconds"] = (
        candidate_events["end_ms"].to_numpy(dtype=np.float64)
        - candidate_events["start_ms"].to_numpy(dtype=np.float64)
    ) / 1000.0
    if len(candidate_events):
        short_threshold, long_threshold = np.quantile(
            candidate_events["duration_seconds"].to_numpy(dtype=np.float64),
            [1.0 / 3.0, 2.0 / 3.0],
        )
    else:
        short_threshold = long_threshold = 0.0
    relation = candidate_events.get(
        "hand_relation", pd.Series("unknown", index=candidate_events.index, dtype=object)
    ).fillna("unknown").astype(str)
    candidate_events = candidate_events.assign(
        same_event=(relation == "same").astype(np.int64),
        different_event=(relation == "different").astype(np.int64),
        short_event=(candidate_events["duration_seconds"] <= short_threshold).astype(np.int64),
        long_event=(candidate_events["duration_seconds"] >= long_threshold).astype(np.int64),
    )
    event_summary = (
        candidate_events.groupby("subject_key", as_index=False)
        .agg(
            event_count=("duration_seconds", "size"),
            same_event_count=("same_event", "sum"),
            different_event_count=("different_event", "sum"),
            short_event_count=("short_event", "sum"),
            long_event_count=("long_event", "sum"),
            event_duration_seconds=("duration_seconds", "sum"),
        )
        .assign(subject_key=lambda frame: frame["subject_key"].astype(str))
    )
    summary = summary.merge(event_summary, on="subject_key", how="left")
    anchor_keys = [
        column
        for column in ("subject_key", "session_id", "timestamp_ms")
        if column in anchors
    ]
    if {"subject_key", "timestamp_ms"}.issubset(anchor_keys):
        observation = (
            anchors[anchors["subject_key"].astype(str).isin(subjects)]
            .assign(subject_key=lambda frame: frame["subject_key"].astype(str))
            .drop_duplicates(anchor_keys)
            .groupby("subject_key", as_index=False)
            .agg(observation_anchor_count=("timestamp_ms", "size"))
        )
        summary = summary.merge(observation, on="subject_key", how="left")
    else:
        summary["observation_anchor_count"] = 0
    value_columns = [
        "event_count",
        "same_event_count",
        "different_event_count",
        "short_event_count",
        "long_event_count",
        "event_duration_seconds",
        "observation_anchor_count",
    ]
    summary[value_columns] = summary[value_columns].fillna(0.0).astype(np.float64)
    return summary, {
        "short_event_threshold_seconds": float(short_threshold),
        "long_event_threshold_seconds": float(long_threshold),
    }


def _selector_split(
    subjects: set[str],
    fraction: float,
    seed: int,
    *,
    events: pd.DataFrame,
    anchors: pd.DataFrame,
) -> tuple[set[str], set[str], dict[str, Any]]:
    if not 0 < fraction < 0.5:
        raise ValueError("Selector fraction must be in (0, 0.5)")
    if len(subjects) < 2:
        raise ValueError("Selector split requires at least two subjects")
    summary, duration_thresholds = _selector_subject_summary(subjects, events, anchors)
    rng = np.random.default_rng(seed)
    ordered = summary["subject_key"].to_numpy(dtype=object)
    rng.shuffle(ordered)
    summary = summary.set_index("subject_key").loc[ordered].reset_index()
    count = min(len(ordered) - 1, max(2, round(len(ordered) * fraction)))
    value_columns = [column for column in summary.columns if column != "subject_key"]
    values = summary[value_columns].to_numpy(dtype=np.float64)
    total = values.sum(axis=0)
    target_ratio = count / len(ordered)
    target = total * target_ratio
    scale = np.maximum(target, 1.0)
    coverage_columns = {
        value_columns.index(column)
        for column in (
            "event_count",
            "same_event_count",
            "different_event_count",
            "short_event_count",
            "long_event_count",
        )
        if column in value_columns
    }

    def score(indices: tuple[int, ...]) -> float:
        selected = values[np.asarray(indices, dtype=np.int64)].sum(axis=0)
        active = total > 0
        error = float(np.square((selected[active] - target[active]) / scale[active]).mean())
        missing_coverage = sum(
            1 for index in coverage_columns if total[index] > 0 and selected[index] == 0
        )
        return error + 4.0 * missing_coverage

    combination_count = math.comb(len(ordered), count)
    if combination_count <= 250_000:
        candidates = combinations(range(len(ordered)), count)
    else:
        sampled = {
            tuple(sorted(rng.choice(len(ordered), size=count, replace=False).tolist()))
            for _ in range(4096)
        }
        candidates = iter(sorted(sampled))
    best_indices: tuple[int, ...] | None = None
    best_score = math.inf
    for indices in candidates:
        current = score(indices)
        if current < best_score - 1e-12:
            best_indices = indices
            best_score = current
    if best_indices is None:
        raise RuntimeError("Selector balancing did not produce a subject subset")
    selected_indices = set(best_indices)
    selector = {str(ordered[index]) for index in selected_indices}
    fit = {str(value) for index, value in enumerate(ordered) if index not in selected_indices}
    assert_disjoint_subjects(fit=fit, selector=selector)
    selector_totals = values[np.asarray(best_indices, dtype=np.int64)].sum(axis=0)
    report = {
        "strategy": "event_stratified_subject_subset_v1",
        "seed": int(seed),
        "requested_fraction": float(fraction),
        "actual_fraction": float(len(selector) / len(subjects)),
        "fit_subject_count": len(fit),
        "selector_subject_count": len(selector),
        "balance_score": float(best_score),
        "balance_columns": value_columns,
        "overall_totals": {
            column: float(value) for column, value in zip(value_columns, total, strict=True)
        },
        "target_totals": {
            column: float(value) for column, value in zip(value_columns, target, strict=True)
        },
        "selector_totals": {
            column: float(value)
            for column, value in zip(value_columns, selector_totals, strict=True)
        },
        **duration_thresholds,
    }
    return fit, selector, report


def _geometry(config: dict[str, Any]) -> SequenceGeometry:
    values = config["sequence"]
    return SequenceGeometry(
        supervised_steps=int(values["supervised_steps"]),
        short_receptive_field_steps=int(values["short_receptive_field_steps"]),
        long_receptive_field_tokens=int(values["long_receptive_field_tokens"]),
        long_pool_factor=int(values["long_pool_factor"]),
        step_seconds=int(values["step_seconds"]),
    )


def _fit_scaler_and_transform(
    inputs: V4Inputs, subjects: set[str]
) -> tuple[FoldRobustScaler, pd.DataFrame]:
    scaler = FoldRobustScaler.fit(
        inputs.statistics,
        training_subjects=subjects,
    )
    return scaler, _transform_with_scaler(inputs, scaler)


def _transform_with_scaler(inputs: V4Inputs, scaler: FoldRobustScaler) -> pd.DataFrame:
    transformed = scaler.transform_frame(inputs.statistics)
    alignment_keys = (
        LEGACY_ALIGNMENT_KEYS
        if "segment_id" in transformed.columns
        else SESSION_ALIGNMENT_KEYS
    )
    statistics_columns = [
        *(f"stat_{name}" for name in STATS_FEATURE_COLUMNS),
        *(f"stat_{name}_missing" for name in STATS_FEATURE_COLUMNS),
    ]
    anchors = inputs.anchors.merge(
        transformed[[*alignment_keys, *statistics_columns]],
        on=alignment_keys,
        how="left",
        validate="one_to_one",
    )
    if anchors[statistics_columns].isna().any().any():
        raise RuntimeError("V4 statistics failed to align with anchors")
    return anchors


def _statistics_columns() -> list[str]:
    return [
        *(f"stat_{name}" for name in STATS_FEATURE_COLUMNS),
        *(f"stat_{name}_missing" for name in STATS_FEATURE_COLUMNS),
    ]


def _device(config: dict[str, Any]) -> torch.device:
    requested = str(config["training"].get("device", "cpu"))
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V4 configuration requires CUDA but CUDA is unavailable")
    return torch.device(requested)


def _make_dataset(
    anchors: pd.DataFrame,
    inputs: V4Inputs,
    events: pd.DataFrame,
    normalization: Normalization,
    config: dict[str, Any],
    *,
    training: bool,
    seed: int,
) -> StatsFusionSequenceDataset:
    return StatsFusionSequenceDataset(
        anchors,
        inputs.segments,
        events,
        normalization,
        statistics_columns=_statistics_columns(),
        geometry=_geometry(config),
        training=training,
        ppg_modality_dropout=(
            float(config["training"]["ppg_modality_dropout"]) if training else 0.0
        ),
        seed=seed,
    )


def _state_loss(config: dict[str, Any]) -> StatsFusionStateLoss:
    return StatsFusionStateLoss(
        smooth_weight=float(config["loss"]["smooth_weight"]),
        smooth_tau=float(config["loss"]["smooth_tau"]),
        boundary_weight=float(config["loss"]["boundary_weight"]),
    )


def _train_state_epochs(
    model: torch.nn.Module,
    dataset: StatsFusionSequenceDataset,
    config: dict[str, Any],
    *,
    epochs: int,
    seed: int,
    optimizer: torch.optim.Optimizer | None = None,
    epoch_offset: int = 0,
    progress_label: str = "state",
    total_epochs: int | None = None,
    scheduler_total_epochs: int | None = None,
) -> torch.optim.Optimizer:
    device = _device(config)
    model.to(device)
    model.train()
    batch_size = int(config["training"]["batch_size"])
    sampler = ClipMixtureSampler(
        dataset.anchors,
        samples_per_epoch=int(config["training"]["steps_per_epoch"]) * batch_size,
        mixture=config["sequence"]["sampling_mixture"],
        seed=seed,
        supervised_steps=dataset.geometry.supervised_steps,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    criterion = _state_loss(config)
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
    accumulation = int(config["training"]["gradient_accumulation"])
    display_total = total_epochs or epoch_offset + int(epochs)
    schedule_total = scheduler_total_epochs or display_total
    scheduler = getattr(optimizer, "_bme_scheduler", None)
    if scheduler is None:
        updates_per_epoch = math.ceil(len(loader) / accumulation)
        total_updates = max(1, updates_per_epoch * int(schedule_total))
        warmup_updates = max(1, round(total_updates * float(config["training"]["warmup_fraction"])))

        def learning_rate_multiplier(step: int) -> float:
            if step < warmup_updates:
                return float(step + 1) / warmup_updates
            progress = (step - warmup_updates + 1) / max(total_updates - warmup_updates, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_multiplier)
        optimizer._bme_scheduler = scheduler
    amp_enabled = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if config["training"].get("amp_dtype") == "bfloat16" else torch.float16
    )
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epoch_offset, epoch_offset + int(epochs)):
        sampler.set_epoch(epoch)
        progress = tqdm(
            loader,
            desc=f"{progress_label} epoch {epoch + 1}/{display_total}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for step, batch in enumerate(progress, start=1):
            tensors = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(tensors)
                loss, _ = criterion(output, tensors)
                group_start = ((step - 1) // accumulation) * accumulation + 1
                group_size = min(accumulation, len(loader) - group_start + 1)
                loss = loss / group_size
            loss.backward()
            if step % accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["training"]["gradient_clip_norm"])
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 10 == 0 or step == len(loader):
                progress.set_postfix(loss=f"{float(loss.detach().cpu()) * accumulation:.4f}")
    return optimizer


def _selector_score(
    model: torch.nn.Module,
    dataset: StatsFusionSequenceDataset,
    config: dict[str, Any],
    fit_events: pd.DataFrame,
    selector_events: pd.DataFrame,
    progress_label: str = "state selector",
) -> dict[str, float | bool]:
    predictions = infer_state_windows(
        model,
        dataset,
        config,
        stacking_partition=-1,
        progress_label=progress_label,
    )
    subjects = sorted(predictions["subject_key"].astype(str).unique())
    if len(subjects) < 2:
        raise RuntimeError("State selector requires at least two disjoint subjects")
    subject_partition = {subject: index for index, subject in enumerate(subjects)}
    predictions["stacking_partition"] = (
        predictions["subject_key"].astype(str).map(subject_partition)
    )
    calibrated, _ = subject_crossfit_platt(
        predictions,
        fit_mask_column="state_loss_mask",
    )
    calibrated["onset_probability"] = 1.0 / (1.0 + np.exp(-calibrated["onset_logit"]))
    calibrated["offset_probability"] = 1.0 / (1.0 + np.exp(-calibrated["offset_logit"]))
    calibrated["state_probability_derivative"] = (
        calibrated.groupby(["subject_key", "session_id"], sort=False)["state_probability"]
        .diff()
        .fillna(0.0)
    )
    calibration_rows = calibrated[calibrated["state_loss_mask"].to_numpy(dtype=float) > 0]
    calibration = state_calibration_metrics(
        calibration_rows["state_target"].to_numpy(),
        calibration_rows["state_logit"].to_numpy(),
        calibration_rows["state_probability"].to_numpy(),
        low_threshold=float(config["decoder"]["low_threshold"]),
        bins=int(config["calibration"]["ece_bins"]),
    )
    calibration_passed = bool(
        calibration["ece"] <= float(config["promotion_gate"]["maximum_state_ece"])
        and calibration["brier"] < calibration["uncalibrated_brier"]
        and calibration["mean_probability_to_prevalence"]
        <= float(config["promotion_gate"]["maximum_state_prevalence_ratio"])
    )
    fit_subjects = set(fit_events["subject_key"].astype(str))
    durations = _truth_event_durations(fit_events, fit_subjects)
    decoder_config = config["decoder"]
    prior = TruncatedLogNormalDurationPrior.fit(
        durations,
        lower_quantile=float(decoder_config["duration_lower_quantile"]),
        upper_quantile=float(decoder_config["duration_upper_quantile"]),
        minimum_floor_seconds=float(decoder_config["minimum_duration_floor_seconds"]),
        maximum_ceiling_seconds=float(decoder_config["maximum_duration_ceiling_seconds"]),
    )
    decoder = FixedLagSemiMarkovDecoder(
        prior,
        grid_seconds=int(decoder_config["grid_seconds"]),
        fixed_lag_seconds=int(decoder_config["fixed_lag_seconds"]),
    )
    proposals = generate_event_candidates_v4(
        calibrated, decoder, decoder_config, split_role="state_selector"
    )
    truth, ignore = partition_evaluation_events(selector_events, set(subjects))
    labeled = exclude_ignored_candidates(label_event_candidates(proposals, truth, 0.25), ignore)
    recall = _candidate_recall(proposals, truth)["candidate_recall"]
    if len(labeled):
        point = _best_verifier_operating_point(
            labeled, truth, calibrated, "generator_score", config, ignore
        )
        f1 = float(point["f1"])
    else:
        f1 = 0.0
    fragments = hysteresis_fragment_diagnostics(calibrated, decoder_config)["state_fragment_count"]
    binary_target = binary_state_targets(calibration_rows["state_target"].to_numpy())
    raw_probability = 1.0 / (1.0 + np.exp(-calibration_rows["state_logit"].to_numpy()))
    auprc = (
        float(average_precision_score(binary_target, raw_probability))
        if np.any(binary_target > 0)
        else 0.0
    )
    return {
        "candidate_recall": float(recall),
        "calibration_passed": calibration_passed,
        "event_f1": f1,
        "state_fragment_count": float(fragments),
        "ece": float(calibration["ece"]),
        "window_auprc": auprc,
    }


def _build_seeded_state_model(config: dict[str, Any], seed: int) -> torch.nn.Module:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return build_state_model(config["model"])


def _add_robust_epoch_metrics(
    epoch_metrics: list[dict[str, Any]], rolling_epochs: int
) -> None:
    if rolling_epochs <= 0:
        raise ValueError("Selector rolling epoch count must be positive")
    metric_names = (
        "candidate_recall",
        "event_f1",
        "state_fragment_count",
        "ece",
        "window_auprc",
    )
    for index, metrics in enumerate(epoch_metrics):
        values = epoch_metrics[max(0, index - rolling_epochs + 1) : index + 1]
        for name in metric_names:
            metrics[f"robust_{name}"] = float(np.median([float(value[name]) for value in values]))
        passed = sum(bool(value["calibration_passed"]) for value in values)
        metrics["robust_calibration_passed"] = passed >= math.ceil(len(values) / 2)


def _selector_early_stopping_improved(
    current: dict[str, Any],
    best: dict[str, Any] | None,
    *,
    minimum_recall: float,
    minimum_delta: float,
) -> bool:
    if best is None:
        return True
    current_qualified = bool(
        current["robust_candidate_recall"] >= minimum_recall
        and current["robust_calibration_passed"]
    )
    best_qualified = bool(
        best["robust_candidate_recall"] >= minimum_recall
        and best["robust_calibration_passed"]
    )
    if current_qualified != best_qualified:
        return current_qualified
    primary_name = "robust_event_f1" if current_qualified else "robust_candidate_recall"
    secondary_name = "robust_candidate_recall" if current_qualified else "robust_event_f1"
    current_primary = float(current[primary_name])
    best_primary = float(best[primary_name])
    if current_primary > best_primary + minimum_delta:
        return True
    return bool(
        current_primary >= best_primary - minimum_delta
        and float(current[secondary_name]) > float(best[secondary_name]) + minimum_delta
    )


def _select_epoch(
    fit_anchors: pd.DataFrame,
    selector_anchors: pd.DataFrame,
    fit_subjects: set[str],
    inputs: V4Inputs,
    config: dict[str, Any],
    seed: int,
) -> tuple[int, dict[str, Any]]:
    scaler, transformed = _fit_scaler_and_transform(inputs, fit_subjects)
    del scaler
    fit_rows = transformed[transformed["subject_key"].astype(str).isin(fit_subjects)].reset_index(
        drop=True
    )
    selector_subjects = set(selector_anchors["subject_key"].astype(str))
    selector_rows = transformed[
        transformed["subject_key"].astype(str).isin(selector_subjects)
    ].reset_index(drop=True)
    normalization = compute_normalization(inputs.segments, fit_subjects)
    fit_events = inputs.events[inputs.events["subject_key"].astype(str).isin(fit_subjects)]
    selector_events = inputs.events[
        inputs.events["subject_key"].astype(str).isin(selector_subjects)
    ]
    train_dataset = _make_dataset(
        fit_rows, inputs, fit_events, normalization, config, training=True, seed=seed
    )
    selector_dataset = _make_dataset(
        selector_rows,
        inputs,
        selector_events,
        normalization,
        config,
        training=False,
        seed=seed,
    )
    model = _build_seeded_state_model(config, seed)
    epoch_metrics: list[dict[str, Any]] = []
    optimizer = None
    maximum_epochs = int(config["training"]["max_epochs"])
    rolling_epochs = int(config["training"].get("selector_rolling_epochs", 1))
    validation_interval = int(config["training"].get("validation_every_epochs", 1))
    minimum_training_epochs = int(
        config["training"].get("early_stopping_min_epochs", maximum_epochs)
    )
    patience_checks = int(
        config["training"].get("early_stopping_patience_checks", maximum_epochs)
    )
    minimum_delta = float(config["training"].get("early_stopping_min_delta", 0.0))
    minimum_recall = float(config["promotion_gate"]["minimum_candidate_recall"])
    if validation_interval <= 0 or minimum_training_epochs <= 0 or patience_checks <= 0:
        raise ValueError("State selector early-stopping intervals must be positive")
    best_progress: dict[str, Any] | None = None
    checks_without_improvement = 0
    stopped_early = False
    completed_training_epochs = 0
    for epoch in range(1, maximum_epochs + 1):
        optimizer = _train_state_epochs(
            model,
            train_dataset,
            config,
            epochs=1,
            seed=seed,
            optimizer=optimizer,
            epoch_offset=epoch - 1,
            progress_label=f"state select seed={seed}",
            total_epochs=maximum_epochs,
        )
        completed_training_epochs = epoch
        if epoch % validation_interval != 0 and epoch != maximum_epochs:
            continue
        score = _selector_score(
            model,
            selector_dataset,
            config,
            fit_events,
            selector_events,
            progress_label=f"state validate {epoch}/{maximum_epochs}",
        )
        epoch_metrics.append({"epoch": epoch, **score})
        _add_robust_epoch_metrics(epoch_metrics, rolling_epochs)
        current = epoch_metrics[-1]
        if _selector_early_stopping_improved(
            current,
            best_progress,
            minimum_recall=minimum_recall,
            minimum_delta=minimum_delta,
        ):
            best_progress = dict(current)
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
        tqdm.write(
            f"[state selector seed={seed}] epoch {epoch}/{maximum_epochs} "
            f"recall={score['candidate_recall']:.4f} F1={score['event_f1']:.4f} "
            f"ECE={score['ece']:.4f}"
        )
        if epoch >= minimum_training_epochs and checks_without_improvement >= patience_checks:
            stopped_early = True
            tqdm.write(
                f"[state selector seed={seed}] early stop after epoch {epoch}; "
                f"no robust improvement for {checks_without_improvement} checks"
            )
            break
    qualified = [
        value
        for value in epoch_metrics
        if value["robust_candidate_recall"] >= minimum_recall
        and bool(value["robust_calibration_passed"])
    ]
    if qualified:
        best = max(
            qualified,
            key=lambda value: (
                value["robust_event_f1"],
                value["robust_candidate_recall"],
                -value["robust_state_fragment_count"],
                -value["robust_ece"],
                value["robust_window_auprc"],
                value["event_f1"],
            ),
        )
        promotion_eligible = True
    else:
        best = max(
            epoch_metrics,
            key=lambda value: (
                value["robust_candidate_recall"],
                value["robust_event_f1"],
                -value["robust_state_fragment_count"],
                -value["robust_ece"],
                value["robust_window_auprc"],
                value["candidate_recall"],
            ),
        )
        promotion_eligible = False
    return int(best["epoch"]), {
        "selected_epoch": int(best["epoch"]),
        "promotion_eligible": promotion_eligible,
        "minimum_candidate_recall": minimum_recall,
        "selector_calibration_protocol": "leave_one_subject_out",
        "selector_rolling_epochs": rolling_epochs,
        "validation_every_epochs": validation_interval,
        "early_stopping_min_epochs": minimum_training_epochs,
        "early_stopping_patience_checks": patience_checks,
        "early_stopping_min_delta": minimum_delta,
        "stopped_early": stopped_early,
        "completed_training_epochs": completed_training_epochs,
        "validation_checks": len(epoch_metrics),
        "selected_metrics": dict(best),
        "epochs": epoch_metrics,
    }


def _inference_endpoints(dataset: StatsFusionSequenceDataset) -> list[int]:
    endpoints: list[int] = []
    step = dataset.geometry.supervised_steps
    for group in dataset.session_groups.values():
        rows = group["_row_id"].to_numpy(dtype=np.int64)
        local = list(range(min(step - 1, len(rows) - 1), len(rows), step))
        if not local or local[-1] != len(rows) - 1:
            local.append(len(rows) - 1)
        endpoints.extend(int(rows[index]) for index in local)
    return endpoints


@torch.no_grad()
def infer_state_windows(
    model: torch.nn.Module,
    dataset: StatsFusionSequenceDataset,
    config: dict[str, Any],
    *,
    stacking_partition: int,
    progress_label: str = "state inference",
) -> pd.DataFrame:
    device = _device(config)
    model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["inference_batch_size"]),
        sampler=_inference_endpoints(dataset),
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    frames: list[pd.DataFrame] = []
    statistic_names = [f"stat_{name}" for name in STATS_FEATURE_COLUMNS]
    for batch in tqdm(
        loader,
        desc=progress_label,
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    ):
        tensors = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        output = model(tensors)
        for sample in range(output["state_logit"].shape[0]):
            mask = tensors["supervision_mask"][sample].bool().cpu().numpy()
            timestamps = tensors["timestamp_ms"][sample].cpu().numpy()[mask]
            frame = pd.DataFrame(
                {
                    "subject_key": str(batch["subject_key"][sample]),
                    "session_id": str(batch["session_id"][sample]),
                    "timestamp_ms": timestamps,
                    "state_logit": output["state_logit"][sample].float().cpu().numpy()[mask],
                    "onset_logit": output["onset_logit"][sample].float().cpu().numpy()[mask],
                    "offset_logit": output["offset_logit"][sample].float().cpu().numpy()[mask],
                    "ppg_gate": output["ppg_gate"][sample].float().cpu().numpy()[mask],
                    "statistics_gate": output["statistics_gate"][sample]
                    .float()
                    .cpu()
                    .numpy()[mask],
                    "long_gate": output["long_gate"][sample].float().cpu().numpy()[mask],
                    "missing_fraction": output["missing_fraction"][sample]
                    .float()
                    .cpu()
                    .numpy()[mask],
                    "motion_valid_fraction": output["motion_valid_fraction"][sample]
                    .float()
                    .cpu()
                    .numpy()[mask],
                    "ppg_valid_fraction": output["ppg_valid_fraction"][sample]
                    .float()
                    .cpu()
                    .numpy()[mask],
                    "statistics_missing_fraction": output["statistics_missing_fraction"][sample]
                    .float()
                    .cpu()
                    .numpy()[mask],
                    "state_target": tensors["state_target"][sample].cpu().numpy()[mask],
                    "state_loss_mask": tensors["state_loss_mask"][sample].cpu().numpy()[mask],
                    "stacking_partition": int(stacking_partition),
                }
            )
            statistics = tensors["statistics"][sample].float().cpu().numpy()[mask]
            for index, name in enumerate(statistic_names):
                frame[name] = statistics[:, index]
            frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    return output.drop_duplicates(["subject_key", "session_id", "timestamp_ms"], keep="last")


def _gate_diagnostics(windows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    gate_columns = [
        "ppg_gate",
        "statistics_gate",
        "long_gate",
        "motion_valid_fraction",
        "ppg_valid_fraction",
        "statistics_missing_fraction",
        "missing_fraction",
    ]
    rows: list[dict[str, Any]] = []
    for subject, group in windows.groupby("subject_key", sort=True):
        row: dict[str, Any] = {"subject_key": str(subject)}
        for column in gate_columns:
            values = group[column].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            if not len(finite):
                for statistic in ("mean", "std", "p05", "p50", "p95"):
                    row[f"{column}_{statistic}"] = float("nan")
                continue
            row[f"{column}_mean"] = float(finite.mean())
            row[f"{column}_std"] = float(finite.std())
            for quantile, name in ((0.05, "p05"), (0.50, "p50"), (0.95, "p95")):
                row[f"{column}_{name}"] = float(np.quantile(finite, quantile))
        rows.append(row)
    diagnostics = pd.DataFrame(rows)
    statistics_mean = diagnostics.get("statistics_gate_mean", pd.Series(dtype=np.float64)).to_numpy(
        dtype=np.float64
    )
    finite = statistics_mean[np.isfinite(statistics_mean)]
    summary = {
        "subject_count": len(diagnostics),
        "statistics_gate_collapsed_low_subjects": int(np.count_nonzero(finite <= 0.01)),
        "statistics_gate_collapsed_high_subjects": int(np.count_nonzero(finite >= 0.99)),
        "collapse_thresholds": {"low": 0.01, "high": 0.99},
    }
    return diagnostics, summary


def _save_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _average_state_prediction_frames(
    frames: list[pd.DataFrame], *, primary_seed_index: int = 0
) -> pd.DataFrame:
    if not frames:
        raise ValueError("State ensemble requires at least one prediction frame")
    keys = ["subject_key", "session_id", "timestamp_ms"]
    ordered = [frame.sort_values(keys).reset_index(drop=True) for frame in frames]
    reference = ordered[primary_seed_index].copy()
    for frame in ordered:
        if not reference[keys].equals(frame[keys]):
            raise RuntimeError("State seed predictions are not timeline-aligned")
    for column in ("state_logit", "onset_logit", "offset_logit"):
        reference[column] = np.mean(
            [frame[column].to_numpy(dtype=np.float64) for frame in ordered], axis=0
        )
    reference["state_seed_count"] = len(ordered)
    return reference


def train_state_crossfit_v4(
    run: HierarchicalRun,
    config: dict[str, Any],
    inputs: V4Inputs,
    *,
    resume: bool,
) -> None:
    run.require_stage("CREATED")
    outer_train, outer_test = outer_subject_sets(inputs, int(run.payload["outer_fold"]))
    partitions = stacking_partitions(
        outer_train,
        int(config["hierarchical"]["verifier_crossfit_partitions"]),
        int(config["training"]["random_seed"]),
    )
    state_seeds = [int(value) for value in config["final_training"]["state_seeds"]]
    if state_seeds != [2026, 2027, 2028]:
        raise ValueError("statsfusion-r2 requires state seeds 2026/2027/2028")
    all_predictions: list[pd.DataFrame] = []
    artifacts: list[Path] = []
    selected_epochs: dict[int, list[int]] = {seed: [] for seed in state_seeds}
    for partition in sorted(set(partitions.values())):
        tqdm.write(f"[state] starting stacking partition {partition}")
        holdout = {subject for subject, value in partitions.items() if value == partition}
        training_subjects = outer_train - holdout
        fit_subjects, selector_subjects, selector_split_report = _selector_split(
            training_subjects,
            float(config["training"]["selector_fraction"]),
            int(config["training"]["random_seed"]) + partition,
            events=inputs.events,
            anchors=inputs.anchors,
        )
        assert_disjoint_subjects(
            fit=fit_subjects, selector=selector_subjects, holdout=holdout, outer=outer_test
        )
        partition_root = run.root / "crossfit" / f"partition_{partition}" / "state"
        scaler_path = partition_root / "statistics_scaler.json"
        normalization_path = partition_root / "sensor_normalization.json"
        if resume and scaler_path.is_file() and normalization_path.is_file():
            scaler = FoldRobustScaler.from_json(json.loads(scaler_path.read_text(encoding="utf-8")))
            normalization = load_normalization(normalization_path)
            transformed = _transform_with_scaler(inputs, scaler)
        else:
            scaler, transformed = _fit_scaler_and_transform(inputs, training_subjects)
            normalization = compute_normalization(inputs.segments, training_subjects)
            write_json_atomic(scaler_path, scaler.to_json())
            save_normalization(normalization, normalization_path)
        artifacts.extend((scaler_path, normalization_path))
        holdout_rows = transformed[
            transformed["subject_key"].astype(str).isin(holdout)
        ].reset_index(drop=True)
        holdout_events = inputs.events[inputs.events["subject_key"].astype(str).isin(holdout)]
        holdout_dataset = _make_dataset(
            holdout_rows,
            inputs,
            holdout_events,
            normalization,
            config,
            training=False,
            seed=int(config["training"]["random_seed"]),
        )
        seed_frames: list[pd.DataFrame] = []
        for seed in state_seeds:
            checkpoint_path = partition_root / f"seed_{seed}.pt"
            selector_report_path = partition_root / f"selector_seed_{seed}.json"
            if resume and checkpoint_path.is_file() and selector_report_path.is_file():
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                epochs = int(checkpoint["epochs"])
                model = build_state_model(checkpoint["model_config"])
                model.load_state_dict(checkpoint["model"])
            else:
                epochs, selector_report = _select_epoch(
                    inputs.anchors[inputs.anchors["subject_key"].astype(str).isin(fit_subjects)],
                    inputs.anchors[
                        inputs.anchors["subject_key"].astype(str).isin(selector_subjects)
                    ],
                    fit_subjects,
                    inputs,
                    config,
                    seed + partition * 10_000,
                )
                selector_report = {
                    **selector_report,
                    "fit_subjects": sorted(fit_subjects),
                    "selector_subjects": sorted(selector_subjects),
                    "selector_split": selector_split_report,
                }
                write_json_atomic(selector_report_path, selector_report)
                train_rows = transformed[
                    transformed["subject_key"].astype(str).isin(training_subjects)
                ].reset_index(drop=True)
                train_events = inputs.events[
                    inputs.events["subject_key"].astype(str).isin(training_subjects)
                ]
                dataset = _make_dataset(
                    train_rows,
                    inputs,
                    train_events,
                    normalization,
                    config,
                    training=True,
                    seed=seed,
                )
                torch.manual_seed(seed)
                model = build_state_model(config["model"])
                _train_state_epochs(
                    model,
                    dataset,
                    config,
                    epochs=epochs,
                    seed=seed + partition * 10_000,
                    progress_label=f"state partition={partition} seed={seed} retrain",
                    total_epochs=epochs,
                    scheduler_total_epochs=int(config["training"]["max_epochs"]),
                )
                _save_torch_atomic(
                    checkpoint_path,
                    {
                        "model": model.state_dict(),
                        "model_config": config["model"],
                        "epochs": epochs,
                        "seed": seed,
                        "training_subjects": sorted(training_subjects),
                        "holdout_subjects": sorted(holdout),
                        "prediction_subjects": sorted(holdout),
                        "globally_excluded_subjects": sorted(holdout),
                        "parent_artifact_sha256": {
                            "statistics_scaler": sha256_file(scaler_path),
                            "sensor_normalization": sha256_file(normalization_path),
                        },
                    },
                )
            seed_frames.append(
                infer_state_windows(
                    model,
                    holdout_dataset,
                    config,
                    stacking_partition=partition,
                    progress_label=f"state partition={partition} seed={seed} OOF inference",
                )
            )
            selected_epochs[seed].append(epochs)
            artifacts.extend((checkpoint_path, selector_report_path))
        all_predictions.append(_average_state_prediction_frames(seed_frames))
    oof_predictions = pd.concat(all_predictions, ignore_index=True)
    oof_path = run.root / "oof" / "window_logits.parquet"
    gate_path = run.root / "oof" / "gate_diagnostics.parquet"
    gate_summary_path = run.root / "oof" / "gate_diagnostics.json"
    write_parquet_atomic(oof_path, oof_predictions)
    gate_diagnostics, gate_summary = _gate_diagnostics(oof_predictions)
    write_parquet_atomic(gate_path, gate_diagnostics)
    write_json_atomic(gate_summary_path, gate_summary)
    artifacts.extend((oof_path, gate_path, gate_summary_path))
    fixed_outer_epochs = {seed: int(np.median(values)) for seed, values in selected_epochs.items()}
    outer_scaler, outer_transformed = _fit_scaler_and_transform(inputs, outer_train)
    outer_normalization = compute_normalization(inputs.segments, outer_train)
    outer_train_rows = outer_transformed[
        outer_transformed["subject_key"].astype(str).isin(outer_train)
    ].reset_index(drop=True)
    outer_train_events = inputs.events[inputs.events["subject_key"].astype(str).isin(outer_train)]
    outer_state_root = run.root / "outer" / "state"
    outer_scaler_path = outer_state_root / "statistics_scaler.json"
    outer_normalization_path = outer_state_root / "sensor_normalization.json"
    write_json_atomic(outer_scaler_path, outer_scaler.to_json())
    save_normalization(outer_normalization, outer_normalization_path)
    outer_rows = outer_transformed[
        outer_transformed["subject_key"].astype(str).isin(outer_test)
    ].reset_index(drop=True)
    for target in ("state_target", "start_target", "end_target"):
        if target in outer_rows:
            outer_rows[target] = 0.0
    for mask in ("state_loss_mask", "start_loss_mask", "end_loss_mask"):
        if mask in outer_rows:
            outer_rows[mask] = 0.0
    outer_inference_dataset = _make_dataset(
        outer_rows,
        inputs,
        inputs.events.iloc[0:0],
        outer_normalization,
        config,
        training=False,
        seed=int(config["training"]["random_seed"]),
    )
    outer_logits_path = run.root / "outer" / "window_logits.parquet"
    outer_frames: list[pd.DataFrame] = []
    outer_checkpoints: list[Path] = []
    for seed in state_seeds:
        outer_dataset = _make_dataset(
            outer_train_rows,
            inputs,
            outer_train_events,
            outer_normalization,
            config,
            training=True,
            seed=seed,
        )
        torch.manual_seed(seed)
        outer_model = build_state_model(config["model"])
        epochs = fixed_outer_epochs[seed]
        _train_state_epochs(
            outer_model,
            outer_dataset,
            config,
            epochs=epochs,
            seed=seed + 50_000,
            progress_label=f"state outer seed={seed} retrain",
            total_epochs=epochs,
            scheduler_total_epochs=int(config["training"]["max_epochs"]),
        )
        checkpoint_path = outer_state_root / f"state_seed_{seed}.pt"
        _save_torch_atomic(
            checkpoint_path,
            {
                "model": outer_model.state_dict(),
                "model_config": config["model"],
                "epochs": epochs,
                "seed": seed,
                "training_subjects": sorted(outer_train),
                "prediction_subjects": sorted(outer_test),
                "globally_excluded_subjects": sorted(outer_test),
                "parent_artifact_sha256": {
                    "statistics_scaler": sha256_file(outer_scaler_path),
                    "sensor_normalization": sha256_file(outer_normalization_path),
                },
            },
        )
        outer_checkpoints.append(checkpoint_path)
        outer_frames.append(
            infer_state_windows(
                outer_model,
                outer_inference_dataset,
                config,
                stacking_partition=-1,
                progress_label=f"state outer seed={seed} inference",
            )
        )
    write_parquet_atomic(
        outer_logits_path,
        _average_state_prediction_frames(outer_frames).drop(
            columns=["state_target", "state_loss_mask"]
        ),
    )
    artifacts.extend(
        (*outer_checkpoints, outer_scaler_path, outer_normalization_path, outer_logits_path)
    )
    epoch_path = run.root / "selection" / "state_epochs.json"
    write_json_atomic(
        epoch_path,
        {
            "state_seeds": state_seeds,
            "partition_epochs_by_seed": {
                str(seed): values for seed, values in selected_epochs.items()
            },
            "fixed_outer_epoch_by_seed": {
                str(seed): value for seed, value in fixed_outer_epochs.items()
            },
        },
    )
    artifacts.append(epoch_path)
    run.transition(
        "STATE_COMPLETE",
        artifacts,
        updates={
            "outer_train_subjects": sorted(outer_train),
            "outer_test_subjects": sorted(outer_test),
            "stacking_partitions": partitions,
        },
    )


def _candidate_recall(proposals: pd.DataFrame, events: pd.DataFrame) -> dict[str, float]:
    valid_events = events[events["evaluable"].fillna(False)] if "evaluable" in events else events
    matched: set[str] = set()
    grouped = {
        str(subject): group for subject, group in proposals.groupby("subject_key", sort=False)
    }
    for event in valid_events.itertuples(index=False):
        candidates = grouped.get(str(event.subject_key), pd.DataFrame())
        if any(
            interval_iou(
                int(event.start_ms),
                int(event.end_ms),
                int(candidate.coarse_start_ms),
                int(candidate.coarse_end_ms),
            )
            > 0.25
            for candidate in candidates.itertuples(index=False)
        ):
            matched.add(str(event.event_id))
    metrics = {"candidate_recall": len(matched) / len(valid_events) if len(valid_events) else 0.0}
    for relation in ("same", "different"):
        subset = valid_events[valid_events.get("hand_relation", "unknown") == relation]
        identifiers = set(subset["event_id"].astype(str)) if "event_id" in subset else set()
        metrics[f"{relation}_candidate_recall"] = (
            len(matched & identifiers) / len(identifiers) if identifiers else float("nan")
        )
    return metrics


def build_candidates_v4(
    run: HierarchicalRun,
    config: dict[str, Any],
    inputs: V4Inputs,
) -> None:
    run.require_stage("STATE_COMPLETE")
    logits = pd.read_parquet(run.root / "oof" / "window_logits.parquet")
    calibrated, calibrator = subject_crossfit_platt(
        logits,
        fit_mask_column="state_loss_mask",
    )
    calibrated["onset_probability"] = 1.0 / (1.0 + np.exp(-calibrated["onset_logit"]))
    calibrated["offset_probability"] = 1.0 / (1.0 + np.exp(-calibrated["offset_logit"]))
    calibrated["state_probability_derivative"] = (
        calibrated.groupby(["subject_key", "session_id"], sort=False)["state_probability"]
        .diff()
        .fillna(0.0)
    )
    calibration_rows = calibrated[calibrated["state_loss_mask"].to_numpy(dtype=float) > 0]
    metrics = state_calibration_metrics(
        calibration_rows["state_target"].to_numpy(),
        calibration_rows["state_logit"].to_numpy(),
        calibration_rows["state_probability"].to_numpy(),
        low_threshold=float(config["decoder"]["low_threshold"]),
        bins=int(config["calibration"]["ece_bins"]),
    )
    calibration_checks = {
        "ece": metrics["ece"] <= float(config["promotion_gate"]["maximum_state_ece"]),
        "brier": metrics["brier"] < metrics["uncalibrated_brier"],
        "prevalence_ratio": metrics["mean_probability_to_prevalence"]
        <= float(config["promotion_gate"]["maximum_state_prevalence_ratio"]),
    }
    strict_gates = str(config["experiment"].get("ablation_id", "S4")) == "S4"
    if strict_gates and not all(calibration_checks.values()):
        failed = [name for name, passed in calibration_checks.items() if not passed]
        raise RuntimeError(f"State calibration gate failed: {failed}")
    training_subjects = set(logits["subject_key"].astype(str))
    durations = _truth_event_durations(inputs.events, training_subjects)
    decoder_config = config["decoder"]
    prior = TruncatedLogNormalDurationPrior.fit(
        durations,
        lower_quantile=float(decoder_config["duration_lower_quantile"]),
        upper_quantile=float(decoder_config["duration_upper_quantile"]),
        minimum_floor_seconds=float(decoder_config["minimum_duration_floor_seconds"]),
        maximum_ceiling_seconds=float(decoder_config["maximum_duration_ceiling_seconds"]),
    )
    decoder = FixedLagSemiMarkovDecoder(
        prior,
        grid_seconds=int(decoder_config["grid_seconds"]),
        fixed_lag_seconds=int(decoder_config["fixed_lag_seconds"]),
    )
    proposals = generate_event_candidates_v4(
        calibrated, decoder, decoder_config, split_role="outer_train_oof"
    )
    evaluable, ignored = partition_evaluation_events(inputs.events, training_subjects)
    labeled = exclude_ignored_candidates(
        label_event_candidates(proposals, evaluable, 0.25), ignored
    )
    candidate_metrics = _candidate_recall(proposals, evaluable)
    candidate_metrics.update(hysteresis_fragment_diagnostics(calibrated, decoder_config))
    candidate_metrics["candidate_count"] = float(len(proposals))
    candidate_metrics["candidates_per_hour"] = float(
        len(proposals) / max(_observed_hours(calibrated), 1e-9)
    )
    if len(labeled):
        state_only_point = _best_verifier_operating_point(
            labeled,
            evaluable,
            calibrated,
            "generator_score",
            config,
            ignored,
        )
        state_only_accepted = _accepted_from_point(labeled, state_only_point, "generator_score")
    else:
        state_only_point = {
            "acceptance_threshold": float("inf"),
            "nms_iou_threshold": float(config["calibration"]["nms_iou_thresholds"][0]),
            "f1": 0.0,
            "fp_per_hour": 0.0,
        }
        state_only_accepted = labeled.copy()
    state_only_hand = _hand_metrics(evaluable, _prediction_events(state_only_accepted), ignored)
    candidate_metrics.update(
        {
            "state_only_f1": float(state_only_point["f1"]),
            "state_only_fp_per_hour": float(state_only_point["fp_per_hour"]),
            "state_only_acceptance_threshold": float(state_only_point["acceptance_threshold"]),
            "state_only_nms_iou_threshold": float(state_only_point["nms_iou_threshold"]),
            **state_only_hand,
            "state_calibration_gate_passed": bool(all(calibration_checks.values())),
        }
    )
    candidate_gate_passed = candidate_metrics["candidate_recall"] >= float(
        config["promotion_gate"]["minimum_candidate_recall"]
    )
    candidate_metrics["candidate_gate_passed"] = bool(candidate_gate_passed)
    if strict_gates and not candidate_gate_passed:
        raise RuntimeError(
            f"XGBoost-free candidate recall gate failed: {candidate_metrics['candidate_recall']:.6f}"
        )
    window_path = run.root / "oof" / "window_predictions.parquet"
    proposal_path = run.root / "oof" / "proposals_labeled.parquet"
    outer_logits = pd.read_parquet(run.root / "outer" / "window_logits.parquet")
    outer_logits["state_probability"] = calibrator.transform(outer_logits["state_logit"].to_numpy())
    outer_logits["onset_probability"] = 1.0 / (1.0 + np.exp(-outer_logits["onset_logit"]))
    outer_logits["offset_probability"] = 1.0 / (1.0 + np.exp(-outer_logits["offset_logit"]))
    outer_logits["state_probability_derivative"] = (
        outer_logits.groupby(["subject_key", "session_id"], sort=False)["state_probability"]
        .diff()
        .fillna(0.0)
    )
    outer_proposals = generate_event_candidates_v4(
        outer_logits, decoder, decoder_config, split_role="outer_test_unlabeled"
    )
    outer_window_path = run.root / "outer" / "window_predictions.parquet"
    outer_proposal_path = run.root / "outer" / "proposals.parquet"
    calibration_path = run.root / "decoder" / "state_calibration.json"
    duration_path = run.root / "decoder" / "duration_prior.json"
    metrics_path = run.root / "decoder" / "candidate_metrics.json"
    write_parquet_atomic(window_path, calibrated)
    write_parquet_atomic(proposal_path, labeled)
    write_parquet_atomic(outer_window_path, outer_logits)
    write_parquet_atomic(outer_proposal_path, outer_proposals)
    write_json_atomic(calibration_path, calibrator.to_json())
    write_json_atomic(duration_path, prior.to_json())
    write_json_atomic(metrics_path, {**metrics, **candidate_metrics})
    run.transition(
        "PROPOSALS_COMPLETE",
        [
            window_path,
            proposal_path,
            outer_window_path,
            outer_proposal_path,
            calibration_path,
            duration_path,
            metrics_path,
        ],
    )


def _assert_nested_lineage(
    *,
    training_subjects: set[str],
    prediction_subjects: set[str],
    globally_excluded_subjects: set[str],
) -> None:
    training = {str(value) for value in training_subjects}
    prediction = {str(value) for value in prediction_subjects}
    excluded = {str(value) for value in globally_excluded_subjects}
    if training & prediction:
        raise RuntimeError(
            f"Nested state training includes prediction subjects: {sorted(training & prediction)}"
        )
    if training & excluded:
        raise RuntimeError(
            f"Nested state training includes globally excluded subjects: {sorted(training & excluded)}"
        )


def _state_probabilities(
    oof_logits: pd.DataFrame, prediction_logits: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    calibrated, calibrator = subject_crossfit_platt(
        oof_logits,
        fit_mask_column="state_loss_mask",
    )
    prediction = prediction_logits.copy()
    prediction["state_probability"] = calibrator.transform(
        prediction["state_logit"].to_numpy(dtype=float)
    )
    for frame in (calibrated, prediction):
        frame["onset_probability"] = 1.0 / (1.0 + np.exp(-frame["onset_logit"]))
        frame["offset_probability"] = 1.0 / (1.0 + np.exp(-frame["offset_logit"]))
        frame["state_probability_derivative"] = (
            frame.groupby(["subject_key", "session_id"], sort=False)["state_probability"]
            .diff()
            .fillna(0.0)
        )
    return calibrated, prediction, calibrator


def _nested_cache_key(
    config: dict[str, Any],
    training_subjects: set[str],
    excluded_subjects: set[str],
    parent_artifact_sha256: dict[str, str] | None = None,
) -> str:
    payload = {
        "protocol_version": "statsfusion-r2",
        "training_subjects": sorted(training_subjects),
        "globally_excluded_subjects": sorted(excluded_subjects),
        "model": config["model"],
        "sequence": config["sequence"],
        "training": config["training"],
        "decoder": config["decoder"],
        "state_seeds": config["final_training"]["state_seeds"],
        "parent_artifact_sha256": parent_artifact_sha256 or {},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _nested_cache_artifacts(
    run: HierarchicalRun,
    lineage: dict[str, Any],
    required: dict[str, Path],
    lineage_path: Path,
    *,
    training_subjects: set[str],
    holdout_subjects: set[str],
    state_seeds: list[int],
    cache_key: str,
    parent_artifact_sha256: dict[str, str],
) -> list[Path]:
    if lineage.get("protocol_version") != "statsfusion-r2":
        raise RuntimeError("Nested state cache uses a blocked protocol")
    if lineage.get("meta_crossfit_protocol") != "fully_nested_v1":
        raise RuntimeError("Nested state cache is not fully nested")
    if lineage.get("cache_key") != cache_key:
        raise RuntimeError("Nested state cache identity changed; use a fresh run name")
    if set(lineage.get("training_subjects", [])) != training_subjects:
        raise RuntimeError("Nested state cache training subjects changed")
    if set(lineage.get("prediction_subjects", [])) != holdout_subjects:
        raise RuntimeError("Nested state cache prediction subjects changed")
    if set(lineage.get("globally_excluded_subjects", [])) != holdout_subjects:
        raise RuntimeError("Nested state cache exclusions changed")
    if [int(value) for value in lineage.get("state_seeds", [])] != state_seeds:
        raise RuntimeError("Nested state cache seed mirror changed")
    if lineage.get("parent_artifact_sha256") != parent_artifact_sha256:
        raise RuntimeError("Nested state cache parent artifacts changed")
    _assert_nested_lineage(
        training_subjects=training_subjects,
        prediction_subjects=holdout_subjects,
        globally_excluded_subjects=holdout_subjects,
    )
    artifacts = [*required.values(), lineage_path]
    for name, path in required.items():
        if not path.is_file():
            raise RuntimeError(f"Nested state cache artifact is missing: {name}")
        if lineage.get("artifact_sha256", {}).get(name) != sha256_file(path):
            raise RuntimeError(f"Nested state cache artifact changed: {name}")
    inner_models = lineage.get("inner_models", [])
    if not inner_models:
        raise RuntimeError("Nested state cache has no inner model lineage")
    prediction_counts = {subject: 0 for subject in training_subjects}
    partition_seeds: dict[int, set[int]] = {}
    for model in inner_models:
        model_training = {str(value) for value in model.get("training_subjects", [])}
        model_prediction = {str(value) for value in model.get("prediction_subjects", [])}
        model_excluded = {str(value) for value in model.get("globally_excluded_subjects", [])}
        _assert_nested_lineage(
            training_subjects=model_training,
            prediction_subjects=model_prediction,
            globally_excluded_subjects=model_excluded,
        )
        if model_excluded != holdout_subjects:
            raise RuntimeError("Nested inner model exclusions changed")
        if model_training | model_prediction != training_subjects:
            raise RuntimeError("Nested inner model does not partition the meta-training subjects")
        for subject in model_prediction:
            prediction_counts[subject] += 1
        partition = int(model["inner_partition"])
        partition_seeds.setdefault(partition, set()).add(int(model["seed"]))
        for path_key, hash_key in (
            ("checkpoint_path", "checkpoint_sha256"),
            ("selector_path", "selector_sha256"),
            ("scaler_path", "scaler_sha256"),
            ("normalization_path", "normalization_sha256"),
        ):
            relative = Path(str(model[path_key]))
            artifact = run.root / relative
            if not artifact.is_file() or sha256_file(artifact) != model.get(hash_key):
                raise RuntimeError(f"Nested inner model artifact changed: {relative.as_posix()}")
            artifacts.append(artifact)
        checkpoint_path = run.root / Path(str(model["checkpoint_path"]))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(checkpoint.get("epochs", -1)) != int(model.get("selected_epoch", -2)):
            raise RuntimeError("Nested inner model epoch lineage changed")
        expected_preprocessing_parents = {
            "statistics_scaler": model["scaler_sha256"],
            "sensor_normalization": model["normalization_sha256"],
        }
        if checkpoint.get("parent_artifact_sha256") != expected_preprocessing_parents:
            raise RuntimeError("Nested inner model preprocessing parents changed")
    expected_seeds = set(state_seeds)
    if any(seeds != expected_seeds for seeds in partition_seeds.values()):
        raise RuntimeError("Nested state cache has an incomplete seed mirror")
    if any(count != len(state_seeds) for count in prediction_counts.values()):
        raise RuntimeError("Nested state cache does not predict every training subject once per seed")
    return list(dict.fromkeys(artifacts))


def _state_holdout_parent_lineage(
    run: HierarchicalRun,
    *,
    meta_partition: int,
    training_subjects: set[str],
    holdout_subjects: set[str],
    state_seeds: list[int],
) -> dict[str, str]:
    state_root = run.root / "crossfit" / f"partition_{meta_partition}" / "state"
    scaler_path = state_root / "statistics_scaler.json"
    normalization_path = state_root / "sensor_normalization.json"
    if not scaler_path.is_file() or not normalization_path.is_file():
        raise FileNotFoundError("Nested meta holdout state preprocessing lineage is missing")
    scaler_payload = json.loads(scaler_path.read_text(encoding="utf-8"))
    if set(scaler_payload.get("training_subjects", [])) != training_subjects:
        raise RuntimeError("Meta holdout state scaler was not fit only on its training subjects")
    parent_artifact_sha256 = {
        "outer_state_statistics_scaler": sha256_file(scaler_path),
        "outer_state_sensor_normalization": sha256_file(normalization_path),
    }
    for seed in state_seeds:
        checkpoint_path = state_root / f"seed_{seed}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Meta holdout state checkpoint is missing for seed {seed}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        prediction_subjects = set(
            checkpoint.get("prediction_subjects", checkpoint.get("holdout_subjects", []))
        )
        excluded_subjects = set(checkpoint.get("globally_excluded_subjects", []))
        _assert_nested_lineage(
            training_subjects=set(checkpoint.get("training_subjects", [])),
            prediction_subjects=prediction_subjects,
            globally_excluded_subjects=excluded_subjects,
        )
        if set(checkpoint.get("training_subjects", [])) != training_subjects:
            raise RuntimeError("Meta holdout state checkpoint training subjects changed")
        if prediction_subjects != holdout_subjects or excluded_subjects != holdout_subjects:
            raise RuntimeError("Meta holdout state checkpoint prediction lineage changed")
        expected_parents = {
            "statistics_scaler": sha256_file(scaler_path),
            "sensor_normalization": sha256_file(normalization_path),
        }
        if checkpoint.get("parent_artifact_sha256") != expected_parents:
            raise RuntimeError("Meta holdout state checkpoint preprocessing parents changed")
        parent_artifact_sha256[f"outer_state_checkpoint_seed_{seed}"] = sha256_file(
            checkpoint_path
        )
    return parent_artifact_sha256


def _prepare_nested_meta_cache(
    run: HierarchicalRun,
    config: dict[str, Any],
    inputs: V4Inputs,
    *,
    meta_partition: int,
    training_subjects: set[str],
    holdout_subjects: set[str],
) -> tuple[Path, list[Path]]:
    _assert_nested_lineage(
        training_subjects=training_subjects,
        prediction_subjects=holdout_subjects,
        globally_excluded_subjects=holdout_subjects,
    )
    root = run.root / "nested" / f"meta_{meta_partition}"
    lineage_path = root / "lineage.json"
    required = {
        "train_windows": root / "train_window_predictions.parquet",
        "train_proposals": root / "train_proposals_labeled.parquet",
        "holdout_windows": root / "holdout_window_predictions.parquet",
        "holdout_proposals": root / "holdout_proposals.parquet",
        "calibration": root / "state_calibration.json",
        "duration_prior": root / "duration_prior.json",
    }
    outer_oof_path = run.root / "oof" / "window_logits.parquet"
    if not outer_oof_path.is_file():
        raise FileNotFoundError("Nested meta-crossfit requires outer-train OOF state logits")
    state_seeds = [int(value) for value in config["final_training"]["state_seeds"]]
    parent_artifact_sha256 = {
        "outer_oof_window_logits": sha256_file(outer_oof_path),
        **_state_holdout_parent_lineage(
            run,
            meta_partition=meta_partition,
            training_subjects=training_subjects,
            holdout_subjects=holdout_subjects,
            state_seeds=state_seeds,
        ),
    }
    cache_key = _nested_cache_key(
        config,
        training_subjects,
        holdout_subjects,
        parent_artifact_sha256,
    )
    if lineage_path.is_file() and all(path.is_file() for path in required.values()):
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        return root, _nested_cache_artifacts(
            run,
            lineage,
            required,
            lineage_path,
            training_subjects=training_subjects,
            holdout_subjects=holdout_subjects,
            state_seeds=state_seeds,
            cache_key=cache_key,
            parent_artifact_sha256=parent_artifact_sha256,
        )

    inner_partitions = stacking_partitions(
        training_subjects,
        int(config["hierarchical"]["verifier_crossfit_partitions"]),
        int(config["training"]["random_seed"]) + 100_000 + meta_partition,
    )
    oof_parts: list[pd.DataFrame] = []
    artifacts: list[Path] = []
    model_lineage: list[dict[str, Any]] = []
    for inner_partition in sorted(set(inner_partitions.values())):
        prediction_subjects = {
            subject for subject, value in inner_partitions.items() if value == inner_partition
        }
        model_subjects = training_subjects - prediction_subjects
        _assert_nested_lineage(
            training_subjects=model_subjects,
            prediction_subjects=prediction_subjects,
            globally_excluded_subjects=holdout_subjects,
        )
        fit_subjects, selector_subjects, selector_split_report = _selector_split(
            model_subjects,
            float(config["training"]["selector_fraction"]),
            int(config["training"]["random_seed"])
            + meta_partition * 10_000
            + inner_partition,
            events=inputs.events,
            anchors=inputs.anchors,
        )
        scaler, transformed = _fit_scaler_and_transform(inputs, model_subjects)
        normalization = compute_normalization(inputs.segments, model_subjects)
        train_rows = transformed[
            transformed["subject_key"].astype(str).isin(model_subjects)
        ].reset_index(drop=True)
        train_events = inputs.events[
            inputs.events["subject_key"].astype(str).isin(model_subjects)
        ]
        prediction_rows = transformed[
            transformed["subject_key"].astype(str).isin(prediction_subjects)
        ].reset_index(drop=True)
        prediction_events = inputs.events[
            inputs.events["subject_key"].astype(str).isin(prediction_subjects)
        ]
        prediction_dataset = _make_dataset(
            prediction_rows,
            inputs,
            prediction_events,
            normalization,
            config,
            training=False,
            seed=int(config["training"]["random_seed"]),
        )
        seed_frames: list[pd.DataFrame] = []
        inner_root = root / "state" / f"inner_{inner_partition}"
        scaler_path = inner_root / "statistics_scaler.json"
        normalization_path = inner_root / "sensor_normalization.json"
        write_json_atomic(scaler_path, scaler.to_json())
        save_normalization(normalization, normalization_path)
        artifacts.extend((scaler_path, normalization_path))
        for seed in state_seeds:
            epochs, selector_report = _select_epoch(
                inputs.anchors[inputs.anchors["subject_key"].astype(str).isin(fit_subjects)],
                inputs.anchors[
                    inputs.anchors["subject_key"].astype(str).isin(selector_subjects)
                ],
                fit_subjects,
                inputs,
                config,
                seed + meta_partition * 100_000 + inner_partition * 10_000,
            )
            dataset = _make_dataset(
                train_rows,
                inputs,
                train_events,
                normalization,
                config,
                training=True,
                seed=seed,
            )
            torch.manual_seed(seed)
            model = build_state_model(config["model"])
            _train_state_epochs(
                model,
                dataset,
                config,
                epochs=epochs,
                seed=seed + meta_partition * 100_000 + inner_partition * 10_000,
                progress_label=(
                    f"nested state meta={meta_partition} inner={inner_partition} seed={seed}"
                ),
                total_epochs=epochs,
                scheduler_total_epochs=int(config["training"]["max_epochs"]),
            )
            checkpoint = inner_root / f"seed_{seed}.pt"
            selector_path = inner_root / f"selector_seed_{seed}.json"
            checkpoint_payload = {
                "model": model.state_dict(),
                "model_config": config["model"],
                "epochs": epochs,
                "seed": seed,
                "training_subjects": sorted(model_subjects),
                "prediction_subjects": sorted(prediction_subjects),
                "globally_excluded_subjects": sorted(holdout_subjects),
                "parent_artifact_sha256": {
                    "statistics_scaler": sha256_file(scaler_path),
                    "sensor_normalization": sha256_file(normalization_path),
                },
            }
            _save_torch_atomic(checkpoint, checkpoint_payload)
            write_json_atomic(
                selector_path,
                {
                    **selector_report,
                    "fit_subjects": sorted(fit_subjects),
                    "selector_subjects": sorted(selector_subjects),
                    "selector_split": selector_split_report,
                    "training_subjects": sorted(model_subjects),
                    "prediction_subjects": sorted(prediction_subjects),
                    "globally_excluded_subjects": sorted(holdout_subjects),
                    "parent_artifact_sha256": {
                        "statistics_scaler": sha256_file(scaler_path),
                        "sensor_normalization": sha256_file(normalization_path),
                    },
                },
            )
            artifacts.extend((checkpoint, selector_path))
            model_lineage.append(
                {
                    "seed": seed,
                    "inner_partition": inner_partition,
                    "training_subjects": sorted(model_subjects),
                    "prediction_subjects": sorted(prediction_subjects),
                    "globally_excluded_subjects": sorted(holdout_subjects),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "checkpoint_path": checkpoint.relative_to(run.root).as_posix(),
                    "selector_sha256": sha256_file(selector_path),
                    "selector_path": selector_path.relative_to(run.root).as_posix(),
                    "scaler_sha256": sha256_file(scaler_path),
                    "scaler_path": scaler_path.relative_to(run.root).as_posix(),
                    "normalization_sha256": sha256_file(normalization_path),
                    "normalization_path": normalization_path.relative_to(run.root).as_posix(),
                    "selected_epoch": int(epochs),
                }
            )
            seed_frames.append(
                infer_state_windows(
                    model,
                    prediction_dataset,
                    config,
                    stacking_partition=inner_partition,
                    progress_label=(
                        f"nested state meta={meta_partition} inner={inner_partition} inference"
                    ),
                )
            )
        oof_parts.append(_average_state_prediction_frames(seed_frames))

    nested_oof_logits = pd.concat(oof_parts, ignore_index=True)
    holdout_logits = pd.read_parquet(outer_oof_path)
    holdout_logits = holdout_logits[
        holdout_logits["subject_key"].astype(str).isin(holdout_subjects)
    ].reset_index(drop=True)
    observed_holdout_subjects = set(holdout_logits["subject_key"].astype(str))
    if observed_holdout_subjects != holdout_subjects:
        raise RuntimeError("Nested meta holdout logits do not cover the expected subjects")
    if "stacking_partition" not in holdout_logits:
        raise RuntimeError("Nested meta holdout logits lack state stacking lineage")
    if set(holdout_logits["stacking_partition"].astype(int)) != {int(meta_partition)}:
        raise RuntimeError("Nested meta holdout logits came from a different state partition")
    calibrated_train, calibrated_holdout, calibrator = _state_probabilities(
        nested_oof_logits, holdout_logits
    )
    durations = _truth_event_durations(inputs.events, training_subjects)
    decoder_config = config["decoder"]
    prior = TruncatedLogNormalDurationPrior.fit(
        durations,
        lower_quantile=float(decoder_config["duration_lower_quantile"]),
        upper_quantile=float(decoder_config["duration_upper_quantile"]),
        minimum_floor_seconds=float(decoder_config["minimum_duration_floor_seconds"]),
        maximum_ceiling_seconds=float(decoder_config["maximum_duration_ceiling_seconds"]),
    )
    decoder = FixedLagSemiMarkovDecoder(
        prior,
        grid_seconds=int(decoder_config["grid_seconds"]),
        fixed_lag_seconds=int(decoder_config["fixed_lag_seconds"]),
    )
    train_proposals = generate_event_candidates_v4(
        calibrated_train, decoder, decoder_config, split_role="nested_meta_train_oof"
    )
    train_truth, train_ignore = partition_evaluation_events(inputs.events, training_subjects)
    train_labeled = exclude_ignored_candidates(
        label_event_candidates(train_proposals, train_truth, 0.25), train_ignore
    )
    holdout_proposals = generate_event_candidates_v4(
        calibrated_holdout, decoder, decoder_config, split_role="nested_meta_holdout"
    )
    write_parquet_atomic(required["train_windows"], calibrated_train)
    write_parquet_atomic(required["train_proposals"], train_labeled)
    write_parquet_atomic(required["holdout_windows"], calibrated_holdout)
    write_parquet_atomic(required["holdout_proposals"], holdout_proposals)
    write_json_atomic(required["calibration"], calibrator.to_json())
    write_json_atomic(required["duration_prior"], prior.to_json())
    artifact_sha256 = {name: sha256_file(path) for name, path in required.items()}
    write_json_atomic(
        lineage_path,
        {
            "protocol_version": "statsfusion-r2",
            "meta_crossfit_protocol": "fully_nested_v1",
            "cache_key": cache_key,
            "meta_partition": meta_partition,
            "training_subjects": sorted(training_subjects),
            "prediction_subjects": sorted(holdout_subjects),
            "globally_excluded_subjects": sorted(holdout_subjects),
            "state_seeds": state_seeds,
            "inner_models": model_lineage,
            "parent_artifact_sha256": parent_artifact_sha256,
            "state_calibration_source": "nested_meta_train_oof",
            "duration_prior_source": "nested_meta_training_truth",
            "artifact_sha256": artifact_sha256,
        },
    )
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    return root, _nested_cache_artifacts(
        run,
        lineage,
        required,
        lineage_path,
        training_subjects=training_subjects,
        holdout_subjects=holdout_subjects,
        state_seeds=state_seeds,
        cache_key=cache_key,
        parent_artifact_sha256=parent_artifact_sha256,
    )


def _slice_proposal_features(features, indices: np.ndarray):
    from bme_eating.models.event_verifier_v4 import ProposalFeatureBatchV4

    return ProposalFeatureBatchV4(
        proposal_ids=features.proposal_ids[indices],
        sequence=features.sequence[indices],
        sequence_mask=features.sequence_mask[indices],
        scalar=features.scalar[indices],
        event_target=features.event_target[indices] if features.event_target is not None else None,
        iou_target=features.iou_target[indices] if features.iou_target is not None else None,
        sample_weight=features.sample_weight[indices]
        if features.sample_weight is not None
        else None,
    )


def _verifier_matrix(features) -> np.ndarray:
    mask = features.sequence_mask[..., None]
    count = mask.sum(axis=1).clip(min=1)
    mean = (features.sequence * mask).sum(axis=1) / count
    maximum = np.where(mask, features.sequence, -np.inf).max(axis=1)
    maximum[~np.isfinite(maximum)] = 0.0
    return np.concatenate((mean, maximum, features.scalar), axis=1)


def _train_verifier_model_v4(
    features,
    categories: np.ndarray,
    config: dict[str, Any],
    *,
    seed: int,
    epochs: int | None = None,
) -> EventVerifierV4:
    torch.manual_seed(seed)
    device = _device(config)
    model = EventVerifierV4(
        features.sequence.shape[-1], features.scalar.shape[-1], config["verifier"]
    ).to(device)
    dataset = ProposalDatasetV4(features)
    batch_size = int(config["verifier"]["batch_size"])
    steps = max(1, math.ceil(len(dataset) / batch_size))
    sampler = HardNegativeBatchSampler(
        categories,
        batch_size=batch_size,
        ratios=config["verifier"]["batch_composition"],
        steps_per_epoch=steps,
        seed=seed,
        target_weights=features.sample_weight,
    )
    loader = DataLoader(dataset, batch_sampler=sampler)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["verifier"]["learning_rate"]),
        weight_decay=float(config["verifier"]["weight_decay"]),
    )
    maximum_epochs = int(epochs or config["verifier"]["max_epochs"])
    progress = tqdm(
        range(maximum_epochs),
        desc=f"verifier seed={seed}",
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in progress:
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss_total = torch.zeros((), device=device)
        epoch_steps = 0
        for batch in loader:
            tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            output = model(tensors)
            loss, _ = verifier_loss_v4(
                output, tensors, float(config["verifier"]["iou_loss_weight"])
            )
            epoch_loss_total += loss.detach()
            epoch_steps += 1
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        progress.set_postfix(loss=f"{float(epoch_loss_total.cpu()) / max(epoch_steps, 1):.4f}")
    return model


def _select_verifier_epoch(
    fit_features,
    fit_categories: np.ndarray,
    selector_features,
    selector_proposals: pd.DataFrame,
    selector_truth: pd.DataFrame,
    selector_ignore: pd.DataFrame,
    selector_windows: pd.DataFrame,
    config: dict[str, Any],
    *,
    seed: int,
) -> tuple[int, list[dict[str, float]]]:
    torch.manual_seed(seed)
    device = _device(config)
    model = EventVerifierV4(
        fit_features.sequence.shape[-1], fit_features.scalar.shape[-1], config["verifier"]
    ).to(device)
    dataset = ProposalDatasetV4(fit_features)
    batch_size = int(config["verifier"]["batch_size"])
    sampler = HardNegativeBatchSampler(
        fit_categories,
        batch_size=batch_size,
        ratios=config["verifier"]["batch_composition"],
        steps_per_epoch=max(1, math.ceil(len(dataset) / batch_size)),
        seed=seed,
        target_weights=fit_features.sample_weight,
    )
    loader = DataLoader(dataset, batch_sampler=sampler)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["verifier"]["learning_rate"]),
        weight_decay=float(config["verifier"]["weight_decay"]),
    )
    best_epoch = 1
    best_key = (-math.inf, -math.inf)
    patience = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(config["verifier"]["max_epochs"]) + 1):
        sampler.set_epoch(epoch - 1)
        model.train()
        for batch in loader:
            tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            output = model(tensors)
            loss, _ = verifier_loss_v4(
                output, tensors, float(config["verifier"]["iou_loss_weight"])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        event_logit, iou_logit = _infer_verifier_model(model, selector_features, config)
        scored = selector_proposals.copy().reset_index(drop=True)
        scored["event_logit"] = event_logit
        scored["iou_logit"] = iou_logit
        scored["final_score"] = 1.0 / (1.0 + np.exp(-event_logit))
        point = _best_verifier_operating_point(
            scored,
            selector_truth,
            selector_windows,
            "final_score",
            config,
            selector_ignore,
        )
        key = (float(point["f1"]), -float(point["fp_per_hour"]))
        history.append(
            {
                "epoch": float(epoch),
                "event_f1": float(point["f1"]),
                "fp_per_hour": float(point["fp_per_hour"]),
            }
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= int(config["verifier"]["patience"]):
            break
    return best_epoch, history


@torch.no_grad()
def _infer_verifier_model(model: EventVerifierV4, features, config: dict[str, Any]):
    device = _device(config)
    model.to(device).eval()
    loader = DataLoader(
        ProposalDatasetV4(features),
        batch_size=int(config["verifier"]["batch_size"]),
        shuffle=False,
    )
    event: list[np.ndarray] = []
    iou: list[np.ndarray] = []
    for batch in tqdm(
        loader,
        desc="verifier inference",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    ):
        tensors = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        output = model(tensors)
        event.append(output["event_logit"].cpu().numpy())
        iou.append(output["iou_logit"].cpu().numpy())
    return np.concatenate(event), np.concatenate(iou)


def _nms(frame: pd.DataFrame, threshold: float, score_column: str) -> pd.DataFrame:
    kept: list[int] = []
    ranked = frame.sort_values(
        [score_column, "coarse_start_ms", "coarse_end_ms", "proposal_id"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    for index in ranked.index:
        candidate = frame.loc[index]
        if any(
            interval_iou(
                int(candidate.coarse_start_ms),
                int(candidate.coarse_end_ms),
                int(frame.loc[other].coarse_start_ms),
                int(frame.loc[other].coarse_end_ms),
            )
            > threshold
            for other in kept
        ):
            continue
        kept.append(int(index))
    return frame.loc[kept].copy()


def _gap_aware_nms(
    frame: pd.DataFrame,
    threshold: float,
    score_column: str,
    minimum_gap_seconds: int = 3,
) -> pd.DataFrame:
    candidates = _nms(frame, threshold, score_column)
    gap_ms = int(minimum_gap_seconds) * 1000
    kept: list[int] = []
    ranked = candidates.sort_values(
        [score_column, "coarse_start_ms", "coarse_end_ms", "proposal_id"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    for index, candidate in ranked.iterrows():
        start = int(candidate.coarse_start_ms)
        end = int(candidate.coarse_end_ms)
        conflicts = any(
            not (
                end + gap_ms <= int(candidates.loc[other].coarse_start_ms)
                or int(candidates.loc[other].coarse_end_ms) + gap_ms <= start
            )
            for other in kept
        )
        if not conflicts:
            kept.append(int(index))
    return candidates.loc[kept].copy()


def _prediction_events(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["subject_key", "start_ms", "end_ms", "proposal_id"])
    return frame.rename(columns={"coarse_start_ms": "start_ms", "coarse_end_ms": "end_ms"})[
        ["subject_key", "start_ms", "end_ms", "proposal_id"]
    ]


def _observed_hours(windows: pd.DataFrame) -> float:
    milliseconds = 0
    for _, group in windows.groupby(["subject_key", "session_id"], sort=False):
        if len(group):
            step = int(np.median(np.diff(group["timestamp_ms"]))) if len(group) > 1 else 3000
            milliseconds += int(group["timestamp_ms"].max() - group["timestamp_ms"].min() + step)
    return milliseconds / 3_600_000.0


def _best_verifier_operating_point(
    scores: pd.DataFrame,
    events: pd.DataFrame,
    windows: pd.DataFrame,
    score_column: str,
    config: dict[str, Any],
    ignore: pd.DataFrame | None = None,
) -> dict[str, float]:
    best: dict[str, float] | None = None
    values = scores[score_column].to_numpy(dtype=np.float64)
    for quantile in config["calibration"]["acceptance_quantiles"]:
        threshold = float(np.quantile(values, float(quantile)))
        for nms_threshold in config["calibration"]["nms_iou_thresholds"]:
            accepted = scores[scores[score_column] >= threshold]
            accepted = (
                pd.concat(
                    [
                        _gap_aware_nms(
                            group,
                            float(nms_threshold),
                            score_column,
                            int(config["boundary"]["safety_gap_seconds"]),
                        )
                        for _, group in accepted.groupby(["subject_key", "session_id"], sort=False)
                    ],
                    ignore_index=True,
                )
                if len(accepted)
                else accepted
            )
            metrics, _ = evaluate_events(
                events,
                _prediction_events(accepted),
                method="max_cardinality_iou",
                ignore=ignore,
            )
            metrics["fp_per_hour"] = metrics["false_positive"] / max(_observed_hours(windows), 1e-9)
            point = {
                "acceptance_threshold": threshold,
                "nms_iou_threshold": float(nms_threshold),
                **metrics,
            }
            if best is None or (point["f1"], -point["fp_per_hour"]) > (
                best["f1"],
                -best["fp_per_hour"],
            ):
                best = point
    if best is None:
        raise RuntimeError("No verifier operating point could be selected")
    return best


def _matching_ranking_reversal(
    maximum_cardinality_delta: float,
    greedy_delta: float,
    *,
    tolerance: float,
) -> bool:
    return bool(
        maximum_cardinality_delta * greedy_delta < 0
        and abs(maximum_cardinality_delta) > tolerance
        and abs(greedy_delta) > tolerance
    )


def _nested_verifier_oof_scores(
    root: Path,
    proposals: pd.DataFrame,
    features: ProposalFeatureBatchV4,
    categories: np.ndarray,
    selected_epochs: dict[int, int],
    config: dict[str, Any],
    *,
    meta_partition: int,
    globally_excluded_subjects: set[str],
) -> tuple[pd.DataFrame, list[Path]]:
    subjects = set(proposals["subject_key"].astype(str))
    parent_paths = {
        "nested_train_windows": root / "train_window_predictions.parquet",
        "nested_train_proposals": root / "train_proposals_labeled.parquet",
        "state_calibration": root / "state_calibration.json",
        "duration_prior": root / "duration_prior.json",
    }
    missing_parents = [name for name, path in parent_paths.items() if not path.is_file()]
    if missing_parents:
        raise FileNotFoundError(f"Nested verifier parents are missing: {missing_parents}")
    parent_artifact_sha256 = {
        name: sha256_file(path) for name, path in parent_paths.items()
    }
    partition_count = min(int(config["hierarchical"]["verifier_crossfit_partitions"]), len(subjects))
    if partition_count < 2:
        raise RuntimeError("Nested verifier OOF requires at least two training subjects")
    partitions = stacking_partitions(
        subjects,
        partition_count,
        int(config["training"]["random_seed"]) + 200_000 + meta_partition,
    )
    proposal_partition = proposals["subject_key"].astype(str).map(partitions).to_numpy(dtype=int)
    deep_event = np.zeros(len(proposals), dtype=np.float64)
    deep_iou = np.zeros(len(proposals), dtype=np.float64)
    logistic_score = np.zeros(len(proposals), dtype=np.float64)
    artifacts: list[Path] = []
    for partition in sorted(set(proposal_partition)):
        holdout_indices = np.flatnonzero(proposal_partition == partition)
        train_indices = np.flatnonzero(proposal_partition != partition)
        training_subjects = set(proposals.iloc[train_indices]["subject_key"].astype(str))
        prediction_subjects = set(proposals.iloc[holdout_indices]["subject_key"].astype(str))
        _assert_nested_lineage(
            training_subjects=training_subjects,
            prediction_subjects=prediction_subjects,
            globally_excluded_subjects=globally_excluded_subjects,
        )
        train_features = _slice_proposal_features(features, train_indices)
        holdout_features = _slice_proposal_features(features, holdout_indices)
        seed_predictions: list[tuple[np.ndarray, np.ndarray]] = []
        for seed in config["verifier"]["seeds"]:
            seed = int(seed)
            model = _train_verifier_model_v4(
                train_features,
                categories[train_indices],
                config,
                seed=seed + meta_partition * 10_000 + partition * 100,
                epochs=int(selected_epochs[seed]),
            )
            checkpoint = root / "verifier_oof" / f"partition_{partition}_seed_{seed}.pt"
            _save_torch_atomic(
                checkpoint,
                {
                    "model": model.state_dict(),
                    "sequence_dim": features.sequence.shape[-1],
                    "scalar_dim": features.scalar.shape[-1],
                    "config": config["verifier"],
                    "epochs": int(selected_epochs[seed]),
                    "seed": seed,
                    "training_subjects": sorted(training_subjects),
                    "prediction_subjects": sorted(prediction_subjects),
                    "globally_excluded_subjects": sorted(globally_excluded_subjects),
                    "parent_artifact_sha256": parent_artifact_sha256,
                    "state_calibration_source": "nested_meta_train_oof",
                    "duration_prior_source": "nested_meta_training_truth",
                },
            )
            artifacts.append(checkpoint)
            seed_predictions.append(_infer_verifier_model(model, holdout_features, config))
        deep_event[holdout_indices] = np.mean([value[0] for value in seed_predictions], axis=0)
        deep_iou[holdout_indices] = np.mean([value[1] for value in seed_predictions], axis=0)
        logistic = LogisticScoreCombiner.fit(
            _verifier_matrix(train_features),
            train_features.event_target,
            sample_weight=train_features.sample_weight,
        )
        logistic_score[holdout_indices] = logistic.predict(_verifier_matrix(holdout_features))
    scored = proposals.copy().reset_index(drop=True)
    scored["stacking_partition"] = proposal_partition
    scored["category"] = categories
    scored["sample_weight"] = features.sample_weight
    scored["event_logit"] = deep_event
    scored["iou_logit"] = deep_iou
    scored["predicted_iou"] = 1.0 / (1.0 + np.exp(-deep_iou))
    scored["state_score"] = features.scalar[:, 2]
    scored["logistic_score"] = logistic_score
    return scored, artifacts


def train_verifier_crossfit_v4(
    run: HierarchicalRun,
    config: dict[str, Any],
    inputs: V4Inputs,
) -> None:
    run.require_stage("PROPOSALS_COMPLETE")
    global_proposals = pd.read_parquet(run.root / "oof" / "proposals_labeled.parquet")
    global_windows = pd.read_parquet(run.root / "oof" / "window_predictions.parquet")
    global_features = build_proposal_features_v4(
        global_proposals,
        global_windows,
        [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
        config["verifier"],
    )
    _assert_proposal_feature_alignment(
        global_features, global_proposals, context="Final verifier training"
    )
    global_categories = classify_proposals(global_proposals)
    partitions = run.payload["stacking_partitions"]
    outer_train_subjects = set(partitions)
    artifacts: list[Path] = []
    scored_parts: list[pd.DataFrame] = []
    selected_verifier_epochs: dict[int, list[int]] = {
        int(seed): [] for seed in config["verifier"]["seeds"]
    }
    for partition in sorted(set(partitions.values())):
        holdout_subjects = {
            subject for subject, value in partitions.items() if int(value) == int(partition)
        }
        train_subjects = outer_train_subjects - holdout_subjects
        nested_root, nested_artifacts = _prepare_nested_meta_cache(
            run,
            config,
            inputs,
            meta_partition=int(partition),
            training_subjects=train_subjects,
            holdout_subjects=holdout_subjects,
        )
        artifacts.extend(nested_artifacts)
        train_proposals = pd.read_parquet(nested_root / "train_proposals_labeled.parquet")
        train_windows = pd.read_parquet(nested_root / "train_window_predictions.parquet")
        holdout_proposals = pd.read_parquet(nested_root / "holdout_proposals.parquet")
        holdout_windows = pd.read_parquet(nested_root / "holdout_window_predictions.parquet")
        nested_parent_paths = {
            "nested_train_windows": nested_root / "train_window_predictions.parquet",
            "nested_train_proposals": nested_root / "train_proposals_labeled.parquet",
            "nested_holdout_windows": nested_root / "holdout_window_predictions.parquet",
            "nested_holdout_proposals": nested_root / "holdout_proposals.parquet",
            "state_calibration": nested_root / "state_calibration.json",
            "duration_prior": nested_root / "duration_prior.json",
        }
        nested_parent_sha256 = {
            name: sha256_file(path) for name, path in nested_parent_paths.items()
        }
        holdout_truth, holdout_ignore = partition_evaluation_events(
            inputs.events, holdout_subjects
        )
        holdout_proposals = exclude_ignored_candidates(
            label_event_candidates(holdout_proposals, holdout_truth, 0.25), holdout_ignore
        )
        train_features = build_proposal_features_v4(
            train_proposals,
            train_windows,
            [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
            config["verifier"],
        )
        holdout_features = build_proposal_features_v4(
            holdout_proposals,
            holdout_windows,
            [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
            config["verifier"],
        )
        _assert_proposal_feature_alignment(
            train_features, train_proposals, context=f"Nested verifier train {partition}"
        )
        _assert_proposal_feature_alignment(
            holdout_features, holdout_proposals, context=f"Nested verifier holdout {partition}"
        )
        categories = classify_proposals(train_proposals)
        fit_subjects, selector_subjects, selector_split_report = _selector_split(
            train_subjects,
            float(config["training"]["selector_fraction"]),
            int(config["training"]["random_seed"]) + int(partition) + 60_000,
            events=inputs.events,
            anchors=inputs.anchors,
        )
        fit_indices = np.flatnonzero(
            train_proposals["subject_key"].astype(str).isin(fit_subjects).to_numpy()
        )
        selector_indices = np.flatnonzero(
            train_proposals["subject_key"].astype(str).isin(selector_subjects).to_numpy()
        )
        fit_features = _slice_proposal_features(train_features, fit_indices)
        selector_features = _slice_proposal_features(train_features, selector_indices)
        selector_proposals = train_proposals.iloc[selector_indices].reset_index(drop=True)
        selector_truth, selector_ignore = partition_evaluation_events(
            inputs.events, selector_subjects
        )
        selector_windows = train_windows[
            train_windows["subject_key"].astype(str).isin(selector_subjects)
        ]
        selected_epochs: dict[int, int] = {}
        holdout_seed_predictions: list[tuple[np.ndarray, np.ndarray]] = []
        for seed in config["verifier"]["seeds"]:
            seed = int(seed)
            selected_epoch, selection_history = _select_verifier_epoch(
                fit_features,
                categories[fit_indices],
                selector_features,
                selector_proposals,
                selector_truth,
                selector_ignore,
                selector_windows,
                config,
                seed=seed + int(partition) * 100,
            )
            selected_epochs[seed] = selected_epoch
            selected_verifier_epochs[seed].append(selected_epoch)
            model = _train_verifier_model_v4(
                train_features,
                categories,
                config,
                seed=seed + int(partition) * 100,
                epochs=selected_epoch,
            )
            checkpoint = run.root / "verifier" / f"partition_{partition}_seed_{seed}.pt"
            selector_path = (
                run.root / "verifier" / f"partition_{partition}_seed_{seed}_selector.json"
            )
            _save_torch_atomic(
                checkpoint,
                {
                    "model": model.state_dict(),
                    "sequence_dim": train_features.sequence.shape[-1],
                    "scalar_dim": train_features.scalar.shape[-1],
                    "config": config["verifier"],
                    "epochs": selected_epoch,
                    "seed": seed,
                    "training_subjects": sorted(train_subjects),
                    "prediction_subjects": sorted(holdout_subjects),
                    "globally_excluded_subjects": sorted(holdout_subjects),
                    "fit_subjects": sorted(fit_subjects),
                    "selector_subjects": sorted(selector_subjects),
                    "selector_split": selector_split_report,
                    "parent_artifact_sha256": nested_parent_sha256,
                    "state_calibration_source": "nested_meta_train_oof",
                    "duration_prior_source": "nested_meta_training_truth",
                },
            )
            write_json_atomic(
                selector_path,
                {
                    "selected_epoch": selected_epoch,
                    "history": selection_history,
                    "training_subjects": sorted(train_subjects),
                    "prediction_subjects": sorted(holdout_subjects),
                    "globally_excluded_subjects": sorted(holdout_subjects),
                    "fit_subjects": sorted(fit_subjects),
                    "selector_subjects": sorted(selector_subjects),
                    "selector_split": selector_split_report,
                    "parent_artifact_sha256": nested_parent_sha256,
                    "state_calibration_source": "nested_meta_train_oof",
                    "duration_prior_source": "nested_meta_training_truth",
                },
            )
            artifacts.extend((checkpoint, selector_path))
            holdout_seed_predictions.append(
                _infer_verifier_model(model, holdout_features, config)
            )
        train_oof_scores, verifier_oof_artifacts = _nested_verifier_oof_scores(
            nested_root,
            train_proposals,
            train_features,
            categories,
            selected_epochs,
            config,
            meta_partition=int(partition),
            globally_excluded_subjects=holdout_subjects,
        )
        artifacts.extend(verifier_oof_artifacts)
        calibration = ProposalCalibrationV4.fit(train_oof_scores)
        calibration_path = nested_root / "proposal_calibration.json"
        write_json_atomic(calibration_path, calibration.to_json())
        calibration_lineage_path = nested_root / "proposal_calibration_lineage.json"
        write_json_atomic(
            calibration_lineage_path,
            {
                "protocol_version": "statsfusion-r2",
                "training_subjects": sorted(train_subjects),
                "prediction_subjects": sorted(holdout_subjects),
                "globally_excluded_subjects": sorted(holdout_subjects),
                "parent_artifact_sha256": {
                    **nested_parent_sha256,
                    **{
                        checkpoint.relative_to(nested_root).as_posix(): sha256_file(checkpoint)
                        for checkpoint in verifier_oof_artifacts
                    },
                },
                "calibration_source": "nested_verifier_oof",
            },
        )
        artifacts.extend((calibration_path, calibration_lineage_path))
        logistic = LogisticScoreCombiner.fit(
            _verifier_matrix(train_features),
            train_features.event_target,
            sample_weight=train_features.sample_weight,
        )
        holdout_scored = holdout_proposals.copy().reset_index(drop=True)
        holdout_scored["stacking_partition"] = int(partition)
        holdout_scored["category"] = classify_proposals(holdout_proposals)
        holdout_scored["sample_weight"] = holdout_features.sample_weight
        holdout_scored["event_logit"] = np.mean(
            [value[0] for value in holdout_seed_predictions], axis=0
        )
        holdout_scored["iou_logit"] = np.mean(
            [value[1] for value in holdout_seed_predictions], axis=0
        )
        holdout_scored["predicted_iou"] = 1.0 / (
            1.0 + np.exp(-holdout_scored["iou_logit"])
        )
        holdout_scored["state_score"] = holdout_features.scalar[:, 2]
        holdout_scored["logistic_score"] = logistic.predict(
            _verifier_matrix(holdout_features)
        )
        holdout_scored = calibration.apply(holdout_scored)
        scored_parts.append(holdout_scored)

    scored = pd.concat(scored_parts, ignore_index=True).sort_values("proposal_id")
    final_calibration = ProposalCalibrationV4.fit(scored)
    logistic_final = LogisticScoreCombiner.fit(
        _verifier_matrix(global_features),
        global_features.event_target,
        sample_weight=global_features.sample_weight,
    )
    outer_proposals = pd.read_parquet(run.root / "outer" / "proposals.parquet")
    outer_windows = pd.read_parquet(run.root / "outer" / "window_predictions.parquet")
    outer_features = build_proposal_features_v4(
        outer_proposals,
        outer_windows,
        [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
        config["verifier"],
    )
    _assert_proposal_feature_alignment(
        features=outer_features, frame=outer_proposals, context="Outer"
    )
    outer_seed_predictions: list[tuple[np.ndarray, np.ndarray]] = []
    final_verifier_parent_sha256 = {
        "oof_window_predictions": sha256_file(run.root / "oof" / "window_predictions.parquet"),
        "oof_proposals_labeled": sha256_file(run.root / "oof" / "proposals_labeled.parquet"),
    }
    for seed in config["verifier"]["seeds"]:
        seed = int(seed)
        final_epochs = int(np.median(selected_verifier_epochs[seed]))
        model = _train_verifier_model_v4(
            global_features,
            global_categories,
            config,
            seed=seed + 90_000,
            epochs=final_epochs,
        )
        checkpoint = run.root / "verifier" / f"final_seed_{seed}.pt"
        _save_torch_atomic(
            checkpoint,
            {
                "model": model.state_dict(),
                "sequence_dim": global_features.sequence.shape[-1],
                "scalar_dim": global_features.scalar.shape[-1],
                "config": config["verifier"],
                "training_subjects": sorted(outer_train_subjects),
                "prediction_subjects": sorted(set(run.payload["outer_test_subjects"])),
                "globally_excluded_subjects": sorted(set(run.payload["outer_test_subjects"])),
                "epochs": final_epochs,
                "parent_artifact_sha256": final_verifier_parent_sha256,
                "state_calibration_source": "outer_train_subject_crossfit_oof",
                "duration_prior_source": "outer_train_truth",
            },
        )
        artifacts.append(checkpoint)
        if len(outer_proposals):
            outer_seed_predictions.append(_infer_verifier_model(model, outer_features, config))
    outer_scored = outer_proposals.copy().reset_index(drop=True)
    if len(outer_scored):
        outer_scored["event_logit"] = np.mean(
            [value[0] for value in outer_seed_predictions], axis=0
        )
        outer_scored["iou_logit"] = np.mean(
            [value[1] for value in outer_seed_predictions], axis=0
        )
        outer_scored["predicted_iou"] = 1.0 / (1.0 + np.exp(-outer_scored["iou_logit"]))
        outer_scored["state_score"] = outer_features.scalar[:, 2]
        outer_scored["logistic_score"] = logistic_final.predict(
            _verifier_matrix(outer_features)
        )
        outer_scored = final_calibration.apply(outer_scored)
    else:
        for column in (
            "event_logit",
            "iou_logit",
            "predicted_iou",
            "state_score",
            "logistic_score",
            "calibrated_event_probability",
            "calibrated_iou",
            "final_score",
        ):
            outer_scored[column] = pd.Series(dtype=float)
    evaluable, ignored = partition_evaluation_events(inputs.events, outer_train_subjects)
    deep_point = _best_verifier_operating_point(
        scored, evaluable, global_windows, "final_score", config, ignored
    )
    logistic_point = _best_verifier_operating_point(
        scored, evaluable, global_windows, "logistic_score", config, ignored
    )
    matching_sensitivity: dict[str, dict[str, float]] = {}
    for name, point, score_column in (
        ("deep", deep_point, "final_score"),
        ("logistic", logistic_point, "logistic_score"),
    ):
        accepted = _accepted_from_point(scored, point, score_column)
        greedy, _ = evaluate_events(
            evaluable, _prediction_events(accepted), method="greedy", ignore=ignored
        )
        matching_sensitivity[name] = {
            "max_cardinality_f1": float(point["f1"]),
            "greedy_f1": float(greedy["f1"]),
        }
    if _matching_ranking_reversal(
        matching_sensitivity["deep"]["max_cardinality_f1"]
        - matching_sensitivity["logistic"]["max_cardinality_f1"],
        matching_sensitivity["deep"]["greedy_f1"]
        - matching_sensitivity["logistic"]["greedy_f1"],
        tolerance=float(config["calibration"].get("matching_reversal_tolerance", 0.005)),
    ):
        raise RuntimeError("Verifier ranking reverses between matching methods")
    deep_passed = deep_point["f1"] >= logistic_point["f1"] + float(
        config["promotion_gate"]["minimum_verifier_f1_improvement"]
    ) or (
        deep_point["f1"]
        >= logistic_point["f1"]
        - float(config["promotion_gate"]["maximum_verifier_f1_drop"])
        and deep_point["fp_per_hour"]
        <= logistic_point["fp_per_hour"]
        * (1.0 - float(config["promotion_gate"]["minimum_verifier_fp_reduction"]))
    )
    preselection = {
        "protocol_version": "statsfusion-r2",
        "meta_crossfit_protocol": "fully_nested_v1",
        "verifier_kind": "deep" if deep_passed else "logistic",
        "deep": deep_point,
        "logistic": logistic_point,
        "deep_gate_passed": bool(deep_passed),
        "matching_sensitivity": matching_sensitivity,
    }
    if not deep_passed:
        scored["final_score"] = scored["logistic_score"]
        outer_scored["final_score"] = outer_scored["logistic_score"]
    score_path = run.root / "oof" / "proposal_scores.parquet"
    outer_score_path = run.root / "outer" / "proposal_scores.parquet"
    calibration_path = run.root / "verifier" / "proposal_calibration.json"
    logistic_path = run.root / "verifier" / "logistic_verifier.json"
    preselection_path = run.root / "verifier" / "preselection.json"
    write_parquet_atomic(score_path, scored)
    write_parquet_atomic(outer_score_path, outer_scored)
    write_json_atomic(calibration_path, final_calibration.to_json())
    write_json_atomic(logistic_path, logistic_final.to_json())
    write_json_atomic(preselection_path, preselection)
    artifacts.extend(
        (score_path, outer_score_path, calibration_path, logistic_path, preselection_path)
    )
    run.transition("VERIFIER_COMPLETE", artifacts)


def _train_verifier_crossfit_v4_r1_blocked(
    run: HierarchicalRun,
    config: dict[str, Any],
    inputs: V4Inputs,
) -> None:
    run.require_stage("PROPOSALS_COMPLETE")
    proposals = pd.read_parquet(run.root / "oof" / "proposals_labeled.parquet")
    windows = pd.read_parquet(run.root / "oof" / "window_predictions.parquet")
    features = build_proposal_features_v4(
        proposals,
        windows,
        [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
        config["verifier"],
    )
    _assert_proposal_feature_alignment(features, proposals, context="Verifier crossfit")
    categories = classify_proposals(proposals)
    partitions = run.payload["stacking_partitions"]
    proposal_partition = proposals["subject_key"].astype(str).map(partitions).to_numpy(dtype=int)
    deep_event = np.zeros(len(proposals), dtype=np.float64)
    deep_iou = np.zeros(len(proposals), dtype=np.float64)
    logistic_score = np.zeros(len(proposals), dtype=np.float64)
    artifacts: list[Path] = []
    selected_verifier_epochs: dict[int, list[int]] = {
        int(seed): [] for seed in config["verifier"]["seeds"]
    }
    for partition in sorted(set(proposal_partition)):
        holdout_indices = np.flatnonzero(proposal_partition == partition)
        train_indices = np.flatnonzero(proposal_partition != partition)
        train_subjects = set(proposals.iloc[train_indices]["subject_key"].astype(str))
        holdout_subjects = set(proposals.iloc[holdout_indices]["subject_key"].astype(str))
        assert_disjoint_subjects(verifier_train=train_subjects, verifier_holdout=holdout_subjects)
        train_features = _slice_proposal_features(features, train_indices)
        holdout_features = _slice_proposal_features(features, holdout_indices)
        fit_subjects, selector_subjects, selector_split_report = _selector_split(
            train_subjects,
            float(config["training"]["selector_fraction"]),
            int(config["training"]["random_seed"]) + partition + 60_000,
            events=inputs.events,
            anchors=inputs.anchors,
        )
        fit_local = np.flatnonzero(
            proposals.iloc[train_indices]["subject_key"].astype(str).isin(fit_subjects).to_numpy()
        )
        selector_local = np.flatnonzero(
            proposals.iloc[train_indices]["subject_key"]
            .astype(str)
            .isin(selector_subjects)
            .to_numpy()
        )
        fit_features = _slice_proposal_features(train_features, fit_local)
        selector_features = _slice_proposal_features(train_features, selector_local)
        selector_proposals = (
            proposals.iloc[train_indices].iloc[selector_local].reset_index(drop=True)
        )
        selector_truth, selector_ignore = partition_evaluation_events(
            inputs.events, selector_subjects
        )
        selector_windows = windows[windows["subject_key"].astype(str).isin(selector_subjects)]
        seed_predictions: list[tuple[np.ndarray, np.ndarray]] = []
        for seed in config["verifier"]["seeds"]:
            seed = int(seed)
            selected_epoch, selection_history = _select_verifier_epoch(
                fit_features,
                categories[train_indices][fit_local],
                selector_features,
                selector_proposals,
                selector_truth,
                selector_ignore,
                selector_windows,
                config,
                seed=seed + int(partition) * 100,
            )
            model = _train_verifier_model_v4(
                train_features,
                categories[train_indices],
                config,
                seed=seed + int(partition) * 100,
                epochs=selected_epoch,
            )
            checkpoint = run.root / "verifier" / f"partition_{partition}_seed_{seed}.pt"
            _save_torch_atomic(
                checkpoint,
                {
                    "model": model.state_dict(),
                    "sequence_dim": features.sequence.shape[-1],
                    "scalar_dim": features.scalar.shape[-1],
                    "config": config["verifier"],
                    "training_subjects": sorted(train_subjects),
                    "holdout_subjects": sorted(holdout_subjects),
                    "fit_subjects": sorted(fit_subjects),
                    "selector_subjects": sorted(selector_subjects),
                    "epochs": selected_epoch,
                },
            )
            selector_path = (
                run.root / "verifier" / f"partition_{partition}_seed_{seed}_selector.json"
            )
            write_json_atomic(
                selector_path,
                {
                    "selected_epoch": selected_epoch,
                    "history": selection_history,
                    "fit_subjects": sorted(fit_subjects),
                    "selector_subjects": sorted(selector_subjects),
                    "selector_split": selector_split_report,
                },
            )
            artifacts.append(selector_path)
            selected_verifier_epochs[seed].append(selected_epoch)
            artifacts.append(checkpoint)
            seed_predictions.append(_infer_verifier_model(model, holdout_features, config))
        deep_event[holdout_indices] = np.mean([value[0] for value in seed_predictions], axis=0)
        deep_iou[holdout_indices] = np.mean([value[1] for value in seed_predictions], axis=0)
        logistic = LogisticScoreCombiner.fit(
            _verifier_matrix(train_features),
            train_features.event_target,
            sample_weight=train_features.sample_weight,
        )
        logistic_score[holdout_indices] = logistic.predict(_verifier_matrix(holdout_features))
    scored = proposals.copy().reset_index(drop=True)
    scored["stacking_partition"] = proposal_partition
    scored["category"] = categories
    scored["sample_weight"] = features.sample_weight
    scored["event_logit"] = deep_event
    scored["iou_logit"] = deep_iou
    scored["predicted_iou"] = 1.0 / (1.0 + np.exp(-deep_iou))
    scored["state_score"] = features.scalar[:, 2]
    scored["logistic_score"] = logistic_score
    calibrated_parts: list[pd.DataFrame] = []
    for partition in sorted(set(proposal_partition)):
        holdout = scored["stacking_partition"] == partition
        calibration = ProposalCalibrationV4.fit(scored.loc[~holdout])
        calibrated_parts.append(calibration.apply(scored.loc[holdout]))
    scored = pd.concat(calibrated_parts, ignore_index=True).sort_values("proposal_id")
    final_calibration = ProposalCalibrationV4.fit(scored)
    logistic_final = LogisticScoreCombiner.fit(
        _verifier_matrix(features),
        features.event_target,
        sample_weight=features.sample_weight,
    )
    outer_proposals = pd.read_parquet(run.root / "outer" / "proposals.parquet")
    outer_windows = pd.read_parquet(run.root / "outer" / "window_predictions.parquet")
    outer_features = build_proposal_features_v4(
        outer_proposals,
        outer_windows,
        [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
        config["verifier"],
    )
    _assert_proposal_feature_alignment(features=outer_features, frame=outer_proposals, context="Outer")
    outer_seed_predictions: list[tuple[np.ndarray, np.ndarray]] = []
    for seed in config["verifier"]["seeds"]:
        seed = int(seed)
        final_epochs = int(np.median(selected_verifier_epochs[seed]))
        model = _train_verifier_model_v4(
            features,
            categories,
            config,
            seed=seed + 90_000,
            epochs=final_epochs,
        )
        checkpoint = run.root / "verifier" / f"final_seed_{seed}.pt"
        _save_torch_atomic(
            checkpoint,
            {
                "model": model.state_dict(),
                "sequence_dim": features.sequence.shape[-1],
                "scalar_dim": features.scalar.shape[-1],
                "config": config["verifier"],
                "training_subjects": sorted(set(proposals["subject_key"].astype(str))),
                "epochs": final_epochs,
            },
        )
        artifacts.append(checkpoint)
        if len(outer_proposals):
            outer_seed_predictions.append(_infer_verifier_model(model, outer_features, config))
    outer_scored = outer_proposals.copy().reset_index(drop=True)
    if len(outer_scored):
        outer_scored["event_logit"] = np.mean(
            [value[0] for value in outer_seed_predictions], axis=0
        )
        outer_scored["iou_logit"] = np.mean([value[1] for value in outer_seed_predictions], axis=0)
        outer_scored["predicted_iou"] = 1.0 / (1.0 + np.exp(-outer_scored["iou_logit"]))
        outer_scored["state_score"] = outer_features.scalar[:, 2]
        outer_scored["logistic_score"] = logistic_final.predict(_verifier_matrix(outer_features))
        outer_scored = final_calibration.apply(outer_scored)
    else:
        for column in (
            "event_logit",
            "iou_logit",
            "predicted_iou",
            "state_score",
            "logistic_score",
            "calibrated_event_probability",
            "calibrated_iou",
            "final_score",
        ):
            outer_scored[column] = pd.Series(dtype=float)
    training_subjects = set(proposals["subject_key"].astype(str))
    evaluable, ignored = partition_evaluation_events(inputs.events, training_subjects)
    deep_point = _best_verifier_operating_point(
        scored, evaluable, windows, "final_score", config, ignored
    )
    logistic_point = _best_verifier_operating_point(
        scored, evaluable, windows, "logistic_score", config, ignored
    )
    matching_sensitivity: dict[str, dict[str, float]] = {}
    for name, point, score_column in (
        ("deep", deep_point, "final_score"),
        ("logistic", logistic_point, "logistic_score"),
    ):
        accepted = _accepted_from_point(scored, point, score_column)
        greedy, _ = evaluate_events(
            evaluable, _prediction_events(accepted), method="greedy", ignore=ignored
        )
        matching_sensitivity[name] = {
            "max_cardinality_f1": float(point["f1"]),
            "greedy_f1": float(greedy["f1"]),
        }
    if _matching_ranking_reversal(
        matching_sensitivity["deep"]["max_cardinality_f1"]
        - matching_sensitivity["logistic"]["max_cardinality_f1"],
        matching_sensitivity["deep"]["greedy_f1"] - matching_sensitivity["logistic"]["greedy_f1"],
        tolerance=float(config["calibration"].get("matching_reversal_tolerance", 0.005)),
    ):
        raise RuntimeError("Verifier ranking reverses between max-cardinality and greedy matching")
    deep_passed = deep_point["f1"] >= logistic_point["f1"] + float(
        config["promotion_gate"]["minimum_verifier_f1_improvement"]
    ) or (
        deep_point["f1"]
        >= logistic_point["f1"] - float(config["promotion_gate"]["maximum_verifier_f1_drop"])
        and deep_point["fp_per_hour"]
        <= logistic_point["fp_per_hour"]
        * (1.0 - float(config["promotion_gate"]["minimum_verifier_fp_reduction"]))
    )
    preselection = {
        "verifier_kind": "deep" if deep_passed else "logistic",
        "deep": deep_point,
        "logistic": logistic_point,
        "deep_gate_passed": bool(deep_passed),
        "matching_sensitivity": matching_sensitivity,
    }
    if not deep_passed:
        scored["final_score"] = scored["logistic_score"]
        outer_scored["final_score"] = outer_scored["logistic_score"]
    score_path = run.root / "oof" / "proposal_scores.parquet"
    outer_score_path = run.root / "outer" / "proposal_scores.parquet"
    calibration_path = run.root / "verifier" / "proposal_calibration.json"
    logistic_path = run.root / "verifier" / "logistic_verifier.json"
    preselection_path = run.root / "verifier" / "preselection.json"
    write_parquet_atomic(score_path, scored)
    write_parquet_atomic(outer_score_path, outer_scored)
    write_json_atomic(calibration_path, final_calibration.to_json())
    write_json_atomic(logistic_path, logistic_final.to_json())
    write_json_atomic(preselection_path, preselection)
    artifacts.extend(
        (score_path, outer_score_path, calibration_path, logistic_path, preselection_path)
    )
    run.transition("VERIFIER_COMPLETE", artifacts)


def _accepted_from_point(
    scores: pd.DataFrame,
    point: dict[str, Any],
    score_column: str,
    minimum_gap_seconds: int = 3,
) -> pd.DataFrame:
    selected = scores[scores[score_column] >= float(point["acceptance_threshold"])]
    if selected.empty:
        return selected.copy()
    return pd.concat(
        [
            _gap_aware_nms(
                group,
                float(point["nms_iou_threshold"]),
                score_column,
                minimum_gap_seconds,
            )
            for _, group in selected.groupby(["subject_key", "session_id"], sort=False)
        ],
        ignore_index=True,
    )


def _train_endpoint_model(
    features,
    config: dict[str, Any],
    *,
    seed: int,
    epochs: int | None = None,
) -> EndpointRefiner:
    torch.manual_seed(seed)
    device = _device(config)
    model = EndpointRefiner(features.start_sequence.shape[-1], config["boundary"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["boundary"]["learning_rate"]),
        weight_decay=float(config["boundary"]["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    batch_size = int(config["boundary"]["batch_size"])
    maximum_epochs = int(epochs or config["boundary"]["max_epochs"])
    progress = tqdm(
        range(maximum_epochs),
        desc=f"boundary seed={seed}",
        unit="epoch",
        dynamic_ncols=True,
    )
    for _ in progress:
        order = rng.permutation(len(features.sample_ids))
        model.train()
        epoch_loss_total = torch.zeros((), device=device)
        epoch_steps = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch = {
                "start_sequence": torch.from_numpy(features.start_sequence[indices]).to(device),
                "end_sequence": torch.from_numpy(features.end_sequence[indices]).to(device),
                "start_mask": torch.from_numpy(features.start_mask[indices]).to(device),
                "end_mask": torch.from_numpy(features.end_mask[indices]).to(device),
                "start_target": torch.from_numpy(features.start_target[indices]).to(device),
                "end_target": torch.from_numpy(features.end_target[indices]).to(device),
                "sample_weight": torch.from_numpy(features.sample_weight[indices]).to(device),
                "start_weight": torch.from_numpy(features.start_weight[indices]).to(device),
                "end_weight": torch.from_numpy(features.end_weight[indices]).to(device),
            }
            output = model(batch)
            loss, _ = endpoint_loss(output, batch)
            epoch_loss_total += loss.detach()
            epoch_steps += 1
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        progress.set_postfix(loss=f"{float(epoch_loss_total.cpu()) / max(epoch_steps, 1):.4f}")
    return model


def _select_boundary_epoch(
    fit_features,
    selector_features,
    config: dict[str, Any],
    *,
    seed: int,
) -> tuple[int, list[dict[str, float]]]:
    if not len(fit_features.sample_ids) or not len(selector_features.sample_ids):
        raise ValueError("Boundary selector requires non-empty fit and selector samples")
    torch.manual_seed(seed)
    device = _device(config)
    model = EndpointRefiner(fit_features.start_sequence.shape[-1], config["boundary"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["boundary"]["learning_rate"]),
        weight_decay=float(config["boundary"]["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    batch_size = int(config["boundary"]["batch_size"])
    patience_limit = int(config["boundary"]["patience"])
    best_epoch = 1
    best_mae = math.inf
    patience = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(config["boundary"]["max_epochs"]) + 1):
        order = rng.permutation(len(fit_features.sample_ids))
        model.train()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch = {
                "start_sequence": torch.from_numpy(fit_features.start_sequence[indices]).to(device),
                "end_sequence": torch.from_numpy(fit_features.end_sequence[indices]).to(device),
                "start_mask": torch.from_numpy(fit_features.start_mask[indices]).to(device),
                "end_mask": torch.from_numpy(fit_features.end_mask[indices]).to(device),
                "start_target": torch.from_numpy(fit_features.start_target[indices]).to(device),
                "end_target": torch.from_numpy(fit_features.end_target[indices]).to(device),
                "sample_weight": torch.from_numpy(fit_features.sample_weight[indices]).to(device),
                "start_weight": torch.from_numpy(fit_features.start_weight[indices]).to(device),
                "end_weight": torch.from_numpy(fit_features.end_weight[indices]).to(device),
            }
            output = model(batch)
            loss, _ = endpoint_loss(output, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        predicted = _infer_endpoint_model(model, selector_features, config)
        errors: list[np.ndarray] = []
        for predicted_offset, target, endpoint_weight, grid in (
            (
                predicted[0],
                selector_features.start_target,
                selector_features.start_weight,
                selector_features.start_offsets_seconds,
            ),
            (
                predicted[1],
                selector_features.end_target,
                selector_features.end_weight,
                selector_features.end_offsets_seconds,
            ),
        ):
            valid = endpoint_weight > 0
            target_offset = (target * grid[None, :]).sum(axis=1)
            errors.append(np.abs(predicted_offset[valid] - target_offset[valid]))
        finite_errors = np.concatenate([value[np.isfinite(value)] for value in errors])
        mae = float(finite_errors.mean()) if len(finite_errors) else math.inf
        history.append({"epoch": float(epoch), "endpoint_mae_seconds": mae})
        if np.isfinite(mae) and mae < best_mae - 1e-9:
            best_mae = mae
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= patience_limit:
            break
    if not np.isfinite(best_mae):
        raise RuntimeError("Boundary selector never produced a finite endpoint MAE")
    return best_epoch, history


@torch.no_grad()
def _infer_endpoint_model(model, features, config: dict[str, Any]):
    device = _device(config)
    model.to(device).eval()
    batch_size = int(config["boundary"]["batch_size"])
    start_offsets: list[np.ndarray] = []
    end_offsets: list[np.ndarray] = []
    start_entropies: list[np.ndarray] = []
    end_entropies: list[np.ndarray] = []
    start_grid = torch.from_numpy(features.start_offsets_seconds).to(device)
    end_grid = torch.from_numpy(features.end_offsets_seconds).to(device)
    for first in tqdm(
        range(0, len(features.sample_ids), batch_size),
        desc="boundary inference",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    ):
        selected = slice(first, first + batch_size)
        batch = {
            "start_sequence": torch.from_numpy(features.start_sequence[selected]).to(device),
            "end_sequence": torch.from_numpy(features.end_sequence[selected]).to(device),
            "start_mask": torch.from_numpy(features.start_mask[selected]).to(device),
            "end_mask": torch.from_numpy(features.end_mask[selected]).to(device),
        }
        output = model(batch)
        start_offset, start_entropy = local_soft_argmax(
            output["start_logit"],
            start_grid,
            int(config["boundary"]["local_softargmax_radius_bins"]),
            batch["start_mask"],
        )
        end_offset, end_entropy = local_soft_argmax(
            output["end_logit"],
            end_grid,
            int(config["boundary"]["local_softargmax_radius_bins"]),
            batch["end_mask"],
        )
        start_offsets.append(start_offset.cpu().numpy())
        end_offsets.append(end_offset.cpu().numpy())
        start_entropies.append(start_entropy.cpu().numpy())
        end_entropies.append(end_entropy.cpu().numpy())
    return tuple(
        np.concatenate(values) if values else np.empty(0, dtype=np.float32)
        for values in (start_offsets, end_offsets, start_entropies, end_entropies)
    )


def _attach_truth_boundaries(proposals: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    truth = events[["event_id", "start_ms", "end_ms"]].rename(
        columns={
            "event_id": "matched_event_id",
            "start_ms": "truth_start_ms",
            "end_ms": "truth_end_ms",
        }
    )
    output = proposals.merge(truth, on="matched_event_id", how="left", validate="many_to_one")
    return output


def train_boundary_crossfit_v4(
    run: HierarchicalRun,
    config: dict[str, Any],
    inputs: V4Inputs,
) -> None:
    run.require_stage("VERIFIER_COMPLETE")
    scores = pd.read_parquet(run.root / "oof" / "proposal_scores.parquet")
    windows = pd.read_parquet(run.root / "oof" / "window_predictions.parquet")
    preselection = json.loads(
        (run.root / "verifier" / "preselection.json").read_text(encoding="utf-8")
    )
    verifier_kind = str(preselection["verifier_kind"])
    score_column = "final_score" if verifier_kind == "deep" else "logistic_score"
    point = preselection[verifier_kind]
    accepted = _accepted_from_point(scores, point, score_column)
    positive = scores[scores["max_iou"] > 0.25].copy()
    positive = _attach_truth_boundaries(positive, inputs.events)
    partitions = run.payload["stacking_partitions"]
    minimum_events = int(config["boundary"]["minimum_independent_events"])
    artifacts: list[Path] = []
    outputs: list[pd.DataFrame] = []
    disabled_reason: str | None = None
    selected_boundary_epochs: dict[int, list[int]] = {
        int(seed): [] for seed in config["boundary"]["seeds"]
    }
    for partition in sorted(set(partitions.values())):
        holdout_subjects = {
            subject for subject, value in partitions.items() if int(value) == int(partition)
        }
        nested_root = run.root / "nested" / f"meta_{partition}"
        lineage_path = nested_root / "lineage.json"
        if not lineage_path.is_file():
            raise FileNotFoundError(f"Nested meta cache is missing for partition {partition}")
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        if set(lineage.get("globally_excluded_subjects", [])) != holdout_subjects:
            raise RuntimeError("Boundary nested cache exclusion does not match its holdout")
        nested_train = pd.read_parquet(nested_root / "train_proposals_labeled.parquet")
        train = nested_train[nested_train["max_iou"] > 0.25].copy()
        train = _attach_truth_boundaries(train, inputs.events)
        holdout = accepted[accepted["subject_key"].astype(str).map(partitions) == partition].copy()
        train_windows = pd.read_parquet(nested_root / "train_window_predictions.parquet")
        holdout_windows = pd.read_parquet(nested_root / "holdout_window_predictions.parquet")
        independent_events = train["matched_event_id"].nunique()
        if independent_events < minimum_events:
            disabled_reason = (
                f"partition {partition} has {independent_events} independent events; "
                f"requires {minimum_events}"
            )
            break
        start_residual = (
            train["truth_start_ms"].to_numpy(dtype=float)
            - train["coarse_start_ms"].to_numpy(dtype=float)
        ) / 1000.0
        end_residual = (
            train["truth_end_ms"].to_numpy(dtype=float)
            - train["coarse_end_ms"].to_numpy(dtype=float)
        ) / 1000.0
        boundary_range = select_boundary_range(
            start_residual,
            end_residual,
            quantile=float(config["boundary"]["residual_quantile"]),
            minimum_seconds=int(config["boundary"]["minimum_range_seconds"]),
            maximum_seconds=int(config["boundary"]["maximum_range_seconds"]),
        )
        if boundary_range.clipped_fraction > float(config["boundary"]["maximum_clipped_fraction"]):
            raise RuntimeError("Boundary residual clipping exceeds 5%; repair candidates first")
        train_subjects = set(train["subject_key"].astype(str))
        fit_subjects, selector_subjects, selector_split_report = _selector_split(
            train_subjects,
            float(config["training"]["selector_fraction"]),
            int(config["training"]["random_seed"]) + partition + 70_000,
            events=inputs.events,
            anchors=inputs.anchors,
        )
        fit_positive = train[train["subject_key"].astype(str).isin(fit_subjects)]
        selector_positive = train[train["subject_key"].astype(str).isin(selector_subjects)]
        selection_start_residual = (
            fit_positive["truth_start_ms"].to_numpy(dtype=float)
            - fit_positive["coarse_start_ms"].to_numpy(dtype=float)
        ) / 1000.0
        selection_end_residual = (
            fit_positive["truth_end_ms"].to_numpy(dtype=float)
            - fit_positive["coarse_end_ms"].to_numpy(dtype=float)
        ) / 1000.0
        selection_range = select_boundary_range(
            selection_start_residual,
            selection_end_residual,
            quantile=float(config["boundary"]["residual_quantile"]),
            minimum_seconds=int(config["boundary"]["minimum_range_seconds"]),
            maximum_seconds=int(config["boundary"]["maximum_range_seconds"]),
        )
        fit_features = build_endpoint_features(
            augment_boundary_training_proposals(
                fit_positive,
                maximum_jitters_per_event=int(config["boundary"]["maximum_jitters_per_event"]),
                jitter_seconds=int(config["boundary"]["jitter_seconds"]),
            ),
            train_windows,
            [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
            selection_range,
            config["boundary"],
        )
        selector_features = build_endpoint_features(
            augment_boundary_training_proposals(
                selector_positive,
                maximum_jitters_per_event=int(config["boundary"]["maximum_jitters_per_event"]),
                jitter_seconds=int(config["boundary"]["jitter_seconds"]),
            ),
            train_windows,
            [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
            selection_range,
            config["boundary"],
        )
        augmented = augment_boundary_training_proposals(
            train,
            maximum_jitters_per_event=int(config["boundary"]["maximum_jitters_per_event"]),
            jitter_seconds=int(config["boundary"]["jitter_seconds"]),
        )
        train_features = build_endpoint_features(
            augmented,
            train_windows,
            [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
            boundary_range,
            config["boundary"],
        )
        holdout_features = build_endpoint_features(
            holdout,
            holdout_windows,
            [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
            boundary_range,
            config["boundary"],
        )
        boundary_parent_paths = {
            "nested_train_windows": nested_root / "train_window_predictions.parquet",
            "nested_train_proposals": nested_root / "train_proposals_labeled.parquet",
            "nested_holdout_windows": nested_root / "holdout_window_predictions.parquet",
            "oof_proposal_scores": run.root / "oof" / "proposal_scores.parquet",
            "verifier_preselection": run.root / "verifier" / "preselection.json",
        }
        boundary_parent_sha256 = {
            name: sha256_file(path) for name, path in boundary_parent_paths.items()
        }
        seed_outputs: dict[int, tuple[np.ndarray, ...]] = {}
        for seed in config["boundary"]["seeds"]:
            seed = int(seed)
            selected_epoch, selection_history = _select_boundary_epoch(
                fit_features,
                selector_features,
                config,
                seed=seed + partition * 100,
            )
            model = _train_endpoint_model(
                train_features,
                config,
                seed=seed + partition * 100,
                epochs=selected_epoch,
            )
            checkpoint = run.root / "boundary" / f"partition_{partition}_seed_{seed}.pt"
            _save_torch_atomic(
                checkpoint,
                {
                    "model": model.state_dict(),
                    "input_dim": train_features.start_sequence.shape[-1],
                    "config": config["boundary"],
                    "range": boundary_range.__dict__,
                    "training_subjects": sorted(set(train["subject_key"].astype(str))),
                    "prediction_subjects": sorted(holdout_subjects),
                    "globally_excluded_subjects": sorted(holdout_subjects),
                    "fit_subjects": sorted(fit_subjects),
                    "selector_subjects": sorted(selector_subjects),
                    "epochs": selected_epoch,
                    "parent_artifact_sha256": boundary_parent_sha256,
                    "state_calibration_source": "nested_meta_train_oof",
                    "duration_prior_source": "nested_meta_training_truth",
                },
            )
            selector_path = (
                run.root / "boundary" / f"partition_{partition}_seed_{seed}_selector.json"
            )
            write_json_atomic(
                selector_path,
                {
                    "selected_epoch": selected_epoch,
                    "history": selection_history,
                    "training_subjects": sorted(set(train["subject_key"].astype(str))),
                    "prediction_subjects": sorted(holdout_subjects),
                    "globally_excluded_subjects": sorted(holdout_subjects),
                    "fit_subjects": sorted(fit_subjects),
                    "selector_subjects": sorted(selector_subjects),
                    "selector_split": selector_split_report,
                    "parent_artifact_sha256": boundary_parent_sha256,
                    "state_calibration_source": "nested_meta_train_oof",
                    "duration_prior_source": "nested_meta_training_truth",
                },
            )
            artifacts.append(selector_path)
            selected_boundary_epochs[seed].append(selected_epoch)
            artifacts.append(checkpoint)
            seed_outputs[seed] = _infer_endpoint_model(model, holdout_features, config)
        holdout_output = holdout[["proposal_id"]].copy()
        for index, name in enumerate(
            ("start_offset_seconds", "end_offset_seconds", "start_entropy", "end_entropy")
        ):
            holdout_output[name] = np.mean(
                [value[index] for value in seed_outputs.values()], axis=0
            )
            for seed, value in seed_outputs.items():
                holdout_output[f"{name}_seed_{seed}"] = value[index]
        outputs.append(holdout_output)
        range_path = run.root / "boundary" / f"partition_{partition}_range.json"
        write_json_atomic(range_path, boundary_range.__dict__)
        artifacts.append(range_path)
    status_path = run.root / "boundary" / "status.json"
    if disabled_reason is not None:
        write_json_atomic(status_path, {"enabled": False, "reason": disabled_reason})
        empty_path = run.root / "oof" / "boundary_scores.parquet"
        write_parquet_atomic(
            empty_path,
            pd.DataFrame(
                columns=[
                    "proposal_id",
                    "start_offset_seconds",
                    "end_offset_seconds",
                    "start_entropy",
                    "end_entropy",
                ]
            ),
        )
        run.transition("BOUNDARY_COMPLETE", [status_path, empty_path])
        return
    boundary_scores = pd.concat(outputs, ignore_index=True)
    boundary_score_path = run.root / "oof" / "boundary_scores.parquet"
    write_parquet_atomic(boundary_score_path, boundary_scores)
    write_json_atomic(status_path, {"enabled": True})
    artifacts.extend((boundary_score_path, status_path))

    final_start_residual = (
        positive["truth_start_ms"].to_numpy(dtype=float)
        - positive["coarse_start_ms"].to_numpy(dtype=float)
    ) / 1000.0
    final_end_residual = (
        positive["truth_end_ms"].to_numpy(dtype=float)
        - positive["coarse_end_ms"].to_numpy(dtype=float)
    ) / 1000.0
    final_range = select_boundary_range(
        final_start_residual,
        final_end_residual,
        quantile=float(config["boundary"]["residual_quantile"]),
        minimum_seconds=int(config["boundary"]["minimum_range_seconds"]),
        maximum_seconds=int(config["boundary"]["maximum_range_seconds"]),
    )
    if final_range.clipped_fraction > float(config["boundary"]["maximum_clipped_fraction"]):
        raise RuntimeError("Final boundary residual clipping exceeds 5%; repair candidates first")
    final_augmented = augment_boundary_training_proposals(
        positive,
        maximum_jitters_per_event=int(config["boundary"]["maximum_jitters_per_event"]),
        jitter_seconds=int(config["boundary"]["jitter_seconds"]),
    )
    final_features = build_endpoint_features(
        final_augmented,
        windows,
        [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
        final_range,
        config["boundary"],
    )
    outer_scores = pd.read_parquet(run.root / "outer" / "proposal_scores.parquet")
    outer_accepted = _accepted_from_point(outer_scores, point, score_column)
    outer_windows = pd.read_parquet(run.root / "outer" / "window_predictions.parquet")
    outer_features = build_endpoint_features(
        outer_accepted,
        outer_windows,
        [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
        final_range,
        config["boundary"],
    )
    outer_seed_outputs: dict[int, tuple[np.ndarray, ...]] = {}
    final_boundary_parent_sha256 = {
        "oof_window_predictions": sha256_file(run.root / "oof" / "window_predictions.parquet"),
        "oof_proposal_scores": sha256_file(run.root / "oof" / "proposal_scores.parquet"),
        "verifier_preselection": sha256_file(run.root / "verifier" / "preselection.json"),
    }
    for seed in config["boundary"]["seeds"]:
        seed = int(seed)
        final_epochs = int(np.median(selected_boundary_epochs[seed]))
        model = _train_endpoint_model(
            final_features, config, seed=seed + 90_000, epochs=final_epochs
        )
        checkpoint = run.root / "boundary" / f"final_seed_{seed}.pt"
        _save_torch_atomic(
            checkpoint,
            {
                "model": model.state_dict(),
                "input_dim": final_features.start_sequence.shape[-1],
                "config": config["boundary"],
                "range": final_range.__dict__,
                "epochs": final_epochs,
                "training_subjects": sorted(set(positive["subject_key"].astype(str))),
                "prediction_subjects": sorted(set(run.payload["outer_test_subjects"])),
                "globally_excluded_subjects": sorted(set(run.payload["outer_test_subjects"])),
                "parent_artifact_sha256": final_boundary_parent_sha256,
                "state_calibration_source": "outer_train_subject_crossfit_oof",
                "duration_prior_source": "outer_train_truth",
            },
        )
        artifacts.append(checkpoint)
        if len(outer_accepted):
            outer_seed_outputs[seed] = _infer_endpoint_model(model, outer_features, config)
    outer_boundary = outer_accepted[["proposal_id"]].copy()
    for index, name in enumerate(
        ("start_offset_seconds", "end_offset_seconds", "start_entropy", "end_entropy")
    ):
        outer_boundary[name] = (
            np.mean([value[index] for value in outer_seed_outputs.values()], axis=0)
            if outer_seed_outputs
            else np.empty(0, dtype=float)
        )
        for seed in config["boundary"]["seeds"]:
            seed = int(seed)
            outer_boundary[f"{name}_seed_{seed}"] = (
                outer_seed_outputs[seed][index]
                if seed in outer_seed_outputs
                else np.empty(0, dtype=float)
            )
    outer_boundary_path = run.root / "outer" / "boundary_scores.parquet"
    final_range_path = run.root / "boundary" / "final_range.json"
    write_parquet_atomic(outer_boundary_path, outer_boundary)
    write_json_atomic(final_range_path, final_range.__dict__)
    artifacts.extend((outer_boundary_path, final_range_path))
    run.transition("BOUNDARY_COMPLETE", artifacts)


def _refine_from_scores(
    accepted: pd.DataFrame,
    boundary_scores: pd.DataFrame,
    entropy_threshold: float,
    safety_gap_seconds: int,
) -> pd.DataFrame:
    merged = accepted.merge(boundary_scores, on="proposal_id", how="left", validate="one_to_one")
    if merged[["start_offset_seconds", "end_offset_seconds"]].isna().any().any():
        raise RuntimeError("Accepted proposal is missing crossfit boundary output")
    return apply_boundary_refinement(
        merged,
        merged["start_offset_seconds"].to_numpy(),
        merged["end_offset_seconds"].to_numpy(),
        merged["start_entropy"].to_numpy(),
        merged["end_entropy"].to_numpy(),
        entropy_threshold=entropy_threshold,
        safety_gap_seconds=safety_gap_seconds,
    )


def _seeded_boundary_scores(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    output = frame.copy()
    for name in (
        "start_offset_seconds",
        "end_offset_seconds",
        "start_entropy",
        "end_entropy",
    ):
        column = f"{name}_seed_{int(seed)}"
        if column not in output:
            raise ValueError(f"Boundary scores are missing seed output: {column}")
        output[name] = output[column]
    return output


def _hand_metrics(
    events: pd.DataFrame,
    predictions: pd.DataFrame,
    ignore: pd.DataFrame | None = None,
) -> dict[str, float]:
    overall, matches = evaluate_events(
        events, predictions, method="max_cardinality_iou", ignore=ignore
    )
    output: dict[str, float] = {}
    for relation in ("same", "different"):
        truth = events[events["hand_relation"] == relation]
        true_positive = int(
            np.count_nonzero(matches.get("hand_relation", pd.Series(dtype=str)) == relation)
        )
        sensitivity = true_positive / len(truth) if len(truth) else float("nan")
        precision = float(overall["precision"])
        output[f"{relation}_sensitivity"] = sensitivity
        output[f"{relation}_f1"] = (
            2.0 * precision * sensitivity / (precision + sensitivity)
            if np.isfinite(sensitivity) and precision + sensitivity > 0
            else float("nan")
        )
        relation_matches = matches[matches.get("hand_relation", pd.Series(dtype=str)) == relation]
        output[f"{relation}_start_mae_seconds"] = (
            float(relation_matches["start_absolute_error_ms"].mean() / 1000.0)
            if len(relation_matches)
            else float("nan")
        )
        output[f"{relation}_end_mae_seconds"] = (
            float(relation_matches["end_absolute_error_ms"].mean() / 1000.0)
            if len(relation_matches)
            else float("nan")
        )
    return output


def _per_subject_metrics_v4(
    events: pd.DataFrame,
    predictions: pd.DataFrame,
    windows: pd.DataFrame,
    ignore: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subjects = sorted(
        set(events.get("subject_key", pd.Series(dtype=str)).astype(str))
        | set(predictions.get("subject_key", pd.Series(dtype=str)).astype(str))
    )
    for subject in subjects:
        truth = events[events["subject_key"].astype(str) == subject]
        predicted = predictions[predictions["subject_key"].astype(str) == subject]
        observed = windows[windows["subject_key"].astype(str) == subject]
        subject_ignore = (
            ignore[ignore["subject_key"].astype(str) == subject] if ignore is not None else None
        )
        metrics, _ = evaluate_events(
            truth, predicted, method="max_cardinality_iou", ignore=subject_ignore
        )
        rows.append(
            {
                "subject_key": subject,
                **metrics,
                **_hand_metrics(truth, predicted, subject_ignore),
                "observed_hours": _observed_hours(observed),
                "fp_per_hour": metrics["false_positive"] / max(_observed_hours(observed), 1e-9),
            }
        )
    return pd.DataFrame(rows)


def select_v4_pipeline(
    run: HierarchicalRun,
    config: dict[str, Any],
    inputs: V4Inputs,
) -> None:
    run.require_stage("BOUNDARY_COMPLETE")
    scores = pd.read_parquet(run.root / "oof" / "proposal_scores.parquet")
    windows = pd.read_parquet(run.root / "oof" / "window_predictions.parquet")
    preselection = json.loads(
        (run.root / "verifier" / "preselection.json").read_text(encoding="utf-8")
    )
    verifier_kind = str(preselection["verifier_kind"])
    score_column = "final_score" if verifier_kind == "deep" else "logistic_score"
    point = preselection[verifier_kind]
    accepted = _accepted_from_point(scores, point, score_column)
    subjects = set(scores["subject_key"].astype(str))
    truth, ignore = partition_evaluation_events(inputs.events, subjects)
    coarse_predictions = _prediction_events(accepted)
    coarse_metrics, coarse_matches = evaluate_events(
        truth, coarse_predictions, method="max_cardinality_iou", ignore=ignore
    )
    boundary_status = json.loads(
        (run.root / "boundary" / "status.json").read_text(encoding="utf-8")
    )
    selected_entropy = None
    refined_predictions = coarse_predictions
    refined_metrics = coarse_metrics
    boundary_passed = False
    selected_boundary_seed: int | None = None
    boundary_seed_results: dict[str, dict[str, float | bool]] = {}
    if boundary_status["enabled"]:
        boundary_scores = pd.read_parquet(run.root / "oof" / "boundary_scores.parquet")
        coarse_mae = np.nanmean(
            [coarse_metrics["start_mae_seconds"], coarse_metrics["end_mae_seconds"]]
        )
        seed_improvements = 0
        seed_winners: list[
            tuple[int, float, dict[str, float], pd.DataFrame, pd.DataFrame, float]
        ] = []
        for seed in config["boundary"]["seeds"]:
            seed = int(seed)
            seeded = _seeded_boundary_scores(boundary_scores, seed)
            candidates: list[tuple[float, dict[str, float], pd.DataFrame, pd.DataFrame, float]] = []
            for threshold in config["boundary"]["entropy_thresholds"]:
                seed_refined = _refine_from_scores(
                    accepted,
                    seeded,
                    float(threshold),
                    int(config["boundary"]["safety_gap_seconds"]),
                )
                seed_predictions = seed_refined.rename(
                    columns={
                        "refined_start_ms": "start_ms",
                        "refined_end_ms": "end_ms",
                    }
                )[["subject_key", "start_ms", "end_ms", "proposal_id"]]
                seed_metrics, seed_matches = evaluate_events(
                    truth, seed_predictions, method="max_cardinality_iou", ignore=ignore
                )
                seed_mae = np.nanmean(
                    [
                        seed_metrics["start_mae_seconds"],
                        seed_metrics["end_mae_seconds"],
                    ]
                )
                candidates.append(
                    (
                        float(threshold),
                        seed_metrics,
                        seed_predictions,
                        seed_matches,
                        float(seed_mae),
                    )
                )
            winner = min(candidates, key=lambda value: (value[4], -value[1]["f1"]))
            seed_winners.append((seed, *winner))
            improved = bool(np.isfinite(winner[4]) and winner[4] < coarse_mae)
            seed_improvements += int(improved)
            boundary_seed_results[str(seed)] = {
                "entropy_threshold": float(winner[0]),
                "mean_mae_seconds": float(winner[4]),
                "f1": float(winner[1]["f1"]),
                "improved_mae": improved,
            }
        (
            selected_boundary_seed,
            selected_entropy,
            refined_metrics,
            refined_predictions,
            refined_matches,
            refined_mae,
        ) = min(seed_winners, key=lambda value: (value[5], -value[2]["f1"]))
        coarse_ids = set(
            coarse_matches.get("prediction_event_id", pd.Series(dtype=str)).astype(str)
        )
        refined_ids = set(
            refined_matches.get("prediction_event_id", pd.Series(dtype=str)).astype(str)
        )
        lost_fraction = len(coarse_ids - refined_ids) / max(len(coarse_ids), 1)
        seed_consistency_passed = seed_improvements >= min(2, len(config["boundary"]["seeds"]))
        hand_gate: dict[str, Any] = {}
        hand_gate_passed = True
        for relation in ("same", "different"):
            coarse_relation = coarse_matches[
                coarse_matches.get("hand_relation", pd.Series(dtype=str)) == relation
            ]
            refined_relation = refined_matches[
                refined_matches.get("hand_relation", pd.Series(dtype=str)) == relation
            ]
            enough = len(coarse_relation) >= 5 and len(refined_relation) >= 5
            mae_checks: dict[str, bool] = {}
            for endpoint in ("start", "end"):
                column = f"{endpoint}_absolute_error_ms"
                coarse_value = (
                    float(coarse_relation[column].mean()) if len(coarse_relation) else float("nan")
                )
                refined_value = (
                    float(refined_relation[column].mean())
                    if len(refined_relation)
                    else float("nan")
                )
                mae_checks[endpoint] = bool(
                    np.isfinite(coarse_value)
                    and np.isfinite(refined_value)
                    and refined_value <= coarse_value * 1.05
                )
            relation_passed = enough and all(mae_checks.values())
            hand_gate_passed &= relation_passed
            hand_gate[relation] = {
                "matched_events": len(refined_relation),
                "minimum_matched_events": 5,
                "start_mae_not_worse_than_5pct": mae_checks["start"],
                "end_mae_not_worse_than_5pct": mae_checks["end"],
                "passed": relation_passed,
            }
        boundary_passed = (
            refined_mae
            <= coarse_mae
            * (1.0 - float(config["promotion_gate"]["minimum_boundary_mae_improvement"]))
            and refined_metrics["f1"]
            >= coarse_metrics["f1"] - float(config["promotion_gate"]["maximum_boundary_f1_drop"])
            and lost_fraction <= float(config["promotion_gate"]["maximum_tp_to_fp_fraction"])
            and seed_consistency_passed
            and hand_gate_passed
        )
        if not boundary_passed:
            refined_predictions = coarse_predictions
            refined_metrics = coarse_metrics
            selected_entropy = None
            selected_boundary_seed = None
    greedy_metrics, _ = evaluate_events(truth, refined_predictions, method="greedy", ignore=ignore)
    metrics = {
        "coarse": {**coarse_metrics, **_hand_metrics(truth, coarse_predictions, ignore)},
        "selected": {**refined_metrics, **_hand_metrics(truth, refined_predictions, ignore)},
        "greedy_selected": greedy_metrics,
        "fp_per_hour": refined_metrics["false_positive"] / max(_observed_hours(windows), 1e-9),
        "boundary_gate_passed": bool(boundary_passed),
        "boundary_seed_consistency": boundary_seed_results,
        "boundary_hand_gate": hand_gate if boundary_status["enabled"] else {},
    }
    selection = {
        "schema_version": 4,
        "protocol_version": "statsfusion-r2",
        "anchor_semantics": "right_endpoint_half_open",
        "masking_protocol": "zero_mask_layernorm_v1",
        "candidate_budget_scope": "session",
        "meta_crossfit_protocol": "fully_nested_v1",
        "minimum_event_gap_seconds": int(config["boundary"]["safety_gap_seconds"]),
        "verifier_kind": verifier_kind,
        "score_column": "final_score",
        "acceptance_threshold": float(point["acceptance_threshold"]),
        "nms_iou_threshold": float(point["nms_iou_threshold"]),
        "boundary_enabled": bool(boundary_passed),
        "boundary_seed": selected_boundary_seed,
        "boundary_entropy_threshold": selected_entropy,
        "strict_iou_operator": ">",
        "strict_iou_threshold": 0.25,
        "maximum_future_context_seconds": 60,
    }
    selection_path = run.root / "selection" / "selected_pipeline.json"
    metrics_path = run.root / "selection" / "oof_metrics.json"
    write_json_atomic(selection_path, selection)
    write_json_atomic(metrics_path, metrics)
    run.transition("SELECTED", [selection_path, metrics_path])


def evaluate_v4_outer(
    run: HierarchicalRun,
    config: dict[str, Any],
    outer_inputs: V4Inputs,
) -> None:
    run.require_stage("SELECTED")
    selection = json.loads(
        (run.root / "selection" / "selected_pipeline.json").read_text(encoding="utf-8")
    )
    scores = pd.read_parquet(run.root / "outer" / "proposal_scores.parquet")
    accepted = _accepted_from_point(
        scores,
        {
            "acceptance_threshold": selection["acceptance_threshold"],
            "nms_iou_threshold": selection["nms_iou_threshold"],
        },
        selection["score_column"],
        int(selection.get("minimum_event_gap_seconds", 3)),
    )
    if selection["boundary_enabled"]:
        boundary_scores = pd.read_parquet(run.root / "outer" / "boundary_scores.parquet")
        boundary_scores = _seeded_boundary_scores(boundary_scores, int(selection["boundary_seed"]))
        refined = _refine_from_scores(
            accepted,
            boundary_scores,
            float(selection["boundary_entropy_threshold"]),
            int(config["boundary"]["safety_gap_seconds"]),
        )
        predictions = refined.rename(
            columns={"refined_start_ms": "start_ms", "refined_end_ms": "end_ms"}
        )[["subject_key", "start_ms", "end_ms", "proposal_id", "final_score"]]
    else:
        predictions = accepted.rename(
            columns={"coarse_start_ms": "start_ms", "coarse_end_ms": "end_ms"}
        )[["subject_key", "start_ms", "end_ms", "proposal_id", "final_score"]]
    outer_subjects = set(predictions["subject_key"].astype(str)) | set(
        outer_inputs.events["subject_key"].astype(str)
    )
    truth, ignore = partition_evaluation_events(outer_inputs.events, outer_subjects)
    max_metrics, matches = evaluate_events(
        truth, predictions, method="max_cardinality_iou", ignore=ignore
    )
    greedy_metrics, _ = evaluate_events(truth, predictions, method="greedy", ignore=ignore)
    windows = pd.read_parquet(run.root / "outer" / "window_predictions.parquet")
    outer_window_labels = outer_inputs.anchors[
        ["subject_key", "session_id", "timestamp_ms", "state_target", "state_loss_mask"]
    ]
    calibrated_windows = windows.merge(
        outer_window_labels,
        on=["subject_key", "session_id", "timestamp_ms"],
        how="left",
        validate="one_to_one",
    )
    if calibrated_windows[["state_target", "state_loss_mask"]].isna().any().any():
        raise RuntimeError("Outer state predictions failed to align with evaluation labels")
    calibration_windows, calibration_counts = _evaluable_state_calibration_rows(
        calibrated_windows
    )
    outer_calibration = state_calibration_metrics(
        calibration_windows["state_target"].to_numpy(),
        calibration_windows["state_logit"].to_numpy(),
        calibration_windows["state_probability"].to_numpy(),
        low_threshold=float(config["decoder"]["low_threshold"]),
        bins=int(config["calibration"]["ece_bins"]),
    )
    outer_calibration.update(calibration_counts)
    state_only_point = json.loads(
        (run.root / "decoder" / "candidate_metrics.json").read_text(encoding="utf-8")
    )
    state_only_proposals = pd.read_parquet(run.root / "outer" / "proposals.parquet")
    state_only_accepted = _accepted_from_point(
        state_only_proposals,
        {
            "acceptance_threshold": state_only_point["state_only_acceptance_threshold"],
            "nms_iou_threshold": state_only_point["state_only_nms_iou_threshold"],
        },
        "generator_score",
    )
    state_only_predictions = _prediction_events(state_only_accepted)
    state_only_metrics, _ = evaluate_events(
        truth, state_only_predictions, method="max_cardinality_iou", ignore=ignore
    )
    outer_candidate_metrics = _candidate_recall(state_only_proposals, truth)
    outer_candidate_metrics.update(
        {
            "candidate_count": float(len(state_only_proposals)),
            "candidates_per_hour": float(
                len(state_only_proposals) / max(_observed_hours(windows), 1e-9)
            ),
        }
    )
    metrics = {
        "max_cardinality_iou": max_metrics,
        "greedy": greedy_metrics,
        "hand": _hand_metrics(truth, predictions, ignore),
        "fp_per_hour": max_metrics["false_positive"] / max(_observed_hours(windows), 1e-9),
        "state_calibration": outer_calibration,
        "candidate": outer_candidate_metrics,
        "state_only": {
            **state_only_metrics,
            **_hand_metrics(truth, state_only_predictions, ignore),
            "fp_per_hour": state_only_metrics["false_positive"]
            / max(_observed_hours(windows), 1e-9),
        },
    }
    matched_truth = set(matches.get("event_id", pd.Series(dtype=str)).astype(str))
    matched_prediction = set(matches.get("prediction_event_id", pd.Series(dtype=str)).astype(str))
    ignored_mask = prediction_ignore_mask(predictions, ignore)
    ignored_predictions = predictions.loc[ignored_mask].copy()
    failures = pd.concat(
        (
            truth.loc[~truth["event_id"].astype(str).isin(matched_truth)].assign(
                failure_type="false_negative"
            ),
            predictions.loc[
                ~predictions["proposal_id"].astype(str).isin(matched_prediction) & ~ignored_mask
            ].assign(failure_type="false_positive"),
        ),
        ignore_index=True,
        sort=False,
    )
    prediction_path = run.root / "evaluation" / "outer_predictions.parquet"
    event_path = run.root / "evaluation" / "events.csv"
    metric_path = run.root / "evaluation" / "metrics.json"
    failure_path = run.root / "evaluation" / "failure_cases.csv"
    per_subject_path = run.root / "evaluation" / "per_subject_metrics.csv"
    state_only_subject_path = run.root / "evaluation" / "state_only_per_subject_metrics.csv"
    ignored_prediction_path = run.root / "evaluation" / "ignored_predictions.parquet"
    write_parquet_atomic(prediction_path, predictions)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_event = event_path.with_name(event_path.name + ".tmp")
    predictions.to_csv(temporary_event, index=False)
    temporary_event.replace(event_path)
    write_json_atomic(metric_path, metrics)
    temporary_failure = failure_path.with_name(failure_path.name + ".tmp")
    failures.to_csv(temporary_failure, index=False)
    temporary_failure.replace(failure_path)
    write_parquet_atomic(ignored_prediction_path, ignored_predictions)
    per_subject = _per_subject_metrics_v4(truth, predictions, windows, ignore)
    temporary_subject = per_subject_path.with_name(per_subject_path.name + ".tmp")
    per_subject.to_csv(temporary_subject, index=False)
    temporary_subject.replace(per_subject_path)
    state_only_subjects = _per_subject_metrics_v4(truth, state_only_predictions, windows, ignore)
    temporary_state_subject = state_only_subject_path.with_name(
        state_only_subject_path.name + ".tmp"
    )
    state_only_subjects.to_csv(temporary_state_subject, index=False)
    temporary_state_subject.replace(state_only_subject_path)
    run.transition(
        "EVALUATED",
        [
            prediction_path,
            event_path,
            metric_path,
            failure_path,
            per_subject_path,
            state_only_subject_path,
            ignored_prediction_path,
        ],
    )


def _concatenate_proposal_features(parts: list[ProposalFeatureBatchV4]) -> ProposalFeatureBatchV4:
    if not parts:
        raise ValueError("No proposal feature batches were provided")

    def optional(name: str):
        values = [getattr(part, name) for part in parts]
        return np.concatenate(values) if all(value is not None for value in values) else None

    return ProposalFeatureBatchV4(
        proposal_ids=np.concatenate([part.proposal_ids for part in parts]),
        sequence=np.concatenate([part.sequence for part in parts]),
        sequence_mask=np.concatenate([part.sequence_mask for part in parts]),
        scalar=np.concatenate([part.scalar for part in parts]),
        event_target=optional("event_target"),
        iou_target=optional("iou_target"),
        sample_weight=optional("sample_weight"),
    )


def train_hierarchical_final_v4(
    config: dict[str, Any],
    input_root: Path,
    output_root: Path,
    run_name: str,
    *,
    fresh: bool,
    resume: bool,
) -> Path:
    if fresh == resume:
        raise ValueError("Choose exactly one of fresh or resume for final v4 training")
    final_root = output_root / "final" / run_name
    manifest_path = final_root / "final_manifest.json"
    existing_manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        if fresh:
            raise FileExistsError(f"V4 final run already exists: {final_root}")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative, expected in existing_manifest.get("artifact_hashes", {}).items():
            if (
                not (final_root / relative).is_file()
                or sha256_file(final_root / relative) != expected
            ):
                raise RuntimeError(f"V4 final artifact changed: {relative}")
    elif resume:
        raise FileNotFoundError("V4 final run does not exist; start with --fresh")
    elif final_root.exists() and any(final_root.iterdir()):
        raise RuntimeError("V4 final directory exists without a valid final manifest")
    identity = current_v4_identity(config, input_root)
    experiment_root = output_root / "experiments" / run_name
    fold_roots = [experiment_root / f"fold_{fold}" for fold in range(5)]
    manifests = []
    parent_artifact_hashes: dict[str, str] = {}
    expected_state_seeds = [int(value) for value in config["final_training"]["state_seeds"]]
    expected_verifier_seeds = [int(value) for value in config["verifier"]["seeds"]]
    for fold, root in enumerate(fold_roots):
        path = root / "run_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"V4 fold {fold} manifest is missing")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("stage") != "EVALUATED":
            raise RuntimeError(f"V4 fold {fold} must be EVALUATED before final training")
        if manifest.get("protocol_version") != "statsfusion-r2":
            raise RuntimeError(f"V4 fold {fold} was not produced by statsfusion-r2")
        if manifest.get("resolved_config_sha256") != identity["resolved_config_sha256"]:
            raise RuntimeError(f"V4 fold {fold} configuration differs from final training")
        if manifest.get("git") != identity["git"]:
            raise RuntimeError(f"V4 fold {fold} worktree differs from final training")
        if manifest.get("input_hashes") != identity["input_hashes"]:
            raise RuntimeError(f"V4 fold {fold} v2 inputs differ from final training")
        if manifest.get("random_seeds", {}).get("state") != expected_state_seeds:
            raise RuntimeError(f"V4 fold {fold} state seeds differ from statsfusion-r2")
        if manifest.get("random_seeds", {}).get("verifier") != expected_verifier_seeds:
            raise RuntimeError(f"V4 fold {fold} verifier seeds differ from statsfusion-r2")
        HierarchicalRun(root, path, manifest).verify_artifacts()
        manifests.append(manifest)
        required_parent_paths = [
            path,
            root / "resolved_config.yaml",
            root / "selection" / "selected_pipeline.json",
            root / "selection" / "state_epochs.json",
            root / "outer" / "window_logits.parquet",
            root / "outer" / "window_predictions.parquet",
            root / "outer" / "proposals.parquet",
            root / "outer" / "proposal_scores.parquet",
            *(root / "outer" / f"state_seed_{seed}.pt" for seed in expected_state_seeds),
            *(root / "verifier" / f"final_seed_{seed}.pt" for seed in expected_verifier_seeds),
        ]
        missing_parent_paths = [
            item.relative_to(root).as_posix()
            for item in required_parent_paths
            if not item.is_file()
        ]
        if missing_parent_paths:
            raise FileNotFoundError(
                f"V4 fold {fold} final evidence is missing: {missing_parent_paths}"
            )
        for item in required_parent_paths:
            relative = item.relative_to(experiment_root).as_posix()
            parent_artifact_hashes[relative] = sha256_file(item)
        boundary_scores = root / "outer" / "boundary_scores.parquet"
        if boundary_scores.is_file():
            parent_artifact_hashes[boundary_scores.relative_to(experiment_root).as_posix()] = (
                sha256_file(boundary_scores)
            )
    stress_path = experiment_root / "stress_gate.json"
    if not stress_path.is_file():
        raise FileNotFoundError("V4 stress gate is missing")
    stress = json.loads(stress_path.read_text(encoding="utf-8"))
    if not bool(stress.get("passed", False)):
        raise RuntimeError("V4 frozen stress gate did not pass; final training is blocked")
    verify_gate_evidence(output_root.parent, stress)
    parent_artifact_hashes[stress_path.relative_to(experiment_root).as_posix()] = sha256_file(
        stress_path
    )
    resume_identity = {
        **identity,
        "state_seeds": expected_state_seeds,
        "verifier_seeds": expected_verifier_seeds,
        "parent_artifact_hashes": parent_artifact_hashes,
    }
    if existing_manifest is not None:
        if existing_manifest.get("protocol_version") != "statsfusion-r2":
            raise RuntimeError("Legacy or blocked v4 final runs cannot be resumed")
        if existing_manifest.get("resume_identity") != resume_identity:
            raise RuntimeError("V4 final resume identity differs from its locked evidence")
        return final_root
    final_root.mkdir(parents=True, exist_ok=True)
    inputs = load_v4_inputs(
        config,
        input_root,
        fold=0,
        event_role="all",
        allow_outer_labels=True,
    )
    expected_partition = {"truth": 161, "ignore": 100, "invalid_duration": 6}
    actual_partition = evaluation_event_partition_summary(inputs.events)
    if actual_partition != expected_partition:
        raise RuntimeError(
            f"V4 truth/ignore contract changed: expected {expected_partition}, got {actual_partition}"
        )
    fold_selections = [
        json.loads((root / "selection" / "selected_pipeline.json").read_text(encoding="utf-8"))
        for root in fold_roots
    ]
    verifier_kind = str(fold_selections[0]["verifier_kind"])
    for current in fold_selections[1:]:
        if current["verifier_kind"] != verifier_kind:
            raise RuntimeError("Final folds disagree on the frozen verifier kind")
    scaler, transformed = _fit_scaler_and_transform(
        inputs, set(inputs.anchors["subject_key"].astype(str))
    )
    normalization = compute_normalization(
        inputs.segments, set(inputs.anchors["subject_key"].astype(str))
    )
    epochs_by_seed: dict[int, list[int]] = {
        int(seed): [] for seed in config["final_training"]["state_seeds"]
    }
    for root in fold_roots:
        payload = json.loads((root / "selection" / "state_epochs.json").read_text(encoding="utf-8"))
        for seed, values in epochs_by_seed.items():
            values.append(int(payload["fixed_outer_epoch_by_seed"][str(seed)]))
    fixed_epochs = {seed: int(np.median(values)) for seed, values in epochs_by_seed.items()}
    artifacts: list[Path] = []
    scaler_path = final_root / "statistics_scaler.json"
    normalization_path = final_root / "sensor_normalization.json"
    write_json_atomic(scaler_path, scaler.to_json())
    save_normalization(normalization, normalization_path)
    artifacts.extend((scaler_path, normalization_path))
    final_state_parent_sha256 = {
        "statistics_scaler": sha256_file(scaler_path),
        "sensor_normalization": sha256_file(normalization_path),
    }
    final_training_subjects = set(inputs.anchors["subject_key"].astype(str))
    for seed in config["final_training"]["state_seeds"]:
        seed = int(seed)
        full_dataset = _make_dataset(
            transformed,
            inputs,
            inputs.events,
            normalization,
            config,
            training=True,
            seed=seed,
        )
        torch.manual_seed(int(seed))
        model = build_state_model(config["model"])
        _train_state_epochs(
            model,
            full_dataset,
            config,
            epochs=fixed_epochs[seed],
            seed=int(seed),
            total_epochs=fixed_epochs[seed],
            scheduler_total_epochs=int(config["training"]["max_epochs"]),
        )
        path = final_root / f"state_seed_{seed}.pt"
        _save_torch_atomic(
            path,
            {
                "model": model.state_dict(),
                "model_config": config["model"],
                "epochs": fixed_epochs[seed],
                "training_subject_count": int(inputs.anchors["subject_key"].nunique()),
                "training_subjects": sorted(final_training_subjects),
                "prediction_subjects": [],
                "globally_excluded_subjects": [],
                "parent_artifact_sha256": final_state_parent_sha256,
            },
        )
        artifacts.append(path)

    outer_logits_parts: list[pd.DataFrame] = []
    proposal_feature_parts: list[ProposalFeatureBatchV4] = []
    scored_parts: list[pd.DataFrame] = []
    boundary_positive_parts: list[pd.DataFrame] = []
    window_parts: list[pd.DataFrame] = []
    truth_parts: list[pd.DataFrame] = []
    ignore_parts: list[pd.DataFrame] = []
    pooled_evidence: list[dict[str, Any]] = []
    for fold, root in enumerate(fold_roots):
        outer_subjects = {
            subject
            for subject, subject_fold in inputs.subject_folds.items()
            if subject_fold == fold
        }
        evaluable, ignored = partition_evaluation_events(inputs.events, outer_subjects)
        anchors = inputs.anchors[inputs.anchors["subject_key"].astype(str).isin(outer_subjects)][
            ["subject_key", "session_id", "timestamp_ms", "state_target", "state_loss_mask"]
        ]
        logits = pd.read_parquet(root / "outer" / "window_logits.parquet").merge(
            anchors,
            on=["subject_key", "session_id", "timestamp_ms"],
            how="left",
            validate="one_to_one",
        )
        if logits["state_target"].isna().any():
            raise RuntimeError(f"Fold {fold} outer logits failed to align with final labels")
        logits = _mask_ignored_state_rows(
            logits,
            ignored,
            step_ms=int(config["sequence"]["step_seconds"]) * 1000,
        )
        outer_logits_parts.append(logits)
        windows = pd.read_parquet(root / "outer" / "window_predictions.parquet")
        proposals = pd.read_parquet(root / "outer" / "proposals.parquet")
        labeled = (
            exclude_ignored_candidates(label_event_candidates(proposals, evaluable, 0.25), ignored)
            .sort_values("proposal_id")
            .reset_index(drop=True)
        )
        proposal_feature_parts.append(
            build_proposal_features_v4(
                labeled,
                windows,
                [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
                config["verifier"],
            )
        )
        scored = pd.read_parquet(root / "outer" / "proposal_scores.parquet")
        scored = scored[
            scored["proposal_id"].astype(str).isin(set(labeled["proposal_id"].astype(str)))
        ].merge(
            labeled[["proposal_id", "max_iou", "matched_event_id", "is_positive"]],
            on="proposal_id",
            how="left",
            validate="one_to_one",
        )
        scored = scored.sort_values("proposal_id").reset_index(drop=True)
        scored_parts.append(scored)
        positive = scored[scored["max_iou"] > 0.25]
        boundary_positive_parts.append(_attach_truth_boundaries(positive, evaluable))
        window_parts.append(windows)
        truth_parts.append(evaluable)
        ignore_parts.append(ignored)
        pooled_evidence.append(
            {
                "fold": fold,
                "run_manifest_sha256": sha256_file(root / "run_manifest.json"),
                "resolved_config_sha256": sha256_file(root / "resolved_config.yaml"),
                "window_logits_sha256": sha256_file(root / "outer" / "window_logits.parquet"),
                "window_predictions_sha256": sha256_file(
                    root / "outer" / "window_predictions.parquet"
                ),
                "proposals_sha256": sha256_file(root / "outer" / "proposals.parquet"),
                "proposal_scores_sha256": sha256_file(root / "outer" / "proposal_scores.parquet"),
                "selection_sha256": sha256_file(root / "selection" / "selected_pipeline.json"),
                "truth_labels_sha256": _dataframe_sha256(
                    evaluable, ["subject_key", "event_id", "start_ms", "end_ms"]
                ),
                "ignore_labels_sha256": _dataframe_sha256(
                    ignored, ["subject_key", "event_id", "start_ms", "end_ms"]
                ),
                "state_checkpoint_sha256": {
                    str(seed): sha256_file(root / "outer" / f"state_seed_{seed}.pt")
                    for seed in expected_state_seeds
                },
                "verifier_checkpoint_sha256": {
                    str(seed): sha256_file(root / "verifier" / f"final_seed_{seed}.pt")
                    for seed in expected_verifier_seeds
                },
                "state_seeds": expected_state_seeds,
                "verifier_seeds": expected_verifier_seeds,
            }
        )
    final_logits = pd.concat(outer_logits_parts, ignore_index=True)
    from bme_eating.calibration_v4 import PlattCalibration

    calibration_rows = final_logits[final_logits["state_loss_mask"].to_numpy(dtype=float) > 0]
    state_calibration = PlattCalibration.fit(
        calibration_rows["state_logit"].to_numpy(),
        calibration_rows["state_target"].to_numpy(),
    )
    state_calibration_path = final_root / "state_calibration.json"
    write_json_atomic(state_calibration_path, state_calibration.to_json())
    artifacts.append(state_calibration_path)
    all_subjects = set(inputs.events["subject_key"].astype(str))
    durations = _truth_event_durations(inputs.events, all_subjects)
    prior = TruncatedLogNormalDurationPrior.fit(
        durations,
        lower_quantile=float(config["decoder"]["duration_lower_quantile"]),
        upper_quantile=float(config["decoder"]["duration_upper_quantile"]),
        minimum_floor_seconds=float(config["decoder"]["minimum_duration_floor_seconds"]),
        maximum_ceiling_seconds=float(config["decoder"]["maximum_duration_ceiling_seconds"]),
    )
    prior_path = final_root / "duration_prior.json"
    write_json_atomic(prior_path, prior.to_json())
    artifacts.append(prior_path)

    verifier_features = _concatenate_proposal_features(proposal_feature_parts)
    calibration_frame = pd.concat(scored_parts, ignore_index=True)
    pooled_verifier_subjects = set(calibration_frame["subject_key"].astype(str))
    _assert_proposal_feature_alignment(
        verifier_features,
        calibration_frame,
        context="Pooled outer OOF",
    )
    categories = classify_proposals(calibration_frame)
    logistic = LogisticScoreCombiner.fit(
        _verifier_matrix(verifier_features),
        verifier_features.event_target,
        sample_weight=verifier_features.sample_weight,
    )
    logistic_path = final_root / "logistic_verifier.json"
    write_json_atomic(logistic_path, logistic.to_json())
    artifacts.append(logistic_path)
    verifier_epoch_by_seed: dict[int, int] = {}
    pooled_verifier_parent_sha256 = {
        **parent_artifact_hashes,
        "state_calibration": sha256_file(state_calibration_path),
        "duration_prior": sha256_file(prior_path),
    }
    if verifier_kind == "deep":
        for seed_value in config["verifier"]["seeds"]:
            seed = int(seed_value)
            selected_epochs = []
            for root in fold_roots:
                for path in sorted(root.glob(f"verifier/partition_*_seed_{seed}_selector.json")):
                    selected_epochs.append(
                        int(json.loads(path.read_text(encoding="utf-8"))["selected_epoch"])
                    )
            if not selected_epochs:
                raise RuntimeError(f"Missing verifier selector evidence for seed {seed}")
            epochs = int(np.median(selected_epochs))
            verifier_epoch_by_seed[seed] = epochs
            verifier = _train_verifier_model_v4(
                verifier_features,
                categories,
                config,
                seed=seed + 120_000,
                epochs=epochs,
            )
            verifier_path = final_root / f"verifier_seed_{seed}.pt"
            _save_torch_atomic(
                verifier_path,
                {
                    "model": verifier.state_dict(),
                    "sequence_dim": verifier_features.sequence.shape[-1],
                    "scalar_dim": verifier_features.scalar.shape[-1],
                    "config": config["verifier"],
                    "seed": seed,
                    "epochs": epochs,
                    "training_subjects": sorted(pooled_verifier_subjects),
                    "prediction_subjects": [],
                    "globally_excluded_subjects": [],
                    "parent_artifact_sha256": pooled_verifier_parent_sha256,
                    "state_calibration_source": "pooled_outer_oof",
                    "duration_prior_source": "pooled_outer_training_truth",
                },
            )
            artifacts.append(verifier_path)
    proposal_calibration = ProposalCalibrationV4.fit(calibration_frame)
    proposal_calibration_path = final_root / "proposal_calibration.json"
    write_json_atomic(proposal_calibration_path, proposal_calibration.to_json())
    artifacts.append(proposal_calibration_path)

    pooled_scores = proposal_calibration.apply(calibration_frame)
    pooled_scores["logistic_score"] = logistic.predict(_verifier_matrix(verifier_features))
    if verifier_kind == "logistic":
        pooled_scores["final_score"] = pooled_scores["logistic_score"]
    pooled_truth = pd.concat(truth_parts, ignore_index=True)
    pooled_ignore = pd.concat(ignore_parts, ignore_index=True)
    pooled_windows = pd.concat(window_parts, ignore_index=True)
    point = _best_verifier_operating_point(
        pooled_scores,
        pooled_truth,
        pooled_windows,
        "final_score",
        config,
        pooled_ignore,
    )
    selection: dict[str, Any] = {
        "schema_version": 4,
        "protocol_version": "statsfusion-r2",
        "selection_source": "pooled_outer_oof",
        "anchor_semantics": "right_endpoint_half_open",
        "masking_protocol": "zero_mask_layernorm_v1",
        "candidate_budget_scope": "session",
        "meta_crossfit_protocol": "fully_nested_v1",
        "minimum_event_gap_seconds": int(config["boundary"]["safety_gap_seconds"]),
        "verifier_kind": verifier_kind,
        "score_column": "final_score",
        "acceptance_threshold": float(point["acceptance_threshold"]),
        "nms_iou_threshold": float(point["nms_iou_threshold"]),
        "boundary_enabled": False,
        "boundary_seed": None,
        "boundary_entropy_threshold": None,
        "state_seeds": [int(value) for value in config["final_training"]["state_seeds"]],
        "verifier_seeds": [int(value) for value in config["verifier"]["seeds"]],
        "ignore_protocol_version": "valid-duration-truth-ignore-v1",
        "strict_iou_operator": ">",
        "strict_iou_threshold": 0.25,
        "maximum_future_context_seconds": 60,
        "input_evidence_sha256": {
            "folds": pooled_evidence,
            "labels": _dataframe_sha256(
                pooled_truth, ["subject_key", "event_id", "start_ms", "end_ms"]
            ),
            "ignore": _dataframe_sha256(
                pooled_ignore, ["subject_key", "event_id", "start_ms", "end_ms"]
            ),
            "config": manifests[0]["resolved_config_sha256"],
            "events_file": manifests[0]["input_hashes"]["events"],
        },
    }

    all_fold_boundary_enabled = all(
        bool(value.get("boundary_enabled", False)) for value in fold_selections
    )
    boundary_score_files_complete = all(
        (root / "outer" / "boundary_scores.parquet").is_file() for root in fold_roots
    )
    selection["boundary_selection_diagnostics"] = {
        "all_folds_enabled": all_fold_boundary_enabled,
        "score_files_complete": boundary_score_files_complete,
        "full_accepted_coverage": False,
        "accepted_count": 0,
        "covered_count": 0,
    }
    if all_fold_boundary_enabled and boundary_score_files_complete:
        accepted = _accepted_from_point(pooled_scores, point, "final_score")
        pooled_boundary = pd.concat(
            [pd.read_parquet(root / "outer" / "boundary_scores.parquet") for root in fold_roots],
            ignore_index=True,
        )
        if pooled_boundary["proposal_id"].astype(str).duplicated().any():
            raise RuntimeError("Pooled boundary proposal IDs are not globally unique")
        accepted_ids = set(accepted["proposal_id"].astype(str))
        covered_ids = set(pooled_boundary["proposal_id"].astype(str))
        full_coverage = accepted_ids.issubset(covered_ids)
        selection["boundary_selection_diagnostics"].update(
            {
                "full_accepted_coverage": full_coverage,
                "accepted_count": len(accepted_ids),
                "covered_count": len(accepted_ids & covered_ids),
            }
        )
        if full_coverage:
            coarse_predictions = _prediction_events(accepted)
            coarse_metrics, coarse_matches = evaluate_events(
                pooled_truth,
                coarse_predictions,
                method="max_cardinality_iou",
                ignore=pooled_ignore,
            )
            coarse_mae = np.nanmean(
                [coarse_metrics["start_mae_seconds"], coarse_metrics["end_mae_seconds"]]
            )
            winners: list[tuple[float, int, float]] = []
            for seed_value in config["boundary"]["seeds"]:
                seed = int(seed_value)
                seeded = _seeded_boundary_scores(pooled_boundary, seed)
                for threshold in config["boundary"]["entropy_thresholds"]:
                    refined = _refine_from_scores(
                        accepted,
                        seeded,
                        float(threshold),
                        int(config["boundary"]["safety_gap_seconds"]),
                    )
                    predictions = refined.rename(
                        columns={"refined_start_ms": "start_ms", "refined_end_ms": "end_ms"}
                    )[["subject_key", "start_ms", "end_ms", "proposal_id"]]
                    metrics, matches = evaluate_events(
                        pooled_truth,
                        predictions,
                        method="max_cardinality_iou",
                        ignore=pooled_ignore,
                    )
                    mae = np.nanmean([metrics["start_mae_seconds"], metrics["end_mae_seconds"]])
                    coarse_ids = set(
                        coarse_matches.get("prediction_event_id", pd.Series(dtype=str)).astype(str)
                    )
                    refined_ids = set(
                        matches.get("prediction_event_id", pd.Series(dtype=str)).astype(str)
                    )
                    lost = len(coarse_ids - refined_ids) / max(len(coarse_ids), 1)
                    hand_safe = True
                    for relation in ("same", "different"):
                        coarse_relation = coarse_matches[
                            coarse_matches.get("hand_relation", pd.Series(dtype=str)) == relation
                        ]
                        refined_relation = matches[
                            matches.get("hand_relation", pd.Series(dtype=str)) == relation
                        ]
                        if len(coarse_relation) < 5 or len(refined_relation) < 5:
                            hand_safe = False
                            break
                        for endpoint in ("start", "end"):
                            column = f"{endpoint}_absolute_error_ms"
                            if (
                                float(refined_relation[column].mean())
                                > float(coarse_relation[column].mean()) * 1.05
                            ):
                                hand_safe = False
                    passed = bool(
                        np.isfinite(mae)
                        and mae
                        <= coarse_mae
                        * (
                            1.0
                            - float(config["promotion_gate"]["minimum_boundary_mae_improvement"])
                        )
                        and metrics["f1"]
                        >= coarse_metrics["f1"]
                        - float(config["promotion_gate"]["maximum_boundary_f1_drop"])
                        and lost <= float(config["promotion_gate"]["maximum_tp_to_fp_fraction"])
                        and hand_safe
                    )
                    if passed:
                        winners.append((float(mae), seed, float(threshold)))
            if winners:
                _, boundary_seed, entropy_threshold = min(winners)
                selection.update(
                    {
                        "boundary_enabled": True,
                        "boundary_seed": boundary_seed,
                        "boundary_entropy_threshold": entropy_threshold,
                    }
                )

    if selection["boundary_enabled"]:
        positive = pd.concat(boundary_positive_parts, ignore_index=True)
        independent_events = positive["matched_event_id"].nunique()
        if independent_events < int(config["boundary"]["minimum_independent_events"]):
            raise RuntimeError("Final boundary training has fewer independent events than required")
        start_residual = (
            positive["truth_start_ms"].to_numpy(dtype=float)
            - positive["coarse_start_ms"].to_numpy(dtype=float)
        ) / 1000.0
        end_residual = (
            positive["truth_end_ms"].to_numpy(dtype=float)
            - positive["coarse_end_ms"].to_numpy(dtype=float)
        ) / 1000.0
        boundary_range = select_boundary_range(
            start_residual,
            end_residual,
            quantile=float(config["boundary"]["residual_quantile"]),
            minimum_seconds=int(config["boundary"]["minimum_range_seconds"]),
            maximum_seconds=int(config["boundary"]["maximum_range_seconds"]),
        )
        if boundary_range.clipped_fraction > float(config["boundary"]["maximum_clipped_fraction"]):
            raise RuntimeError("Final OOF boundary residual clipping exceeds the configured limit")
        augmented = augment_boundary_training_proposals(
            positive,
            maximum_jitters_per_event=int(config["boundary"]["maximum_jitters_per_event"]),
            jitter_seconds=int(config["boundary"]["jitter_seconds"]),
        )
        boundary_features = build_endpoint_features(
            augmented,
            pd.concat(window_parts, ignore_index=True),
            [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
            boundary_range,
            config["boundary"],
        )
        boundary_seed = int(selection["boundary_seed"])
        boundary_epochs = []
        for root in fold_roots:
            for path in sorted(
                root.glob(f"boundary/partition_*_seed_{boundary_seed}_selector.json")
            ):
                boundary_epochs.append(
                    int(json.loads(path.read_text(encoding="utf-8"))["selected_epoch"])
                )
        if not boundary_epochs:
            raise RuntimeError("Missing pooled boundary selector evidence")
        fixed_boundary_epoch = int(np.median(boundary_epochs))
        boundary = _train_endpoint_model(
            boundary_features,
            config,
            seed=boundary_seed + 120_000,
            epochs=fixed_boundary_epoch,
        )
        boundary_path = final_root / "boundary.pt"
        range_path = final_root / "boundary_range.json"
        _save_torch_atomic(
            boundary_path,
            {
                "model": boundary.state_dict(),
                "input_dim": boundary_features.start_sequence.shape[-1],
                "config": config["boundary"],
                "seed": boundary_seed,
                "epochs": fixed_boundary_epoch,
                "training_subjects": sorted(set(positive["subject_key"].astype(str))),
                "prediction_subjects": [],
                "globally_excluded_subjects": [],
                "parent_artifact_sha256": parent_artifact_hashes,
                "state_calibration_source": "pooled_outer_oof",
                "duration_prior_source": "pooled_outer_training_truth",
            },
        )
        write_json_atomic(range_path, boundary_range.__dict__)
        artifacts.extend((boundary_path, range_path))
    selected_path = final_root / "selected_pipeline.json"
    resolved_path = final_root / "resolved_config.yaml"
    write_json_atomic(selected_path, selection)
    write_yaml_atomic(
        resolved_path,
        {key: value for key, value in config.items() if not key.startswith("_")},
    )
    artifacts.extend((selected_path, resolved_path))
    snapshot_root = final_root / "oof_training_snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_root / "source_folds.json"
    write_json_atomic(
        snapshot_path,
        {
            "run_name": run_name,
            "protocol_version": "statsfusion-r2",
            "selection_source": "pooled_outer_oof",
            "pooled_evidence": pooled_evidence,
            "fold_manifests": [
                {
                    "fold": fold,
                    "manifest_sha256": sha256_file(root / "run_manifest.json"),
                    "selected_pipeline_sha256": sha256_file(
                        root / "selection" / "selected_pipeline.json"
                    ),
                }
                for fold, root in enumerate(fold_roots)
            ],
        },
    )
    artifacts.append(snapshot_path)
    manifest = {
        "version": 4,
        "stage": "COMPLETE",
        "run_name": run_name,
        "protocol_version": "statsfusion-r2",
        "state_seeds": [int(value) for value in config["final_training"]["state_seeds"]],
        "fixed_state_epoch_by_seed": {str(seed): epoch for seed, epoch in fixed_epochs.items()},
        "verifier_seeds": [int(value) for value in config["verifier"]["seeds"]],
        "fixed_verifier_epoch_by_seed": {
            str(seed): epoch for seed, epoch in verifier_epoch_by_seed.items()
        },
        "resume_identity": resume_identity,
        "stress_gate_sha256": sha256_file(stress_path),
        "artifact_hashes": {
            path.relative_to(final_root).as_posix(): sha256_file(path) for path in artifacts
        },
        "bundle_exclusions": [
            "training_labels",
            "raw_data",
            "outer_predictions",
            "personal_absolute_paths",
            "credentials",
            "xgboost_models",
            "xgboost_scores",
        ],
    }
    write_json_atomic(manifest_path, manifest)
    return final_root
