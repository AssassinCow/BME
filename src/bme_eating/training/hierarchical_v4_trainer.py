from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from bme_eating.calibration_v4 import (
    LogisticScoreCombiner,
    ProposalCalibrationV4,
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
from bme_eating.hierarchical_v4_gates import verify_gate_evidence
from bme_eating.metrics import evaluate_events
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

ALIGNMENT_KEYS = ["segment_id", "session_id", "subject_key", "timestamp_ms"]
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
    feature_path = input_root / "features" / f"{feature_artifact_name(config)}.parquet"
    statistics = pd.read_parquet(
        feature_path,
        columns=[*ALIGNMENT_KEYS, *STATS_FEATURE_COLUMNS],
    )
    if statistics.duplicated(ALIGNMENT_KEYS).any():
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


def _selector_split(subjects: set[str], fraction: float, seed: int) -> tuple[set[str], set[str]]:
    if not 0 < fraction < 0.5:
        raise ValueError("Selector fraction must be in (0, 0.5)")
    rng = np.random.default_rng(seed)
    ordered = np.asarray(sorted(subjects), dtype=object)
    rng.shuffle(ordered)
    count = max(1, min(len(ordered) - 1, round(len(ordered) * fraction)))
    selector = {str(value) for value in ordered[:count]}
    fit = {str(value) for value in ordered[count:]}
    assert_disjoint_subjects(fit=fit, selector=selector)
    return fit, selector


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


def _transform_with_scaler(
    inputs: V4Inputs, scaler: FoldRobustScaler
) -> pd.DataFrame:
    transformed = scaler.transform_frame(inputs.statistics)
    statistics_columns = [
        *(f"stat_{name}" for name in STATS_FEATURE_COLUMNS),
        *(f"stat_{name}_missing" for name in STATS_FEATURE_COLUMNS),
    ]
    anchors = inputs.anchors.merge(
        transformed[[*ALIGNMENT_KEYS, *statistics_columns]],
        on=ALIGNMENT_KEYS,
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
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if config["training"].get("amp_dtype") == "bfloat16" else torch.float16
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epoch_offset, epoch_offset + int(epochs)):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(loader, start=1):
            tensors = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(tensors)
                loss, _ = criterion(output, tensors)
                loss = loss / accumulation
            loss.backward()
            if step % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["training"]["gradient_clip_norm"])
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if len(loader) % accumulation:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return optimizer


def _selector_score(
    model: torch.nn.Module,
    dataset: StatsFusionSequenceDataset,
    config: dict[str, Any],
    seed: int,
) -> float:
    device = _device(config)
    sampler = ClipMixtureSampler(
        dataset.anchors,
        samples_per_epoch=min(128, len(dataset)),
        mixture={"uniform": 1.0, "event": 0.0, "boundary": 0.0},
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), sampler=sampler)
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            output = model(tensors)
            mask = tensors["supervision_mask"] > 0
            targets.append(tensors["state_target"][mask].cpu().numpy())
            probabilities.append(torch.sigmoid(output["state_logit"][mask]).cpu().numpy())
    target = np.concatenate(targets)
    probability = np.concatenate(probabilities)
    return float(average_precision_score(target, probability)) if np.any(target > 0) else 0.0


def _select_epoch(
    fit_anchors: pd.DataFrame,
    selector_anchors: pd.DataFrame,
    fit_subjects: set[str],
    inputs: V4Inputs,
    config: dict[str, Any],
    seed: int,
) -> int:
    scaler, transformed = _fit_scaler_and_transform(inputs, fit_subjects)
    del scaler
    fit_rows = transformed[transformed["subject_key"].astype(str).isin(fit_subjects)].reset_index(drop=True)
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
    model = build_state_model(config["model"])
    best_epoch = 1
    best_score = -math.inf
    optimizer = None
    for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
        optimizer = _train_state_epochs(
            model,
            train_dataset,
            config,
            epochs=1,
            seed=seed,
            optimizer=optimizer,
            epoch_offset=epoch - 1,
        )
        score = _selector_score(model, selector_dataset, config, seed + 10_000 + epoch)
        if score > best_score:
            best_score = score
            best_epoch = epoch
    return best_epoch


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
    for batch in loader:
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
                    "statistics_gate": output["statistics_gate"][sample].float().cpu().numpy()[mask],
                    "long_gate": output["long_gate"][sample].float().cpu().numpy()[mask],
                    "missing_fraction": output["missing_fraction"][sample].float().cpu().numpy()[mask],
                    "state_target": tensors["state_target"][sample].cpu().numpy()[mask],
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
    gate_columns = ["ppg_gate", "statistics_gate", "long_gate"]
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
    statistics_mean = diagnostics.get(
        "statistics_gate_mean", pd.Series(dtype=np.float64)
    ).to_numpy(dtype=np.float64)
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
    all_predictions: list[pd.DataFrame] = []
    artifacts: list[Path] = []
    selected_epochs: list[int] = []
    for partition in sorted(set(partitions.values())):
        holdout = {subject for subject, value in partitions.items() if value == partition}
        training_subjects = outer_train - holdout
        fit_subjects, selector_subjects = _selector_split(
            training_subjects,
            float(config["training"]["selector_fraction"]),
            int(config["training"]["random_seed"]) + partition,
        )
        assert_disjoint_subjects(fit=fit_subjects, selector=selector_subjects, holdout=holdout, outer=outer_test)
        partition_root = run.root / "crossfit" / f"partition_{partition}" / "state"
        checkpoint_path = partition_root / "best.pt"
        scaler_path = partition_root / "statistics_scaler.json"
        normalization_path = partition_root / "sensor_normalization.json"
        if resume and checkpoint_path.is_file() and scaler_path.is_file() and normalization_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            epochs = int(checkpoint["epochs"])
            scaler = FoldRobustScaler.from_json(json.loads(scaler_path.read_text(encoding="utf-8")))
            normalization = load_normalization(normalization_path)
            model = build_state_model(checkpoint["model_config"])
            model.load_state_dict(checkpoint["model"])
            transformed = _transform_with_scaler(inputs, scaler)
        else:
            epochs = _select_epoch(
                inputs.anchors[inputs.anchors["subject_key"].astype(str).isin(fit_subjects)],
                inputs.anchors[
                    inputs.anchors["subject_key"].astype(str).isin(selector_subjects)
                ],
                fit_subjects,
                inputs,
                config,
                int(config["training"]["random_seed"]) + partition,
            )
            scaler, transformed = _fit_scaler_and_transform(inputs, training_subjects)
            normalization = compute_normalization(inputs.segments, training_subjects)
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
                seed=int(config["training"]["random_seed"]) + partition,
            )
            torch.manual_seed(int(config["training"]["random_seed"]) + partition)
            model = build_state_model(config["model"])
            _train_state_epochs(
                model,
                dataset,
                config,
                epochs=epochs,
                seed=int(config["training"]["random_seed"]) + partition,
            )
            write_json_atomic(scaler_path, scaler.to_json())
            save_normalization(normalization, normalization_path)
            _save_torch_atomic(
                checkpoint_path,
                {
                    "model": model.state_dict(),
                    "model_config": config["model"],
                    "epochs": epochs,
                    "training_subjects": sorted(training_subjects),
                    "holdout_subjects": sorted(holdout),
                },
            )
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
        all_predictions.append(
            infer_state_windows(
                model, holdout_dataset, config, stacking_partition=partition
            )
        )
        selected_epochs.append(epochs)
        artifacts.extend((checkpoint_path, scaler_path, normalization_path))
    oof_predictions = pd.concat(all_predictions, ignore_index=True)
    oof_path = run.root / "oof" / "window_logits.parquet"
    gate_path = run.root / "oof" / "gate_diagnostics.parquet"
    gate_summary_path = run.root / "oof" / "gate_diagnostics.json"
    write_parquet_atomic(oof_path, oof_predictions)
    gate_diagnostics, gate_summary = _gate_diagnostics(oof_predictions)
    write_parquet_atomic(gate_path, gate_diagnostics)
    write_json_atomic(gate_summary_path, gate_summary)
    artifacts.extend((oof_path, gate_path, gate_summary_path))
    fixed_outer_epoch = int(np.median(selected_epochs))
    outer_scaler, outer_transformed = _fit_scaler_and_transform(inputs, outer_train)
    outer_normalization = compute_normalization(inputs.segments, outer_train)
    outer_train_rows = outer_transformed[
        outer_transformed["subject_key"].astype(str).isin(outer_train)
    ].reset_index(drop=True)
    outer_train_events = inputs.events[
        inputs.events["subject_key"].astype(str).isin(outer_train)
    ]
    outer_dataset = _make_dataset(
        outer_train_rows,
        inputs,
        outer_train_events,
        outer_normalization,
        config,
        training=True,
        seed=int(config["training"]["random_seed"]) + 50_000,
    )
    torch.manual_seed(int(config["training"]["random_seed"]) + 50_000)
    outer_model = build_state_model(config["model"])
    _train_state_epochs(
        outer_model,
        outer_dataset,
        config,
        epochs=fixed_outer_epoch,
        seed=int(config["training"]["random_seed"]) + 50_000,
    )
    outer_state_root = run.root / "outer" / "state"
    outer_checkpoint = outer_state_root / "model.pt"
    outer_scaler_path = outer_state_root / "statistics_scaler.json"
    outer_normalization_path = outer_state_root / "sensor_normalization.json"
    _save_torch_atomic(
        outer_checkpoint,
        {
            "model": outer_model.state_dict(),
            "model_config": config["model"],
            "epochs": fixed_outer_epoch,
            "training_subjects": sorted(outer_train),
        },
    )
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
    write_parquet_atomic(
        outer_logits_path,
        infer_state_windows(
            outer_model,
            outer_inference_dataset,
            config,
            stacking_partition=-1,
        ).drop(columns=["state_target"]),
    )
    artifacts.extend(
        (outer_checkpoint, outer_scaler_path, outer_normalization_path, outer_logits_path)
    )
    epoch_path = run.root / "selection" / "state_epochs.json"
    write_json_atomic(
        epoch_path,
        {"partition_epochs": selected_epochs, "fixed_outer_epoch": fixed_outer_epoch},
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
    valid_events = (
        events[events["evaluable"].fillna(False)] if "evaluable" in events else events
    )
    matched: set[str] = set()
    grouped = {
        str(subject): group
        for subject, group in proposals.groupby("subject_key", sort=False)
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
    metrics = {
        "candidate_recall": len(matched) / len(valid_events) if len(valid_events) else 0.0
    }
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
    calibrated, calibrator = subject_crossfit_platt(logits)
    calibrated["onset_probability"] = 1.0 / (1.0 + np.exp(-calibrated["onset_logit"]))
    calibrated["offset_probability"] = 1.0 / (1.0 + np.exp(-calibrated["offset_logit"]))
    calibrated["state_probability_derivative"] = calibrated.groupby(
        ["subject_key", "session_id"], sort=False
    )["state_probability"].diff().fillna(0.0)
    metrics = state_calibration_metrics(
        calibrated["state_target"].to_numpy(),
        calibrated["state_logit"].to_numpy(),
        calibrated["state_probability"].to_numpy(),
        low_threshold=float(config["decoder"]["low_threshold"]),
        bins=int(config["calibration"]["ece_bins"]),
    )
    calibration_checks = {
        "ece": metrics["ece"]
        <= float(config["promotion_gate"]["maximum_state_ece"]),
        "brier": metrics["brier"] < metrics["uncalibrated_brier"],
        "prevalence_ratio": metrics["mean_probability_to_prevalence"]
        <= float(config["promotion_gate"]["maximum_state_prevalence_ratio"]),
    }
    strict_gates = str(config["experiment"].get("ablation_id", "S4")) == "S4"
    if strict_gates and not all(calibration_checks.values()):
        failed = [name for name, passed in calibration_checks.items() if not passed]
        raise RuntimeError(f"State calibration gate failed: {failed}")
    durations = (
        inputs.events.loc[inputs.events["valid_duration"], "end_ms"].to_numpy(dtype=float)
        - inputs.events.loc[inputs.events["valid_duration"], "start_ms"].to_numpy(dtype=float)
    ) / 1000.0
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
    evaluable = inputs.events[
        inputs.events.get("evaluable", pd.Series(True, index=inputs.events.index)).fillna(False)
    ]
    ignored = inputs.events.loc[~inputs.events.index.isin(evaluable.index)]
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
        )
        state_only_accepted = _accepted_from_point(
            labeled, state_only_point, "generator_score"
        )
    else:
        state_only_point = {
            "acceptance_threshold": float("inf"),
            "nms_iou_threshold": float(config["calibration"]["nms_iou_thresholds"][0]),
            "f1": 0.0,
            "fp_per_hour": 0.0,
        }
        state_only_accepted = labeled.copy()
    state_only_hand = _hand_metrics(evaluable, _prediction_events(state_only_accepted))
    candidate_metrics.update(
        {
            "state_only_f1": float(state_only_point["f1"]),
            "state_only_fp_per_hour": float(state_only_point["fp_per_hour"]),
            "state_only_acceptance_threshold": float(
                state_only_point["acceptance_threshold"]
            ),
            "state_only_nms_iou_threshold": float(
                state_only_point["nms_iou_threshold"]
            ),
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
    outer_logits["state_probability"] = calibrator.transform(
        outer_logits["state_logit"].to_numpy()
    )
    outer_logits["onset_probability"] = 1.0 / (1.0 + np.exp(-outer_logits["onset_logit"]))
    outer_logits["offset_probability"] = 1.0 / (1.0 + np.exp(-outer_logits["offset_logit"]))
    outer_logits["state_probability_derivative"] = outer_logits.groupby(
        ["subject_key", "session_id"], sort=False
    )["state_probability"].diff().fillna(0.0)
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


def _slice_proposal_features(features, indices: np.ndarray):
    from bme_eating.models.event_verifier_v4 import ProposalFeatureBatchV4

    return ProposalFeatureBatchV4(
        proposal_ids=features.proposal_ids[indices],
        sequence=features.sequence[indices],
        sequence_mask=features.sequence_mask[indices],
        scalar=features.scalar[indices],
        event_target=features.event_target[indices] if features.event_target is not None else None,
        iou_target=features.iou_target[indices] if features.iou_target is not None else None,
        sample_weight=features.sample_weight[indices] if features.sample_weight is not None else None,
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
    )
    loader = DataLoader(dataset, batch_sampler=sampler)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["verifier"]["learning_rate"]),
        weight_decay=float(config["verifier"]["weight_decay"]),
    )
    for epoch in range(int(config["verifier"]["max_epochs"])):
        sampler.set_epoch(epoch)
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
    return model


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
    for batch in loader:
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
    for index in frame.sort_values(score_column, ascending=False).index:
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


def _prediction_events(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["subject_key", "start_ms", "end_ms", "proposal_id"])
    return frame.rename(
        columns={"coarse_start_ms": "start_ms", "coarse_end_ms": "end_ms"}
    )[["subject_key", "start_ms", "end_ms", "proposal_id"]]


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
) -> dict[str, float]:
    best: dict[str, float] | None = None
    values = scores[score_column].to_numpy(dtype=np.float64)
    for quantile in config["calibration"]["acceptance_quantiles"]:
        threshold = float(np.quantile(values, float(quantile)))
        for nms_threshold in config["calibration"]["nms_iou_thresholds"]:
            accepted = scores[scores[score_column] >= threshold]
            accepted = pd.concat(
                [
                    _nms(group, float(nms_threshold), score_column)
                    for _, group in accepted.groupby(["subject_key", "session_id"], sort=False)
                ],
                ignore_index=True,
            ) if len(accepted) else accepted
            metrics, _ = evaluate_events(events, _prediction_events(accepted), method="max_cardinality_iou")
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


def train_verifier_crossfit_v4(
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
    categories = classify_proposals(proposals)
    partitions = run.payload["stacking_partitions"]
    proposal_partition = proposals["subject_key"].astype(str).map(partitions).to_numpy(dtype=int)
    deep_event = np.zeros(len(proposals), dtype=np.float64)
    deep_iou = np.zeros(len(proposals), dtype=np.float64)
    logistic_score = np.zeros(len(proposals), dtype=np.float64)
    artifacts: list[Path] = []
    for partition in sorted(set(proposal_partition)):
        holdout_indices = np.flatnonzero(proposal_partition == partition)
        train_indices = np.flatnonzero(proposal_partition != partition)
        train_subjects = set(proposals.iloc[train_indices]["subject_key"].astype(str))
        holdout_subjects = set(proposals.iloc[holdout_indices]["subject_key"].astype(str))
        assert_disjoint_subjects(verifier_train=train_subjects, verifier_holdout=holdout_subjects)
        train_features = _slice_proposal_features(features, train_indices)
        holdout_features = _slice_proposal_features(features, holdout_indices)
        seed_predictions: list[tuple[np.ndarray, np.ndarray]] = []
        for seed in config["verifier"]["seeds"]:
            model = _train_verifier_model_v4(
                train_features,
                categories[train_indices],
                config,
                seed=int(seed) + int(partition) * 100,
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
    outer_seed_predictions: list[tuple[np.ndarray, np.ndarray]] = []
    for seed in config["verifier"]["seeds"]:
        model = _train_verifier_model_v4(
            features,
            categories,
            config,
            seed=int(seed) + 90_000,
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
    evaluable = inputs.events[
        inputs.events.get("evaluable", pd.Series(True, index=inputs.events.index)).fillna(False)
    ]
    deep_point = _best_verifier_operating_point(
        scored, evaluable, windows, "final_score", config
    )
    logistic_point = _best_verifier_operating_point(
        scored, evaluable, windows, "logistic_score", config
    )
    matching_sensitivity: dict[str, dict[str, float]] = {}
    for name, point, score_column in (
        ("deep", deep_point, "final_score"),
        ("logistic", logistic_point, "logistic_score"),
    ):
        accepted = _accepted_from_point(scored, point, score_column)
        greedy, _ = evaluate_events(
            evaluable, _prediction_events(accepted), method="greedy"
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
        raise RuntimeError(
            "Verifier ranking reverses between max-cardinality and greedy matching"
        )
    deep_passed = (
        deep_point["f1"] >= logistic_point["f1"] + float(
            config["promotion_gate"]["minimum_verifier_f1_improvement"]
        )
        or (
            deep_point["f1"] >= logistic_point["f1"] - float(
                config["promotion_gate"]["maximum_verifier_f1_drop"]
            )
            and deep_point["fp_per_hour"]
            <= logistic_point["fp_per_hour"]
            * (1.0 - float(config["promotion_gate"]["minimum_verifier_fp_reduction"]))
        )
    )
    preselection = {
        "verifier_kind": "deep" if deep_passed else "logistic",
        "deep": deep_point,
        "logistic": logistic_point,
        "deep_gate_passed": bool(deep_passed),
        "matching_sensitivity": matching_sensitivity,
    }
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
    scores: pd.DataFrame, point: dict[str, Any], score_column: str
) -> pd.DataFrame:
    selected = scores[scores[score_column] >= float(point["acceptance_threshold"])]
    if selected.empty:
        return selected.copy()
    return pd.concat(
        [
            _nms(group, float(point["nms_iou_threshold"]), score_column)
            for _, group in selected.groupby(["subject_key", "session_id"], sort=False)
        ],
        ignore_index=True,
    )


def _train_endpoint_model(
    features,
    config: dict[str, Any],
    *,
    seed: int,
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
    for _ in range(int(config["boundary"]["max_epochs"])):
        order = rng.permutation(len(features.sample_ids))
        model.train()
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
            }
            output = model(batch)
            loss, _ = endpoint_loss(output, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


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
    for first in range(0, len(features.sample_ids), batch_size):
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
        )
        end_offset, end_entropy = local_soft_argmax(
            output["end_logit"],
            end_grid,
            int(config["boundary"]["local_softargmax_radius_bins"]),
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
    for partition in sorted(set(partitions.values())):
        train = positive[
            positive["subject_key"].astype(str).map(partitions) != partition
        ].copy()
        holdout = accepted[
            accepted["subject_key"].astype(str).map(partitions) == partition
        ].copy()
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
        if boundary_range.clipped_fraction > float(
            config["boundary"]["maximum_clipped_fraction"]
        ):
            raise RuntimeError("Boundary residual clipping exceeds 5%; repair candidates first")
        augmented = augment_boundary_training_proposals(
            train,
            maximum_jitters_per_event=int(config["boundary"]["maximum_jitters_per_event"]),
            jitter_seconds=int(config["boundary"]["jitter_seconds"]),
        )
        train_features = build_endpoint_features(
            augmented,
            windows,
            [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
            boundary_range,
            config["boundary"],
        )
        holdout_features = build_endpoint_features(
            holdout,
            windows,
            [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
            boundary_range,
            config["boundary"],
        )
        seed_outputs: dict[int, tuple[np.ndarray, ...]] = {}
        for seed in config["boundary"]["seeds"]:
            seed = int(seed)
            model = _train_endpoint_model(
                train_features, config, seed=seed + partition * 100
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
                },
            )
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
    if final_range.clipped_fraction > float(
        config["boundary"]["maximum_clipped_fraction"]
    ):
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
    for seed in config["boundary"]["seeds"]:
        seed = int(seed)
        model = _train_endpoint_model(final_features, config, seed=seed + 90_000)
        checkpoint = run.root / "boundary" / f"final_seed_{seed}.pt"
        _save_torch_atomic(
            checkpoint,
            {
                "model": model.state_dict(),
                "input_dim": final_features.start_sequence.shape[-1],
                "config": config["boundary"],
                "range": final_range.__dict__,
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


def _hand_metrics(events: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    overall, matches = evaluate_events(
        events, predictions, method="max_cardinality_iou"
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
        relation_matches = matches[
            matches.get("hand_relation", pd.Series(dtype=str)) == relation
        ]
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
    events: pd.DataFrame, predictions: pd.DataFrame, windows: pd.DataFrame
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
        metrics, _ = evaluate_events(truth, predicted, method="max_cardinality_iou")
        rows.append(
            {
                "subject_key": subject,
                **metrics,
                **_hand_metrics(truth, predicted),
                "observed_hours": _observed_hours(observed),
                "fp_per_hour": metrics["false_positive"]
                / max(_observed_hours(observed), 1e-9),
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
    truth = inputs.events[
        inputs.events.get("evaluable", pd.Series(True, index=inputs.events.index)).fillna(False)
    ]
    coarse_predictions = _prediction_events(accepted)
    coarse_metrics, coarse_matches = evaluate_events(
        truth, coarse_predictions, method="max_cardinality_iou"
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
            candidates: list[
                tuple[float, dict[str, float], pd.DataFrame, pd.DataFrame, float]
            ] = []
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
                    truth, seed_predictions, method="max_cardinality_iou"
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
        seed_consistency_passed = seed_improvements >= min(
            2, len(config["boundary"]["seeds"])
        )
        boundary_passed = (
            refined_mae
            <= coarse_mae
            * (
                1.0
                - float(
                    config["promotion_gate"]["minimum_boundary_mae_improvement"]
                )
            )
            and refined_metrics["f1"]
            >= coarse_metrics["f1"] - float(config["promotion_gate"]["maximum_boundary_f1_drop"])
            and lost_fraction <= float(config["promotion_gate"]["maximum_tp_to_fp_fraction"])
            and seed_consistency_passed
        )
        if not boundary_passed:
            refined_predictions = coarse_predictions
            refined_metrics = coarse_metrics
            selected_entropy = None
            selected_boundary_seed = None
    greedy_metrics, _ = evaluate_events(truth, refined_predictions, method="greedy")
    metrics = {
        "coarse": {**coarse_metrics, **_hand_metrics(truth, coarse_predictions)},
        "selected": {**refined_metrics, **_hand_metrics(truth, refined_predictions)},
        "greedy_selected": greedy_metrics,
        "fp_per_hour": refined_metrics["false_positive"] / max(_observed_hours(windows), 1e-9),
        "boundary_gate_passed": bool(boundary_passed),
        "boundary_seed_consistency": boundary_seed_results,
    }
    selection = {
        "schema_version": 4,
        "verifier_kind": verifier_kind,
        "score_column": score_column,
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
    )
    if selection["boundary_enabled"]:
        boundary_scores = pd.read_parquet(run.root / "outer" / "boundary_scores.parquet")
        boundary_scores = _seeded_boundary_scores(
            boundary_scores, int(selection["boundary_seed"])
        )
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
    truth = outer_inputs.events[
        outer_inputs.events.get(
            "evaluable", pd.Series(True, index=outer_inputs.events.index)
        ).fillna(False)
    ]
    max_metrics, matches = evaluate_events(
        truth, predictions, method="max_cardinality_iou"
    )
    greedy_metrics, _ = evaluate_events(truth, predictions, method="greedy")
    windows = pd.read_parquet(run.root / "outer" / "window_predictions.parquet")
    outer_window_labels = outer_inputs.anchors[
        ["subject_key", "session_id", "timestamp_ms", "state_target"]
    ]
    calibrated_windows = windows.merge(
        outer_window_labels,
        on=["subject_key", "session_id", "timestamp_ms"],
        how="left",
        validate="one_to_one",
    )
    if calibrated_windows["state_target"].isna().any():
        raise RuntimeError("Outer state predictions failed to align with evaluation labels")
    outer_calibration = state_calibration_metrics(
        calibrated_windows["state_target"].to_numpy(),
        calibrated_windows["state_logit"].to_numpy(),
        calibrated_windows["state_probability"].to_numpy(),
        low_threshold=float(config["decoder"]["low_threshold"]),
        bins=int(config["calibration"]["ece_bins"]),
    )
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
        truth, state_only_predictions, method="max_cardinality_iou"
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
        "hand": _hand_metrics(truth, predictions),
        "fp_per_hour": max_metrics["false_positive"] / max(_observed_hours(windows), 1e-9),
        "state_calibration": outer_calibration,
        "candidate": outer_candidate_metrics,
        "state_only": {
            **state_only_metrics,
            **_hand_metrics(truth, state_only_predictions),
            "fp_per_hour": state_only_metrics["false_positive"]
            / max(_observed_hours(windows), 1e-9),
        },
    }
    matched_truth = set(matches.get("event_id", pd.Series(dtype=str)).astype(str))
    matched_prediction = set(
        matches.get("prediction_event_id", pd.Series(dtype=str)).astype(str)
    )
    failures = pd.concat(
        (
            truth.loc[~truth["event_id"].astype(str).isin(matched_truth)].assign(
                failure_type="false_negative"
            ),
            predictions.loc[
                ~predictions["proposal_id"].astype(str).isin(matched_prediction)
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
    state_only_subject_path = (
        run.root / "evaluation" / "state_only_per_subject_metrics.csv"
    )
    write_parquet_atomic(prediction_path, predictions)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_event = event_path.with_name(event_path.name + ".tmp")
    predictions.to_csv(temporary_event, index=False)
    temporary_event.replace(event_path)
    write_json_atomic(metric_path, metrics)
    temporary_failure = failure_path.with_name(failure_path.name + ".tmp")
    failures.to_csv(temporary_failure, index=False)
    temporary_failure.replace(failure_path)
    per_subject = _per_subject_metrics_v4(truth, predictions, windows)
    temporary_subject = per_subject_path.with_name(per_subject_path.name + ".tmp")
    per_subject.to_csv(temporary_subject, index=False)
    temporary_subject.replace(per_subject_path)
    state_only_subjects = _per_subject_metrics_v4(
        truth, state_only_predictions, windows
    )
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
    final_root = output_root / "final" / run_name
    manifest_path = final_root / "final_manifest.json"
    if manifest_path.is_file():
        if fresh:
            raise FileExistsError(f"V4 final run already exists: {final_root}")
        if not resume:
            raise RuntimeError("Existing V4 final run requires --resume")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative, expected in manifest.get("artifact_hashes", {}).items():
            if not (final_root / relative).is_file() or sha256_file(final_root / relative) != expected:
                raise RuntimeError(f"V4 final artifact changed: {relative}")
        return final_root
    if resume:
        raise FileNotFoundError("V4 final run does not exist; start with --fresh")
    final_root.mkdir(parents=True, exist_ok=True)
    experiment_root = output_root / "experiments" / run_name
    fold_roots = [experiment_root / f"fold_{fold}" for fold in range(5)]
    manifests = []
    for fold, root in enumerate(fold_roots):
        path = root / "run_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"V4 fold {fold} manifest is missing")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("stage") != "EVALUATED":
            raise RuntimeError(f"V4 fold {fold} must be EVALUATED before final training")
        HierarchicalRun(root, path, manifest).verify_artifacts()
        manifests.append(manifest)
    stress_path = experiment_root / "stress_gate.json"
    stress = json.loads(stress_path.read_text(encoding="utf-8"))
    if not bool(stress.get("passed", False)):
        raise RuntimeError("V4 frozen stress gate did not pass; final training is blocked")
    verify_gate_evidence(output_root.parent, stress)
    inputs = load_v4_inputs(
        config,
        input_root,
        fold=0,
        event_role="all",
        allow_outer_labels=True,
    )
    selection = json.loads(
        (fold_roots[0] / "selection" / "selected_pipeline.json").read_text(encoding="utf-8")
    )
    for root in fold_roots[1:]:
        current = json.loads(
            (root / "selection" / "selected_pipeline.json").read_text(encoding="utf-8")
        )
        if current["verifier_kind"] != selection["verifier_kind"]:
            raise RuntimeError("Final folds disagree on the frozen verifier kind")
    scaler, transformed = _fit_scaler_and_transform(
        inputs, set(inputs.anchors["subject_key"].astype(str))
    )
    normalization = compute_normalization(
        inputs.segments, set(inputs.anchors["subject_key"].astype(str))
    )
    epochs = []
    for root in fold_roots:
        payload = json.loads((root / "selection" / "state_epochs.json").read_text(encoding="utf-8"))
        epochs.append(int(payload["fixed_outer_epoch"]))
    fixed_epoch = int(np.median(epochs))
    full_dataset = _make_dataset(
        transformed,
        inputs,
        inputs.events,
        normalization,
        config,
        training=True,
        seed=int(config["training"]["random_seed"]),
    )
    artifacts: list[Path] = []
    for seed in config["final_training"]["state_seeds"]:
        torch.manual_seed(int(seed))
        model = build_state_model(config["model"])
        _train_state_epochs(
            model,
            full_dataset,
            config,
            epochs=fixed_epoch,
            seed=int(seed),
        )
        path = final_root / f"state_seed_{seed}.pt"
        _save_torch_atomic(
            path,
            {
                "model": model.state_dict(),
                "model_config": config["model"],
                "epochs": fixed_epoch,
                "training_subject_count": int(inputs.anchors["subject_key"].nunique()),
            },
        )
        artifacts.append(path)
    scaler_path = final_root / "statistics_scaler.json"
    normalization_path = final_root / "sensor_normalization.json"
    write_json_atomic(scaler_path, scaler.to_json())
    save_normalization(normalization, normalization_path)
    artifacts.extend((scaler_path, normalization_path))

    outer_logits_parts: list[pd.DataFrame] = []
    proposal_feature_parts: list[ProposalFeatureBatchV4] = []
    scored_parts: list[pd.DataFrame] = []
    boundary_positive_parts: list[pd.DataFrame] = []
    window_parts: list[pd.DataFrame] = []
    for fold, root in enumerate(fold_roots):
        outer_subjects = {
            subject for subject, subject_fold in inputs.subject_folds.items() if subject_fold == fold
        }
        anchors = inputs.anchors[
            inputs.anchors["subject_key"].astype(str).isin(outer_subjects)
        ][["subject_key", "session_id", "timestamp_ms", "state_target"]]
        logits = pd.read_parquet(root / "outer" / "window_logits.parquet").merge(
            anchors,
            on=["subject_key", "session_id", "timestamp_ms"],
            how="left",
            validate="one_to_one",
        )
        if logits["state_target"].isna().any():
            raise RuntimeError(f"Fold {fold} outer logits failed to align with final labels")
        outer_logits_parts.append(logits)
        windows = pd.read_parquet(root / "outer" / "window_predictions.parquet")
        proposals = pd.read_parquet(root / "outer" / "proposals.parquet")
        fold_truth = inputs.events[inputs.events["subject_key"].astype(str).isin(outer_subjects)]
        evaluable = fold_truth[
            fold_truth.get("evaluable", pd.Series(True, index=fold_truth.index)).fillna(False)
        ]
        labeled = label_event_candidates(proposals, evaluable, 0.25)
        proposal_feature_parts.append(
            build_proposal_features_v4(
                labeled,
                windows,
                [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
                config["verifier"],
            )
        )
        scored = pd.read_parquet(root / "outer" / "proposal_scores.parquet").merge(
            labeled[["proposal_id", "max_iou", "matched_event_id", "is_positive"]],
            on="proposal_id",
            how="left",
            validate="one_to_one",
        )
        scored_parts.append(scored)
        positive = scored[scored["max_iou"] > 0.25]
        boundary_positive_parts.append(_attach_truth_boundaries(positive, evaluable))
        window_parts.append(windows)
    final_logits = pd.concat(outer_logits_parts, ignore_index=True)
    from bme_eating.calibration_v4 import PlattCalibration

    state_calibration = PlattCalibration.fit(
        final_logits["state_logit"].to_numpy(), final_logits["state_target"].to_numpy()
    )
    state_calibration_path = final_root / "state_calibration.json"
    write_json_atomic(state_calibration_path, state_calibration.to_json())
    artifacts.append(state_calibration_path)
    durations = (
        inputs.events.loc[inputs.events["valid_duration"], "end_ms"].to_numpy(dtype=float)
        - inputs.events.loc[inputs.events["valid_duration"], "start_ms"].to_numpy(dtype=float)
    ) / 1000.0
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
    categories = classify_proposals(pd.concat([
        pd.read_parquet(root / "outer" / "proposals.parquet").assign(
            max_iou=part.iou_target,
        )
        for root, part in zip(fold_roots, proposal_feature_parts)
    ], ignore_index=True))
    logistic = LogisticScoreCombiner.fit(
        _verifier_matrix(verifier_features),
        verifier_features.event_target,
        sample_weight=verifier_features.sample_weight,
    )
    logistic_path = final_root / "logistic_verifier.json"
    write_json_atomic(logistic_path, logistic.to_json())
    artifacts.append(logistic_path)
    if selection["verifier_kind"] == "deep":
        verifier = _train_verifier_model_v4(
            verifier_features,
            categories,
            config,
            seed=int(config["verifier"]["seeds"][0]) + 120_000,
        )
        verifier_path = final_root / "verifier.pt"
        _save_torch_atomic(
            verifier_path,
            {
                "model": verifier.state_dict(),
                "sequence_dim": verifier_features.sequence.shape[-1],
                "scalar_dim": verifier_features.scalar.shape[-1],
                "config": config["verifier"],
            },
        )
        artifacts.append(verifier_path)
    calibration_frame = pd.concat(scored_parts, ignore_index=True)
    proposal_calibration = ProposalCalibrationV4.fit(calibration_frame)
    proposal_calibration_path = final_root / "proposal_calibration.json"
    write_json_atomic(proposal_calibration_path, proposal_calibration.to_json())
    artifacts.append(proposal_calibration_path)

    if selection["boundary_enabled"]:
        positive = pd.concat(boundary_positive_parts, ignore_index=True)
        independent_events = positive["matched_event_id"].nunique()
        if independent_events < int(config["boundary"]["minimum_independent_events"]):
            raise RuntimeError(
                "Final boundary training has fewer independent events than required"
            )
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
        if boundary_range.clipped_fraction > float(
            config["boundary"]["maximum_clipped_fraction"]
        ):
            raise RuntimeError(
                "Final OOF boundary residual clipping exceeds the configured limit"
            )
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
        boundary = _train_endpoint_model(
            boundary_features,
            config,
            seed=int(selection["boundary_seed"]) + 120_000,
        )
        boundary_path = final_root / "boundary.pt"
        range_path = final_root / "boundary_range.json"
        _save_torch_atomic(
            boundary_path,
            {
                "model": boundary.state_dict(),
                "input_dim": boundary_features.start_sequence.shape[-1],
                "config": config["boundary"],
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
        "state_seeds": [int(value) for value in config["final_training"]["state_seeds"]],
        "fixed_state_epoch": fixed_epoch,
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
