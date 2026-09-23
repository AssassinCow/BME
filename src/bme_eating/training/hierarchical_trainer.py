from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from bme_eating.calibration import crossfit_calibrate_scores, proposal_nms
from bme_eating.config import feature_artifact_name
from bme_eating.fusion import align_prediction_frames
from bme_eating.hierarchical_artifacts import (
    RUN_STAGES,
    HierarchicalRun,
    OuterLabelGuard,
    assert_disjoint_subjects,
    assert_oof_provenance,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
)
from bme_eating.metrics import evaluate_events, partition_evaluation_events
from bme_eating.models.boundary_refiner import (
    BoundaryRefiner,
    boundary_loss,
    build_boundary_targets,
    decode_boundaries,
)
from bme_eating.models.event_verifier import (
    EventVerifier,
    ProposalBatchSampler,
    ProposalFeatureBatch,
    ProposalTensorDataset,
    build_proposal_features,
    verifier_loss,
)
from bme_eating.models.xgb_baseline import assign_train_validation_test
from bme_eating.postprocess import probabilities_to_events
from bme_eating.proposals import (
    SOURCE_HINT,
    SOURCE_STATE,
    SOURCE_XGBOOST,
    duration_bounds_from_training_events,
    exclude_ignored_candidates,
    generate_event_candidates,
    label_event_candidates,
)
from bme_eating.training.dtp_trainer import seed_everything, train_dtp_fold

ALIGNMENT_KEYS = ["subject_key", "session_id", "timestamp_ms"]


@dataclass(frozen=True)
class HierarchicalInputs:
    anchors: pd.DataFrame
    segments: pd.DataFrame
    events: pd.DataFrame
    features: pd.DataFrame
    subject_folds: dict[str, int]


def _read_subject_folds(path: Path) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in json.loads(path.read_text(encoding="utf-8")).items()
    }


def load_hierarchical_inputs(
    config: dict[str, Any],
    input_root: Path,
    *,
    fold: int | None = None,
    event_role: str = "all",
) -> HierarchicalInputs:
    subject_folds = _read_subject_folds(input_root / "indices" / "subject_folds.json")
    if event_role not in {"all", "outer_train", "outer_test"}:
        raise ValueError(f"Unknown hierarchical event role: {event_role}")
    if event_role != "all" and fold is None:
        raise ValueError("Fold-scoped event loading requires an outer fold")
    if fold is not None and fold not in range(int(config["data"]["subject_folds"])):
        raise ValueError("Outer fold is outside the configured fold range")
    anchors = pd.read_parquet(input_root / "indices" / "anchors.parquet")
    segments = pd.read_parquet(input_root / "indices" / "segments.parquet")
    if event_role == "all":
        events = pd.read_parquet(input_root / "indices" / "events.parquet")
    else:
        selected_subjects = sorted(
            subject
            for subject, subject_fold in subject_folds.items()
            if (subject_fold != fold) == (event_role == "outer_train")
        )
        events = pd.read_parquet(
            input_root / "indices" / "events.parquet",
            filters=[("subject_key", "in", selected_subjects)],
        )
    feature_name = feature_artifact_name(config)
    stable_columns = list(config["hierarchical"]["stable_feature_columns"])
    feature_columns = [*ALIGNMENT_KEYS, *stable_columns]
    features = pd.read_parquet(
        input_root / "features" / f"{feature_name}.parquet", columns=feature_columns
    )
    if features.duplicated(ALIGNMENT_KEYS).any():
        raise RuntimeError("Stable feature rows contain duplicate timeline keys")
    return HierarchicalInputs(anchors, segments, events, features, subject_folds)


def _outer_subject_sets(inputs: HierarchicalInputs, fold: int) -> tuple[set[str], set[str]]:
    all_subjects = {str(value) for value in inputs.anchors["subject_key"].unique()}
    outer = {subject for subject in all_subjects if inputs.subject_folds[subject] == fold}
    train = all_subjects - outer
    assert_disjoint_subjects(outer_train=train, outer_test=outer)
    if not train or not outer:
        raise ValueError("Outer fold must have non-empty train and test subjects")
    return train, outer


def _sanitized_training_anchors(
    inputs: HierarchicalInputs,
    config: dict[str, Any],
    outer_subjects: set[str],
) -> pd.DataFrame:
    anchors = inputs.anchors.copy()
    if bool(config["model"].get("use_stable_state_features", False)):
        anchors = anchors.merge(
            inputs.features,
            on=ALIGNMENT_KEYS,
            how="left",
            validate="one_to_one",
        )
        stable_columns = list(config["model"]["stable_feature_columns"])
        if anchors[stable_columns].isna().all(axis=1).any():
            raise RuntimeError("Stable state features failed to align with anchors")
    outer_mask = anchors["subject_key"].astype(str).isin(outer_subjects)
    for target in ("state_target", "start_target", "end_target"):
        anchors.loc[outer_mask, target] = 0.0
    for mask in ("state_loss_mask", "start_loss_mask", "end_loss_mask"):
        anchors.loc[outer_mask, mask] = 0.0
    if "event_id" in anchors:
        anchors.loc[outer_mask, "event_id"] = None
    if "hand_relation" in anchors:
        anchors.loc[outer_mask, "hand_relation"] = "unknown"
    if "distance_to_event_seconds" in anchors:
        anchors.loc[outer_mask, "distance_to_event_seconds"] = np.inf
    return anchors


def _partition_signature(
    config: dict[str, Any], fold: int, partition: int, seed: int
) -> str:
    payload = {
        "protocol_version": 3,
        "outer_fold": fold,
        "inner_partition": partition,
        "model": config["model"],
        "training": {**config["training"], "random_seed": seed},
        "loss": config["loss"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _average_outer_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("Outer prediction ensemble is empty")
    reference = frames[0]
    aligned = [align_prediction_frames(reference, frame)[1] for frame in frames]
    output = aligned[0].copy()
    embedding_columns = [
        column for column in output.columns if column.startswith("state_embedding_")
    ]
    averaged = [
        "state_probability",
        "start_probability",
        "end_probability",
        "state_logit",
        "start_logit",
        "end_logit",
        "ppg_gate_mean",
        "ppg_gate_recent",
        "ppg_valid_fraction",
        "motion_valid_fraction",
        "missing_fraction",
    ]
    for column in averaged:
        if all(column in frame for frame in aligned):
            output[column] = np.stack(
                [frame[column].to_numpy(dtype=np.float64) for frame in aligned]
            ).mean(axis=0).astype(np.float32)
    output["representation_partition"] = 0
    output["ensemble_members"] = len(frames)
    for column in embedding_columns:
        output[column] = aligned[0][column].to_numpy(dtype=np.float32)
    return output


def _state_checkpoint_selector(
    validation_truth: pd.DataFrame,
    validation_ignore: pd.DataFrame,
    duration_bounds: tuple[float, float],
    config: dict[str, Any],
):
    def select(predictions: pd.DataFrame, epoch: int) -> dict[str, Any]:
        proposals = generate_event_candidates(
            predictions,
            config["proposals"],
            duration_bounds,
            split_role="state_checkpoint_validation",
        )
        proposals = label_event_candidates(
            proposals,
            validation_truth,
            float(config["postprocess"]["iou_threshold"]),
        )
        matched = set(
            proposals.loc[proposals["is_positive"], "matched_event_id"]
            .dropna()
            .astype(str)
        )
        candidate_recall = len(matched) / max(len(validation_truth), 1)
        if proposals.empty:
            metrics = {
                "f1": 0.0,
                "start_mae_seconds": float("nan"),
                "end_mae_seconds": float("nan"),
            }
        else:
            scored = proposals.copy()
            scored["final_score"] = scored["generator_score"]
            best_rank: tuple[float, ...] | None = None
            metrics = {}
            thresholds = np.unique(
                np.quantile(scored["final_score"], [0.5, 0.6, 0.7, 0.8, 0.9])
            )
            for threshold in thresholds:
                accepted = proposal_nms(
                    scored[scored["final_score"] >= threshold], 0.5
                )
                events = _events_from_proposals(accepted, refined=False)
                current, _ = evaluate_events(
                    validation_truth,
                    events,
                    iou_threshold=float(config["postprocess"]["iou_threshold"]),
                    method=str(config["postprocess"]["matching_method"]),
                    ignore=validation_ignore,
                )
                boundary_values = [
                    float(current[name])
                    for name in ("start_mae_seconds", "end_mae_seconds")
                    if np.isfinite(float(current[name]))
                ]
                boundary_mae = (
                    float(np.mean(boundary_values)) if boundary_values else math.inf
                )
                rank = (float(current["f1"]), -boundary_mae, float(threshold))
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    metrics = current
        boundary_values = [
            float(metrics[name])
            for name in ("start_mae_seconds", "end_mae_seconds")
            if np.isfinite(float(metrics[name]))
        ]
        boundary_mae = float(np.mean(boundary_values)) if boundary_values else math.inf
        return {
            "epoch": epoch,
            "candidate_recall": candidate_recall,
            "candidate_count": len(proposals),
            "metrics": metrics,
            "rank": [
                float(metrics["f1"]),
                candidate_recall,
                -boundary_mae,
                -float(epoch),
            ],
        }

    return select


def train_state_crossfit(
    run: HierarchicalRun,
    config: dict[str, Any],
    inputs: HierarchicalInputs,
    *,
    resume: bool,
) -> None:
    run.require_stage("CREATED")
    fold = int(run.payload["outer_fold"])
    outer_train_subjects, outer_subjects = _outer_subject_sets(inputs, fold)
    anchors = _sanitized_training_anchors(inputs, config, outer_subjects)
    outer_train_events = inputs.events[
        inputs.events["subject_key"].astype(str).isin(outer_train_subjects)
    ].copy()
    validation_frames: list[pd.DataFrame] = []
    outer_frames: list[pd.DataFrame] = []
    subject_partition: dict[str, int] = {}
    artifacts: list[Path] = []
    for partition in range(3):
        train, validation, test = assign_train_validation_test(
            anchors,
            inputs.subject_folds,
            fold,
            inner_validation_partition=partition,
        )
        validation_subjects = {str(value) for value in validation["subject_key"].unique()}
        train_subjects = {str(value) for value in train["subject_key"].unique()}
        test_subjects = {str(value) for value in test["subject_key"].unique()}
        if test_subjects != outer_subjects:
            raise RuntimeError("Inner state partition changed the outer-test subject set")
        if set(subject_partition) & validation_subjects:
            raise RuntimeError("State crossfit validation subjects overlap partitions")
        subject_partition.update({subject: partition for subject in validation_subjects})
        partition_dir = run.root / f"crossfit_{partition}" / "state"
        seed = int(config["training"]["random_seed"]) + partition
        partition_training = {**config["training"], "random_seed": seed}
        signature = _partition_signature(config, fold, partition, seed)
        validation_truth, validation_ignore = partition_evaluation_events(
            outer_train_events, validation_subjects
        )
        train_events = outer_train_events[
            outer_train_events["subject_key"].astype(str).isin(train_subjects)
        ]
        train_truth, _ = partition_evaluation_events(train_events, train_subjects)
        duration_bounds = duration_bounds_from_training_events(
            train_truth, config["proposals"]
        )
        required = (
            partition_dir / "best.pt",
            partition_dir / "best_validation_predictions.parquet",
            partition_dir / "outer_window_predictions.parquet",
            partition_dir / "metadata.json",
        )
        if not all(path.is_file() for path in required):
            existing = partition_dir.exists() and any(partition_dir.iterdir())
            last = partition_dir / "last.pt"
            if existing and not resume:
                raise RuntimeError(
                    f"State partition {partition} is incomplete; rerun with --resume"
                )
            train_dtp_fold(
                anchors,
                inputs.segments,
                outer_train_events,
                inputs.subject_folds,
                fold,
                config["model"],
                partition_training,
                config["loss"],
                config["postprocess"],
                partition_dir,
                last if resume and last.is_file() else None,
                validation_selector=_state_checkpoint_selector(
                    validation_truth,
                    validation_ignore,
                    duration_bounds,
                    config,
                ),
                selection_signature=signature,
                test_predictions_name="outer_window_predictions.parquet",
                inner_validation_partition=partition,
                checkpoint_selection_metric="event_f1",
            )
        validation_frame = pd.read_parquet(required[1])
        validation_frame["calibration_fold"] = partition
        validation_frames.append(validation_frame)
        outer_frames.append(pd.read_parquet(required[2]))
        artifacts.extend(path for path in partition_dir.rglob("*") if path.is_file())

    if set(subject_partition) != outer_train_subjects:
        raise RuntimeError("State crossfit OOF predictions do not cover outer-train subjects")
    oof = pd.concat(validation_frames, ignore_index=True).sort_values(ALIGNMENT_KEYS)
    assert_oof_provenance(oof, subject_partition)
    outer = _average_outer_predictions(outer_frames).sort_values(ALIGNMENT_KEYS)
    oof_path = run.root / "oof" / "window_predictions.parquet"
    outer_path = run.root / "outer" / "window_predictions.parquet"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    outer_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(oof_path, oof)
    write_parquet_atomic(outer_path, outer)
    write_json_atomic(run.root / "selection" / "subject_partition.json", subject_partition)
    run.transition(
        "STATE_COMPLETE",
        [*artifacts, oof_path, outer_path, run.root / "selection" / "subject_partition.json"],
        {
            "subjects": {
                "outer_train": sorted(outer_train_subjects),
                "outer_test": sorted(outer_subjects),
            },
            "state_crossfit_partitions": 3,
        },
    )


def reuse_hierarchical_state_artifacts(
    source: HierarchicalRun,
    target: HierarchicalRun,
    source_config: dict[str, Any],
    target_config: dict[str, Any],
) -> None:
    target.require_stage("CREATED")
    source.verify_artifacts()
    if RUN_STAGES.index(source.stage) < RUN_STAGES.index("STATE_COMPLETE"):
        raise RuntimeError("Source run has not completed hierarchical state training")
    for section in ("model", "training", "loss"):
        if source_config[section] != target_config[section]:
            raise RuntimeError(f"State cache reuse requires identical {section} configuration")
    if source.payload["input_hashes"] != target.payload["input_hashes"]:
        raise RuntimeError("State cache reuse requires identical v2 input hashes")
    paths: list[Path] = []
    relative_files = [
        path.relative_to(source.root)
        for partition in range(3)
        for path in (source.root / f"crossfit_{partition}" / "state").rglob("*")
        if path.is_file()
    ]
    relative_files.extend(
        Path(value)
        for value in (
            "oof/window_predictions.parquet",
            "outer/window_predictions.parquet",
            "selection/subject_partition.json",
        )
    )
    for relative in relative_files:
        source_path = source.root / relative
        target_path = target.root / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = target_path.with_name(target_path.name + ".tmp")
        shutil.copy2(source_path, temporary)
        temporary.replace(target_path)
        paths.append(target_path)
    target.transition(
        "STATE_COMPLETE",
        paths,
        {
            "subjects": source.payload["subjects"],
            "state_crossfit_partitions": source.payload["state_crossfit_partitions"],
            "reused_state_from": {
                "run_name": source.payload["run_name"],
                "outer_fold": source.payload["outer_fold"],
                "manifest_sha256": sha256_file(source.manifest_path),
            },
        },
    )


def _load_xgb_candidate_events(
    input_root: Path,
    fold: int,
    outer_train_subjects: set[str],
    outer_subjects: set[str],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    directory = input_root / "experiments" / "baseline" / f"fold_{fold}"
    required = (
        directory / "validation_predictions.parquet",
        directory / "test_predictions.parquet",
        directory / "selected_postprocess.json",
    )
    if not all(path.is_file() for path in required):
        missing = [path.name for path in required if not path.is_file()]
        raise FileNotFoundError(f"Frozen XGBoost candidate artifacts are missing: {missing}")
    manifest_path = directory / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Frozen XGBoost candidates require a run manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_hashes = manifest.get("artifact_hashes", {})
    for path in required:
        expected = artifact_hashes.get(path.name)
        if not expected or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen XGBoost artifact hash mismatch: {path.name}")
    parameters = json.loads(required[2].read_text(encoding="utf-8"))
    accepted_parameters = {
        key: value
        for key, value in parameters.items()
        if key
        in {
            "ema_half_life_seconds",
            "high_threshold",
            "low_threshold",
            "minimum_event_seconds",
            "merge_gap_seconds",
            "boundary_lookback_seconds",
            "detector_mode",
            "fast_ema_half_life_seconds",
            "slow_ema_half_life_seconds",
            "fast_high_threshold",
            "slow_high_threshold",
            "exit_threshold_ratio",
            "off_duration_seconds",
        }
    }
    validation_predictions = pd.read_parquet(required[0])
    test_predictions = pd.read_parquet(required[1])
    if set(validation_predictions["subject_key"].astype(str)) != outer_train_subjects:
        raise RuntimeError("Frozen XGBoost OOF predictions do not cover outer-train subjects")
    if set(test_predictions["subject_key"].astype(str)) != outer_subjects:
        raise RuntimeError("Frozen XGBoost predictions do not match outer-test subjects")
    validation = probabilities_to_events(validation_predictions, **accepted_parameters)
    test = probabilities_to_events(test_predictions, **accepted_parameters)
    return validation, test


def build_fold_candidates(
    run: HierarchicalRun,
    config: dict[str, Any],
    inputs: HierarchicalInputs,
    input_root: Path,
) -> None:
    run.require_stage("STATE_COMPLETE")
    fold = int(run.payload["outer_fold"])
    outer_train_subjects, outer_subjects = _outer_subject_sets(inputs, fold)
    guard = OuterLabelGuard(frozenset(outer_subjects), allow_outer_labels=False)
    training_events = guard.select_labels(inputs.events, outer_train_subjects)
    training_truth, training_ignore = partition_evaluation_events(
        training_events, outer_train_subjects
    )
    duration_bounds = duration_bounds_from_training_events(training_truth, config["proposals"])
    oof_windows = pd.read_parquet(run.root / "oof" / "window_predictions.parquet")
    outer_windows = pd.read_parquet(run.root / "outer" / "window_predictions.parquet")
    xgb_oof, xgb_outer = _load_xgb_candidate_events(
        input_root, fold, outer_train_subjects, outer_subjects
    )
    oof = generate_event_candidates(
        oof_windows,
        config["proposals"],
        duration_bounds,
        split_role="outer_train_oof",
        xgb_events=xgb_oof,
    )
    subject_partition = json.loads(
        (run.root / "selection" / "subject_partition.json").read_text(encoding="utf-8")
    )
    oof["calibration_fold"] = oof["subject_key"].map(subject_partition)
    if oof["calibration_fold"].isna().any():
        raise RuntimeError("OOF proposal subjects are missing calibration partitions")
    oof["calibration_fold"] = oof["calibration_fold"].astype(int)
    oof = label_event_candidates(
        oof, training_truth, float(config["postprocess"]["iou_threshold"])
    )
    oof = exclude_ignored_candidates(oof, training_ignore)
    outer = generate_event_candidates(
        outer_windows,
        config["proposals"],
        duration_bounds,
        split_role="outer_test_unlabeled",
        xgb_events=xgb_outer,
    )
    forbidden = {"max_iou", "matched_event_id", "is_positive", "negative_type"} & set(
        outer.columns
    )
    if forbidden:
        raise RuntimeError(f"Outer proposals unexpectedly contain labels: {sorted(forbidden)}")
    oof_path = run.root / "oof" / "proposals_labeled.parquet"
    outer_path = run.root / "outer" / "proposals.parquet"
    write_parquet_atomic(oof_path, oof)
    write_parquet_atomic(outer_path, outer)
    bounds_path = run.root / "selection" / "duration_bounds.json"
    write_json_atomic(
        bounds_path,
        {"minimum_seconds": duration_bounds[0], "maximum_seconds": duration_bounds[1]},
    )
    run.transition("PROPOSALS_COMPLETE", [oof_path, outer_path, bounds_path])


def _slice_feature_batch(features: ProposalFeatureBatch, indices: np.ndarray) -> ProposalFeatureBatch:
    return ProposalFeatureBatch(
        proposal_ids=features.proposal_ids[indices],
        sequence=features.sequence[indices],
        scalar=features.scalar[indices],
        event_target=(
            features.event_target[indices] if features.event_target is not None else None
        ),
        iou_target=features.iou_target[indices] if features.iou_target is not None else None,
    )


def _save_feature_batch(path: Path, features: ProposalFeatureBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            proposal_ids=features.proposal_ids.astype(str),
            sequence=features.sequence,
            scalar=features.scalar,
            event_target=features.event_target
            if features.event_target is not None
            else np.empty(0, dtype=np.float32),
            iou_target=features.iou_target
            if features.iou_target is not None
            else np.empty(0, dtype=np.float32),
        )
    temporary.replace(path)


def _proposal_batch_sampler(
    proposals: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
) -> ProposalBatchSampler:
    if "negative_type" not in proposals:
        raise ValueError("Verifier proposals are missing sampling categories")
    batch_size = int(config["batch_size"])
    steps_per_epoch = max(math.ceil(len(proposals) / batch_size), 20)
    return ProposalBatchSampler(
        proposals["negative_type"].to_numpy(dtype=str),
        batch_size=batch_size,
        ratios={
            str(name): float(value)
            for name, value in config["batch_composition"].items()
        },
        steps_per_epoch=steps_per_epoch,
        seed=seed,
    )


def _infer_verifier(
    model: EventVerifier,
    features: ProposalFeatureBatch,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        ProposalTensorDataset(features), batch_size=batch_size, shuffle=False, num_workers=0
    )
    event_logits: list[np.ndarray] = []
    iou_logits: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = model(
                {
                    "sequence": batch["sequence"].to(device),
                    "scalar": batch["scalar"].to(device),
                }
            )
            event_logits.append(output["event_logit"].cpu().numpy())
            iou_logits.append(output["iou_logit"].cpu().numpy())
    return np.concatenate(event_logits), np.concatenate(iou_logits)


def _verifier_event_f1(
    proposals: pd.DataFrame,
    event_logits: np.ndarray,
    truth: pd.DataFrame,
    ignore: pd.DataFrame,
    iou_threshold: float,
    matching_method: str,
) -> float:
    scored = proposals.copy()
    scored["final_score"] = 1.0 / (1.0 + np.exp(-event_logits))
    quantiles = np.unique(np.quantile(scored["final_score"], [0.5, 0.6, 0.7, 0.8, 0.9]))
    best = 0.0
    for threshold in quantiles:
        selected = proposal_nms(scored[scored["final_score"] >= threshold], 0.5)
        events = selected.rename(
            columns={
                "coarse_start_ms": "start_ms",
                "coarse_end_ms": "end_ms",
                "final_score": "score",
            }
        )[["subject_key", "start_ms", "end_ms", "score"]]
        metrics, _ = evaluate_events(
            truth,
            events,
            iou_threshold=iou_threshold,
            method=matching_method,
            ignore=ignore,
        )
        best = max(best, float(metrics["f1"]))
    return best


def _train_verifier_model(
    train_proposals: pd.DataFrame,
    train_features: ProposalFeatureBatch,
    validation_proposals: pd.DataFrame,
    validation_features: ProposalFeatureBatch,
    validation_truth: pd.DataFrame,
    validation_ignore: pd.DataFrame,
    config: dict[str, Any],
    output_path: Path,
    seed: int,
    iou_threshold: float,
    matching_method: str,
    *,
    resume: bool,
) -> EventVerifier:
    if train_features.event_target is None or train_features.iou_target is None:
        raise ValueError("Verifier training features require event and IoU targets")
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventVerifier(
        train_features.sequence.shape[-1], train_features.scalar.shape[-1], config
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    sampler = _proposal_batch_sampler(train_proposals, config, seed)
    best_f1 = -1.0
    patience = 0
    start_epoch = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_path = output_path.with_name(output_path.stem + ".last.pt")
    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if int(checkpoint.get("seed", -1)) != seed:
            raise RuntimeError("Verifier resume checkpoint seed mismatch")
        if bool(checkpoint.get("training_complete", False)):
            best = torch.load(output_path, map_location=device, weights_only=False)
            model.load_state_dict(best["model"])
            return model
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_f1 = float(checkpoint["best_f1"])
        patience = int(checkpoint["patience"])
        start_epoch = int(checkpoint["epoch"]) + 1
    for epoch in range(start_epoch, int(config["max_epochs"])):
        seed_everything(seed + epoch)
        sampler.set_epoch(epoch)
        loader = DataLoader(
            ProposalTensorDataset(train_features),
            batch_sampler=sampler,
            num_workers=0,
        )
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            moved = {
                "sequence": batch["sequence"].to(device),
                "scalar": batch["scalar"].to(device),
            }
            output = model(moved)
            loss, _ = verifier_loss(
                output,
                batch["event_target"].to(device),
                batch["iou_target"].to(device),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Verifier training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        validation_logits, _ = _infer_verifier(
            model, validation_features, device, int(config["batch_size"])
        )
        validation_f1 = _verifier_event_f1(
            validation_proposals,
            validation_logits,
            validation_truth,
            validation_ignore,
            iou_threshold,
            matching_method,
        )
        if validation_f1 > best_f1 + 1e-6:
            best_f1 = validation_f1
            patience = 0
            temporary = output_path.with_name(output_path.name + ".tmp")
            torch.save(
                {
                    "model": model.state_dict(),
                    "sequence_dim": train_features.sequence.shape[-1],
                    "scalar_dim": train_features.scalar.shape[-1],
                    "config": config,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_event_f1": validation_f1,
                },
                temporary,
            )
            temporary.replace(output_path)
        else:
            patience += 1
        temporary = last_path.with_name(last_path.name + ".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "seed": seed,
                "epoch": epoch,
                "best_f1": best_f1,
                "patience": patience,
                "training_complete": False,
            },
            temporary,
        )
        temporary.replace(last_path)
        if patience >= int(config["patience"]):
            break
    if not output_path.is_file():
        raise RuntimeError("Verifier training did not produce a best checkpoint")
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    last["training_complete"] = True
    temporary = last_path.with_name(last_path.name + ".tmp")
    torch.save(last, temporary)
    temporary.replace(last_path)
    checkpoint = torch.load(output_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model


def train_verifier_crossfit(
    run: HierarchicalRun,
    config: dict[str, Any],
    inputs: HierarchicalInputs,
    *,
    resume: bool = False,
) -> None:
    run.require_stage("PROPOSALS_COMPLETE")
    fold = int(run.payload["outer_fold"])
    outer_train_subjects, outer_subjects = _outer_subject_sets(inputs, fold)
    proposals = pd.read_parquet(run.root / "oof" / "proposals_labeled.parquet")
    outer_proposals = pd.read_parquet(run.root / "outer" / "proposals.parquet")
    windows = pd.read_parquet(run.root / "oof" / "window_predictions.parquet")
    outer_windows = pd.read_parquet(run.root / "outer" / "window_predictions.parquet")
    stable_columns = list(config["hierarchical"]["stable_feature_columns"])
    verifier_config = dict(config["verifier"])
    features = build_proposal_features(
        proposals, windows, inputs.features, stable_columns, verifier_config
    )
    outer_features = build_proposal_features(
        outer_proposals, outer_windows, inputs.features, stable_columns, verifier_config
    )
    feature_path = run.root / "oof" / "proposal_features.npz"
    outer_feature_path = run.root / "outer" / "proposal_features.npz"
    _save_feature_batch(feature_path, features)
    _save_feature_batch(outer_feature_path, outer_features)
    guard = OuterLabelGuard(frozenset(outer_subjects), allow_outer_labels=False)
    training_events = guard.select_labels(inputs.events, outer_train_subjects)

    oof_parts: list[pd.DataFrame] = []
    outer_event_logits: list[np.ndarray] = []
    outer_iou_logits: list[np.ndarray] = []
    outer_event_by_seed: dict[int, list[np.ndarray]] = {
        int(seed): [] for seed in config["verifier"]["seeds"]
    }
    outer_iou_by_seed: dict[int, list[np.ndarray]] = {
        int(seed): [] for seed in config["verifier"]["seeds"]
    }
    model_paths: list[Path] = []
    seeds = [int(value) for value in config["verifier"]["seeds"]]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Verifier requires exactly three distinct registered seeds")
    for partition in range(int(config["hierarchical"]["verifier_crossfit_partitions"])):
        train_indices = np.flatnonzero(proposals["calibration_fold"].to_numpy() != partition)
        validation_indices = np.flatnonzero(
            proposals["calibration_fold"].to_numpy() == partition
        )
        train_proposals = proposals.iloc[train_indices].reset_index(drop=True)
        validation_proposals = proposals.iloc[validation_indices].reset_index(drop=True)
        assert_disjoint_subjects(
            verifier_train=set(train_proposals["subject_key"].astype(str)),
            verifier_validation=set(validation_proposals["subject_key"].astype(str)),
        )
        train_features = _slice_feature_batch(features, train_indices)
        validation_features = _slice_feature_batch(features, validation_indices)
        validation_subjects = set(validation_proposals["subject_key"].astype(str))
        validation_truth, validation_ignore = partition_evaluation_events(
            training_events, validation_subjects
        )
        validation_seed_events: list[np.ndarray] = []
        validation_seed_ious: list[np.ndarray] = []
        outer_seed_events: list[np.ndarray] = []
        outer_seed_ious: list[np.ndarray] = []
        for seed in seeds:
            model_path = (
                run.root
                / "models"
                / f"verifier_crossfit_{partition}_seed_{seed}.pt"
            )
            model = _train_verifier_model(
                train_proposals,
                train_features,
                validation_proposals,
                validation_features,
                validation_truth,
                validation_ignore,
                verifier_config,
                model_path,
                seed + partition * 10_000,
                float(config["postprocess"]["iou_threshold"]),
                str(config["postprocess"]["matching_method"]),
                resume=resume,
            )
            device = next(model.parameters()).device
            validation_event, validation_iou = _infer_verifier(
                model, validation_features, device, int(verifier_config["batch_size"])
            )
            validation_seed_events.append(validation_event)
            validation_seed_ious.append(validation_iou)
            outer_event, outer_iou = _infer_verifier(
                model, outer_features, device, int(verifier_config["batch_size"])
            )
            outer_seed_events.append(outer_event)
            outer_seed_ious.append(outer_iou)
            model_paths.extend((model_path, model_path.with_name(model_path.stem + ".last.pt")))
        validation_event = np.mean(np.stack(validation_seed_events), axis=0)
        validation_iou = np.mean(np.stack(validation_seed_ious), axis=0)
        part = validation_proposals.copy()
        part["event_logit"] = validation_event
        part["iou_logit"] = validation_iou
        for seed, event_values, iou_values in zip(
            seeds, validation_seed_events, validation_seed_ious
        ):
            part[f"event_logit_seed_{seed}"] = event_values
            part[f"iou_logit_seed_{seed}"] = iou_values
        part["state_score"] = validation_features.sequence[:, :, 0].mean(axis=1)
        oof_parts.append(part)
        outer_event_logits.append(np.mean(np.stack(outer_seed_events), axis=0))
        outer_iou_logits.append(np.mean(np.stack(outer_seed_ious), axis=0))
        for seed, event_values, iou_values in zip(
            seeds, outer_seed_events, outer_seed_ious
        ):
            outer_event_by_seed[seed].append(event_values)
            outer_iou_by_seed[seed].append(iou_values)

    oof_scores = pd.concat(oof_parts, ignore_index=True).sort_values("proposal_id")
    assert_oof_provenance(
        oof_scores,
        {
            str(subject): int(partition)
            for subject, partition in proposals[
                ["subject_key", "calibration_fold"]
            ].drop_duplicates().itertuples(index=False)
        },
    )
    calibrated_oof, final_calibration = crossfit_calibrate_scores(oof_scores)
    outer_scores = outer_proposals.copy()
    outer_scores["event_logit"] = np.mean(np.stack(outer_event_logits), axis=0)
    outer_scores["iou_logit"] = np.mean(np.stack(outer_iou_logits), axis=0)
    for seed in seeds:
        outer_scores[f"event_logit_seed_{seed}"] = np.mean(
            np.stack(outer_event_by_seed[seed]), axis=0
        )
        outer_scores[f"iou_logit_seed_{seed}"] = np.mean(
            np.stack(outer_iou_by_seed[seed]), axis=0
        )
    outer_scores["predicted_iou"] = 1.0 / (1.0 + np.exp(-outer_scores["iou_logit"]))
    outer_scores["state_score"] = outer_features.sequence[:, :, 0].mean(axis=1)
    outer_scores = final_calibration.apply(outer_scores)
    oof_path = run.root / "oof" / "proposal_scores.parquet"
    outer_path = run.root / "outer" / "proposal_scores.parquet"
    calibration_path = run.root / "calibration.json"
    write_parquet_atomic(oof_path, calibrated_oof)
    write_parquet_atomic(outer_path, outer_scores)
    write_json_atomic(calibration_path, final_calibration.to_json())
    run.transition(
        "VERIFIER_COMPLETE",
        [feature_path, outer_feature_path, *model_paths, oof_path, outer_path, calibration_path],
    )


class _BoundaryDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        features: ProposalFeatureBatch,
        targets: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return len(self.features.proposal_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item: dict[str, torch.Tensor | str] = {
            "proposal_id": str(self.features.proposal_ids[index]),
            "sequence": torch.from_numpy(self.features.sequence[index]),
            "scalar": torch.from_numpy(self.features.scalar[index]),
        }
        if self.targets is not None:
            item.update(
                {
                    "start_distribution": torch.from_numpy(
                        self.targets["start_distribution"][index]
                    ),
                    "end_distribution": torch.from_numpy(
                        self.targets["end_distribution"][index]
                    ),
                    "start_fine": torch.tensor(
                        float(self.targets["start_fine"][index]), dtype=torch.float32
                    ),
                    "end_fine": torch.tensor(
                        float(self.targets["end_fine"][index]), dtype=torch.float32
                    ),
                }
            )
        return item


def _infer_boundary(
    model: BoundaryRefiner,
    features: ProposalFeatureBatch,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(_BoundaryDataset(features), batch_size=batch_size, shuffle=False)
    values: dict[str, list[np.ndarray]] = {
        "start_distribution_logit": [],
        "end_distribution_logit": [],
        "start_fine_seconds": [],
        "end_fine_seconds": [],
    }
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = model(
                {
                    "sequence": batch["sequence"].to(device),
                    "scalar": batch["scalar"].to(device),
                }
            )
            for name, parts in values.items():
                parts.append(output[name].cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in values.items()}


def _train_boundary_model(
    train_features: ProposalFeatureBatch,
    train_targets: dict[str, np.ndarray],
    validation_features: ProposalFeatureBatch,
    validation_targets: dict[str, np.ndarray],
    config: dict[str, Any],
    output_path: Path,
    seed: int,
    *,
    resume: bool,
) -> BoundaryRefiner:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoundaryRefiner(
        train_features.sequence.shape[-1], train_features.scalar.shape[-1], config
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    validation_loader = DataLoader(
        _BoundaryDataset(validation_features, validation_targets),
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )
    best_loss = math.inf
    patience = 0
    start_epoch = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_path = output_path.with_name(output_path.stem + ".last.pt")
    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if int(checkpoint.get("seed", -1)) != seed:
            raise RuntimeError("Boundary resume checkpoint seed mismatch")
        if bool(checkpoint.get("training_complete", False)):
            best = torch.load(output_path, map_location=device, weights_only=False)
            model.load_state_dict(best["model"])
            return model
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_loss = float(checkpoint["best_loss"])
        patience = int(checkpoint["patience"])
        start_epoch = int(checkpoint["epoch"]) + 1
    for epoch in range(start_epoch, int(config["max_epochs"])):
        seed_everything(seed + epoch)
        train_loader = DataLoader(
            _BoundaryDataset(train_features, train_targets),
            batch_size=int(config["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(
                {
                    "sequence": batch["sequence"].to(device),
                    "scalar": batch["scalar"].to(device),
                }
            )
            loss, _ = boundary_loss(
                output,
                batch["start_distribution"].to(device),
                batch["end_distribution"].to(device),
                batch["start_fine"].to(device),
                batch["end_fine"].to(device),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Boundary training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        losses: list[float] = []
        with torch.inference_mode():
            for batch in validation_loader:
                output = model(
                    {
                        "sequence": batch["sequence"].to(device),
                        "scalar": batch["scalar"].to(device),
                    }
                )
                loss, _ = boundary_loss(
                    output,
                    batch["start_distribution"].to(device),
                    batch["end_distribution"].to(device),
                    batch["start_fine"].to(device),
                    batch["end_fine"].to(device),
                )
                losses.append(float(loss.cpu()))
        validation_loss = float(np.mean(losses))
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            patience = 0
            temporary = output_path.with_name(output_path.name + ".tmp")
            torch.save(
                {
                    "model": model.state_dict(),
                    "sequence_dim": train_features.sequence.shape[-1],
                    "scalar_dim": train_features.scalar.shape[-1],
                    "config": config,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                },
                temporary,
            )
            temporary.replace(output_path)
        else:
            patience += 1
        temporary = last_path.with_name(last_path.name + ".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "seed": seed,
                "epoch": epoch,
                "best_loss": best_loss,
                "patience": patience,
                "training_complete": False,
            },
            temporary,
        )
        temporary.replace(last_path)
        if patience >= int(config["patience"]):
            break
    if not output_path.is_file():
        raise RuntimeError("Boundary training did not produce a best checkpoint")
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    last["training_complete"] = True
    temporary = last_path.with_name(last_path.name + ".tmp")
    torch.save(last, temporary)
    temporary.replace(last_path)
    checkpoint = torch.load(output_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model


def _write_boundary_outputs(
    path: Path,
    proposal_ids: np.ndarray,
    output: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, proposal_ids=proposal_ids.astype(str), **output)
    temporary.replace(path)


def train_boundary_crossfit(
    run: HierarchicalRun,
    config: dict[str, Any],
    inputs: HierarchicalInputs,
    *,
    resume: bool = False,
) -> None:
    run.require_stage("VERIFIER_COMPLETE")
    fold = int(run.payload["outer_fold"])
    outer_train_subjects, outer_subjects = _outer_subject_sets(inputs, fold)
    proposals = pd.read_parquet(run.root / "oof" / "proposals_labeled.parquet")
    outer_proposals = pd.read_parquet(run.root / "outer" / "proposals.parquet")
    windows = pd.read_parquet(run.root / "oof" / "window_predictions.parquet")
    outer_windows = pd.read_parquet(run.root / "outer" / "window_predictions.parquet")
    stable_columns = list(config["hierarchical"]["stable_feature_columns"])
    features = build_proposal_features(
        proposals, windows, inputs.features, stable_columns, config["verifier"]
    )
    outer_features = build_proposal_features(
        outer_proposals, outer_windows, inputs.features, stable_columns, config["verifier"]
    )
    guard = OuterLabelGuard(frozenset(outer_subjects), allow_outer_labels=False)
    training_events = guard.select_labels(inputs.events, outer_train_subjects)
    positive = proposals["is_positive"].to_numpy(dtype=bool)
    if not positive.any():
        raise RuntimeError("Boundary training has no positive OOF proposals")

    oof_output_parts: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    outer_outputs: list[dict[str, np.ndarray]] = []
    oof_seed_parts: dict[int, list[tuple[np.ndarray, dict[str, np.ndarray]]]] = {}
    outer_seed_parts: dict[int, list[dict[str, np.ndarray]]] = {}
    model_paths: list[Path] = []
    partitions = int(config["hierarchical"]["boundary_crossfit_partitions"])
    seeds = [int(value) for value in config["boundary"]["seeds"]]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Boundary refiner requires exactly three distinct registered seeds")
    oof_seed_parts = {seed: [] for seed in seeds}
    outer_seed_parts = {seed: [] for seed in seeds}
    for partition in range(partitions):
        train_indices = np.flatnonzero(
            positive & (proposals["calibration_fold"].to_numpy() != partition)
        )
        validation_indices = np.flatnonzero(
            positive & (proposals["calibration_fold"].to_numpy() == partition)
        )
        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise RuntimeError("Every boundary crossfit partition needs positive proposals")
        train_proposals = proposals.iloc[train_indices].reset_index(drop=True)
        validation_proposals = proposals.iloc[validation_indices].reset_index(drop=True)
        assert_disjoint_subjects(
            boundary_train=set(train_proposals["subject_key"].astype(str)),
            boundary_validation=set(validation_proposals["subject_key"].astype(str)),
        )
        train_features = _slice_feature_batch(features, train_indices)
        validation_features = _slice_feature_batch(features, validation_indices)
        train_targets = build_boundary_targets(
            train_proposals, training_events, config["boundary"]
        )
        validation_targets = build_boundary_targets(
            validation_proposals, training_events, config["boundary"]
        )
        heldout_indices = np.flatnonzero(
            proposals["calibration_fold"].to_numpy() == partition
        )
        heldout_features = _slice_feature_batch(features, heldout_indices)
        heldout_seed_outputs: list[dict[str, np.ndarray]] = []
        outer_seed_outputs: list[dict[str, np.ndarray]] = []
        for seed in seeds:
            model_path = (
                run.root
                / "models"
                / f"boundary_crossfit_{partition}_seed_{seed}.pt"
            )
            model = _train_boundary_model(
                train_features,
                train_targets,
                validation_features,
                validation_targets,
                config["boundary"],
                model_path,
                seed + partition * 10_000,
                resume=resume,
            )
            device = next(model.parameters()).device
            heldout_seed_outputs.append(
                _infer_boundary(
                    model,
                    heldout_features,
                    device,
                    int(config["boundary"]["batch_size"]),
                )
            )
            outer_seed_outputs.append(
                _infer_boundary(
                    model,
                    outer_features,
                    device,
                    int(config["boundary"]["batch_size"]),
                )
            )
            oof_seed_parts[seed].append(
                (features.proposal_ids[heldout_indices], heldout_seed_outputs[-1])
            )
            outer_seed_parts[seed].append(outer_seed_outputs[-1])
            model_paths.extend((model_path, model_path.with_name(model_path.stem + ".last.pt")))
        heldout_output = {
            name: np.mean(np.stack([value[name] for value in heldout_seed_outputs]), axis=0)
            for name in heldout_seed_outputs[0]
        }
        oof_output_parts.append(
            (
                features.proposal_ids[heldout_indices],
                heldout_output,
            )
        )
        outer_outputs.append(
            {
                name: np.mean(
                    np.stack([value[name] for value in outer_seed_outputs]), axis=0
                )
                for name in outer_seed_outputs[0]
            }
        )

    combined_ids = np.concatenate([identifiers for identifiers, _ in oof_output_parts])
    combined = {
        name: np.concatenate([output[name] for _, output in oof_output_parts])
        for name in oof_output_parts[0][1]
    }
    order = np.argsort(combined_ids.astype(str))
    combined_ids = combined_ids[order]
    combined = {name: value[order] for name, value in combined.items()}
    for seed in seeds:
        seed_ids = np.concatenate([value[0] for value in oof_seed_parts[seed]])
        seed_order = np.argsort(seed_ids.astype(str))
        if not np.array_equal(seed_ids[seed_order], combined_ids):
            raise RuntimeError("Boundary seed outputs do not align with ensemble OOF outputs")
        for name in oof_seed_parts[seed][0][1]:
            values = np.concatenate([value[1][name] for value in oof_seed_parts[seed]])
            combined[f"{name}_seed_{seed}"] = values[seed_order]
    outer_combined = {
        name: np.mean(np.stack([output[name] for output in outer_outputs]), axis=0)
        for name in outer_outputs[0]
    }
    for seed in seeds:
        for name in outer_seed_parts[seed][0]:
            outer_combined[f"{name}_seed_{seed}"] = np.mean(
                np.stack([value[name] for value in outer_seed_parts[seed]]), axis=0
            )
    oof_path = run.root / "oof" / "boundary_outputs.npz"
    outer_path = run.root / "outer" / "boundary_outputs.npz"
    _write_boundary_outputs(oof_path, combined_ids, combined)
    _write_boundary_outputs(outer_path, outer_features.proposal_ids, outer_combined)
    run.transition("BOUNDARY_COMPLETE", [*model_paths, oof_path, outer_path])


def _load_boundary_outputs(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path) as archive:
        identifiers = archive["proposal_ids"].astype(str)
        output = {
            name: archive[name]
            for name in (
                "start_distribution_logit",
                "end_distribution_logit",
                "start_fine_seconds",
                "end_fine_seconds",
            )
        }
    return identifiers, output


def _load_boundary_seed_outputs(
    path: Path,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path) as archive:
        identifiers = archive["proposal_ids"].astype(str)
        output = {
            name: archive[f"{name}_seed_{seed}"]
            for name in (
                "start_distribution_logit",
                "end_distribution_logit",
                "start_fine_seconds",
                "end_fine_seconds",
            )
        }
    return identifiers, output


def _select_boundary_output(
    selected_ids: pd.Series,
    all_ids: np.ndarray,
    output: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    lookup = {str(value): index for index, value in enumerate(all_ids)}
    try:
        indices = np.asarray([lookup[str(value)] for value in selected_ids], dtype=np.int64)
    except KeyError as error:
        raise RuntimeError(f"Boundary output is missing proposal {error.args[0]}") from error
    return {name: torch.from_numpy(value[indices]) for name, value in output.items()}


def _observed_hours(windows: pd.DataFrame) -> float:
    duration_ms = windows.groupby(["subject_key", "session_id"])["timestamp_ms"].agg(
        lambda values: max(0, int(values.max()) - int(values.min()))
    )
    return max(float(duration_ms.sum()) / 3_600_000, 1e-9)


def _events_from_proposals(frame: pd.DataFrame, refined: bool) -> pd.DataFrame:
    start = "refined_start_ms" if refined else "coarse_start_ms"
    end = "refined_end_ms" if refined else "coarse_end_ms"
    return frame.rename(
        columns={start: "start_ms", end: "end_ms", "final_score": "score"}
    )[["proposal_id", "subject_key", "session_id", "start_ms", "end_ms", "score"]]


def _metrics_with_diagnostics(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    ignore: pd.DataFrame,
    windows: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, float], pd.DataFrame]:
    metrics, matches = evaluate_events(
        truth,
        prediction,
        iou_threshold=float(config["postprocess"]["iou_threshold"]),
        method=str(config["postprocess"]["matching_method"]),
        ignore=ignore,
    )
    metrics["false_positives_per_observed_hour"] = float(
        metrics["false_positive"] / _observed_hours(windows)
    )
    boundary = [
        float(metrics[name])
        for name in ("start_mae_seconds", "end_mae_seconds")
        if np.isfinite(float(metrics[name]))
    ]
    metrics["boundary_mae_seconds"] = float(np.mean(boundary)) if boundary else math.inf
    truth_relation = (
        truth["hand_relation"].fillna("unknown").astype(str)
        if "hand_relation" in truth
        else pd.Series("unknown", index=truth.index, dtype=str)
    )
    match_relation = (
        matches["hand_relation"].fillna("unknown").astype(str)
        if "hand_relation" in matches
        else pd.Series("unknown", index=matches.index, dtype=str)
    )
    for relation in ("same", "different"):
        total = int((truth_relation == relation).sum())
        relation_matches = matches[match_relation == relation]
        matched = len(relation_matches)
        metrics[f"{relation}_sensitivity"] = matched / total if total else float("nan")
        metrics[f"{relation}_start_mae_seconds"] = (
            float(relation_matches["start_absolute_error_ms"].mean() / 1000.0)
            if matched
            else float("nan")
        )
        metrics[f"{relation}_end_mae_seconds"] = (
            float(relation_matches["end_absolute_error_ms"].mean() / 1000.0)
            if matched
            else float("nan")
        )
    return metrics, matches


def _apply_selected_pipeline(
    scores: pd.DataFrame,
    boundary_ids: np.ndarray,
    boundary_output: dict[str, np.ndarray],
    windows: pd.DataFrame,
    config: dict[str, Any],
    selection: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = _accepted_pipeline_records(
        scores,
        boundary_ids,
        boundary_output,
        windows,
        config,
        selection,
    )
    coarse = _events_from_proposals(accepted, refined=False)
    return coarse, _events_from_proposals(accepted, refined=True)


def _accepted_pipeline_records(
    scores: pd.DataFrame,
    boundary_ids: np.ndarray,
    boundary_output: dict[str, np.ndarray],
    windows: pd.DataFrame,
    config: dict[str, Any],
    selection: dict[str, Any],
) -> pd.DataFrame:
    score_column = str(selection.get("score_column", "final_score"))
    if score_column not in scores:
        raise ValueError(f"Selected score column is unavailable: {score_column}")
    working = scores.copy()
    if score_column != "final_score":
        working["combined_final_score"] = working["final_score"]
        working["final_score"] = working[score_column]
    accepted = working[
        working["final_score"] >= float(selection["acceptance_threshold"])
    ].copy()
    accepted = proposal_nms(accepted, float(selection["nms_iou_threshold"]))
    if accepted.empty or not bool(selection.get("use_boundary", True)):
        fallback = accepted.copy()
        fallback["refined_start_ms"] = fallback["coarse_start_ms"]
        fallback["refined_end_ms"] = fallback["coarse_end_ms"]
        fallback["start_entropy"] = 1.0
        fallback["end_entropy"] = 1.0
        fallback["boundary_fallback"] = True
        return fallback
    selected_output = _select_boundary_output(
        accepted["proposal_id"], boundary_ids, boundary_output
    )
    observation_end = {
        (str(subject), str(session)): int(group["timestamp_ms"].max())
        for (subject, session), group in windows.groupby(
            ["subject_key", "session_id"], sort=False
        )
    }
    refined = decode_boundaries(
        accepted,
        selected_output,
        config["boundary"],
        float(selection["boundary_entropy_threshold"]),
        observation_end,
    )
    if set(refined["proposal_id"]) != set(accepted["proposal_id"]):
        raise RuntimeError("Boundary refinement changed accepted proposal identities")
    return refined


def _proposal_score_audit_frame(
    scores: pd.DataFrame,
    accepted: pd.DataFrame,
) -> pd.DataFrame:
    output = scores.copy()
    output["accepted"] = False
    output["refined_start_ms"] = output["coarse_start_ms"]
    output["refined_end_ms"] = output["coarse_end_ms"]
    output["start_entropy"] = np.nan
    output["end_entropy"] = np.nan
    output["boundary_fallback"] = True
    if accepted.empty:
        return output
    columns = [
        "proposal_id",
        "final_score",
        "refined_start_ms",
        "refined_end_ms",
        "start_entropy",
        "end_entropy",
        "boundary_fallback",
    ]
    accepted_values = accepted[columns].set_index("proposal_id")
    indices = output["proposal_id"].isin(accepted_values.index)
    output.loc[indices, "accepted"] = True
    for column in columns[1:]:
        output.loc[indices, column] = output.loc[indices, "proposal_id"].map(
            accepted_values[column]
        )
    return output


def _registered_baseline_ablations(
    config: dict[str, Any],
    fold: int,
    outer_train_subjects: set[str],
    outer_subjects: set[str],
    truth: pd.DataFrame,
    ignore: pd.DataFrame,
    windows: pd.DataFrame,
) -> list[dict[str, Any]]:
    input_root = Path(config["_input_artifact_root"])
    xgb_oof, _ = _load_xgb_candidate_events(
        input_root, fold, outer_train_subjects, outer_subjects
    )
    xgb_metrics, _ = _metrics_with_diagnostics(
        truth,
        xgb_oof if xgb_oof is not None else pd.DataFrame(),
        ignore,
        windows,
        config,
    )
    rows: list[dict[str, Any]] = [
        {
            "ablation": "A0_frozen_xgboost",
            "eligible": True,
            "evidence_scope": "outer_train_subject_oof_frozen_v2",
            **{f"refined_{key}": value for key, value in xgb_metrics.items()},
        }
    ]
    if fold != 0:
        rows.append(
            {
                "ablation": "A1_old_hard_dyadic",
                "eligible": False,
                "evidence_scope": "registered_fold_0_history_only",
                "reason": "The registered old hard-dyadic diagnostic exists only for fold 0",
            }
        )
        return rows
    historical = config["historical_baselines"]
    source_root = (
        input_root
        / "experiments"
        / str(historical["hard_dyadic_experiment"])
        / "fold_0"
    )
    manifest_path = source_root / "run_manifest.json"
    metrics_path = source_root / str(historical["hard_dyadic_metrics_file"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("artifacts", {}).get(metrics_path.name)
    if not expected or sha256_file(metrics_path) != expected:
        raise RuntimeError("Registered hard-dyadic diagnostic hash mismatch")
    diagnostics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = dict(
        diagnostics["oof_event_metrics_in_sample_postprocess_selection"][
            str(config["postprocess"]["matching_method"])
        ]
    )
    hand = diagnostics["oof_event_metrics_in_sample_postprocess_selection"][
        "hand_relation"
    ]
    metrics["same_sensitivity"] = hand["same"]["sensitivity"]
    metrics["different_sensitivity"] = hand["different"]["sensitivity"]
    rows.append(
        {
            "ablation": "A1_old_hard_dyadic",
            "eligible": True,
            "evidence_scope": "historical_fold_0_oof_postprocess_diagnostic",
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_metrics_sha256": expected,
            **{f"refined_{key}": value for key, value in metrics.items()},
        }
    )
    return rows


def select_hierarchical_pipeline(
    run: HierarchicalRun,
    config: dict[str, Any],
    inputs: HierarchicalInputs,
) -> None:
    run.require_stage("BOUNDARY_COMPLETE")
    fold = int(run.payload["outer_fold"])
    outer_train_subjects, outer_subjects = _outer_subject_sets(inputs, fold)
    guard = OuterLabelGuard(frozenset(outer_subjects), allow_outer_labels=False)
    training_events = guard.select_labels(inputs.events, outer_train_subjects)
    truth, ignore = partition_evaluation_events(training_events, outer_train_subjects)
    scores = pd.read_parquet(run.root / "oof" / "proposal_scores.parquet")
    windows = pd.read_parquet(run.root / "oof" / "window_predictions.parquet")
    boundary_ids, boundary_output = _load_boundary_outputs(
        run.root / "oof" / "boundary_outputs.npz"
    )
    source_mask = scores["source_mask"].to_numpy(dtype=np.int64)
    state_only = scores[(source_mask & SOURCE_STATE) > 0]
    state_hint = scores[
        ((source_mask & (SOURCE_STATE | SOURCE_HINT)) > 0)
        & ((source_mask & SOURCE_XGBOOST) == 0)
    ]
    ablation_specs = (
        ("A2_state_only", state_only, "generator_score", (False,)),
        ("A3_state_hints", state_hint, "generator_score", (False,)),
        ("A4_eventness", state_hint, "calibrated_event_probability", (False,)),
        ("A5_predicted_iou", state_hint, "final_score", (False,)),
        ("A6_boundary", state_hint, "final_score", (True,)),
        ("A7_current_mode", scores, "final_score", (False, True)),
    )
    ablation_rows: list[dict[str, Any]] = []
    ablation_rows.extend(
        _registered_baseline_ablations(
            config,
            fold,
            outer_train_subjects,
            outer_subjects,
            truth,
            ignore,
            windows,
        )
    )
    ablation_selections: dict[str, Any] = {}
    for name, variant_scores, score_column, boundary_options in ablation_specs:
        try:
            variant, _ = select_pipeline_parameters(
                variant_scores,
                windows,
                boundary_ids,
                boundary_output,
                truth,
                ignore,
                config,
                score_column=score_column,
                boundary_options=boundary_options,
            )
            ablation_selections[name] = variant
            ablation_rows.append({"ablation": name, "eligible": True, **variant})
        except (RuntimeError, ValueError) as error:
            ablation_selections[name] = {"eligible": False, "reason": str(error)}
            ablation_rows.append(
                {"ablation": name, "eligible": False, "reason": str(error)}
            )
    eventness = ablation_selections["A4_eventness"]
    predicted_iou = ablation_selections["A5_predicted_iou"]
    iou_gate = False
    if eventness.get("eligible", True) and predicted_iou.get("eligible", True):
        f1_gain = float(predicted_iou["refined_f1"]) - float(eventness["refined_f1"])
        eventness_mae = float(eventness["refined_boundary_mae_seconds"])
        predicted_mae = float(predicted_iou["refined_boundary_mae_seconds"])
        mae_gain = (
            (eventness_mae - predicted_mae) / eventness_mae
            if np.isfinite(eventness_mae) and eventness_mae > 0
            else -math.inf
        )
        iou_gate = f1_gain >= 0.005 or (mae_gain >= 0.05 and f1_gain >= -0.005)
    selected_score_column = "final_score" if iou_gate else "calibrated_event_probability"
    selected, trial_frame = select_pipeline_parameters(
        scores,
        windows,
        boundary_ids,
        boundary_output,
        truth,
        ignore,
        config,
        score_column=selected_score_column,
    )
    selected.update(
        {
            "protocol_version": 3,
            "run_name": run.payload["run_name"],
            "outer_fold": fold,
            "selection_scope": "outer_train_subject_crossfit_oof_only",
            "xgb_mode": config["hierarchical"]["xgb_mode"],
            "outer_evaluated": False,
            "predicted_iou_gate_passed": iou_gate,
        }
    )
    verifier_seed_diagnostics: list[dict[str, Any]] = []
    a3_f1 = float(ablation_selections["A3_state_hints"].get("refined_f1", 0.0))
    for seed in (int(value) for value in config["verifier"]["seeds"]):
        seed_scores = scores.copy()
        seed_scores["event_logit"] = seed_scores[f"event_logit_seed_{seed}"]
        seed_scores["iou_logit"] = seed_scores[f"iou_logit_seed_{seed}"]
        seed_calibrated, _ = crossfit_calibrate_scores(seed_scores)
        seed_source_mask = seed_calibrated["source_mask"].to_numpy(dtype=np.int64)
        seed_state_hint = seed_calibrated[
            ((seed_source_mask & (SOURCE_STATE | SOURCE_HINT)) > 0)
            & ((seed_source_mask & SOURCE_XGBOOST) == 0)
        ]
        seed_selection, _ = select_pipeline_parameters(
            seed_state_hint,
            windows,
            boundary_ids,
            boundary_output,
            truth,
            ignore,
            config,
            score_column=selected_score_column,
            boundary_options=(False,),
        )
        verifier_seed_diagnostics.append(
            {
                "seed": seed,
                "f1": seed_selection["refined_f1"],
                "direction_positive_vs_a3": float(seed_selection["refined_f1"])
                > a3_f1,
            }
        )
    boundary_seed_diagnostics: list[dict[str, Any]] = []
    coarse_selection = {**selected, "use_boundary": False}
    coarse_events, _ = _apply_selected_pipeline(
        scores, boundary_ids, boundary_output, windows, config, coarse_selection
    )
    coarse_metrics, _ = _metrics_with_diagnostics(
        truth, coarse_events, ignore, windows, config
    )
    if bool(selected.get("use_boundary", False)):
        for seed in (int(value) for value in config["boundary"]["seeds"]):
            seed_ids, seed_output = _load_boundary_seed_outputs(
                run.root / "oof" / "boundary_outputs.npz", seed
            )
            _, seed_events = _apply_selected_pipeline(
                scores, seed_ids, seed_output, windows, config, selected
            )
            seed_metrics, _ = _metrics_with_diagnostics(
                truth, seed_events, ignore, windows, config
            )
            coarse_mae = float(coarse_metrics["boundary_mae_seconds"])
            seed_mae = float(seed_metrics["boundary_mae_seconds"])
            mae_improvement = (
                (coarse_mae - seed_mae) / coarse_mae
                if np.isfinite(coarse_mae) and coarse_mae > 0
                else -math.inf
            )
            direction = (
                mae_improvement > 0
                and float(seed_metrics["f1"])
                >= float(coarse_metrics["f1"])
                - float(config["promotion_gate"]["maximum_boundary_f1_drop"])
            )
            boundary_seed_diagnostics.append(
                {
                    "seed": seed,
                    "f1": seed_metrics["f1"],
                    "boundary_mae_seconds": seed_mae,
                    "mae_improvement": mae_improvement,
                    "direction_positive": direction,
                }
            )
    selected["verifier_seed_direction_passed"] = (
        sum(bool(row["direction_positive_vs_a3"]) for row in verifier_seed_diagnostics)
        >= 2
    )
    selected["boundary_seed_direction_passed"] = (
        not bool(selected.get("use_boundary", False))
        or sum(bool(row["direction_positive"]) for row in boundary_seed_diagnostics) >= 2
    )
    selection_path = run.root / "selection" / "selected_pipeline.json"
    trials_path = run.root / "selection" / "selection_trials.csv"
    metrics_path = run.root / "selection" / "oof_metrics.json"
    ablations_path = run.root / "selection" / "module_ablations.csv"
    ablation_selection_path = run.root / "selection" / "selected_ablations.json"
    seed_diagnostics_path = run.root / "selection" / "seed_diagnostics.json"
    write_json_atomic(selection_path, selected)
    write_csv_atomic(trials_path, trial_frame)
    write_csv_atomic(ablations_path, pd.DataFrame(ablation_rows))
    write_json_atomic(ablation_selection_path, ablation_selections)
    write_json_atomic(
        seed_diagnostics_path,
        {
            "verifier": verifier_seed_diagnostics,
            "boundary": boundary_seed_diagnostics,
        },
    )
    write_json_atomic(
        metrics_path,
        {
            "selected": {
                key: value
                for key, value in selected.items()
                if key.startswith(("coarse_", "refined_"))
            },
            "candidate_count": len(scores),
            "selection_trials": len(trial_frame),
        },
    )
    run.transition(
        "SELECTED",
        [
            selection_path,
            trials_path,
            metrics_path,
            ablations_path,
            ablation_selection_path,
            seed_diagnostics_path,
        ],
    )


def select_pipeline_parameters(
    scores: pd.DataFrame,
    windows: pd.DataFrame,
    boundary_ids: np.ndarray,
    boundary_output: dict[str, np.ndarray],
    truth: pd.DataFrame,
    ignore: pd.DataFrame,
    config: dict[str, Any],
    *,
    score_column: str = "final_score",
    boundary_options: tuple[bool, ...] = (False, True),
) -> tuple[dict[str, Any], pd.DataFrame]:
    if scores.empty:
        raise ValueError("Pipeline selection requires non-empty OOF proposal scores")
    if score_column not in scores:
        raise ValueError(f"Pipeline score column is missing: {score_column}")
    if not boundary_options or not set(boundary_options) <= {False, True}:
        raise ValueError("Boundary options must contain False and/or True")
    quantiles = sorted({float(value) for value in config["calibration"]["acceptance_quantiles"]})
    thresholds = sorted(
        {
            float(np.quantile(scores[score_column], quantile))
            for quantile in quantiles
        }
    )
    trials: list[dict[str, Any]] = []
    for threshold in thresholds:
        for nms_threshold in config["calibration"]["nms_iou_thresholds"]:
            coarse_selection = {
                "acceptance_threshold": threshold,
                "nms_iou_threshold": float(nms_threshold),
                "boundary_entropy_threshold": 1.0,
                "use_boundary": False,
                "score_column": score_column,
            }
            coarse, _ = _apply_selected_pipeline(
                scores,
                boundary_ids,
                boundary_output,
                windows,
                config,
                coarse_selection,
            )
            coarse_metrics, coarse_matches = _metrics_with_diagnostics(
                truth, coarse, ignore, windows, config
            )
            if False in boundary_options:
                trials.append(
                    {
                        **coarse_selection,
                        **{f"coarse_{key}": value for key, value in coarse_metrics.items()},
                        **{f"refined_{key}": value for key, value in coarse_metrics.items()},
                        "boundary_mae_improvement": 0.0,
                        "tp_to_fp_fraction": 0.0,
                        "boundary_gate_passed": True,
                    }
                )
            entropy_thresholds = (
                config["boundary"]["entropy_thresholds"]
                if True in boundary_options
                else ()
            )
            for entropy_threshold in entropy_thresholds:
                selection = {
                    "acceptance_threshold": threshold,
                    "nms_iou_threshold": float(nms_threshold),
                    "boundary_entropy_threshold": float(entropy_threshold),
                    "use_boundary": True,
                    "score_column": score_column,
                }
                coarse, refined = _apply_selected_pipeline(
                    scores,
                    boundary_ids,
                    boundary_output,
                    windows,
                    config,
                    selection,
                )
                refined_metrics, refined_matches = _metrics_with_diagnostics(
                    truth, refined, ignore, windows, config
                )
                coarse_mae = float(coarse_metrics["boundary_mae_seconds"])
                refined_mae = float(refined_metrics["boundary_mae_seconds"])
                improvement = (
                    (coarse_mae - refined_mae) / coarse_mae
                    if np.isfinite(coarse_mae) and coarse_mae > 0
                    else -math.inf
                )
                coarse_tp_ids = set(
                    coarse_matches.get(
                        "prediction_event_id", pd.Series(dtype=str)
                    ).astype(str)
                ) - {""}
                refined_tp_ids = set(
                    refined_matches.get(
                        "prediction_event_id", pd.Series(dtype=str)
                    ).astype(str)
                ) - {""}
                tp_to_fp = len(coarse_tp_ids - refined_tp_ids) / max(
                    len(coarse_tp_ids), 1
                )
                relation_changes: dict[str, float | bool] = {}
                relation_gate = True
                for relation in ("same", "different"):
                    coarse_values = np.asarray(
                        [
                            coarse_metrics[f"{relation}_start_mae_seconds"],
                            coarse_metrics[f"{relation}_end_mae_seconds"],
                        ],
                        dtype=np.float64,
                    )
                    refined_values = np.asarray(
                        [
                            refined_metrics[f"{relation}_start_mae_seconds"],
                            refined_metrics[f"{relation}_end_mae_seconds"],
                        ],
                        dtype=np.float64,
                    )
                    coarse_relation_mae = (
                        float(np.nanmean(coarse_values))
                        if np.isfinite(coarse_values).any()
                        else math.nan
                    )
                    refined_relation_mae = (
                        float(np.nanmean(refined_values))
                        if np.isfinite(refined_values).any()
                        else math.nan
                    )
                    relation_not_worse = (
                        not np.isfinite(coarse_relation_mae)
                        or (
                            np.isfinite(refined_relation_mae)
                            and refined_relation_mae <= coarse_relation_mae + 1e-9
                        )
                    )
                    relation_changes[f"{relation}_boundary_mae_change_seconds"] = (
                        refined_relation_mae - coarse_relation_mae
                        if np.isfinite(coarse_relation_mae)
                        and np.isfinite(refined_relation_mae)
                        else math.nan
                    )
                    relation_changes[f"{relation}_boundary_not_worse"] = relation_not_worse
                    relation_gate = relation_gate and relation_not_worse
                gate = (
                    improvement
                    >= float(config["promotion_gate"]["minimum_boundary_mae_improvement"])
                    and float(refined_metrics["f1"])
                    >= float(coarse_metrics["f1"])
                    - float(config["promotion_gate"]["maximum_boundary_f1_drop"])
                    and tp_to_fp
                    <= float(config["promotion_gate"]["maximum_tp_to_fp_fraction"])
                    and relation_gate
                )
                trials.append(
                    {
                        **selection,
                        **{f"coarse_{key}": value for key, value in coarse_metrics.items()},
                        **{f"refined_{key}": value for key, value in refined_metrics.items()},
                        "boundary_mae_improvement": improvement,
                        "tp_to_fp_fraction": tp_to_fp,
                        **relation_changes,
                        "boundary_gate_passed": gate,
                    }
                )
    trial_frame = pd.DataFrame(trials)
    eligible = trial_frame[trial_frame["boundary_gate_passed"]].copy()
    if eligible.empty:
        raise RuntimeError("No hierarchical selection satisfies the boundary safety gate")
    eligible = eligible.sort_values(
        [
            "refined_f1",
            "refined_false_positives_per_observed_hour",
            "refined_boundary_mae_seconds",
            "acceptance_threshold",
        ],
        ascending=[False, True, True, False],
    )
    selected = eligible.iloc[0].to_dict()
    return selected, trial_frame


def _failure_cases(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    matches: pd.DataFrame,
    ignore: pd.DataFrame,
) -> pd.DataFrame:
    matched_truth = set(matches.get("event_id", pd.Series(dtype=str)).astype(str))
    false_negative = truth[
        ~truth.get("event_id", pd.Series("", index=truth.index)).astype(str).isin(matched_truth)
    ].copy()
    false_negative["failure_type"] = "false_negative"
    false_negative = false_negative.rename(
        columns={"start_ms": "truth_start_ms", "end_ms": "truth_end_ms"}
    )
    matched_predictions = {
        (
            str(row.subject_key),
            int(row.prediction_start_ms),
            int(row.prediction_end_ms),
        )
        for row in matches.itertuples(index=False)
    }
    unmatched_mask = np.asarray(
        [
            (
                str(row.subject_key),
                int(row.start_ms),
                int(row.end_ms),
            )
            not in matched_predictions
            for row in prediction.itertuples(index=False)
        ],
        dtype=bool,
    )
    false_positive = prediction.loc[unmatched_mask].copy()
    ignored_indices: list[int] = []
    if not ignore.empty:
        for index, row in false_positive.iterrows():
            ignored = ignore[ignore["subject_key"].astype(str) == str(row.subject_key)]
            if ignored.empty:
                continue
            overlap = np.maximum(
                0,
                np.minimum(int(row.end_ms), ignored["end_ms"].to_numpy(dtype=np.int64))
                - np.maximum(
                    int(row.start_ms), ignored["start_ms"].to_numpy(dtype=np.int64)
                ),
            )
            if np.any(overlap > 0):
                ignored_indices.append(index)
    false_positive = false_positive.drop(index=ignored_indices)
    false_positive["failure_type"] = "false_positive"
    false_positive = false_positive.rename(
        columns={"start_ms": "prediction_start_ms", "end_ms": "prediction_end_ms"}
    )
    return pd.concat((false_negative, false_positive), ignore_index=True, sort=False)


def _per_subject_metrics(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    ignore: pd.DataFrame,
    windows: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subjects = sorted(
        set(truth.get("subject_key", pd.Series(dtype=str)).astype(str))
        | set(prediction.get("subject_key", pd.Series(dtype=str)).astype(str))
    )
    for subject in subjects:
        selected_truth = truth[truth["subject_key"].astype(str) == subject]
        selected_prediction = prediction[
            prediction["subject_key"].astype(str) == subject
        ]
        selected_ignore = ignore[ignore["subject_key"].astype(str) == subject]
        selected_windows = windows[windows["subject_key"].astype(str) == subject]
        metrics, _ = _metrics_with_diagnostics(
            selected_truth,
            selected_prediction,
            selected_ignore,
            selected_windows,
            config,
        )
        rows.append({"subject_key": subject, **metrics})
    return pd.DataFrame(rows)


def evaluate_hierarchical_outer(
    run: HierarchicalRun,
    config: dict[str, Any],
    inputs: HierarchicalInputs,
) -> None:
    run.require_stage("SELECTED")
    fold = int(run.payload["outer_fold"])
    _, outer_subjects = _outer_subject_sets(inputs, fold)
    selection_path = run.root / "selection" / "selected_pipeline.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("outer_evaluated") is not False:
        raise RuntimeError("Outer evaluation was already recorded")
    scores = pd.read_parquet(run.root / "outer" / "proposal_scores.parquet")
    windows = pd.read_parquet(run.root / "outer" / "window_predictions.parquet")
    boundary_ids, boundary_output = _load_boundary_outputs(
        run.root / "outer" / "boundary_outputs.npz"
    )
    coarse, refined = _apply_selected_pipeline(
        scores, boundary_ids, boundary_output, windows, config, selection
    )
    accepted_records = _accepted_pipeline_records(
        scores, boundary_ids, boundary_output, windows, config, selection
    )
    audit_scores = _proposal_score_audit_frame(scores, accepted_records)
    guard = OuterLabelGuard(frozenset(outer_subjects), allow_outer_labels=True)
    outer_events = guard.select_labels(inputs.events, outer_subjects)
    truth, ignore = partition_evaluation_events(outer_events, outer_subjects)
    coarse_metrics, _ = _metrics_with_diagnostics(truth, coarse, ignore, windows, config)
    refined_metrics, matches = _metrics_with_diagnostics(
        truth, refined, ignore, windows, config
    )
    ablation_selections = json.loads(
        (run.root / "selection" / "selected_ablations.json").read_text(encoding="utf-8")
    )
    source_mask = scores["source_mask"].to_numpy(dtype=np.int64)
    ablation_frames = {
        "A2_state_only": scores[(source_mask & SOURCE_STATE) > 0],
        "A3_state_hints": scores[
            ((source_mask & (SOURCE_STATE | SOURCE_HINT)) > 0)
            & ((source_mask & SOURCE_XGBOOST) == 0)
        ],
        "A4_eventness": scores[
            ((source_mask & (SOURCE_STATE | SOURCE_HINT)) > 0)
            & ((source_mask & SOURCE_XGBOOST) == 0)
        ],
        "A5_predicted_iou": scores[
            ((source_mask & (SOURCE_STATE | SOURCE_HINT)) > 0)
            & ((source_mask & SOURCE_XGBOOST) == 0)
        ],
        "A6_boundary": scores[
            ((source_mask & (SOURCE_STATE | SOURCE_HINT)) > 0)
            & ((source_mask & SOURCE_XGBOOST) == 0)
        ],
        "A7_current_mode": scores,
    }
    ablation_metrics: dict[str, Any] = {}
    ablation_subject_frames: list[pd.DataFrame] = []
    for name, variant_scores in ablation_frames.items():
        variant_selection = ablation_selections.get(name, {})
        if variant_selection.get("eligible") is False:
            ablation_metrics[name] = variant_selection
            continue
        _, variant_events = _apply_selected_pipeline(
            variant_scores,
            boundary_ids,
            boundary_output,
            windows,
            config,
            variant_selection,
        )
        variant_metrics, _ = _metrics_with_diagnostics(
            truth, variant_events, ignore, windows, config
        )
        ablation_metrics[name] = variant_metrics
        variant_subjects = _per_subject_metrics(
            truth, variant_events, ignore, windows, config
        )
        variant_subjects.insert(0, "ablation", name)
        ablation_subject_frames.append(variant_subjects)
    evaluation_dir = run.root / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = evaluation_dir / "outer_predictions.parquet"
    events_path = evaluation_dir / "events.csv"
    metrics_path = evaluation_dir / "metrics.json"
    failures_path = evaluation_dir / "failure_cases.csv"
    per_subject_path = evaluation_dir / "per_subject_metrics.csv"
    ablation_metrics_path = evaluation_dir / "ablation_metrics.json"
    ablation_subjects_path = evaluation_dir / "ablation_per_subject_metrics.csv"
    write_parquet_atomic(predictions_path, audit_scores)
    write_csv_atomic(events_path, refined)
    write_json_atomic(
        metrics_path,
        {"coarse": coarse_metrics, "refined": refined_metrics},
    )
    write_csv_atomic(failures_path, _failure_cases(truth, refined, matches, ignore))
    write_csv_atomic(
        per_subject_path,
        _per_subject_metrics(truth, refined, ignore, windows, config),
    )
    write_json_atomic(ablation_metrics_path, ablation_metrics)
    write_csv_atomic(
        ablation_subjects_path,
        pd.concat(ablation_subject_frames, ignore_index=True)
        if ablation_subject_frames
        else pd.DataFrame(columns=["ablation", "subject_key"]),
    )
    run.transition(
        "EVALUATED",
        [
            predictions_path,
            events_path,
            metrics_path,
            failures_path,
            per_subject_path,
            ablation_metrics_path,
            ablation_subjects_path,
        ],
    )
