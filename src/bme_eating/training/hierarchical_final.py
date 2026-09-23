from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from bme_eating.calibration import crossfit_calibrate_scores
from bme_eating.data.deep_dataset import (
    DTPDataset,
    SegmentBalancedBatchSampler,
    compute_normalization,
)
from bme_eating.hierarchical_artifacts import (
    initialize_hierarchical_run,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
    write_yaml_atomic,
)
from bme_eating.metrics import partition_evaluation_events
from bme_eating.models.boundary_refiner import (
    BoundaryRefiner,
    boundary_loss,
    build_boundary_targets,
)
from bme_eating.models.event_verifier import (
    EventVerifier,
    ProposalFeatureBatch,
    ProposalTensorDataset,
    build_proposal_features,
    verifier_loss,
)
from bme_eating.models.factory import build_state_model
from bme_eating.models.losses import HierarchicalStateLoss
from bme_eating.proposals import (
    SOURCE_HINT,
    SOURCE_STATE,
    SOURCE_XGBOOST,
    duration_bounds_from_training_events,
    label_event_candidates,
)
from bme_eating.training.dtp_trainer import (
    configured_state_feature_columns,
    seed_everything,
)
from bme_eating.training.hierarchical_trainer import (
    HierarchicalInputs,
    _BoundaryDataset,
    _load_boundary_outputs,
    _proposal_batch_sampler,
    _save_feature_batch,
    _write_boundary_outputs,
    select_pipeline_parameters,
)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if not key.startswith("_") and key not in {"credentials", "secrets"}
    }


def _final_manifest_path(final_root: Path) -> Path:
    return final_root / "final_manifest.json"


def _update_final_manifest(
    final_root: Path,
    stage: str,
    artifacts: list[Path],
    updates: dict[str, Any] | None = None,
) -> None:
    path = _final_manifest_path(final_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    hashes = dict(payload.get("artifact_hashes", {}))
    for artifact in artifacts:
        relative = artifact.resolve().relative_to(final_root.resolve()).as_posix()
        hashes[relative] = sha256_file(artifact)
    payload["stage"] = stage
    payload["artifact_hashes"] = hashes
    if updates:
        payload.update(updates)
    write_json_atomic(path, payload)


def _initialize_final_root(
    final_root: Path,
    config: dict[str, Any],
    source_runs: list[Any],
    *,
    fresh: bool,
) -> None:
    manifest_path = _final_manifest_path(final_root)
    if manifest_path.is_file():
        if fresh:
            raise FileExistsError(f"Final run already exists: {final_root}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative, expected in payload.get("artifact_hashes", {}).items():
            path = final_root / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"Final artifact changed after manifesting: {relative}")
        current_sources = {
            str(run.payload["outer_fold"]): sha256_file(run.manifest_path)
            for run in source_runs
        }
        if payload.get("source_fold_manifests") != current_sources:
            raise RuntimeError("Source fold manifests changed after final training began")
        return
    if not fresh:
        raise FileNotFoundError("Final run does not exist; start it with --fresh")
    if final_root.exists() and any(final_root.iterdir()):
        raise RuntimeError("Final run directory exists without a manifest")
    final_root.mkdir(parents=True, exist_ok=True)
    resolved_path = final_root / "resolved_config.yaml"
    write_yaml_atomic(resolved_path, _public_config(config))
    write_json_atomic(
        manifest_path,
        {
            "version": 3,
            "stage": "CREATED",
            "run_name": source_runs[0].payload["run_name"],
            "source_fold_manifests": {
                str(run.payload["outer_fold"]): sha256_file(run.manifest_path)
                for run in source_runs
            },
            "artifact_hashes": {"resolved_config.yaml": sha256_file(resolved_path)},
        },
    )


def _selected_epoch_count(source_runs: list[Any], component: str) -> int:
    epochs: list[int] = []
    if component == "state":
        paths = [
            run.root / f"crossfit_{partition}" / "state" / "best.pt"
            for run in source_runs
            for partition in range(3)
        ]
    else:
        paths = sorted(
            path
            for run in source_runs
            for path in (run.root / "models").glob(f"{component}_crossfit_*_seed_*.pt")
            if not path.name.endswith(".last.pt")
        )
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        epochs.append(int(checkpoint["epoch"]) + 1)
    if not epochs:
        raise FileNotFoundError(f"No crossfit checkpoints found for {component}")
    return max(1, round(median(epochs)))


def _fit_stable_normalization(
    anchors: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if not columns:
        return anchors, None
    values = anchors[columns].to_numpy(dtype=np.float64)
    center = np.nanmedian(values, axis=0)
    scale = np.nanpercentile(values, 75, axis=0) - np.nanpercentile(values, 25, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("Final stable-feature normalization is not finite")
    transformed = anchors.copy()
    filled = np.nan_to_num(values, nan=center, posinf=center, neginf=center)
    transformed.loc[:, columns] = np.clip((filled - center) / scale, -10.0, 10.0)
    return transformed, {
        "columns": columns,
        "median": center.tolist(),
        "iqr": scale.tolist(),
    }


def _train_final_state_seed(
    anchors: pd.DataFrame,
    inputs: HierarchicalInputs,
    config: dict[str, Any],
    normalization: Any,
    stable_normalization: dict[str, Any] | None,
    epochs: int,
    seed: int,
    output_path: Path,
    *,
    resume: bool,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Final hierarchical state training requires CUDA")
    training = config["training"]
    model_config = config["model"]
    stable_columns = list(configured_state_feature_columns(model_config))
    dataset = DTPDataset(
        anchors,
        inputs.segments,
        normalization,
        future_context_seconds=0,
        training=True,
        ppg_augmentation_probability=float(training["ppg_augmentation_probability"]),
        seed=seed,
        motion_block_seconds=int(model_config["motion_block_seconds"]),
        ppg_block_seconds=int(model_config["ppg_block_seconds"]),
        motion_bucket_counts=model_config["motion_bucket_counts"],
        ppg_bucket_counts=model_config["ppg_bucket_counts"],
        stable_feature_columns=stable_columns,
    )
    sampler = SegmentBalancedBatchSampler(
        anchors,
        batch_size=int(training["batch_size"]),
        steps_per_epoch=int(training["steps_per_epoch"]),
        positive_fraction=float(training["positive_sampling_fraction"]),
        seed=seed,
    )
    workers = int(training["num_workers"])
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    device = torch.device("cuda")
    model = build_state_model(model_config).to(device)
    encoder_parameters = list(model.motion_encoder.parameters()) + list(
        model.ppg_encoder.parameters()
    )
    encoder_ids = {id(value) for value in encoder_parameters}
    optimizer = AdamW(
        [
            {"params": encoder_parameters, "lr": float(training["encoder_learning_rate"])},
            {
                "params": [value for value in model.parameters() if id(value) not in encoder_ids],
                "lr": float(training["learning_rate"]),
            },
        ],
        weight_decay=float(training["weight_decay"]),
    )
    accumulation = int(training["gradient_accumulation"])
    total_steps = math.ceil(len(loader) / accumulation) * epochs
    warmup_steps = max(1, int(total_steps * float(training["warmup_fraction"])))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = LambdaLR(optimizer, schedule)
    criterion = HierarchicalStateLoss(
        positive_alpha=float(training["focal_positive_alpha"]),
        focal_gamma=float(config["loss"]["focal_gamma"]),
        dice_weight=float(config["loss"]["dice_weight"]),
        boundary_weight=float(config["loss"]["boundary_weight"]),
        sqi_weight=float(config["loss"]["sqi_weight"]),
        boundary_positive_weight=float(training["boundary_positive_weight"]),
        smooth_weight=float(config["loss"]["smooth_weight"]),
        smooth_tau=float(config["loss"]["smooth_tau"]),
    ).to(device)
    amp_dtype = torch.bfloat16 if str(training["amp_dtype"]).lower() == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    last_path = output_path.with_name(output_path.stem + ".last.pt")
    start_epoch = 0
    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if int(checkpoint.get("seed", -1)) != seed or int(checkpoint.get("epochs", -1)) != epochs:
            raise RuntimeError("Final state resume checkpoint identity mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        seed_everything(seed + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            moved = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                loss, _ = criterion(model(moved), moved)
                scaled = loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError("Final state training produced a non-finite loss")
            scaler.scale(scaled).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        _atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "seed": seed,
                "epoch": epoch,
                "epochs": epochs,
            },
            last_path,
        )
    _atomic_torch_save(
        {
            "model": model.state_dict(),
            "model_config": model_config,
            "normalization": normalization.to_json(),
            "stable_feature_normalization": stable_normalization,
            "seed": seed,
            "epochs": epochs,
            "parameter_count": sum(value.numel() for value in model.parameters()),
        },
        output_path,
    )


def _aggregate_oof_snapshot(
    source_runs: list[Any],
    inputs: HierarchicalInputs,
    config: dict[str, Any],
    snapshot_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, ProposalFeatureBatch, np.ndarray, dict[str, np.ndarray]]:
    windows: list[pd.DataFrame] = []
    proposals: list[pd.DataFrame] = []
    scores: list[pd.DataFrame] = []
    boundary_ids: list[np.ndarray] = []
    boundary_parts: list[dict[str, np.ndarray]] = []
    sources: dict[str, Any] = {}
    for run in source_runs:
        fold = int(run.payload["outer_fold"])
        window_path = run.root / "outer" / "window_predictions.parquet"
        proposal_path = run.root / "outer" / "proposals.parquet"
        score_path = run.root / "outer" / "proposal_scores.parquet"
        boundary_path = run.root / "outer" / "boundary_outputs.npz"
        fold_windows = pd.read_parquet(window_path)
        fold_subjects = set(fold_windows["subject_key"].astype(str))
        fold_truth, _ = partition_evaluation_events(inputs.events, fold_subjects)
        fold_proposals = label_event_candidates(
            pd.read_parquet(proposal_path),
            fold_truth,
            float(config["postprocess"]["iou_threshold"]),
        )
        fold_proposals["calibration_fold"] = fold
        fold_scores = pd.read_parquet(score_path).drop(
            columns=[
                "max_iou",
                "matched_event_id",
                "is_positive",
                "negative_type",
                "calibration_fold",
            ],
            errors="ignore",
        )
        fold_scores = fold_scores.merge(
            fold_proposals[
                [
                    "proposal_id",
                    "max_iou",
                    "matched_event_id",
                    "is_positive",
                    "negative_type",
                    "calibration_fold",
                ]
            ],
            on="proposal_id",
            how="inner",
            validate="one_to_one",
        )
        identifiers, output = _load_boundary_outputs(boundary_path)
        windows.append(fold_windows)
        proposals.append(fold_proposals)
        scores.append(fold_scores)
        boundary_ids.append(identifiers)
        boundary_parts.append(output)
        sources[str(fold)] = {
            "manifest_sha256": sha256_file(run.manifest_path),
            "window_predictions_sha256": sha256_file(window_path),
            "proposals_sha256": sha256_file(proposal_path),
            "proposal_scores_sha256": sha256_file(score_path),
            "boundary_outputs_sha256": sha256_file(boundary_path),
        }
    combined_windows = pd.concat(windows, ignore_index=True)
    combined_proposals = pd.concat(proposals, ignore_index=True)
    combined_scores = pd.concat(scores, ignore_index=True)
    if combined_proposals["proposal_id"].duplicated().any():
        raise RuntimeError("Final OOF snapshot contains duplicate proposal identifiers")
    identifiers = np.concatenate(boundary_ids)
    output = {
        name: np.concatenate([part[name] for part in boundary_parts])
        for name in boundary_parts[0]
    }
    features = build_proposal_features(
        combined_proposals,
        combined_windows,
        inputs.features,
        list(config["hierarchical"]["stable_feature_columns"]),
        config["verifier"],
    )
    snapshot_root.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(snapshot_root / "window_predictions.parquet", combined_windows)
    write_parquet_atomic(snapshot_root / "proposals_labeled.parquet", combined_proposals)
    calibrated_scores, calibration = crossfit_calibrate_scores(combined_scores)
    write_parquet_atomic(snapshot_root / "proposal_scores.parquet", calibrated_scores)
    _write_boundary_outputs(snapshot_root / "boundary_outputs.npz", identifiers, output)
    _save_feature_batch(snapshot_root / "proposal_features.npz", features)
    write_json_atomic(snapshot_root / "sources.json", sources)
    truth, ignore = partition_evaluation_events(
        inputs.events, set(inputs.subject_folds)
    )
    source_mask = calibrated_scores["source_mask"].to_numpy(dtype=np.int64)
    state_hint = calibrated_scores[
        ((source_mask & (SOURCE_STATE | SOURCE_HINT)) > 0)
        & ((source_mask & SOURCE_XGBOOST) == 0)
    ]
    eventness, _ = select_pipeline_parameters(
        state_hint,
        combined_windows,
        identifiers,
        output,
        truth,
        ignore,
        config,
        score_column="calibrated_event_probability",
        boundary_options=(False,),
    )
    predicted_iou, _ = select_pipeline_parameters(
        state_hint,
        combined_windows,
        identifiers,
        output,
        truth,
        ignore,
        config,
        score_column="final_score",
        boundary_options=(False,),
    )
    f1_gain = float(predicted_iou["refined_f1"]) - float(eventness["refined_f1"])
    eventness_mae = float(eventness["refined_boundary_mae_seconds"])
    predicted_mae = float(predicted_iou["refined_boundary_mae_seconds"])
    mae_gain = (
        (eventness_mae - predicted_mae) / eventness_mae
        if np.isfinite(eventness_mae) and eventness_mae > 0
        else -math.inf
    )
    iou_gate = f1_gain >= 0.005 or (mae_gain >= 0.05 and f1_gain >= -0.005)
    score_column = "final_score" if iou_gate else "calibrated_event_probability"
    selected, trials = select_pipeline_parameters(
        calibrated_scores,
        combined_windows,
        identifiers,
        output,
        truth,
        ignore,
        config,
        score_column=score_column,
    )
    selected.update(
        {
            "protocol_version": 3,
            "selection_scope": "five_fold_subject_oof",
            "xgb_mode": config["hierarchical"]["xgb_mode"],
            "predicted_iou_gate_passed": iou_gate,
        }
    )
    write_json_atomic(snapshot_root.parent / "calibration.json", calibration.to_json())
    write_json_atomic(snapshot_root.parent / "selected_pipeline.json", selected)
    write_csv_atomic(snapshot_root / "selection_trials.csv", trials)
    return combined_proposals, combined_windows, features, identifiers, output


def _train_full_verifier(
    proposals: pd.DataFrame,
    features: ProposalFeatureBatch,
    config: dict[str, Any],
    epochs: int,
    output_path: Path,
    *,
    resume: bool,
) -> None:
    seed = int(config["verifier"]["seeds"][0])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventVerifier(features.sequence.shape[-1], features.scalar.shape[-1], config["verifier"]).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["verifier"]["learning_rate"]),
        weight_decay=float(config["verifier"]["weight_decay"]),
    )
    sampler = _proposal_batch_sampler(proposals, config["verifier"], seed)
    last_path = output_path.with_name(output_path.stem + ".last.pt")
    start_epoch = 0
    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
    for epoch in range(start_epoch, epochs):
        seed_everything(seed + epoch)
        sampler.set_epoch(epoch)
        loader = DataLoader(
            ProposalTensorDataset(features),
            batch_sampler=sampler,
        )
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            result = model(
                {
                    "sequence": batch["sequence"].to(device),
                    "scalar": batch["scalar"].to(device),
                }
            )
            loss, _ = verifier_loss(
                result,
                batch["event_target"].to(device),
                batch["iou_target"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        _atomic_torch_save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
            last_path,
        )
    _atomic_torch_save(
        {
            "model": model.state_dict(),
            "sequence_dim": features.sequence.shape[-1],
            "scalar_dim": features.scalar.shape[-1],
            "config": config["verifier"],
            "seed": seed,
            "epochs": epochs,
        },
        output_path,
    )


def _train_full_boundary(
    proposals: pd.DataFrame,
    features: ProposalFeatureBatch,
    events: pd.DataFrame,
    config: dict[str, Any],
    epochs: int,
    output_path: Path,
    *,
    resume: bool,
) -> None:
    positive = proposals["is_positive"].to_numpy(dtype=bool)
    positive_proposals = proposals.loc[positive].reset_index(drop=True)
    positive_features = ProposalFeatureBatch(
        proposal_ids=features.proposal_ids[positive],
        sequence=features.sequence[positive],
        scalar=features.scalar[positive],
        event_target=features.event_target[positive] if features.event_target is not None else None,
        iou_target=features.iou_target[positive] if features.iou_target is not None else None,
    )
    targets = build_boundary_targets(positive_proposals, events, config["boundary"])
    seed = int(config["boundary"]["seeds"][0])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoundaryRefiner(
        positive_features.sequence.shape[-1],
        positive_features.scalar.shape[-1],
        config["boundary"],
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["boundary"]["learning_rate"]),
        weight_decay=float(config["boundary"]["weight_decay"]),
    )
    last_path = output_path.with_name(output_path.stem + ".last.pt")
    start_epoch = 0
    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
    for epoch in range(start_epoch, epochs):
        seed_everything(seed + epoch)
        loader = DataLoader(
            _BoundaryDataset(positive_features, targets),
            batch_size=int(config["boundary"]["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            result = model(
                {
                    "sequence": batch["sequence"].to(device),
                    "scalar": batch["scalar"].to(device),
                }
            )
            loss, _ = boundary_loss(
                result,
                batch["start_distribution"].to(device),
                batch["end_distribution"].to(device),
                batch["start_fine"].to(device),
                batch["end_fine"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        _atomic_torch_save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
            last_path,
        )
    _atomic_torch_save(
        {
            "model": model.state_dict(),
            "sequence_dim": positive_features.sequence.shape[-1],
            "scalar_dim": positive_features.scalar.shape[-1],
            "config": config["boundary"],
            "seed": seed,
            "epochs": epochs,
        },
        output_path,
    )


def _train_full_xgboost(
    input_root: Path,
    final_root: Path,
    config: dict[str, Any],
) -> list[Path]:
    experiment = str(config["final_training"]["xgboost_source_experiment"])
    source_fold = int(config["final_training"]["xgboost_parameter_source_fold"])
    source_root = input_root / "experiments" / experiment / f"fold_{source_fold}"
    metadata_path = source_root / "metadata.json"
    model_path = source_root / "model.json"
    if not metadata_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("Frozen XGBoost parameter source is incomplete")
    source_manifest_path = source_root / "run_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    for path in (metadata_path, model_path, source_root / "selected_postprocess.json"):
        expected = source_manifest.get("artifact_hashes", {}).get(path.name)
        if not expected or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen XGBoost source hash mismatch: {path.name}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    columns = [str(value) for value in metadata["feature_columns"]]
    features = pd.read_parquet(input_root / "features" / "baseline.parquet")
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"Full XGBoost features are missing columns: {sorted(missing)}")
    eligible = features[features["state_loss_mask"].fillna(0).astype(float) > 0].copy()
    weights = np.ones(len(eligible), dtype=np.float32)
    positive = eligible["state_target"].to_numpy(dtype=float) > 0
    if positive.any():
        event_ids = eligible.loc[positive, "event_id"].astype(str)
        counts = event_ids.value_counts()
        positive_weights = np.asarray([1.0 / counts[value] for value in event_ids])
        positive_weights *= positive.sum() / positive_weights.sum()
        weights[positive] = positive_weights.astype(np.float32)
    parameters = dict(metadata["best_parameters"])
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=int(metadata["final_estimators"]),
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_state=int(config["project"]["seed"]),
        n_jobs=int(config["final_training"]["xgboost_n_jobs"]),
        **parameters,
    )
    model.fit(
        eligible[columns].to_numpy(dtype=np.float32),
        eligible["state_target"].to_numpy(dtype=np.int64),
        sample_weight=weights,
        verbose=False,
    )
    output_path = final_root / "xgboost_full.json"
    temporary = output_path.with_name(output_path.stem + ".tmp.json")
    model.save_model(temporary)
    temporary.replace(output_path)
    output_metadata = final_root / "xgboost_full.metadata.json"
    write_json_atomic(
        output_metadata,
        {
            "feature_columns": columns,
            "parameters": parameters,
            "n_estimators": int(metadata["final_estimators"]),
            "source_experiment": experiment,
            "source_fold": source_fold,
            "source_model_sha256": sha256_file(model_path),
        },
    )
    postprocess_source = source_root / "selected_postprocess.json"
    postprocess_output = final_root / "xgboost_postprocess.json"
    write_json_atomic(
        postprocess_output,
        json.loads(postprocess_source.read_text(encoding="utf-8")),
    )
    return [output_path, output_metadata, postprocess_output]


def train_hierarchical_final(
    config: dict[str, Any],
    inputs: HierarchicalInputs,
    input_root: Path,
    output_root: Path,
    run_name: str,
    *,
    fresh: bool,
    resume: bool,
) -> Path:
    source_runs = [
        initialize_hierarchical_run(
            config, input_root, output_root, run_name, fold, fresh=False
        )
        for fold in range(int(config["data"]["subject_folds"]))
    ]
    for run in source_runs:
        run.require_stage("EVALUATED")
    final_root = output_root / "final" / run_name
    _initialize_final_root(final_root, config, source_runs, fresh=fresh)
    stage = json.loads(_final_manifest_path(final_root).read_text(encoding="utf-8"))["stage"]
    if stage == "EXPORTED":
        return final_root

    anchors = inputs.anchors.copy()
    stable_columns: list[str] = []
    if bool(config["model"].get("use_stable_state_features", False)):
        stable_columns = list(config["model"]["stable_feature_columns"])
        anchors = anchors.merge(
            inputs.features,
            on=["subject_key", "session_id", "timestamp_ms"],
            how="left",
            validate="one_to_one",
        )
    anchors, stable_normalization = _fit_stable_normalization(anchors, stable_columns)
    normalization = compute_normalization(inputs.segments, set(inputs.subject_folds))
    normalization_path = final_root / "sensor_normalization.json"
    stable_path = final_root / "stable_feature_normalization.json"
    if stage == "CREATED":
        write_json_atomic(normalization_path, normalization.to_json())
        write_json_atomic(stable_path, stable_normalization or {"columns": []})
        state_epochs = _selected_epoch_count(source_runs, "state")
        state_paths: list[Path] = []
        state_seeds = [int(value) for value in config["final_training"]["state_seeds"]]
        if state_seeds != [2026, 2027, 2028]:
            raise ValueError("Final state seeds must remain frozen at 2026/2027/2028")
        for seed in state_seeds:
            path = final_root / f"state_seed_{seed}.pt"
            _train_final_state_seed(
                anchors,
                inputs,
                config,
                normalization,
                stable_normalization,
                state_epochs,
                seed,
                path,
                resume=resume,
            )
            state_paths.append(path)
        _update_final_manifest(
            final_root,
            "STATE_COMPLETE",
            [normalization_path, stable_path, *state_paths],
            {"state_epochs": state_epochs},
        )
        stage = "STATE_COMPLETE"

    snapshot_root = final_root / "oof_training_snapshot"
    if stage == "STATE_COMPLETE":
        proposals, _, features, _, _ = _aggregate_oof_snapshot(
            source_runs, inputs, config, snapshot_root
        )
        final_truth, _ = partition_evaluation_events(
            inputs.events, set(inputs.subject_folds)
        )
        duration_bounds = duration_bounds_from_training_events(
            final_truth, config["proposals"]
        )
        duration_path = final_root / "duration_bounds.json"
        write_json_atomic(
            duration_path,
            {
                "minimum_seconds": duration_bounds[0],
                "maximum_seconds": duration_bounds[1],
            },
        )
        snapshot_files = [path for path in snapshot_root.rglob("*") if path.is_file()]
        _update_final_manifest(
            final_root,
            "OOF_SNAPSHOT_COMPLETE",
            [
                *snapshot_files,
                final_root / "calibration.json",
                final_root / "selected_pipeline.json",
                duration_path,
            ],
        )
        stage = "OOF_SNAPSHOT_COMPLETE"
    else:
        proposals = pd.read_parquet(snapshot_root / "proposals_labeled.parquet")
        features = build_proposal_features(
            proposals,
            pd.read_parquet(snapshot_root / "window_predictions.parquet"),
            inputs.features,
            list(config["hierarchical"]["stable_feature_columns"]),
            config["verifier"],
        )

    trained: list[Path] = []
    if stage == "OOF_SNAPSHOT_COMPLETE":
        verifier_epochs = _selected_epoch_count(source_runs, "verifier")
        verifier_path = final_root / "verifier.pt"
        _train_full_verifier(
            proposals, features, config, verifier_epochs, verifier_path, resume=resume
        )
        _update_final_manifest(
            final_root,
            "VERIFIER_COMPLETE",
            [verifier_path],
            {"verifier_epochs": verifier_epochs},
        )
        trained.append(verifier_path)
        stage = "VERIFIER_COMPLETE"
    if stage == "VERIFIER_COMPLETE":
        boundary_epochs = _selected_epoch_count(source_runs, "boundary")
        boundary_path = final_root / "boundary.pt"
        _train_full_boundary(
            proposals,
            features,
            inputs.events,
            config,
            boundary_epochs,
            boundary_path,
            resume=resume,
        )
        _update_final_manifest(
            final_root,
            "BOUNDARY_COMPLETE",
            [boundary_path],
            {"boundary_epochs": boundary_epochs},
        )
        trained.append(boundary_path)
        stage = "BOUNDARY_COMPLETE"
    if stage == "BOUNDARY_COMPLETE":
        xgb_paths = _train_full_xgboost(input_root, final_root, config)
        _update_final_manifest(final_root, "COMPLETE", xgb_paths)
    elif stage != "COMPLETE":
        raise RuntimeError(f"Unknown final training stage: {stage}")
    return final_root
