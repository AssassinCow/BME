from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from bme_eating.data.deep_dataset import (
    DTPDataset,
    SegmentBalancedBatchSampler,
    compute_normalization,
    save_normalization,
)
from bme_eating.metrics import (
    evaluate_events,
    masked_average_precision,
    partition_evaluation_events,
)
from bme_eating.models.dtp_sqf import DTPSQF, logits_to_probability_arrays
from bme_eating.models.losses import DTPLoss
from bme_eating.models.xgb_baseline import assign_train_validation_test
from bme_eating.postprocess import probabilities_to_events, tune_postprocess_parameters
from bme_eating.reproducibility import epoch_random_seed, should_validate_epoch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _learning_rate_schedule(warmup_fraction: float, total_steps: int):
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return schedule


def _prediction_frame(
    model: DTPSQF,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    description: str = "Inference",
) -> tuple[pd.DataFrame, float]:
    model.eval()
    rows: list[dict[str, object]] = []
    targets: list[float] = []
    probabilities: list[float] = []
    state_masks: list[float] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, unit="batch", leave=False):
            moved = _move_batch(batch, device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                output = model(moved)
            state, start, end = logits_to_probability_arrays(output)
            if not all(np.isfinite(values).all() for values in (state, start, end)):
                raise FloatingPointError(
                    f"{description} produced NaN or infinite probabilities"
                )
            target = batch["state_target"].numpy()
            state_mask = batch.get("state_loss_mask", torch.ones_like(batch["state_target"]))
            targets.extend(target.tolist())
            probabilities.extend(state.tolist())
            state_masks.extend(state_mask.numpy().tolist())
            for index in range(len(state)):
                rows.append(
                    {
                        "subject_key": batch["subject_key"][index],
                        "segment_id": batch["segment_id"][index],
                        "session_id": batch["session_id"][index],
                        "timestamp_ms": int(batch["timestamp_ms"][index]),
                        "state_probability": float(state[index]),
                        "start_probability": float(start[index]),
                        "end_probability": float(end[index]),
                    }
                )
    auprc = masked_average_precision(
        np.asarray(targets), np.asarray(probabilities), np.asarray(state_masks)
    )
    frame = pd.DataFrame(rows)
    for column in ("state_probability", "start_probability", "end_probability"):
        if column in frame:
            frame[column] = frame[column].astype(np.float32)
    return frame, float(auprc)


def select_checkpoint_validation_anchors(
    anchors: pd.DataFrame,
    maximum_rows: int,
    seed: int,
) -> pd.DataFrame:
    if maximum_rows <= 0:
        raise ValueError("checkpoint_validation_max_rows must be positive")
    state_mask = anchors.get("state_loss_mask", pd.Series(1.0, index=anchors.index))
    eligible = anchors[state_mask.fillna(0.0).astype(float) > 0].reset_index(drop=True)
    if eligible.empty:
        raise ValueError("Checkpoint validation has no evaluable anchors")
    if len(eligible) <= maximum_rows:
        return eligible

    positive = np.flatnonzero(eligible["state_target"].to_numpy(dtype=float) > 0)
    negative = np.flatnonzero(eligible["state_target"].to_numpy(dtype=float) <= 0)
    rng = np.random.default_rng(seed)
    if len(positive) == 0 or len(negative) == 0 or maximum_rows == 1:
        positions = rng.choice(len(eligible), size=maximum_rows, replace=False)
    else:
        positive_rows = round(maximum_rows * len(positive) / len(eligible))
        positive_rows = min(max(1, positive_rows), len(positive), maximum_rows - 1)
        negative_rows = maximum_rows - positive_rows
        if negative_rows > len(negative):
            negative_rows = len(negative)
            positive_rows = maximum_rows - negative_rows
        positions = np.concatenate(
            (
                rng.choice(positive, size=positive_rows, replace=False),
                rng.choice(negative, size=negative_rows, replace=False),
            )
        )
    return eligible.iloc[np.sort(positions)].reset_index(drop=True)


def _save_torch_checkpoint(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _postprocess_kwargs(config: dict[str, Any]) -> dict[str, float]:
    return {
        "ema_half_life_seconds": float(config["ema_half_life_seconds"]),
        "high_threshold": float(config["high_threshold"]),
        "low_threshold": float(config["low_threshold"]),
        "minimum_event_seconds": float(config["minimum_event_seconds"]),
        "merge_gap_seconds": float(config["merge_gap_seconds"]),
        "boundary_lookback_seconds": float(config["boundary_lookback_seconds"]),
    }


def resolve_focal_positive_alpha(training_config: dict[str, Any]) -> float:
    positive_fraction = float(training_config["positive_sampling_fraction"])
    positive_alpha = float(training_config["focal_positive_alpha"])
    if not 0.0 < positive_fraction < 1.0:
        raise ValueError("positive_sampling_fraction must be between zero and one")
    if not 0.0 < positive_alpha < 1.0:
        raise ValueError("focal_positive_alpha must be between zero and one")
    if math.isclose(positive_fraction, 0.5) and not math.isclose(positive_alpha, 0.5):
        raise ValueError(
            "Balanced 50% positive sampling requires focal_positive_alpha=0.5 "
            "to avoid double class compensation"
        )
    return positive_alpha


def checkpoint_selection_rank(
    selection_metric: str,
    validation_auprc: float,
    metrics: dict[str, float],
    epoch: int,
) -> tuple[float, ...]:
    if selection_metric == "window_auprc":
        score = validation_auprc if math.isfinite(validation_auprc) else -math.inf
        return (float(score), -float(epoch))
    if selection_metric != "event_f1":
        raise ValueError(f"Unsupported checkpoint selection metric: {selection_metric}")
    boundary_values = [
        float(metrics[name])
        for name in ("start_mae_seconds", "end_mae_seconds")
        if math.isfinite(float(metrics[name]))
    ]
    boundary_mae = float(np.mean(boundary_values)) if boundary_values else math.inf
    return (float(metrics["f1"]), -boundary_mae, -float(epoch))


def train_dtp_fold(
    anchors: pd.DataFrame,
    segments: pd.DataFrame,
    events: pd.DataFrame,
    subject_folds: dict[str, int],
    outer_fold: int,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    loss_config: dict[str, Any],
    postprocess_config: dict[str, Any],
    output_dir: Path,
    resume_path: Path | None = None,
    validation_selector: Callable[[pd.DataFrame, int], dict[str, Any]] | None = None,
    selection_signature: str | None = None,
    selection_gate: Callable[[dict[str, Any]], None] | None = None,
    test_predictions_name: str = "test_predictions.parquet",
    inner_validation_partition: int = 0,
    checkpoint_selection_metric: str = "event_f1",
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("DTP-SQF training requires the RTX 4080 CUDA environment")
    seed = int(training_config["random_seed"])
    seed_everything(seed)
    if inner_validation_partition not in range(3):
        raise ValueError("inner_validation_partition must be 0, 1, or 2")
    train_anchors, validation_anchors, test_anchors = assign_train_validation_test(
        anchors,
        subject_folds,
        outer_fold,
        inner_validation_partition=inner_validation_partition,
    )
    train_anchors = train_anchors.reset_index(drop=True)
    validation_anchors = validation_anchors.reset_index(drop=True)
    test_anchors = test_anchors.reset_index(drop=True)
    if train_anchors.empty or validation_anchors.empty or test_anchors.empty:
        raise ValueError(
            f"Fold {outer_fold} must have non-empty train, validation, and test anchors"
        )
    train_subjects = set(train_anchors["subject_key"].unique())
    normalization = compute_normalization(segments, train_subjects)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_normalization(normalization, output_dir / "normalization.json")

    future_seconds = int(model_config.get("future_context_seconds", 0))
    train_dataset = DTPDataset(
        train_anchors,
        segments,
        normalization,
        future_context_seconds=future_seconds,
        training=True,
        ppg_augmentation_probability=float(training_config["ppg_augmentation_probability"]),
        seed=seed,
        motion_block_seconds=int(model_config.get("motion_block_seconds", 3)),
        ppg_block_seconds=int(model_config.get("ppg_block_seconds", 15)),
        motion_bucket_counts=model_config["motion_bucket_counts"],
        ppg_bucket_counts=model_config["ppg_bucket_counts"],
    )
    use_checkpoint_subset = (
        checkpoint_selection_metric == "window_auprc" and validation_selector is None
    )
    if use_checkpoint_subset:
        checkpoint_validation_anchors = select_checkpoint_validation_anchors(
            validation_anchors,
            int(training_config.get("checkpoint_validation_max_rows", len(validation_anchors))),
            seed,
        )
    else:
        checkpoint_validation_anchors = validation_anchors
    checkpoint_validation_dataset = DTPDataset(
        checkpoint_validation_anchors,
        segments,
        normalization,
        future_context_seconds=future_seconds,
        training=False,
        seed=seed,
        motion_block_seconds=int(model_config.get("motion_block_seconds", 3)),
        ppg_block_seconds=int(model_config.get("ppg_block_seconds", 15)),
        motion_bucket_counts=model_config["motion_bucket_counts"],
        ppg_bucket_counts=model_config["ppg_bucket_counts"],
    )
    test_dataset = DTPDataset(
        test_anchors,
        segments,
        normalization,
        future_context_seconds=future_seconds,
        training=False,
        seed=seed,
        motion_block_seconds=int(model_config.get("motion_block_seconds", 3)),
        ppg_block_seconds=int(model_config.get("ppg_block_seconds", 15)),
        motion_bucket_counts=model_config["motion_bucket_counts"],
        ppg_bucket_counts=model_config["ppg_bucket_counts"],
    )
    batch_sampler = SegmentBalancedBatchSampler(
        train_anchors,
        batch_size=int(training_config["batch_size"]),
        steps_per_epoch=int(training_config["steps_per_epoch"]),
        positive_fraction=float(training_config["positive_sampling_fraction"]),
        seed=seed,
    )
    training_workers = int(training_config["num_workers"])
    inference_workers = int(
        training_config.get("inference_num_workers", min(training_workers, 2))
    )
    if training_workers < 0 or inference_workers < 0:
        raise ValueError("DataLoader worker counts must be non-negative")
    training_loader_arguments = {
        "num_workers": training_workers,
        "pin_memory": True,
        "persistent_workers": training_workers > 0,
    }
    inference_loader_arguments = {
        "num_workers": inference_workers,
        "pin_memory": True,
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        **training_loader_arguments,
    )
    checkpoint_validation_loader = DataLoader(
        checkpoint_validation_dataset,
        batch_size=int(training_config["inference_batch_size"]),
        shuffle=False,
        **inference_loader_arguments,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(training_config["inference_batch_size"]),
        shuffle=False,
        **inference_loader_arguments,
    )

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = DTPSQF(model_config).to(device)
    encoder_parameters = list(model.motion_encoder.parameters()) + list(
        model.ppg_encoder.parameters()
    )
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]
    optimizer = AdamW(
        [
            {"params": encoder_parameters, "lr": float(training_config["encoder_learning_rate"])},
            {"params": head_parameters, "lr": float(training_config["learning_rate"])},
        ],
        weight_decay=float(training_config["weight_decay"]),
    )
    accumulation = int(training_config["gradient_accumulation"])
    max_epochs = int(training_config["max_epochs"])
    if accumulation <= 0 or max_epochs <= 0:
        raise ValueError("gradient_accumulation and max_epochs must be positive")
    total_steps = math.ceil(len(train_loader) / accumulation) * max_epochs
    scheduler = LambdaLR(
        optimizer,
        _learning_rate_schedule(float(training_config["warmup_fraction"]), total_steps),
    )
    train_state_mask = train_anchors.get(
        "state_loss_mask", pd.Series(1.0, index=train_anchors.index)
    )
    eligible_train = train_anchors[train_state_mask.fillna(0.0).astype(float) > 0]
    positive_rate = float((eligible_train["state_target"] > 0).mean())
    positive_alpha = resolve_focal_positive_alpha(training_config)
    criterion = DTPLoss(
        positive_alpha=positive_alpha,
        focal_gamma=float(loss_config["focal_gamma"]),
        dice_weight=float(loss_config["dice_weight"]),
        boundary_weight=float(loss_config["boundary_weight"]),
        sqi_weight=float(loss_config["sqi_weight"]),
        boundary_positive_weight=float(training_config["boundary_positive_weight"]),
    ).to(device)
    amp_name = str(training_config["amp_dtype"]).lower()
    amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    start_epoch = 0
    best_f1 = -1.0
    best_boundary_mae = float("inf")
    best_validation_auprc = -math.inf
    patience = 0
    history: list[dict[str, float]] = []
    best_selection: dict[str, Any] | None = None
    best_selection_rank: tuple[float, ...] | None = None
    checkpoint_path = output_dir / "best.pt"
    last_checkpoint_path = output_dir / "last.pt"
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint["best_f1"])
        best_boundary_mae = float(checkpoint["best_boundary_mae"])
        best_validation_auprc = float(checkpoint.get("best_validation_auprc", -math.inf))
        patience = int(checkpoint.get("patience", 0))
        history = list(checkpoint.get("history", []))
        best_selection = checkpoint.get("best_selection")
        stored_rank = checkpoint.get("best_selection_rank")
        best_selection_rank = tuple(float(value) for value in stored_rank) if stored_rank else None
        stored_signature = checkpoint.get("selection_signature")
        if stored_signature != selection_signature:
            raise RuntimeError(
                "Resume checkpoint signature does not match the active cross-fit configuration"
            )
        if int(checkpoint.get("inner_validation_partition", 0)) != inner_validation_partition:
            raise RuntimeError("Resume checkpoint belongs to a different inner partition")
        if checkpoint.get("checkpoint_selection_metric", "event_f1") != checkpoint_selection_metric:
            raise RuntimeError("Resume checkpoint uses a different checkpoint selection metric")
        tqdm.write(f"Resuming DTP-SQF at epoch {start_epoch + 1}/{max_epochs}")

    validation_subjects = set(validation_anchors["subject_key"].unique())
    validation_truth, validation_ignore = partition_evaluation_events(events, validation_subjects)
    validation_interval = int(training_config["validation_every_epochs"])
    if validation_interval <= 0:
        raise ValueError("validation_every_epochs must be positive")
    if use_checkpoint_subset:
        tqdm.write(
            "Checkpoint selection validation: "
            f"{len(checkpoint_validation_anchors)}/{len(validation_anchors)} evaluable anchors"
        )
    for epoch in range(start_epoch, max_epochs):
        batch_sampler.set_epoch(epoch)
        # Deriving randomness from fold and epoch makes end-of-epoch resume equivalent
        # to an uninterrupted run, including dropout and CUDA stochastic operations.
        seed_everything(epoch_random_seed(seed, outer_fold, epoch))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{max_epochs}",
            unit="batch",
            leave=False,
        )
        for batch_index, batch in enumerate(train_progress):
            moved = _move_batch(batch, device)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                output = model(moved)
                loss, _ = criterion(output, moved)
                scaled_loss = loss / accumulation
            loss_value = float(loss.detach().cpu())
            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch + 1}, batch {batch_index + 1}"
                )
            scaler.scale(scaled_loss).backward()
            running_loss += loss_value
            train_progress.set_postfix(loss=f"{running_loss / (batch_index + 1):.4f}")
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training_config["gradient_clip_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        should_validate = should_validate_epoch(epoch, validation_interval)
        if not should_validate:
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": running_loss / max(len(train_loader), 1),
                    "validation_auprc": float("nan"),
                    "validation_f1": float("nan"),
                    "validation_boundary_mae_seconds": float("nan"),
                }
            )
            pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
            _save_torch_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_f1": best_f1,
                    "best_boundary_mae": best_boundary_mae,
                    "best_validation_auprc": best_validation_auprc,
                    "patience": patience,
                    "history": history,
                    "best_selection": best_selection,
                    "best_selection_rank": best_selection_rank,
                    "selection_signature": selection_signature,
                    "model_config": model_config,
                    "training_config": training_config,
                    "outer_fold": outer_fold,
                    "inner_validation_partition": inner_validation_partition,
                    "checkpoint_selection_metric": checkpoint_selection_metric,
                },
                last_checkpoint_path,
            )
            tqdm.write(
                f"Epoch {epoch + 1}/{max_epochs}: "
                f"train_loss={history[-1]['train_loss']:.4f}; validation skipped"
            )
            continue

        validation_predictions, validation_auprc = _prediction_frame(
            model,
            checkpoint_validation_loader,
            device,
            amp_dtype,
            description=f"Validating epoch {epoch + 1}",
        )
        current_selection: dict[str, Any] | None = None
        if checkpoint_selection_metric == "window_auprc" and validation_selector is None:
            metrics = {
                "f1": float("nan"),
                "start_mae_seconds": float("nan"),
                "end_mae_seconds": float("nan"),
            }
        elif validation_selector is not None:
            current_selection = validation_selector(validation_predictions, epoch)
            selected_metrics = current_selection["metrics"]
            metrics = {
                "f1": float(selected_metrics["f1"]),
                "start_mae_seconds": float(selected_metrics["start_mae_seconds"]),
                "end_mae_seconds": float(selected_metrics["end_mae_seconds"]),
            }
        elif "search" in postprocess_config:
            selected_postprocess, _ = tune_postprocess_parameters(
                validation_predictions,
                validation_truth,
                postprocess_config["search"],
                float(postprocess_config["iou_threshold"]),
                show_progress=False,
                ignore=validation_ignore,
                matching_method=str(
                    postprocess_config.get("matching_method", "max_cardinality_iou")
                ),
            )
            validation_events = probabilities_to_events(
                validation_predictions, **selected_postprocess
            )
        else:
            validation_events = probabilities_to_events(
                validation_predictions, **_postprocess_kwargs(postprocess_config)
            )
        if current_selection is None and checkpoint_selection_metric != "window_auprc":
            metrics, _ = evaluate_events(
                validation_truth,
                validation_events,
                iou_threshold=float(postprocess_config["iou_threshold"]),
                method=str(postprocess_config.get("matching_method", "max_cardinality_iou")),
                ignore=validation_ignore,
            )
        boundary_values = [
            value
            for value in (metrics["start_mae_seconds"], metrics["end_mae_seconds"])
            if np.isfinite(value)
        ]
        boundary_mae = float(np.mean(boundary_values)) if boundary_values else float("inf")
        if current_selection is not None:
            current_rank = tuple(float(value) for value in current_selection["rank"])
        else:
            current_rank = checkpoint_selection_rank(
                checkpoint_selection_metric,
                validation_auprc,
                metrics,
                epoch,
            )
        improved = best_selection_rank is None or current_rank > best_selection_rank
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / max(len(train_loader), 1),
                "validation_auprc": validation_auprc,
                "validation_f1": metrics["f1"],
                "validation_boundary_mae_seconds": boundary_mae,
            }
        )
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        if improved:
            best_f1 = metrics["f1"]
            best_boundary_mae = boundary_mae
            best_validation_auprc = validation_auprc
            best_selection = current_selection
            best_selection_rank = current_rank
            patience = 0
        else:
            patience += 1
        checkpoint_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_f1": best_f1,
            "best_boundary_mae": best_boundary_mae,
            "best_validation_auprc": best_validation_auprc,
            "patience": patience,
            "history": history,
            "best_selection": best_selection,
            "best_selection_rank": best_selection_rank,
            "selection_signature": selection_signature,
            "model_config": model_config,
            "training_config": training_config,
            "outer_fold": outer_fold,
            "inner_validation_partition": inner_validation_partition,
            "checkpoint_selection_metric": checkpoint_selection_metric,
        }
        if improved:
            _save_torch_checkpoint(checkpoint_payload, checkpoint_path)
            validation_predictions.to_parquet(
                output_dir
                / (
                    "best_checkpoint_validation_predictions.parquet"
                    if use_checkpoint_subset
                    else "best_validation_predictions.parquet"
                ),
                index=False,
            )
        _save_torch_checkpoint(checkpoint_payload, last_checkpoint_path)
        tqdm.write(
            f"Epoch {epoch + 1}/{max_epochs}: "
            f"train_loss={history[-1]['train_loss']:.4f}, "
            f"validation_auprc={validation_auprc:.4f}, "
            f"validation_f1={metrics['f1']:.4f}, patience={patience}"
        )
        if patience >= int(training_config["early_stopping_epochs"]):
            tqdm.write("Early stopping threshold reached.")
            break

    if not checkpoint_path.exists():
        raise RuntimeError("Training ended without producing a best checkpoint")
    if validation_selector is not None:
        if best_selection is None:
            raise RuntimeError("Fusion training ended without a selected validation candidate")
        (output_dir / "best_validation_selection.json").write_text(
            json.dumps(_json_safe(best_selection), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if selection_gate is not None:
            selection_gate(best_selection)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if use_checkpoint_subset:
        full_validation_dataset = DTPDataset(
            validation_anchors,
            segments,
            normalization,
            future_context_seconds=future_seconds,
            training=False,
            seed=seed,
            motion_block_seconds=int(model_config.get("motion_block_seconds", 3)),
            ppg_block_seconds=int(model_config.get("ppg_block_seconds", 15)),
            motion_bucket_counts=model_config["motion_bucket_counts"],
            ppg_bucket_counts=model_config["ppg_bucket_counts"],
        )
        full_validation_loader = DataLoader(
            full_validation_dataset,
            batch_size=int(training_config["inference_batch_size"]),
            shuffle=False,
            **inference_loader_arguments,
        )
        full_validation_predictions, full_validation_auprc = _prediction_frame(
            model,
            full_validation_loader,
            device,
            amp_dtype,
            description="Predicting full validation partition",
        )
        full_validation_predictions.to_parquet(
            output_dir / "best_validation_predictions.parquet", index=False
        )
    else:
        full_validation_auprc = best_validation_auprc
    test_predictions, test_auprc = _prediction_frame(
        model, test_loader, device, amp_dtype, description="Predicting test fold"
    )
    test_predictions.to_parquet(output_dir / test_predictions_name, index=False)
    metadata = {
        "outer_fold": outer_fold,
        "inner_validation_partition": inner_validation_partition,
        "checkpoint_selection_metric": checkpoint_selection_metric,
        "best_checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_f1": best_f1,
        "best_validation_auprc": best_validation_auprc,
        "full_validation_auprc": full_validation_auprc,
        "best_validation_boundary_mae_seconds": best_boundary_mae
        if np.isfinite(best_boundary_mae)
        else None,
        "test_window_auprc": test_auprc,
        "train_rows": len(train_anchors),
        "validation_rows": len(validation_anchors),
        "validation_full_timeline_rows": len(validation_anchors),
        "checkpoint_validation_rows": len(checkpoint_validation_anchors),
        "checkpoint_validation_strategy": "deterministic_stratified_evaluable"
        if use_checkpoint_subset
        else "full_timeline",
        "checkpoint_validation_seed": seed,
        "checkpoint_validation_positive_rate": float(
            (checkpoint_validation_anchors["state_target"] > 0).mean()
        ),
        "test_rows": len(test_anchors),
        "train_subjects": sorted(str(value) for value in train_subjects),
        "validation_subjects": sorted(str(value) for value in validation_subjects),
        "focal_positive_alpha": positive_alpha,
        "raw_training_positive_rate": positive_rate,
        "training_num_workers": training_workers,
        "inference_num_workers": inference_workers,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "validation_selection_signature": selection_signature,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return checkpoint_path
