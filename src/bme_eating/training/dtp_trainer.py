from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from bme_eating.data.deep_dataset import (
    DTPDataset,
    SegmentBalancedBatchSampler,
    compute_normalization,
    save_normalization,
)
from bme_eating.metrics import evaluate_events
from bme_eating.models.dtp_sqf import DTPSQF
from bme_eating.models.losses import DTPLoss
from bme_eating.models.xgb_baseline import assign_train_validation_test
from bme_eating.postprocess import probabilities_to_events


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


def _validation_proxy(
    anchors: pd.DataFrame,
    background_segments_per_subject: int,
) -> pd.DataFrame:
    near_segment_ids = set(
        anchors.loc[anchors["distance_to_event_seconds"] <= 1800.0, "segment_id"].unique()
    )
    selected_segment_ids = set(near_segment_ids)
    for _, subject_rows in anchors.groupby("subject_key", sort=True):
        background_ids = sorted(
            set(subject_rows["segment_id"].unique()) - near_segment_ids
        )[:background_segments_per_subject]
        selected_segment_ids.update(background_ids)
    return anchors[anchors["segment_id"].isin(selected_segment_ids)].reset_index(drop=True)


def _prediction_frame(
    model: DTPSQF,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[pd.DataFrame, float]:
    model.eval()
    rows: list[dict[str, object]] = []
    targets: list[float] = []
    probabilities: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            moved = _move_batch(batch, device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                output = model(moved)
            state = torch.sigmoid(output["state_logit"]).cpu().numpy()
            start = torch.sigmoid(output["start_logit"]).cpu().numpy()
            end = torch.sigmoid(output["end_logit"]).cpu().numpy()
            target = batch["state_target"].numpy()
            targets.extend(target.tolist())
            probabilities.extend(state.tolist())
            for index in range(len(state)):
                rows.append(
                    {
                        "subject_key": batch["subject_key"][index],
                        "segment_id": batch["segment_id"][index],
                        "timestamp_ms": int(batch["timestamp_ms"][index]),
                        "state_probability": float(state[index]),
                        "start_probability": float(start[index]),
                        "end_probability": float(end[index]),
                    }
                )
    auprc = average_precision_score(np.asarray(targets) > 0, probabilities) if targets else 0.0
    return pd.DataFrame(rows), float(auprc)


def _postprocess_kwargs(config: dict[str, Any]) -> dict[str, float]:
    return {
        "ema_half_life_seconds": float(config["ema_half_life_seconds"]),
        "high_threshold": float(config["high_threshold"]),
        "low_threshold": float(config["low_threshold"]),
        "minimum_event_seconds": float(config["minimum_event_seconds"]),
        "merge_gap_seconds": float(config["merge_gap_seconds"]),
        "boundary_lookback_seconds": float(config["boundary_lookback_seconds"]),
    }


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
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("DTP-SQF training requires the RTX 4080 CUDA environment")
    seed = int(training_config["random_seed"])
    seed_everything(seed)
    train_anchors, validation_anchors, test_anchors = assign_train_validation_test(
        anchors, subject_folds, outer_fold
    )
    train_anchors = train_anchors.reset_index(drop=True)
    validation_anchors = validation_anchors.reset_index(drop=True)
    test_anchors = test_anchors.reset_index(drop=True)
    validation_proxy = _validation_proxy(
        validation_anchors,
        int(training_config["validation_background_segments_per_subject"]),
    )
    train_subjects = set(train_anchors["subject_key"].unique())
    normalization = compute_normalization(segments, train_subjects)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_normalization(normalization, output_dir / "normalization.json")

    future_seconds = int(model_config.get("future_context_seconds", 0))
    train_dataset = DTPDataset(
        train_anchors,
        normalization,
        future_context_seconds=future_seconds,
        training=True,
        ppg_augmentation_probability=float(training_config["ppg_augmentation_probability"]),
        seed=seed,
    )
    validation_dataset = DTPDataset(
        validation_proxy,
        normalization,
        future_context_seconds=future_seconds,
        training=False,
        seed=seed,
    )
    test_dataset = DTPDataset(
        test_anchors,
        normalization,
        future_context_seconds=future_seconds,
        training=False,
        seed=seed,
    )
    batch_sampler = SegmentBalancedBatchSampler(
        train_anchors,
        batch_size=int(training_config["batch_size"]),
        steps_per_epoch=int(training_config["steps_per_epoch"]),
        positive_fraction=float(training_config["positive_sampling_fraction"]),
        seed=seed,
    )
    loader_arguments = {
        "num_workers": int(training_config["num_workers"]),
        "pin_memory": True,
        "persistent_workers": int(training_config["num_workers"]) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        **loader_arguments,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training_config["inference_batch_size"]),
        shuffle=False,
        **loader_arguments,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(training_config["inference_batch_size"]),
        shuffle=False,
        **loader_arguments,
    )

    device = torch.device("cuda")
    model = DTPSQF(model_config).to(device)
    encoder_parameters = list(model.motion_encoder.parameters()) + list(model.ppg_encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    optimizer = AdamW(
        [
            {"params": encoder_parameters, "lr": float(training_config["encoder_learning_rate"])},
            {"params": head_parameters, "lr": float(training_config["learning_rate"])},
        ],
        weight_decay=float(training_config["weight_decay"]),
    )
    accumulation = int(training_config["gradient_accumulation"])
    total_steps = (
        math.ceil(len(train_loader) / accumulation) * int(training_config["max_epochs"])
    )
    scheduler = LambdaLR(
        optimizer,
        _learning_rate_schedule(float(training_config["warmup_fraction"]), total_steps),
    )
    positive_rate = float((train_anchors["state_target"] > 0).mean())
    positive_alpha = float(np.clip(1.0 - positive_rate, 0.5, 0.95))
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
    patience = 0
    history: list[dict[str, float]] = []
    checkpoint_path = output_dir / "best.pt"
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint["best_f1"])
        best_boundary_mae = float(checkpoint["best_boundary_mae"])

    validation_subjects = set(validation_anchors["subject_key"].unique())
    validation_truth = events[
        events["subject_key"].isin(validation_subjects)
        & events["valid_duration"]
        & (events["coverage"] == "full")
    ]
    for epoch in range(start_epoch, int(training_config["max_epochs"])):
        batch_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for batch_index, batch in enumerate(train_loader):
            moved = _move_batch(batch, device)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                output = model(moved)
                loss, _ = criterion(output, moved)
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.detach().cpu())
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training_config["gradient_clip_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        validation_interval = int(training_config["validation_every_epochs"])
        should_validate = epoch == start_epoch or (epoch + 1) % validation_interval == 0
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
            continue

        validation_predictions, validation_auprc = _prediction_frame(
            model, validation_loader, device, amp_dtype
        )
        validation_events = probabilities_to_events(
            validation_predictions, **_postprocess_kwargs(postprocess_config)
        )
        metrics, _ = evaluate_events(
            validation_truth,
            validation_events,
            iou_threshold=float(postprocess_config["iou_threshold"]),
            method="hungarian",
        )
        boundary_values = [
            value
            for value in (metrics["start_mae_seconds"], metrics["end_mae_seconds"])
            if np.isfinite(value)
        ]
        boundary_mae = float(np.mean(boundary_values)) if boundary_values else float("inf")
        improved = metrics["f1"] > best_f1 or (
            math.isclose(metrics["f1"], best_f1) and boundary_mae < best_boundary_mae
        )
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
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_f1": best_f1,
                    "best_boundary_mae": best_boundary_mae,
                    "model_config": model_config,
                    "training_config": training_config,
                    "outer_fold": outer_fold,
                },
                checkpoint_path,
            )
            validation_predictions.to_parquet(
                output_dir / "best_validation_predictions.parquet", index=False
            )
        else:
            patience += 1
            if patience >= int(training_config["early_stopping_epochs"]):
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_predictions, test_auprc = _prediction_frame(model, test_loader, device, amp_dtype)
    test_predictions.to_parquet(output_dir / "test_predictions.parquet", index=False)
    metadata = {
        "outer_fold": outer_fold,
        "best_validation_f1": best_f1,
        "best_validation_boundary_mae_seconds": best_boundary_mae
        if np.isfinite(best_boundary_mae)
        else None,
        "test_window_auprc": test_auprc,
        "train_rows": len(train_anchors),
        "validation_rows": len(validation_anchors),
        "validation_proxy_rows": len(validation_proxy),
        "test_rows": len(test_anchors),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return checkpoint_path
