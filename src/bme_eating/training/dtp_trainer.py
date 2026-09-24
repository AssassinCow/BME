from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from bme_eating.data.deep_dataset import (
    DTPDataset,
    Normalization,
    SegmentBalancedBatchSampler,
    compute_normalization,
    load_normalization,
    save_normalization,
)
from bme_eating.metrics import (
    evaluate_events,
    masked_average_precision,
    partition_evaluation_events,
)
from bme_eating.models.dtp_sqf import DTPSQF, logits_to_probability_arrays
from bme_eating.models.factory import build_state_model
from bme_eating.models.losses import DTPLoss, HierarchicalStateLoss
from bme_eating.models.xgb_baseline import assign_train_validation_test
from bme_eating.postprocess import probabilities_to_events, tune_postprocess_parameters
from bme_eating.reproducibility import epoch_random_seed, should_validate_epoch

PREDICTION_COLUMNS = (
    "subject_key",
    "segment_id",
    "session_id",
    "timestamp_ms",
    "state_probability",
    "start_probability",
    "end_probability",
)
PREDICTION_KEYS = ("subject_key", "segment_id", "session_id", "timestamp_ms")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configured_state_feature_columns(model_config: dict[str, Any]) -> tuple[str, ...]:
    if not bool(model_config.get("use_stable_state_features", False)):
        return ()
    return tuple(str(value) for value in model_config.get("stable_feature_columns", ()))


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
    model: nn.Module,
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
            ppg_gate = output["ppg_gate"].detach().float().cpu().numpy()
            ppg_valid = batch["ppg_valid"].numpy().astype(np.float64)
            motion_valid = batch["motion_valid"].numpy().astype(np.float64)
            valid_gate_count = np.maximum(ppg_valid.sum(axis=1), 1.0)
            ppg_gate_mean = (ppg_gate * ppg_valid).sum(axis=1) / valid_gate_count
            ppg_gate_recent = ppg_gate[:, -1]
            ppg_valid_fraction = ppg_valid.mean(axis=1)
            motion_valid_fraction = motion_valid.mean(axis=1)
            state_logit = output["state_logit"].detach().float().cpu().numpy()
            start_logit = output["start_logit"].detach().float().cpu().numpy()
            end_logit = output["end_logit"].detach().float().cpu().numpy()
            embedding = output.get("state_embedding")
            embedding_array = (
                embedding.detach().float().cpu().numpy() if embedding is not None else None
            )
            missing = output.get("missing_fraction")
            missing_array = (
                missing.detach().float().cpu().numpy().reshape(-1)
                if missing is not None
                else 1.0 - 0.5 * (ppg_valid_fraction + motion_valid_fraction)
            )
            for index in range(len(state)):
                row: dict[str, object] = {
                        "subject_key": batch["subject_key"][index],
                        "segment_id": batch["segment_id"][index],
                        "session_id": batch["session_id"][index],
                        "timestamp_ms": int(batch["timestamp_ms"][index]),
                        "state_probability": float(state[index]),
                        "start_probability": float(start[index]),
                        "end_probability": float(end[index]),
                        "ppg_gate_mean": float(ppg_gate_mean[index]),
                        "ppg_gate_recent": float(ppg_gate_recent[index]),
                        "ppg_valid_fraction": float(ppg_valid_fraction[index]),
                        "motion_valid_fraction": float(motion_valid_fraction[index]),
                        "missing_fraction": float(missing_array[index]),
                        "state_logit": float(state_logit[index]),
                        "start_logit": float(start_logit[index]),
                        "end_logit": float(end_logit[index]),
                    }
                if embedding_array is not None:
                    row.update(
                        {
                            f"state_embedding_{dimension:02d}": float(value)
                            for dimension, value in enumerate(embedding_array[index])
                        }
                    )
                rows.append(row)
    auprc = masked_average_precision(
        np.asarray(targets), np.asarray(probabilities), np.asarray(state_masks)
    )
    frame = pd.DataFrame(rows)
    for column in (
        "state_probability",
        "start_probability",
        "end_probability",
        "ppg_gate_mean",
        "ppg_gate_recent",
        "ppg_valid_fraction",
        "motion_valid_fraction",
        "missing_fraction",
        "state_logit",
        "start_logit",
        "end_logit",
    ):
        if column in frame:
            frame[column] = frame[column].astype(np.float32)
    return frame, float(auprc)


def _complete_session_chunks(
    anchors: pd.DataFrame,
    maximum_rows: int,
) -> list[pd.DataFrame]:
    if maximum_rows <= 0:
        raise ValueError("inference_resume_chunk_rows must be positive")
    required = {"subject_key", "session_id"}
    missing = required - set(anchors.columns)
    if missing:
        raise ValueError(f"Resumable inference requires columns: {sorted(missing)}")
    if anchors.empty:
        return []
    grouped_positions = list(
        anchors.groupby(["subject_key", "session_id"], sort=False, dropna=False).indices.values()
    )
    chunks: list[pd.DataFrame] = []
    current: list[np.ndarray] = []
    current_rows = 0
    for positions in grouped_positions:
        positions = np.asarray(positions, dtype=np.int64)
        if current and current_rows + len(positions) > maximum_rows:
            selected = np.concatenate(current)
            chunks.append(anchors.iloc[selected].reset_index(drop=True))
            current = []
            current_rows = 0
        current.append(positions)
        current_rows += len(positions)
    if current:
        selected = np.concatenate(current)
        chunks.append(anchors.iloc[selected].reset_index(drop=True))
    return chunks


def _prepare_stable_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    columns: tuple[str, ...],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not columns:
        return train, validation, test
    missing = sorted(set(columns) - set(train.columns))
    if missing:
        raise ValueError(f"Training anchors are missing stable features: {missing}")
    values = train.loc[:, columns].to_numpy(dtype=np.float64)
    median = np.nanmedian(values, axis=0)
    upper = np.nanpercentile(values, 75, axis=0)
    lower = np.nanpercentile(values, 25, axis=0)
    scale = np.where(upper - lower > 1e-6, upper - lower, 1.0)
    if not np.isfinite(median).all() or not np.isfinite(scale).all():
        raise ValueError("Stable feature normalization is not finite")

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy()
        array = transformed.loc[:, columns].to_numpy(dtype=np.float64)
        array = np.nan_to_num(array, nan=median, posinf=median, neginf=median)
        transformed.loc[:, columns] = np.clip((array - median) / scale, -10.0, 10.0)
        return transformed

    _write_json_atomic(
        {"columns": list(columns), "median": median.tolist(), "iqr": scale.tolist()},
        output_dir / "stable_feature_normalization.json",
    )
    return transform(train), transform(validation), transform(test)


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


def _session_bounds(frame: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic timestamp bounds for each subject/session pair."""

    required = {"subject_key", "session_id", "timestamp_ms"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Session sampling requires columns: {sorted(missing)}")
    normalized = frame.loc[:, ["subject_key", "session_id", "timestamp_ms"]].copy()
    normalized["_subject_key"] = normalized["subject_key"].astype(str)
    normalized["_session_id"] = normalized["session_id"].astype(str)
    bounds = (
        normalized.groupby(["_subject_key", "_session_id"], sort=True, dropna=False)[
            "timestamp_ms"
        ]
        .agg(start_ms="min", end_ms="max", row_count="size")
        .reset_index()
        .rename(columns={"_subject_key": "subject_norm", "_session_id": "session_norm"})
    )
    return bounds


def _attach_event_sessions(events: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """Map event intervals to the anchor session they overlap most strongly."""

    if events.empty:
        output = events.copy()
        output["_checkpoint_session_key"] = pd.Series(dtype=object, index=output.index)
        return output
    required = {"subject_key", "start_ms", "end_ms"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Event session mapping requires columns: {sorted(missing)}")
    bounds = _session_bounds(anchors)
    by_subject: dict[str, list[tuple[str, int, int]]] = {}
    known_keys: set[tuple[str, str]] = set()
    for row in bounds.itertuples(index=False):
        subject = str(row.subject_norm)
        session = str(row.session_norm)
        item = (session, int(row.start_ms), int(row.end_ms))
        by_subject.setdefault(subject, []).append(item)
        known_keys.add((subject, session))

    assignments: list[tuple[str, str] | None] = []
    for event in events.itertuples(index=False):
        subject = str(event.subject_key)
        try:
            start = int(event.start_ms)
            end = int(event.end_ms)
        except (TypeError, ValueError):
            assignments.append(None)
            continue
        if end <= start or subject not in by_subject:
            assignments.append(None)
            continue
        explicit_session = getattr(event, "session_id", None)
        if explicit_session is not None and not pd.isna(explicit_session):
            explicit_key = (subject, str(explicit_session))
            if explicit_key in known_keys:
                assignments.append(explicit_key)
                continue
        candidates = []
        for session, session_start, session_end in by_subject[subject]:
            overlap = min(end, session_end) - max(start, session_start)
            if overlap > 0:
                candidates.append((overlap, session_start, session))
        if not candidates:
            assignments.append(None)
            continue
        _, _, session = max(candidates, key=lambda value: (value[0], -value[1], value[2]))
        assignments.append((subject, session))
    output = events.copy()
    output["_checkpoint_session_key"] = assignments
    return output


def events_overlapping_sessions(events: pd.DataFrame, session_frame: pd.DataFrame) -> pd.DataFrame:
    """Keep events that overlap at least one session represented by ``session_frame``."""

    if events.empty or session_frame.empty:
        return events.iloc[0:0].copy()
    mapped = _attach_event_sessions(events, session_frame)
    selected_keys = set(zip(
        session_frame["subject_key"].astype(str),
        session_frame["session_id"].astype(str),
    ))
    mask = mapped["_checkpoint_session_key"].map(
        lambda value: value in selected_keys if value is not None else False
    )
    return mapped.loc[mask].drop(columns=["_checkpoint_session_key"]).reset_index(drop=True)


def select_checkpoint_validation_sessions(
    anchors: pd.DataFrame,
    validation_truth: pd.DataFrame,
    validation_ignore: pd.DataFrame,
    maximum_rows: int,
    seed: int,
    *,
    minimum_subjects: int = 8,
    minimum_events: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select complete sessions for frequent validation checkpoints.

    Sessions are the atomic sampling unit: no timestamp is removed from a selected
    session.  Evaluable event sessions and subject coverage are selected first,
    followed by background sessions until the row budget is reached.
    """

    if maximum_rows <= 0:
        raise ValueError("checkpoint_validation_max_rows must be positive")
    if minimum_subjects < 0 or minimum_events < 0:
        raise ValueError("Checkpoint validation coverage minima must be non-negative")
    required = {"subject_key", "session_id", "timestamp_ms"}
    missing = required - set(anchors.columns)
    if missing:
        raise ValueError(f"Checkpoint session sampling requires columns: {sorted(missing)}")
    state_mask = anchors.get("state_loss_mask", pd.Series(1.0, index=anchors.index))
    eligible_mask = state_mask.fillna(0.0).astype(float) > 0
    if not eligible_mask.any():
        raise ValueError("Checkpoint validation has no evaluable anchors")

    normalized = anchors.copy()
    normalized["_subject_key"] = normalized["subject_key"].astype(str)
    normalized["_session_id"] = normalized["session_id"].astype(str)
    eligible_sessions = set(
        zip(
            normalized.loc[eligible_mask, "_subject_key"],
            normalized.loc[eligible_mask, "_session_id"],
        )
    )
    mapped_truth = _attach_event_sessions(validation_truth, anchors)
    mapped_truth = mapped_truth[mapped_truth["_checkpoint_session_key"].notna()].copy()
    event_counts = mapped_truth["_checkpoint_session_key"].value_counts().to_dict()

    records: list[dict[str, Any]] = []
    for (subject, session), group in normalized.groupby(
        ["_subject_key", "_session_id"], sort=True, dropna=False
    ):
        key = (str(subject), str(session))
        if key not in eligible_sessions:
            continue
        records.append(
            {
                "key": key,
                "subject": str(subject),
                "rows": len(group),
                "events": int(event_counts.get(key, 0)),
            }
        )
    if not records:
        raise ValueError("Checkpoint validation has no complete eligible sessions")

    rng = np.random.default_rng(seed)
    tie_order = {
        records[int(index)]["key"]: int(rank)
        for rank, index in enumerate(rng.permutation(len(records)))
    }
    available_subjects = {record["subject"] for record in records}
    available_events = int(sum(record["events"] for record in records))
    target_subjects = min(int(minimum_subjects), len(available_subjects))
    target_events = min(int(minimum_events), available_events)
    selected_set: set[tuple[str, str]] = set()
    selected_rows = 0

    def add(record: dict[str, Any]) -> None:
        nonlocal selected_rows
        key = record["key"]
        if key not in selected_set:
            selected_set.add(key)
            selected_rows += int(record["rows"])

    by_subject: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_subject.setdefault(record["subject"], []).append(record)
    event_subjects = {record["subject"] for record in records if record["events"] > 0}
    if len(event_subjects) < target_subjects:
        for subject in sorted(by_subject):
            if len({item[0] for item in selected_set}) >= target_subjects:
                break
            candidates = sorted(
                by_subject[subject],
                key=lambda item: (-item["events"], item["rows"], tie_order[item["key"]]),
            )
            if candidates:
                add(candidates[0])

    event_candidates = sorted(
        records,
        key=lambda item: (-item["events"], item["rows"], tie_order[item["key"]]),
    )
    for record in event_candidates:
        selected_subjects = {item[0] for item in selected_set}
        selected_events = sum(item["events"] for item in records if item["key"] in selected_set)
        if selected_events >= target_events and len(selected_subjects) >= target_subjects:
            break
        add(record)

    selected_subjects = {item[0] for item in selected_set}
    for subject in sorted(available_subjects - selected_subjects):
        if len(selected_subjects) >= target_subjects:
            break
        candidates = sorted(by_subject[subject], key=lambda item: (item["rows"], tie_order[item["key"]]))
        if candidates:
            add(candidates[0])
            selected_subjects.add(subject)

    remaining = [record for record in records if record["key"] not in selected_set]
    remaining.sort(key=lambda item: (item["events"] == 0, item["rows"], tie_order[item["key"]]))
    for record in remaining:
        if selected_rows >= maximum_rows:
            break
        if selected_rows + int(record["rows"]) <= maximum_rows:
            add(record)

    selected_frame = normalized[
        normalized.apply(lambda row: (row["_subject_key"], row["_session_id"]) in selected_set, axis=1)
    ].drop(columns=["_subject_key", "_session_id"])
    selected_frame = selected_frame.reset_index(drop=True)
    selected_truth = events_overlapping_sessions(validation_truth, selected_frame)
    selected_ignore = events_overlapping_sessions(validation_ignore, selected_frame)
    selected_subject_count = int(selected_frame[["subject_key"]].astype(str).nunique().iloc[0])
    selected_event_count = len(selected_truth)

    coverage_satisfied = (
        selected_subject_count >= target_subjects and selected_event_count >= target_events
    )
    if not coverage_satisfied:
        selected_frame = anchors.reset_index(drop=True).copy()
        selected_truth = validation_truth.reset_index(drop=True).copy()
        selected_ignore = validation_ignore.reset_index(drop=True).copy()
        selected_set = set(zip(
            selected_frame["subject_key"].astype(str),
            selected_frame["session_id"].astype(str),
        ))
        selected_subject_count = int(selected_frame["subject_key"].astype(str).nunique())
        selected_event_count = len(selected_truth)

    metadata = {
        "target_rows": int(maximum_rows),
        "selected_rows": len(selected_frame),
        "available_rows": len(anchors),
        "selected_sessions": len(selected_set),
        "available_sessions": len(records),
        "selected_subjects": selected_subject_count,
        "available_subjects": len(available_subjects),
        "selected_events": selected_event_count,
        "available_events": available_events,
        "minimum_subjects": int(minimum_subjects),
        "minimum_events": int(minimum_events),
        "effective_minimum_subjects": target_subjects,
        "effective_minimum_events": target_events,
        "coverage_satisfied": bool(coverage_satisfied),
        "seed": int(seed),
        "strategy": "event_stratified_complete_sessions",
    }
    return selected_frame, selected_truth, selected_ignore, metadata


def _save_torch_checkpoint(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)


def _save_prediction_frame(frame: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


def _save_csv_frame(frame: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _prediction_manifest_path(path: Path) -> Path:
    return path.with_name(path.name + ".manifest.json")


def _expected_prediction_rows(anchors: pd.DataFrame) -> pd.DataFrame:
    expected = anchors.copy()
    if "session_id" not in expected:
        expected["session_id"] = expected["segment_id"]
    required = set(PREDICTION_KEYS) | {"state_target"}
    missing = sorted(required - set(expected.columns))
    if missing:
        raise ValueError(f"Prediction anchors are missing columns: {missing}")
    if "state_loss_mask" not in expected:
        expected["state_loss_mask"] = 1.0
    return expected[
        [*PREDICTION_KEYS, "state_target", "state_loss_mask"]
    ].sort_values(list(PREDICTION_KEYS)).reset_index(drop=True)


def validate_prediction_frame(
    predictions: pd.DataFrame,
    anchors: pd.DataFrame,
    description: str,
) -> float:
    missing = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
    if missing:
        raise ValueError(f"{description} is missing columns: {missing}")
    if predictions.duplicated(list(PREDICTION_KEYS)).any():
        raise ValueError(f"{description} contains duplicate timeline keys")
    probabilities = predictions[
        ["state_probability", "start_probability", "end_probability"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{description} contains NaN or infinite probabilities")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError(f"{description} contains probabilities outside [0, 1]")

    actual = predictions[list(PREDICTION_KEYS)].copy()
    actual["subject_key"] = actual["subject_key"].astype(str)
    actual["segment_id"] = actual["segment_id"].astype(str)
    actual["session_id"] = actual["session_id"].astype(str)
    actual["timestamp_ms"] = actual["timestamp_ms"].astype(np.int64)
    actual = actual.sort_values(list(PREDICTION_KEYS)).reset_index(drop=True)
    expected = _expected_prediction_rows(anchors)
    for column in ("subject_key", "segment_id", "session_id"):
        expected[column] = expected[column].astype(str)
    expected["timestamp_ms"] = expected["timestamp_ms"].astype(np.int64)
    if not actual.equals(expected[list(PREDICTION_KEYS)]):
        raise ValueError(f"{description} does not exactly match the expected timeline")

    ordered_predictions = predictions.sort_values(list(PREDICTION_KEYS)).reset_index(drop=True)
    return float(
        masked_average_precision(
            expected["state_target"].to_numpy(dtype=np.float64),
            ordered_predictions["state_probability"].to_numpy(dtype=np.float64),
            expected["state_loss_mask"].fillna(0.0).to_numpy(dtype=np.float64),
        )
    )


def _prediction_cache_identity(
    checkpoint_path: Path,
    selection_signature: str | None,
    outer_fold: int,
    inner_validation_partition: int,
) -> dict[str, Any]:
    return {
        "version": 2,
        "best_checkpoint_sha256": _sha256(checkpoint_path),
        "selection_signature": selection_signature,
        "outer_fold": int(outer_fold),
        "inner_validation_partition": int(inner_validation_partition),
    }


def _write_prediction_cache_manifest(
    path: Path,
    predictions: pd.DataFrame,
    auprc: float,
    identity: dict[str, Any],
) -> None:
    _write_json_atomic(
        {
            **identity,
            "prediction_file": path.name,
            "prediction_sha256": _sha256(path),
            "rows": len(predictions),
            "window_auprc": auprc,
        },
        _prediction_manifest_path(path),
    )


def _load_prediction_cache(
    path: Path,
    anchors: pd.DataFrame,
    checkpoint_path: Path,
    identity: dict[str, Any],
    description: str,
    *,
    allow_legacy_manifest: bool = False,
) -> tuple[pd.DataFrame, float] | None:
    if not path.is_file():
        return None
    try:
        predictions = pd.read_parquet(path)
        auprc = validate_prediction_frame(predictions, anchors, description)
        manifest_path = _prediction_manifest_path(path)
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key, value in identity.items():
                if manifest.get(key) != value:
                    raise ValueError(f"{description} manifest mismatch for {key}")
            if manifest.get("prediction_sha256") != _sha256(path):
                raise ValueError(f"{description} file hash does not match its manifest")
            if int(manifest.get("rows", -1)) != len(predictions):
                raise ValueError(f"{description} row count does not match its manifest")
            manifest_auprc = float(manifest.get("window_auprc"))
            if not math.isfinite(manifest_auprc) or not math.isclose(
                manifest_auprc, auprc, rel_tol=1e-7, abs_tol=1e-9
            ):
                raise ValueError(f"{description} AUPRC does not match its manifest")
        else:
            if not allow_legacy_manifest:
                raise ValueError(f"{description} has no cache manifest")
            if path.stat().st_mtime_ns < checkpoint_path.stat().st_mtime_ns:
                raise ValueError(f"{description} predates the selected checkpoint")
            _write_prediction_cache_manifest(path, predictions, auprc, identity)
        return predictions, auprc
    except (OSError, ValueError, KeyError, TypeError, pa.ArrowException) as error:
        tqdm.write(f"Ignoring invalid {description} cache: {error}")
        return None


def _resumable_prediction_frame(
    model: nn.Module,
    anchors: pd.DataFrame,
    segments: pd.DataFrame,
    normalization: Normalization,
    dataset_arguments: dict[str, Any],
    loader_arguments: dict[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype,
    output_path: Path,
    checkpoint_path: Path,
    identity: dict[str, Any],
    description: str,
    maximum_chunk_rows: int,
) -> tuple[pd.DataFrame, float]:
    chunks = _complete_session_chunks(anchors, maximum_chunk_rows)
    if not chunks:
        raise ValueError(f"{description} has no anchors")
    parts_dir = output_path.parent / f".{output_path.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for part_index, chunk in enumerate(chunks):
        part_path = parts_dir / f"part_{part_index:05d}.parquet"
        part_identity = {
            **identity,
            "cache_role": description,
            "part_index": part_index,
            "part_count": len(chunks),
        }
        cached = _load_prediction_cache(
            part_path,
            chunk,
            checkpoint_path,
            part_identity,
            f"{description} part {part_index + 1}/{len(chunks)}",
        )
        if cached is not None:
            frame, _ = cached
            frames.append(frame)
            tqdm.write(
                f"Reusing {description} part {part_index + 1}/{len(chunks)} "
                f"({len(frame)} rows)."
            )
            continue
        dataset = DTPDataset(
            chunk,
            segments,
            normalization,
            **dataset_arguments,
        )
        loader = DataLoader(
            dataset,
            shuffle=False,
            **loader_arguments,
        )
        frame, part_auprc = _prediction_frame(
            model,
            loader,
            device,
            amp_dtype,
            description=f"{description} part {part_index + 1}/{len(chunks)}",
        )
        validate_prediction_frame(
            frame,
            chunk,
            f"{description} part {part_index + 1}/{len(chunks)}",
        )
        _save_prediction_frame(frame, part_path)
        _write_prediction_cache_manifest(part_path, frame, part_auprc, part_identity)
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    auprc = validate_prediction_frame(predictions, anchors, description)
    _save_prediction_frame(predictions, output_path)
    _write_prediction_cache_manifest(output_path, predictions, auprc, identity)
    return predictions, auprc


def resume_training_is_complete(
    start_epoch: int,
    max_epochs: int,
    patience: int,
    early_stopping_patience_checks: int,
) -> bool:
    return start_epoch >= max_epochs or patience >= early_stopping_patience_checks


def resolve_early_stopping_config(training_config: dict[str, Any]) -> tuple[int, float]:
    patience_key = "early_stopping_patience_checks"
    legacy_key = "early_stopping_epochs"
    if patience_key in training_config and legacy_key in training_config:
        raise ValueError(
            f"Use only {patience_key}; {legacy_key} counted validation checks despite its name"
        )
    if patience_key in training_config:
        patience_checks = int(training_config[patience_key])
    elif legacy_key in training_config:
        patience_checks = int(training_config[legacy_key])
    else:
        raise ValueError(f"Missing training setting: {patience_key}")
    minimum_delta = float(training_config.get("early_stopping_min_delta", 0.0))
    if patience_checks <= 0:
        raise ValueError("early_stopping_patience_checks must be positive")
    if not math.isfinite(minimum_delta) or minimum_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be finite and non-negative")
    return patience_checks, minimum_delta


def update_early_stopping(
    reference_score: float | None,
    current_score: float,
    patience: int,
    minimum_delta: float,
) -> tuple[float | None, int, bool]:
    if patience < 0:
        raise ValueError("Early-stopping patience cannot be negative")
    if not math.isfinite(minimum_delta) or minimum_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be finite and non-negative")
    if not math.isfinite(current_score):
        return reference_score, patience + 1, False
    if reference_score is None or not math.isfinite(reference_score):
        return float(current_score), 0, True
    improvement = float(current_score) - float(reference_score)
    if improvement > 0.0 and improvement >= minimum_delta:
        return float(current_score), 0, True
    return reference_score, patience + 1, False


def export_dtp_quality_predictions(
    anchors: pd.DataFrame,
    segments: pd.DataFrame,
    subject_folds: dict[str, int],
    outer_fold: int,
    inner_validation_partition: int,
    checkpoint_path: Path,
    normalization_path: Path,
    output_dir: Path,
    *,
    selection_signature: str,
    inference_batch_size: int | None = None,
    inference_num_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not checkpoint_path.is_file() or not normalization_path.is_file():
        raise FileNotFoundError("DTP quality re-inference requires best.pt and normalization.json")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("outer_fold", -1)) != outer_fold:
        raise RuntimeError("DTP checkpoint belongs to a different outer fold")
    if int(checkpoint.get("inner_validation_partition", -1)) != inner_validation_partition:
        raise RuntimeError("DTP checkpoint belongs to a different inner partition")
    if checkpoint.get("selection_signature") != selection_signature:
        raise RuntimeError("DTP checkpoint selection signature does not match its source record")
    model_config = dict(checkpoint["model_config"])
    training_config = dict(checkpoint["training_config"])
    if int(model_config.get("future_context_seconds", 0)) != 0:
        raise ValueError("Quality re-inference only supports causal DTP checkpoints")
    _, validation_anchors, test_anchors = assign_train_validation_test(
        anchors,
        subject_folds,
        outer_fold,
        inner_validation_partition=inner_validation_partition,
    )
    validation_anchors = validation_anchors.reset_index(drop=True)
    test_anchors = test_anchors.reset_index(drop=True)
    if validation_anchors.empty or test_anchors.empty:
        raise ValueError("Quality re-inference requires non-empty validation and test anchors")
    normalization = load_normalization(normalization_path)
    seed = int(training_config["random_seed"])
    seed_everything(seed)
    dataset_arguments = {
        "future_context_seconds": 0,
        "training": False,
        "seed": seed,
        "motion_block_seconds": int(model_config.get("motion_block_seconds", 3)),
        "ppg_block_seconds": int(model_config.get("ppg_block_seconds", 15)),
        "motion_bucket_counts": model_config["motion_bucket_counts"],
        "ppg_bucket_counts": model_config["ppg_bucket_counts"],
    }
    validation_dataset = DTPDataset(
        validation_anchors,
        segments,
        normalization,
        **dataset_arguments,
    )
    test_dataset = DTPDataset(test_anchors, segments, normalization, **dataset_arguments)
    batch_size = int(
        inference_batch_size
        if inference_batch_size is not None
        else training_config["inference_batch_size"]
    )
    workers = int(
        inference_num_workers
        if inference_num_workers is not None
        else training_config.get("inference_num_workers", 2)
    )
    if batch_size <= 0 or workers < 0:
        raise ValueError("Inference batch size must be positive and workers non-negative")
    loader_arguments = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": False,
    }
    validation_loader = DataLoader(validation_dataset, **loader_arguments)
    test_loader = DataLoader(test_dataset, **loader_arguments)
    if not torch.cuda.is_available():
        raise RuntimeError("DTP quality re-inference requires CUDA")
    device = torch.device("cuda")
    model = DTPSQF(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    amp_name = str(training_config.get("amp_dtype", "bfloat16")).lower()
    amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
    validation_predictions, validation_auprc = _prediction_frame(
        model,
        validation_loader,
        device,
        amp_dtype,
        description=f"Re-inferring quality validation partition {inner_validation_partition}",
    )
    test_predictions, test_auprc = _prediction_frame(
        model,
        test_loader,
        device,
        amp_dtype,
        description=f"Re-inferring quality test partition {inner_validation_partition}",
    )
    validate_prediction_frame(
        validation_predictions, validation_anchors, "quality validation predictions"
    )
    validate_prediction_frame(test_predictions, test_anchors, "quality test predictions")
    output_dir.mkdir(parents=True, exist_ok=False)
    save_normalization(normalization, output_dir / "normalization.json")
    validation_path = output_dir / "best_validation_predictions.parquet"
    test_path = output_dir / "dtp_test_predictions.parquet"
    _save_prediction_frame(validation_predictions, validation_path)
    _save_prediction_frame(test_predictions, test_path)
    identity = _prediction_cache_identity(
        checkpoint_path,
        selection_signature,
        outer_fold,
        inner_validation_partition,
    )
    _write_prediction_cache_manifest(
        validation_path, validation_predictions, validation_auprc, identity
    )
    _write_prediction_cache_manifest(test_path, test_predictions, test_auprc, identity)
    record = {
        "inner_validation_partition": inner_validation_partition,
        "selection_signature": selection_signature,
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "source_normalization_sha256": _sha256(normalization_path),
        "best_checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_window_auprc": validation_auprc,
        "test_window_auprc": test_auprc,
        "validation_rows": len(validation_predictions),
        "test_rows": len(test_predictions),
    }
    return validation_predictions, test_predictions, record


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
    validation_subjects = set(validation_anchors["subject_key"].unique())
    validation_truth, validation_ignore = partition_evaluation_events(events, validation_subjects)
    train_subjects = set(train_anchors["subject_key"].unique())
    normalization = compute_normalization(segments, train_subjects)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_normalization(normalization, output_dir / "normalization.json")
    stable_feature_columns = configured_state_feature_columns(model_config)
    train_anchors, validation_anchors, test_anchors = _prepare_stable_features(
        train_anchors,
        validation_anchors,
        test_anchors,
        stable_feature_columns,
        output_dir,
    )

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
        stable_feature_columns=stable_feature_columns,
    )
    checkpoint_strategy = str(
        training_config.get("checkpoint_validation_strategy", "full_timeline")
    )
    checkpoint_metadata: dict[str, Any] = {
        "strategy": "full_timeline",
        "selected_rows": len(validation_anchors),
        "available_rows": len(validation_anchors),
    }
    use_checkpoint_subset = False
    if checkpoint_strategy == "event_stratified_complete_sessions":
        (
            checkpoint_validation_anchors,
            _checkpoint_truth,
            _checkpoint_ignore,
            checkpoint_metadata,
        ) = select_checkpoint_validation_sessions(
            validation_anchors,
            validation_truth,
            validation_ignore,
            int(training_config.get("checkpoint_validation_max_rows", len(validation_anchors))),
            seed,
            minimum_subjects=int(training_config.get("checkpoint_validation_min_subjects", 0)),
            minimum_events=int(training_config.get("checkpoint_validation_min_events", 0)),
        )
        use_checkpoint_subset = len(checkpoint_validation_anchors) < len(validation_anchors)
    elif checkpoint_selection_metric == "window_auprc" and validation_selector is None:
        checkpoint_validation_anchors = select_checkpoint_validation_anchors(
            validation_anchors,
            int(training_config.get("checkpoint_validation_max_rows", len(validation_anchors))),
            seed,
        )
        use_checkpoint_subset = len(checkpoint_validation_anchors) < len(validation_anchors)
        checkpoint_metadata = {
            "strategy": "deterministic_stratified_evaluable",
            "selected_rows": len(checkpoint_validation_anchors),
            "available_rows": len(validation_anchors),
        }
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
        stable_feature_columns=stable_feature_columns,
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
    inference_batch_size = int(training_config["inference_batch_size"])
    inference_chunk_rows = int(training_config.get("inference_resume_chunk_rows", 32768))
    if inference_batch_size <= 0 or inference_chunk_rows <= 0:
        raise ValueError("Inference batch and resume chunk sizes must be positive")
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
        batch_size=inference_batch_size,
        shuffle=False,
        **inference_loader_arguments,
    )
    resumable_loader_arguments = {
        "batch_size": inference_batch_size,
        **inference_loader_arguments,
    }
    inference_dataset_arguments = {
        "future_context_seconds": future_seconds,
        "training": False,
        "seed": seed,
        "motion_block_seconds": int(model_config.get("motion_block_seconds", 3)),
        "ppg_block_seconds": int(model_config.get("ppg_block_seconds", 15)),
        "motion_bucket_counts": model_config["motion_bucket_counts"],
        "ppg_bucket_counts": model_config["ppg_bucket_counts"],
        "stable_feature_columns": stable_feature_columns,
    }

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = build_state_model(model_config).to(device)
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
    early_stopping_patience_checks, early_stopping_min_delta = (
        resolve_early_stopping_config(training_config)
    )
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
    loss_arguments = {
        "positive_alpha": positive_alpha,
        "focal_gamma": float(loss_config["focal_gamma"]),
        "dice_weight": float(loss_config["dice_weight"]),
        "boundary_weight": float(loss_config["boundary_weight"]),
        "sqi_weight": float(loss_config["sqi_weight"]),
        "boundary_positive_weight": float(training_config["boundary_positive_weight"]),
    }
    if str(model_config.get("architecture", "dtp_sqf")) == "hierarchical_state":
        criterion = HierarchicalStateLoss(
            **loss_arguments,
            smooth_weight=float(loss_config.get("smooth_weight", 0.05)),
            smooth_tau=float(loss_config.get("smooth_tau", 0.25)),
        ).to(device)
    else:
        criterion = DTPLoss(**loss_arguments).to(device)
    amp_name = str(training_config["amp_dtype"]).lower()
    amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    start_epoch = 0
    best_f1 = -1.0
    best_boundary_mae = float("inf")
    best_validation_auprc = -math.inf
    patience = 0
    early_stopping_reference_score: float | None = None
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
        stored_reference = checkpoint.get("early_stopping_reference_score")
        if stored_reference is not None:
            early_stopping_reference_score = float(stored_reference)
        elif best_selection_rank is not None and math.isfinite(best_selection_rank[0]):
            early_stopping_reference_score = float(best_selection_rank[0])
        stored_signature = checkpoint.get("selection_signature")
        if stored_signature != selection_signature:
            raise RuntimeError(
                "Resume checkpoint signature does not match the active cross-fit configuration"
            )
        if int(checkpoint.get("inner_validation_partition", 0)) != inner_validation_partition:
            raise RuntimeError("Resume checkpoint belongs to a different inner partition")
        if checkpoint.get("checkpoint_selection_metric", "event_f1") != checkpoint_selection_metric:
            raise RuntimeError("Resume checkpoint uses a different checkpoint selection metric")
        tqdm.write(f"Loaded DTP-SQF resume checkpoint after epoch {start_epoch}/{max_epochs}")

    validation_interval = int(training_config["validation_every_epochs"])
    if validation_interval <= 0:
        raise ValueError("validation_every_epochs must be positive")
    training_complete = resume_training_is_complete(
        start_epoch, max_epochs, patience, early_stopping_patience_checks
    )
    if resume_path is not None and training_complete:
        tqdm.write("Training was already complete; resuming post-training inference only.")
    if checkpoint_strategy != "full_timeline":
        tqdm.write(
            "Checkpoint selection validation: "
            f"{len(checkpoint_validation_anchors)}/{len(validation_anchors)} anchors; "
            f"{checkpoint_metadata.get('selected_sessions', 'n/a')} sessions; "
            f"{checkpoint_metadata.get('selected_events', 'n/a')} events"
        )
    epoch_range = range(0) if training_complete else range(start_epoch, max_epochs)
    for epoch in epoch_range:
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
                    "early_stopping_patience": float(patience),
                    "early_stopping_reference_score": early_stopping_reference_score,
                }
            )
            _save_csv_frame(pd.DataFrame(history), output_dir / "history.csv")
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
                    "early_stopping_reference_score": early_stopping_reference_score,
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
        checkpoint_improved = best_selection_rank is None or current_rank > best_selection_rank
        current_primary_score = float(current_rank[0])
        early_stopping_reference_score, patience, meaningful_improvement = (
            update_early_stopping(
                early_stopping_reference_score,
                current_primary_score,
                patience,
                early_stopping_min_delta,
            )
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / max(len(train_loader), 1),
                "validation_auprc": validation_auprc,
                "validation_f1": metrics["f1"],
                "validation_boundary_mae_seconds": boundary_mae,
                "early_stopping_patience": float(patience),
                "early_stopping_reference_score": early_stopping_reference_score,
            }
        )
        _save_csv_frame(pd.DataFrame(history), output_dir / "history.csv")
        if checkpoint_improved:
            best_f1 = metrics["f1"]
            best_boundary_mae = boundary_mae
            best_validation_auprc = validation_auprc
            best_selection = current_selection
            best_selection_rank = current_rank
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
            "early_stopping_reference_score": early_stopping_reference_score,
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
        if checkpoint_improved:
            _save_torch_checkpoint(checkpoint_payload, checkpoint_path)
            _save_prediction_frame(
                validation_predictions,
                output_dir
                / (
                    "best_checkpoint_validation_predictions.parquet"
                    if use_checkpoint_subset
                    else "best_validation_predictions.parquet"
                ),
            )
        _save_torch_checkpoint(checkpoint_payload, last_checkpoint_path)
        tqdm.write(
            f"Epoch {epoch + 1}/{max_epochs}: "
            f"train_loss={history[-1]['train_loss']:.4f}, "
            f"validation_auprc={validation_auprc:.4f}, "
            f"validation_f1={metrics['f1']:.4f}, patience={patience}, "
            f"meaningful_improvement={meaningful_improvement}"
        )
        if patience >= early_stopping_patience_checks:
            tqdm.write("Early stopping threshold reached.")
            break

    if not checkpoint_path.exists():
        raise RuntimeError("Training ended without producing a best checkpoint")
    if validation_selector is not None:
        if best_selection is None:
            raise RuntimeError("Fusion training ended without a selected validation candidate")
        _write_json_atomic(
            best_selection,
            output_dir / "best_validation_selection.json",
        )
        if selection_gate is not None:
            selection_gate(best_selection)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("selection_signature") != selection_signature:
        raise RuntimeError("Best checkpoint signature does not match the active configuration")
    if int(checkpoint.get("inner_validation_partition", 0)) != inner_validation_partition:
        raise RuntimeError("Best checkpoint belongs to a different inner partition")
    if checkpoint.get("checkpoint_selection_metric", "event_f1") != checkpoint_selection_metric:
        raise RuntimeError("Best checkpoint uses a different checkpoint selection metric")
    model.load_state_dict(checkpoint["model"])
    cache_identity = _prediction_cache_identity(
        checkpoint_path,
        selection_signature,
        outer_fold,
        inner_validation_partition,
    )
    if use_checkpoint_subset:
        validation_path = output_dir / "best_validation_predictions.parquet"
        validation_cache = _load_prediction_cache(
            validation_path,
            validation_anchors,
            checkpoint_path,
            cache_identity,
            "full validation predictions",
            allow_legacy_manifest=True,
        )
        if validation_cache is None:
            _, full_validation_auprc = _resumable_prediction_frame(
                model,
                validation_anchors,
                segments,
                normalization,
                inference_dataset_arguments,
                resumable_loader_arguments,
                device,
                amp_dtype,
                validation_path,
                checkpoint_path,
                cache_identity,
                "full validation predictions",
                inference_chunk_rows,
            )
        else:
            _, full_validation_auprc = validation_cache
            tqdm.write("Reusing verified full validation predictions.")
    else:
        full_validation_auprc = best_validation_auprc
    test_path = output_dir / test_predictions_name
    test_cache = _load_prediction_cache(
        test_path,
        test_anchors,
        checkpoint_path,
        cache_identity,
        "test predictions",
    )
    if test_cache is None:
        _, test_auprc = _resumable_prediction_frame(
            model,
            test_anchors,
            segments,
            normalization,
            inference_dataset_arguments,
            resumable_loader_arguments,
            device,
            amp_dtype,
            test_path,
            checkpoint_path,
            cache_identity,
            "test predictions",
            inference_chunk_rows,
        )
    else:
        _, test_auprc = test_cache
        tqdm.write("Reusing verified test predictions.")
    metadata = {
        "outer_fold": outer_fold,
        "inner_validation_partition": inner_validation_partition,
        "checkpoint_selection_metric": checkpoint_selection_metric,
        "training_complete_before_resume": training_complete if resume_path is not None else False,
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
        "checkpoint_validation_strategy": checkpoint_metadata.get("strategy", checkpoint_strategy),
        "checkpoint_validation_seed": seed,
        "checkpoint_validation_metadata": checkpoint_metadata,
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
        "early_stopping_patience_checks": early_stopping_patience_checks,
        "early_stopping_min_delta": early_stopping_min_delta,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "validation_selection_signature": selection_signature,
    }
    _write_json_atomic(metadata, output_dir / "metadata.json")
    return checkpoint_path
