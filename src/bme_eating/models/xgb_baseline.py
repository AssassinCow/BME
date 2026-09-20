from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor


def _configure_conda_cuda_runtime() -> None:
    """Make Conda's CUDA runtime discoverable by CuPy on Windows."""

    if os.name != "nt" or os.environ.get("CUDA_PATH"):
        return
    conda_root = Path(sys.prefix)
    if (conda_root / "include" / "cuda.h").exists():
        os.environ["CUDA_PATH"] = str(conda_root)
        os.environ["PATH"] = f"{conda_root / 'bin'};{os.environ.get('PATH', '')}"


_configure_conda_cuda_runtime()

try:  # Optional: only needed to keep CUDA prediction data on the GPU.
    import cupy as cp
except ImportError:  # pragma: no cover - depends on the user's GPU environment.
    cp = None  # type: ignore[assignment]


METADATA_COLUMNS = {
    "segment_id",
    "session_id",
    "subject_key",
    "timestamp_ms",
    "state_target",
    "state_loss_mask",
    "censor_mask",
    "start_target",
    "end_target",
    "start_loss_mask",
    "end_loss_mask",
    "distance_to_event_seconds",
    "hand_relation",
    "event_id",
    "motion_history_available_seconds",
    "ppg_history_available_seconds",
    "fold",
}


def _event_balanced_weights(
    frame: pd.DataFrame, different_hand_weight: float = 1.0
) -> np.ndarray:
    if not np.isfinite(different_hand_weight) or different_hand_weight <= 0:
        raise ValueError("different_hand_weight must be positive and finite")
    weights = np.ones(len(frame), dtype=np.float32)
    positive = frame["state_target"].to_numpy() > 0
    if "event_id" not in frame.columns or not positive.any():
        return weights
    event_ids = frame["event_id"].astype(str).to_numpy()
    positive_ids = event_ids[positive]
    counts = pd.Series(positive_ids).value_counts()
    raw = np.asarray([1.0 / max(int(counts.get(value, 1)), 1) for value in positive_ids])
    raw *= positive.sum() / max(raw.sum(), 1e-12)
    if "hand_relation" in frame.columns and different_hand_weight != 1.0:
        relation = frame.loc[positive, "hand_relation"].astype(str).to_numpy()
        raw[relation == "different"] *= different_hand_weight
        raw *= positive.sum() / max(raw.sum(), 1e-12)
    weights[positive] = raw.astype(np.float32)
    return weights


def feature_columns(frame: pd.DataFrame, ablation: str = "fused") -> list[str]:
    columns = [
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if ablation == "fused":
        return columns
    if ablation == "motion_only":
        return [column for column in columns if "ppg" not in column.lower()]
    if ablation == "ppg_only":
        return [column for column in columns if "ppg" in column.lower()]
    raise ValueError(f"Unknown feature ablation: {ablation}")


def _inner_partition(subject_key: str, outer_fold: int, partitions: int = 3) -> int:
    digest = hashlib.sha256(f"{outer_fold}|{subject_key}".encode()).digest()
    return int.from_bytes(digest[:4], "little") % partitions


def _balanced_inner_partitions(
    frame: pd.DataFrame,
    outer_fold: int,
    partitions: int = 3,
) -> dict[str, int]:
    if partitions <= 1:
        raise ValueError("Inner CV requires at least two partitions")
    positive = frame[frame["state_target"] > 0].copy()
    if "event_id" in positive.columns:
        positive = positive[positive["event_id"].astype(str) != ""].drop_duplicates(
            ["subject_key", "event_id"]
        )
    summary = pd.DataFrame({"subject_key": sorted(frame["subject_key"].astype(str).unique())})
    if len(positive):
        relation = positive["hand_relation"] if "hand_relation" in positive else pd.Series(
            "unknown", index=positive.index
        )
        counts = (
            positive.assign(
                same=(relation == "same").astype(int),
                different=(relation == "different").astype(int),
            )
            .groupby("subject_key", as_index=False)
            .agg(events=("state_target", "size"), same=("same", "sum"), different=("different", "sum"))
        )
        summary = summary.merge(counts, on="subject_key", how="left")
    for column in ("events", "same", "different"):
        if column not in summary:
            summary[column] = 0
        summary[column] = summary[column].fillna(0).astype(int)
    summary["tie"] = summary["subject_key"].map(
        lambda value: hashlib.sha256(f"{outer_fold}|{value}".encode()).hexdigest()
    )
    summary = summary.sort_values(
        ["events", "same", "different", "tie"], ascending=[False, False, False, True]
    )
    targets = np.asarray(
        [len(summary), summary["events"].sum(), summary["same"].sum(), summary["different"].sum()],
        dtype=np.float64,
    ) / partitions
    totals = np.zeros((partitions, 4), dtype=np.float64)
    assignments: dict[str, int] = {}
    for row in summary.itertuples(index=False):
        contribution = np.asarray([1.0, row.events, row.same, row.different])
        scores: list[float] = []
        for partition in range(partitions):
            proposed = totals.copy()
            proposed[partition] += contribution
            normalized = (proposed - targets[None, :]) / np.maximum(targets[None, :], 1.0)
            scores.append(float(np.square(normalized).sum()))
        chosen = int(np.argmin(scores))
        totals[chosen] += contribution
        assignments[str(row.subject_key)] = chosen
    return assignments


def assign_train_validation_test(
    frame: pd.DataFrame,
    subject_folds: dict[str, int],
    outer_fold: int,
    inner_validation_partition: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold = frame["subject_key"].map(subject_folds)
    if fold.isna().any():
        missing = sorted(frame.loc[fold.isna(), "subject_key"].unique())
        raise ValueError(f"Subjects missing from fold map: {missing}")
    test = frame[fold == outer_fold].copy()
    outer_train = frame[fold != outer_fold].copy()
    inner_map = _balanced_inner_partitions(outer_train, outer_fold)
    inner = outer_train["subject_key"].astype(str).map(inner_map)
    validation = outer_train[inner == inner_validation_partition].copy()
    train = outer_train[inner != inner_validation_partition].copy()
    return train, validation, test


def sample_training_rows(
    frame: pd.DataFrame,
    near_event_minutes: float,
    far_negative_to_positive_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive = frame[frame["state_target"] > 0].copy()
    negative = frame[frame["state_target"] <= 0].copy()
    near = negative[
        negative["distance_to_event_seconds"] <= near_event_minutes * 60.0
    ]
    far = negative[
        negative["distance_to_event_seconds"] > near_event_minutes * 60.0
    ]
    maximum_far = int(max(1, round(len(positive) * far_negative_to_positive_ratio)))
    selected_far = far.sample(n=min(maximum_far, len(far)), random_state=seed)
    selected_indices = set(positive.index) | set(near.index) | set(selected_far.index)
    selected = frame.loc[sorted(selected_indices)].copy()
    excluded_far = far.loc[~far.index.isin(selected_far.index)].copy()
    return selected, excluded_far


def _parameter_space() -> dict[str, list[Any]]:
    return {
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.02, 0.04, 0.06, 0.1],
        "min_child_weight": [1, 3, 8, 15],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0.0, 0.2, 0.5, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [1.0, 3.0, 10.0],
    }


def _xgb_search_signature(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    validation_x: pd.DataFrame,
    validation_y: np.ndarray,
    config: dict[str, Any],
    outer_fold: int,
    sampled_parameters: list[dict[str, Any]],
) -> str:
    digest = hashlib.sha256()
    payload = {
        "version": 1,
        "outer_fold": outer_fold,
        "config": config,
        "sampled_parameters": sampled_parameters,
    }
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    for matrix, target in ((train_x, train_y), (validation_x, validation_y)):
        row_hashes = pd.util.hash_pandas_object(matrix, index=True).to_numpy(dtype=np.uint64)
        digest.update(row_hashes.tobytes())
        digest.update(np.asarray(target, dtype=np.int8).tobytes())
    return digest.hexdigest()


def _load_xgb_checkpoint(path: Path, signature: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            header = json.loads(handle.readline())
            if header != {"version": 1, "signature": signature}:
                return []
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and "trial" in row and "validation_auprc" in row:
                    rows.append(row)
    except (OSError, json.JSONDecodeError, TypeError):
        return []
    return rows


def _initialize_xgb_checkpoint(path: Path, signature: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps({"version": 1, "signature": signature}) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _append_xgb_checkpoint(path: Path, row: dict[str, Any]) -> None:
    needs_separator = path.exists() and path.stat().st_size > 0
    if needs_separator:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            needs_separator = handle.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as handle:
        prefix = "\n" if needs_separator else ""
        handle.write(prefix + json.dumps(row, separators=(",", ":")) + "\n")
        handle.flush()


def _write_trials_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).sort_values("trial").to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _make_model(config: dict[str, Any], parameters: dict[str, Any]) -> XGBClassifier:
    return XGBClassifier(
        objective=config["objective"],
        eval_metric=config["eval_metric"],
        n_estimators=int(config["n_estimators"]),
        early_stopping_rounds=int(config["early_stopping_rounds"]),
        tree_method="hist",
        device=str(config["device"]),
        random_state=int(config["random_seed"]),
        n_jobs=int(config.get("n_jobs", -1)),
        **parameters,
    )


def _sample_boundary_rows(
    frame: pd.DataFrame,
    target_column: str,
    mask_column: str,
    random_negative_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = frame[frame.get(mask_column, pd.Series(0.0, index=frame.index)) > 0].copy()
    positive = eligible[eligible[target_column] > 0]
    negative = eligible[eligible[target_column] <= 0]
    if positive.empty:
        return eligible.iloc[:0].copy(), negative
    maximum = min(len(negative), math.ceil(len(positive) * random_negative_ratio))
    sampled_negative = negative.sample(n=maximum, random_state=seed) if maximum else negative.iloc[:0]
    selected = pd.concat([positive, sampled_negative]).sort_index()
    remaining = negative.loc[~negative.index.isin(sampled_negative.index)]
    return selected, remaining


def _boundary_weights(frame: pd.DataFrame, target_column: str, positive_weight: float) -> np.ndarray:
    weights = np.ones(len(frame), dtype=np.float32)
    weights[frame[target_column].to_numpy(dtype=np.float64) > 0] = float(positive_weight)
    return weights


def _make_boundary_model(
    config: dict[str, Any],
    parameters: dict[str, Any],
    n_estimators: int,
    *,
    early_stopping_rounds: int | None,
) -> XGBRegressor:
    kwargs: dict[str, Any] = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": int(n_estimators),
        "tree_method": "hist",
        "device": str(config["device"]),
        "random_state": int(config["random_seed"]),
        "n_jobs": int(config.get("n_jobs", -1)),
        **parameters,
    }
    if early_stopping_rounds:
        kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
    return XGBRegressor(**kwargs)


def _predict_regression(model: XGBRegressor, matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.to_numpy(dtype=np.float32, copy=False)
    prediction = model.predict(values)
    return np.clip(np.asarray(prediction, dtype=np.float64), 0.0, 1.0)


def _fit_boundary_head(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    state_config: dict[str, Any],
    parameters: dict[str, Any],
    boundary_config: dict[str, Any],
    target_column: str,
    mask_column: str,
    seed_offset: int,
    n_estimators: int,
) -> tuple[XGBRegressor | None, int]:
    selected, remaining = _sample_boundary_rows(
        train,
        target_column,
        mask_column,
        float(boundary_config.get("random_negative_to_positive_ratio", 20.0)),
        int(state_config["random_seed"]) + seed_offset,
    )
    validation = validation[
        validation.get(mask_column, pd.Series(0.0, index=validation.index)) > 0
    ]
    if selected.empty or not (selected[target_column] > 0).any():
        return None, 0
    train_x = selected[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    validation_x = (
        validation[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if len(validation)
        else None
    )
    early_stopping = (
        int(state_config.get("early_stopping_rounds", 0)) or None
        if validation_x is not None
        else None
    )
    model = _make_boundary_model(
        state_config,
        parameters,
        n_estimators,
        early_stopping_rounds=early_stopping,
    )
    fit_kwargs: dict[str, Any] = {}
    if validation_x is not None:
        fit_kwargs["eval_set"] = [
            (validation_x, validation[target_column].to_numpy(dtype=np.float32))
        ]
        fit_kwargs["verbose"] = False
    model.fit(
        train_x,
        selected[target_column].to_numpy(dtype=np.float32),
        sample_weight=_boundary_weights(
            selected, target_column, float(boundary_config.get("positive_weight", 20.0))
        ),
        **fit_kwargs,
    )
    if len(remaining):
        remaining_x = remaining[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        hard_count = min(
            len(remaining),
            math.ceil(
                (selected[target_column] > 0).sum()
                * float(boundary_config.get("hard_negative_to_positive_ratio", 10.0))
            ),
        )
        if hard_count:
            hard_indices = np.argsort(_predict_regression(model, remaining_x))[-hard_count:]
            selected = pd.concat([selected, remaining.iloc[hard_indices]], ignore_index=True)
            train_x = selected[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            model = _make_boundary_model(
                state_config,
                parameters,
                n_estimators,
                early_stopping_rounds=early_stopping,
            )
            model.fit(
                train_x,
                selected[target_column].to_numpy(dtype=np.float32),
                sample_weight=_boundary_weights(
                    selected,
                    target_column,
                    float(boundary_config.get("positive_weight", 20.0)),
                ),
                **fit_kwargs,
            )
    best_iteration = int(getattr(model, "best_iteration", n_estimators - 1)) + 1
    return model, max(1, best_iteration)


def _attach_boundary_models(
    state_model: XGBClassifier,
    start_model: XGBRegressor | None,
    end_model: XGBRegressor | None,
) -> None:
    state_model._bme_boundary_models = {  # type: ignore[attr-defined]
        "start": start_model,
        "end": end_model,
    }


def _model_uses_cuda(model: XGBClassifier) -> bool:
    """Return whether the fitted XGBoost model is configured for CUDA."""

    params = model.get_xgb_params()
    device = str(params.get("device", "cpu")).lower()
    return device.startswith("cuda") or params.get("tree_method") == "gpu_hist"


def _predict_probability(
    model: XGBClassifier,
    matrix: pd.DataFrame,
    *,
    batch_size: int = 131_072,
) -> np.ndarray:
    """Predict the positive-class probability without needless device copies.

    XGBoost's CUDA predictor expects CUDA-backed input.  Pandas/NumPy input is
    still valid, but XGBoost must fall back to a CPU DMatrix and emits a device
    mismatch warning.  CuPy is optional so the baseline remains runnable on a
    CPU-only machine; when it is unavailable, this function deliberately uses
    the regular CPU path.
    """

    values = matrix.to_numpy(dtype=np.float32, copy=False)
    uses_cuda = _model_uses_cuda(model)
    if uses_cuda and cp is not None:
        try:
            probabilities: list[np.ndarray] = []
            for start in range(0, len(values), batch_size):
                gpu_values = cp.asarray(values[start : start + batch_size])
                prediction = model.predict_proba(gpu_values)
                if isinstance(prediction, cp.ndarray):
                    prediction = cp.asnumpy(prediction)
                probabilities.append(np.asarray(prediction)[:, 1])
            if not probabilities:
                return np.empty(0, dtype=np.float32)
            return np.concatenate(probabilities)
        except (RuntimeError, TypeError, ValueError) as error:
            warnings.warn(
                f"CUDA prediction input failed ({error}); falling back to CPU input.",
                RuntimeWarning,
                stacklevel=2,
            )

    if uses_cuda and cp is None:
        warnings.warn(
            "XGBoost is configured for CUDA but CuPy is not installed; "
            "prediction is falling back to CPU input. Install cupy-cuda12x "
            "to avoid device-mismatch transfers.",
            RuntimeWarning,
            stacklevel=2,
        )
    prediction = model.predict_proba(values)
    return np.asarray(prediction)[:, 1]


def train_xgboost_fold(
    features: pd.DataFrame,
    subject_folds: dict[str, int],
    outer_fold: int,
    config: dict[str, Any],
    output_dir: Path,
    resume_search: bool = True,
) -> tuple[XGBClassifier, list[str], pd.DataFrame, pd.DataFrame]:
    fold = features["subject_key"].map(subject_folds)
    if fold.isna().any():
        missing = sorted(features.loc[fold.isna(), "subject_key"].unique())
        raise ValueError(f"Subjects missing from fold map: {missing}")
    outer_train = features[fold != outer_fold].copy()
    test = features[fold == outer_fold].copy()
    if outer_train.empty:
        raise ValueError(f"No outer-training rows available for fold {outer_fold}")
    if test.empty:
        raise ValueError(f"No test rows available for outer fold {outer_fold}")
    modeling_rows = outer_train[
        outer_train.get("state_loss_mask", pd.Series(1.0, index=outer_train.index)) > 0
    ].copy()
    columns = feature_columns(features, str(config.get("feature_ablation", "fused")))
    if not columns:
        raise ValueError("No numeric feature columns available for XGBoost")
    inner_map = _balanced_inner_partitions(modeling_rows, outer_fold, partitions=3)
    inner_partition = modeling_rows["subject_key"].astype(str).map(inner_map)
    if inner_partition.nunique() != 3:
        raise ValueError("Balanced inner CV did not produce three non-empty subject groups")
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_parameters = list(
        ParameterSampler(
            _parameter_space(),
            n_iter=int(config["trials"]),
            random_state=int(config["random_seed"]),
        )
    )
    checkpoint_path = output_dir / "xgboost_search.checkpoint.jsonl"
    signature_matrix = modeling_rows[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    signature_target = (modeling_rows["state_target"].to_numpy() > 0).astype(np.int8)
    signature = _xgb_search_signature(
        signature_matrix,
        signature_target,
        signature_matrix.iloc[:0],
        signature_target[:0],
        config,
        outer_fold,
        sampled_parameters,
    )
    trial_results = _load_xgb_checkpoint(checkpoint_path, signature) if resume_search else []
    completed_trials = {int(row["trial"]) for row in trial_results}
    if not completed_trials:
        _initialize_xgb_checkpoint(checkpoint_path, signature)
    best_score = max(
        (float(row["validation_auprc"]) for row in trial_results), default=-np.inf
    )
    progress = tqdm(
        total=len(sampled_parameters),
        initial=len(completed_trials),
        desc="Tuning XGBoost",
        unit="trial",
    )
    for trial, parameters in enumerate(sampled_parameters):
        if trial in completed_trials:
            continue
        fold_scores: list[float] = []
        for inner_fold in range(3):
            inner_train = modeling_rows[inner_partition != inner_fold]
            inner_validation = modeling_rows[inner_partition == inner_fold]
            selected, _ = sample_training_rows(
                inner_train,
                float(config["near_event_minutes"]),
                float(config["far_negative_to_positive_ratio"]),
                int(config["random_seed"]) + inner_fold,
            )
            train_y = (selected["state_target"].to_numpy() > 0).astype(np.int8)
            validation_y = (
                inner_validation["state_target"].to_numpy() > 0
            ).astype(np.int8)
            if np.unique(train_y).size < 2 or np.unique(validation_y).size < 2:
                raise ValueError(f"Inner fold {inner_fold} does not contain both classes")
            train_x = selected[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            validation_x = inner_validation[columns].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
            model = _make_model(config, parameters)
            model.fit(
                train_x,
                train_y,
                sample_weight=_event_balanced_weights(selected),
                eval_set=[(validation_x, validation_y)],
                verbose=False,
            )
            probability = _predict_probability(model, validation_x)
            fold_scores.append(float(average_precision_score(validation_y, probability)))
        score = float(np.mean(fold_scores))
        row = {
            "trial": trial,
            "validation_auprc": score,
            "inner_fold_auprc": fold_scores,
            **parameters,
        }
        trial_results.append(row)
        completed_trials.add(trial)
        _append_xgb_checkpoint(checkpoint_path, row)
        _write_trials_csv(output_dir / "trials.csv", trial_results)
        best_score = max(best_score, score)
        progress.update(1)
        progress.set_postfix(auprc=f"{score:.4f}", best=f"{best_score:.4f}")
    progress.close()
    if not trial_results:
        raise RuntimeError("No XGBoost trial completed")

    parameter_names = list(_parameter_space())
    best_row = max(trial_results, key=lambda row: float(row["validation_auprc"]))
    best_score = float(best_row["validation_auprc"])
    best_parameters = {name: best_row[name] for name in parameter_names}
    relation_candidates = [
        float(value) for value in config.get("different_hand_weight_candidates", [1.0])
    ]
    if not relation_candidates:
        raise ValueError("different_hand_weight_candidates must not be empty")
    candidate_oof: dict[float, list[pd.DataFrame]] = {}
    candidate_iterations: dict[float, list[int]] = {}
    relation_scores: dict[float, float] = {}
    for relation_weight in relation_candidates:
        weight_predictions: list[pd.DataFrame] = []
        iterations: list[int] = []
        fold_scores: list[float] = []
        for inner_fold in range(3):
            inner_train = modeling_rows[inner_partition != inner_fold]
            inner_validation_model = modeling_rows[inner_partition == inner_fold]
            validation_subjects = set(inner_validation_model["subject_key"].astype(str))
            inner_validation_full = outer_train[
                outer_train["subject_key"].astype(str).isin(validation_subjects)
            ]
            selected, _ = sample_training_rows(
                inner_train,
                float(config["near_event_minutes"]),
                float(config["far_negative_to_positive_ratio"]),
                int(config["random_seed"]) + inner_fold,
            )
            train_x = selected[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            validation_x = inner_validation_model[columns].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
            model = _make_model(config, best_parameters)
            model.fit(
                train_x,
                (selected["state_target"].to_numpy() > 0).astype(np.int8),
                sample_weight=_event_balanced_weights(selected, relation_weight),
                eval_set=[
                    (
                        validation_x,
                        (inner_validation_model["state_target"].to_numpy() > 0).astype(np.int8),
                    )
                ],
                verbose=False,
            )
            iterations.append(int(getattr(model, "best_iteration", 0)) + 1)
            validation_probability = _predict_probability(model, validation_x)
            fold_scores.append(
                float(
                    average_precision_score(
                        (inner_validation_model["state_target"].to_numpy() > 0).astype(np.int8),
                        validation_probability,
                    )
                )
            )
            prediction = predict_xgboost(model, inner_validation_full, columns)
            prediction["calibration_fold"] = inner_fold
            weight_predictions.append(prediction)
        candidate_oof[relation_weight] = weight_predictions
        candidate_iterations[relation_weight] = iterations
        relation_scores[relation_weight] = float(np.mean(fold_scores))
    selected_relation_weight = max(
        relation_candidates, key=lambda value: (relation_scores[value], -value)
    )
    oof_predictions = candidate_oof[selected_relation_weight]
    best_iterations = candidate_iterations[selected_relation_weight]

    boundary_config = dict(config.get("boundary_heads", {}))
    boundary_enabled = bool(boundary_config.get("enabled", False))
    boundary_iterations: dict[str, list[int]] = {"start": [], "end": []}
    if boundary_enabled:
        oof_predictions = []
        for inner_fold in range(3):
            inner_train = modeling_rows[inner_partition != inner_fold]
            inner_validation_model = modeling_rows[inner_partition == inner_fold]
            validation_subjects = set(inner_validation_model["subject_key"].astype(str))
            training_subjects = set(inner_train["subject_key"].astype(str))
            boundary_inner_train = outer_train[
                outer_train["subject_key"].astype(str).isin(training_subjects)
            ]
            boundary_inner_validation = outer_train[
                outer_train["subject_key"].astype(str).isin(validation_subjects)
            ]
            inner_validation_full = outer_train[
                outer_train["subject_key"].astype(str).isin(validation_subjects)
            ]
            selected, _ = sample_training_rows(
                inner_train,
                float(config["near_event_minutes"]),
                float(config["far_negative_to_positive_ratio"]),
                int(config["random_seed"]) + inner_fold,
            )
            state_model = _make_model(config, best_parameters)
            state_model.fit(
                selected[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0),
                (selected["state_target"].to_numpy() > 0).astype(np.int8),
                sample_weight=_event_balanced_weights(selected, selected_relation_weight),
                eval_set=[
                    (
                        inner_validation_model[columns]
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(0.0),
                        (inner_validation_model["state_target"].to_numpy() > 0).astype(np.int8),
                    )
                ],
                verbose=False,
            )
            start_model, start_iterations = _fit_boundary_head(
                boundary_inner_train,
                boundary_inner_validation,
                columns,
                config,
                best_parameters,
                boundary_config,
                "start_target",
                "start_loss_mask",
                inner_fold * 2,
                int(config["n_estimators"]),
            )
            end_model, end_iterations = _fit_boundary_head(
                boundary_inner_train,
                boundary_inner_validation,
                columns,
                config,
                best_parameters,
                boundary_config,
                "end_target",
                "end_loss_mask",
                inner_fold * 2 + 1,
                int(config["n_estimators"]),
            )
            if start_iterations:
                boundary_iterations["start"].append(start_iterations)
            if end_iterations:
                boundary_iterations["end"].append(end_iterations)
            _attach_boundary_models(state_model, start_model, end_model)
            prediction = predict_xgboost(state_model, inner_validation_full, columns)
            prediction["calibration_fold"] = inner_fold
            oof_predictions.append(prediction)

    selected_train, excluded_far = sample_training_rows(
        modeling_rows,
        float(config["near_event_minutes"]),
        float(config["far_negative_to_positive_ratio"]),
        int(config["random_seed"]),
    )
    final_estimators = max(1, round(float(np.median(best_iterations))))
    final_config = dict(config)
    final_config["n_estimators"] = final_estimators
    final_config["early_stopping_rounds"] = 0
    best_model = XGBClassifier(
        objective=final_config["objective"],
        eval_metric=final_config["eval_metric"],
        n_estimators=final_estimators,
        tree_method="hist",
        device=str(final_config["device"]),
        random_state=int(final_config["random_seed"]),
        n_jobs=int(final_config.get("n_jobs", -1)),
        **best_parameters,
    )
    train_x = selected_train[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_y = (selected_train["state_target"].to_numpy() > 0).astype(np.int8)
    best_model.fit(
        train_x,
        train_y,
        sample_weight=_event_balanced_weights(selected_train, selected_relation_weight),
    )
    if len(excluded_far):
        excluded_x = excluded_far[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        excluded_probability = _predict_probability(best_model, excluded_x)
        hard_count = min(
            len(excluded_far),
            int(
                max(1, (selected_train["state_target"] > 0).sum())
                * float(config["hard_negative_to_positive_ratio"])
            ),
        )
        hard_indices = np.argsort(excluded_probability)[-hard_count:]
        selected_train = pd.concat(
            [selected_train, excluded_far.iloc[hard_indices]], ignore_index=True
        )
        train_y = (selected_train["state_target"].to_numpy() > 0).astype(np.int8)
        train_x = selected_train[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        best_model = XGBClassifier(
            objective=final_config["objective"],
            eval_metric=final_config["eval_metric"],
            n_estimators=final_estimators,
            tree_method="hist",
            device=str(final_config["device"]),
            random_state=int(final_config["random_seed"]),
            n_jobs=int(final_config.get("n_jobs", -1)),
            **best_parameters,
        )
        best_model.fit(
            train_x,
            train_y,
            sample_weight=_event_balanced_weights(selected_train, selected_relation_weight),
        )

    if boundary_enabled:
        start_estimators = max(
            1,
            round(float(np.median(boundary_iterations["start"])))
            if boundary_iterations["start"]
            else final_estimators,
        )
        end_estimators = max(
            1,
            round(float(np.median(boundary_iterations["end"])))
            if boundary_iterations["end"]
            else final_estimators,
        )
        start_model, _ = _fit_boundary_head(
            outer_train,
            outer_train,
            columns,
            {**config, "early_stopping_rounds": 0},
            best_parameters,
            boundary_config,
            "start_target",
            "start_loss_mask",
            100,
            start_estimators,
        )
        end_model, _ = _fit_boundary_head(
            outer_train,
            outer_train,
            columns,
            {**config, "early_stopping_rounds": 0},
            best_parameters,
            boundary_config,
            "end_target",
            "end_loss_mask",
            101,
            end_estimators,
        )
        _attach_boundary_models(best_model, start_model, end_model)
        if start_model is not None:
            start_model.save_model(output_dir / "start_boundary_model.json")
        if end_model is not None:
            end_model.save_model(output_dir / "end_boundary_model.json")

    best_model.save_model(output_dir / "model.json")
    metadata = {
        "outer_fold": outer_fold,
        "feature_columns": columns,
        "best_parameters": best_parameters,
        "validation_auprc": float(best_score),
        "inner_partitions": inner_map,
        "inner_best_iterations": best_iterations,
        "final_estimators": final_estimators,
        "different_hand_weight": selected_relation_weight,
        "different_hand_weight_oof_auprc": relation_scores,
        "boundary_heads_enabled": boundary_enabled,
        "boundary_inner_best_iterations": boundary_iterations,
        "train_rows": len(selected_train),
        "validation_rows": len(outer_train),
        "test_rows": len(test),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_trials_csv(output_dir / "trials.csv", trial_results)
    validation_predictions = pd.concat(oof_predictions, ignore_index=True).sort_values(
        ["subject_key", "session_id", "timestamp_ms"]
    ).reset_index(drop=True)
    return best_model, columns, validation_predictions, test


def predict_xgboost(
    model: XGBClassifier,
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    matrix = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    probability = _predict_probability(model, matrix)
    output = frame[
        ["subject_key", "segment_id", "session_id", "timestamp_ms"]
    ].copy()
    output["state_probability"] = probability
    output["start_probability"] = 0.0
    output["end_probability"] = 0.0
    for indices in output.groupby(["subject_key", "session_id"], sort=False).groups.values():
        ordered = output.loc[indices].sort_values("timestamp_ms")
        values = ordered["state_probability"].to_numpy()
        derivative = np.diff(values, prepend=values[0])
        output.loc[ordered.index, "start_probability"] = np.clip(derivative, 0, 1)
        output.loc[ordered.index, "end_probability"] = np.clip(-derivative, 0, 1)
    boundary_models = getattr(model, "_bme_boundary_models", {})
    start_model = boundary_models.get("start") if isinstance(boundary_models, dict) else None
    end_model = boundary_models.get("end") if isinstance(boundary_models, dict) else None
    if start_model is not None:
        output["start_probability"] = _predict_regression(start_model, matrix)
    if end_model is not None:
        output["end_probability"] = _predict_regression(end_model, matrix)
    return output


def load_xgboost_fold(
    output_dir: Path,
) -> tuple[XGBClassifier, list[str]]:
    """Load a persisted state model and its optional learned boundary heads."""

    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing XGBoost metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    columns = [str(value) for value in metadata.get("feature_columns", [])]
    if not columns:
        raise ValueError("Persisted XGBoost metadata has no feature columns")
    state_model = XGBClassifier()
    state_model.load_model(output_dir / "model.json")
    boundary_models: dict[str, XGBRegressor | None] = {"start": None, "end": None}
    for name in ("start", "end"):
        path = output_dir / f"{name}_boundary_model.json"
        if path.exists():
            model = XGBRegressor()
            model.load_model(path)
            boundary_models[name] = model
    _attach_boundary_models(
        state_model, boundary_models["start"], boundary_models["end"]
    )
    return state_model, columns
