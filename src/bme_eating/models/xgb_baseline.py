from __future__ import annotations

import hashlib
import json
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
from xgboost import XGBClassifier


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
    "subject_key",
    "timestamp_ms",
    "state_target",
    "start_target",
    "end_target",
    "start_loss_mask",
    "end_loss_mask",
    "distance_to_event_seconds",
    "hand_relation",
    "fold",
}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _inner_partition(subject_key: str, outer_fold: int, partitions: int = 3) -> int:
    digest = hashlib.sha256(f"{outer_fold}|{subject_key}".encode()).digest()
    return int.from_bytes(digest[:4], "little") % partitions


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
    inner = outer_train["subject_key"].map(
        lambda value: _inner_partition(str(value), outer_fold)
    )
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
        n_jobs=-1,
        **parameters,
    )


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
    train, validation, test = assign_train_validation_test(features, subject_folds, outer_fold)
    if train.empty:
        raise ValueError(f"No training rows available for outer fold {outer_fold}")
    if validation.empty:
        raise ValueError(f"No validation rows available for outer fold {outer_fold}")
    if test.empty:
        raise ValueError(f"No test rows available for outer fold {outer_fold}")
    selected_train, excluded_far = sample_training_rows(
        train,
        float(config["near_event_minutes"]),
        float(config["far_negative_to_positive_ratio"]),
        int(config["random_seed"]),
    )
    train_y = (selected_train["state_target"].to_numpy() > 0).astype(np.int8)
    validation_y = (validation["state_target"].to_numpy() > 0).astype(np.int8)
    if np.unique(train_y).size < 2:
        raise ValueError("XGBoost training rows must contain both target classes")
    if np.unique(validation_y).size < 2:
        raise ValueError("XGBoost validation rows must contain both target classes")
    columns = feature_columns(features)
    if not columns:
        raise ValueError("No numeric feature columns available for XGBoost")
    train_x = selected_train[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    validation_x = validation[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_parameters = list(
        ParameterSampler(
            _parameter_space(),
            n_iter=int(config["trials"]),
            random_state=int(config["random_seed"]),
        )
    )
    checkpoint_path = output_dir / "xgboost_search.checkpoint.jsonl"
    signature = _xgb_search_signature(
        train_x,
        train_y,
        validation_x,
        validation_y,
        config,
        outer_fold,
        sampled_parameters,
    )
    trial_results = _load_xgb_checkpoint(checkpoint_path, signature) if resume_search else []
    completed_trials = {int(row["trial"]) for row in trial_results}
    if not completed_trials:
        _initialize_xgb_checkpoint(checkpoint_path, signature)
    best_model: XGBClassifier | None = None
    best_model_parameters: dict[str, Any] | None = None
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
        model = _make_model(config, parameters)
        model.fit(
            train_x,
            train_y,
            eval_set=[(validation_x, validation_y)],
            verbose=False,
        )
        probability = _predict_probability(model, validation_x)
        score = float(average_precision_score(validation_y, probability))
        row = {"trial": trial, "validation_auprc": score, **parameters}
        trial_results.append(row)
        completed_trials.add(trial)
        _append_xgb_checkpoint(checkpoint_path, row)
        _write_trials_csv(output_dir / "trials.csv", trial_results)
        if score > best_score:
            best_score = score
            best_model = model
            best_model_parameters = dict(parameters)
        progress.update(1)
        progress.set_postfix(auprc=f"{score:.4f}", best=f"{best_score:.4f}")
    progress.close()
    if not trial_results:
        raise RuntimeError("No XGBoost trial completed")

    parameter_names = list(_parameter_space())
    best_row = max(trial_results, key=lambda row: float(row["validation_auprc"]))
    best_score = float(best_row["validation_auprc"])
    best_parameters = {name: best_row[name] for name in parameter_names}
    if best_model is None or best_model_parameters != best_parameters:
        tqdm.write("Refitting the best completed XGBoost trial for resumed search...")
        best_model = _make_model(config, best_parameters)
        best_model.fit(
            train_x,
            train_y,
            eval_set=[(validation_x, validation_y)],
            verbose=False,
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
        best_model = _make_model(config, best_parameters)
        best_model.fit(
            train_x,
            train_y,
            eval_set=[(validation_x, validation_y)],
            verbose=False,
        )

    best_model.save_model(output_dir / "model.json")
    metadata = {
        "outer_fold": outer_fold,
        "feature_columns": columns,
        "best_parameters": best_parameters,
        "validation_auprc": float(best_score),
        "train_rows": len(selected_train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_trials_csv(output_dir / "trials.csv", trial_results)
    return best_model, columns, validation, test


def predict_xgboost(
    model: XGBClassifier,
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    matrix = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    probability = _predict_probability(model, matrix)
    output = frame[["subject_key", "segment_id", "timestamp_ms"]].copy()
    output["state_probability"] = probability
    output["start_probability"] = 0.0
    output["end_probability"] = 0.0
    for indices in output.groupby(["subject_key", "segment_id"], sort=False).groups.values():
        ordered = output.loc[indices].sort_values("timestamp_ms")
        values = ordered["state_probability"].to_numpy()
        derivative = np.diff(values, prepend=values[0])
        output.loc[ordered.index, "start_probability"] = np.clip(derivative, 0, 1)
        output.loc[ordered.index, "end_probability"] = np.clip(-derivative, 0, 1)
    return output
