from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier


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
    digest = hashlib.sha256(f"{outer_fold}|{subject_key}".encode("utf-8")).digest()
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


def train_xgboost_fold(
    features: pd.DataFrame,
    subject_folds: dict[str, int],
    outer_fold: int,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[XGBClassifier, list[str], pd.DataFrame, pd.DataFrame]:
    train, validation, test = assign_train_validation_test(features, subject_folds, outer_fold)
    selected_train, excluded_far = sample_training_rows(
        train,
        float(config["near_event_minutes"]),
        float(config["far_negative_to_positive_ratio"]),
        int(config["random_seed"]),
    )
    columns = feature_columns(features)
    train_x = selected_train[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    validation_x = validation[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    trial_results: list[dict[str, Any]] = []
    best_model: XGBClassifier | None = None
    best_score = -np.inf
    best_parameters: dict[str, Any] = {}
    sampler = ParameterSampler(
        _parameter_space(),
        n_iter=int(config["trials"]),
        random_state=int(config["random_seed"]),
    )
    for trial, parameters in enumerate(sampler):
        model = _make_model(config, parameters)
        model.fit(
            train_x,
            selected_train["state_target"].to_numpy(),
            eval_set=[(validation_x, validation["state_target"].to_numpy())],
            verbose=False,
        )
        probability = model.predict_proba(validation_x)[:, 1]
        score = average_precision_score(validation["state_target"] > 0, probability)
        trial_results.append({"trial": trial, "validation_auprc": score, **parameters})
        if score > best_score:
            best_score = score
            best_model = model
            best_parameters = dict(parameters)
    if best_model is None:
        raise RuntimeError("No XGBoost trial completed")

    if len(excluded_far):
        excluded_x = excluded_far[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        excluded_probability = best_model.predict_proba(excluded_x)[:, 1]
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
        train_x = selected_train[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        best_model = _make_model(config, best_parameters)
        best_model.fit(
            train_x,
            selected_train["state_target"].to_numpy(),
            eval_set=[(validation_x, validation["state_target"].to_numpy())],
            verbose=False,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    best_model.save_model(output_dir / "model.json")
    metadata = {
        "outer_fold": outer_fold,
        "feature_columns": columns,
        "best_parameters": best_parameters,
        "validation_auprc": float(best_score),
        "train_rows": int(len(selected_train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(trial_results).to_csv(output_dir / "trials.csv", index=False)
    return best_model, columns, validation, test


def predict_xgboost(
    model: XGBClassifier,
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    matrix = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    probability = model.predict_proba(matrix)[:, 1]
    output = frame[["subject_key", "segment_id", "timestamp_ms"]].copy()
    output["state_probability"] = probability
    output["start_probability"] = 0.0
    output["end_probability"] = 0.0
    for _, indices in output.groupby(["subject_key", "segment_id"], sort=False).groups.items():
        ordered = output.loc[indices].sort_values("timestamp_ms")
        values = ordered["state_probability"].to_numpy()
        derivative = np.diff(values, prepend=values[0])
        output.loc[ordered.index, "start_probability"] = np.clip(derivative, 0, 1)
        output.loc[ordered.index, "end_probability"] = np.clip(-derivative, 0, 1)
    return output
