from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    output = np.empty_like(values)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def binary_state_targets(targets: np.ndarray) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float64).reshape(-1)
    if not len(values):
        raise ValueError("State targets must be non-empty")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("State targets must be finite values in [0, 1]")
    return (values > 0.0).astype(np.int64)


@dataclass(frozen=True)
class TemperatureCalibrationV4:
    temperature: float

    @classmethod
    def fit(cls, logits: np.ndarray, targets: np.ndarray) -> TemperatureCalibrationV4:
        logits = np.asarray(logits, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if logits.shape != targets.shape or not logits.size:
            raise ValueError("Temperature calibration requires aligned non-empty arrays")

        def objective(log_temperature: float) -> float:
            probability = sigmoid(logits / np.exp(log_temperature))
            return float(
                -np.mean(
                    targets * np.log(np.clip(probability, 1e-8, 1.0))
                    + (1.0 - targets) * np.log(np.clip(1.0 - probability, 1e-8, 1.0))
                )
            )

        result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
        if not result.success:
            raise RuntimeError("Temperature calibration failed")
        return cls(float(np.exp(result.x)))

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return sigmoid(np.asarray(logits, dtype=np.float64) / self.temperature)

    def to_json(self) -> dict[str, float]:
        return {"temperature": self.temperature}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> TemperatureCalibrationV4:
        return cls(float(payload["temperature"]))


@dataclass(frozen=True)
class PlattCalibration:
    coefficient: float
    intercept: float

    @classmethod
    def fit(
        cls, logits: np.ndarray, targets: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> PlattCalibration:
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        targets = binary_state_targets(targets)
        if len(logits) != len(targets) or not len(logits):
            raise ValueError("Platt calibration requires aligned non-empty arrays")
        if len(np.unique(targets)) != 2:
            raise ValueError("Platt calibration requires both target classes")
        model = LogisticRegression(C=1e6, solver="lbfgs", random_state=2026)
        model.fit(logits[:, None], targets, sample_weight=sample_weight)
        return cls(float(model.coef_[0, 0]), float(model.intercept_[0]))

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return sigmoid(self.coefficient * np.asarray(logits, dtype=np.float64) + self.intercept)

    def to_json(self) -> dict[str, float]:
        return {"coefficient": self.coefficient, "intercept": self.intercept}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> PlattCalibration:
        return cls(float(payload["coefficient"]), float(payload["intercept"]))


def expected_calibration_error(
    targets: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(targets) != len(probabilities) or not len(targets):
        raise ValueError("ECE requires aligned non-empty arrays")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    indices = np.minimum(np.searchsorted(edges, probabilities, side="right") - 1, bins - 1)
    indices = np.maximum(indices, 0)
    error = 0.0
    for index in range(int(bins)):
        selected = indices == index
        if selected.any():
            error += selected.mean() * abs(
                probabilities[selected].mean() - targets[selected].mean()
            )
    return float(error)


def state_calibration_metrics(
    targets: np.ndarray,
    raw_logits: np.ndarray,
    calibrated: np.ndarray,
    *,
    low_threshold: float,
    bins: int = 15,
) -> dict[str, float]:
    targets = binary_state_targets(targets).astype(np.float64)
    raw = sigmoid(raw_logits)
    calibrated = np.asarray(calibrated, dtype=np.float64)
    prevalence = float(targets.mean())
    mean_probability = float(calibrated.mean())
    background = targets <= 0
    return {
        "uncalibrated_brier": float(brier_score_loss(targets, raw)),
        "brier": float(brier_score_loss(targets, calibrated)),
        "ece": expected_calibration_error(targets, calibrated, bins),
        "prevalence": prevalence,
        "mean_probability": mean_probability,
        "mean_probability_to_prevalence": (
            mean_probability / prevalence if prevalence > 0 else float("inf")
        ),
        "background_above_low_threshold": float(
            (calibrated[background] >= low_threshold).mean() if background.any() else 0.0
        ),
    }


def subject_crossfit_platt(
    frame: pd.DataFrame,
    *,
    partition_column: str = "stacking_partition",
    logit_column: str = "state_logit",
    target_column: str = "state_target",
    fit_mask_column: str | None = None,
) -> tuple[pd.DataFrame, PlattCalibration]:
    required = {"subject_key", partition_column, logit_column, target_column}
    if fit_mask_column is not None:
        required.add(fit_mask_column)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Calibration frame is missing columns: {sorted(missing)}")
    output = frame.copy()
    calibrated = np.full(len(output), np.nan, dtype=np.float64)
    eligible = (
        output[fit_mask_column].fillna(0.0).to_numpy(dtype=np.float64) > 0
        if fit_mask_column is not None
        else np.ones(len(output), dtype=bool)
    )
    subject_partitions = output.groupby("subject_key")[partition_column].nunique()
    if (subject_partitions != 1).any():
        raise RuntimeError("A subject appears in multiple calibration partitions")
    for partition in sorted(output[partition_column].unique()):
        holdout = output[partition_column] == partition
        fit = ~holdout
        fit_subjects = set(output.loc[fit, "subject_key"].astype(str))
        holdout_subjects = set(output.loc[holdout, "subject_key"].astype(str))
        if fit_subjects & holdout_subjects:
            raise RuntimeError("Subject leakage in Platt crossfit")
        fit_eligible = fit.to_numpy() & eligible
        calibrator = PlattCalibration.fit(
            output.loc[fit_eligible, logit_column].to_numpy(),
            output.loc[fit_eligible, target_column].to_numpy(),
        )
        calibrated[holdout.to_numpy()] = calibrator.transform(
            output.loc[holdout, logit_column].to_numpy()
        )
    if not np.isfinite(calibrated).all():
        raise RuntimeError("Crossfit calibration left unscored rows")
    output["state_probability"] = calibrated
    final = PlattCalibration.fit(
        output.loc[eligible, logit_column].to_numpy(),
        output.loc[eligible, target_column].to_numpy(),
    )
    return output, final


@dataclass(frozen=True)
class LogisticScoreCombiner:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

    @classmethod
    def fit(
        cls, features: np.ndarray, targets: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> LogisticScoreCombiner:
        values = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.int64)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        standardized = (values - mean) / scale
        model = LogisticRegression(C=1.0, solver="lbfgs", random_state=2026)
        model.fit(standardized, targets, sample_weight=sample_weight)
        return cls(
            tuple(float(value) for value in mean),
            tuple(float(value) for value in scale),
            tuple(float(value) for value in model.coef_[0]),
            float(model.intercept_[0]),
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        standardized = (values - np.asarray(self.mean)) / np.asarray(self.scale)
        return sigmoid(standardized @ np.asarray(self.coefficients) + self.intercept)

    def to_json(self) -> dict[str, Any]:
        return {
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> LogisticScoreCombiner:
        return cls(
            tuple(float(value) for value in payload["mean"]),
            tuple(float(value) for value in payload["scale"]),
            tuple(float(value) for value in payload["coefficients"]),
            float(payload["intercept"]),
        )


@dataclass(frozen=True)
class ProposalCalibrationV4:
    event: TemperatureCalibrationV4
    iou: TemperatureCalibrationV4
    combiner: LogisticScoreCombiner

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> ProposalCalibrationV4:
        required = {"event_logit", "iou_logit", "state_score", "is_positive", "max_iou"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Proposal calibration frame is missing columns: {sorted(missing)}")
        event = TemperatureCalibrationV4.fit(
            frame["event_logit"].to_numpy(), frame["is_positive"].to_numpy()
        )
        iou = TemperatureCalibrationV4.fit(
            frame["iou_logit"].to_numpy(), frame["max_iou"].to_numpy()
        )
        event_probability = event.transform_logits(frame["event_logit"].to_numpy())
        iou_probability = iou.transform_logits(frame["iou_logit"].to_numpy())
        features = np.column_stack(
            (logit(event_probability), logit(iou_probability), logit(frame["state_score"]))
        )
        weight = frame["sample_weight"].to_numpy() if "sample_weight" in frame else None
        combiner = LogisticScoreCombiner.fit(
            features, frame["is_positive"].to_numpy(), sample_weight=weight
        )
        return cls(event, iou, combiner)

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        output["calibrated_event_probability"] = self.event.transform_logits(
            output["event_logit"].to_numpy()
        )
        output["calibrated_iou"] = self.iou.transform_logits(output["iou_logit"].to_numpy())
        features = np.column_stack(
            (
                logit(output["calibrated_event_probability"]),
                logit(output["calibrated_iou"]),
                logit(output["state_score"]),
            )
        )
        output["final_score"] = self.combiner.predict(features)
        return output

    def to_json(self) -> dict[str, Any]:
        return {
            "event": self.event.to_json(),
            "iou": self.iou.to_json(),
            "combiner": self.combiner.to_json(),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ProposalCalibrationV4:
        return cls(
            TemperatureCalibrationV4.from_json(payload["event"]),
            TemperatureCalibrationV4.from_json(payload["iou"]),
            LogisticScoreCombiner.from_json(payload["combiner"]),
        )
