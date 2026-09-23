from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from bme_eating.proposals import interval_iou


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    output = np.empty_like(values)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float

    @classmethod
    def fit(cls, logits: np.ndarray, targets: np.ndarray) -> TemperatureCalibration:
        logits = np.asarray(logits, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if logits.shape != targets.shape or logits.size == 0:
            raise ValueError("Temperature calibration requires aligned non-empty arrays")

        def objective(log_temperature: float) -> float:
            probability = _sigmoid(logits / np.exp(log_temperature))
            return float(
                -np.mean(
                    targets * np.log(np.clip(probability, 1e-8, 1.0))
                    + (1.0 - targets) * np.log(np.clip(1.0 - probability, 1e-8, 1.0))
                )
            )

        result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
        if not result.success:
            raise RuntimeError("Temperature calibration failed to converge")
        return cls(float(np.exp(result.x)))

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return _sigmoid(np.asarray(logits, dtype=np.float64) / self.temperature)

    def to_json(self) -> dict[str, float]:
        return {"temperature": self.temperature}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> TemperatureCalibration:
        return cls(float(payload["temperature"]))


@dataclass(frozen=True)
class ScoreCombiner:
    coefficients: tuple[float, float, float]
    intercept: float

    @classmethod
    def fit(
        cls,
        event_probability: np.ndarray,
        iou_probability: np.ndarray,
        state_probability: np.ndarray,
        targets: np.ndarray,
    ) -> ScoreCombiner:
        features = np.column_stack(
            (
                _logit(event_probability),
                _logit(iou_probability),
                _logit(state_probability),
            )
        )
        targets = np.asarray(targets, dtype=np.int64)
        if len(np.unique(targets)) != 2:
            raise ValueError("Score combiner requires positive and negative proposals")
        model = LogisticRegression(C=1.0, solver="lbfgs", random_state=2026)
        model.fit(features, targets)
        return cls(tuple(float(value) for value in model.coef_[0]), float(model.intercept_[0]))

    def predict(
        self,
        event_probability: np.ndarray,
        iou_probability: np.ndarray,
        state_probability: np.ndarray,
    ) -> np.ndarray:
        features = np.column_stack(
            (
                _logit(event_probability),
                _logit(iou_probability),
                _logit(state_probability),
            )
        )
        return _sigmoid(features @ np.asarray(self.coefficients) + self.intercept)

    def to_json(self) -> dict[str, Any]:
        return {"coefficients": list(self.coefficients), "intercept": self.intercept}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ScoreCombiner:
        return cls(
            tuple(float(value) for value in payload["coefficients"]),
            float(payload["intercept"]),
        )


@dataclass(frozen=True)
class CalibrationBundle:
    event: TemperatureCalibration
    iou: TemperatureCalibration
    state: TemperatureCalibration
    combiner: ScoreCombiner

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> CalibrationBundle:
        required = {"event_logit", "iou_logit", "state_score", "is_positive", "max_iou"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Calibration frame is missing columns: {sorted(missing)}")
        target = frame["is_positive"].to_numpy(dtype=np.float64)
        event = TemperatureCalibration.fit(frame["event_logit"].to_numpy(), target)
        iou = TemperatureCalibration.fit(frame["iou_logit"].to_numpy(), frame["max_iou"].to_numpy())
        state_logit = _logit(frame["state_score"].to_numpy())
        state = TemperatureCalibration.fit(state_logit, target)
        event_probability = event.transform_logits(frame["event_logit"].to_numpy())
        iou_probability = iou.transform_logits(frame["iou_logit"].to_numpy())
        state_probability = state.transform_logits(state_logit)
        combiner = ScoreCombiner.fit(
            event_probability,
            iou_probability,
            state_probability,
            target,
        )
        return cls(event, iou, state, combiner)

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        output["calibrated_event_probability"] = self.event.transform_logits(
            output["event_logit"].to_numpy()
        )
        output["calibrated_iou"] = self.iou.transform_logits(
            output["iou_logit"].to_numpy()
        )
        output["calibrated_state_probability"] = self.state.transform_logits(
            _logit(output["state_score"].to_numpy())
        )
        output["final_score"] = self.combiner.predict(
            output["calibrated_event_probability"].to_numpy(),
            output["calibrated_iou"].to_numpy(),
            output["calibrated_state_probability"].to_numpy(),
        )
        return output

    def to_json(self) -> dict[str, Any]:
        return {
            "event": self.event.to_json(),
            "iou": self.iou.to_json(),
            "state": self.state.to_json(),
            "combiner": self.combiner.to_json(),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> CalibrationBundle:
        return cls(
            TemperatureCalibration.from_json(payload["event"]),
            TemperatureCalibration.from_json(payload["iou"]),
            TemperatureCalibration.from_json(payload["state"]),
            ScoreCombiner.from_json(payload["combiner"]),
        )


def crossfit_calibrate_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, CalibrationBundle]:
    if "calibration_fold" not in frame:
        raise ValueError("Cross-fit calibration requires calibration_fold")
    folds = sorted(int(value) for value in frame["calibration_fold"].unique())
    if folds != list(range(len(folds))) or len(folds) < 2:
        raise ValueError("Calibration folds must be contiguous and contain at least two folds")
    parts: list[pd.DataFrame] = []
    for fold in folds:
        train = frame[frame["calibration_fold"] != fold]
        validation = frame[frame["calibration_fold"] == fold]
        bundle = CalibrationBundle.fit(train)
        parts.append(bundle.apply(validation))
    calibrated = pd.concat(parts, ignore_index=True).sort_values("proposal_id").reset_index(drop=True)
    return calibrated, CalibrationBundle.fit(frame)


def proposal_nms(frame: pd.DataFrame, iou_threshold: float) -> pd.DataFrame:
    selected_indices: list[int] = []
    for _, group in frame.groupby(["subject_key", "session_id"], sort=False):
        kept: list[int] = []
        for index in group.sort_values("final_score", ascending=False).index:
            candidate = frame.loc[index]
            if any(
                interval_iou(
                    int(candidate.coarse_start_ms),
                    int(candidate.coarse_end_ms),
                    int(frame.loc[other].coarse_start_ms),
                    int(frame.loc[other].coarse_end_ms),
                )
                > iou_threshold
                for other in kept
            ):
                continue
            kept.append(int(index))
        selected_indices.extend(kept)
    return frame.loc[selected_indices].sort_values(
        ["subject_key", "session_id", "coarse_start_ms"]
    ).reset_index(drop=True)
