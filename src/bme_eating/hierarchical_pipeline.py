from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from xgboost import XGBClassifier

from bme_eating.calibration import CalibrationBundle, proposal_nms
from bme_eating.models.boundary_refiner import BoundaryRefiner, decode_boundaries
from bme_eating.models.event_verifier import EventVerifier, build_proposal_features
from bme_eating.models.factory import build_state_model
from bme_eating.postprocess import probabilities_to_events
from bme_eating.proposals import generate_event_candidates
from bme_eating.types import Event


class HierarchicalEatingDetector:
    def __init__(
        self,
        verifier: EventVerifier,
        boundary: BoundaryRefiner,
        calibration: CalibrationBundle,
        config: dict[str, Any],
        stable_feature_columns: list[str],
        *,
        acceptance_threshold: float,
        nms_iou_threshold: float,
        boundary_entropy_threshold: float,
        score_column: str = "final_score",
        device: torch.device | str = "cpu",
        state_models: Sequence[torch.nn.Module] = (),
        xgb_model: XGBClassifier | None = None,
        xgb_feature_columns: Sequence[str] = (),
        xgb_postprocess: dict[str, Any] | None = None,
        duration_bounds: tuple[float, float] | None = None,
    ) -> None:
        self.verifier = verifier.to(device).eval()
        self.boundary = boundary.to(device).eval()
        self.calibration = calibration
        self.config = config
        self.stable_feature_columns = list(stable_feature_columns)
        self.acceptance_threshold = float(acceptance_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.boundary_entropy_threshold = float(boundary_entropy_threshold)
        self.score_column = str(score_column)
        self.device = torch.device(device)
        self.state_models = [model.to(device).eval() for model in state_models]
        self.xgb_model = xgb_model
        self.xgb_feature_columns = list(xgb_feature_columns)
        self.xgb_postprocess = dict(xgb_postprocess or {})
        self.duration_bounds = duration_bounds
        maximum_latency = int(config["hierarchical"]["maximum_event_latency_seconds"])
        if int(config["verifier"]["right_context_seconds"]) > maximum_latency:
            raise ValueError("Verifier right context exceeds the maximum event latency")

    def predict_state_windows(
        self,
        batches: Iterable[dict[str, Any]],
    ) -> pd.DataFrame:
        if not self.state_models:
            raise RuntimeError("No state models are attached to this detector")
        rows: list[dict[str, Any]] = []
        for batch in batches:
            moved = {
                key: value.to(self.device)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            with torch.inference_mode():
                outputs = [model(moved) for model in self.state_models]
            primary = outputs[0]
            state_logit = torch.stack([value["state_logit"] for value in outputs]).mean(0)
            start_logit = torch.stack([value["start_logit"] for value in outputs]).mean(0)
            end_logit = torch.stack([value["end_logit"] for value in outputs]).mean(0)
            ppg_valid = moved["ppg_valid"].to(primary["ppg_gate"].dtype)
            motion_valid = moved["motion_valid"].to(primary["ppg_gate"].dtype)
            ppg_gate_mean = (primary["ppg_gate"] * ppg_valid).sum(dim=1) / ppg_valid.sum(
                dim=1
            ).clamp_min(1)
            batch_size = len(state_logit)
            for index in range(batch_size):
                try:
                    row: dict[str, Any] = {
                        "subject_key": str(batch["subject_key"][index]),
                        "segment_id": str(batch["segment_id"][index]),
                        "session_id": str(batch["session_id"][index]),
                        "timestamp_ms": int(batch["timestamp_ms"][index]),
                        "state_logit": float(state_logit[index].cpu()),
                        "start_logit": float(start_logit[index].cpu()),
                        "end_logit": float(end_logit[index].cpu()),
                        "state_probability": float(torch.sigmoid(state_logit[index]).cpu()),
                        "start_probability": float(torch.sigmoid(start_logit[index]).cpu()),
                        "end_probability": float(torch.sigmoid(end_logit[index]).cpu()),
                        "ppg_gate_mean": float(ppg_gate_mean[index].cpu()),
                        "ppg_gate_recent": float(primary["ppg_gate"][index, -1].cpu()),
                        "ppg_valid_fraction": float(ppg_valid[index].mean().cpu()),
                        "motion_valid_fraction": float(motion_valid[index].mean().cpu()),
                        "missing_fraction": float(
                            primary["missing_fraction"][index].reshape(-1)[0].cpu()
                        ),
                    }
                except KeyError as error:
                    raise ValueError(
                        f"State inference batch is missing identifier {error.args[0]}"
                    ) from error
                row.update(
                    {
                        f"state_embedding_{dimension:02d}": float(value)
                        for dimension, value in enumerate(
                            primary["state_embedding"][index].float().cpu().tolist()
                        )
                    }
                )
                rows.append(row)
        return pd.DataFrame(rows).sort_values(
            ["subject_key", "session_id", "timestamp_ms"]
        ).reset_index(drop=True)

    def _xgboost_events(self, features: pd.DataFrame) -> pd.DataFrame | None:
        if self.xgb_model is None:
            return None
        missing = set(self.xgb_feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"XGBoost inference features are missing: {sorted(missing)}")
        predictions = features[
            ["subject_key", "segment_id", "session_id", "timestamp_ms"]
        ].copy()
        matrix = features[self.xgb_feature_columns].to_numpy(dtype=np.float32)
        predictions["state_probability"] = self.xgb_model.predict_proba(matrix)[:, 1]
        predictions["start_probability"] = 0.0
        predictions["end_probability"] = 0.0
        for indices in predictions.groupby(["subject_key", "session_id"]).groups.values():
            ordered = predictions.loc[indices].sort_values("timestamp_ms")
            values = ordered["state_probability"].to_numpy(dtype=np.float64)
            derivative = np.diff(values, prepend=values[0])
            predictions.loc[ordered.index, "start_probability"] = np.clip(derivative, 0, 1)
            predictions.loc[ordered.index, "end_probability"] = np.clip(-derivative, 0, 1)
        accepted = {
            key: value
            for key, value in self.xgb_postprocess.items()
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
        return probabilities_to_events(predictions, **accepted)

    def _tensor_batch(self, sequence: np.ndarray, scalar: np.ndarray) -> dict[str, torch.Tensor]:
        return {
            "sequence": torch.from_numpy(sequence).to(self.device),
            "scalar": torch.from_numpy(scalar).to(self.device),
        }

    def score_proposals(
        self,
        proposals: pd.DataFrame,
        window_predictions: pd.DataFrame,
        stable_features: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, torch.Tensor]]:
        features = build_proposal_features(
            proposals,
            window_predictions,
            stable_features,
            self.stable_feature_columns,
            self.config["verifier"],
        )
        batch = self._tensor_batch(features.sequence, features.scalar)
        with torch.inference_mode():
            verifier_output = self.verifier(batch)
        scored = proposals.copy()
        scored["event_logit"] = verifier_output["event_logit"].cpu().numpy()
        scored["iou_logit"] = verifier_output["iou_logit"].cpu().numpy()
        scored["predicted_iou"] = verifier_output["predicted_iou"].cpu().numpy()
        scored["state_score"] = features.sequence[:, :, 0].mean(axis=1)
        return self.calibration.apply(scored), batch

    def predict_session(
        self,
        window_predictions: pd.DataFrame | Iterable[dict[str, Any]],
        stable_features: pd.DataFrame,
        duration_bounds: tuple[float, float] | None = None,
        *,
        xgb_events: pd.DataFrame | None = None,
    ) -> list[Event]:
        if not isinstance(window_predictions, pd.DataFrame):
            window_predictions = self.predict_state_windows(window_predictions)
        if duration_bounds is None:
            duration_bounds = self.duration_bounds
        if duration_bounds is None:
            raise ValueError("Inference duration bounds were not provided or bundled")
        sessions = window_predictions[["subject_key", "session_id"]].drop_duplicates()
        if len(sessions) != 1:
            raise ValueError("predict_session requires exactly one subject/session timeline")
        if xgb_events is None:
            xgb_events = self._xgboost_events(stable_features)
        proposals = generate_event_candidates(
            window_predictions,
            self.config["proposals"],
            duration_bounds,
            split_role="inference",
            xgb_events=xgb_events,
        )
        if proposals.empty:
            return []
        scored, _ = self.score_proposals(proposals, window_predictions, stable_features)
        if self.score_column not in scored:
            raise RuntimeError(f"Bundled score column is unavailable: {self.score_column}")
        if self.score_column != "final_score":
            scored["combined_final_score"] = scored["final_score"]
            scored["final_score"] = scored[self.score_column]
        accepted = scored[scored["final_score"] >= self.acceptance_threshold].copy()
        accepted = proposal_nms(accepted, self.nms_iou_threshold)
        if accepted.empty:
            return []

        accepted_features = build_proposal_features(
            accepted,
            window_predictions,
            stable_features,
            self.stable_feature_columns,
            self.config["verifier"],
        )
        boundary_batch = self._tensor_batch(
            accepted_features.sequence, accepted_features.scalar
        )
        with torch.inference_mode():
            boundary_output = self.boundary(boundary_batch)
        observation_end = {
            (str(subject), str(session)): int(group["timestamp_ms"].max())
            for (subject, session), group in window_predictions.groupby(
                ["subject_key", "session_id"], sort=False
            )
        }
        refined = decode_boundaries(
            accepted,
            boundary_output,
            self.config["boundary"],
            self.boundary_entropy_threshold,
            observation_end,
        )
        if len(refined) != len(accepted) or set(refined["proposal_id"]) != set(
            accepted["proposal_id"]
        ):
            raise RuntimeError("Boundary refinement changed the accepted event identity set")
        return [
            Event(
                subject_key=str(row.subject_key),
                start_ms=int(row.refined_start_ms),
                end_ms=int(row.refined_end_ms),
                score=float(row.final_score),
                event_id=str(row.proposal_id),
            )
            for row in refined.itertuples(index=False)
        ]


def load_hierarchical_bundle(
    bundle_root: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> HierarchicalEatingDetector:
    root = Path(bundle_root)
    config = yaml.safe_load((root / "resolved_config.yaml").read_text(encoding="utf-8"))
    state_models: list[torch.nn.Module] = []
    for seed in (2026, 2027, 2028):
        checkpoint = torch.load(
            root / f"state_seed_{seed}.pt", map_location=device, weights_only=False
        )
        model = build_state_model(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"])
        state_models.append(model)
    verifier_checkpoint = torch.load(
        root / "verifier.pt", map_location=device, weights_only=False
    )
    verifier = EventVerifier(
        int(verifier_checkpoint["sequence_dim"]),
        int(verifier_checkpoint["scalar_dim"]),
        verifier_checkpoint["config"],
    )
    verifier.load_state_dict(verifier_checkpoint["model"])
    boundary_checkpoint = torch.load(
        root / "boundary.pt", map_location=device, weights_only=False
    )
    boundary = BoundaryRefiner(
        int(boundary_checkpoint["sequence_dim"]),
        int(boundary_checkpoint["scalar_dim"]),
        boundary_checkpoint["config"],
    )
    boundary.load_state_dict(boundary_checkpoint["model"])
    xgb_metadata = json.loads(
        (root / "xgboost_full.metadata.json").read_text(encoding="utf-8")
    )
    xgb_model = XGBClassifier()
    xgb_model.load_model(root / "xgboost_full.json")
    selection = json.loads((root / "selected_pipeline.json").read_text(encoding="utf-8"))
    bounds = json.loads((root / "duration_bounds.json").read_text(encoding="utf-8"))
    return HierarchicalEatingDetector(
        verifier,
        boundary,
        CalibrationBundle.from_json(
            json.loads((root / "calibration.json").read_text(encoding="utf-8"))
        ),
        config,
        list(config["hierarchical"]["stable_feature_columns"]),
        acceptance_threshold=float(selection["acceptance_threshold"]),
        nms_iou_threshold=float(selection["nms_iou_threshold"]),
        boundary_entropy_threshold=float(selection["boundary_entropy_threshold"]),
        score_column=str(selection.get("score_column", "final_score")),
        device=device,
        state_models=state_models,
        xgb_model=xgb_model,
        xgb_feature_columns=xgb_metadata["feature_columns"],
        xgb_postprocess=json.loads(
            (root / "xgboost_postprocess.json").read_text(encoding="utf-8")
        ),
        duration_bounds=(float(bounds["minimum_seconds"]), float(bounds["maximum_seconds"])),
    )
