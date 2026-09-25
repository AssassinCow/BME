from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from bme_eating.calibration_v4 import (
    LogisticScoreCombiner,
    PlattCalibration,
    ProposalCalibrationV4,
)
from bme_eating.models.endpoint_refiner import (
    BoundaryRange,
    EndpointRefiner,
    apply_boundary_refinement,
    build_endpoint_features,
    local_soft_argmax,
)
from bme_eating.models.event_verifier_v4 import (
    EventVerifierV4,
    ProposalFeatureBatchV4,
    build_proposal_features_v4,
)
from bme_eating.models.factory import build_state_model
from bme_eating.proposals_v4 import generate_event_candidates_v4
from bme_eating.stats_features import STATS_FEATURE_COLUMNS, FoldRobustScaler
from bme_eating.structured_decoder import (
    FixedLagSemiMarkovDecoder,
    TruncatedLogNormalDurationPrior,
)
from bme_eating.types import Event


def _proposal_nms(frame: pd.DataFrame, iou_threshold: float) -> pd.DataFrame:
    from bme_eating.proposals_v4 import interval_iou

    kept: list[int] = []
    for index in frame.sort_values("final_score", ascending=False).index:
        row = frame.loc[index]
        if any(
            interval_iou(
                int(row.coarse_start_ms),
                int(row.coarse_end_ms),
                int(frame.loc[other].coarse_start_ms),
                int(frame.loc[other].coarse_end_ms),
            )
            > iou_threshold
            for other in kept
        ):
            continue
        kept.append(int(index))
    return frame.loc[kept].copy().reset_index(drop=True)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _feature_batch_to_tensors(
    features: ProposalFeatureBatchV4, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "sequence": torch.from_numpy(features.sequence).to(device),
        "sequence_mask": torch.from_numpy(features.sequence_mask).to(device),
        "scalar": torch.from_numpy(features.scalar).to(device),
    }


class HierarchicalEatingDetectorV4:
    def __init__(
        self,
        *,
        state_models: Sequence[torch.nn.Module],
        state_calibration: PlattCalibration,
        verifier: EventVerifierV4 | None,
        logistic_verifier: LogisticScoreCombiner | None,
        proposal_calibration: ProposalCalibrationV4 | None,
        boundary: EndpointRefiner | None,
        statistics_scaler: FoldRobustScaler,
        duration_prior: TruncatedLogNormalDurationPrior,
        boundary_range: BoundaryRange | None,
        config: dict[str, Any],
        selection: dict[str, Any],
        device: torch.device | str = "cpu",
    ) -> None:
        if not state_models:
            raise ValueError("V4 inference requires at least one state model")
        if int(config["hierarchical"]["maximum_event_latency_seconds"]) > 60:
            raise ValueError("V4 inference exceeds the 60-second future-data limit")
        if int(config["decoder"]["fixed_lag_seconds"]) > 60:
            raise ValueError("V4 decoder exceeds the 60-second fixed lag")
        verifier_kind = str(selection.get("verifier_kind", "deep"))
        if verifier_kind == "deep" and (verifier is None or proposal_calibration is None):
            raise ValueError("Deep verifier selection requires model and calibration")
        if verifier_kind == "logistic" and logistic_verifier is None:
            raise ValueError("Logistic verifier selection requires its fitted model")
        self.device = torch.device(device)
        self.state_models = list(state_models)
        self.state_calibration = state_calibration
        self.verifier = verifier
        self.logistic_verifier = logistic_verifier
        self.proposal_calibration = proposal_calibration
        self.boundary = boundary
        self.statistics_scaler = statistics_scaler
        self.duration_prior = duration_prior
        self.boundary_range = boundary_range
        self.config = config
        self.selection = selection
        self.statistics_columns = [f"stat_{name}" for name in STATS_FEATURE_COLUMNS]
        for model in self.state_models:
            model.to(self.device).eval()
        if self.verifier is not None:
            self.verifier.to(self.device).eval()
        if self.boundary is not None:
            self.boundary.to(self.device).eval()

    @property
    def decoder(self) -> FixedLagSemiMarkovDecoder:
        return FixedLagSemiMarkovDecoder(
            self.duration_prior,
            grid_seconds=int(self.config["decoder"]["grid_seconds"]),
            fixed_lag_seconds=int(self.config["decoder"]["fixed_lag_seconds"]),
        )

    @torch.no_grad()
    def predict_state_sequence(
        self,
        state_batch: dict[str, Any],
        *,
        subject_key: str,
        session_id: str,
    ) -> pd.DataFrame:
        batch = _to_device(state_batch, self.device)
        outputs = [model(batch) for model in self.state_models]
        names = ("state_logit", "onset_logit", "offset_logit")
        averaged = {
            name: torch.stack([output[name].float() for output in outputs]).mean(dim=0)
            for name in names
        }
        primary = outputs[0]
        if averaged["state_logit"].shape[0] != 1:
            raise ValueError("predict_session currently accepts one session batch")
        timestamps = batch["timestamp_ms"][0].detach().cpu().numpy().astype(np.int64)
        supervision = batch.get("supervision_mask")
        selected = (
            supervision[0].detach().cpu().numpy() > 0
            if supervision is not None
            else np.ones(len(timestamps), dtype=bool)
        )
        state_logit = averaged["state_logit"][0].cpu().numpy()
        state_probability = self.state_calibration.transform(state_logit)
        statistics = batch["statistics"][0].float().cpu().numpy()
        frame = pd.DataFrame(
            {
                "subject_key": subject_key,
                "session_id": session_id,
                "timestamp_ms": timestamps,
                "state_logit": state_logit,
                "state_probability": state_probability,
                "onset_probability": torch.sigmoid(averaged["onset_logit"][0]).cpu().numpy(),
                "offset_probability": torch.sigmoid(averaged["offset_logit"][0]).cpu().numpy(),
                "ppg_gate": primary["ppg_gate"][0].float().cpu().numpy(),
                "statistics_gate": primary["statistics_gate"][0].float().cpu().numpy(),
                "missing_fraction": primary["missing_fraction"][0].float().cpu().numpy(),
            }
        )
        for index, name in enumerate(self.statistics_columns):
            frame[name] = statistics[:, index]
        frame["state_probability_derivative"] = frame["state_probability"].diff().fillna(0.0)
        return frame.loc[selected].reset_index(drop=True)

    def _logistic_features(self, features: ProposalFeatureBatchV4) -> np.ndarray:
        mask = features.sequence_mask[..., None]
        count = mask.sum(axis=1).clip(min=1)
        mean = (features.sequence * mask).sum(axis=1) / count
        maximum = np.where(mask, features.sequence, -np.inf).max(axis=1)
        maximum[~np.isfinite(maximum)] = 0.0
        return np.concatenate((mean, maximum, features.scalar), axis=1)

    @torch.no_grad()
    def _score_proposals(
        self, proposals: pd.DataFrame, windows: pd.DataFrame
    ) -> pd.DataFrame:
        features = build_proposal_features_v4(
            proposals,
            windows,
            self.statistics_columns,
            self.config["verifier"],
        )
        output = proposals.copy().reset_index(drop=True)
        output["state_score"] = features.scalar[:, 2]
        if self.selection.get("verifier_kind", "deep") == "logistic":
            output["final_score"] = self.logistic_verifier.predict(
                self._logistic_features(features)
            )
            output["event_logit"] = np.nan
            output["predicted_iou"] = np.nan
            return output
        tensors = _feature_batch_to_tensors(features, self.device)
        prediction = self.verifier(tensors)
        output["event_logit"] = prediction["event_logit"].cpu().numpy()
        output["iou_logit"] = prediction["iou_logit"].cpu().numpy()
        output["predicted_iou"] = prediction["predicted_iou"].cpu().numpy()
        return self.proposal_calibration.apply(output)

    @torch.no_grad()
    def _refine(self, accepted: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
        if self.boundary is None or self.boundary_range is None or accepted.empty:
            output = accepted.copy()
            output["refined_start_ms"] = output["coarse_start_ms"]
            output["refined_end_ms"] = output["coarse_end_ms"]
            output["start_entropy"] = 1.0
            output["end_entropy"] = 1.0
            output["boundary_fallback"] = True
            return output
        features = build_endpoint_features(
            accepted,
            windows,
            self.statistics_columns,
            self.boundary_range,
            self.config["boundary"],
        )
        batch = {
            "start_sequence": torch.from_numpy(features.start_sequence).to(self.device),
            "end_sequence": torch.from_numpy(features.end_sequence).to(self.device),
            "start_mask": torch.from_numpy(features.start_mask).to(self.device),
            "end_mask": torch.from_numpy(features.end_mask).to(self.device),
        }
        prediction = self.boundary(batch)
        start_offset, start_entropy = local_soft_argmax(
            prediction["start_logit"],
            torch.from_numpy(features.start_offsets_seconds).to(self.device),
            int(self.config["boundary"]["local_softargmax_radius_bins"]),
        )
        end_offset, end_entropy = local_soft_argmax(
            prediction["end_logit"],
            torch.from_numpy(features.end_offsets_seconds).to(self.device),
            int(self.config["boundary"]["local_softargmax_radius_bins"]),
        )
        return apply_boundary_refinement(
            accepted,
            start_offset.cpu().numpy(),
            end_offset.cpu().numpy(),
            start_entropy.cpu().numpy(),
            end_entropy.cpu().numpy(),
            entropy_threshold=float(self.selection.get("boundary_entropy_threshold", 0.75)),
            safety_gap_seconds=int(self.config["boundary"]["safety_gap_seconds"]),
        )

    def predict_session(
        self,
        state_batch: dict[str, Any],
        *,
        subject_key: str,
        session_id: str,
    ) -> list[Event]:
        windows = self.predict_state_sequence(
            state_batch, subject_key=subject_key, session_id=session_id
        )
        proposals = generate_event_candidates_v4(
            windows,
            self.decoder,
            self.config["decoder"],
            split_role="inference",
        )
        if proposals.empty:
            return []
        scored = self._score_proposals(proposals, windows)
        accepted = scored[
            scored["final_score"] >= float(self.selection["acceptance_threshold"])
        ].copy()
        accepted = _proposal_nms(
            accepted, float(self.selection.get("nms_iou_threshold", 0.5))
        )
        refined = self._refine(accepted, windows)
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


def load_hierarchical_v4_bundle(
    bundle_root: str | Path, *, device: torch.device | str = "cpu"
) -> HierarchicalEatingDetectorV4:
    root = Path(bundle_root)
    config = yaml.safe_load((root / "resolved_config.yaml").read_text(encoding="utf-8"))
    selection = json.loads((root / "selected_pipeline.json").read_text(encoding="utf-8"))
    state_models: list[torch.nn.Module] = []
    for seed in (2026, 2027, 2028):
        checkpoint = torch.load(root / f"state_seed_{seed}.pt", map_location=device, weights_only=False)
        model = build_state_model(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"])
        state_models.append(model)
    verifier = None
    verifier_path = root / "verifier.pt"
    if verifier_path.is_file():
        checkpoint = torch.load(verifier_path, map_location=device, weights_only=False)
        verifier = EventVerifierV4(
            int(checkpoint["sequence_dim"]), int(checkpoint["scalar_dim"]), checkpoint["config"]
        )
        verifier.load_state_dict(checkpoint["model"])
    boundary = None
    boundary_path = root / "boundary.pt"
    if boundary_path.is_file():
        checkpoint = torch.load(boundary_path, map_location=device, weights_only=False)
        boundary = EndpointRefiner(int(checkpoint["input_dim"]), checkpoint["config"])
        boundary.load_state_dict(checkpoint["model"])
    logistic_path = root / "logistic_verifier.json"
    proposal_calibration_path = root / "proposal_calibration.json"
    boundary_range_path = root / "boundary_range.json"
    return HierarchicalEatingDetectorV4(
        state_models=state_models,
        state_calibration=PlattCalibration.from_json(
            json.loads((root / "state_calibration.json").read_text(encoding="utf-8"))
        ),
        verifier=verifier,
        logistic_verifier=(
            LogisticScoreCombiner.from_json(json.loads(logistic_path.read_text(encoding="utf-8")))
            if logistic_path.is_file()
            else None
        ),
        proposal_calibration=(
            ProposalCalibrationV4.from_json(
                json.loads(proposal_calibration_path.read_text(encoding="utf-8"))
            )
            if proposal_calibration_path.is_file()
            else None
        ),
        boundary=boundary,
        statistics_scaler=FoldRobustScaler.from_json(
            json.loads((root / "statistics_scaler.json").read_text(encoding="utf-8"))
        ),
        duration_prior=TruncatedLogNormalDurationPrior.from_json(
            json.loads((root / "duration_prior.json").read_text(encoding="utf-8"))
        ),
        boundary_range=(
            BoundaryRange(**json.loads(boundary_range_path.read_text(encoding="utf-8")))
            if boundary_range_path.is_file()
            else None
        ),
        config=config,
        selection=selection,
        device=device,
    )
