from __future__ import annotations

import hashlib
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
from bme_eating.data.deep_dataset import Normalization, load_normalization
from bme_eating.data.stats_fusion_preprocess import (
    RawSessionInput,
    StatsFusionRawSessionPreprocessor,
)
from bme_eating.data.stats_fusion_sequence import SequenceGeometry
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _proposal_nms(
    frame: pd.DataFrame, iou_threshold: float, minimum_gap_seconds: int = 3
) -> pd.DataFrame:
    from bme_eating.proposals_v4 import interval_iou

    kept: list[int] = []
    ranked_input = frame.sort_values(
        ["final_score", "coarse_start_ms", "coarse_end_ms", "proposal_id"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    for index in ranked_input.index:
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
    candidates = frame.loc[kept].copy()
    gap_ms = int(minimum_gap_seconds) * 1000
    separated: list[int] = []
    ranked = candidates.sort_values(
        ["final_score", "coarse_start_ms", "coarse_end_ms", "proposal_id"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    for index, row in ranked.iterrows():
        start = int(row.coarse_start_ms)
        end = int(row.coarse_end_ms)
        if any(
            not (
                end + gap_ms <= int(candidates.loc[other].coarse_start_ms)
                or int(candidates.loc[other].coarse_end_ms) + gap_ms <= start
            )
            for other in separated
        ):
            continue
        separated.append(int(index))
    return candidates.loc[separated].copy().reset_index(drop=True)


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
        verifier: EventVerifierV4 | Sequence[EventVerifierV4] | None,
        logistic_verifier: LogisticScoreCombiner | None,
        proposal_calibration: ProposalCalibrationV4 | None,
        boundary: EndpointRefiner | None,
        statistics_scaler: FoldRobustScaler,
        sensor_normalization: Normalization,
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
        self.verifiers = (
            []
            if verifier is None
            else list(verifier)
            if isinstance(verifier, Sequence)
            else [verifier]
        )
        self.logistic_verifier = logistic_verifier
        self.proposal_calibration = proposal_calibration
        self.boundary = boundary
        self.statistics_scaler = statistics_scaler
        self.sensor_normalization = sensor_normalization
        self.duration_prior = duration_prior
        self.boundary_range = boundary_range
        self.config = config
        self.selection = selection
        self.statistics_columns = [f"stat_{name}" for name in STATS_FEATURE_COLUMNS]
        for model in self.state_models:
            model.to(self.device).eval()
        for model in self.verifiers:
            model.to(self.device).eval()
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
    def _score_proposals(self, proposals: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
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
        predictions = [model(tensors) for model in self.verifiers]
        output["event_logit"] = (
            torch.stack([value["event_logit"].float() for value in predictions])
            .mean(dim=0)
            .cpu()
            .numpy()
        )
        output["iou_logit"] = (
            torch.stack([value["iou_logit"].float() for value in predictions])
            .mean(dim=0)
            .cpu()
            .numpy()
        )
        output["predicted_iou"] = 1.0 / (1.0 + np.exp(-output["iou_logit"]))
        return self.proposal_calibration.apply(output)

    @torch.no_grad()
    def _refine(self, accepted: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
        if self.boundary is None or self.boundary_range is None or accepted.empty:
            output = accepted.copy()
            output["refined_start_ms"] = output["coarse_start_ms"]
            output["refined_end_ms"] = output["coarse_end_ms"]
            output["start_entropy"] = 1.0
            output["end_entropy"] = 1.0
            output["start_fallback"] = True
            output["end_fallback"] = True
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
            batch["start_mask"],
        )
        end_offset, end_entropy = local_soft_argmax(
            prediction["end_logit"],
            torch.from_numpy(features.end_offsets_seconds).to(self.device),
            int(self.config["boundary"]["local_softargmax_radius_bins"]),
            batch["end_mask"],
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

    def _events_from_windows(self, windows: pd.DataFrame) -> list[Event]:
        if windows.empty:
            return []
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
            accepted,
            float(self.selection.get("nms_iou_threshold", 0.5)),
            int(self.selection.get("minimum_event_gap_seconds", 3)),
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

    def predict_preprocessed_session(
        self,
        state_batch: dict[str, Any],
        *,
        subject_key: str,
        session_id: str,
    ) -> list[Event]:
        windows = self.predict_state_sequence(
            state_batch, subject_key=subject_key, session_id=session_id
        )
        return self._events_from_windows(windows)

    def predict_session(self, session: RawSessionInput) -> list[Event]:
        sequence = self.config["sequence"]
        geometry = SequenceGeometry(
            supervised_steps=int(sequence["supervised_steps"]),
            short_receptive_field_steps=int(sequence["short_receptive_field_steps"]),
            long_receptive_field_tokens=int(sequence["long_receptive_field_tokens"]),
            long_pool_factor=int(sequence["long_pool_factor"]),
            step_seconds=int(sequence["step_seconds"]),
        )
        preprocessor = StatsFusionRawSessionPreprocessor(
            normalization=self.sensor_normalization,
            statistics_scaler=self.statistics_scaler,
            geometry=geometry,
        )
        frames = [
            self.predict_state_sequence(
                batch,
                subject_key=session.subject_key,
                session_id=session.session_id,
            )
            for batch in preprocessor.iter_state_batches(session)
        ]
        if not frames:
            return []
        windows = (
            pd.concat(frames, ignore_index=True)
            .sort_values("timestamp_ms")
            .drop_duplicates(["subject_key", "session_id", "timestamp_ms"], keep="last")
            .reset_index(drop=True)
        )
        return self._events_from_windows(windows)


def load_hierarchical_v4_bundle(
    bundle_root: str | Path, *, device: torch.device | str = "cpu"
) -> HierarchicalEatingDetectorV4:
    root = Path(bundle_root)
    hashes_path = root / "SHA256SUMS.json"
    if not hashes_path.is_file():
        raise FileNotFoundError("V4 bundle SHA256SUMS.json is required")
    expected_files = json.loads(hashes_path.read_text(encoding="utf-8"))["files"]
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS.json"
        and path.suffix.lower() != ".pyc"
        and "__pycache__" not in path.parts
    }
    if actual_files != set(expected_files):
        raise RuntimeError("V4 bundle file set differs from its hash manifest")
    for relative, expected in expected_files.items():
        if _sha256_file(root / relative) != expected:
            raise RuntimeError(f"V4 bundle artifact hash mismatch: {relative}")
    config = yaml.safe_load((root / "resolved_config.yaml").read_text(encoding="utf-8"))
    selection = json.loads((root / "selected_pipeline.json").read_text(encoding="utf-8"))
    if selection.get("protocol_version") != "statsfusion-r2":
        raise RuntimeError("Only statsfusion-r2 bundles are supported")
    if selection.get("selection_source") != "pooled_outer_oof":
        raise RuntimeError("V4 bundle selection must come from pooled outer OOF")
    state_seeds = [int(value) for value in selection.get("state_seeds", [])]
    if state_seeds != [2026, 2027, 2028]:
        raise RuntimeError("V4 bundle requires state seeds 2026/2027/2028")
    state_models: list[torch.nn.Module] = []
    for seed in state_seeds:
        checkpoint = torch.load(
            root / f"state_seed_{seed}.pt", map_location=device, weights_only=False
        )
        model = build_state_model(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"])
        state_models.append(model)
    verifier_models: list[EventVerifierV4] = []
    verifier_kind = str(selection.get("verifier_kind", ""))
    verifier_paths: list[Path] = []
    if verifier_kind == "deep":
        verifier_seeds = [int(value) for value in selection.get("verifier_seeds", [])]
        if verifier_seeds != [2026, 2027, 2028]:
            raise RuntimeError("Deep v4 bundle requires verifier seeds 2026/2027/2028")
        verifier_paths = [root / f"verifier_seed_{seed}.pt" for seed in verifier_seeds]
        missing = [path.name for path in verifier_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Deep v4 verifier checkpoints are missing: {missing}")
    elif verifier_kind != "logistic":
        raise RuntimeError(f"Unsupported v4 verifier kind: {verifier_kind}")
    for verifier_path in verifier_paths:
        checkpoint = torch.load(verifier_path, map_location=device, weights_only=False)
        verifier_model = EventVerifierV4(
            int(checkpoint["sequence_dim"]), int(checkpoint["scalar_dim"]), checkpoint["config"]
        )
        verifier_model.load_state_dict(checkpoint["model"])
        verifier_models.append(verifier_model)
    boundary = None
    boundary_path = root / "boundary.pt"
    if bool(selection.get("boundary_enabled", False)):
        if not boundary_path.is_file():
            raise FileNotFoundError("Selected v4 boundary checkpoint is missing")
        if not (root / "boundary_range.json").is_file():
            raise FileNotFoundError("Selected v4 boundary range is missing")
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
        verifier=verifier_models or None,
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
        sensor_normalization=load_normalization(root / "sensor_normalization.json"),
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
