from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bme_eating.data.deep_dataset import Normalization
from bme_eating.data.joint_sessions import joint_geometry
from bme_eating.data.stats_fusion_preprocess import (
    RawSessionInput,
    StatsFusionRawSessionPreprocessor,
)
from bme_eating.hierarchical_artifacts import sha256_file
from bme_eating.joint_evaluation import NonnegativePlatt
from bme_eating.joint_protocol import (
    PROTOCOL,
    checkpoint_qualified,
    config_identity,
    configure_numerics,
    digest,
    validate_config,
    validate_gate_structure,
    verify_artifact_hashes,
)
from bme_eating.joint_stream import (
    FrozenSourceCDF,
    ImmutableCommitter,
    OnlineBudget,
    StreamingProposals,
)
from bme_eating.models.statsfusion_joint_event import StatsFusionJointEvent
from bme_eating.stats_features import FoldRobustScaler
from bme_eating.training.statsfusion_joint_trainer import (
    decoder_settings,
    pack_features,
    refine_candidate,
    score_candidate,
)
from bme_eating.types import Event


class JointChunkDetector:
    def __init__(self, model, config, point, mapping, identity, boundary, subject, session):
        configure_numerics(config)
        self.model, self.config, self.point = model.eval(), config, point
        self.mapping, self.identity, self.boundary = mapping, identity, boundary
        self.device = next(model.parameters()).device
        self.stream = StreamingProposals(subject, session, identity, decoder_settings(config)[point["setting"]])
        self.budget = OnlineBudget(point["budget"], config["decoder"]["quotas"])
        self.committer = ImmutableCommitter(point["acceptance"], point["nms"])
        self.calibrator = NonnegativePlatt(**point["calibration"])
        self.times = torch.empty(0, dtype=torch.int64)
        self.features = torch.empty(0, 137)
        self.finished = False

    @torch.no_grad()
    def push_features(self, times: torch.Tensor, features: torch.Tensor, *, final: bool = False):
        if self.finished:
            raise RuntimeError("Finished session cannot accept future chunks")
        if (times.ndim != 1 or times.dtype != torch.int64 or features.shape != (len(times), 137)
                or not torch.isfinite(features).all() or (torch.diff(times) != 3000).any()
                or ((features[:, 128:] < 0) | (features[:, 128:] > 1)).any()):
            raise ValueError("Invalid streaming timeline, feature shape, finite values or masks")
        if len(self.times) and len(times) and int(times[0]) != int(self.times[-1]) + 3000:
            raise ValueError("Feature chunks overlap, move backwards or skip missing bins")
        self.times = torch.cat((self.times, times.cpu()))
        self.features = torch.cat((self.features, features.cpu()))
        emitted = []
        for timestamp, values in zip(times.tolist(), features.cpu().numpy()):
            self.budget.observe(3, float(values[128]))
            probabilities = {"state": float(values[129]),
                             **{f"proposal_{scale}": float(values[130 + index])
                                for index, scale in enumerate((3, 15, 60))}}
            candidates = self.stream.observe(int(timestamp), probabilities, float(values[128]),
                                              float(values[134]))
            emitted.extend(self._decide(candidates))
        if final:
            emitted.extend(self._decide(self.stream.finish()))
            self.finished = True
        return [candidate.to_dict() for candidate in emitted]

    def _decide(self, candidates):
        chosen = self.budget.select(candidates, self.mapping) if candidates else []
        for index, candidate in enumerate(chosen):
            score_candidate(self.model, candidate, self.times, self.features, self.device)
            if self.point["boundary_enabled"]:
                candidate = refine_candidate(self.model, candidate, self.times, self.features, self.device,
                                               self.boundary, self.point["boundary_entropy"])
                chosen[index] = candidate
            candidate.event_probability = float(self.calibrator.transform(candidate.event_logit))
            candidate.rank_score = candidate.event_probability * candidate.predicted_quality
            if candidate.coarse_event_logit is not None:
                candidate.coarse_probability = float(self.calibrator.transform(candidate.coarse_event_logit))
        return self.committer.submit(chosen)

    @torch.no_grad()
    def push_batch(self, batch, *, final=False):
        batch = {name: value.to(self.device) if isinstance(value, torch.Tensor) else value
                 for name, value in batch.items()}
        outputs = self.model(batch)
        selected = batch["supervision_mask"][0] > 0
        return self.push_features(batch["timestamp_ms"][0, selected], pack_features(outputs, selected),
                                   final=final)


def causal_raw_batches(session: RawSessionInput, preprocessing: dict[str, Any], config: dict[str, Any],
                       new_steps: int = 256, *, after_ms: int | None = None, session_finished: bool = True):
    session = session.validated()
    geometry = joint_geometry(config)
    processor = StatsFusionRawSessionPreprocessor(
        normalization=Normalization.from_json(preprocessing["normalization"]),
        statistics_scaler=FoldRobustScaler.from_json(preprocessing["scaler"]), geometry=geometry)
    anchors = processor.anchors(session)
    if not len(anchors):
        return
    origin = int(anchors[0]) - 3000
    initial = int(np.searchsorted(anchors, after_ms, side="right")) if after_ms is not None else 0
    for first in range(initial, len(anchors), new_steps):
        stop = min(len(anchors), first + new_steps)
        end = int(anchors[stop - 1])
        steps = geometry.history_steps + stop - first
        times = end - np.arange(steps - 1, -1, -1, dtype=np.int64) * 3000
        motion, motion_valid, invariant = processor._motion_blocks(session, int(times[0]), steps)
        ppg, quality, ppg_valid, mapping, ends = processor._ppg_blocks(session, times, origin)
        statistics = np.zeros((steps, 24), dtype=np.float32)
        statistics[:, 12:] = 1
        observed = times >= anchors[0]
        statistics[observed] = processor._statistics(session, times[observed])
        mask = np.zeros(steps, dtype=np.float32)
        mask[-(stop - first):] = 1
        arrays = {"motion_blocks": motion, "motion_valid": motion_valid,
                  "motion_invariant_blocks": invariant, "ppg_blocks": ppg,
                  "ppg_quality": quality, "ppg_valid": ppg_valid,
                  "ppg_to_motion_index": mapping, "long_block_end_indices": ends,
                  "statistics": statistics, "timestamp_ms": times,
                  "supervision_mask": mask, "timeline_position": (times - origin) // 3000 - 1}
        yield {name: torch.from_numpy(value)[None] for name, value in arrays.items()}, session_finished and stop == len(anchors)


def public_events(diagnostics):
    return [Event(subject_key=event["subject_key"], start_ms=event["start_ms"], end_ms=event["end_ms"],
                  score=event["rank_score"], event_id=event["proposal_id"], hand_relation="unknown")
            for event in diagnostics]


class JointRawStream:
    def __init__(self, detector: JointChunkDetector, preprocessing, config):
        self.detector, self.preprocessing, self.config = detector, preprocessing, config
        self.raw: RawSessionInput | None = None
        self.last_input_ms: int | None = None
        self.diagnostics = []

    def push(self, chunk: RawSessionInput, *, final: bool = False):
        if self.detector.finished:
            raise RuntimeError("Finished raw session cannot accept future samples")
        chunk = chunk.validated()
        if chunk.subject_key != self.detector.stream.subject or chunk.session_id != self.detector.stream.session:
            raise ValueError("Incremental raw subject/session identity changed")
        if self.raw is None:
            self.raw = chunk
        else:
            if any(getattr(self.raw, name) != getattr(chunk, name) for name in
                   ("schema_version", "timestamp_unit", "motion_channel_order", "unit_contract_id", "resampling_state")):
                raise ValueError("Incremental raw sensor contract changed")
            arrays = {}
            for modality in ("motion", "ppg"):
                previous_time = getattr(self.raw, modality + "_timestamp_ms")
                new_time = getattr(chunk, modality + "_timestamp_ms")
                if len(previous_time) and len(new_time) and new_time[0] <= previous_time[-1]:
                    raise ValueError("Incremental raw chunks overlap or move backwards")
                if self.last_input_ms is not None and len(new_time) and new_time[0] <= self.last_input_ms:
                    raise ValueError("Late raw samples would change an already encoded prefix")
                for suffix in ("timestamp_ms", "values", "mask"):
                    name = modality + "_" + suffix
                    arrays[name] = np.concatenate((getattr(self.raw, name), getattr(chunk, name)))
            self.raw = replace(self.raw, **arrays)
        emitted = []
        for batch, finished in causal_raw_batches(self.raw, self.preprocessing, self.config,
                                                  after_ms=self.last_input_ms, session_finished=final):
            emitted.extend(self.detector.push_batch(batch, final=finished))
            self.last_input_ms = int(batch["timestamp_ms"][0, -1])
        if final and not self.detector.finished:
            emitted.extend(self.detector.push_features(torch.empty(0, dtype=torch.int64),
                                                       torch.empty(0, 137), final=True))
        self.diagnostics.extend(emitted)
        return public_events(emitted)


class StatsFusionJointDetector:
    def __init__(self, bundle: str | Path, *, device: str = "cpu") -> None:
        root = Path(bundle)
        manifest = json.loads((root / "final_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("protocol") != PROTOCOL or not manifest.get("promotable"):
            raise RuntimeError("Bundle has no qualified joint manifest")
        checksums = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
        verify_artifact_hashes(root, checksums)
        files = {path.relative_to(root).as_posix() for path in root.rglob("*")
                 if path.is_file() and path.name != "SHA256SUMS.json" and "__pycache__" not in path.parts}
        if files != set(checksums):
            raise RuntimeError("Joint bundle contains unlisted files or incomplete checksum coverage")
        if not checksums or "checkpoint.pt" not in checksums:
            raise RuntimeError("Incomplete joint bundle checksum set")
        required = {"checkpoint.pt", "final_manifest.json", "joint_gate_manifest.json", "identity.json",
                    "resolved_config.json", "operating_point.json", "source_mapping.json"}
        if not required.issubset(checksums):
            raise RuntimeError("Joint bundle lacks audited deployment artifacts")
        gate = json.loads((root / "joint_gate_manifest.json").read_text(encoding="utf-8"))
        identity_hash = sha256_file(root / "identity.json")
        if (not gate.get("promotable") or gate.get("protocol") != PROTOCOL
                or manifest.get("identity_sha256") != identity_hash
                or gate.get("identity_sha256") != identity_hash
                or manifest.get("gate_sha256") != sha256_file(root / "joint_gate_manifest.json")):
            raise RuntimeError("Joint bundle gate/identity lock failed")
        validate_gate_structure(gate)
        self.config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
        validate_config(self.config)
        configure_numerics(self.config)
        self.point = json.loads((root / "operating_point.json").read_text(encoding="utf-8"))
        payload = torch.load(root / "checkpoint.pt", map_location=device, weights_only=True)
        identity = json.loads((root / "identity.json").read_text(encoding="utf-8"))
        subjects = {subject for fold in gate["folds"] for subject in fold["prediction_subjects"]}
        if (set(manifest.get("training_subjects", [])) != subjects
                or payload.get("data_sha256") != digest(identity.get("data"))
                or payload.get("source_sha256") != digest(identity.get("source"))
                or payload.get("excluded_subjects")
                or set(payload.get("preprocessing", {}).get("training_subjects", [])) != subjects):
            raise RuntimeError("Bundle checkpoint data/source/preprocessing/cohort lineage differs")
        for relative, expected in identity.get("source", {}).items():
            if relative.startswith("src/") and checksums.get(relative) != expected:
                raise RuntimeError("Bundle source differs from the frozen runtime identity")
        if not checkpoint_qualified(payload, manifest["model_identity"], set(payload["training_subjects"]),
                                     self.config["training"]["checkpoint_min_epoch"]):
            raise RuntimeError("Bundle checkpoint is unqualified")
        if payload["config_sha256"] != config_identity(self.config):
            raise RuntimeError("Bundle checkpoint configuration differs")
        from bme_eating.joint_workflow import verify_oof_lineage
        if not manifest.get("internal_oof_sha256"):
            raise RuntimeError("Bundle lacks full joint internal OOF qualification")
        verify_oof_lineage(manifest.get("lineage", []), subjects, set(), identities={
            name: payload[name] for name in ("config_sha256", "source_sha256", "data_sha256")})
        for name, relative in (("checkpoint", "checkpoint.pt"), ("operating_point", "operating_point.json"),
                               ("source_mapping", "source_mapping.json")):
            if manifest.get(name + "_sha256") != sha256_file(root / relative):
                raise RuntimeError("Bundle final manifest artifact binding failed")
        if not self.point.get("promotion_eligible") or not self.point["calibration"].get("qualified"):
            raise RuntimeError("Bundle has an unqualified operating point")
        if self.point["boundary_enabled"] != manifest.get("boundary_enabled") or (
                self.point["boundary_enabled"] and not payload["boundary"].get("qualified")):
            raise RuntimeError("Bundle Boundary gate failed")
        self.model = StatsFusionJointEvent(self.config).to(device).eval()
        self.model.load_state_dict(payload["model"])
        self.preprocessing, self.boundary = payload["preprocessing"], payload["boundary"]
        self.identity = payload["identity"]
        mapping = json.loads((root / "source_mapping.json").read_text(encoding="utf-8"))
        self.mapping = FrozenSourceCDF(**mapping)
        if self.mapping.training_subjects != sorted(payload["training_subjects"]):
            raise RuntimeError("Bundle source mapping model lineage differs")

    def new_session(self, subject: str, session: str):
        return JointChunkDetector(self.model, self.config, self.point, self.mapping, self.identity,
                                   self.boundary, subject, session)

    def new_raw_session(self, subject: str, session: str):
        return JointRawStream(self.new_session(subject, session), self.preprocessing, self.config)

    def predict_session(self, raw: RawSessionInput, subject: str | None = None, session: str | None = None,
                        *, new_steps=256):
        subject, session = subject or raw.subject_key, session or raw.session_id
        if subject != raw.subject_key or session != raw.session_id:
            raise ValueError("Prediction subject/session differs from raw input identity")
        detector, events = self.new_session(subject, session), []
        for batch, final in causal_raw_batches(raw, self.preprocessing, self.config, new_steps):
            events.extend(detector.push_batch(batch, final=final))
        self.last_event_diagnostics = events
        return public_events(events)
