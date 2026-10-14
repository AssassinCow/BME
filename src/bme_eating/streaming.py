from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch

from bme_eating.candidates import (
    ImmutableCommitter,
    OnlineBudget,
)
from bme_eating.data.deep_dataset import Normalization
from bme_eating.data.joint_sessions import joint_geometry
from bme_eating.data.stats_fusion_preprocess import (
    RawSessionInput,
    StatsFusionRawSessionPreprocessor,
)
from bme_eating.evaluation import NonnegativePlatt
from bme_eating.protocol import (
    capabilities,
    configure_numerics,
)
from bme_eating.stats_features import FoldRobustScaler
from bme_eating.training.statsfusion_joint_trainer import (
    decoder_settings,
    pack_features,
    refine_candidate,
)
from bme_eating.types import Event


class JointChunkDetector:
    def __init__(self, model, config, point, mapping, identity, boundary, subject, session):
        configure_numerics(config)
        self.model, self.config, self.point = model.eval(), config, point
        self.mapping, self.identity, self.boundary = mapping, identity, boundary
        self.device = next(model.parameters()).device
        from bme_eating.proposal_stream import make_stream
        self.stream = make_stream(subject, session, identity, decoder_settings(config)[point["setting"]])
        self.budget = OnlineBudget(point["budget"], config["decoder"]["quotas"])
        self.committer = ImmutableCommitter(point["acceptance"], point["nms"])
        self.calibrator = NonnegativePlatt(**point["calibration"])
        self.times = torch.empty(0, dtype=torch.int64)
        self.features = torch.empty(0, capabilities(self.model).feature_width)
        self.finished = False
        from bme_eating.feature_cache import BlockedHistory
        self.history = BlockedHistory()

    @torch.no_grad()
    def push_features(self, times: torch.Tensor, features: torch.Tensor, *, final: bool = False):
        if self.finished:
            raise RuntimeError("Finished session cannot accept future chunks")
        if (times.ndim != 1 or times.dtype != torch.int64 or features.shape != (len(times), capabilities(self.model).feature_width)
                or not torch.isfinite(features).all() or (torch.diff(times) != 3000).any()
                or ((features[:, 128:137] < 0) | (features[:, 128:137] > 1)).any()):
            raise ValueError("Invalid streaming timeline, feature shape, finite values or masks")
        if len(self.times) and len(times) and int(times[0]) != int(self.times[-1]) + 3000:
            raise ValueError("Feature chunks overlap, move backwards or skip missing bins")
        self.history.append(times, features)
        self.times, self.features = self.history.times, self.history.features
        emitted = []
        available_time = int(times[-1]) if len(times) else int(self.times[-1]) if len(self.times) else None
        for index, (timestamp, values) in enumerate(zip(times.tolist(), features.cpu().numpy())):
            self.budget.observe(3, float(values[128]))
            probabilities = {"state": float(values[129]),
                             **{f"proposal_{scale}": float(values[130 + index])
                                for index, scale in enumerate((3, 15, 60))}}
            arguments = (int(timestamp), probabilities, float(values[128]), float(values[134]))
            candidates = (self.stream.observe(*arguments, onset_probability=float(values[133])))
            emitted.extend(self._decide(candidates))
            if final and index == len(times) - 1:
                emitted.extend(self._decide(self.stream.finish()))
        if final:
            if not len(times):
                emitted.extend(self._decide(self.stream.finish()))
            self.finished = True
        for candidate in emitted:
            candidate.input_available_time_ms = available_time
            candidate.emitted_at_ms = available_time
        return [candidate.to_dict() for candidate in emitted]

    def _decide(self, candidates):
        chosen = self.budget.select(candidates, self.mapping) if candidates else []
        from bme_eating.candidate_scoring import score_candidates
        score_candidates(self.model, chosen, self.times, self.features, self.device, batch_size=128)
        for index, candidate in enumerate(chosen):
            if self.point["boundary_enabled"]:
                candidate = refine_candidate(self.model, candidate, self.times, self.features, self.device,
                                               {**self.boundary, "mode": self.point.get("boundary_mode", "both")}, self.point["boundary_entropy"])
                chosen[index] = candidate
            candidate.event_probability = float(self.calibrator.transform(candidate.event_logit))
            candidate.rank_score = candidate.event_probability
            candidate.acceptance_score, candidate.quality_score = candidate.rank_score, candidate.predicted_quality
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
                                                       torch.empty(0, capabilities(self.detector.model).feature_width), final=True))
        available_time = max((int(values[-1]) for values in
            (self.raw.motion_timestamp_ms, self.raw.ppg_timestamp_ms) if len(values)), default=None)
        for candidate in emitted:
            candidate["input_available_time_ms"] = available_time
            candidate["emitted_at_ms"] = available_time
        self.diagnostics.extend(emitted)
        return public_events(emitted)


class JointDetector:

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
        available_time = max((int(values[-1]) for values in
            (raw.motion_timestamp_ms, raw.ppg_timestamp_ms) if len(values)), default=None)
        for candidate in events:
            candidate["input_available_time_ms"] = available_time
            candidate["emitted_at_ms"] = available_time
        self.last_event_diagnostics = events
        return public_events(events)
