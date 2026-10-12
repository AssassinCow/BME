from __future__ import annotations

import copy
from collections import deque

from bme_eating.joint_stream import StreamingProposals, stable_id


class LearnedStreamingProposals(StreamingProposals):
    def __init__(self, *arguments):
        super().__init__(*arguments)
        self.onset_samples = deque()
        self.onset_peaks = []

    def observe(self, timestamp, probabilities, valid, offset_probability=0.0,
                onset_probability=0.0):
        import math
        if not math.isfinite(onset_probability) or not 0 <= onset_probability <= 1:
            raise ValueError("Invalid onset probability")
        self.onset_samples.append((timestamp, onset_probability, valid))
        if len(self.onset_samples) >= 7:
            samples = list(self.onset_samples)[-7:]
            center = samples[3]
            if (center[2] > 0 and center[1] >= self.config["onset_peak_threshold"]
                    and samples[-1][0] - center[0] == 9000
                    and center[0] - samples[0][0] == 9000
                    and all(center[1] > value[1] for value in samples[:3])
                    and all(center[1] >= value[1] for value in samples[4:])):
                self.onset_peaks.append((center[0], center[1]))
        while len(self.onset_samples) > 7:
            self.onset_samples.popleft()
        return super().observe(timestamp, probabilities, valid, offset_probability)

    def _ready(self, decision, eof=False):
        legacy = super()._ready(decision, eof)
        if self.config.get("parent_proposals_only"):
            for candidate in legacy:
                candidate.score_formula = self.config.get("score_formula", "calibrated_event_probability_v411")
            return legacy
        parents = {}
        for candidate in legacy:
            parents.setdefault(candidate.ancestor_id, []).append(candidate)
        result = []
        for pool in parents.values():
            coarse = next(candidate for candidate in pool if candidate.source in {"state", "proposal"})
            peaks = sorted((peak for peak in self.onset_peaks
                            if abs(peak[0] - coarse.coarse_start_ms) <= 900_000
                            and int(self.origin) <= peak[0] < coarse.end_ms),
                           key=lambda peak: (-peak[1], peak[0]))
            selected = []
            for peak in peaks:
                if all(abs(peak[0] - other[0]) >= 15_000 for other in selected):
                    selected.append(peak)
                if len(selected) == 2:
                    break
            if self.config.get("ablation") == "no_onset":
                variants = pool
            elif selected:
                variants = [coarse]
                for timestamp, probability in selected:
                    version = copy.deepcopy(coarse)
                    version.start_ms, version.source = timestamp, "learned_onset"
                    version.proposal_id = stable_id(coarse.ancestor_id, timestamp, coarse.end_ms,
                                                    "learned_onset")
                    version.generation_parameters.update({"onset_peak_probability": probability})
                    variants.append(version)
            else:
                variants = [coarse]
                geometries = {(coarse.start_ms, coarse.end_ms)}
                for item in sorted(pool, key=lambda item: item.proposal_id):
                    geometry = item.start_ms, item.end_ms
                    if geometry not in geometries:
                        variants.append(item)
                        geometries.add(geometry)
                    if len(variants) == 3:
                        break
            unique = {}
            for candidate in variants:
                geometry = candidate.start_ms, candidate.end_ms
                flags = [candidate.source, candidate.generation_parameters["parent_source"]]
                if geometry in unique:
                    previous = unique[geometry]
                    previous.generation_parameters["source_flags"] = sorted(set(
                        previous.generation_parameters["source_flags"] + flags))
                    continue
                candidate.generation_parameters.update({"v411": True,
                    "source_flags": sorted(set(flags)), "parent_raw_score": coarse.raw_score,
                    "learned_peak_count": len(selected), "compatibility_fallback": not bool(selected),
                    "coarse_parent_id": coarse.ancestor_id})
                candidate.score_formula = self.config.get("score_formula", "calibrated_event_probability_v411")
                candidate.validate()
                unique[geometry] = candidate
            for item in pool:
                geometry = item.start_ms, item.end_ms
                if geometry in unique:
                    kept = unique[geometry]
                    kept.generation_parameters["source_flags"] = sorted(set(
                        kept.generation_parameters["source_flags"] + [item.source]))
            result.extend(unique.values())
        return result


def make_stream(subject, session, identity, decoder):
    if decoder.get("learned_localization"):
        return LearnedStreamingProposals(subject, session, identity, decoder)
    return StreamingProposals(subject, session, identity, decoder)
