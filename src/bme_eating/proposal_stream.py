from __future__ import annotations

import copy

from bme_eating.candidates import StreamingProposals, candidate_order_key, stable_id
from bme_eating.endpoint_peaks import CausalOnsetPeaks
from bme_eating.protocol import PROFILE


class LearnedStreamingProposals(StreamingProposals):
    def __init__(self, *arguments):
        super().__init__(*arguments)
        self.peak_detector = CausalOnsetPeaks(self.config["onset_peak_threshold"])
        self.onset_peaks = []

    def observe(self, timestamp, probabilities, valid, offset_probability=0.0,
                onset_probability=0.0):
        import math
        if not math.isfinite(onset_probability) or not 0 <= onset_probability <= 1:
            raise ValueError("Invalid onset probability")
        peak = self.peak_detector.observe(timestamp, onset_probability, valid)
        if peak is not None:
            self.onset_peaks.append(peak[:2])
        return super().observe(timestamp, probabilities, valid, offset_probability)

    def _ready(self, decision, eof=False):
        legacy = super()._ready(decision, eof)
        if self.config.get("parent_proposals_only"):
            for candidate in legacy:
                candidate.score_formula = self.config.get("score_formula", PROFILE.score_formula)
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
                protected_geometries = set()
            else:
                fallback = [coarse]
                geometries = {(coarse.start_ms, coarse.end_ms)}
                def fallback_priority(item):
                    parent = item.generation_parameters
                    return stable_id(2026, item.subject_key, item.session_id,
                        parent.get("parent_source", item.source), item.scale_seconds,
                        item.coarse_start_ms, item.coarse_end_ms, item.start_ms, item.end_ms), candidate_order_key(item)
                for item in sorted(pool, key=fallback_priority):
                    geometry = item.start_ms, item.end_ms
                    if geometry not in geometries:
                        fallback.append(item)
                        geometries.add(geometry)
                    if len(fallback) == 3:
                        break
                variants = list(fallback)
                protected_geometries = {(item.start_ms, item.end_ms) for item in fallback[1:]}
                if selected:
                    geometries = {(item.start_ms, item.end_ms) for item in variants}
                    for timestamp, probability in selected:
                        version = copy.deepcopy(coarse)
                        version.start_ms, version.source = timestamp, "learned_onset"
                        version.proposal_id = stable_id(coarse.ancestor_id, timestamp, coarse.end_ms,
                                                        "learned_onset")
                        version.generation_parameters.update({"onset_peak_probability": probability})
                        variants.append(version)
                        geometry = version.start_ms, version.end_ms
                        if geometry not in geometries:
                            geometries.add(geometry)
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
                candidate.generation_parameters.update({PROFILE.schema: True,
                    "schema": PROFILE.schema, "protocol": PROFILE.protocol, "repair_revision": PROFILE.repair_revision,
                    "source_flags": sorted(set(flags)), "parent_raw_score": coarse.raw_score,
                    "learned_peak_count": len(selected), "compatibility_fallback": not bool(selected),
                    "compatibility_guard": geometry in protected_geometries,
                    "candidate_version_policy": "fallback_three_plus_one_learned_v1",
                    "coarse_parent_id": coarse.ancestor_id})
                candidate.score_formula = self.config.get("score_formula", PROFILE.score_formula)
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
