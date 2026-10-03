from __future__ import annotations

import numpy as np
import torch

from bme_eating.data.stats_fusion_sequence import StatsFusionSequenceDataset


class V48ProposalSequenceDataset(StatsFusionSequenceDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.anchors["proposal_target"] = 0.0
        self.anchors["proposal_weight"] = 0.0
        step_ms = self.geometry.step_seconds * 1000
        for (subject_key, session_id), group in self.anchors.groupby(
            ["subject_key", "session_id"], sort=False
        ):
            session_events = self.events.loc[
                self.events["subject_key"].astype(str).eq(str(subject_key))
                & self.events["session_id"].astype(str).eq(str(session_id))
            ]
            timestamps = group["timestamp_ms"].to_numpy(dtype=np.int64)
            eligible = group["state_loss_mask"].to_numpy(dtype=float) > 0
            for event in session_events.itertuples(index=False):
                hits = (
                    (timestamps > int(event.start_ms))
                    & (timestamps - 2 * step_ms < int(event.end_ms))
                    & eligible
                )
                if hits.any():
                    self.anchors.loc[group.index[hits], "proposal_target"] = 1.0
                    self.anchors.loc[group.index[hits], "proposal_weight"] += 1.0 / hits.sum()
        for _, group in self.anchors.groupby("subject_key", sort=False):
            background = (
                group["proposal_target"].to_numpy(dtype=float) == 0
            ) & (group["state_loss_mask"].to_numpy(dtype=float) > 0)
            if background.any():
                self.anchors.loc[group.index[background], "proposal_weight"] = (
                    1.0 / background.sum()
                )
        self.session_groups = {
            (str(subject), str(session)): group.sort_values("timestamp_ms").reset_index(drop=True)
            for (subject, session), group in self.anchors.groupby(
                ["subject_key", "session_id"], sort=False
            )
        }

    def __getitem__(self, index):
        result = super().__getitem__(index)
        key = (str(result["subject_key"]), str(result["session_id"]))
        group = self.session_groups[key]
        source = group["timestamp_ms"].to_numpy(dtype=np.int64)
        timestamps = result["timestamp_ms"].numpy()
        positions = self._nearest_indices(
            source, timestamps, self.geometry.step_seconds * 500
        )
        valid = positions >= 0
        for name in ("proposal_target", "proposal_weight"):
            output = np.zeros(len(timestamps), dtype=np.float32)
            output[valid] = group[name].to_numpy(dtype=np.float32)[positions[valid]]
            result[name] = torch.from_numpy(output)
        return result
