from __future__ import annotations

from collections import OrderedDict

import torch


class FeatureBlocks:
    def __init__(self, blocks, width=None):
        self.blocks = [block for block in blocks if len(block)]
        width = width if width is not None else blocks[0].shape[1] if blocks else 137
        if any(block.shape[1] != width for block in self.blocks):
            raise ValueError("History blocks use incompatible feature layouts")
        self.shape = (sum(len(block) for block in self.blocks), width)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, selection):
        rows, columns = selection if isinstance(selection, tuple) else (selection, slice(None))
        outputs, offset = [], 0
        if isinstance(rows, torch.Tensor) and rows.dtype == torch.bool:
            for block in self.blocks:
                outputs.append(block[rows[offset:offset + len(block)], columns])
                offset += len(block)
        elif isinstance(rows, slice):
            start, stop, step = rows.indices(len(self))
            if step != 1:
                raise ValueError("Blocked history requires contiguous indexed slices")
            for block in self.blocks:
                first, last = max(0, start - offset), min(len(block), stop - offset)
                if first < last:
                    outputs.append(block[first:last, columns])
                offset += len(block)
        else:
            raise TypeError("Blocked history requires slice or Boolean mask")
        if outputs:
            return torch.cat(outputs)
        width = torch.empty(0, self.shape[1])[:, columns].shape
        return torch.empty(width)


class BlockedHistory:
    def __init__(self):
        self.time_blocks, self.feature_blocks = [], []

    def append(self, times, features):
        self.time_blocks.append(times.cpu())
        self.feature_blocks.append(features.cpu())

    @property
    def times(self):
        return torch.cat(self.time_blocks) if self.time_blocks else torch.empty(0, dtype=torch.int64)

    @property
    def features(self):
        return FeatureBlocks(self.feature_blocks)


class ExactROICache:
    def __init__(self, binding, maximum=1024):
        self.binding, self.maximum = binding, maximum
        self.values = OrderedDict()
        self.hits, self.misses = 0, 0

    def key(self, candidate):
        return (self.binding, candidate.model_identity, candidate.subject_key, candidate.session_id,
                candidate.start_ms, candidate.end_ms, candidate.decision_time_ms,
                candidate.max_input_time_ms)

    def get(self, candidate):
        key = self.key(candidate)
        if key not in self.values:
            self.misses += 1
            return None
        self.hits += 1
        self.values.move_to_end(key)
        return self.values[key]

    def put(self, candidate, roi):
        if roi.requires_grad:
            raise RuntimeError("Training ROI cannot be cached across backward graphs")
        self.values[self.key(candidate)] = roi
        while len(self.values) > self.maximum:
            self.values.popitem(last=False)
