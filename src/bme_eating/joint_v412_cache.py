from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_capabilities import capabilities
from bme_eating.joint_protocol import config_identity, digest
from bme_eating.joint_stream import TimelineCache


def feature_binding(model, inputs, config, preprocessing, identity):
    weights = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        weights.update(name.encode())
        weights.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        weights.update(tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
    source = Path(__file__).parent
    paths = sorted(source.rglob("*.py"))
    return {"weights_sha256": weights.hexdigest(), "identity": identity,
            "config_sha256": config_identity(config), "data_sha256": digest(inputs.identity),
            "preprocessing_sha256": digest(preprocessing),
            "feature_source_sha256": digest({path.relative_to(source).as_posix(): sha256_file(path)
                                              for path in paths}),
            "view": "natural", "width": capabilities(config).feature_width,
            "layout": capabilities(config).schema}


class PredictionFeatureCache:
    def __init__(self, path, binding):
        self.path, self.binding, self.cursor = path, binding, 0
        completed = path / "complete.json"
        self.reused = completed.exists()
        if self.reused:
            manifest = json.loads(completed.read_text(encoding="utf-8"))
            if manifest["binding"] != binding:
                raise RuntimeError("Prediction feature cache identity changed")
            for name, checksum in manifest["sha256"].items():
                if sha256_file(path / name) != checksum:
                    raise RuntimeError("Prediction feature cache checksum failed")
            self.cache = TimelineCache(path, binding, binding.get("width", 137), committed_rows=manifest["rows"],
                                       committed_sha256=manifest["cache_sha256"])
            self.times, self.features = self.cache.tensors()
        else:
            self.cache = TimelineCache(path, binding, binding.get("width", 137))

    def read(self, timestamps):
        stop = self.cursor + len(timestamps)
        if stop > len(self.times) or not np.array_equal(self.times[self.cursor:stop].numpy(), timestamps):
            raise RuntimeError("Prediction cache traversal differs from session timestamps")
        times, features = self.times[self.cursor:stop], self.features[self.cursor:stop]
        self.cursor = stop
        return times, features

    def append(self, times, features):
        self.cache.append(times.numpy(), features.numpy())
        self.cursor = self.cache.rows

    def prefix(self):
        times, features = self.cache.tensors()
        return times[:self.cursor], features[:self.cursor]

    def finish(self):
        if self.reused:
            if self.cursor != self.cache.rows:
                raise RuntimeError("Completed prediction cache contains unobserved trailing timestamps")
            return
        write_json_atomic(self.path / "complete.json", {"binding": self.binding, "rows": self.cache.rows,
            "cache_sha256": self.cache.checksum,
            "sha256": {name: sha256_file(self.path / name)
                       for name in ("times.bin", "features.bin", "checksums.jsonl")}})
