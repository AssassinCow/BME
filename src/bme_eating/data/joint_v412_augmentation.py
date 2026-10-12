from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import firwin, lfilter

from bme_eating.data.joint_sessions import JointSessionDataset
from bme_eating.data.stats_fusion_preprocess import build_motion_blocks, build_ppg_blocks
from bme_eating.features.signal import robust_statistics
from bme_eating.joint_capabilities import supported
from bme_eating.joint_protocol import digest
from bme_eating.stats_features import STATS_FEATURE_COLUMNS


def augmentation_description(seed, epoch, subject, session, training):
    training = dict(training)
    if training.get("_stabilized"):
        ramp = min(1.0, max(0.0, (epoch - 2) / 6))
        training["gyro_modality_dropout"] = 0.10 + 0.10 * ramp
        training["ppg_modality_dropout"] = 0.10 + 0.10 * ramp
        training["low_bandwidth_probability"] = 0.20 * ramp
    identity = f"{seed}|{epoch}|{subject}|{session}"
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8]))
    rotations = JointSessionDataset._proper_signed_permutations()
    rotation = rotations[int(rng.integers(len(rotations)))] if rng.random() < training["rotation_augmentation_probability"] else np.eye(3)
    return {"identity": identity, "rotation": rotation.tolist(),
            **({"schedule": {name: training[name] for name in ("gyro_modality_dropout",
                "ppg_modality_dropout", "low_bandwidth_probability")}} if training.get("_stabilized") else {}),
            "gyro_dropout": bool(rng.random() < training["gyro_modality_dropout"]),
            "ppg_dropout": bool(rng.random() < training["ppg_modality_dropout"]),
            "bandwidth_hz": float(rng.choice(training["low_bandwidth_rates"]))
                if rng.random() < training["low_bandwidth_probability"] else None}


def low_bandwidth_view(times, values, mask, rate, origin, sampling_rate=100):
    output = np.zeros_like(values)
    valid = np.zeros_like(mask, dtype=bool)
    taps = firwin(63, min(rate, sampling_rate) * 0.4, fs=sampling_rate)
    for channel in range(values.shape[1]):
        available = mask[:, channel] & np.isfinite(values[:, channel])
        edges = np.diff(np.r_[False, available, False].astype(np.int8))
        for first, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            filtered = lfilter(taps, [1.0], values[first:stop, channel])
            ticks = np.floor((times[first:stop] - origin) * rate / 1000).astype(np.int64)
            updates = np.r_[True, np.diff(ticks) > 0]
            held = np.maximum.accumulate(np.where(updates, np.arange(stop - first), 0))
            output[first:stop, channel] = filtered[held]
            valid[first:stop, channel] = True
    return output, valid


def transform_payload(payload, description, origin):
    result = {name: value.copy() for name, value in payload.items()}
    rotation = np.asarray(description["rotation"], dtype=np.float32)
    values, mask = result["motion_values"], result["motion_mask"].astype(bool)
    for first in (0, 3):
        values[:, first:first + 3] = values[:, first:first + 3] @ rotation.T
        mask[:, first:first + 3] = mask[:, first:first + 3] @ np.abs(rotation).astype(bool).T
    if description["bandwidth_hz"] is not None:
        values, mask = low_bandwidth_view(result["motion_timestamp_ms"], values, mask,
                                          description["bandwidth_hz"], origin)
    if description["gyro_dropout"]:
        mask[:, 3:] = False
    values[~mask] = 0
    result["motion_values"], result["motion_mask"] = values, mask
    if description["ppg_dropout"]:
        result["ppg_values"].fill(0)
        result["ppg_mask"].fill(False)
    elif description["bandwidth_hz"] is not None:
        shape = result["ppg_values"].shape
        values, mask = low_bandwidth_view(result["ppg_timestamp_ms"], result["ppg_values"].reshape(-1, 1),
            result["ppg_mask"].reshape(-1, 1), description["bandwidth_hz"], origin, sampling_rate=50)
        result["ppg_values"], result["ppg_mask"] = values.reshape(shape), mask.reshape(result["ppg_mask"].shape)
    return result


def selected_statistics(payload, end_ms):
    times = payload["motion_timestamp_ms"]
    selection = slice(np.searchsorted(times, end_ms - 15_000), np.searchsorted(times, end_ms, side="right"))
    values, mask = payload["motion_values"][selection], payload["motion_mask"][selection]
    features = {}
    for channel, name in ((0, "acc_x"), (1, "acc_y"), (2, "acc_z"), (4, "gyro_y"), (5, "gyro_z")):
        statistics = robust_statistics(values[mask[:, channel], channel])
        for metric in ("mean", "median", "mad", "zero_crossing_rate"):
            features[f"local_{name}_{metric}"] = statistics[metric]
    for first, name in ((0, "acc_mag"), (3, "gyro_mag")):
        valid = mask[:, first:first + 3].all(1)
        magnitude = np.linalg.norm(values[:, first:first + 3], axis=1)
        statistics = robust_statistics(magnitude[valid])
        for metric in ("std", "iqr"):
            features[f"local_{name}_{metric}"] = statistics[metric]
        adjacent = valid[:-1] & valid[1:]
        jerk = np.diff(magnitude)[adjacent] * 100.0
        features[f"local_{name}_jerk_rms"] = float(np.sqrt(np.mean(jerk ** 2))) if len(jerk) else np.nan
        features[f"local_{name}_jerk_p95"] = float(np.percentile(np.abs(jerk), 95)) if len(jerk) else np.nan
    times = payload["ppg_timestamp_ms"]
    selection = slice(np.searchsorted(times, end_ms - 15_000), np.searchsorted(times, end_ms, side="right"))
    ppg_mask = payload["ppg_mask"][selection]
    features["local_ppg_valid_fraction"] = float(ppg_mask.mean()) if ppg_mask.size else 0.0
    return np.asarray([features[name] for name in STATS_FEATURE_COLUMNS])


class AugmentedReader:
    def __init__(self, reader, dataset):
        self.reader, self.dataset = reader, dataset
        self.payload = None

    def __getattr__(self, name):
        reader = self.__dict__.get("reader")
        if reader is None:
            raise AttributeError(name)
        return getattr(reader, name)

    def read(self, session_id, start_ms, end_ms, *, subject_key):
        payload = self.reader.read(session_id, start_ms - 3_000, end_ms, subject_key=subject_key)
        if self.dataset.training:
            payload = transform_payload(payload, self.dataset.description,
                                        self.dataset.session_origins[(subject_key, session_id)])
        result = {}
        for prefix in ("motion", "ppg"):
            selected = payload[prefix + "_timestamp_ms"] >= start_ms
            for suffix in ("timestamp_ms", "values", "mask"):
                result[prefix + "_" + suffix] = payload[prefix + "_" + suffix][selected]
        self.payload = result
        return result


class JointSessionDatasetV412(JointSessionDataset):
    def configure(self, config, scaler, input_identity):
        self.augmentation_config = {**config["training"], "_stabilized": supported(config, "stabilized")}
        self.scaler = scaler
        self.description = None
        self.statistics_cache = OrderedDict()
        self.cache_bytes = 0
        self.cache_limit = config["training"]["derived_cache_mib"] * 1024 ** 2
        self.cache_identity = digest({"input": input_identity, "scaler": scaler.to_json(),
                                      "normalization": self.normalization.to_json(),
                                      "feature_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        self.reader = AugmentedReader(self.reader, self)

    def __getitem__(self, index):
        row, epoch, _ = index
        subject, session = self.row_location[int(row)]
        self.description = augmentation_description(self.seed, epoch, subject, session, self.augmentation_config)
        result = super().__getitem__(index)
        result["augmentation_identity"] = digest(self.description) if self.training else "natural"
        result["augmentation_description"] = self.description if self.training else {"view": "natural"}
        for name in ("onset", "offset"):
            counts = self.endpoint_counts((subject, session), name) if self.training else (1, 1)
            result[name + "_positive_count"] = torch.tensor(counts[0], dtype=torch.float32)
            result[name + "_background_count"] = torch.tensor(counts[1], dtype=torch.float32)
        if self.augmentation_config.get("_stabilized") and self.training:
            for name, count in self.timeline_counts((subject, session)).items():
                result[name + "_session_count"] = torch.tensor(count, dtype=torch.float32)
        return result

    def timeline_counts(self, key):
        cache = self.__dict__.setdefault("_timeline_counts", {})
        identity = (*key, self.description["gyro_dropout"], self.description["ppg_dropout"])
        if identity not in cache:
            group = self.session_groups[key]
            times = group.timestamp_ms.to_numpy(np.int64)
            sensor = self._session_sensor_valid(key, times)
            weights = group.state_loss_mask.to_numpy(np.float32) * sensor
            counts = {"state": float(weights.sum())}
            for seconds in (3, 15, 60):
                factor = seconds // 3
                total = 0.0
                for end in range(factor - 1, len(times), factor):
                    members = slice(end + 1 - factor, end + 1)
                    if sensor[members].any() and group.state_loss_mask.to_numpy()[members].sum() > 0:
                        total += weights[end]
                counts["proposal_" + str(seconds)] = total
            smooth = group.smooth_loss_mask.to_numpy(np.float32) if "smooth_loss_mask" in group else np.ones(len(group))
            _, _, smooth_target = self._transition_targets(*key, times)
            counts["smooth"] = float((weights[1:] * weights[:-1] * smooth[1:] * smooth_target[1:]).sum())
            cache[identity] = counts
        return cache[identity]

    def _motion_blocks(self, payload, first_timestamp_ms, steps, rng):
        return build_motion_blocks(timestamp_ms=payload["motion_timestamp_ms"], values=payload["motion_values"],
            mask=payload["motion_mask"], normalization=self.normalization, first_timestamp_ms=first_timestamp_ms,
            steps=steps, step_seconds=self.geometry.step_seconds, return_invariant=True)

    def _ppg_blocks(self, payload, timestamps, session_origin_ms, rng):
        return build_ppg_blocks(timestamp_ms=payload["ppg_timestamp_ms"], values=payload["ppg_values"],
            mask=payload["ppg_mask"], normalization=self.normalization, timestamps=timestamps,
            session_origin_ms=session_origin_ms, step_ms=3000, factor=self.geometry.long_pool_factor)

    def _aligned_anchor_arrays(self, group, timestamps):
        valid, arrays = super()._aligned_anchor_arrays(group, timestamps)
        origin = int(group.timestamp_ms.iloc[0]) - 3000
        selected = self.events[(self.events.subject_key.astype(str) == str(group.subject_key.iloc[0]))
                               & (self.events.session_id.astype(str) == str(group.session_id.iloc[0]))]
        for field, mask_field, support in (("start_ms", "start_loss_mask", 30_000),
                                          ("end_ms", "end_loss_mask", 60_000)):
            for endpoint in selected[field].to_numpy(np.int64):
                if endpoint <= origin:
                    arrays[mask_field][timestamps <= origin + support] = 0
        if self.training:
            transformed = []
            identity = (self.cache_identity, digest(self.description))
            for timestamp in timestamps:
                key = (*identity, int(timestamp))
                if key in self.statistics_cache:
                    value = self.statistics_cache.pop(key)
                else:
                    value = self.scaler.transform_array(selected_statistics(self.reader.payload, int(timestamp)))
                    self.cache_bytes += value.nbytes + 256
                self.statistics_cache[key] = value
                transformed.append(value)
            arrays["statistics"] = np.stack(transformed)
            arrays["statistics"][~valid, :12] = 0
            arrays["statistics"][~valid, 12:] = 1
            while self.cache_bytes > self.cache_limit and self.statistics_cache:
                _, value = self.statistics_cache.popitem(last=False)
                self.cache_bytes -= value.nbytes + 256
        return valid, arrays

    def _transition_targets(self, subject_key, session_id, timestamps):
        _, _, smooth = super()._transition_targets(subject_key, session_id, timestamps)
        group = self.session_groups[(subject_key, session_id)]
        origin = self.session_origins[(subject_key, session_id)]
        selected = self.events[(self.events.subject_key.astype(str) == subject_key)
                               & (self.events.session_id.astype(str) == session_id)]
        targets = []
        for field, support in (("start_ms", 30_000), ("end_ms", 60_000)):
            target = np.zeros(len(timestamps), dtype=np.float32)
            for endpoint in selected[field].to_numpy(dtype=np.int64):
                peak = origin + int(np.ceil((endpoint - origin) / 3000)) * 3000
                if endpoint <= origin or endpoint > int(group.timestamp_ms.iloc[-1]):
                    continue
                delta = timestamps - peak
                values = np.where((delta >= 0) & (delta <= support), np.exp(-0.5 * (delta / 9000) ** 2), 0)
                target = np.maximum(target, values)
            targets.append(target.astype(np.float32))
        return *targets, smooth

    def endpoint_counts(self, key, name):
        cache = self.__dict__.setdefault("_endpoint_counts", {})
        cache_key = (*key, self.description["gyro_dropout"], self.description["ppg_dropout"])
        if cache_key not in cache:
            group = self.session_groups[key]
            timestamps = group.timestamp_ms.to_numpy(np.int64)
            targets = self._transition_targets(*key, timestamps)
            sensor_valid = self._session_sensor_valid(key, timestamps)
            cache[cache_key] = {}
            for index, field in enumerate(("start_loss_mask", "end_loss_mask")):
                valid = group[field].to_numpy() > 0 if field in group else np.ones(len(group), bool)
                valid &= sensor_valid
                endpoint_field, support = (("start_ms", 30_000), ("end_ms", 60_000))[index]
                selected = self.events[(self.events.subject_key.astype(str) == key[0])
                                       & (self.events.session_id.astype(str) == key[1])]
                if (selected[endpoint_field] <= self.session_origins[key]).any():
                    valid &= timestamps > self.session_origins[key] + support
                cache[cache_key][("onset", "offset")[index]] = (int(((targets[index] == 1) & valid).sum()),
                    int(((targets[index] < 1) & valid).sum()))
        return cache[cache_key][name]

    def _session_sensor_valid(self, key, timestamps):
        result = np.zeros(len(timestamps), dtype=bool)
        origin = self.session_origins[key]
        for first in range(0, len(timestamps), 256):
            times = timestamps[first:first + 256]
            payload = self.reader.reader.read(key[1], int(times[0]) - 30_000,
                                              int(times[-1]), subject_key=key[0])
            motion_mask = payload["motion_mask"].copy()
            if self.description["gyro_dropout"]:
                motion_mask[:, 3:] = False
            for prefix, available, starts, ends in (
                ("motion", motion_mask.any(1), times - 3000, times),
                ("ppg", payload["ppg_mask"].reshape(-1) & (not self.description["ppg_dropout"]),
                 origin + ((times - origin) // 15_000) * 15_000 - 15_000,
                 origin + ((times - origin) // 15_000) * 15_000)):
                cumulative = np.r_[0, np.cumsum(available)]
                observed = payload[prefix + "_timestamp_ms"]
                left, right = np.searchsorted(observed, starts), np.searchsorted(observed, ends)
                result[first:first + len(times)] |= cumulative[right] > cumulative[left]
        return result
