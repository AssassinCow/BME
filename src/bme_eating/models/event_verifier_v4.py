from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, Sampler

from bme_eating.proposals_v4 import proposal_source_family

CATEGORY_ORDER = (
    "positive",
    "near_miss",
    "hard_false_positive",
    "random_background",
)


@dataclass(frozen=True)
class ProposalFeatureBatchV4:
    proposal_ids: np.ndarray
    sequence: np.ndarray
    sequence_mask: np.ndarray
    scalar: np.ndarray
    event_target: np.ndarray | None = None
    iou_target: np.ndarray | None = None
    sample_weight: np.ndarray | None = None
    raw_imu: np.ndarray | None = None
    sampling_metadata: dict[str, np.ndarray] | None = None


def pooled_logistic_features(features: ProposalFeatureBatchV4) -> np.ndarray:
    sequence = np.asarray(features.sequence)
    mask = np.asarray(features.sequence_mask, dtype=bool)
    scalar = np.asarray(features.scalar)
    if sequence.ndim != 3 or mask.shape != sequence.shape[:2]:
        raise ValueError("Pooled Logistic sequence and mask shapes are incompatible")
    if scalar.ndim != 2 or scalar.shape[0] != sequence.shape[0]:
        raise ValueError("Pooled Logistic scalar features are not aligned")
    expanded_mask = mask[..., None]
    count = expanded_mask.sum(axis=1).clip(min=1)
    mean = np.where(expanded_mask, sequence, 0.0).sum(axis=1) / count
    maximum = np.where(expanded_mask, sequence, -np.inf).max(axis=1)
    maximum[~np.isfinite(maximum)] = 0.0
    matrix = np.concatenate((mean, maximum, scalar), axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("Pooled Logistic features must be finite on valid bins and scalars")
    return matrix


class ProposalDatasetV4(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(self, features: ProposalFeatureBatchV4) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features.proposal_ids)

    def __getitem__(self, index: int | tuple[int, float]) -> dict[str, torch.Tensor | str]:
        if isinstance(index, tuple):
            row_index, sampling_probability = index
        else:
            row_index, sampling_probability = index, 1.0
        index = int(row_index)
        output: dict[str, torch.Tensor | str] = {
            "proposal_id": str(self.features.proposal_ids[index]),
            "sequence": torch.from_numpy(self.features.sequence[index]),
            "sequence_mask": torch.from_numpy(self.features.sequence_mask[index]),
            "scalar": torch.from_numpy(self.features.scalar[index]),
        }
        if self.features.raw_imu is not None:
            output["raw_imu"] = torch.from_numpy(self.features.raw_imu[index].astype(np.float32))
        for name in ("event_target", "iou_target", "sample_weight"):
            values = getattr(self.features, name)
            if values is not None:
                output[name] = torch.tensor(float(values[index]), dtype=torch.float32)
        if self.features.sample_weight is not None:
            probability = max(float(sampling_probability), np.finfo(np.float32).tiny)
            output["sampling_probability"] = torch.tensor(probability, dtype=torch.float32)
            output["importance_weight"] = torch.tensor(
                float(self.features.sample_weight[index]) / probability,
                dtype=torch.float32,
            )
        return output


class HardNegativeBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        categories: np.ndarray,
        *,
        batch_size: int,
        ratios: dict[str, float],
        steps_per_epoch: int,
        seed: int,
        target_weights: np.ndarray | None = None,
        sampling_metadata: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.categories = np.asarray(categories, dtype=str)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        self.sampling_metadata = self._validate_metadata(sampling_metadata)
        if set(ratios) != set(CATEGORY_ORDER):
            raise ValueError("Verifier ratios must define all four categories")
        weights = np.asarray([ratios[name] for name in CATEGORY_ORDER], dtype=np.float64)
        if (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Verifier ratios must be non-negative and sum to one")
        raw = weights * self.batch_size
        counts = np.floor(raw).astype(int)
        for index in np.argsort(-(raw - counts), kind="stable")[: self.batch_size - counts.sum()]:
            counts[index] += 1
        pools = {name: np.flatnonzero(self.categories == name) for name in CATEGORY_ORDER}
        if not any(len(value) for value in pools.values()):
            raise ValueError("Verifier sampler received no proposals")
        redistribution = ("near_miss", "hard_false_positive", "random_background")
        for index, name in enumerate(CATEGORY_ORDER):
            if len(pools[name]) or counts[index] == 0:
                continue
            missing = int(counts[index])
            counts[index] = 0
            available = [value for value in redistribution if len(pools[value]) and value != name]
            if not available:
                available = [value for value in CATEGORY_ORDER if len(pools[value])]
            counts[CATEGORY_ORDER.index(available[0])] += missing
        self.pools = pools
        self.counts = {name: int(counts[index]) for index, name in enumerate(CATEGORY_ORDER)}
        target = (
            np.ones(len(self.categories), dtype=np.float64)
            if target_weights is None
            else np.asarray(target_weights, dtype=np.float64)
        )
        if (
            target.shape != self.categories.shape
            or not np.isfinite(target).all()
            or np.any(target < 0)
        ):
            raise ValueError("Verifier target weights must be finite, non-negative, and aligned")
        self.within_category_probability: dict[str, np.ndarray] = {}
        for name, pool in pools.items():
            if not len(pool):
                self.within_category_probability[name] = np.empty(0, dtype=np.float64)
                continue
            selected = target[pool]
            total = selected.sum()
            self.within_category_probability[name] = (
                selected / total if total > 0 else np.full(len(pool), 1.0 / len(pool))
            )

        self._strata = self._build_strata()

    def _validate_metadata(
        self, metadata: dict[str, np.ndarray] | None
    ) -> dict[str, np.ndarray] | None:
        if metadata is None:
            return None
        normalized: dict[str, np.ndarray] = {}
        for name, values in metadata.items():
            array = np.asarray(values)
            if array.shape != self.categories.shape:
                raise ValueError(f"Verifier sampling metadata {name!r} is not aligned")
            normalized[name] = array.astype(str)
        required = {"subject_key", "hand_relation", "iou_bin", "gyro_missingness_bin",
                    "duration_bin", "source_mask", "proposal_family_id", "event_group"}
        missing = required - set(normalized)
        if missing:
            raise ValueError(f"Verifier sampling metadata is missing: {sorted(missing)}")
        normalized.setdefault("wear_hand", np.full(len(self.categories), "unknown", dtype=str))
        normalized.setdefault("source_family", np.asarray([
            proposal_source_family(int(value)) for value in normalized["source_mask"]
        ]))
        ownership = pd.DataFrame({"family": normalized["proposal_family_id"], "subject": normalized["subject_key"],
                                  "session": normalized.get("session_id", normalized["subject_key"])})
        if ownership.groupby("family")[["subject", "session"]].nunique().gt(1).any().any():
            raise ValueError("Verifier proposal family crosses subject boundaries")
        return normalized

    def _build_strata(self) -> dict[str, list[tuple[str, str, np.ndarray]]]:
        if self.sampling_metadata is None:
            return {name: [("all", str(index), np.asarray([index])) for index in pool] for name, pool in self.pools.items()}
        metadata = self.sampling_metadata
        key_columns = ("subject_key", "wear_hand", "hand_relation", "iou_bin", "gyro_missingness_bin",
                       "duration_bin", "source_family")
        strata: dict[str, list[tuple[str, str, np.ndarray]]] = {}
        for name, pool in self.pools.items():
            if not len(pool):
                strata[name] = []
                continue
            frame = pd.DataFrame({column: metadata[column][pool] for column in key_columns})
            groups: list[tuple[str, str, np.ndarray]] = []
            grouping_column = "event_group" if name == "positive" else "proposal_family_id"
            frame["group"] = metadata[grouping_column][pool]
            for group_key, grouped in frame.groupby(["subject_key", "group"], sort=True):
                representative = grouped.sort_values(list(key_columns), kind="stable").iloc[0]
                groups.append((
                    "|".join(str(representative[column]) for column in key_columns),
                    "|".join(map(str, group_key)), pool[grouped.index.to_numpy()],
                ))
            strata[name] = groups
        return strata

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        for _ in range(self.steps_per_epoch):
            parts = []
            used_groups: set[str] = set()
            for name, count in self.counts.items():
                if not count:
                    continue
                strata = self._strata[name]
                if not strata:
                    continue
                selected_rows: list[int] = []
                selected_probabilities: list[float] = []
                for _position in range(count):
                    if self.sampling_metadata is None:
                        pool = self.pools[name]
                        row = int(rng.choice(pool, p=self.within_category_probability[name]))
                        probability = self.within_category_probability[name][int(np.searchsorted(pool, row))]
                        selected_rows.append(row)
                        selected_probabilities.append((count / self.batch_size) * probability)
                        continue
                    candidates = [
                        index for index, group in enumerate(strata) if group[1] not in used_groups
                    ]
                    if not candidates:
                        candidates = list(range(len(strata)))
                    probabilities = np.ones(len(candidates), dtype=float)
                    paths = [strata[index][0].split("|") for index in candidates]
                    branching: dict[tuple[str, ...], set[str]] = defaultdict(set)
                    for path in paths:
                        for depth in range(len(path)):
                            branching[tuple(path[:depth])].add(path[depth])
                    for depth in range(len(paths[0])):
                        for candidate_index, path in enumerate(paths):
                            probabilities[candidate_index] /= len(branching[tuple(path[:depth])])
                    leaves = Counter(tuple(path) for path in paths)
                    for candidate_index, path in enumerate(paths):
                        probabilities[candidate_index] /= leaves[tuple(path)]
                    probabilities /= probabilities.sum()
                    choice = int(rng.choice(len(candidates), p=probabilities))
                    stratum_index = candidates[choice]
                    _, group_name, stratum_rows = strata[stratum_index]
                    row = int(rng.choice(stratum_rows))
                    selected_rows.append(row)
                    used_groups.add(group_name)
                    selected_probabilities.append(
                        (count / self.batch_size)
                        * float(probabilities[choice]) / len(stratum_rows)
                    )
                parts.append(np.column_stack((selected_rows, selected_probabilities)))
            if not parts:
                continue
            batch = np.concatenate(parts)
            rng.shuffle(batch)
            yield [(int(row[0]), float(row[1])) for row in batch]


class DepthwiseVerifierBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(1).to(values.dtype)
        masked = values * weights
        update = self.pointwise(self.depthwise(masked)).transpose(1, 2)
        update = self.dropout(torch.nn.functional.silu(self.norm(update))).transpose(1, 2)
        return (masked + update * weights) * weights


class EventVerifierV4(nn.Module):
    def __init__(self, sequence_dim: int, scalar_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        channels = int(config.get("hidden_channels", 64))
        hidden = int(config.get("hidden_dim", 128))
        dropout = float(config.get("dropout", 0.1))
        self.use_learned_query_pooling = bool(config.get("use_learned_query_pooling", False))
        attention_heads = int(config.get("attention_heads", 4))
        if self.use_learned_query_pooling:
            if attention_heads <= 0 or channels % attention_heads:
                raise ValueError("attention_heads must divide hidden_channels")
            self.learned_query = nn.Parameter(torch.zeros(1, 1, channels))
            self.query_attention = nn.MultiheadAttention(
                channels,
                attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            nn.init.normal_(self.learned_query, mean=0.0, std=0.02)
        self.input_projection = nn.Conv1d(sequence_dim, channels, kernel_size=1, bias=False)
        self.blocks = nn.ModuleList(
            [
                DepthwiseVerifierBlock(channels, dropout),
                DepthwiseVerifierBlock(channels, dropout),
            ]
        )
        pooled_dim = (3 if self.use_learned_query_pooling else 2) * channels
        self.projection = nn.Sequential(
            nn.Linear(pooled_dim + scalar_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.SiLU(),
        )
        self.event_head = nn.Linear(64, 1)
        self.iou_head = nn.Linear(64, 1)
        self.use_raw_imu_branch = bool(config.get("use_raw_imu_branch", False))
        if self.use_raw_imu_branch:
            self.use_fold_raw_normalization = config.get("raw_imu_normalization") == "training_fold_robust_v1"
            if self.use_fold_raw_normalization:
                self.register_buffer("raw_imu_center", torch.zeros(6))
                self.register_buffer("raw_imu_scale", torch.ones(6))
            self.raw_imu_encoder = nn.Sequential(
                nn.Conv1d(12, 16, kernel_size=7, stride=2, padding=3),
                nn.SiLU(),
                nn.Conv1d(16, 16, kernel_size=5, stride=2, padding=2, groups=16),
                nn.Conv1d(16, 16, kernel_size=1),
                nn.SiLU(),
            )
            self.raw_imu_projection = nn.Linear(3 * 32, 64)
            self.raw_imu_gate = nn.Parameter(torch.tensor(-3.0))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = batch["sequence_mask"].bool()
        values = torch.where(
            mask.unsqueeze(-1),
            torch.nan_to_num(batch["sequence"]),
            torch.zeros_like(batch["sequence"]),
        )
        sequence = self.input_projection(values.transpose(1, 2))
        for block in self.blocks:
            sequence = block(sequence, mask)
        weights = mask.unsqueeze(1).to(sequence.dtype)
        mean = (sequence * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
        maximum = sequence.masked_fill(~mask.unsqueeze(1), -torch.inf).amax(dim=-1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        pooled = [mean, maximum]
        if self.use_learned_query_pooling:
            sequence_tokens = sequence.transpose(1, 2)
            valid_rows = mask.any(dim=-1)
            attention_pool = torch.zeros_like(mean)
            if valid_rows.any():
                query = self.learned_query.expand(int(valid_rows.sum()), -1, -1)
                attended, _ = self.query_attention(
                    query,
                    sequence_tokens[valid_rows],
                    sequence_tokens[valid_rows],
                    key_padding_mask=~mask[valid_rows],
                    need_weights=False,
                )
                attention_pool[valid_rows] = attended[:, 0, :]
            pooled.append(attention_pool)
        scalar = torch.nan_to_num(batch["scalar"])
        hidden = self.projection(torch.cat((*pooled, scalar), dim=-1))
        if self.use_raw_imu_branch:
            raw_imu = batch.get("raw_imu")
            if raw_imu is None or raw_imu.ndim != 4 or raw_imu.shape[1:3] != (3, 12):
                raise ValueError("Raw IMU verifier requires three aligned 12-channel snippets")
            raw_imu = torch.nan_to_num(raw_imu)
            raw_mask = raw_imu[:, :, 6:, :].clamp(0, 1)
            raw_values = raw_imu[:, :, :6, :]
            if self.use_fold_raw_normalization:
                raw_values = (raw_values - self.raw_imu_center[None, None, :, None]) / self.raw_imu_scale[None, None, :, None]
                raw_values = raw_values.clamp(-10, 10)
            raw_imu = torch.cat((torch.where(raw_mask.bool(), raw_values, 0.0), raw_mask), dim=2)
            raw_validity = raw_imu[:, :, 6:, :].mean(dim=(1, 2, 3)).clamp(0, 1)
            encoded = self.raw_imu_encoder(
                raw_imu.reshape(-1, 12, raw_imu.shape[-1])
            )
            raw_summary = torch.cat((encoded.mean(dim=-1), encoded.amax(dim=-1)), dim=-1)
            raw_summary = raw_summary.reshape(raw_imu.shape[0], -1)
            hidden = hidden + (
                torch.sigmoid(self.raw_imu_gate)
                * raw_validity.unsqueeze(-1)
                * self.raw_imu_projection(raw_summary)
            )
        event_logit = self.event_head(hidden).squeeze(-1)
        iou_logit = self.iou_head(hidden).squeeze(-1)
        return {
            "event_logit": event_logit,
            "iou_logit": iou_logit,
            "predicted_iou": torch.sigmoid(iou_logit),
        }


def verifier_loss_v4(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], iou_weight: float = 0.5
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weight = batch.get(
        "importance_weight",
        batch.get("sample_weight", torch.ones_like(batch["event_target"])),
    )
    event_element = nn.functional.binary_cross_entropy_with_logits(
        output["event_logit"], batch["event_target"], reduction="none"
    )
    iou_element = nn.functional.smooth_l1_loss(
        output["predicted_iou"], batch["iou_target"], reduction="none"
    )
    denominator = weight.sum().clamp_min(1.0)
    event = (event_element * weight).sum() / denominator
    quality = (iou_element * weight).sum() / denominator
    return event + float(iou_weight) * quality, {"event": event, "quality": quality}


def classify_proposals(frame: pd.DataFrame) -> np.ndarray:
    required = {"max_iou", "generator_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Proposal labels are missing columns: {sorted(missing)}")
    iou = frame["max_iou"].to_numpy(dtype=np.float64)
    score = frame["generator_score"].to_numpy(dtype=np.float64)
    hard_threshold = float(np.quantile(score[iou < 0.10], 0.75)) if np.any(iou < 0.10) else np.inf
    historical = frame.get("historical_hard_false_positive", False)
    historical = np.asarray(historical, dtype=bool)
    category = np.full(len(frame), "random_background", dtype=object)
    category[(iou >= 0.10) & (iou <= 0.25)] = "near_miss"
    category[(iou < 0.10) & ((score >= hard_threshold) | historical)] = "hard_false_positive"
    category[iou > 0.25] = "positive"
    return category.astype(str)


def normalized_proposal_weights(frame: pd.DataFrame) -> np.ndarray:
    required = {"subject_key", "proposal_id", "max_iou"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Proposal weights are missing columns: {sorted(missing)}")
    if frame.empty:
        return np.empty(0, dtype=np.float32)
    positive = frame["max_iou"].to_numpy(dtype=np.float64) > 0.25
    matched = frame.get("matched_event_id", pd.Series("", index=frame.index)).astype(str)
    session = frame.get("session_id", pd.Series("", index=frame.index)).astype(str)
    family = frame.get("proposal_family_id", frame["proposal_id"]).astype(str)
    group = np.where(positive, "event:" + session + ":" + matched, "family:" + family)
    work = pd.DataFrame(
        {
            "subject_key": frame["subject_key"].astype(str).to_numpy(),
            "group": group,
            "row": np.arange(len(frame), dtype=np.int64),
        }
    )
    weights = np.zeros(len(frame), dtype=np.float64)
    for _, subject in work.groupby("subject_key", sort=False):
        subject_weight = np.zeros(len(subject), dtype=np.float64)
        for _, grouped in subject.groupby("group", sort=False):
            positions = subject.index.get_indexer(grouped.index)
            subject_weight[positions] = 1.0 / len(grouped)
        subject_weight /= subject_weight.sum()
        weights[subject["row"].to_numpy(dtype=np.int64)] = subject_weight
    weights *= len(work["subject_key"].unique()) / weights.sum()
    return weights.astype(np.float32)


def _bin_specs(start_ms: int, end_ms: int, config: dict[str, Any]):
    specifications = (
        (
            start_ms - int(config["left_context_seconds"]) * 1000,
            start_ms,
            int(config["left_bins"]),
            0,
        ),
        (start_ms, end_ms, int(config["event_bins"]), 1),
        (
            end_ms,
            end_ms + int(config["right_context_seconds"]) * 1000,
            int(config["right_bins"]),
            2,
        ),
    )
    total = sum(value[2] for value in specifications)
    position = 0
    for left, right, count, region in specifications:
        edges = np.linspace(left, right, count + 1)
        for index in range(count):
            yield int(edges[index]), int(edges[index + 1]), region, position / max(total - 1, 1)
            position += 1


def build_proposal_features_v4(
    proposals: pd.DataFrame,
    windows: pd.DataFrame,
    statistics_columns: list[str],
    config: dict[str, Any],
    *,
    raw_segments: pd.DataFrame | None = None,
    raw_session: Any | None = None,
) -> ProposalFeatureBatchV4:
    use_raw_imu = bool(config.get("use_raw_imu_branch", False))
    source_bits = 7 if bool(config.get("include_v48_source_flags", False)) else 4
    include_v49_lineage = bool(config.get("include_v49_lineage", False))
    if include_v49_lineage:
        from bme_eating.proposals_v4 import validate_proposal_lineage

        validate_proposal_lineage(proposals)
    if use_raw_imu and (raw_segments is None) == (raw_session is None):
        raise ValueError("Raw IMU verifier requires exactly one raw input source")
    reader = None
    if raw_segments is not None and use_raw_imu:
        from bme_eating.data.session import SessionWindowReader

        reader = SessionWindowReader(raw_segments)
    base_columns = [
        "state_probability",
        "onset_probability",
        "offset_probability",
        "ppg_gate",
        "statistics_gate",
        "long_gate",
        "gyro_gate",
        "invariant_gate",
        "missing_fraction",
        "acc_valid_fraction",
        "gyro_valid_fraction",
        "ppg_valid_fraction",
        "statistics_missing_fraction",
        *statistics_columns,
    ]
    use_latent_bridge = bool(config.get("use_state_latent_bridge", False))
    use_boundary_quality = bool(config.get("use_boundary_quality_features", False))
    latent_columns = []
    if use_latent_bridge:
        latent_dim = int(config.get("state_latent_dim", 64))
        if latent_dim <= 0:
            raise ValueError("state_latent_dim must be positive")
        latent_columns = [f"state_hidden_{index:03d}" for index in range(latent_dim)]
        missing_latent = set(latent_columns) - set(windows.columns)
        if missing_latent:
            raise ValueError(
                "Verifier windows are missing state latent columns: "
                f"{sorted(missing_latent)[:5]}"
            )
    if use_boundary_quality:
        base_columns.append("state_probability_derivative")
    required = {"subject_key", "session_id", "timestamp_ms", *base_columns}
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"Verifier windows are missing columns: {sorted(missing)}")
    groups = {
        (str(subject), str(session)): group.sort_values("timestamp_ms")
        for (subject, session), group in windows.groupby(["subject_key", "session_id"], sort=False)
    }
    sequences: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    scalars: list[np.ndarray] = []
    latent_summaries: list[np.ndarray] = []
    boundary_summaries: list[np.ndarray] = []
    raw_imu_snippets: list[np.ndarray] = []
    raw_by_index: dict[int, np.ndarray] = {}
    if reader is not None:
        indexed = proposals.reset_index(drop=True).reset_index(names="proposal_row_index")
        for (subject_key, session_id), session_proposals in indexed.groupby(
            ["subject_key", "session_id"], sort=False
        ):
            payload = reader.read(
                str(session_id),
                int(session_proposals["coarse_start_ms"].min()) - 15_000,
                int(session_proposals["coarse_end_ms"].max()) + 15_000,
                subject_key=str(subject_key),
            )
            for candidate in session_proposals.itertuples(index=False):
                raw_by_index[int(candidate.proposal_row_index)] = _raw_imu_proposal_snippets(
                    payload["motion_timestamp_ms"],
                    payload["motion_values"],
                    payload["motion_mask"],
                    int(candidate.coarse_start_ms),
                    int(candidate.coarse_end_ms),
                    normalize_locally=config.get("raw_imu_normalization") != "training_fold_robust_v1",
                )
    for proposal_index, proposal in enumerate(proposals.itertuples(index=False)):
        if use_raw_imu:
            if reader is not None:
                raw_imu_snippets.append(raw_by_index[proposal_index])
            else:
                if (str(raw_session.subject_key), str(raw_session.session_id)) != (
                    str(proposal.subject_key), str(proposal.session_id)
                ):
                    raise ValueError("Raw session identity does not match verifier proposal")
                raw_imu_snippets.append(_raw_imu_proposal_snippets(
                    raw_session.motion_timestamp_ms,
                    raw_session.motion_values,
                    raw_session.motion_mask,
                    int(proposal.coarse_start_ms), int(proposal.coarse_end_ms),
                    normalize_locally=config.get("raw_imu_normalization") != "training_fold_robust_v1",
                ))
        group = groups.get((str(proposal.subject_key), str(proposal.session_id)))
        if group is None:
            raise ValueError(f"No verifier windows for proposal {proposal.proposal_id}")
        timestamps = group["timestamp_ms"].to_numpy(dtype=np.int64)
        values = group[base_columns].to_numpy(dtype=np.float64)
        latent_values = (
            group[latent_columns].to_numpy(dtype=np.float64)
            if latent_columns
            else None
        )
        bins: list[np.ndarray] = []
        valid_bins: list[bool] = []
        selected_latent_rows: list[np.ndarray] = []
        selected_boundary_rows: list[np.ndarray] = []
        for left, right, region, relative in _bin_specs(
            int(proposal.coarse_start_ms), int(proposal.coarse_end_ms), config
        ):
            selected = (timestamps > left) & (timestamps <= right)
            selected_values = values[selected]
            if len(selected_values) and use_latent_bridge:
                if latent_values is None:
                    raise RuntimeError("Latent bridge values were not loaded")
                selected_latent_rows.append(latent_values[selected])
            if len(selected_values) and use_boundary_quality:
                derivative_index = base_columns.index("state_probability_derivative")
                selected_boundary_rows.append(
                    selected_values[:, [1, 2, derivative_index]]
                )
            valid_bins.append(bool(len(selected_values)))
            if len(selected_values):
                finite = np.isfinite(selected_values)
                cleaned = np.where(finite, selected_values, np.nan)
                with np.errstate(all="ignore"):
                    mean = np.nanmean(cleaned, axis=0)
                    maximum = np.nanmax(cleaned, axis=0)
                    standard_deviation = np.nanstd(cleaned, axis=0)
                last = np.zeros(len(base_columns), dtype=np.float64)
                for column in range(len(base_columns)):
                    indices = np.flatnonzero(finite[:, column])
                    if len(indices):
                        last[column] = selected_values[indices[-1], column]
                count = finite.sum(axis=0).astype(np.float64)
            else:
                mean = maximum = standard_deviation = last = count = np.zeros(
                    len(base_columns), dtype=np.float64
                )
            region_one_hot = np.eye(3, dtype=np.float64)[region]
            metadata = np.asarray(
                [
                    *region_one_hot,
                    relative,
                    (right - left) / 1000.0,
                    float(not len(selected_values)),
                ]
            )
            bins.append(
                np.nan_to_num(
                    np.concatenate((mean, maximum, standard_deviation, last, count, metadata)),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).astype(np.float32)
            )
        duration_seconds = (int(proposal.coarse_end_ms) - int(proposal.coarse_start_ms)) / 1000.0
        in_event = (timestamps > int(proposal.coarse_start_ms)) & (
            timestamps <= int(proposal.coarse_end_ms)
        )
        state_mean = float(np.nanmean(values[in_event, 0])) if in_event.any() else 0.0
        source_mask = int(proposal.source_mask)
        scalars.append(
            np.asarray(
                [
                    np.log1p(max(duration_seconds, 0.0)),
                    float(proposal.generator_score),
                    state_mean,
                    *((source_mask & (1 << bit)) > 0 for bit in range(source_bits)),
                    float(np.mean(valid_bins)),
                    *(
                        [
                            (float(proposal.coarse_start_ms) - float(proposal.parent_start_ms))
                            / 600_000.0,
                            (float(proposal.coarse_end_ms) - float(proposal.parent_end_ms))
                            / 600_000.0,
                            float(np.mean(np.isfinite(values[in_event, 0]))) if in_event.any() else 0.0,
                            float(np.mean(np.isfinite(values[in_event, 1]))) if in_event.any() else 0.0,
                        ]
                        if include_v49_lineage
                        else []
                    ),
                ],
                dtype=np.float32,
            )
        )
        if use_latent_bridge and selected_latent_rows:
            latent_values = np.concatenate(selected_latent_rows, axis=0)
            latent_summaries.append(
                np.concatenate(
                    (np.nanmean(latent_values, axis=0), np.nanmax(latent_values, axis=0))
                ).astype(np.float32)
            )
        elif use_latent_bridge:
            latent_summaries.append(np.zeros(2 * len(latent_columns), dtype=np.float32))
        if use_boundary_quality and selected_boundary_rows:
            boundary_values = np.concatenate(selected_boundary_rows, axis=0)
            boundary_summaries.append(
                np.asarray(
                    [
                        float(np.nanmean(boundary_values[:, 0])),
                        float(np.nanmean(boundary_values[:, 1])),
                        float(np.nanmean(np.abs(boundary_values[:, 2]))),
                        float(boundary_values[0, 0] + boundary_values[-1, 1]),
                    ],
                    dtype=np.float32,
                )
            )
        elif use_boundary_quality:
            boundary_summaries.append(np.zeros(4, dtype=np.float32))
        sequences.append(np.stack(bins))
        masks.append(np.asarray(valid_bins, dtype=bool))
    event_target = (
        (proposals["max_iou"].to_numpy(dtype=np.float32) > 0.25).astype(np.float32)
        if "max_iou" in proposals
        else None
    )
    iou_target = proposals["max_iou"].to_numpy(dtype=np.float32) if "max_iou" in proposals else None
    weights = normalized_proposal_weights(proposals) if "max_iou" in proposals else None
    bin_count = int(config["left_bins"]) + int(config["event_bins"]) + int(config["right_bins"])
    feature_dim = 5 * len(base_columns) + 6
    scalar_values = np.stack(scalars) if scalars else np.empty((0, 4 + source_bits + (4 if include_v49_lineage else 0)), np.float32)
    if use_latent_bridge:
        latent_values = (
            np.stack(latent_summaries)
            if latent_summaries
            else np.empty((0, 2 * len(latent_columns)), np.float32)
        )
        scalar_values = np.concatenate((scalar_values, latent_values), axis=1)
    if use_boundary_quality:
        boundary_values = (
            np.stack(boundary_summaries)
            if boundary_summaries
            else np.empty((0, 4), np.float32)
        )
        scalar_values = np.concatenate((scalar_values, boundary_values), axis=1)
    return ProposalFeatureBatchV4(
        proposal_ids=proposals["proposal_id"].astype(str).to_numpy(),
        sequence=np.stack(sequences)
        if sequences
        else np.empty((0, bin_count, feature_dim), np.float32),
        sequence_mask=np.stack(masks) if masks else np.empty((0, bin_count), bool),
        scalar=scalar_values,
        event_target=event_target,
        iou_target=iou_target,
        sample_weight=weights,
        raw_imu=np.stack(raw_imu_snippets) if use_raw_imu and raw_imu_snippets else None,
        sampling_metadata=(
            _proposal_sampling_metadata(proposals, windows) if include_v49_lineage else None
        ),
    )


def _proposal_sampling_metadata(proposals: pd.DataFrame, windows: pd.DataFrame | None = None) -> dict[str, np.ndarray] | None:
    if proposals.empty:
        return {name: np.empty(0, dtype=str) for name in (
            "subject_key", "wear_hand", "hand_relation", "iou_bin", "gyro_missingness_bin",
            "duration_bin", "source_mask", "source_family", "session_id", "proposal_family_id", "event_group")}
    def values(name: str, default: str = "unknown") -> np.ndarray:
        if name in proposals:
            return proposals[name].fillna(default).astype(str).to_numpy()
        return np.full(len(proposals), default, dtype=str)

    iou = proposals.get("max_iou", pd.Series(0.0, index=proposals.index)).fillna(0.0).to_numpy(dtype=float)
    gyro = proposals.get("gyro_valid_fraction", pd.Series(np.nan, index=proposals.index)).to_numpy(dtype=float)
    hand = values("hand_relation").astype(object)
    if windows is not None:
        groups = {
            (str(subject), str(session)): group.sort_values("timestamp_ms")
            for (subject, session), group in windows.groupby(["subject_key", "session_id"], sort=False)
        }
        for proposal_index, proposal in enumerate(proposals.itertuples(index=False)):
            group = groups.get((str(proposal.subject_key), str(proposal.session_id)))
            if group is None:
                continue
            timestamps = group["timestamp_ms"].to_numpy()
            left = np.searchsorted(timestamps, int(proposal.coarse_start_ms), side="right")
            right = np.searchsorted(timestamps, int(proposal.coarse_end_ms), side="right")
            event_windows = group.iloc[left:right]
            if "gyro_valid_fraction" in event_windows and len(event_windows):
                gyro[proposal_index] = event_windows["gyro_valid_fraction"].mean()
            if hand[proposal_index] not in {"same", "different"} and "hand_relation" in event_windows and len(event_windows):
                modes = event_windows["hand_relation"].dropna().astype(str).mode()
                if len(modes):
                    hand[proposal_index] = modes.iloc[0]
    if "duration_seconds" in proposals:
        duration = proposals["duration_seconds"].fillna(0.0).to_numpy(dtype=float)
    elif {"coarse_start_ms", "coarse_end_ms"}.issubset(proposals.columns):
        duration = (proposals["coarse_end_ms"].to_numpy(dtype=float) - proposals["coarse_start_ms"].to_numpy(dtype=float)) / 1000.0
    elif {"start_ms", "end_ms"}.issubset(proposals.columns):
        duration = (proposals["end_ms"].to_numpy(dtype=float) - proposals["start_ms"].to_numpy(dtype=float)) / 1000.0
    else:
        duration = np.zeros(len(proposals), dtype=float)
    return {
        "subject_key": values("subject_key"),
        "wear_hand": values("wear_hand"),
        "hand_relation": hand,
        "iou_bin": pd.cut(iou, bins=[-np.inf, 0.0, 0.10, 0.25, np.inf], labels=False).astype(str),
        "gyro_missingness_bin": np.where(~np.isfinite(gyro) | (gyro < 0.5), "missing", "observed"),
        "duration_bin": pd.cut(duration, bins=[-np.inf, 5.0, 15.0, 30.0, 60.0, np.inf], labels=False).astype(str),
        "source_mask": values("source_mask", "0"),
        "source_family": np.asarray([proposal_source_family(int(value)) for value in values("source_mask", "0")]),
        "session_id": values("session_id"),
        "proposal_family_id": values("proposal_family_id", "unknown"),
        "event_group": (
            proposals.get("session_id", pd.Series("", index=proposals.index)).fillna("").astype(str)
            + ":"
            + proposals.get("matched_event_id", pd.Series("", index=proposals.index)).fillna("").astype(str)
        ).to_numpy(),
    }


def _raw_imu_proposal_snippets(
    timestamps: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    start_ms: int,
    end_ms: int,
    *,
    normalize_locally: bool = True,
) -> np.ndarray:
    from bme_eating.data.deep_dataset import _sample_grid

    centers = (start_ms, (start_ms + end_ms) // 2, end_ms)
    snippets: list[np.ndarray] = []
    for center in centers:
        sampled, sampled_mask = _sample_grid(
            np.asarray(timestamps, dtype=np.int64),
            np.asarray(values, dtype=np.float32),
            np.asarray(mask, dtype=bool),
            int(center) - 15_000,
            30,
            10,
        )
        normalized = np.zeros_like(sampled)
        sampled_mask &= np.isfinite(sampled)
        for channel in range(6):
            valid = sampled_mask[:, channel]
            if not valid.any():
                continue
            if not normalize_locally:
                normalized[valid, channel] = sampled[valid, channel]
                continue
            median = np.median(sampled[valid, channel])
            spread = max(float(np.percentile(sampled[valid, channel], 75)
                               - np.percentile(sampled[valid, channel], 25)), 1e-3)
            normalized[valid, channel] = np.clip(
                (sampled[valid, channel] - median) / spread, -10, 10
            )
        snippets.append(np.concatenate((normalized, sampled_mask.astype(np.float32)), axis=1).T)
    return np.stack(snippets).astype(np.float16)


def fit_raw_imu_normalization(model: EventVerifierV4, features: ProposalFeatureBatchV4) -> None:
    if not model.use_raw_imu_branch or not model.use_fold_raw_normalization:
        return
    raw = features.raw_imu
    if raw is None or raw.shape[1:] != (3, 12, 300):
        raise ValueError("v4.9 raw IMU normalization requires training snippets")
    centers, scales = [], []
    sample_stride = max(1, int(np.ceil(len(raw) * 900 / 1_000_000)))
    for channel in range(6):
        values = raw[::sample_stride, :, channel, :].reshape(-1)
        valid = raw[::sample_stride, :, channel + 6, :].reshape(-1).astype(bool) & np.isfinite(values)
        selected = values[valid].astype(np.float32)
        selected = selected[::max(1, int(np.ceil(len(selected) / 1_000_000)))]
        centers.append(float(np.median(selected)) if len(selected) else 0.0)
        scales.append(max(float(np.percentile(selected, 75) - np.percentile(selected, 25)), 1e-3) if len(selected) else 1.0)
    with torch.no_grad():
        model.raw_imu_center.copy_(torch.tensor(centers, device=model.raw_imu_center.device))
        model.raw_imu_scale.copy_(torch.tensor(scales, device=model.raw_imu_scale.device))
