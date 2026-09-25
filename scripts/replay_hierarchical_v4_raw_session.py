from __future__ import annotations

import argparse
import json
import warnings

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.data.deep_dataset import compute_normalization
from bme_eating.data.session import SessionWindowReader
from bme_eating.data.stats_fusion_inputs import (
    canonical_input_paths,
    verify_canonical_statsfusion_inputs,
)
from bme_eating.data.stats_fusion_preprocess import (
    RawSessionInput,
    StatsFusionRawSessionPreprocessor,
)
from bme_eating.data.stats_fusion_sequence import SequenceGeometry, StatsFusionSequenceDataset
from bme_eating.models.factory import build_state_model
from bme_eating.stats_features import STATS_FEATURE_COLUMNS, FoldRobustScaler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay one real multi-fragment session through canonical v4 preprocessing"
    )
    parser.add_argument("--config", default="configs/hierarchical_v4_statsfusion.yaml")
    parser.add_argument("--segment-id")
    parser.add_argument("--session-id")
    args = parser.parse_args()
    config = load_config(args.config)
    _, input_root, output_root = resolve_artifact_roots(config)
    verify_canonical_statsfusion_inputs(input_root, output_root)
    canonical = canonical_input_paths(output_root)
    segments = pd.read_parquet(input_root / "indices" / "segments.parquet")
    if args.segment_id and args.session_id:
        raise ValueError("Specify either --segment-id or --session-id, not both")
    if args.session_id:
        session_id = str(args.session_id)
    elif args.segment_id:
        matched = segments[segments["segment_id"].astype(str) == args.segment_id]
        if len(matched) != 1:
            raise ValueError("Requested segment_id was not found uniquely")
        session_id = str(matched.iloc[0].session_id)
    else:
        counts = segments.groupby("session_id", sort=False).size().sort_values(ascending=False)
        if counts.empty or int(counts.iloc[0]) < 2:
            raise RuntimeError("No multi-fragment v2 session is available for raw replay")
        session_id = str(counts.index[0])
    selected = segments[segments["session_id"].astype(str) == session_id].copy()
    if not len(selected):
        raise ValueError("Requested session_id was not found")
    selected = selected.sort_values(["start_ms", "end_ms", "segment_id"], kind="stable")
    subject = str(selected.iloc[0].subject_key)
    anchors = pd.read_parquet(
        canonical["anchors"],
        filters=[("session_id", "==", session_id)],
    ).sort_values("timestamp_ms")
    statistics = pd.read_parquet(
        canonical["statistics"],
        filters=[("session_id", "==", session_id)],
    )
    keys = ["session_id", "subject_key", "timestamp_ms"]
    statistics = statistics[[*keys, *STATS_FEATURE_COLUMNS]]
    merged = anchors.merge(
        statistics,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scaler = FoldRobustScaler.fit(
            merged, STATS_FEATURE_COLUMNS, training_subjects={subject}
        )
    transformed = scaler.transform_frame(merged)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        normalization = compute_normalization(selected, {subject})
    sequence = config["sequence"]
    geometry = SequenceGeometry(
        supervised_steps=int(sequence["supervised_steps"]),
        short_receptive_field_steps=int(sequence["short_receptive_field_steps"]),
        long_receptive_field_tokens=int(sequence["long_receptive_field_tokens"]),
        long_pool_factor=int(sequence["long_pool_factor"]),
        step_seconds=int(sequence["step_seconds"]),
    )
    dataset = StatsFusionSequenceDataset(
        transformed,
        selected,
        pd.DataFrame(),
        normalization,
        statistics_columns=[
            *(f"stat_{name}" for name in STATS_FEATURE_COLUMNS),
            *(f"stat_{name}_missing" for name in STATS_FEATURE_COLUMNS),
        ],
        geometry=geometry,
        training=False,
    )
    reader = SessionWindowReader(selected)
    payload = reader.read(
        session_id,
        int(selected["start_ms"].min()) - 15_000,
        int(selected["end_ms"].max()),
    )
    raw = RawSessionInput(
        subject,
        session_id,
        payload["motion_timestamp_ms"],
        payload["motion_values"],
        payload["motion_mask"],
        payload["ppg_timestamp_ms"],
        payload["ppg_values"],
        payload["ppg_mask"],
    )
    preprocessor = StatsFusionRawSessionPreprocessor(
        normalization=normalization,
        statistics_scaler=scaler,
        geometry=geometry,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw_batch = next(iter(preprocessor.iter_state_batches(raw)))
    endpoint_ms = int(raw_batch["timestamp_ms"][0, -1])
    row_index = int(
        transformed.index[transformed["timestamp_ms"].astype(np.int64) == endpoint_ms][0]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dataset_batch = dataset[row_index]
    errors = {}
    for name in (
        "motion_blocks",
        "motion_valid",
        "ppg_blocks",
        "ppg_quality",
        "ppg_valid",
        "ppg_to_motion_index",
        "long_block_end_indices",
        "statistics",
    ):
        left = raw_batch[name][0].numpy()
        right = dataset_batch[name].numpy()
        errors[name] = float(np.max(np.abs(left - right)))
    if max(errors.values()) > 1e-6:
        statistics_error = np.abs(
            raw_batch["statistics"][0].numpy() - dataset_batch["statistics"].numpy()
        )
        statistics_columns = [
            *(f"stat_{name}" for name in STATS_FEATURE_COLUMNS),
            *(f"stat_{name}_missing" for name in STATS_FEATURE_COLUMNS),
        ]
        feature_errors = {
            name: float(statistics_error[:, index].max())
            for index, name in enumerate(statistics_columns)
            if statistics_error[:, index].max() > 1e-6
        }
        maximum_index = np.unravel_index(int(statistics_error.argmax()), statistics_error.shape)
        maximum_detail = {
            "timestamp_ms": int(raw_batch["timestamp_ms"][0, maximum_index[0]]),
            "feature": statistics_columns[maximum_index[1]],
            "raw_value": float(raw_batch["statistics"][0, maximum_index[0], maximum_index[1]]),
            "dataset_value": float(dataset_batch["statistics"][maximum_index]),
        }
        raise RuntimeError(
            f"Training/raw preprocessing mismatch: {errors}; "
            f"statistics_feature_errors={feature_errors}; maximum_detail={maximum_detail}"
        )
    device = torch.device(config["training"]["device"])
    model = build_state_model(config["model"]).to(device).eval()
    model_batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in raw_batch.items()
    }
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        first = model(model_batch)
        second = model(model_batch)
    repeat_error = float(
        torch.max(torch.abs(first["state_logit"].float() - second["state_logit"].float())).cpu()
    )
    if repeat_error > 1e-6:
        raise RuntimeError(f"Real-session repeated inference differs by {repeat_error}")
    print(
        json.dumps(
            {
                "protocol_version": "statsfusion-r2",
                "session_id": session_id,
                "fragment_count": len(selected),
                "subject_key": subject,
                "endpoint_ms": endpoint_ms,
                "preprocessing_max_errors": errors,
                "repeat_logit_max_error": repeat_error,
                "device": str(device),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
