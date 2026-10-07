from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

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
from bme_eating.data.stats_fusion_sequence import (
    StatsFusionSequenceDataset,
    sequence_geometry_from_config,
)
from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.hierarchical_v4_pipeline import load_hierarchical_v4_bundle
from bme_eating.models.factory import build_state_model
from bme_eating.stats_features import STATS_FEATURE_COLUMNS, FoldRobustScaler
from bme_eating.v4_protocol import (
    PROTOCOL_VERSION,
    execution_environment_identity,
    execution_source_identity,
)


def validate_replay_lock(bundle: Path, config: dict) -> dict:
    if config.get("v49", {}).get("protocol") != "integrated_repair_v2":
        return {}
    path = bundle / "v49_deployment_protocol.json"
    if not path.is_file():
        raise RuntimeError("v4.9 replay requires the deployment protocol lock")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (lock.get("protocol") != "v49_deployment_lock_v2"
            or lock.get("execution_source_identity") != execution_source_identity(Path(__file__).resolve().parents[1])
            or lock.get("execution_environment") != execution_environment_identity()
            or lock.get("config_sha256") != sha256_file(bundle / "resolved_config.yaml")
            or lock.get("nested_state_protocol") != "fully_excluded_nested_state_oof_v1"
            or lock.get("boundary_passed") is not True
            or lock.get("state_seeds") != [2026] or lock.get("verifier_seeds") != [2026, 2027, 2028]):
        raise RuntimeError("v4.9 replay protocol/runtime identity changed")
    for name, protocol in (("v49_deep_gate.json", "v49_deep_promotion_v1"),
                           ("v49_raw_imu_gate.json", "v49_raw_imu_nested_crossfit_gate_v2")):
        gate = json.loads((bundle / name).read_text(encoding="utf-8"))
        if gate.get("protocol") != protocol or gate.get("passed") is not True:
            raise RuntimeError("v4.9 replay requires qualified Deep and raw IMU gates")
    return lock


def _repeat_frame_error(first: pd.DataFrame, second: pd.DataFrame, identity: list[str]) -> float:
    if list(first.columns) != list(second.columns) or not first[identity].equals(second[identity]):
        raise RuntimeError("Bundle replay repeated inference changes intermediate identities")
    numeric = first.select_dtypes(include=["number"]).columns.difference(identity)
    if first.empty or not len(numeric):
        return 0.0
    first_values = first[numeric].to_numpy(dtype=np.float64)
    second_values = second[numeric].to_numpy(dtype=np.float64)
    if not np.isfinite(first_values).all() or not np.isfinite(second_values).all():
        raise RuntimeError("Bundle replay produced non-finite intermediate predictions")
    return float(np.max(np.abs(first_values - second_values)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay one real multi-fragment session through canonical v4 preprocessing"
    )
    parser.add_argument(
        "--config", default="configs/hierarchical_v4_r32_pooled_heads_early_select.yaml"
    )
    parser.add_argument("--segment-id")
    parser.add_argument("--session-id")
    parser.add_argument("--output")
    parser.add_argument("--bundle")
    parser.add_argument("--forbid-xgboost", action="store_true")
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
        scaler = FoldRobustScaler.fit(merged, STATS_FEATURE_COLUMNS, training_subjects={subject})
    transformed = scaler.transform_frame(merged)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        normalization = compute_normalization(
            selected,
            {subject},
            require_motion=bool(config["model"].get("use_motion", True)),
            require_ppg=bool(config["model"].get("use_ppg", True)),
        )
    geometry = sequence_geometry_from_config(config)
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
        subject_key=subject,
        session_id=session_id,
        motion_timestamp_ms=payload["motion_timestamp_ms"],
        motion_values=payload["motion_values"],
        motion_mask=payload["motion_mask"],
        ppg_timestamp_ms=payload["ppg_timestamp_ms"],
        ppg_values=payload["ppg_values"],
        ppg_mask=payload["ppg_mask"],
    )
    if args.bundle:
        replay_lock = validate_replay_lock(Path(args.bundle), config)
        if args.forbid_xgboost:
            class BlockXGBoost:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "xgboost" or fullname.startswith("xgboost."):
                        raise ImportError("XGBoost is unavailable during bundle replay")

            if any(name == "xgboost" or name.startswith("xgboost.") for name in sys.modules):
                raise RuntimeError("XGBoost was already imported before the isolated replay")
            sys.meta_path.insert(0, BlockXGBoost())
        device = torch.device(config["training"]["device"])
        detector = load_hierarchical_v4_bundle(args.bundle, device=device)
        state_trace: list[pd.DataFrame] = []
        score_trace: list[pd.DataFrame] = []
        predict_state = detector.predict_state_sequence
        score_proposals = detector._score_proposals

        def traced_state(*arguments, **keywords):
            frame = predict_state(*arguments, **keywords)
            columns = ["subject_key", "session_id", "timestamp_ms", "state_logit",
                       "onset_logit", "offset_logit", "state_probability", "proposal_logit"]
            state_trace.append(frame[[column for column in columns if column in frame]].copy())
            return frame

        def traced_scores(*arguments, **keywords):
            frame = score_proposals(*arguments, **keywords)
            columns = ["proposal_id", "final_score"]
            if detector.selection.get("verifier_kind", "deep") == "deep":
                columns.extend(("event_logit", "iou_logit"))
            score_trace.append(frame[[column for column in columns if column in frame]].copy())
            return frame

        detector.predict_state_sequence = traced_state
        detector._score_proposals = traced_scores
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        first = detector.predict_session(raw)
        first_state = pd.concat(state_trace, ignore_index=True)
        first_scores = pd.concat(score_trace, ignore_index=True) if score_trace else pd.DataFrame(columns=["proposal_id"])
        state_trace.clear()
        score_trace.clear()
        second = detector.predict_session(raw)
        second_state = pd.concat(state_trace, ignore_index=True)
        second_scores = pd.concat(score_trace, ignore_index=True) if score_trace else pd.DataFrame(columns=["proposal_id"])
        if [event.event_id for event in first] != [event.event_id for event in second]:
            raise RuntimeError("Bundle replay repeated inference changes event identities")
        maximum_error = max((
            max(abs(left.start_ms - right.start_ms), abs(left.end_ms - right.end_ms), abs(left.score - right.score))
            for left, right in zip(first, second)
        ), default=0.0)
        state_error = _repeat_frame_error(first_state, second_state, ["subject_key", "session_id", "timestamp_ms"])
        score_error = _repeat_frame_error(first_scores, second_scores, ["proposal_id"])
        maximum_error = max(maximum_error, state_error, score_error)
        peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
        if not np.isfinite(maximum_error) or maximum_error > 1e-6 or peak >= 10.5:
            raise RuntimeError("Bundle replay fails determinism or memory qualification")
        report = {
            "protocol": "v49_checkpoint_raw_session_replay_v1", "session_id": session_id,
            "bundle": str(Path(args.bundle).resolve()), "event_count": len(first),
            "repeat_max_error": maximum_error, "peak_memory_gb": peak,
            "repeat_state_max_error": state_error, "repeat_proposal_max_error": score_error,
            "xgboost_import_forbidden": bool(args.forbid_xgboost),
            "checkpoint_manifest": json.loads((Path(args.bundle) / "SHA256SUMS.json").read_text(encoding="utf-8")),
            "protocol_lock": replay_lock,
        }
        report_path = Path(args.output).resolve() if args.output else Path(args.bundle).parent / "bundle_replay.json"
        write_json_atomic(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
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
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        first = model(model_batch)
        second = model(model_batch)
    repeat_error = float(
        torch.max(torch.abs(first["state_logit"].float() - second["state_logit"].float())).cpu()
    )
    if repeat_error > 1e-6:
        raise RuntimeError(f"Real-session repeated inference differs by {repeat_error}")
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "session_id": session_id,
        "fragment_count": len(selected),
        "subject_key": subject,
        "endpoint_ms": endpoint_ms,
        "sampling_diagnostics": raw.sampling_diagnostics(),
        "preprocessing_max_errors": errors,
        "repeat_logit_max_error": repeat_error,
        "device": str(device),
    }
    if args.output:
        report_path = Path(args.output).expanduser().resolve()
    else:
        identity = hashlib.sha256(f"{subject}\0{session_id}".encode()).hexdigest()[:16]
        report_path = output_root / "diagnostics" / "raw_session_replay" / f"{identity}.json"
    write_json_atomic(report_path, report)
    print(json.dumps({**report, "report_path": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
