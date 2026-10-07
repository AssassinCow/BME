from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np
import torch

from bme_eating.config import load_config
from bme_eating.data.stats_fusion_preprocess import causal_completed_block_layout
from bme_eating.data.stats_fusion_sequence import sequence_geometry_from_config
from bme_eating.models.event_verifier_v4 import EventVerifierV4, verifier_loss_v4
from bme_eating.models.factory import build_state_model
from bme_eating.models.stats_fusion_loss import StatsFusionStateLoss


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the StatsFusion v4 bf16 smoke test")
    parser.add_argument(
        "--config", default="configs/hierarchical_v4_r32_pooled_heads_early_select.yaml"
    )
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(config["training"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the v4 smoke configuration")
    geometry = sequence_geometry_from_config(config)
    batch_size = args.batch_size or int(config["training"]["batch_size"])
    steps = geometry.total_steps
    timestamps = np.arange(1, steps + 1, dtype=np.int64) * geometry.step_seconds * 1000
    _, block_end_indices, mapping = causal_completed_block_layout(
        timestamps,
        session_origin_ms=0,
        step_ms=geometry.step_seconds * 1000,
        factor=geometry.long_pool_factor,
    )
    ppg_steps = len(block_end_indices)
    motion_blocks = torch.randn(batch_size, steps, 12, 300, device=device)
    motion_blocks[:, :, 6:] = 1.0
    invariant_blocks = torch.randn(batch_size, steps, 6, 300, device=device)
    invariant_blocks[:, :, 3:] = 1.0
    ppg_blocks = torch.randn(batch_size, ppg_steps, 2, 750, device=device)
    ppg_blocks[:, :, 1] = 1.0
    statistics = torch.randn(batch_size, steps, 24, device=device)
    statistics[:, :, 12:] = 0.0
    batch = {
        "motion_blocks": motion_blocks,
        "motion_valid": torch.ones(batch_size, steps, device=device),
        "motion_invariant_blocks": invariant_blocks,
        "ppg_blocks": ppg_blocks,
        "ppg_quality": torch.randn(batch_size, ppg_steps, 8, device=device),
        "ppg_valid": torch.ones(batch_size, ppg_steps, device=device),
        "ppg_to_motion_index": torch.from_numpy(mapping)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .to(device),
        "long_block_end_indices": torch.from_numpy(block_end_indices)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .to(device),
        "statistics": statistics,
        "state_target": torch.randint(0, 2, (batch_size, steps), device=device).float(),
        "onset_target": torch.zeros(batch_size, steps, device=device),
        "offset_target": torch.zeros(batch_size, steps, device=device),
        "state_loss_mask": torch.ones(batch_size, steps, device=device),
        "supervision_mask": torch.zeros(batch_size, steps, device=device),
        "smooth_mask": torch.ones(batch_size, steps, device=device),
    }
    batch["supervision_mask"][:, -geometry.supervised_steps :] = 1.0
    model = build_state_model(config["model"]).to(device).train()
    criterion = StatsFusionStateLoss(
        smooth_weight=float(config["loss"]["smooth_weight"]),
        smooth_beta=float(config["loss"]["smooth_beta"]),
        boundary_weight=float(config["loss"]["boundary_weight"]),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        output = model(batch)
        loss, _ = criterion(output, batch)
    loss.backward()
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    )
    if device.type == "cuda" and peak_gb >= 10.5:
        raise RuntimeError(f"V4 smoke peak memory {peak_gb:.3f} GB exceeds 10.5 GB")
    model.eval()
    with torch.no_grad():
        first = torch.sigmoid(model(batch)["state_logit"].float())
        second = torch.sigmoid(model(batch)["state_logit"].float())
    maximum_error = float(torch.max(torch.abs(first - second)).cpu())
    if maximum_error > 1e-6:
        raise RuntimeError(f"Repeated inference differs by {maximum_error:.3e}")
    del output, loss, first, second, model, batch, motion_blocks, invariant_blocks, ppg_blocks, statistics
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    verifier_batch = {
        "sequence": torch.randn(batch_size, 32, 160, device=device),
        "sequence_mask": torch.ones(batch_size, 32, dtype=torch.bool, device=device),
        "scalar": torch.randn(batch_size, 128, device=device),
        "event_target": torch.ones(batch_size, device=device),
        "iou_target": torch.full((batch_size,), 0.5, device=device),
    }
    if config["verifier"].get("use_raw_imu_branch"):
        raw = torch.randn(batch_size, 3, 12, 300, device=device)
        raw[:, :, 6:] = 1
        raw[:, :, 9:] = 0
        raw[:, :, 3:6] = 0
        verifier_batch["raw_imu"] = raw
    verifier = EventVerifierV4(160, 128, config["verifier"]).to(device).train()
    verifier_output = verifier(verifier_batch)
    verifier_loss, _ = verifier_loss_v4(verifier_output, verifier_batch)
    verifier_loss.backward()
    if not torch.isfinite(verifier_loss) or any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in verifier.parameters()):
        raise RuntimeError("Deep verifier smoke has non-finite loss or gradients")
    verifier.eval()
    with torch.no_grad():
        first_deep = verifier(verifier_batch)["event_logit"]
        second_deep = verifier(verifier_batch)["event_logit"]
    deep_error = float((first_deep - second_deep).abs().max().cpu())
    deep_peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    if deep_error > 1e-6 or deep_peak >= 10.5:
        raise RuntimeError("Deep verifier smoke fails determinism or memory qualification")
    print(
        json.dumps(
            {
                "device": str(device),
                "batch_size": batch_size,
                "gradient_accumulation": int(config["training"]["gradient_accumulation"]),
                "effective_batch_size": batch_size
                * int(config["training"]["gradient_accumulation"]),
                "steps": steps,
                "peak_memory_gb": peak_gb,
                "repeat_probability_max_error": maximum_error,
                "deep_repeat_logit_max_error": deep_error,
                "deep_peak_memory_gb": deep_peak,
                "raw_imu_branch": bool(config["verifier"].get("use_raw_imu_branch")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
