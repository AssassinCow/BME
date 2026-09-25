from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np
import torch

from bme_eating.config import load_config
from bme_eating.data.stats_fusion_preprocess import causal_completed_block_layout
from bme_eating.data.stats_fusion_sequence import SequenceGeometry
from bme_eating.models.factory import build_state_model
from bme_eating.models.stats_fusion_loss import StatsFusionStateLoss


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the StatsFusion v4 bf16 smoke test")
    parser.add_argument("--config", default="configs/hierarchical_v4_statsfusion.yaml")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(config["training"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the v4 smoke configuration")
    geometry = SequenceGeometry(
        supervised_steps=int(config["sequence"]["supervised_steps"]),
        short_receptive_field_steps=int(config["sequence"]["short_receptive_field_steps"]),
        long_receptive_field_tokens=int(config["sequence"]["long_receptive_field_tokens"]),
        long_pool_factor=int(config["sequence"]["long_pool_factor"]),
        step_seconds=int(config["sequence"]["step_seconds"]),
    )
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
    batch = {
        "motion_blocks": torch.randn(batch_size, steps, 12, 300, device=device),
        "motion_valid": torch.ones(batch_size, steps, device=device),
        "ppg_blocks": torch.randn(batch_size, ppg_steps, 2, 750, device=device),
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
        "statistics": torch.randn(batch_size, steps, 24, device=device),
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
        smooth_tau=float(config["loss"]["smooth_tau"]),
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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
