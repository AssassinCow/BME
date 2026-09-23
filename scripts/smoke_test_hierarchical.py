from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import torch

from bme_eating.config import load_config
from bme_eating.models.factory import build_state_model


def _batch(config: dict[str, object], batch_size: int, device: torch.device):
    motion_count = sum(int(value) for value in config["motion_bucket_counts"])
    ppg_count = sum(int(value) for value in config["ppg_bucket_counts"])
    batch = {
        "motion_blocks": torch.randn(
            batch_size,
            motion_count,
            12,
            int(config["motion_block_seconds"]) * 100,
            device=device,
        ),
        "motion_valid": torch.ones(batch_size, motion_count, device=device),
        "ppg_blocks": torch.randn(
            batch_size,
            ppg_count,
            2,
            int(config["ppg_block_seconds"]) * 50,
            device=device,
        ),
        "ppg_quality": torch.rand(batch_size, ppg_count, 8, device=device),
        "ppg_valid": torch.ones(batch_size, ppg_count, device=device),
    }
    if bool(config.get("use_stable_state_features", False)):
        batch["stable_features"] = torch.randn(
            batch_size, len(config["stable_feature_columns"]), device=device
        )
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v3 bf16 CUDA acceptance smoke test")
    parser.add_argument("--config", default="configs/hierarchical_v3.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run this test on the RTX 4080 Laptop")
    config = load_config(args.config)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = build_state_model(config["model"]).to(device).train()
    batch = _batch(config["model"], args.batch_size, device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(batch)
        loss = (
            output["state_logit"].square().mean()
            + output["start_logit"].square().mean()
            + output["end_logit"].square().mean()
        )
    loss.backward()
    peak = int(torch.cuda.max_memory_allocated(device))
    if peak > int(10.5 * 1024**3):
        raise RuntimeError(f"Peak CUDA memory exceeds 10.5 GiB: {peak}")
    model.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        first = model(batch)
        second = model(batch)
    maximum_error = max(
        float((first[name].float() - second[name].float()).abs().max().cpu())
        for name in ("state_logit", "start_logit", "end_logit")
    )
    if maximum_error > 1e-6:
        raise RuntimeError(f"Repeated inference error exceeds 1e-6: {maximum_error}")
    print(
        {
            "batch_size": args.batch_size,
            "peak_gpu_memory_bytes": peak,
            "maximum_repeated_inference_error": maximum_error,
        }
    )


if __name__ == "__main__":
    main()
