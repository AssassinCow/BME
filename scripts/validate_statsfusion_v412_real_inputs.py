from __future__ import annotations

import argparse
import copy
import json
import time
from types import SimpleNamespace

import numpy as np
import torch
from _bootstrap import PROJECT_ROOT

from bme_eating.config import _load_project_dotenv, load_config, require_environment_path
from bme_eating.data.joint_sessions import (
    JointInputs,
    SessionChunkSampler,
    make_joint_dataset,
    release_session_loader,
    session_loader,
)
from bme_eating.hierarchical_artifacts import write_json_atomic
from bme_eating.joint_protocol import (
    config_identity,
    configure_numerics,
    digest,
    require_idle_gpu,
    source_identity,
)
from bme_eating.joint_v412_training import heatmap_focal_loss
from bme_eating.models.statsfusion_joint_event_v411 import build_joint_model
from bme_eating.training.statsfusion_joint_trainer import enforce_resources


def main():
    parser = argparse.ArgumentParser(description="Read-only canonical audit and bounded v412 real sensor smoke")
    parser.add_argument("--config", default="configs/statsfusion_v412_integrated_repair.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", default="manifests/v412_real_inputs_validation.json")
    arguments = parser.parse_args()
    config = load_config(PROJECT_ROOT / arguments.config)
    _load_project_dotenv(config)
    base = require_environment_path(config, "output_root_env")
    started = time.monotonic()
    inputs = JointInputs.audit(base / config["project"]["input_artifact_schema_version"],
                               base / config["project"]["canonical_schema_version"])
    require_idle_gpu(arguments.device)
    settings = copy.deepcopy(config)
    settings["training"]["device"] = arguments.device
    configure_numerics(settings)
    subject = min(subject for subject, fold in inputs.subject_folds.items() if fold != 0)
    frames = inputs.frames({subject}, labels=True)
    keys = sorted(set(zip(frames[0].subject_key.astype(str), frames[0].session_id.astype(str))))[:2]
    def subset(subjects, *, labels):
        selected = []
        for frame in inputs.frames(subjects, labels=labels):
            keep = np.array([(str(owner), str(session)) in keys
                             for owner, session in zip(frame.subject_key, frame.session_id)]) if len(frame) else []
            selected.append(frame.loc[keep].copy() if len(frame) else frame.copy())
        return tuple(selected)
    bounded = SimpleNamespace(identity=inputs.identity, frames=subset)
    dataset, preprocessing = make_joint_dataset(bounded, {subject}, settings, training=True)
    sampler = SessionChunkSampler(dataset, 3, training=False)
    sampler.chunks = sampler.chunks[:2]
    torch.manual_seed(2026)
    model = build_joint_model(settings).to(arguments.device).train()
    if arguments.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    loader = session_loader(dataset, sampler, settings)
    reports = []
    try:
        for batch in loader:
            inputs.assert_session_unchanged((batch["subject_key"], batch["session_id"]))
            batch = {name: value.to(arguments.device) if isinstance(value, torch.Tensor) else value
                     for name, value in batch.items()}
            model.zero_grad(set_to_none=True)
            with torch.autocast(arguments.device, dtype=torch.bfloat16, enabled=arguments.device == "cuda"):
                output = model(batch)
                valid = batch["supervision_mask"] * (output["sensor_valid"] > 0)
                loss = sum(heatmap_focal_loss(output[name + "_logit"], batch[name + "_target"],
                    valid * batch[name + "_loss_mask"], batch[name + "_positive_count"],
                    batch[name + "_background_count"]) for name in ("onset", "offset"))
                loss = loss + (output["state_logit"].float().square() * valid).sum() / valid.sum().clamp_min(1)
            loss.backward()
            finite = all(torch.isfinite(value).all() for value in output.values() if isinstance(value, torch.Tensor))
            finite &= all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
            if not finite or not torch.isfinite(loss):
                raise RuntimeError("Real sensor smoke finite gate failed")
            reports.append({"subject": batch["subject_key"], "session": batch["session_id"],
                "augmentation": batch["augmentation_description"], "loss": float(loss.detach()),
                "valid_steps": int(valid.sum()), "finite": True, **enforce_resources(settings, torch.device(arguments.device))})
    finally:
        release_session_loader(loader)
    report = {"passed": True, "scope": "bounded_real_sensor_smoke", "promotable": False,
        "formal_training_performed": False, "device": arguments.device, "workers": settings["training"]["num_workers"],
        "prefetch": settings["training"]["loader_prefetch_factor"],
        "reader_cache": settings["training"]["session_reader_cache_size"],
        "normalization_scope": "two_natural_training_subject_sessions_for_smoke_only",
        "preprocessing_sha256": digest(preprocessing), "data_identity": inputs.identity,
        "config_sha256": config_identity(config), "source": source_identity(PROJECT_ROOT),
        "elapsed_seconds": time.monotonic() - started, "chunks": reports}
    write_json_atomic(PROJECT_ROOT / arguments.output, report)
    print(json.dumps({"passed": True, "chunks": len(reports), "workers": report["workers"],
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
