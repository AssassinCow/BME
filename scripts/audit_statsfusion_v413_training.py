from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

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
from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_protocol import configure_numerics, require_idle_gpu
from bme_eating.joint_stream import JointCandidate, candidate_targets, stable_id
from bme_eating.joint_v411_scoring import boundary_loss, score_candidates
from bme_eating.joint_v413_replay import replay_session
from bme_eating.models.statsfusion_joint_event import quality_focal_loss
from bme_eating.training.statsfusion_joint_trainer import (
    enforce_resources,
    pack_features,
    seed_model,
)


def main():
    parser = argparse.ArgumentParser(description="Disposable real-input v413 training audit")
    parser.add_argument("--config", default="configs/statsfusion_v413_integrated_repair.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", default=".modeling/v413_real_training_audit.json")
    arguments = parser.parse_args()
    config = load_config(PROJECT_ROOT / arguments.config)
    config["training"].update({"device": arguments.device, "num_workers": arguments.workers})
    require_idle_gpu(arguments.device)
    configure_numerics(config)
    _load_project_dotenv(config)
    output_base = require_environment_path(config, "output_root_env")
    inputs = JointInputs.audit(output_base / "v2", output_base / "v4")
    legal = {subject for subject, fold in inputs.subject_folds.items() if fold != 0}
    _, _, events, _ = inputs.frames(legal, labels=True)
    from bme_eating.metrics import partition_evaluation_events
    truth, _ = partition_evaluation_events(events, legal)
    event = truth.sort_values(["end_ms", "start_ms"]).iloc[0]
    subjects = {str(event.subject_key)}
    key = (str(event.subject_key), str(event.session_id))
    dataset, preprocessing = make_joint_dataset(inputs, subjects, config, training=True)
    natural, _ = make_joint_dataset(inputs, subjects, config, preprocessing, training=True)
    natural.training = False
    sampler = SessionChunkSampler(dataset, 3, training=False)
    group = dataset.session_groups[key]
    stop = min(len(group), int(np.searchsorted(group.timestamp_ms.to_numpy(), int(event.end_ms) + 30_000)) + 1)
    first = max(0, stop - 512)
    sampler.chunks = []
    for begin in range(first, stop, 256):
        finish = min(begin + 256, stop)
        importance = np.zeros(256, np.float32)
        importance[-(finish - begin):] = 1
        sampler.chunks.append((int(group["_row_id"].iloc[finish - 1]), 3, importance,
                               key, finish == stop, finish - begin))
    model = seed_model(config).to(arguments.device).train()
    disposable = Path(tempfile.gettempdir()) / "bme_v413_validation" / "audit" / arguments.device
    rng = torch.get_rng_state().clone()
    replay, manifest = replay_session(model, inputs, config, preprocessing, natural, sampler, key,
        disposable, "disposable_real_audit", 3, 0)
    if not torch.equal(rng, torch.get_rng_state()) or not model.training:
        raise RuntimeError("Natural replay changed training RNG or mode")
    loader = session_loader(dataset, sampler, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["head_learning_rate"])
    prior_times, prior_features = [], []
    reports = []
    try:
        for index, batch in enumerate(loader):
            batch = {name: value.to(arguments.device) if isinstance(value, torch.Tensor) else value
                     for name, value in batch.items()}
            steps = sampler.chunks[index][5]
            mask = torch.zeros_like(batch["supervision_mask"][0], dtype=torch.bool)
            mask[-steps:] = True
            with torch.autocast(device_type=arguments.device, dtype=torch.bfloat16,
                                enabled=arguments.device == "cuda"):
                outputs = model(batch)
                features = pack_features(outputs, mask)
                times = batch["timestamp_ms"][0, mask].cpu()
                full_times = torch.cat([*prior_times, times])
                full_features = torch.cat([*prior_features, features.cpu()])
                support = (full_times <= int(event.end_ms) + 30_000)
                decision = int(full_times[-1])
                candidate = JointCandidate(*key, int(event.start_ms), min(int(event.end_ms), decision),
                    decision, "state", 3, float(features[:, 129].mean().detach()),
                    int(event.start_ms), min(int(event.end_ms), decision), "audit", stable_id(key, decision),
                    "audit", "audit", "disposable_real_audit", {}, decision)
                if candidate.end_ms <= candidate.start_ms or decision - candidate.end_ms > 60_000 or not support.any():
                    prior_times.append(times)
                    prior_features.append(features.detach().cpu())
                    continue
                score = score_candidates(model, [candidate], full_times, full_features, arguments.device)[0]
                target = candidate_targets(candidate, truth)
                losses = {"event": torch.nn.functional.binary_cross_entropy_with_logits(
                    score["event_logit"].float(), torch.tensor([target[0]], device=arguments.device)),
                    "quality": quality_focal_loss(score["quality_logit"].float(),
                    torch.tensor([target[1]], device=arguments.device)).mean(),
                    "boundary": boundary_loss(model, candidate, target, full_times, full_features,
                    arguments.device, {"start_range_seconds": 900, "end_range_seconds": 30},
                    score["candidate_embedding"][0])}
            norms = {}
            parameter = next(model.encoder.acc_encoder.parameters())
            for name, loss in losses.items():
                gradient = torch.autograd.grad(loss, parameter, retain_graph=True, allow_unused=True)[0]
                norms[name] = float(gradient.float().norm()) if gradient is not None else 0.0
            sum(losses.values()).backward()
            prior_times.append(times)
            prior_features.append(features.detach().cpu())
            reports.append({"chunk": index, "target": list(target), "shared_gradient_norms": norms,
                "augmentation": batch["augmentation_description"], "resources": enforce_resources(config, torch.device(arguments.device))})
    finally:
        release_session_loader(loader)
    if not reports or any(not np.isfinite(value) for report in reports for value in report["shared_gradient_norms"].values()):
        raise RuntimeError("Real-input audit has no finite supervised candidate")
    before = model.candidate_head.event.weight.detach().clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"], error_if_nonfinite=True)
    optimizer.step()
    changed = float((model.candidate_head.event.weight - before).detach().abs().max())
    if changed <= 0:
        raise RuntimeError("Real-input audit optimizer did not update event head")
    from bme_eating.training.statsfusion_joint_trainer import save_torch_atomic
    save_torch_atomic(disposable / "audit_checkpoint.pt", {"model": model.state_dict(),
        "preprocessing": preprocessing, "training_subjects": sorted(subjects),
        "disposable_audit": True, "weights_enter_formal_run": False})
    write_json_atomic(Path(arguments.output), {"disposable_audit": True, "weights_enter_formal_run": False,
        "disposable_root": str(disposable),
        "checkpoint_sha256": sha256_file(disposable / "audit_checkpoint.pt"),
        "subjects": sorted(subjects), "outer_excluded": True, "workers": arguments.workers,
        "device": arguments.device, "model_update_max": changed, "chunks": reports,
        "natural_replay": {"binding": manifest["binding"], "sha256": manifest["sha256"],
            "candidates": len(replay["candidates"]), "seconds": replay["seconds"]}})
    print(f"Real training audit passed: {len(reports)} supervised chunks; optimizer delta={changed:.8g}", flush=True)


if __name__ == "__main__":
    main()
