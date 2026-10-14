from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from _bootstrap import PROJECT_ROOT

from bme_eating.artifacts import write_json_atomic
from bme_eating.config import _load_project_dotenv, load_config, require_environment_path
from bme_eating.data.joint_sessions import JointInputs
from bme_eating.development_split import load_split
from bme_eating.metrics import partition_evaluation_events
from bme_eating.protocol import (
    PROFILE,
    config_identity,
    configure_numerics,
    digest,
    protocol_for,
    require_idle_gpu,
    source_identity,
)
from bme_eating.training.candidate_supervision import ContributionBank
from bme_eating.training.statsfusion_joint_trainer import seed_model, train_joint_model


class SelectedSessions:
    def __init__(self, inputs, keys):
        self.inputs, self.keys = inputs, keys
        self.identity = {**inputs.identity, "disposable_audit_sessions": sorted(keys)}

    def frames(self, subjects, *, labels):
        frames = self.inputs.frames(subjects, labels=labels)
        return tuple(frame.loc[[tuple(map(str, key)) in self.keys for key in
            frame[["subject_key", "session_id"]].itertuples(index=False, name=None)]].copy()
            for frame in frames)

    def file_signature(self):
        return self.inputs.file_signature()

    def assert_session_unchanged(self, key):
        self.inputs.assert_session_unchanged(key)


def main():
    parser = argparse.ArgumentParser(description="Disposable real production-loop learning audit")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--long-session", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.epochs < 8:
        parser.error("Repair acceptance requires at least eight disposable epochs")
    config = load_config(PROJECT_ROOT / "configs/statsfusion_v414_full_data.yaml")
    config["training"]["device"] = arguments.device
    if arguments.workers is not None:
        config["training"]["num_workers"] = arguments.workers
    require_idle_gpu(arguments.device)
    configure_numerics(config)
    _load_project_dotenv(config)
    output_base = require_environment_path(config, "output_root_env")
    inputs = JointInputs.audit(output_base / "v2", output_base / "v4")
    split = load_split(config, inputs)
    fit = set(split["training_subjects"])
    excluded = set(split["development_subjects"])
    anchors, _, events, _ = inputs.frames(fit, labels=True)
    truth, _ = partition_evaluation_events(events, fit)
    lengths = anchors.groupby(["subject_key", "session_id"]).size()
    positive_keys = set(truth[["subject_key", "session_id"]].astype(str).itertuples(index=False, name=None))
    ordered = sorted(positive_keys, key=lambda key: (lengths[key], key))
    missing = [key for key in ordered if split["subject_statistics"][key[0]]["gyro_all_missing"]]
    positives = [ordered[0], missing[0]] if missing else [ordered[0]]
    if arguments.long_session:
        positives.append(ordered[-1])
    positives = list(dict.fromkeys([*positives, *ordered]))[:3]
    negatives = sorted(set(lengths.index) - positive_keys, key=lambda key: (lengths[key], key))[:2]
    keys = set(positives + negatives)
    if len(keys) != 5:
        raise RuntimeError("Production audit requires five real sessions")
    subjects = {key[0] for key in keys}
    if subjects & excluded:
        raise RuntimeError("Production audit subject exclusion failed")
    inputs = SelectedSessions(inputs, keys)
    directory = Path(tempfile.mkdtemp(prefix="bme_v414_production_loop_"))
    prefixes = ("candidate_head.event.", "candidate_head.quality.", "boundary_head.", "onset.", "offset.")
    baseline = {name: parameter.detach().clone().cpu() for name, parameter in seed_model(config).named_parameters()
                if name.startswith(prefixes)}
    original_commit = ContributionBank.commit
    bank_traces = []

    def traced_commit(bank, model):
        contributions = {task: {name: float(value.norm()) for name, value in gradients.items()
            if name.startswith(prefixes)} for task, gradients in bank.state["pending_task_gradients"].items()}
        original_commit(bank, model)
        bank_traces.append({"bank": contributions, "committed_head_gradients": {
            name: float(parameter.grad.norm()) if parameter.grad is not None else None
            for name, parameter in model.named_parameters() if name.startswith(prefixes)}})

    ContributionBank.commit = traced_commit
    try:
        train_joint_model(inputs, subjects, config, directory, "disposable_production_audit",
                          epochs=arguments.epochs, resume=False, outer_excluded=excluded)
    finally:
        ContributionBank.commit = original_commit
    payload = torch.load(directory / "latest.pt", map_location="cpu", weights_only=True)
    sampling = [json.loads(line)["payload"] for line in
        (directory / "sampling.jsonl").read_text(encoding="utf-8").splitlines()]
    epoch_three = [record for record in sampling if record["epoch"] == 3]
    positive_groups = sum(record["positive_loss_groups"] for record in epoch_three)
    negative_groups = sum(record["negative_loss_groups"] for record in epoch_three)
    changes = {name: float((payload["model"][name] - value).abs().max()) for name, value in baseline.items()}
    history = json.loads((directory / "training_history.json").read_text(encoding="utf-8"))
    gradients = {name: max((trace["committed_head_gradients"].get(name) or 0)
                          for trace in bank_traces) for name in baseline}
    diagnostics = [json.loads(line)["payload"] for line in
        (directory / "gradients.jsonl").read_text(encoding="utf-8").splitlines()]
    epoch_evidence = []
    for epoch in range(1, arguments.epochs + 1):
        records = [record for record in sampling if record["epoch"] == epoch]
        counts = json.loads((directory / "endpoint_counts" / f"epoch_{epoch:03d}.json").read_text(encoding="utf-8"))
        endpoint_totals = {name: {term: sum(record["endpoint_supervision"][name][term] for record in records)
            for term in ("peaks", "background_bins")} for name in ("onset", "offset")}
        expected_totals = {name: {term: sum(session[name][index] for session in counts["sessions"].values())
            for index, term in enumerate(("peaks", "background_bins"))} for name in ("onset", "offset")}
        epoch_evidence.append({"epoch": epoch, "endpoint_count_sha256": counts["sha256"],
            "endpoint_scales": counts["scales"], "actual_endpoint_totals": endpoint_totals,
            "expected_endpoint_totals": expected_totals, "endpoint_counts_match": endpoint_totals == expected_totals,
            "positive_groups": sum(record["positive_loss_groups"] for record in records),
            "negative_groups": sum(record["negative_loss_groups"] for record in records),
            "contrastive_pairs": sum(record["contrastive_pairs"] for record in records),
            "historical_background_bins": sum(record["contrastive_supervision"]["history_background_bins"] for record in records),
            "natural_replay_seconds": sum(record["stage_timing"].get("natural_replay_seconds", 0) for record in records),
            "augmentation_descriptions": {"|".join((record["subject"], record["session"])): record["augmentation"] for record in records},
            "optimizer_updates": sum("actual_update_contributions" in record for record in records),
            "atomic_losses": sum(item["atomic_losses"] for record in records for item in record.get("positive_retention", [])
                if "atomic_losses" in item),
            "vjp_fallback_groups": sum(item.get("batched_vjp_fallback", False) for record in records
                for item in record.get("positive_retention", [])),
            "stage_seconds": {name: sum(record["stage_timing"].get(name, 0) for record in records)
                for name in ("encoder_seconds", "sampling_seconds", "candidate_gradient_seconds", "previous_checkpoint_total_seconds")}})
    required_learning = [name for name in baseline if name != "boundary_head.head.2.bias"]
    report = {"disposable_audit": True, "weights_enter_formal_run": False,
        "protocol": protocol_for(config), "repair_revision": config["experiment"]["repair_revision"],
        "config_sha256": config_identity(config), "epoch_evidence": epoch_evidence,
        "endpoint_gradient_diagnostics": [{name: value for name, value in record.items()
            if name in {"epoch", "chunk"} or name.startswith(("onset_", "offset_"))} for record in diagnostics],
        "source_sha256": digest(source_identity(PROJECT_ROOT)), "data_sha256": digest(inputs.identity),
        "directory": str(directory), "device": arguments.device,
        "workers": config["training"]["num_workers"], "epochs": arguments.epochs,
        "training_subjects": sorted(subjects), "fit_subject_count": len(fit),
        "excluded_subjects": sorted(excluded), "sessions": sorted(keys),
        "session_bins": {"|".join(key): int(lengths[key]) for key in keys},
        "epoch_three_positive_groups": positive_groups, "epoch_three_negative_groups": negative_groups,
        "update_contributions": [record["actual_update_contributions"] for record in epoch_three
                                 if "actual_update_contributions" in record],
        "head_parameter_changes": changes, "maximum_committed_head_gradient_norms": gradients,
        "required_learning_parameters": required_learning,
        "softmax_shift_invariant_parameters": ["boundary_head.head.2.bias"],
        "bank_traces": bank_traces,
        "gradient_norms": [record["maximum_gradient_norm"] for record in history],
        "clipping_fractions": [record["clipping_fraction"] for record in history],
        "maximum_mining_calls": max(record["mining_head_candidates"] for record in sampling),
        "resources": {name: max(record["resources"][name] for record in sampling) for name in
            ("peak_gpu_gb", "peak_gpu_reserved_gb", "peak_process_tree_private_ram_gb")}}
    report["split_sha256"] = config["development"]["split_sha256"]
    report["includes_natural_gyro_missing_session"] = any(split["subject_statistics"][key[0]]["gyro_all_missing"] for key in keys)
    report["maximum_session_seconds"] = max(int(lengths[key]) * 3 for key in keys)
    report["passed"] = (positive_groups > 0 and negative_groups > 0
        and all(changes[name] > 0 and gradients[name] > 0 for name in required_learning)
        and all(record["endpoint_counts_match"] and record["optimizer_updates"] == 2 for record in epoch_evidence)
        and all(record["natural_replay_seconds"] > 0 and record["atomic_losses"] > 0 for record in epoch_evidence if record["epoch"] >= 3)
        and sum(record["contrastive_pairs"] for record in epoch_evidence if record["epoch"] >= 6) > 0
        and report["maximum_mining_calls"] <= 96
        and report["resources"]["peak_gpu_gb"] <= config["resources"]["maximum_gpu_gb"]
        and report["resources"]["peak_process_tree_private_ram_gb"] <= config["resources"]["maximum_private_ram_gb"]
        and max(report["gradient_norms"]) < 10_000)
    repair_tag = PROFILE.repair_revision.replace("-", "_")
    write_json_atomic(PROJECT_ROOT / f"manifests/{repair_tag}_production_loop_{arguments.device}.json", report)
    print(json.dumps({name: report[name] for name in ("passed", "directory", "device", "sessions", "epoch_three_positive_groups", "epoch_three_negative_groups", "maximum_mining_calls", "clipping_fractions")}, ensure_ascii=True), flush=True)
    if not report["passed"]:
        raise RuntimeError("Production learning audit failed; inspect head gradients and gradient norms")


if __name__ == "__main__":
    main()
