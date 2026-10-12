from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bme_eating.config import _load_project_dotenv, load_config, require_environment_path
from bme_eating.data.joint_sessions import JointInputs
from bme_eating.hierarchical_artifacts import write_json_atomic
from bme_eating.joint_protocol import configure_numerics, digest, require_idle_gpu, source_identity
from bme_eating.joint_v413_training import ContributionBank
from bme_eating.metrics import partition_evaluation_events
from bme_eating.training.hierarchical_v4_trainer import _selector_split, stacking_partitions
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
    arguments = parser.parse_args()
    config = load_config(PROJECT_ROOT / "configs/statsfusion_v413_integrated_repair.yaml")
    config["training"]["device"] = arguments.device
    if arguments.workers is not None:
        config["training"]["num_workers"] = arguments.workers
    require_idle_gpu(arguments.device)
    configure_numerics(config)
    _load_project_dotenv(config)
    output_base = require_environment_path(config, "output_root_env")
    inputs = JointInputs.audit(output_base / "v2", output_base / "v4")
    outer = {subject for subject, fold in inputs.subject_folds.items() if fold == 0}
    legal = set(inputs.subject_folds) - outer
    partitions = stacking_partitions(legal, 3, 2026)
    internal = {subject for subject, partition in partitions.items() if partition == 0}
    training = legal - internal
    anchors, _, events, _ = inputs.frames(training, labels=True)
    fit, selector, _ = _selector_split(training, config["training"]["selector_fraction"],
                                     2026, anchors=anchors, events=events)
    anchors, _, events, _ = inputs.frames(fit, labels=True)
    truth, _ = partition_evaluation_events(events, fit)
    lengths = anchors.groupby(["subject_key", "session_id"]).size()
    positive_keys = set(truth[["subject_key", "session_id"]].astype(str).itertuples(index=False, name=None))
    positives = sorted(positive_keys, key=lambda key: (lengths[key], key))[:3]
    negatives = sorted(set(lengths.index) - positive_keys, key=lambda key: (lengths[key], key))[:2]
    keys = set(positives + negatives)
    if len(keys) != 5:
        raise RuntimeError("Production audit requires five real sessions")
    subjects = {key[0] for key in keys}
    excluded = outer | internal | selector
    if subjects & excluded:
        raise RuntimeError("Production audit subject exclusion failed")
    inputs = SelectedSessions(inputs, keys)
    directory = Path(tempfile.mkdtemp(prefix="bme_v413_production_loop_"))
    prefixes = ("candidate_head.event.", "candidate_head.quality.", "boundary_head.")
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
                          epochs=3, resume=False, outer_excluded=excluded)
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
    required_learning = [name for name in baseline if name != "boundary_head.head.2.bias"]
    report = {"disposable_audit": True, "weights_enter_formal_run": False,
        "source_sha256": digest(source_identity(PROJECT_ROOT)), "data_sha256": digest(inputs.identity),
        "directory": str(directory), "device": arguments.device,
        "workers": config["training"]["num_workers"], "epochs": 3,
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
    report["passed"] = (positive_groups > 0 and negative_groups > 0
        and all(changes[name] > 0 and gradients[name] > 0 for name in required_learning)
        and max(report["gradient_norms"]) < 10_000)
    write_json_atomic(PROJECT_ROOT / f".modeling/v413_production_loop_{arguments.device}.json", report)
    print(json.dumps(report, ensure_ascii=True), flush=True)
    if not report["passed"]:
        raise RuntimeError("Production learning audit failed; inspect head gradients and gradient norms")


if __name__ == "__main__":
    main()
