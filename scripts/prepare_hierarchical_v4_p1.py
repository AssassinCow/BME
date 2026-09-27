from __future__ import annotations

import argparse
import json
from copy import deepcopy

import _bootstrap  # noqa: F401
import yaml

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_artifacts import (
    sha256_file,
    validate_run_name,
    write_json_atomic,
    write_yaml_atomic,
)
from bme_eating.hierarchical_v4_gates import verify_gate_evidence
from bme_eating.v4_protocol import validate_r3_config

MOTION_ABLATIONS = {
    "R3-S0",
    "R3-S1",
    "R3-S2",
    "R3-D1",
    "R3-D2a",
    "R3-D2b",
    "R3-D2c",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build R3-P1 from the hash-locked motion winner")
    parser.add_argument("--config", default="configs/hierarchical_v4_r3_p1.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--motion-run", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    validate_run_name(args.run_name)
    template = load_config(args.config)
    _, _, output_root = resolve_artifact_roots(template)
    motion_root = output_root / "experiments" / args.motion_run
    motion_config_path = motion_root / "fold_0" / "resolved_config.yaml"
    gate_path = motion_root / "ablation" / "fold_0_report.json"
    if not motion_config_path.is_file() or not gate_path.is_file():
        raise FileNotFoundError("Motion winner config and fold-0 gate report are required")
    gate_report = json.loads(gate_path.read_text(encoding="utf-8"))
    if not bool(gate_report.get("passed", False)) or gate_report.get("selected_run") != args.motion_run:
        raise RuntimeError("R3-P1 parent must be the passing fold-0 selected motion run")
    verify_gate_evidence(output_root.parent, gate_report)
    motion = yaml.safe_load(motion_config_path.read_text(encoding="utf-8")) or {}
    validate_r3_config(motion)
    ablation = str(motion.get("experiment", {}).get("ablation_id", ""))
    if ablation not in MOTION_ABLATIONS or bool(motion.get("model", {}).get("use_ppg", False)):
        raise RuntimeError("R3-P1 parent must be a PPG-free r3 motion ablation")
    resolved = deepcopy(motion)
    resolved["experiment"]["name"] = str(template["experiment"]["name"])
    resolved["experiment"]["ablation_id"] = "R3-P1"
    resolved["experiment"]["motion_parent"] = {
        "run_name": args.motion_run,
        "resolved_config_sha256": sha256_file(motion_config_path),
        "fold0_gate_sha256": sha256_file(gate_path),
    }
    resolved["model"]["use_ppg"] = True
    validate_r3_config(resolved)
    experiment_root = output_root / "experiments" / args.run_name
    config_path = experiment_root / "p1_resolved_config.yaml"
    parent_path = experiment_root / "motion_parent.json"
    parent = resolved["experiment"]["motion_parent"]
    if args.resume:
        if not config_path.is_file() or not parent_path.is_file():
            raise FileNotFoundError("Resolved R3-P1 config does not exist; start with --fresh")
        if load_config(config_path) != {**resolved, "_config_path": str(config_path.resolve())}:
            raise RuntimeError("Resolved R3-P1 config changed after creation")
        if json.loads(parent_path.read_text(encoding="utf-8")) != parent:
            raise RuntimeError("R3-P1 motion-parent evidence changed after creation")
    else:
        if config_path.exists() or parent_path.exists():
            raise FileExistsError("Resolved R3-P1 config already exists; use --resume")
        experiment_root.mkdir(parents=True, exist_ok=True)
        write_yaml_atomic(config_path, resolved)
        write_json_atomic(parent_path, parent)
    print(config_path)


if __name__ == "__main__":
    main()
