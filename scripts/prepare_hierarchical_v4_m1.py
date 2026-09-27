from __future__ import annotations

import argparse
import json
from copy import deepcopy

import _bootstrap  # noqa: F401
import yaml

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_artifacts import (
    validate_run_name,
    write_json_atomic,
    write_yaml_atomic,
)
from bme_eating.hierarchical_v4_gates import evaluate_ppg_promotion, verify_gate_evidence
from bme_eating.v4_protocol import validate_r3_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build R3-M1 from the promoted state architecture")
    parser.add_argument("--config", default="configs/hierarchical_v4_r3_m1.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--motion-run", required=True)
    parser.add_argument("--ppg-run", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    validate_run_name(args.run_name)
    template = load_config(args.config)
    _, _, output_root = resolve_artifact_roots(template)
    decision = evaluate_ppg_promotion(
        output_root,
        s2_run=args.motion_run,
        s3_run=args.ppg_run,
        gate=template["promotion_gate"],
    )
    verify_gate_evidence(output_root.parent, decision)
    selected_run = args.ppg_run if decision["resolved_use_ppg"] else args.motion_run
    selected_path = output_root / "experiments" / selected_run / "fold_0" / "resolved_config.yaml"
    selected = yaml.safe_load(selected_path.read_text(encoding="utf-8")) or {}
    resolved = deepcopy(selected)
    resolved["experiment"]["name"] = str(template["experiment"]["name"])
    resolved["experiment"]["ablation_id"] = "R3-M1"
    resolved["experiment"]["state_promotion"] = decision
    resolved["decoder"]["use_semi_markov"] = True
    validate_r3_config(resolved)
    experiment_root = output_root / "experiments" / args.run_name
    decision_path = experiment_root / "state_promotion.json"
    config_path = experiment_root / "m1_resolved_config.yaml"
    if args.resume:
        if not decision_path.is_file() or not config_path.is_file():
            raise FileNotFoundError("Resolved R3-M1 config does not exist; start with --fresh")
        if json.loads(decision_path.read_text(encoding="utf-8")) != decision:
            raise RuntimeError("R3-M1 promotion evidence changed after creation")
        if load_config(config_path) != {**resolved, "_config_path": str(config_path.resolve())}:
            raise RuntimeError("Resolved R3-M1 config changed after creation")
    else:
        if decision_path.exists() or config_path.exists():
            raise FileExistsError("Resolved R3-M1 config already exists; use --resume")
        experiment_root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(decision_path, decision)
        write_yaml_atomic(config_path, resolved)
    print(config_path)


if __name__ == "__main__":
    main()
