from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_artifacts import (
    validate_run_name,
    write_json_atomic,
    write_yaml_atomic,
)
from bme_eating.hierarchical_v4_gates import evaluate_ppg_promotion, verify_gate_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve the statsfusion-r2 S4 PPG branch")
    parser.add_argument("--config", default="configs/hierarchical_v4_statsfusion.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--s2-run", required=True)
    parser.add_argument("--s3-run", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    validate_run_name(args.run_name)
    config = load_config(args.config)
    if config.get("experiment", {}).get("protocol_version") != "statsfusion-r2":
        raise RuntimeError("S4 preparation requires statsfusion-r2")
    if str(config.get("experiment", {}).get("ablation_id")) != "S4":
        raise RuntimeError("S4 preparation requires an S4 base config")
    _, _, output_root = resolve_artifact_roots(config)
    experiment_root = output_root / "experiments" / args.run_name
    decision_path = experiment_root / "ppg_promotion.json"
    resolved_path = experiment_root / "s4_resolved_config.yaml"
    decision = evaluate_ppg_promotion(
        output_root,
        s2_run=args.s2_run,
        s3_run=args.s3_run,
        gate=config["promotion_gate"],
    )
    verify_gate_evidence(output_root.parent, decision)
    config["model"]["use_ppg"] = bool(decision["resolved_use_ppg"])
    config["experiment"]["ppg_promotion"] = decision
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    if args.resume:
        if not decision_path.is_file() or not resolved_path.is_file():
            raise FileNotFoundError("Resolved S4 config does not exist; start with --fresh")
        existing = json.loads(decision_path.read_text(encoding="utf-8"))
        if existing != decision:
            raise RuntimeError("S2/S3 PPG evidence changed after S4 resolution")
        if load_config(resolved_path) != {**public_config, "_config_path": str(resolved_path.resolve())}:
            raise RuntimeError("Resolved S4 config changed after creation")
    else:
        if decision_path.exists() or resolved_path.exists():
            raise FileExistsError("Resolved S4 config already exists; use --resume")
        experiment_root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(decision_path, decision)
        write_yaml_atomic(resolved_path, public_config)
    print(resolved_path)


if __name__ == "__main__":
    main()
