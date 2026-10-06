from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_artifacts import sha256_file, validate_run_name, write_json_atomic
from bme_eating.hierarchical_v4_artifacts import _tracked_inputs, feature_provenance_artifact_path
from bme_eating.integrated_v49 import (
    is_v49,
    run_cli_with_failure_report,
    validate_sensor_only_reference,
)
from bme_eating.stats_features import audit_feature_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete v4.9 integrated repair protocol")
    parser.add_argument("--config", default="configs/hierarchical_v4_v49_integrated_repair.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--s0-run", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if not is_v49(config):
        raise ValueError("Integrated runner requires the v4.9 repair configuration")
    validate_run_name(args.run_name)
    validate_run_name(args.s0_run)
    if not args.run_name.startswith(str(config["v49"]["run_name_prefix"])) or args.run_name == args.s0_run:
        raise ValueError("v4.9 requires a fresh, separately named repair run")
    _, input_root, output_root = resolve_artifact_roots(config)
    root = output_root / "experiments" / args.run_name
    if args.fresh and root.exists():
        raise FileExistsError("Integrated run already exists; choose a new run name")
    if args.resume and not root.is_dir():
        raise FileNotFoundError("Integrated run does not exist; start with --fresh")
    validate_sensor_only_reference(output_root, args.s0_run)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    project = Path(__file__).resolve().parents[1]

    def execute(script: str, arguments: list[str], label: str) -> None:
        command = [sys.executable, str(project / "scripts" / script), "--config", args.config, *arguments]
        write_json_atomic(root / "execution_status.json", {"stage": label, "command": command, "status": "RUNNING"})
        with (
            (logs / f"{label}.log").open("a", encoding="utf-8") as log,
            subprocess.Popen(command, cwd=project, env=environment, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process,
        ):
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait() != 0:
                raise RuntimeError(f"Integrated v4.9 stopped at {label}; see {log.name}")

    execute("prepare_statsfusion_v4_inputs.py", ["--workers", "8", "--resume"], "prepare")
    validate_sensor_only_reference(
        output_root, args.s0_run,
        expected_input_hashes={name: sha256_file(path) for name, path in _tracked_inputs(config, input_root).items()},
    )
    provenance = audit_feature_provenance(
        project_root=project, input_root=input_root,
        source_paths=config["feature_provenance"]["source_paths"],
        selection_note_path=config["feature_provenance"].get("selection_note_path"),
        assumed_used_all_outer_folds=bool(config["feature_provenance"].get("assumed_used_all_outer_folds", True)),
    )
    provenance_mode = "--resume" if feature_provenance_artifact_path(output_root, provenance).is_file() else "--fresh"
    execute("audit_statsfusion_feature_provenance.py", ["--run-name", args.run_name, "--fold", "0", provenance_mode], "provenance")
    for fold in range(5):
        manifest_path = root / f"fold_{fold}" / "run_manifest.json"
        stage = json.loads(manifest_path.read_text(encoding="utf-8")).get("stage") if manifest_path.is_file() else "MISSING"
        common = ["--run-name", args.run_name, "--fold", str(fold)]
        if stage == "MISSING" or stage == "CREATED":
            execute("train_hierarchical_v4_state.py", [*common, "--fresh" if stage == "MISSING" else "--resume"], f"fold_{fold}_state")
            stage = "STATE_COMPLETE"
        if stage == "STATE_COMPLETE":
            execute("build_event_candidates_v4.py", [*common, "--resume"], f"fold_{fold}_candidates")
            stage = "PROPOSALS_COMPLETE"
        if stage == "PROPOSALS_COMPLETE":
            execute("select_hierarchical_v4_pipeline.py", [*common, "--resume"], f"fold_{fold}_select")
            stage = "SELECTED"
        if stage == "SELECTED":
            execute("evaluate_hierarchical_v4.py", [*common, "--resume"], f"fold_{fold}_evaluate")
            stage = "EVALUATED"
        if stage != "EVALUATED":
            raise RuntimeError(f"Unsupported fold stage: {stage}")
    execute("check_hierarchical_v4_gates.py", ["--mode", "full", "--run-name", args.run_name, "--s0-run", args.s0_run], "outer_gate")
    final_root = output_root / "final" / args.run_name
    final_mode = "--resume" if (final_root / "final_manifest.json").is_file() else "--fresh"
    execute("train_hierarchical_v4_final.py", ["--run-name", args.run_name, final_mode], "final_training")
    execute("export_hierarchical_v4_bundle.py", ["--run-name", args.run_name, "--resume" if (final_root / "model_bundle").exists() else "--fresh"], "export")
    execute("smoke_test_hierarchical_v4.py", [], "smoke")
    execute("replay_hierarchical_v4_raw_session.py", ["--bundle", str(final_root / "model_bundle"), "--forbid-xgboost"], "bundle_replay")
    write_json_atomic(root / "execution_status.json", {"stage": "COMPLETE", "status": "PASSED", "bundle": str(final_root / "model_bundle")})


if __name__ == "__main__":
    run_cli_with_failure_report(main, "integrated_execution")
