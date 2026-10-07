from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback

from _bootstrap import PROJECT_ROOT

from bme_eating.config import _load_project_dotenv, load_config, require_environment_path
from bme_eating.data.joint_sessions import JointInputs
from bme_eating.hierarchical_artifacts import write_json_atomic
from bme_eating.joint_protocol import (
    STAGES,
    execution_failure_status,
    initialize_run,
    require_idle_gpu,
    run_lock,
    write_failure,
)
from bme_eating.joint_workflow import (
    audit_reference,
    execute_final,
    execute_gates,
    execute_outer,
    export_joint_bundle,
)


class LogMirror:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()


@contextlib.contextmanager
def run_logging(run):
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with (run / "console.log").open("a", encoding="utf-8", buffering=1) as log:
        sys.stdout, sys.stderr = LogMirror(original_stdout, log), LogMirror(original_stderr, log)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


def main():
    parser = argparse.ArgumentParser(description="StatsFusion v4.10 joint-event protocol")
    parser.add_argument("--config", default="configs/statsfusion_v410_joint_event.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--skip-s0", action="store_true")
    parser.add_argument("--historical-reference")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.fold is not None and arguments.stage not in {"outer", "all"}:
        parser.error("--fold is only valid with --stage outer or all")
    config = load_config(arguments.config)
    _load_project_dotenv(config)
    output_base = require_environment_path(config, "output_root_env")
    input_root = output_base / config["project"]["input_artifact_schema_version"]
    canonical_root = output_base / config["project"]["canonical_schema_version"]
    inputs = JointInputs.audit(input_root, canonical_root)
    root = PROJECT_ROOT
    run = initialize_run(config, root, output_base / "v410", arguments.run_name, inputs.identity,
                         fresh=arguments.fresh, skip_s0=arguments.skip_s0,
                         historical_reference=arguments.historical_reference)
    reference = output_base / "v4" / "experiments" / arguments.historical_reference if arguments.historical_reference else None
    with run_lock(run), run_logging(run):
        write_json_atomic(run / "resolved_config.json", config)
        execute_stages(arguments, config, inputs, root, run, reference)


def execute_stages(arguments, config, inputs, root, run, reference):
    stages = STAGES if arguments.stage == "all" else [arguments.stage]
    status = json.loads((run / "execution_status.json").read_text(encoding="utf-8"))
    for stage in stages:
        if arguments.resume and status["stages"].get(stage) == "COMPLETE" and stage in {"audit", "validate"}:
            continue
        status.update({"status": "RUNNING", "current_stage": stage, "command": sys.argv,
                       "workers": config["training"]["num_workers"],
                       "prefetch": config["training"]["loader_prefetch_factor"],
                       "cache": config["training"]["session_reader_cache_size"]})
        status["stages"][stage] = "RUNNING"
        write_json_atomic(run / "execution_status.json", status)
        try:
            if stage in {"validate", "outer", "final", "replay"}:
                require_idle_gpu(config["training"]["device"])
            if stage == "audit":
                write_json_atomic(run / "input_audit.json", inputs.identity)
                if reference is not None:
                    write_json_atomic(run / "reference_audit.json", audit_reference(reference, inputs))
            elif stage == "validate":
                from bme_eating.joint_validation import validate_joint_implementation
                write_json_atomic(run / "implementation_validation.json", validate_joint_implementation(config))
            elif stage == "outer":
                for fold in [arguments.fold] if arguments.fold is not None else range(5):
                    status["current_fold"] = fold
                    status.setdefault("outer_folds", {})[str(fold)] = "RUNNING"
                    write_json_atomic(run / "execution_status.json", status)
                    execute_outer(inputs, config, run, fold, arguments.resume)
                    status["outer_folds"][str(fold)] = "COMPLETE"
                    write_json_atomic(run / "execution_status.json", status)
            elif stage == "gates":
                manifest = execute_gates(inputs, config, run, reference)
                if not manifest["promotable"]:
                    raise RuntimeError("Five-fold diagnostics complete; joint promotion gates closed")
            elif stage == "final":
                execute_final(inputs, config, run, arguments.resume)
            elif stage == "export":
                export_joint_bundle(config, run, root, resume=arguments.resume)
            elif stage == "replay":
                from bme_eating.joint_validation import replay_joint_bundle
                write_json_atomic(run / "bundle_replay.json", replay_joint_bundle(run / "bundle", config))
            status["stages"][stage] = "COMPLETE"
            if stage == "outer" and not all((run / f"fold_{fold}" / "completion.json").is_file()
                                             for fold in range(5)):
                status["stages"][stage] = "PAUSED"
                write_json_atomic(run / "execution_status.json", status)
                break
        except BaseException as error:
            status["status"] = status["stages"][stage] = execution_failure_status(error)
            if stage == "outer" and "current_fold" in status:
                status.setdefault("outer_folds", {})[str(status["current_fold"])] = status["status"]
            status["error"] = {"type": type(error).__name__, "message": str(error)}
            write_json_atomic(run / "execution_status.json", status)
            write_failure(run, stage, error)
            traceback.print_exc()
            raise
        write_json_atomic(run / "execution_status.json", status)
    status["status"] = ("FAILED" if any(value == "FAILED" for value in status["stages"].values()) else
                        "COMPLETE" if all(status["stages"].get(stage) == "COMPLETE" for stage in STAGES)
                        else "PAUSED")
    write_json_atomic(run / "execution_status.json", status)
    print(json.dumps({"run": str(run), "status": status["status"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
