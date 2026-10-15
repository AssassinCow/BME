from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from pickle import UnpicklingError

from _bootstrap import PROJECT_ROOT

from bme_eating.artifacts import write_json_atomic
from bme_eating.config import _load_project_dotenv, load_config, require_environment_path
from bme_eating.data.joint_sessions import JointInputs
from bme_eating.protocol import (
    STAGES,
    execution_failure_status,
    initialize_run,
    protocol_for,
    require_idle_gpu,
    run_lock,
    stages_for,
    write_failure,
)


class LogMirror:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def isatty(self):
        return self.stream.isatty()


@contextlib.contextmanager
def run_logging(run):
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with (run / "console.log").open("a", encoding="utf-8", buffering=1) as log:
        sys.stdout, sys.stderr = LogMirror(original_stdout, log), LogMirror(original_stderr, log)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


def record_v414_failure(run, config, status, stage, error):
    from bme_eating.workflow import (
        final39_training_progress,
        read_json,
        save_run_hashes,
        write_resource_log,
    )

    status.update({"status": execution_failure_status(error), "current_stage": stage,
                   "phase": stage, "error": {"type": type(error).__name__, "message": str(error)}})
    status.setdefault("stages", {})[stage] = status["status"]
    reporting_errors = []
    progress = None
    if (run / "final39/model").exists():
        try:
            progress = final39_training_progress(run, config)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, EOFError, UnpicklingError,
                KeyboardInterrupt, SystemExit) as reporting_error:
            reporting_errors.append({"operation": "training_progress", "type": type(reporting_error).__name__,
                                     "message": str(reporting_error)})
        except BaseException as reporting_error:
            raise error from reporting_error

    def save_report():
        try:
            report = read_json(run / "failure_report.json")
        except (OSError, ValueError):
            report = {}
        report.update({"stage": stage, **status["error"], "error_type": type(error).__name__,
                       "status": status["status"], "promotable": False, "final_model_complete": False})
        if progress is not None:
            report["final39_training_progress"] = progress
        if reporting_errors:
            report["reporting_errors"] = reporting_errors
        write_json_atomic(run / "failure_report.json", report)
        write_json_atomic(run / "failures" / f"{stage}.json", report)

    for operation, writer in (
            ("execution_status", lambda: write_json_atomic(run / "execution_status.json", status)),
            ("failure_details", lambda: write_failure(run, stage, error)),
            ("failure_report", save_report), ("resource_log", lambda: write_resource_log(run)),
            ("run_hashes", lambda: save_run_hashes(run))):
        try:
            writer()
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, EOFError, UnpicklingError,
                KeyboardInterrupt, SystemExit) as reporting_error:
            reporting_errors.append({"operation": operation, "type": type(reporting_error).__name__,
                                     "message": str(reporting_error)})
        except BaseException as reporting_error:
            raise error from reporting_error
    if reporting_errors:
        print(json.dumps({"original_error": status["error"], "reporting_errors": reporting_errors},
                         ensure_ascii=False), file=sys.stderr, flush=True)


def run_v414(arguments, config, parser):
    from bme_eating.bundle import export_bundle
    from bme_eating.development_split import load_split
    from bme_eating.workflow import (
        audit,
        execute_dev,
        execute_final39,
        save_run_hashes,
        stage_verified,
        write_resource_log,
    )
    if arguments.evaluation_mode == "final39" and arguments.stage in {"audit", "dev"}:
        parser.error("final39 mode cannot execute development selection")
    _load_project_dotenv(config)
    output = require_environment_path(config, "output_root_env")
    try:
        inputs = JointInputs.audit(output / config["project"]["input_artifact_schema_version"],
                                   output / config["project"]["canonical_schema_version"])
        load_split(config, inputs)
        run = initialize_run(config, PROJECT_ROOT, output / "v414", arguments.run_name, inputs.identity,
            fresh=arguments.fresh, skip_s0=True, historical_reference=arguments.reference_run)
    except BaseException as error:
        candidate = Path(output) / "v414/runs" / arguments.run_name
        if (arguments.resume and candidate.parent.resolve() == (Path(output) / "v414/runs").resolve()
                and arguments.run_name not in {".", ".."} and (candidate / "identity.json").is_file()
                and (candidate / "execution_status.json").is_file()):
            try:
                with run_lock(candidate):
                    identity = json.loads((candidate / "identity.json").read_text(encoding="utf-8"))
                    if identity.get("protocol") == protocol_for(config):
                        status = json.loads((candidate / "execution_status.json").read_text(encoding="utf-8"))
                        record_v414_failure(candidate, config, status, "audit", error)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError, EOFError, UnpicklingError,
                    KeyboardInterrupt, SystemExit) as reporting_error:
                print(f"Failed to record v414 audit failure: {reporting_error}", file=sys.stderr, flush=True)
            except BaseException as reporting_error:
                raise error from reporting_error
        raise
    stages = list(stages_for(config)) if arguments.stage == "all" else [arguments.stage]
    if arguments.evaluation_mode == "final39" and arguments.stage == "all":
        stages = ["final39", "export", "replay"]
    with run_lock(run), run_logging(run):
        status = json.loads((run / "execution_status.json").read_text(encoding="utf-8"))
        stage = "audit"
        try:
            previous = status.get("current_stage")
            if arguments.resume and previous and status["stages"].get(previous) in {"RUNNING", "PAUSED", "FAILED"} and previous not in stages:
                stage = previous
                raise RuntimeError("Resume must finish the interrupted v414 phase before switching stages")
            audit(inputs, config, run)
            for stage in stages:
                if (arguments.resume and status["stages"].get(stage) == "COMPLETE"
                        and stage_verified(run, config, stage)):
                    continue
                status.update({"status": "RUNNING", "current_stage": stage,
                    "evaluation_mode": arguments.evaluation_mode, "phase": stage, "command": sys.argv})
                status["stages"][stage] = "RUNNING"
                status.pop("error", None)
                write_json_atomic(run / "execution_status.json", status)
                if stage in {"dev", "final39", "replay"}:
                    require_idle_gpu(config["training"]["device"])
                if stage == "audit":
                    audit(inputs, config, run)
                elif stage == "dev":
                    from bme_eating.protocol import (
                        config_identity,
                        digest,
                        training_source_identity,
                    )
                    from bme_eating.validation import validate_joint_implementation
                    preflight = run / "implementation_validation.json"
                    if preflight.exists():
                        report = json.loads(preflight.read_text(encoding="utf-8"))
                        if (report.get("config_sha256") != config_identity(config)
                                or report.get("source_sha256") != digest(training_source_identity(PROJECT_ROOT, config))
                                or report.get("device") != config["training"]["device"]):
                            raise RuntimeError("v414 implementation preflight identity differs")
                    else:
                        write_json_atomic(preflight, validate_joint_implementation(config, device=config["training"]["device"]))
                    execute_dev(inputs, config, run, resume=arguments.resume)
                elif stage == "final39":
                    execute_final39(inputs, config, run, resume=arguments.resume)
                elif stage == "export":
                    export_bundle(config, run, PROJECT_ROOT, resume=arguments.resume)
                elif stage == "replay":
                    from bme_eating.validation import replay_joint_bundle
                    write_json_atomic(run / "bundle_replay.json", replay_joint_bundle(run / "bundle", config,
                                                                                     device=config["training"]["device"]))
                status["stages"][stage] = "COMPLETE"
                write_json_atomic(run / "execution_status.json", status)
                write_resource_log(run)
            status["status"] = "COMPLETE" if all(status["stages"].get(stage) == "COMPLETE" for stage in stages_for(config)) else "PAUSED"
            write_json_atomic(run / "execution_status.json", status)
            save_run_hashes(run)
        except BaseException as error:
            record_v414_failure(run, config, status, stage, error)
            raise
        print(json.dumps({"run": str(run), "status": status["status"], "promotable": False}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="StatsFusion v4.14 development and final-data training")
    parser.add_argument("--config", default="configs/statsfusion_v414_full_data.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--evaluation-mode", choices=("dev28_11", "final39"), default="dev28_11")
    parser.add_argument("--skip-s0", action="store_true")
    parser.add_argument("--reference-run")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if config["model"]["architecture"] != "statsfusion_joint_event_v414":
        parser.error("Only the v4.14 configuration is supported")
    run_v414(arguments, config, parser)


if __name__ == "__main__":
    main()
