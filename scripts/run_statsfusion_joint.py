from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from pickle import UnpicklingError

from _bootstrap import PROJECT_ROOT

from bme_eating.config import _load_project_dotenv, load_config, require_environment_path
from bme_eating.data.joint_sessions import JointInputs
from bme_eating.hierarchical_artifacts import write_json_atomic
from bme_eating.joint_capabilities import supported
from bme_eating.joint_protocol import (
    STAGES,
    execution_failure_status,
    execution_folds,
    execution_mode,
    initialize_run,
    require_idle_gpu,
    resolve_reference,
    run_lock,
    stages_for,
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


def main():
    parser = argparse.ArgumentParser(description="StatsFusion joint training, including v4.14 full-data workflow")
    parser.add_argument("--config", default="configs/statsfusion_v414_full_data.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", choices=(*STAGES, "ablation", "dev", "final39", "all"), default="all")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--evaluation-mode", choices=("fold0", "fivefold", "dev28_11", "final39"),
                        help="fold0 saves an independent single-fold diagnostic model; fivefold uses formal gates")
    parser.add_argument("--internal-mode", choices=("single", "oof"),
                        help="single uses partition 0 for independent calibration in fold0 diagnostic mode")
    parser.add_argument("--max-epochs", type=int, help="Epoch ceiling; must match the run's frozen configuration")
    parser.add_argument("--skip-s0", action="store_true")
    parser.add_argument("--historical-reference")
    parser.add_argument("--reference-run")
    parser.add_argument("--migrate-clipping-gate", action="store_true",
                        help="Apply the registered clipping-only migration to the existing failed run")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.reference_run and arguments.historical_reference:
        parser.error("Use one reference identity")
    if arguments.migrate_clipping_gate and not arguments.resume:
        parser.error("--migrate-clipping-gate requires --resume")
    if arguments.fold is not None and arguments.stage not in {"outer", "all"}:
        parser.error("--fold is only valid with --stage outer or all")
    config = load_config(arguments.config)
    if config["model"]["architecture"] != "statsfusion_joint_event_v414":
        parser.error("This isolated v414 runner only writes v414 runs; select the v414 configuration")
    if supported(config, "full_data_workflow"):
        run_v414(arguments, config, parser)
        return
    if arguments.evaluation_mode:
        config.setdefault("execution", {})["mode"] = arguments.evaluation_mode
    if arguments.internal_mode:
        config.setdefault("execution", {})["internal_mode"] = arguments.internal_mode
    if arguments.max_epochs is not None:
        config["training"]["max_epochs"] = arguments.max_epochs
    if execution_mode(config) == "fold0" and arguments.fold not in (None, 0):
        parser.error("fold0 mode can only run fold 0")
    _load_project_dotenv(config)
    output_base = require_environment_path(config, "output_root_env")
    input_root = output_base / config["project"]["input_artifact_schema_version"]
    canonical_root = output_base / config["project"]["canonical_schema_version"]
    inputs = JointInputs.audit(input_root, canonical_root)
    root = PROJECT_ROOT
    run = initialize_run(config, root, output_base / config["project"]["artifact_schema_version"], arguments.run_name, inputs.identity,
                         fresh=arguments.fresh, skip_s0=arguments.skip_s0,
                         historical_reference=arguments.reference_run or arguments.historical_reference,
                         migrate_clipping_gate=arguments.migrate_clipping_gate)
    reference = output_base / "v4" / "experiments" / arguments.historical_reference if arguments.historical_reference else None
    if arguments.reference_run:
        reference = resolve_reference(output_base, arguments.reference_run)
    with run_lock(run), run_logging(run):
        write_json_atomic(run / "resolved_config.json", config)
        execute_stages(arguments, config, inputs, root, run, reference)


def record_v414_failure(run, config, status, stage, error):
    from bme_eating.joint_v414_workflow import (
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
    from bme_eating.joint_v414_bundle import export_bundle
    from bme_eating.joint_v414_split import load_split
    from bme_eating.joint_v414_workflow import (
        audit,
        execute_dev,
        execute_final39,
        save_run_hashes,
        stage_verified,
        write_resource_log,
    )
    if (arguments.evaluation_mode not in {None, "dev28_11", "final39"} or arguments.internal_mode is not None
            or arguments.fold is not None or arguments.migrate_clipping_gate
            or arguments.stage not in {"audit", "dev", "final39", "export", "replay", "all"}):
        parser.error("v414 uses dev28_11/final39 and audit/dev/final39/export/replay stages; no fold/partition migration")
    if arguments.max_epochs is not None and arguments.max_epochs != 64:
        parser.error("v414 retains the registered 64-epoch learning-rate horizon")
    if arguments.evaluation_mode == "final39" and arguments.stage in {"audit", "dev"}:
        parser.error("final39 mode cannot execute development selection")
    _load_project_dotenv(config)
    output = require_environment_path(config, "output_root_env")
    try:
        inputs = JointInputs.audit(output / config["project"]["input_artifact_schema_version"],
                                   output / config["project"]["canonical_schema_version"])
        load_split(config, inputs)
        run = initialize_run(config, PROJECT_ROOT, output / "v414", arguments.run_name, inputs.identity,
            fresh=arguments.fresh, skip_s0=True, historical_reference=arguments.reference_run or arguments.historical_reference)
    except BaseException as error:
        candidate = Path(output) / "v414/runs" / arguments.run_name
        if (arguments.resume and candidate.parent.resolve() == (Path(output) / "v414/runs").resolve()
                and arguments.run_name not in {".", ".."} and (candidate / "identity.json").is_file()
                and (candidate / "execution_status.json").is_file()):
            try:
                with run_lock(candidate):
                    identity = json.loads((candidate / "identity.json").read_text(encoding="utf-8"))
                    if identity.get("protocol") == "statsfusion-joint-event-v5":
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
                    "evaluation_mode": "dev28_11", "phase": stage, "command": sys.argv})
                status["stages"][stage] = "RUNNING"
                status.pop("error", None)
                write_json_atomic(run / "execution_status.json", status)
                if stage in {"dev", "final39", "replay"}:
                    require_idle_gpu(config["training"]["device"])
                if stage == "audit":
                    audit(inputs, config, run)
                elif stage == "dev":
                    from bme_eating.joint_protocol import config_identity, digest, source_identity
                    from bme_eating.joint_validation import validate_joint_implementation
                    preflight = run / "implementation_validation.json"
                    if preflight.exists():
                        report = json.loads(preflight.read_text(encoding="utf-8"))
                        if (report.get("config_sha256") != config_identity(config)
                                or report.get("source_sha256") != digest(source_identity(PROJECT_ROOT))
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
                    from bme_eating.joint_validation import replay_joint_bundle
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


def execute_stages(arguments, config, inputs, root, run, reference):
    stages = stages_for(config) if arguments.stage == "all" else [arguments.stage]
    status = json.loads((run / "execution_status.json").read_text(encoding="utf-8"))
    for stage in stages:
        if arguments.resume and status["stages"].get(stage) == "COMPLETE" and stage in {"audit", "validate"}:
            continue
        status.update({"status": "RUNNING", "current_stage": stage, "command": sys.argv,
                       "evaluation_mode": execution_mode(config),
                       "internal_mode": config.get("execution", {}).get("internal_mode", "oof"),
                       "workers": config["training"]["num_workers"],
                       "prefetch": config["training"]["loader_prefetch_factor"],
                       "cache": config["training"]["session_reader_cache_size"]})
        status["stages"][stage] = "RUNNING"
        status.pop("error", None)
        write_json_atomic(run / "execution_status.json", status)
        try:
            if stage in {"validate", "outer", "ablation", "final", "replay"}:
                require_idle_gpu(config["training"]["device"])
            if stage == "audit":
                write_json_atomic(run / "input_audit.json", inputs.identity)
                if reference is not None:
                    audit = audit_reference(reference, inputs, folds=[0] if
                        supported(config, "consistent_inputs") and execution_mode(config) == "fold0" else None)
                    audit["root"] = str(reference.resolve())
                    audit.pop("prediction_frame", None)
                    audit.pop("exposure_frame", None)
                    write_json_atomic(run / "reference_audit.json", audit)
            elif stage == "validate":
                from bme_eating.joint_validation import validate_joint_implementation
                write_json_atomic(run / "implementation_validation.json", validate_joint_implementation(config, device=config["training"]["device"]))
            elif stage == "ablation":
                from bme_eating.joint_v411_ablation import execute_ablations
                execute_ablations(inputs, config, run, arguments.resume)
            elif stage == "outer":
                for fold in [arguments.fold] if arguments.fold is not None else execution_folds(config):
                    status["current_fold"] = fold
                    status.setdefault("outer_folds", {})[str(fold)] = "RUNNING"
                    write_json_atomic(run / "execution_status.json", status)
                    execute_outer(inputs, config, run, fold, arguments.resume)
                    status["outer_folds"][str(fold)] = "COMPLETE"
                    write_json_atomic(run / "execution_status.json", status)
            elif stage == "gates":
                if execution_mode(config) == "fold0":
                    from bme_eating.joint_single_fold import execute_fold_gate
                    execute_fold_gate(inputs, config, run)
                else:
                    manifest = execute_gates(inputs, config, run, reference)
                    if not manifest["promotable"]:
                        raise RuntimeError("Five-fold diagnostics complete; joint promotion gates closed")
            elif stage == "final":
                if execution_mode(config) == "fold0":
                    from bme_eating.joint_single_fold import save_fold_final
                    save_fold_final(inputs, config, run, resume=arguments.resume)
                else:
                    execute_final(inputs, config, run, arguments.resume)
            elif stage == "export":
                if execution_mode(config) == "fold0":
                    from bme_eating.joint_single_fold import export_fold_bundle
                    export_fold_bundle(inputs, config, run, root, resume=arguments.resume)
                else:
                    export_joint_bundle(config, run, root, resume=arguments.resume)
            elif stage == "replay":
                from bme_eating.joint_validation import replay_joint_bundle
                bundle_name = "fold0_bundle" if execution_mode(config) == "fold0" else "bundle"
                write_json_atomic(run / "bundle_replay.json", replay_joint_bundle(run / bundle_name, config))
            status["stages"][stage] = "COMPLETE"
            if stage == "outer" and not all((run / f"fold_{fold}" / "completion.json").is_file()
                                             for fold in execution_folds(config)):
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
                        "COMPLETE" if all(status["stages"].get(stage) == "COMPLETE" for stage in stages_for(config))
                        else "PAUSED")
    write_json_atomic(run / "execution_status.json", status)
    print(json.dumps({"run": str(run), "status": status["status"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
