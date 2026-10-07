from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_artifacts import sha256_file, validate_run_name, write_json_atomic
from bme_eating.hierarchical_v4_artifacts import (
    _tracked_inputs,
    feature_provenance_artifact_path,
    resume_config_hash,
)
from bme_eating.integrated_v49 import (
    is_v49,
    run_cli_with_failure_report,
    validate_sensor_only_reference,
)
from bme_eating.stats_features import audit_feature_provenance
from bme_eating.v4_protocol import execution_environment_identity, execution_source_identity


def _process_tree_memory(pid: int) -> int | None:
    if os.name != "nt":
        return None
    from ctypes import wintypes

    class ProcessEntry(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD), ("pid", wintypes.DWORD),
                    ("heap", ctypes.c_size_t), ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                    ("parent", wintypes.DWORD), ("priority", wintypes.LONG), ("flags", wintypes.DWORD),
                    ("exe", wintypes.WCHAR * 260)]

    class MemoryCounters(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD),
                    *[(name, ctypes.c_size_t) for name in ("peak_working", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == wintypes.HANDLE(-1).value:
        return None
    entry = ProcessEntry()
    entry.size = ctypes.sizeof(entry)
    parents = {}
    try:
        available = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while available:
            parents[int(entry.pid)] = int(entry.parent)
            available = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    selected = {pid}
    while True:
        descendants = {child for child, parent in parents.items() if parent in selected}
        if descendants <= selected:
            break
        selected.update(descendants)
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    total = 0
    for child in selected:
        handle = kernel.OpenProcess(0x410, False, child)
        if handle:
            counters = MemoryCounters()
            counters.size = ctypes.sizeof(counters)
            try:
                if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.size):
                    total += int(counters.working)
            finally:
                kernel.CloseHandle(handle)
    return total


def _monitor_resources(pid: int, stop: threading.Event, resources: dict) -> None:
    while not stop.is_set():
        try:
            memory = _process_tree_memory(pid)
            if memory is not None:
                resources["peak_process_tree_working_set_bytes"] = max(resources.get("peak_process_tree_working_set_bytes", 0), memory)
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=3, check=False)
            values = [int(value.strip()) for value in gpu.stdout.splitlines() if value.strip().isdigit()]
            if values:
                resources["peak_device_memory_mib"] = max(resources.get("peak_device_memory_mib", 0), max(values))
            resources["samples"] = resources.get("samples", 0) + 1
        except (OSError, subprocess.TimeoutExpired, ValueError):
            resources["probe_unavailable"] = True
        stop.wait(3)


def execute_stage(root: Path, project: Path, config_path: str, environment: dict[str, str],
                  script: str, arguments: list[str], label: str, runtime: dict) -> None:
    command = [sys.executable, str(project / "scripts" / script), "--config", config_path, *arguments]
    status = {"stage": label, "status": "RUNNING", "run_name": root.name,
              "command": command, "runtime_parameters": runtime, "started_at": time.time()}
    write_json_atomic(root / "execution_status.json", status)
    log_path = root / "logs" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = None
    monitor = None
    stop = threading.Event()
    resources = {"gpu_memory_scope": "device_total_including_other_processes", "sampling_interval_seconds": 3}
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(status, ensure_ascii=False) + "\n")
            process = subprocess.Popen(command, cwd=project, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            monitor = threading.Thread(target=_monitor_resources, args=(process.pid, stop, resources), daemon=True)
            monitor.start()
            for line in process.stdout:
                if any(marker in line.lower() for marker in ("out of memory", "memoryerror", "bad allocation")):
                    status["failure_kind"] = "OOM"
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return_code = process.wait()
            status["return_code"] = return_code
            if return_code in {130, -2, -1073741510}:
                raise KeyboardInterrupt("Child process interrupted")
            if return_code != 0:
                raise RuntimeError(f"Integrated v4.9 stopped at {label}; see {log_path}")
        status["status"] = "RUNNING"
        status["stage_completed"] = True
    except BaseException as error:
        if process is not None and process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
            else:
                process.terminate()
            process.wait()
        status.update({"status": "PAUSED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                       "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=4)
        status["finished_at"] = time.time()
        status["resources"] = resources
        write_json_atomic(root / "execution_status.json", status)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(status, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete v4.9 integrated repair protocol")
    parser.add_argument("--config", default="configs/hierarchical_v4_v49_integrated_optimized.yaml")
    parser.add_argument("--run-name", required=True)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--s0-run")
    reference.add_argument("--skip-s0", action="store_true", help="Record S0 as unavailable and retain all frozen-Deep gates")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    from bme_eating.v4_protocol import validate_r3_config

    validate_r3_config(config)
    if not is_v49(config):
        raise ValueError("Integrated runner requires the v4.9 repair configuration")
    validate_run_name(args.run_name)
    if args.s0_run is not None:
        validate_run_name(args.s0_run)
    if not args.run_name.startswith(str(config["v49"]["run_name_prefix"])) or args.run_name == args.s0_run:
        raise ValueError("v4.9 requires a fresh, separately named repair run")
    _, input_root, output_root = resolve_artifact_roots(config)
    root = output_root / "experiments" / args.run_name
    if args.fresh and root.exists():
        raise FileExistsError("Integrated run already exists; choose a new run name")
    if args.resume and not root.is_dir():
        raise FileNotFoundError("Integrated run does not exist; start with --fresh")
    if not args.skip_s0:
        validate_sensor_only_reference(output_root, args.s0_run)
    protocol = {"run_name": args.run_name, "s0_run": args.s0_run, "skip_s0": args.skip_s0}
    protocol_path = root / "execution_protocol.json"
    if args.resume:
        if not protocol_path.is_file() or json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Integrated resume cannot change its S0 comparison policy; choose a new run")
    else:
        write_json_atomic(protocol_path, protocol)
    project = Path(__file__).resolve().parents[1]
    execution_identity = {
        "config_sha256": resume_config_hash(config),
        "source": execution_source_identity(project),
        "environment": execution_environment_identity(),
    }
    identity_path = root / "execution_identity.json"
    if config.get("v49", {}).get("protocol") == "integrated_repair_v2":
        if args.resume:
            if not identity_path.is_file() or json.loads(identity_path.read_text(encoding="utf-8")) != execution_identity:
                raise RuntimeError("v4.9 runtime/config/source identity changed; choose a new run")
        else:
            write_json_atomic(identity_path, execution_identity)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    runtime = {key: value for key, value in config["training"].items()
               if "worker" in key or "prefetch" in key or "cache" in key}

    def write_status(label: str, status: str, **extra: object) -> None:
        write_json_atomic(root / "execution_status.json", {
            "stage": label, "status": status, "run_name": args.run_name,
            "execution_identity": execution_identity, "runtime_parameters": runtime, **extra,
        })

    def execute(script: str, arguments: list[str], label: str) -> None:
        execute_stage(root, project, args.config, environment, script, arguments, label, runtime)

    write_status("initialization", "CREATED" if args.fresh else "RUNNING")
    workers = str(config["training"].get("preprocess_num_workers", config["training"]["num_workers"]))
    execute("prepare_statsfusion_v4_inputs.py", ["--workers", workers, "--resume"], "prepare")
    if not args.skip_s0:
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
    reference_arguments = ["--skip-s0"] if args.skip_s0 else ["--s0-run", args.s0_run]
    execute("check_hierarchical_v4_gates.py", ["--mode", "full", "--run-name", args.run_name, *reference_arguments], "outer_gate")
    final_root = output_root / "final" / args.run_name
    final_mode = "--resume" if (final_root / "final_manifest.json").is_file() else "--fresh"
    execute("train_hierarchical_v4_final.py", ["--run-name", args.run_name, final_mode], "final_training")
    execute("export_hierarchical_v4_bundle.py", ["--run-name", args.run_name, "--resume" if (final_root / "model_bundle").exists() else "--fresh"], "export")
    execute("smoke_test_hierarchical_v4.py", [], "smoke")
    execute("replay_hierarchical_v4_raw_session.py", ["--bundle", str(final_root / "model_bundle"), "--forbid-xgboost"], "bundle_replay")
    write_status("COMPLETE", "COMPLETE", bundle=str(final_root / "model_bundle"))


if __name__ == "__main__":
    run_cli_with_failure_report(main, "integrated_execution")
