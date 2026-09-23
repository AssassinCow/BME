"""Development replay and postprocessing benchmark; cannot authorize outer evaluation."""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import threading
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_roots
from bme_eating.fusion import sha256_file
from bme_eating.fusion_event_rescue import (
    ROOT,
    _identity,
    _inputs,
    _json,
    _manifest,
    _name,
    _write_evidence,
    apply_frozen_event_rescue,
    assert_development_replay,
    crossfit_event_rescue,
    validate_protocol,
)


def _rss_bytes():
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak_working_set",
                    "working_set",
                    "peak_paged",
                    "paged",
                    "peak_nonpaged",
                    "nonpaged",
                    "pagefile",
                    "peak_pagefile",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        api = ctypes.WinDLL("psapi")
        api.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if not api.GetProcessMemoryInfo(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        return int(counters.working_set)
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def benchmark(baseline, dtp, selection):
    sizes = (84_913, 169_826, 339_652)
    rows = []
    for size in sizes:
        if len(dtp) < size:
            raise ValueError("Benchmark requires at least 339652 frozen OOF windows")
        base = baseline.iloc[:size].copy()
        sample = dtp.iloc[:size].copy()
        initial = _rss_bytes()
        peak = [initial]
        stop = threading.Event()

        def monitor(stop=stop, peak=peak):
            while not stop.wait(0.01):
                peak[0] = max(peak[0], _rss_bytes())

        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        started = time.perf_counter()
        try:
            result, _, _ = apply_frozen_event_rescue(base, sample, selection)
        finally:
            elapsed = time.perf_counter() - started
            peak[0] = max(peak[0], _rss_bytes())
            stop.set()
            watcher.join()
        rows.append(
            {
                "windows": size,
                "seconds": elapsed,
                "events": len(result),
                "microseconds_per_window": elapsed / size * 1e6,
                "rss_before_bytes": initial,
                "sampled_peak_rss_bytes": peak[0],
                "sampled_incremental_rss_bytes": max(0, peak[0] - initial),
            }
        )
        print(f"Postprocess benchmark: {size} windows in {elapsed:.3f}s", flush=True)
    return {
        "measurements": rows,
        "neural_inference_included": False,
        "file_loading_included": False,
        "label_reading_included": False,
        "scope": "fixed decoding twice + stable sort + event rescue scan",
        "memory_method": "RSS sampled every 10ms; inputs preloaded; sub-10ms peaks may be missed",
        "complexity": "event scan O(B+D); sorting O(N log N); timings are empirical scaling evidence",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dtp_fusion_event_rescue.yaml")
    parser.add_argument("--source-run", required=True)
    parser.add_argument(
        "--run-name", required=True, help="New diagnostics directory, not a formal run"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    validate_protocol(config)
    identity = _identity(config, clean=False)
    _, root = resolve_roots(config)
    directory = root / "diagnostics" / _name(args.run_name)
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "formal_registration": False,
        "outer_predictions_read": False,
        "dirty_worktree": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        ),
        "identity": identity,
        "folds": {},
    }
    for fold in (0, 1):
        print(f"Development replay fold {fold}...", flush=True)
        baseline, dtp, labels, postprocess, hashes = _inputs(root, config, args.source_run, fold)
        record, combined, base_events, audit = crossfit_event_rescue(
            baseline, dtp, labels, postprocess
        )
        assert_development_replay(record, fold)
        output = directory / f"fold_{fold}"
        output.mkdir()
        _write_evidence(output, dtp, combined, base_events, audit, labels, postprocess)
        _json(output / "development_selection.json", record)
        _manifest(output, identity, {"inputs": hashes, "formal_registration": False})
        report["folds"][str(fold)] = {
            k: record[k]
            for k in ("baseline_metrics", "candidate_metrics", "meta_oof_gate", "partitions")
        }
        if fold == 0:
            # OOF prefix only: this benchmark does not consume even observed outer data.
            report["benchmark"] = benchmark(baseline, dtp, record)
    if _identity(config, clean=False) != identity:
        raise RuntimeError("Code/config changed during development replay; repeat before reporting")
    _json(directory / "validation_report.json", report)
    # Include the nested evidence manifests in the top-level artifact list.
    _manifest(
        directory,
        identity,
        {
            "formal_registration": False,
            "source_run": args.source_run,
            "fold_manifest_hashes": {
                str(f): sha256_file(directory / f"fold_{f}" / "run_manifest.json") for f in (0, 1)
            },
        },
    )
    print(directory, flush=True)


if __name__ == "__main__":
    main()
