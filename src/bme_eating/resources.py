from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

_peak_private = 0.0


def private_process_tree(root_pid=None):
    global _peak_private
    root_pid = os.getpid() if root_pid is None else root_pid
    if os.name != "nt":
        import psutil
        process = psutil.Process(root_pid)
        members = [process, *process.children(recursive=True)]
        private = 0
        for member in members:
            try:
                private += member.memory_full_info().uss
            except psutil.NoSuchProcess:
                continue
        count = len(members)
    else:
        class ProcessEntry(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD),
                        ("pid", wintypes.DWORD), ("heap", ctypes.c_size_t),
                        ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                        ("parent", wintypes.DWORD), ("priority", wintypes.LONG),
                        ("flags", wintypes.DWORD), ("name", wintypes.WCHAR * 260)]
        class Counters(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in (
                            "peak_working", "working", "peak_paged", "paged", "peak_nonpaged",
                            "nonpaged", "pagefile", "peak_pagefile", "private")]]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel.Process32FirstW.argtypes = kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise RuntimeError("Cannot inspect process tree private RAM")
        parents, entry = {}, ProcessEntry()
        entry.size = ctypes.sizeof(entry)
        try:
            available = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
            while available:
                parents[entry.pid] = entry.parent
                available = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            kernel.CloseHandle(snapshot)
        members = {root_pid}
        while True:
            extended = members | {pid for pid, parent in parents.items() if parent in members}
            if members == extended:
                break
            members = extended
        private, count = 0, len(members)
        for pid in members:
            process = kernel.OpenProcess(0x410, False, pid)
            if not process:
                continue
            counters = Counters()
            counters.size = ctypes.sizeof(counters)
            try:
                if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.size):
                    raise RuntimeError("Cannot read owned process private RAM")
                private += counters.private
            finally:
                kernel.CloseHandle(process)
    current = private / 1024 ** 3
    _peak_private = max(_peak_private, current)
    return {"process_tree_private_ram_gb": current, "peak_process_tree_private_ram_gb": _peak_private,
            "worker_process_count": count - 1}


class SearchResourceMonitor:
    def __init__(self, config, root_pid=None):
        self.limit = config["resources"].get("maximum_private_ram_gb")
        self.root_pid = root_pid
        self.last_check = float("-inf")
        self.usage = {}

    def check(self, *, force=False, allocation_bytes=0):
        import time
        now = time.monotonic()
        if self.limit is None or (not force and not allocation_bytes and now - self.last_check < .25):
            return self.usage
        self.usage = private_process_tree(self.root_pid)
        self.last_check = now
        projected = self.usage["process_tree_private_ram_gb"] + allocation_bytes / 1024 ** 3
        if projected >= self.limit:
            raise RuntimeError(f"Search process tree private RAM exceeds registered {self.limit:g} GB limit")
        return self.usage
