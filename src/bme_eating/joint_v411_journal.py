from __future__ import annotations

import hashlib
import json
import os


class CheckedJournal:
    def __init__(self, path, committed_bytes=0):
        self.path, self.offset, self.previous = path, committed_bytes, ""
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
        if path.stat().st_size < committed_bytes:
            raise RuntimeError("Committed diagnostic log is truncated")
        with path.open("r+b") as handle:
            while handle.tell() < committed_bytes:
                line = handle.readline()
                if not line.endswith(b"\n") or handle.tell() > committed_bytes:
                    raise RuntimeError("Committed log ends inside a record")
                record = json.loads(line)
                encoded = json.dumps(record["payload"], sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")
                checksum = hashlib.sha256(self.previous.encode() + encoded).hexdigest()
                if record["sha256"] != checksum:
                    raise RuntimeError("Diagnostic log checksum mismatch")
                self.previous = checksum
            handle.truncate(committed_bytes)

    def append(self, payloads):
        with self.path.open("ab") as handle:
            for payload in payloads:
                encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")
                self.previous = hashlib.sha256(self.previous.encode() + encoded).hexdigest()
                line = json.dumps({"payload": payload, "sha256": self.previous},
                                  ensure_ascii=False, separators=(",", ":")) + "\n"
                handle.write(line.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
            self.offset = handle.tell()

    def records(self):
        with self.path.open("rb") as handle:
            while handle.tell() < self.offset:
                yield json.loads(handle.readline())["payload"]
