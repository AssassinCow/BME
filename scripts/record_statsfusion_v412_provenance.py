from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description="Verify isolated v411 source snapshot without modifying original files")
    parser.add_argument("--source", required=True)
    arguments = parser.parse_args()
    original = Path(arguments.source).resolve()
    snapshot = json.loads((PROJECT_ROOT / "v412_source_snapshot.json").read_text(encoding="utf-8"))
    differences, checked = [], {}
    for relative, checksum in snapshot["files"].items():
        target = original / relative
        if relative.startswith(("src/", "scripts/", "configs/", "tests/", "docs/")):
            actual = sha256_file(target) if target.is_file() else None
            checked[relative] = actual
            if actual != checksum:
                differences.append(relative)
    def git(*command):
        return subprocess.run(["git", *command], cwd=original, check=True,
                              stdout=subprocess.PIPE).stdout.decode("utf-8")
    write_json_atomic(PROJECT_ROOT / "manifests/v412_source_provenance.json", {
        "source_root": str(original), "isolated_root": str(PROJECT_ROOT),
        "original_snapshot_sha256": sha256_file(PROJECT_ROOT / "v412_source_snapshot.json"),
        "snapshot_head": snapshot["head"], "current_source_head": git("rev-parse", "HEAD").strip(),
        "staged_diff": git("diff", "--cached", "--", "src", "scripts", "configs", "tests", "docs", "pyproject.toml"),
        "working_diff": git("diff", "--", "src", "scripts", "configs", "tests", "docs", "pyproject.toml"),
        "original_source_unchanged": not differences, "differences": differences,
        "verified_files": checked})
    if differences:
        raise RuntimeError("Original source changed since isolation: " + ", ".join(differences))
    print(f"Verified {len(checked)} original source files; original project unchanged", flush=True)


if __name__ == "__main__":
    main()
