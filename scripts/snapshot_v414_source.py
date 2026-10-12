from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def hashes(root):
    excluded = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".modeling", ".vscode", ".v414-git-baseline"}
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()
            and not excluded.intersection(path.relative_to(root).parts)
            and path.suffix in {".py", ".yaml", ".yml", ".toml", ".md", ".json", ".ps1"}
            and not path.relative_to(root).as_posix().startswith("manifests/v414_")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    target = Path(__file__).resolve().parents[1]
    source = arguments.source.resolve()
    if source == target or not (source / ".git").exists():
        raise RuntimeError("Expected a distinct v413 Git checkout")
    manifest = target / "manifests/v414_source_provenance.json"
    current = hashes(source)
    if arguments.verify:
        stored = json.loads(manifest.read_text(encoding="utf-8"))
        if current != stored["source_files"]:
            raise RuntimeError("v413 source changed since the isolated snapshot")
        delivery = {"v413_source_unchanged": True, "source_sha256": stored["source_sha256"],
                    "v414_delivery_files": hashes(target), "active_outputs_modified_by_delivery": False}
        (target / "manifests/v414_delivery_verification.json").write_text(
            json.dumps(delivery, ensure_ascii=False, indent=2), encoding="utf-8")
        print("v413 source unchanged; v414 delivery hashes saved")
        return
    if manifest.exists():
        raise FileExistsError("Provenance is immutable")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    def git(*options):
        return subprocess.run(["git", "-C", str(source), *options], check=True, capture_output=True).stdout
    for name, options in (("staged", ("diff", "--cached", "--binary", "HEAD")),
                          ("unstaged", ("diff", "--binary"))):
        (manifest.parent / f"v414_parent_{name}.patch").write_bytes(git(*options))
    initial = hashes(target)
    missing = {name: checksum for name, checksum in current.items() if initial.get(name) != checksum}
    if missing:
        raise RuntimeError(f"Working-tree snapshot differs: {list(missing)[:10]}")
    payload = {"source_directory": str(source), "source_head": git("rev-parse", "HEAD").decode().strip(),
        "source_files": current, "source_sha256": hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest(),
        "initial_target_files": initial, "untracked": git("ls-files", "--others", "--exclude-standard").decode("utf-8"),
        "copy_method": "working_tree_copy_plus_independent_git_clone_no_hardlinks",
        "staged_diff": "v414_parent_staged.patch", "unstaged_diff": "v414_parent_unstaged.patch",
        "outputs_are_read_only_inputs": True}
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Snapshot verified: {len(current)} source files; independent Git objects")


if __name__ == "__main__":
    main()
