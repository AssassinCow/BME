from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_gates import write_freeze_manifest
from bme_eating.reproducibility import require_git_worktree


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock the folds 2-4 v3 stress protocol")
    parser.add_argument("--config", default="configs/hierarchical_v3.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--development-decision", required=True)
    args = parser.parse_args()
    require_git_worktree()
    config = load_config(args.config)
    _, _, output_root = resolve_artifact_roots(config)
    decision_path = Path(args.development_decision).resolve()
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    print(write_freeze_manifest(output_root, args.run_name, decision))


if __name__ == "__main__":
    main()
