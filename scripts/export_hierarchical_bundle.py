from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_export import export_hierarchical_bundle
from bme_eating.reproducibility import require_git_worktree


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a privacy-safe v3 model bundle")
    parser.add_argument("--config", default="configs/hierarchical_v3.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", choices=("all",), default="all")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require_git_worktree()
    config = load_config(args.config)
    _, _, output_root = resolve_artifact_roots(config)
    project_root = Path(__file__).resolve().parents[1]
    bundle = export_hierarchical_bundle(
        project_root,
        output_root / "final" / args.run_name,
        fresh=args.fresh,
        resume=args.resume,
    )
    print(bundle)


if __name__ == "__main__":
    main()
