from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_v4_export import export_hierarchical_v4_bundle
from bme_eating.reproducibility import require_git_worktree


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an XGBoost-free StatsFusion v4 bundle")
    parser.add_argument("--config", default="configs/hierarchical_v4_statsfusion.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require_git_worktree()
    config = load_config(args.config)
    _, _, output_root = resolve_artifact_roots(config)
    final_root = output_root / "final" / args.run_name
    export_hierarchical_v4_bundle(
        Path(__file__).resolve().parents[1],
        final_root,
        fresh=args.fresh,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
