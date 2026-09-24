from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_artifacts import initialize_hierarchical_run
from bme_eating.reproducibility import require_git_worktree
from bme_eating.training.hierarchical_trainer import migrate_completed_state_partition


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate a completed state checkpoint across permitted configuration changes"
    )
    parser.add_argument("--config", default="configs/hierarchical_v3.yaml")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--partition", type=int, choices=range(3), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.source_run == args.run_name:
        raise ValueError("Checkpoint migration requires a new target run name")
    require_git_worktree()
    config = load_config(args.config)
    _, input_root, output_root = resolve_artifact_roots(config)
    source_root = output_root / "experiments" / args.source_run / f"fold_{args.fold}"
    source_config_path = source_root / "resolved_config.yaml"
    if not source_config_path.is_file():
        raise FileNotFoundError(f"Source resolved configuration is missing: {source_config_path}")
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8")) or {}
    target = initialize_hierarchical_run(
        config,
        input_root,
        output_root,
        args.run_name,
        args.fold,
        fresh=args.fresh,
    )
    migration = migrate_completed_state_partition(
        source_root,
        target,
        source_config,
        config,
        args.partition,
    )
    migration_path = (
        target.root
        / f"crossfit_{args.partition}"
        / "state"
        / "checkpoint_migration.json"
    )
    print(
        {
            "target_run": migration["target_run"],
            "partition": migration["partition"],
            "model_weights_unchanged": migration["model_weights_unchanged"],
            "migration_record": str(Path(migration_path).resolve()),
        }
    )


if __name__ == "__main__":
    main()
