from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.v49_resume import RECOVERY_RUN, prepare_resume_migration


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and preserve the 20261006c recovery lineage")
    parser.add_argument("--config", default="configs/hierarchical_v4_v49_integrated_repair.yaml")
    parser.add_argument("--run-name", default=RECOVERY_RUN)
    parser.add_argument("--apply", action="store_true", help="Back up evidence and apply the audited recovery")
    args = parser.parse_args()
    config = load_config(args.config)
    _, input_root, output_root = resolve_artifact_roots(config)
    report = prepare_resume_migration(config, input_root, output_root, args.run_name, apply=args.apply)
    print(json.dumps({
        "mode": "applied" if args.apply else "dry_run", "run_name": report["run_name"],
        "continuation": report["continuation"], "runtime_config": report["active_runtime_config"],
        "legacy_checkpoint_count": len(report["legacy_checkpoints"]),
    }, indent=2))


if __name__ == "__main__":
    main()
