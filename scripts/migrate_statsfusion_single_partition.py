from __future__ import annotations

import argparse
import os
import re

from _bootstrap import PROJECT_ROOT

from bme_eating.config import _load_project_dotenv, load_config, require_environment_path
from bme_eating.data.joint_sessions import JointInputs
from bme_eating.joint_partition_migration import migrate_single_partition


def main():
    parser = argparse.ArgumentParser(description="Preserve paused v411 checkpoints and run only internal partition 0")
    parser.add_argument("--config", default="configs/statsfusion_v411_fold0_single_partition.yaml")
    parser.add_argument("--run-name", required=True)
    arguments = parser.parse_args()
    if arguments.run_name in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", arguments.run_name):
        parser.error("Unsafe run name")
    config = load_config(arguments.config)
    _load_project_dotenv(config)
    output = require_environment_path(config, "output_root_env")
    inputs = JointInputs.audit(output / config["project"]["input_artifact_schema_version"],
                               output / config["project"]["canonical_schema_version"])
    run = output / config["project"]["artifact_schema_version"] / "runs" / arguments.run_name
    receipt = migrate_single_partition(run, config, PROJECT_ROOT, inputs.identity)
    print(f"Single-partition migration {receipt['status']}: {run}; use the new config with --resume")


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
