from __future__ import annotations

import argparse
import os

from _bootstrap import PROJECT_ROOT

from bme_eating.config import _load_project_dotenv, load_config, require_environment_path
from bme_eating.joint_continuation import fork_completed_selector


def main():
    parser = argparse.ArgumentParser(description="Copy completed selector training into a new extended run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent-run", required=True)
    parser.add_argument("--run-name", required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    _load_project_dotenv(config)
    output = require_environment_path(config, "output_root_env") / config["project"]["artifact_schema_version"]
    parent = output / "runs" / arguments.parent_run
    if parent.resolve().parent != (output / "runs").resolve():
        raise ValueError("Unsafe parent run name")
    run = fork_completed_selector(parent, config, PROJECT_ROOT, output, arguments.run_name)
    print(f"Continuation prepared: {run}; use --resume")


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
