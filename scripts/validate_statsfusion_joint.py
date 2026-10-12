from __future__ import annotations

import argparse
import json
import os

from _bootstrap import PROJECT_ROOT

from bme_eating.artifacts import write_json_atomic
from bme_eating.config import load_config
from bme_eating.protocol import require_idle_gpu
from bme_eating.validation import replay_joint_bundle, validate_joint_implementation


def main():
    parser = argparse.ArgumentParser(description="Joint model CPU/GPU smoke and native raw replay")
    parser.add_argument("--config", default="configs/statsfusion_v414_full_data.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bundle")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    config = load_config(PROJECT_ROOT / arguments.config)
    require_idle_gpu(arguments.device)
    if arguments.bundle:
        from pathlib import Path
        report = replay_joint_bundle(Path(arguments.bundle), config, device=arguments.device)
    else:
        report = validate_joint_implementation(config, device=arguments.device)
    if arguments.output:
        from pathlib import Path
        write_json_atomic(Path(arguments.output), report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
