from __future__ import annotations

import json

from _bootstrap import PROJECT_ROOT

from bme_eating.config import _load_project_dotenv, load_config, require_environment_path
from bme_eating.data.joint_sessions import JointInputs
from bme_eating.development_split import generate_split, validate_split


def main():
    config = load_config(PROJECT_ROOT / "configs/statsfusion_v414_full_data.yaml")
    _load_project_dotenv(config)
    output = require_environment_path(config, "output_root_env")
    inputs = JointInputs.audit(output / "v2", output / "v4")
    split = generate_split(inputs)
    validate_split(split, inputs)
    print(json.dumps(split, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
