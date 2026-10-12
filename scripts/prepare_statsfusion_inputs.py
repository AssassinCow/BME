from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from bme_eating.config import load_config
from bme_eating.data.input_preparation import prepare_inputs


def main():
    parser = argparse.ArgumentParser(description="Prepare audited v2 and canonical v4 training inputs")
    parser.add_argument("--config", default="configs/statsfusion_v414_full_data.yaml")
    parser.add_argument("--stage", choices=("v2", "canonical", "all"), default="all")
    parser.add_argument("--workers", type=int, default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = prepare_inputs(load_config(args.config), stage=args.stage, workers=args.workers,
                            fresh=args.fresh, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
