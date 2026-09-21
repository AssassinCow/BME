from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.cli import command_evaluate_fusion_v4
from bme_eating.fusion import FusionGateError


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen protocol-v4 fusion selection on one outer fold."
    )
    parser.add_argument("--config", default="configs/dtp_fusion.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    args = parser.parse_args()
    try:
        command_evaluate_fusion_v4(args)
    except FusionGateError as error:
        print(str(error))
        raise SystemExit(error.exit_code) from error


if __name__ == "__main__":
    main()
