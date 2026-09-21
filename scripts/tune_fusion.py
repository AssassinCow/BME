from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.cli import command_tune_fusion_v4
from bme_eating.fusion import FusionGateError


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune protocol-v4 calibrated gated fusion from frozen baseline and DTP OOF "
            "predictions without reading the outer-fold labels."
        )
    )
    parser.add_argument("--config", default="configs/dtp_fusion.yaml")
    parser.add_argument("--source-run", required=True, help="Run containing frozen DTP predictions.")
    parser.add_argument("--run-name", required=True, help="New isolated protocol-v4 result run.")
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--workers", type=int, default=16)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        command_tune_fusion_v4(args)
    except FusionGateError as error:
        print(str(error))
        raise SystemExit(error.exit_code) from error


if __name__ == "__main__":
    main()
