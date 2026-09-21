from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.cli import command_export_dtp_quality
from bme_eating.fusion import FusionGateError


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-infer existing causal DTP checkpoints into a new frozen source run with "
            "quality diagnostics; no model training is performed."
        )
    )
    parser.add_argument("--config", default="configs/dtp_fusion_quality.yaml")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    try:
        command_export_dtp_quality(args)
    except FusionGateError as error:
        print(str(error))
        raise SystemExit(error.exit_code) from error


if __name__ == "__main__":
    main()
