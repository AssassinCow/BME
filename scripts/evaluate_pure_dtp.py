from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.pure_dtp import evaluate_pure_dtp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune pure DTP postprocessing on outer-train OOF, then diagnose one holdout."
    )
    parser.add_argument("--config", default="configs/dtp_fusion.yaml")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    evaluate_pure_dtp(args)


if __name__ == "__main__":
    main()
