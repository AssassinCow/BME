from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.dtp_postprocess import evaluate_dtp_postprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one frozen pure DTP working point")
    parser.add_argument("--config", default="configs/dtp_postprocess.yaml")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--selection-run", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 5), required=True)
    evaluate_dtp_postprocess(parser.parse_args())


if __name__ == "__main__":
    main()
