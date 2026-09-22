from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.dtp_postprocess import tune_dtp_postprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune pure DTP postprocessing on frozen OOF only")
    parser.add_argument("--config", default="configs/dtp_postprocess.yaml")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=(0,), required=True)
    parser.add_argument("--workers", type=int, default=16)
    tune_dtp_postprocess(parser.parse_args())


if __name__ == "__main__":
    main()
