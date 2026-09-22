from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.fusion_event_ablation import run_ablation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare frozen v4 fusion with rescue and balanced DTP event gates on OOF."
    )
    parser.add_argument("--config", default="configs/dtp_fusion_event_ablation.yaml")
    parser.add_argument("--fusion-run", required=True)
    parser.add_argument("--component-run", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, required=True)
    print(run_ablation(parser.parse_args()))


if __name__ == "__main__":
    main()
