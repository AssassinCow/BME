from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.fusion_end_preservation import run_end_preservation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay frozen fold-0 rescue and test label-free end preservation."
    )
    parser.add_argument("--config", default="configs/dtp_fusion_end_preservation.yaml")
    parser.add_argument("--fusion-run", required=True)
    parser.add_argument("--component-run", required=True)
    parser.add_argument("--ablation-run", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, required=True)
    print(run_end_preservation(parser.parse_args()))


if __name__ == "__main__":
    main()
