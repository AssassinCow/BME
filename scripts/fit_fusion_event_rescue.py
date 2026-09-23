from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.fusion_event_rescue import fit_event_rescue


def main():
    parser = argparse.ArgumentParser(
        description="Fit fixed v5 thresholds from outer-train OOF only"
    )
    parser.add_argument("--config", default="configs/dtp_fusion_event_rescue.yaml")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    fit_event_rescue(parser.parse_args())


if __name__ == "__main__":
    main()
