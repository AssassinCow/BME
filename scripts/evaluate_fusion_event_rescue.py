from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.fusion_event_rescue import evaluate_event_rescue


def main():
    parser = argparse.ArgumentParser(description="Evaluate a frozen v5 outer stress fold once")
    parser.add_argument("--config", default="configs/dtp_fusion_event_rescue.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=(2, 3, 4), required=True)
    evaluate_event_rescue(parser.parse_args())


if __name__ == "__main__":
    main()
