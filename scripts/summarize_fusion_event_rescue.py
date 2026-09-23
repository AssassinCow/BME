from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.fusion_event_rescue import summarize_event_rescue


def main():
    parser = argparse.ArgumentParser(
        description="Apply v5 promotion gates to locked outer folds 2-4"
    )
    parser.add_argument("--config", default="configs/dtp_fusion_event_rescue.yaml")
    parser.add_argument("--run-name", required=True)
    summarize_event_rescue(parser.parse_args())


if __name__ == "__main__":
    main()
