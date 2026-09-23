from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.fusion_event_rescue import register_event_rescue


def main():
    parser = argparse.ArgumentParser(description="Lock v5 after reproducing development folds 0/1")
    parser.add_argument("--config", default="configs/dtp_fusion_event_rescue.yaml")
    parser.add_argument("--run-name", required=True)
    register_event_rescue(parser.parse_args())


if __name__ == "__main__":
    main()
