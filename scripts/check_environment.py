from __future__ import annotations

import json
import platform
import sys

import _bootstrap as _bootstrap
import torch


def main():
    print(json.dumps({"python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
        "devices": [{"name": torch.cuda.get_device_name(index),
                     "total_memory_gb": torch.cuda.get_device_properties(index).total_memory / 1024 ** 3}
                    for index in range(torch.cuda.device_count())]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
