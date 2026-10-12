from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch


@contextmanager
def training_inference(device, *, cache_safe):
    device_type = torch.device(device).type
    autocast = torch.autocast(
        device_type=device_type,
        enabled=torch.is_autocast_enabled(device_type),
        dtype=torch.get_autocast_dtype(device_type),
        cache_enabled=False,
    ) if cache_safe else nullcontext()
    with autocast, torch.no_grad():
        yield
