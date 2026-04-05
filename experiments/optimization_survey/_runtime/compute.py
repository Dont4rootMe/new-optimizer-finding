"""Shared compute-device helpers for optimization_survey experiments."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def is_cuda_device_name(device_name: str) -> bool:
    return str(device_name).strip().startswith("cuda")


def resolve_torch_device(device_name: str) -> torch.device:
    requested = str(device_name).strip()
    if is_cuda_device_name(requested) and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def pin_memory_for_device(device_name: str) -> bool:
    return is_cuda_device_name(device_name)


def autocast_context(*, device: torch.device, precision: str):
    enabled = device.type == "cuda" and str(precision) in {"bf16", "fp16"}
    if not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if str(precision) == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
