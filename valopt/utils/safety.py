"""Numerical safety checks used in train loops."""

from __future__ import annotations

from typing import Iterable

import torch


def loss_is_finite(loss: torch.Tensor) -> bool:
    """Return True when loss tensor contains finite value."""

    if loss.numel() == 0:
        return False
    return bool(torch.isfinite(loss.detach()).all().item())


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Compute global L2 grad norm for parameters that have gradients."""

    total = torch.zeros(1, device="cpu")
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += grad.float().pow(2).sum().cpu()
    return float(total.sqrt().item())


def check_params_finite(model: torch.nn.Module) -> bool:
    """Check that all model parameters are finite."""

    for param in model.parameters():
        if not torch.isfinite(param.detach()).all():
            return False
    return True
