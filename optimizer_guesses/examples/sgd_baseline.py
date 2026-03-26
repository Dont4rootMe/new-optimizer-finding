"""Baseline optimizer controller used for sanity checks and prompts."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

OPTIMIZER_NAME = "SGDBaselineController"


class SGDBaselineController:
    """Reference optimizer compatible with build_optimizer(model, max_steps) contract."""

    def __init__(self, model: nn.Module, max_steps: int) -> None:
        self.named_parameters = [
            (name, param) for name, param in model.named_parameters() if param.requires_grad
        ]

        lr = 0.1
        momentum = 0.9
        weight_decay = 5e-4
        nesterov = True

        params = [param for _, param in self.named_parameters]
        self.optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

        warmup_steps = max(1, int(max_steps * 0.02))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return max(1e-8, float(step + 1) / float(warmup_steps))
            progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def step(self, weights, grads, activations, step_fn) -> None:
        del weights, grads, activations, step_fn
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)


def build_optimizer(model: nn.Module, max_steps: int) -> SGDBaselineController:
    """Create baseline optimizer controller."""

    return SGDBaselineController(model, max_steps)
