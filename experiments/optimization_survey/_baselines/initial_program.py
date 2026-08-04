"""Runnable, task-agnostic generation-0 seed for the optimization survey.

Every optimization task calls the same ``build_optimizer`` contract. The seed
uses conservative Adam without task-specific knowledge, giving every island a
finite, executable starting organism that later mutations can specialize.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# EVOLVE-BLOCK-START
OPTIMIZER_NAME = "SeedAdam"


class SeedAdam:
    """Small controller adapter around a conservative built-in Adam update."""

    def __init__(self, model: nn.Module, max_steps: int) -> None:
        self.model = model
        self.max_steps = max(1, int(max_steps))
        self.step_index = 0
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("SeedAdam requires at least one trainable parameter.")
        self.optimizer = torch.optim.Adam(
            parameters,
            lr=1.0e-3,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=0.0,
        )

    def step(self, weights, grads, activations, step_fn) -> None:
        del weights, grads, activations, step_fn
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.step_index += 1

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)


def build_optimizer(model: nn.Module, max_steps: int):
    return SeedAdam(model, max_steps)
# EVOLVE-BLOCK-END
