"""Optimizer-controller contract for the optimization-survey family."""

from __future__ import annotations

from typing import Callable, Protocol

import torch

NamedParameters = list[tuple[str, torch.nn.Parameter]]
TensorMap = dict[str, torch.Tensor]
StepFn = Callable[[], tuple[float, TensorMap, TensorMap]]


class OptimizerControllerProtocol(Protocol):
    """Runtime optimizer object expected by optimization-survey train loops."""

    def step(self, weights: TensorMap, grads: TensorMap, activations: TensorMap, step_fn: StepFn) -> None:
        """Apply one optimization step using current model signals."""

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients on tracked trainable parameters."""


OptimizerBuilder = Callable[[torch.nn.Module, int], OptimizerControllerProtocol]
