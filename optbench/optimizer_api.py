"""Shared optimizer-controller contract for external and builtin optimizers."""

from __future__ import annotations

from typing import Any, Callable, Protocol

import torch

NamedParameters = list[tuple[str, torch.nn.Parameter]]
TensorMap = dict[str, torch.Tensor]
OptimizerBuildConfig = dict[str, Any]

# step_fn() -> (loss, grads, activations)
# Each call increments the global step counter, preventing cheating.
StepFn = Callable[[], tuple[float, TensorMap, TensorMap]]


class OptimizerControllerProtocol(Protocol):
    """Runtime optimizer object expected by train loops.

    New contract:
      - __init__(model, max_steps) receives the full model and step budget.
      - step(weights, grads, activations, step_fn) applies one optimization step.
        step_fn is a closure that performs another forward-backward pass and
        returns (loss, grads, activations). Every call costs one step.
      - zero_grad(set_to_none) clears gradients on tracked parameters.
    """

    def step(self, weights: TensorMap, grads: TensorMap, activations: TensorMap, step_fn: StepFn) -> None:
        """Apply one optimization step using current model signals."""

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients on tracked trainable parameters."""


OptimizerBuilder = Callable[[torch.nn.Module, int], OptimizerControllerProtocol]
