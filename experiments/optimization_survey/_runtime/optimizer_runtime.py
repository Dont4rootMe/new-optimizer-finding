"""Runtime helpers for optimization-survey controller integration."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from experiments.optimization_survey._runtime.optimizer_api import NamedParameters, TensorMap


class StepBudget:
    """Tracks how many extra steps the candidate consumed via step_fn."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.consumed = 0
        self._last_read = 0

    def increment(self) -> None:
        self.consumed += 1

    @property
    def exhausted(self) -> bool:
        return self.consumed >= self.max_steps

    @property
    def current_cycle_consumed(self) -> int:
        return self.consumed - self._last_read

    @property
    def consumed_since_last(self) -> int:
        delta = self.consumed - self._last_read
        self._last_read = self.consumed
        return delta


class ActivationRecorder:
    """Collect recent module activations via forward hooks."""

    def __init__(self, model: nn.Module, max_modules: int = 32) -> None:
        self._activations: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register(model, max_modules=max_modules)

    def _register(self, model: nn.Module, max_modules: int) -> None:
        tracked = 0
        for name, module in model.named_modules():
            if tracked >= max_modules:
                break
            if not self._should_track(module):
                continue
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)
            tracked += 1

    def _should_track(self, module: nn.Module) -> bool:
        if len(list(module.children())) > 0:
            return False
        return any(param.requires_grad for param in module.parameters(recurse=False))

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = self._extract_tensor(output)
            if tensor is None:
                return
            self._activations[name] = tensor.detach()

        return hook

    def _extract_tensor(self, output: Any) -> torch.Tensor | None:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (list, tuple)):
            for item in output:
                if torch.is_tensor(item):
                    return item
        return None

    def snapshot(self) -> TensorMap:
        return dict(self._activations)

    def clear(self) -> None:
        self._activations.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._activations.clear()


def collect_named_trainable_parameters(model: nn.Module) -> NamedParameters:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def collect_step_tensors(named_parameters: NamedParameters) -> tuple[TensorMap, TensorMap]:
    weights: TensorMap = {}
    grads: TensorMap = {}
    for name, param in named_parameters:
        weights[name] = param.detach()
        if param.grad is not None:
            grads[name] = param.grad.detach()
    return weights, grads
