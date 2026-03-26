"""Rosenbrock function minimization experiment."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig

from .model import build_model
from .train import evaluate, train


class RosenbrockNetExperiment:
    """Minimize Rosenbrock function over learnable parameters — tests narrow valley navigation."""

    name = "rosenbrock_net"

    def build_datamodule(self, cfg: DictConfig) -> dict[str, Any]:
        return {}

    def build_model(self, cfg: DictConfig) -> torch.nn.Module:
        return build_model(cfg)

    def train(self, cfg, model, datamodule, optimizer_factory) -> dict[str, Any]:
        return train(cfg, model, datamodule, optimizer_factory)

    def evaluate(self, cfg, model, datamodule) -> dict[str, float]:
        return evaluate(cfg, model, datamodule)
