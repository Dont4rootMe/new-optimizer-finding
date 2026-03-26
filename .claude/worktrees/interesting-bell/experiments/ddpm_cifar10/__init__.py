"""DDPM CIFAR-10 experiment."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig

from .data import build_datamodule
from .model import build_model
from .train import evaluate, train


class DdpmCifar10Experiment:
    """Unified experiment implementation for DDPM on CIFAR-10."""

    name = "ddpm_cifar10"

    def build_datamodule(self, cfg: DictConfig) -> dict[str, Any]:
        return build_datamodule(cfg)

    def build_model(self, cfg: DictConfig) -> torch.nn.Module:
        return build_model(cfg)

    def train(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        datamodule: dict[str, Any],
        optimizer_factory,
    ) -> dict[str, Any]:
        return train(cfg, model, datamodule, optimizer_factory)

    def evaluate(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        datamodule: dict[str, Any],
    ) -> dict[str, float]:
        return evaluate(cfg, model, datamodule)
