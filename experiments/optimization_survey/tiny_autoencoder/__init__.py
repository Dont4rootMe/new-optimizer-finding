"""Tiny autoencoder experiment."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig

from .data import build_datamodule
from .model import build_model
from .train import evaluate, train


class TinyAutoencoderExperiment:
    """Autoencoder on a synthetic manifold — tests reconstruction learning."""

    name = "tiny_autoencoder"

    def build_datamodule(self, cfg: DictConfig) -> dict[str, Any]:
        return build_datamodule(cfg)

    def build_model(self, cfg: DictConfig) -> torch.nn.Module:
        return build_model(cfg)

    def train(self, cfg, model, datamodule, optimizer_factory) -> dict[str, Any]:
        return train(cfg, model, datamodule, optimizer_factory)

    def evaluate(self, cfg, model, datamodule) -> dict[str, float]:
        return evaluate(cfg, model, datamodule)
