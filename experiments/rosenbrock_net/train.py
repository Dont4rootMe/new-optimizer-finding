"""Training wrappers for Rosenbrock optimization."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from experiments._shared.train_paramonly import (
    evaluate_paramonly as evaluate,
    train as _train,
)


def train(cfg: DictConfig, model, datamodule, optimizer_factory) -> dict[str, Any]:
    return _train(cfg, model, datamodule, optimizer_factory)
