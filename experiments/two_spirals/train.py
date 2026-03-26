"""Training wrappers for two spirals."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from omegaconf import DictConfig

from experiments._shared.train_supervised import (
    evaluate_classification as evaluate,
    train as _train,
)


def train(cfg: DictConfig, model, datamodule, optimizer_factory) -> dict[str, Any]:
    return _train(cfg, model, datamodule, optimizer_factory, criterion_cls=nn.CrossEntropyLoss, eval_fn=evaluate)
