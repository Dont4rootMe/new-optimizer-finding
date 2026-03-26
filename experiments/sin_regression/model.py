"""Sinusoidal regression MLP model."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig


def build_model(cfg: DictConfig) -> nn.Module:
    hidden = int(cfg.model.hidden_dim)
    return nn.Sequential(
        nn.Linear(1, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 1),
    )
