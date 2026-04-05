"""Polynomial regression model."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig


def build_model(cfg: DictConfig) -> nn.Module:
    degree = int(cfg.data.degree)
    return nn.Linear(degree, 1)
