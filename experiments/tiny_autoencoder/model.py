"""Tiny autoencoder model."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig


def build_model(cfg: DictConfig) -> nn.Module:
    ambient_dim = int(cfg.data.ambient_dim)
    bottleneck = int(cfg.model.bottleneck_dim)
    return nn.Sequential(
        nn.Linear(ambient_dim, bottleneck),
        nn.ReLU(),
        nn.Linear(bottleneck, bottleneck),
        nn.ReLU(),
        nn.Linear(bottleneck, ambient_dim),
    )
