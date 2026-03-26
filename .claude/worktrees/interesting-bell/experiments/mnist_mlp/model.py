"""Model definition for MNIST binary MLP experiment."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig


def build_model(cfg: DictConfig) -> nn.Module:
    """Build a small MLP for binary classification on MNIST."""
    input_dim = int(cfg.model.input_dim)
    hidden_dim = int(cfg.model.hidden_dim)
    num_classes = int(cfg.model.num_classes)

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )
