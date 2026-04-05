"""Model definition for synthetic logistic regression experiment."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig


def build_model(cfg: DictConfig) -> nn.Module:
    """Build a single-layer linear classifier."""
    return nn.Linear(int(cfg.data.num_features), int(cfg.model.num_classes))
