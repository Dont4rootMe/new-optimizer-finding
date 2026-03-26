"""Linear denoiser model."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig


def build_model(cfg: DictConfig) -> nn.Module:
    signal_dim = int(cfg.data.signal_dim)
    return nn.Linear(signal_dim, signal_dim)
