"""1D CNN model for waveform classification."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig


def build_model(cfg: DictConfig) -> nn.Module:
    num_classes = int(cfg.model.num_classes)
    channels = int(cfg.model.channels)
    pool_out = int(cfg.model.pool_out)
    return nn.Sequential(
        nn.Conv1d(1, channels, kernel_size=7, padding=3),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(pool_out),
        nn.Flatten(),
        nn.Linear(channels * pool_out, num_classes),
    )
