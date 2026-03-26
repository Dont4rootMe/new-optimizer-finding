"""Synthetic sinusoidal regression data."""

from __future__ import annotations

import math
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset


def build_datamodule(cfg: DictConfig) -> dict[str, Any]:
    num_samples = int(cfg.data.num_samples)
    noise_std = float(cfg.data.get("noise_std", 0.05))
    seed = int(cfg.seed)

    gen = torch.Generator().manual_seed(seed)

    x = torch.rand(num_samples, 1, generator=gen) * 2 * math.pi - math.pi  # Uniform(-pi, pi)
    y = torch.sin(x) + 0.3 * torch.sin(3 * x) + 0.1 * torch.sin(5 * x)
    noise = torch.randn(num_samples, 1, generator=gen) * noise_std
    y = y + noise

    perm = torch.randperm(num_samples, generator=gen)
    x, y = x[perm], y[perm]

    val_split = int(cfg.data.val_split)
    x_train, y_train = x[:-val_split], y[:-val_split]
    x_val, y_val = x[-val_split:], y[-val_split:]

    batch_size = int(cfg.compute.batch_size)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, drop_last=False)

    return {"train": train_loader, "val": val_loader}
