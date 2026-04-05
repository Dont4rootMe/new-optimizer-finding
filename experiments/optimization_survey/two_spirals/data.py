"""Synthetic two-spirals data."""

from __future__ import annotations

import math
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset


def _make_spiral(n: int, offset: float, gen: torch.Generator, noise_std: float) -> tuple[torch.Tensor, torch.Tensor]:
    r = torch.linspace(0.1, 1.0, n)
    theta = r * 4 * math.pi + offset
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    coords = torch.stack([x, y], dim=1)
    noise = torch.randn(n, 2, generator=gen) * noise_std
    return coords + noise


def build_datamodule(cfg: DictConfig) -> dict[str, Any]:
    samples_per_class = int(cfg.data.samples_per_class)
    noise_std = float(cfg.data.get("noise_std", 0.02))
    seed = int(cfg.seed)

    gen = torch.Generator().manual_seed(seed)

    spiral_0 = _make_spiral(samples_per_class, 0.0, gen, noise_std)
    spiral_1 = _make_spiral(samples_per_class, math.pi, gen, noise_std)

    features = torch.cat([spiral_0, spiral_1], dim=0)
    labels = torch.cat([
        torch.zeros(samples_per_class, dtype=torch.long),
        torch.ones(samples_per_class, dtype=torch.long),
    ])

    perm = torch.randperm(len(features), generator=gen)
    features, labels = features[perm], labels[perm]

    val_split = int(cfg.data.val_split)
    x_train, y_train = features[:-val_split], labels[:-val_split]
    x_val, y_val = features[-val_split:], labels[-val_split:]

    batch_size = int(cfg.compute.batch_size)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, drop_last=False)

    return {"train": train_loader, "val": val_loader}
