"""Synthetic polynomial regression data."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset


def build_datamodule(cfg: DictConfig) -> dict[str, Any]:
    num_samples = int(cfg.data.num_samples)
    degree = int(cfg.data.degree)
    seed = int(cfg.seed)
    noise_std = float(cfg.data.get("noise_std", 0.1))

    gen = torch.Generator().manual_seed(seed)

    x = torch.rand(num_samples, 1, generator=gen) * 4.0 - 2.0  # Uniform(-2, 2)
    y_true = 0.5 * x ** 5 - 2.0 * x ** 3 + x
    noise = torch.randn(num_samples, 1, generator=gen) * noise_std
    y = y_true + noise

    # Polynomial features: [x, x^2, ..., x^degree]
    features = torch.cat([x ** i for i in range(1, degree + 1)], dim=1)

    # Shuffle
    perm = torch.randperm(num_samples, generator=gen)
    features, y = features[perm], y[perm]

    val_split = int(cfg.data.val_split)
    x_train, y_train = features[:-val_split], y[:-val_split]
    x_val, y_val = features[-val_split:], y[-val_split:]

    batch_size = int(cfg.compute.batch_size)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, drop_last=False)

    return {"train": train_loader, "val": val_loader}
