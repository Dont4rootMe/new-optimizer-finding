"""Synthetic denoising data."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset


def build_datamodule(cfg: DictConfig) -> dict[str, Any]:
    num_samples = int(cfg.data.num_samples)
    signal_dim = int(cfg.data.signal_dim)
    noise_std = float(cfg.data.get("noise_std", 0.5))
    seed = int(cfg.seed)

    gen = torch.Generator().manual_seed(seed)

    # Clean signals
    clean = torch.randn(num_samples, signal_dim, generator=gen)
    # Noisy inputs
    noise = torch.randn(num_samples, signal_dim, generator=gen) * noise_std
    noisy = clean + noise

    perm = torch.randperm(num_samples, generator=gen)
    noisy, clean = noisy[perm], clean[perm]

    val_split = int(cfg.data.val_split)
    x_train, y_train = noisy[:-val_split], clean[:-val_split]
    x_val, y_val = noisy[-val_split:], clean[-val_split:]

    batch_size = int(cfg.compute.batch_size)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, drop_last=False)

    return {"train": train_loader, "val": val_loader}
