"""Synthetic manifold data for autoencoder."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset


def build_datamodule(cfg: DictConfig) -> dict[str, Any]:
    num_samples = int(cfg.data.num_samples)
    latent_dim = int(cfg.data.latent_dim)
    ambient_dim = int(cfg.data.ambient_dim)
    noise_std = float(cfg.data.get("noise_std", 0.1))
    seed = int(cfg.seed)

    gen = torch.Generator().manual_seed(seed)

    # Random orthogonal embedding matrix
    raw = torch.randn(latent_dim, ambient_dim, generator=gen)
    u, _, vh = torch.linalg.svd(raw, full_matrices=False)
    w_embed = vh[:latent_dim]  # (latent_dim, ambient_dim), orthonormal rows

    z = torch.randn(num_samples, latent_dim, generator=gen)
    x = z @ w_embed
    noise = torch.randn(num_samples, ambient_dim, generator=gen) * noise_std
    x = x + noise

    perm = torch.randperm(num_samples, generator=gen)
    x = x[perm]

    val_split = int(cfg.data.val_split)
    x_train = x[:-val_split]
    x_val = x[-val_split:]

    batch_size = int(cfg.compute.batch_size)
    # Target = input (autoencoder)
    train_loader = DataLoader(TensorDataset(x_train, x_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(x_val, x_val), batch_size=batch_size, shuffle=False, drop_last=False)

    return {"train": train_loader, "val": val_loader}
