"""Synthetic XOR data with noise."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset


def build_datamodule(cfg: DictConfig) -> dict[str, Any]:
    num_samples = int(cfg.data.num_samples)
    noise_std = float(cfg.data.get("noise_std", 0.15))
    seed = int(cfg.seed)

    gen = torch.Generator().manual_seed(seed)

    # 4 canonical XOR points
    xor_inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    xor_labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    samples_per_point = num_samples // 4
    xs, ys = [], []
    for i in range(4):
        noise = torch.randn(samples_per_point, 2, generator=gen) * noise_std
        xs.append(xor_inputs[i].unsqueeze(0).expand(samples_per_point, -1) + noise)
        ys.append(xor_labels[i].expand(samples_per_point))

    features = torch.cat(xs, dim=0)
    labels = torch.cat(ys, dim=0)

    perm = torch.randperm(len(features), generator=gen)
    features, labels = features[perm], labels[perm]

    val_split = int(cfg.data.val_split)
    x_train, y_train = features[:-val_split], labels[:-val_split]
    x_val, y_val = features[-val_split:], labels[-val_split:]

    batch_size = int(cfg.compute.batch_size)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, drop_last=False)

    return {"train": train_loader, "val": val_loader}
