"""Synthetic data generation for logistic regression experiment."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset


def _generate_blobs(
    num_samples: int,
    num_features: int,
    num_classes: int,
    seed: int,
    class_sep: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate Gaussian blobs with fixed seed."""
    gen = torch.Generator().manual_seed(seed)
    samples_per_class = num_samples // num_classes
    xs, ys = [], []
    for cls_idx in range(num_classes):
        center = torch.randn(num_features, generator=gen) * class_sep
        points = torch.randn(samples_per_class, num_features, generator=gen) + center
        xs.append(points)
        ys.append(torch.full((samples_per_class,), cls_idx, dtype=torch.long))
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def build_datamodule(cfg: DictConfig) -> dict[str, Any]:
    """Build train/val dataloaders from synthetic blobs."""
    num_samples = int(cfg.data.num_samples)
    num_features = int(cfg.data.num_features)
    num_classes = int(cfg.model.num_classes)
    seed = int(cfg.seed)
    class_sep = float(cfg.data.get("class_sep", 3.0))

    x, y = _generate_blobs(num_samples, num_features, num_classes, seed, class_sep)

    # Shuffle deterministically
    gen = torch.Generator().manual_seed(seed + 1)
    perm = torch.randperm(len(x), generator=gen)
    x, y = x[perm], y[perm]

    val_split = int(cfg.data.val_split)
    x_train, y_train = x[:-val_split], y[:-val_split]
    x_val, y_val = x[-val_split:], y[-val_split:]

    batch_size = int(cfg.compute.batch_size)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader}
