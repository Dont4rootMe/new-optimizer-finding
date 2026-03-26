"""Data loading for MNIST binary classification experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


def build_datamodule(cfg: DictConfig) -> dict[str, Any]:
    """Build train/val dataloaders for MNIST binary (digit 0 vs 1)."""
    data_root = Path(str(cfg.data.root)).expanduser()
    class_a = int(cfg.data.class_a)
    class_b = int(cfg.data.class_b)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
    ])

    train_full = datasets.MNIST(root=str(data_root), train=True, download=True, transform=transform)
    test_full = datasets.MNIST(root=str(data_root), train=False, download=True, transform=transform)

    def _filter_and_flatten(dataset):
        xs, ys = [], []
        for img, label in dataset:
            if label == class_a or label == class_b:
                xs.append(img.view(-1))  # flatten 28x28 -> 784
                ys.append(0 if label == class_a else 1)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    x_train, y_train = _filter_and_flatten(train_full)
    x_test, y_test = _filter_and_flatten(test_full)

    # Split train into train/val
    val_split = int(cfg.data.val_split)
    gen = torch.Generator().manual_seed(int(cfg.seed))
    perm = torch.randperm(len(x_train), generator=gen)
    x_train, y_train = x_train[perm], y_train[perm]

    x_val, y_val = x_train[-val_split:], y_train[-val_split:]
    x_train, y_train = x_train[:-val_split], y_train[:-val_split]

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
