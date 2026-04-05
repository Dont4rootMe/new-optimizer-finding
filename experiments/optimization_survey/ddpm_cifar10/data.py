"""Data loading for DDPM CIFAR-10."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_datamodule(cfg: DictConfig) -> dict[str, Any]:
    """Build dataloaders for diffusion training on CIFAR-10."""

    data_root = Path(str(cfg.data.root)).expanduser()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )

    train_set = datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=transform)

    batch_size = int(cfg.compute.batch_size)
    num_workers = int(cfg.compute.num_workers)
    pin_memory = str(cfg.compute.device) == "cuda"

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": val_loader,
    }
