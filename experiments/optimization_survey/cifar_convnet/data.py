"""Data loading for CIFAR-10 experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from experiments.optimization_survey._runtime.compute import pin_memory_for_device


def build_datamodule(cfg: DictConfig) -> dict[str, Any]:
    """Build train/val/test dataloaders for CIFAR-10."""

    data_root = Path(str(cfg.data.root)).expanduser()
    val_split = int(cfg.data.val_split)

    mean = tuple(float(x) for x in cfg.data.normalize_mean)
    std = tuple(float(x) for x in cfg.data.normalize_std)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    full_train_aug = datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=train_transform)
    full_train_eval = datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=eval_transform)
    test_set = datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=eval_transform)

    if val_split <= 0 or val_split >= len(full_train_aug):
        raise ValueError(f"Invalid val_split={val_split} for CIFAR-10 train size {len(full_train_aug)}")

    gen = torch.Generator().manual_seed(int(cfg.seed))
    perm = torch.randperm(len(full_train_aug), generator=gen).tolist()
    train_indices = perm[:-val_split]
    val_indices = perm[-val_split:]

    train_set = Subset(full_train_aug, train_indices)
    val_set = Subset(full_train_eval, val_indices)

    batch_size = int(cfg.compute.batch_size)
    num_workers = int(cfg.compute.num_workers)
    pin_memory = pin_memory_for_device(str(cfg.compute.device))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
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
        "test": test_loader,
    }
