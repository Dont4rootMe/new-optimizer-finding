"""Synthetic 1D waveform data for classification."""

from __future__ import annotations

import math
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset


def _generate_waveforms(
    n: int, seq_len: int, gen: torch.Generator, noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate 3 classes of synthetic 1D waveforms.

    Class 0: low-frequency sine
    Class 1: high-frequency sine
    Class 2: clipped (square-like) sine
    """
    t = torch.linspace(0, 2 * math.pi, seq_len).unsqueeze(0)  # (1, seq_len)
    per_class = n // 3

    signals = []
    labels = []

    # Class 0: low freq (freq ~ 1-2 cycles)
    freq = 1.0 + torch.rand(per_class, 1, generator=gen)
    phase = torch.rand(per_class, 1, generator=gen) * 2 * math.pi
    amp = 0.5 + torch.rand(per_class, 1, generator=gen) * 0.5
    wave = amp * torch.sin(freq * t + phase)
    signals.append(wave)
    labels.append(torch.zeros(per_class, dtype=torch.long))

    # Class 1: high freq (freq ~ 5-8 cycles)
    freq = 5.0 + torch.rand(per_class, 1, generator=gen) * 3.0
    phase = torch.rand(per_class, 1, generator=gen) * 2 * math.pi
    amp = 0.5 + torch.rand(per_class, 1, generator=gen) * 0.5
    wave = amp * torch.sin(freq * t + phase)
    signals.append(wave)
    labels.append(torch.ones(per_class, dtype=torch.long))

    # Class 2: clipped sine (square-like)
    freq = 2.0 + torch.rand(per_class, 1, generator=gen) * 2.0
    phase = torch.rand(per_class, 1, generator=gen) * 2 * math.pi
    wave = torch.sign(torch.sin(freq * t + phase)) * 0.8
    signals.append(wave)
    labels.append(torch.full((per_class,), 2, dtype=torch.long))

    all_signals = torch.cat(signals, dim=0)  # (n, seq_len)
    noise = torch.randn_like(all_signals) * noise_std
    all_signals = all_signals + noise
    # Add channel dim: (n, 1, seq_len)
    all_signals = all_signals.unsqueeze(1)

    all_labels = torch.cat(labels, dim=0)
    return all_signals, all_labels


def build_datamodule(cfg: DictConfig) -> dict[str, Any]:
    num_samples = int(cfg.data.num_samples)
    seq_len = int(cfg.data.seq_len)
    noise_std = float(cfg.data.get("noise_std", 0.1))
    seed = int(cfg.seed)

    gen = torch.Generator().manual_seed(seed)

    features, labels = _generate_waveforms(num_samples, seq_len, gen, noise_std)

    perm = torch.randperm(len(features), generator=gen)
    features, labels = features[perm], labels[perm]

    val_split = int(cfg.data.val_split)
    x_train, y_train = features[:-val_split], labels[:-val_split]
    x_val, y_val = features[-val_split:], labels[-val_split:]

    batch_size = int(cfg.compute.batch_size)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, drop_last=False)

    return {"train": train_loader, "val": val_loader}
