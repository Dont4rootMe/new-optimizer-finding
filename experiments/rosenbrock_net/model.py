"""Rosenbrock parametric model."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class RosenbrockModel(nn.Module):
    """Parameters that minimize the Rosenbrock function.

    Rosenbrock(p) = sum_{i=0}^{n/2-1} [100*(p[2i+1] - p[2i]^2)^2 + (1 - p[2i])^2]
    Global minimum = 0 at p = [1, 1, ..., 1].
    """

    def __init__(self, num_params: int) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.zeros(num_params))

    def forward(self) -> torch.Tensor:
        p = self.p
        n = p.shape[0]
        even = p[0 : n - 1 : 2]
        odd = p[1:n:2]
        return torch.sum(100.0 * (odd - even ** 2) ** 2 + (1.0 - even) ** 2)


def build_model(cfg: DictConfig) -> nn.Module:
    num_params = int(cfg.model.num_params)
    return RosenbrockModel(num_params)
