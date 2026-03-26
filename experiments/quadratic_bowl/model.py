"""Ill-conditioned quadratic bowl model."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class QuadraticBowlModel(nn.Module):
    """Minimize x^T A x + b^T x with controllable condition number.

    A = Q diag(eigenvalues) Q^T where eigenvalues are log-spaced.
    Minimum is at x* = -0.5 * A^{-1} b.
    """

    def __init__(self, dim: int, condition_number: float, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)

        # Random orthogonal Q
        raw = torch.randn(dim, dim, generator=gen)
        q, _ = torch.linalg.qr(raw)

        # Eigenvalues log-spaced from 1/cond to 1
        eigenvalues = torch.logspace(
            -torch.log10(torch.tensor(condition_number)).item(), 0.0, dim,
        )
        A = q @ torch.diag(eigenvalues) @ q.T
        b = torch.randn(dim, generator=gen)

        self.register_buffer("A", A)
        self.register_buffer("b", b)

        self.x = nn.Parameter(torch.randn(dim, generator=gen))

    def forward(self) -> torch.Tensor:
        return self.x @ self.A @ self.x + self.b @ self.x


def build_model(cfg: DictConfig) -> nn.Module:
    dim = int(cfg.model.dim)
    cond = float(cfg.model.condition_number)
    seed = int(cfg.seed)
    return QuadraticBowlModel(dim, cond, seed)
