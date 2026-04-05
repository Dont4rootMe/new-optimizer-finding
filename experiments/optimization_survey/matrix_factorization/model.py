"""Matrix factorization model."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class MatrixFactorizationModel(nn.Module):
    """Factorize a target matrix A ~ U @ V^T.

    Loss = MSE(U @ V^T, A).
    """

    def __init__(self, matrix_dim: int, rank: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)

        # Target matrix: random low-rank
        w_true = torch.randn(matrix_dim, rank, generator=gen)
        h_true = torch.randn(matrix_dim, rank, generator=gen)
        target = w_true @ h_true.T
        self.register_buffer("target", target)

        # Learnable factors
        self.U = nn.Parameter(torch.randn(matrix_dim, rank, generator=gen) * 0.1)
        self.V = nn.Parameter(torch.randn(matrix_dim, rank, generator=gen) * 0.1)

    def forward(self) -> torch.Tensor:
        reconstruction = self.U @ self.V.T
        return torch.mean((reconstruction - self.target) ** 2)


def build_model(cfg: DictConfig) -> nn.Module:
    matrix_dim = int(cfg.model.matrix_dim)
    rank = int(cfg.model.rank)
    seed = int(cfg.seed)
    return MatrixFactorizationModel(matrix_dim, rank, seed)
