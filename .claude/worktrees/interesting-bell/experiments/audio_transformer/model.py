"""Model for tiny audio transformer keyword spotting."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class TinyAudioTransformer(nn.Module):
    """Small transformer encoder over log-mel time frames."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        num_classes: int,
        pooling: str = "mean",
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.pooling = pooling

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        if pooling == "attention":
            self.attn_pool = nn.Linear(d_model, 1)
        else:
            self.attn_pool = None

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(features)
        seq_len = x.size(1)
        x = x + self.pos_emb[:, :seq_len, :]
        x = self.encoder(x)
        x = self.norm(x)

        if self.attn_pool is not None:
            weights = torch.softmax(self.attn_pool(x).squeeze(-1), dim=1)
            pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        else:
            pooled = x.mean(dim=1)

        return self.classifier(pooled)


def build_model(cfg: DictConfig) -> nn.Module:
    """Build tiny transformer for Speech Commands classification."""

    return TinyAudioTransformer(
        input_dim=int(cfg.model.input_dim),
        d_model=int(cfg.model.d_model),
        n_layers=int(cfg.model.n_layers),
        n_heads=int(cfg.model.n_heads),
        d_ff=int(cfg.model.d_ff),
        dropout=float(cfg.model.dropout),
        num_classes=int(cfg.model.num_classes),
        pooling=str(cfg.model.pooling),
    )
