"""Small UNet model for CIFAR-10 DDPM."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from omegaconf import DictConfig


def _group_count(channels: int) -> int:
    return 32 if channels % 32 == 0 else 16 if channels % 16 == 0 else 8


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for discrete diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * exponent)
        args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    """Residual block with time conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        time_bias = self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_bias
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Spatial self-attention block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        h = self.norm(x)
        q = self.q(h).view(bsz, channels, height * width).transpose(1, 2)
        k = self.k(h).view(bsz, channels, height * width)
        v = self.v(h).view(bsz, channels, height * width).transpose(1, 2)

        attn = torch.softmax(torch.bmm(q, k) * (channels ** -0.5), dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).view(bsz, channels, height, width)
        return x + self.proj(out)


class SmallUNet(nn.Module):
    """Compact U-Net for 32x32 noise prediction."""

    def __init__(self, base_channels: int = 64, attention: bool = False) -> None:
        super().__init__()
        time_dim = base_channels * 4

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(3, base_channels, kernel_size=3, padding=1)

        self.down1 = ResBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)

        self.mid = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid_attn = AttentionBlock(base_channels * 4) if attention else nn.Identity()

        self.upsample1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up1 = ResBlock(base_channels * 4, base_channels * 2, time_dim)
        self.upsample2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up2 = ResBlock(base_channels * 2, base_channels, time_dim)

        self.out_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(timesteps)

        x0 = self.in_conv(x)
        x1 = self.down1(x0, t_emb)
        x2 = self.downsample1(x1)
        x3 = self.down2(x2, t_emb)
        x4 = self.downsample2(x3)

        mid = self.mid(x4, t_emb)
        mid = self.mid_attn(mid)

        u1 = self.upsample1(mid)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.up1(u1, t_emb)

        u2 = self.upsample2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.up2(u2, t_emb)

        out = self.out_conv(self.out_act(self.out_norm(u2)))
        return out


def build_model(cfg: DictConfig) -> nn.Module:
    """Build a compact U-Net model for DDPM training."""

    return SmallUNet(
        base_channels=int(cfg.model.base_channels),
        attention=bool(cfg.model.attention),
    )
