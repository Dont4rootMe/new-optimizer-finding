"""Model definition for CIFAR-10 experiment."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig
from torchvision.models import resnet18


def build_model(cfg: DictConfig) -> nn.Module:
    """Build CIFAR-adapted ResNet-18."""

    num_classes = int(cfg.model.num_classes)
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
