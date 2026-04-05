"""Metric helpers for MiniGPT experiment."""

from __future__ import annotations

import math
from typing import Any

from omegaconf import DictConfig


def compute_primary(metrics_dict: dict[str, Any], primary_cfg: DictConfig) -> float:
    """Extract primary metric from dict by configured name."""

    metric_name = str(primary_cfg.name)
    if metric_name not in metrics_dict:
        raise KeyError(f"Primary metric '{metric_name}' not found in {metrics_dict.keys()}")
    return float(metrics_dict[metric_name])


def perplexity_from_loss(loss: float) -> float:
    """Convert mean loss to perplexity, clipping to avoid overflow."""

    clipped = min(max(loss, 0.0), 50.0)
    return float(math.exp(clipped))


def is_target_reached(metrics_dict: dict[str, Any], target_cfg: DictConfig) -> bool:
    """Check if perplexity target is reached."""

    ppl = float(metrics_dict["val_ppl"])
    return ppl <= float(target_cfg.value)
