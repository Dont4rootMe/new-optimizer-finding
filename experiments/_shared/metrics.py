"""Generic metric helpers for experiments."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig


def compute_primary(metrics_dict: dict[str, Any], primary_cfg: DictConfig) -> float:
    """Extract primary metric value from metric dictionary."""
    metric_name = str(primary_cfg.name)
    if metric_name not in metrics_dict:
        raise KeyError(f"Primary metric '{metric_name}' not found in metrics: {metrics_dict.keys()}")
    return float(metrics_dict[metric_name])


def is_target_reached(metrics_dict: dict[str, Any], target_cfg: DictConfig) -> bool:
    """Check if configured target is reached, respecting direction."""
    metric_name = str(target_cfg.metric)
    value = float(metrics_dict[metric_name])
    target_value = float(target_cfg.value)
    direction = str(target_cfg.get("direction", "max"))
    if direction == "min":
        return value <= target_value
    return value >= target_value
