"""Metric helpers for MNIST MLP experiment."""

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
    """Check if configured target for accuracy is reached."""
    val_acc = float(metrics_dict["val_acc"])
    return val_acc >= float(target_cfg.value)
