"""Metric helpers for DDPM CIFAR-10 experiment."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig


def compute_primary(metrics_dict: dict[str, Any], primary_cfg: DictConfig) -> float:
    """Extract configured primary metric for DDPM."""

    metric_name = str(primary_cfg.name)
    if metric_name not in metrics_dict:
        raise KeyError(f"Primary metric '{metric_name}' not found in {metrics_dict.keys()}")
    return float(metrics_dict[metric_name])


def is_target_reached(metrics_dict: dict[str, Any], target_cfg: DictConfig) -> bool:
    """Evaluate absolute or relative target conditions for DDPM loss."""

    ema_loss = float(metrics_dict["ema_loss"])
    mode = str(target_cfg.get("mode", "absolute")).lower()

    if mode == "relative":
        start = metrics_dict.get("ema_loss_at_start")
        if start is None:
            return False
        improvement_ratio = float(target_cfg.get("improvement_ratio", 0.0))
        threshold = (1.0 - improvement_ratio) * float(start)
        return ema_loss <= threshold

    return ema_loss <= float(target_cfg.value)
