"""Optimization-survey score computation from raw training reports."""

from __future__ import annotations

import math
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if output != output or math.isinf(output):
        return None
    return output


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compute_score(
    report: dict[str, Any],
    *,
    experiment_name: str,
    baseline_profile: dict[str, Any] | None,
    normalization_cfg: dict[str, Any] | None,
) -> tuple[float | None, str | None]:
    """Compute the optimization-survey score from a raw experiment report."""

    if str(report.get("status", "ok")) != "ok":
        return None, report.get("error_msg")

    objective_name = str(report.get("objective_name") or "train_loss")
    objective_direction = str(report.get("objective_direction") or "min").lower()
    objective_last = _safe_float(report.get("objective_last"))
    first_step = _safe_int(report.get("first_step_at_or_below_baseline"))

    if objective_name != "train_loss":
        return None, f"experiment '{experiment_name}' must report objective_name='train_loss'"
    if objective_direction != "min":
        return None, f"experiment '{experiment_name}' must report objective_direction='min'"
    if objective_last is None:
        return None, f"experiment '{experiment_name}' must report finite objective_last"

    if baseline_profile is None:
        return None, f"missing baseline profile for experiment '{experiment_name}'"

    baseline_last = _safe_float(baseline_profile.get("objective_last"))
    baseline_steps = _safe_float(baseline_profile.get("steps"))
    if baseline_last is None or baseline_steps is None or baseline_steps <= 0:
        return None, f"invalid baseline profile for experiment '{experiment_name}'"

    normalization_cfg = normalization_cfg or {}
    eps = _safe_float(normalization_cfg.get("eps"))
    if eps is None or eps <= 0:
        eps = 1.0e-8

    quality_ratio = max(float(baseline_last / max(objective_last, eps)), 0.0)
    if first_step is None or first_step <= 0:
        steps_ratio = 0.0
    else:
        steps_ratio = max(float(baseline_steps / max(float(first_step), eps)), 0.0)

    if quality_ratio <= 0.0 or steps_ratio <= 0.0:
        return 0.0, None

    score = (2.0 * quality_ratio * steps_ratio) / max(quality_ratio + steps_ratio, eps)
    return float(score), None

