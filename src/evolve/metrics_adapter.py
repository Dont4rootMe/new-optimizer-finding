"""Adapters from evaluator JSON into normalized experiment scoring fields."""

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


def extract_metrics(
    payload: dict[str, Any],
    exp_cfg: dict[str, Any],
    scoring_cfg: dict[str, Any],
    baseline_profile: dict[str, Any] | None = None,
    baseline_error: str | None = None,
) -> dict[str, Any]:
    """Normalize payload to exp_score/time/steps/status plus supporting fields."""

    del scoring_cfg

    direction = str(payload.get("objective_direction") or "min").lower()
    if direction not in {"max", "min"}:
        direction = "min"

    status = str(payload.get("status", "ok"))
    if status not in {"ok", "failed", "partial", "timeout", "skipped", "interrupted"}:
        status = "failed"

    raw_metric = _safe_float(payload.get("objective_last"))

    time_sec = _safe_float(payload.get("time_sec"))
    if time_sec is None:
        time_sec = _safe_float(payload.get("wall_time_sec"))

    steps = _safe_int(payload.get("steps"))
    if steps is None:
        steps = _safe_int(payload.get("iters"))

    objective_name = str(payload.get("objective_name") or "train_loss")
    if status == "ok" and objective_name != "train_loss":
        status = "failed"
    if status == "ok" and direction != "min":
        status = "failed"
    if status == "ok" and raw_metric is None:
        status = "failed"

    error_msg = payload.get("error_msg")
    if status == "ok" and baseline_error is not None:
        status = "failed"
        error_msg = baseline_error

    normalization_cfg = exp_cfg.get("normalization", {}) if isinstance(exp_cfg, dict) else {}
    if not isinstance(normalization_cfg, dict):
        normalization_cfg = {}

    eps = _safe_float(normalization_cfg.get("eps"))
    if eps is None or eps <= 0:
        eps = 1.0e-8

    baseline_last = None
    baseline_total_steps = None
    if baseline_profile is not None:
        baseline_last = _safe_float(baseline_profile.get("objective_last"))
        baseline_total_steps = _safe_float(baseline_profile.get("steps"))

    if status == "ok" and (baseline_last is None or baseline_total_steps is None or baseline_total_steps <= 0):
        status = "failed"
        error_msg = f"invalid baseline profile for experiment '{exp_cfg.get('name', 'unknown')}'"

    quality_ratio: float | None = None
    if status == "ok" and raw_metric is not None and baseline_last is not None:
        quality_ratio = baseline_last / max(raw_metric, eps)
        quality_ratio = max(float(quality_ratio), 0.0)

    steps_ratio: float | None = None
    if status == "ok":
        first_step = _safe_int(payload.get("first_step_at_or_below_baseline"))
        if first_step is None or first_step <= 0:
            steps_ratio = 0.0
        else:
            assert baseline_total_steps is not None
            steps_ratio = max(baseline_total_steps / max(float(first_step), eps), 0.0)

    exp_score: float | None = None
    if status == "ok" and quality_ratio is not None and steps_ratio is not None:
        if quality_ratio <= 0.0 or steps_ratio <= 0.0:
            exp_score = 0.0
        else:
            exp_score = (2.0 * quality_ratio * steps_ratio) / max(quality_ratio + steps_ratio, eps)

    if status == "ok" and exp_score is None:
        status = "failed"

    return {
        "raw_metric": raw_metric,
        "metric_direction": direction,
        "quality_ratio": quality_ratio,
        "steps_ratio": steps_ratio,
        "exp_score": exp_score,
        "time_sec": time_sec,
        "steps": steps,
        "status": status,
        "error_msg": error_msg,
    }
