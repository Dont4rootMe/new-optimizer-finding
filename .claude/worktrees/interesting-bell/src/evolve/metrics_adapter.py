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


def _normalize_score_weights(scoring_cfg: dict[str, Any]) -> tuple[float, float]:
    weights_cfg = scoring_cfg.get("score_weights", {})
    if not isinstance(weights_cfg, dict):
        weights_cfg = {}

    w_quality = _safe_float(weights_cfg.get("quality"))
    w_steps = _safe_float(weights_cfg.get("steps"))
    if w_quality is None:
        w_quality = 0.5
    if w_steps is None:
        w_steps = 0.5

    w_quality = max(w_quality, 0.0)
    w_steps = max(w_steps, 0.0)

    total = w_quality + w_steps
    if total <= 0:
        return 0.5, 0.5
    return w_quality / total, w_steps / total


def extract_metrics(
    payload: dict[str, Any],
    exp_cfg: dict[str, Any],
    scoring_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Normalize payload to exp_score/time/steps/status plus supporting fields."""

    direction = str(
        exp_cfg.get("primary_metric", {}).get("direction")
        or payload.get("metric_direction")
        or "max"
    ).lower()
    if direction not in {"max", "min"}:
        direction = "max"

    status = str(payload.get("status", "ok"))
    if status not in {"ok", "failed", "partial", "timeout", "skipped"}:
        status = "failed"

    raw_metric = _safe_float(payload.get("raw_metric"))
    if raw_metric is None:
        final_score_fallback = _safe_float(payload.get("final_score"))
        if final_score_fallback is not None:
            raw_metric = final_score_fallback if direction == "max" else -final_score_fallback

    time_sec = _safe_float(payload.get("time_sec"))
    if time_sec is None:
        time_sec = _safe_float(payload.get("wall_time_sec"))

    steps = _safe_int(payload.get("steps"))
    if steps is None:
        steps = _safe_int(payload.get("iters"))

    if status == "ok" and raw_metric is None:
        status = "failed"

    normalization_cfg = exp_cfg.get("normalization", {}) if isinstance(exp_cfg, dict) else {}
    if not isinstance(normalization_cfg, dict):
        normalization_cfg = {}

    eps = _safe_float(normalization_cfg.get("eps"))
    if eps is None or eps <= 0:
        eps = 1.0e-8

    quality_ref = _safe_float(normalization_cfg.get("quality_ref"))
    if quality_ref is None:
        quality_ref = abs(raw_metric) if raw_metric is not None else 1.0
    quality_ref = max(quality_ref, eps)

    steps_ref = _safe_float(normalization_cfg.get("steps_ref"))
    if steps_ref is None:
        steps_ref = float(steps if steps is not None else 1.0)
    steps_ref = max(steps_ref, eps)

    quality_ratio: float | None = None
    if raw_metric is not None:
        if direction == "max":
            quality_ratio = raw_metric / quality_ref
        else:
            quality_ratio = quality_ref / max(raw_metric, eps)
        quality_ratio = max(float(quality_ratio), 0.0)

    steps_ratio: float | None = None
    if steps is not None:
        steps_ratio = max(steps_ref / max(float(steps), eps), 0.0)

    exp_score: float | None = None
    if status == "ok" and quality_ratio is not None and steps_ratio is not None:
        w_quality, w_steps = _normalize_score_weights(scoring_cfg)
        exp_score = (w_quality * quality_ratio) + (w_steps * steps_ratio)

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
        "error_msg": payload.get("error_msg"),
    }
