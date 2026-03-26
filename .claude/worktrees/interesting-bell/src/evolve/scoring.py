"""Score aggregation helpers for optimizer candidates."""

from __future__ import annotations

from typing import Any

from src.evolve.metrics_adapter import extract_metrics


def mean_score(
    eval_results: dict[str, dict[str, Any]],
    selected_experiments: list[str],
    experiment_cfgs: dict[str, dict[str, Any]],
    allocation_pi: dict[str, float],
    scoring_cfg: dict[str, Any],
) -> tuple[float | None, str, dict[str, dict[str, Any]]]:
    """Aggregate candidate quality using weighted subset mean over exp_score."""

    per_experiment: dict[str, dict[str, Any]] = {}
    weighted_sum = 0.0
    weight_sum = 0.0
    successful = 0

    for exp_name in selected_experiments:
        payload = eval_results.get(exp_name)
        if payload is None:
            normalized = {
                "raw_metric": None,
                "metric_direction": str(experiment_cfgs.get(exp_name, {}).get("primary_metric", {}).get("direction", "max")),
                "quality_ratio": None,
                "steps_ratio": None,
                "exp_score": None,
                "time_sec": None,
                "steps": None,
                "status": "failed",
                "error_msg": "missing result",
            }
        else:
            normalized = extract_metrics(
                payload=payload,
                exp_cfg=experiment_cfgs.get(exp_name, {}),
                scoring_cfg=scoring_cfg,
            )

        per_experiment[exp_name] = normalized

        score_value = normalized.get("exp_score")
        if normalized["status"] == "ok" and score_value is not None:
            pi_value = allocation_pi.get(exp_name)
            try:
                pi = float(pi_value)
            except (TypeError, ValueError):
                pi = 0.0
            if pi <= 0:
                pi = 1.0

            weighted_sum += pi * float(score_value)
            weight_sum += pi
            successful += 1

    if successful == 0:
        return None, "failed", per_experiment

    aggregate = weighted_sum / max(weight_sum, 1.0e-12)
    if successful == len(selected_experiments):
        status = "ok"
    else:
        status = "partial"

    return float(aggregate), status, per_experiment
