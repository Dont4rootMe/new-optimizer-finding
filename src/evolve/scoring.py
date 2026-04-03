"""Aggregate experiment scores using the effective conditional inclusion probabilities."""

from __future__ import annotations

from typing import Any

from src.evolve.metrics_adapter import extract_metrics


def mean_score(
    eval_results: dict[str, dict[str, Any]],
    selected_experiments: list[str],
    experiment_cfgs: dict[str, dict[str, Any]],
    baseline_profiles: dict[str, dict[str, Any]],
    baseline_errors: dict[str, str],
    inclusion_prob: dict[str, float],
    total_experiments: int,
    scoring_cfg: dict[str, Any],
) -> tuple[float | None, str, dict[str, dict[str, Any]]]:
    """Estimate the full-experiment mean of `exp_score`.

    The estimator is:

        (1 / N) * sum_{i in selected} exp_score_i / pi_i

    where `N` is the full phase experiment count and `pi_i` is the inclusion
    probability from the conditional non-empty sampling design recorded in the
    allocation snapshot. Selected failed experiments contribute zero.
    """

    per_experiment: dict[str, dict[str, Any]] = {}
    weighted_sum = 0.0
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
                baseline_profile=baseline_profiles.get(exp_name),
                baseline_error=baseline_errors.get(exp_name),
            )

        per_experiment[exp_name] = normalized

        score_value = normalized.get("exp_score")
        if normalized["status"] == "ok" and score_value is not None:
            try:
                q_i = float(inclusion_prob.get(exp_name))
            except (TypeError, ValueError):
                q_i = 0.0
            if q_i <= 0.0:
                raise ValueError(
                    f"Inclusion probability for selected experiment '{exp_name}' must be > 0, got {q_i}"
                )

            weighted_sum += float(score_value) / q_i
            successful += 1

    if successful == 0:
        return None, "failed", per_experiment

    aggregate = weighted_sum / max(1, int(total_experiments))
    status = "ok" if successful == len(selected_experiments) else "partial"
    return float(aggregate), status, per_experiment
