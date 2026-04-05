"""Aggregate experiment scores using effective conditional inclusion probabilities."""

from __future__ import annotations

from typing import Any


def mean_score(
    eval_results: dict[str, dict[str, Any]],
    selected_experiments: list[str],
    inclusion_prob: dict[str, float],
    total_experiments: int,
) -> tuple[float | None, str, dict[str, dict[str, Any]]]:
    """Estimate the full-experiment mean of experiment-reported scores."""

    per_experiment: dict[str, dict[str, Any]] = {}
    weighted_sum = 0.0
    successful = 0

    for exp_name in selected_experiments:
        payload = eval_results.get(exp_name)
        if payload is None:
            normalized = {
                "score": None,
                "status": "failed",
                "error_msg": "missing result",
            }
        else:
            status = str(payload.get("status", "ok"))
            score = payload.get("score")
            try:
                parsed_score = float(score) if score is not None else None
            except (TypeError, ValueError):
                parsed_score = None
            if parsed_score is not None and parsed_score != parsed_score:
                parsed_score = None
            if status not in {"ok", "failed", "partial", "timeout", "skipped", "interrupted"}:
                status = "failed"
            normalized = {
                "score": parsed_score,
                "status": status,
                "error_msg": payload.get("error_msg"),
                "raw_report": payload,
            }

        per_experiment[exp_name] = normalized

        score_value = normalized.get("score")
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
