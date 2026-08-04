"""Task-blind evaluator report validation shared by every experiment family."""

from __future__ import annotations

import math
from typing import Any


VALID_REPORT_STATUSES = {
    "ok",
    "failed",
    "partial",
    "timeout",
    "skipped",
    "interrupted",
}


def validate_evaluation_report(report: object, *, experiment_name: str) -> dict[str, Any]:
    """Validate the common ``evaluate_organism`` report contract.

    A successful report must carry a finite numeric score. Non-success reports
    may use ``score=None``; any non-null score is still required to be finite so
    infinities cannot enter population selection through a custom evaluator.
    """

    if not isinstance(report, dict):
        raise TypeError(
            f"Experiment '{experiment_name}' returned non-dict payload: "
            f"{type(report).__name__}"
        )
    if "score" not in report:
        raise ValueError(
            f"Experiment '{experiment_name}' report is missing required field 'score'."
        )

    status = str(report.get("status", "ok"))
    if status not in VALID_REPORT_STATUSES:
        raise ValueError(
            f"Experiment '{experiment_name}' returned unsupported status {status!r}."
        )

    score = report.get("score")
    if score is None:
        if status == "ok":
            raise ValueError(
                f"Experiment '{experiment_name}' returned status='ok' with score=None."
            )
        return report
    if isinstance(score, bool):
        raise ValueError(
            f"Experiment '{experiment_name}' score must be numeric, not bool."
        )
    try:
        numeric_score = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Experiment '{experiment_name}' score must be numeric, got {score!r}."
        ) from exc
    if not math.isfinite(numeric_score):
        raise ValueError(
            f"Experiment '{experiment_name}' score must be finite, got {score!r}."
        )
    return report
