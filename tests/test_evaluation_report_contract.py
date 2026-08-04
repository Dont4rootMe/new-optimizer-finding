"""Shared evaluator report contract tests."""

from __future__ import annotations

import pytest

from src.validate.report_contract import validate_evaluation_report


def test_success_report_requires_finite_numeric_score() -> None:
    assert validate_evaluation_report(
        {"status": "ok", "score": 1.25}, experiment_name="task"
    )["score"] == 1.25
    for bad_score in (None, True, float("nan"), float("inf"), -float("inf"), "not-a-number"):
        with pytest.raises(ValueError):
            validate_evaluation_report(
                {"status": "ok", "score": bad_score}, experiment_name="task"
            )


def test_failed_report_may_have_null_score_but_not_infinity() -> None:
    report = {"status": "failed", "score": None}
    assert validate_evaluation_report(report, experiment_name="task") is report
    with pytest.raises(ValueError, match="finite"):
        validate_evaluation_report(
            {"status": "failed", "score": float("inf")},
            experiment_name="task",
        )


def test_report_rejects_unknown_status_and_missing_score() -> None:
    with pytest.raises(ValueError, match="unsupported status"):
        validate_evaluation_report(
            {"status": "mystery", "score": 1.0}, experiment_name="task"
        )
    with pytest.raises(ValueError, match="missing required field"):
        validate_evaluation_report({"status": "ok"}, experiment_name="task")
