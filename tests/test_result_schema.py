"""Tests for the generic experiment report contract."""

from __future__ import annotations

import pytest


def validate_experiment_report(payload: dict[str, object]) -> None:
    if "score" not in payload:
        raise ValueError("Experiment report is missing required field 'score'.")


def test_experiment_report_requires_score() -> None:
    validate_experiment_report({"status": "ok", "score": 1.23})

    with pytest.raises(ValueError, match="score"):
        validate_experiment_report({"status": "ok"})
