"""Unit tests for optimization-survey score computation."""

from __future__ import annotations

import pytest

from experiments.optimization_survey._runtime.scoring import compute_score


def test_compute_score_harmonic_mean_of_quality_and_speed() -> None:
    score, error = compute_score(
        report={
            "status": "ok",
            "objective_name": "train_loss",
            "objective_direction": "min",
            "objective_last": 0.8,
            "first_step_at_or_below_baseline": 50,
        },
        experiment_name="exp_a",
        baseline_profile={"objective_last": 1.2, "steps": 100},
        normalization_cfg={"eps": 1.0e-8},
    )

    assert error is None
    assert score == pytest.approx(12.0 / 7.0)


def test_compute_score_returns_zero_when_baseline_never_beaten() -> None:
    score, error = compute_score(
        report={
            "status": "ok",
            "objective_name": "train_loss",
            "objective_direction": "min",
            "objective_last": 30.0,
            "first_step_at_or_below_baseline": None,
        },
        experiment_name="exp_b",
        baseline_profile={"objective_last": 45.0, "steps": 100},
        normalization_cfg={"eps": 1.0e-8},
    )

    assert error is None
    assert score == pytest.approx(0.0)


def test_compute_score_requires_baseline_profile() -> None:
    score, error = compute_score(
        report={
            "status": "ok",
            "objective_name": "train_loss",
            "objective_direction": "min",
            "objective_last": 0.0,
        },
        experiment_name="exp_c",
        baseline_profile=None,
        normalization_cfg={"eps": 1.0e-6},
    )

    assert score is None
    assert error == "missing baseline profile for experiment 'exp_c'"
