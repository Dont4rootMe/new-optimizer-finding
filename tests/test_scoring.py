"""Unit tests for score-only aggregate scoring behavior."""

from __future__ import annotations

import pytest

from src.evolve.scoring import mean_score


def test_mean_score_all_ok() -> None:
    agg, status, per_exp = mean_score(
        eval_results={
            "exp_a": {"status": "ok", "score": 2.0, "extra": "a"},
            "exp_b": {"status": "ok", "score": 4.0, "extra": "b"},
        },
        selected_experiments=["exp_a", "exp_b"],
        inclusion_prob={"exp_a": 1.0, "exp_b": 1.0},
        total_experiments=2,
    )

    assert agg == pytest.approx(3.0)
    assert status == "ok"
    assert per_exp["exp_a"]["status"] == "ok"
    assert per_exp["exp_a"]["score"] == pytest.approx(2.0)
    assert per_exp["exp_a"]["raw_report"]["extra"] == "a"


def test_mean_score_partial() -> None:
    agg, status, per_exp = mean_score(
        eval_results={
            "exp_a": {"status": "ok", "score": 1.0},
            "exp_b": {"status": "failed", "error_msg": "boom"},
        },
        selected_experiments=["exp_a", "exp_b"],
        inclusion_prob={"exp_a": 0.5, "exp_b": 0.5},
        total_experiments=2,
    )

    assert agg == pytest.approx(1.0)
    assert status == "partial"
    assert per_exp["exp_b"]["status"] == "failed"
    assert per_exp["exp_b"]["error_msg"] == "boom"


def test_mean_score_uses_effective_conditional_inclusion_probabilities() -> None:
    agg, status, _ = mean_score(
        eval_results={
            "exp_a": {"status": "ok", "score": 1.0},
        },
        selected_experiments=["exp_a"],
        inclusion_prob={"exp_a": 0.25},
        total_experiments=2,
    )

    assert status == "ok"
    assert agg == pytest.approx(2.0)


def test_mean_score_rejects_non_positive_inclusion_probability() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        mean_score(
            eval_results={
                "exp_a": {"status": "ok", "score": 1.0},
            },
            selected_experiments=["exp_a"],
            inclusion_prob={"exp_a": 0.0},
            total_experiments=2,
        )


def test_mean_score_all_failed() -> None:
    agg, status, per_exp = mean_score(
        eval_results={
            "exp_a": {"status": "failed", "error_msg": "x"},
            "exp_b": {"status": "failed", "error_msg": "y"},
        },
        selected_experiments=["exp_a", "exp_b"],
        inclusion_prob={"exp_a": 0.5, "exp_b": 0.5},
        total_experiments=2,
    )

    assert agg is None
    assert status == "failed"
    assert set(per_exp) == {"exp_a", "exp_b"}
