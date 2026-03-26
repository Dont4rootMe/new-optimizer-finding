"""Unit tests for Neyman allocation utilities."""

from __future__ import annotations

from src.evolve.allocation import (
    compute_experiment_stats,
    compute_neyman_pi,
    sample_experiments_without_replacement,
)


def test_compute_neyman_pi_with_costs() -> None:
    stats = {
        "exp_a": {"mean": 0.0, "std": 2.0, "n": 10, "cost": 1.0},
        "exp_b": {"mean": 0.0, "std": 1.0, "n": 10, "cost": 4.0},
    }
    pi = compute_neyman_pi(
        stats=stats,
        min_history_for_variance=3,
        std_floor=1.0e-6,
        fallback="uniform",
    )

    # raw weights: exp_a=2/sqrt(1)=2, exp_b=1/sqrt(4)=0.5
    assert abs(pi["exp_a"] - 0.8) < 1.0e-9
    assert abs(pi["exp_b"] - 0.2) < 1.0e-9


def test_compute_neyman_pi_fallback_uniform_for_sparse_history() -> None:
    stats = {
        "exp_a": {"mean": 0.0, "std": 1.0e-6, "n": 0, "cost": 1.0},
        "exp_b": {"mean": 0.0, "std": 1.0e-6, "n": 1, "cost": 1.0},
        "exp_c": {"mean": 0.0, "std": 1.0e-6, "n": 2, "cost": 1.0},
    }
    pi = compute_neyman_pi(
        stats=stats,
        min_history_for_variance=3,
        std_floor=1.0e-6,
        fallback="uniform",
    )

    assert abs(pi["exp_a"] - (1.0 / 3.0)) < 1.0e-9
    assert abs(pi["exp_b"] - (1.0 / 3.0)) < 1.0e-9
    assert abs(pi["exp_c"] - (1.0 / 3.0)) < 1.0e-9


def test_sample_without_replacement_respects_size() -> None:
    experiments = ["exp_a", "exp_b", "exp_c", "exp_d"]
    pi = {"exp_a": 0.7, "exp_b": 0.2, "exp_c": 0.05, "exp_d": 0.05}
    picked = sample_experiments_without_replacement(
        experiments=experiments,
        pi=pi,
        sample_size=2,
        seed=123,
    )

    assert len(picked) == 2
    assert len(set(picked)) == 2
    assert set(picked).issubset(set(experiments))


def test_compute_experiment_stats_costs_normalized() -> None:
    history = {"exp_a": [1.0, 2.0], "exp_b": [3.0, 4.0]}
    stats = compute_experiment_stats(
        history=history,
        experiments=["exp_a", "exp_b"],
        costs={"exp_a": 1.0, "exp_b": 3.0},
        std_floor=1.0e-6,
    )
    total_cost = float(stats["exp_a"]["cost"]) + float(stats["exp_b"]["cost"])
    assert abs(total_cost - 1.0) < 1.0e-9
