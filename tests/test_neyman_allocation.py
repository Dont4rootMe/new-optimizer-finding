"""Unit tests for canonical Neyman allocation and organism-only history loading."""

from __future__ import annotations

import json
from pathlib import Path

from src.evolve.allocation import (
    build_allocation_snapshot,
    compute_conditional_inclusion_probabilities,
    compute_experiment_stats,
    compute_inclusion_probabilities,
    compute_neyman_weights,
    compute_nonempty_probability,
    sample_experiments_poisson,
)


def test_compute_neyman_weights_with_costs() -> None:
    stats = {
        "exp_a": {"mean": 0.0, "std": 2.0, "n": 10, "cost": 1.0},
        "exp_b": {"mean": 0.0, "std": 1.0, "n": 10, "cost": 4.0},
    }
    weights = compute_neyman_weights(
        stats=stats,
        min_history_for_variance=3,
        std_floor=1.0e-6,
        fallback="uniform",
    )

    assert abs(weights["exp_a"] - 0.8) < 1.0e-9
    assert abs(weights["exp_b"] - 0.2) < 1.0e-9


def test_compute_neyman_weights_fallback_uniform_for_sparse_history() -> None:
    stats = {
        "exp_a": {"mean": 0.0, "std": 1.0e-6, "n": 0, "cost": 1.0},
        "exp_b": {"mean": 0.0, "std": 1.0e-6, "n": 1, "cost": 1.0},
        "exp_c": {"mean": 0.0, "std": 1.0e-6, "n": 2, "cost": 1.0},
    }
    weights = compute_neyman_weights(
        stats=stats,
        min_history_for_variance=3,
        std_floor=1.0e-6,
        fallback="uniform",
    )

    assert abs(weights["exp_a"] - (1.0 / 3.0)) < 1.0e-9
    assert abs(weights["exp_b"] - (1.0 / 3.0)) < 1.0e-9
    assert abs(weights["exp_c"] - (1.0 / 3.0)) < 1.0e-9


def test_compute_inclusion_probabilities() -> None:
    inclusion_prob = compute_inclusion_probabilities(
        {"exp_a": 0.8, "exp_b": 0.2},
        sample_size=2,
    )

    assert inclusion_prob == {"exp_a": 1.0, "exp_b": 0.4}


def test_sample_poisson_is_deterministic_and_never_returns_empty() -> None:
    inclusion_prob = {"exp_a": 0.05, "exp_b": 0.04, "exp_c": 0.03}
    picked_a = sample_experiments_poisson(
        experiments=["exp_a", "exp_b", "exp_c"],
        inclusion_prob=inclusion_prob,
        seed=123,
    )
    picked_b = sample_experiments_poisson(
        experiments=["exp_a", "exp_b", "exp_c"],
        inclusion_prob=inclusion_prob,
        seed=123,
    )

    assert picked_a == picked_b
    assert picked_a
    assert set(picked_a).issubset({"exp_a", "exp_b", "exp_c"})


def test_compute_nonempty_probability_is_hand_checkable() -> None:
    prob_nonempty = compute_nonempty_probability({"exp_a": 0.2, "exp_b": 0.5})

    assert abs(prob_nonempty - 0.6) < 1.0e-9


def test_conditional_inclusion_probabilities_scale_by_nonempty_probability() -> None:
    base = {"exp_a": 0.2, "exp_b": 0.5}
    prob_nonempty = compute_nonempty_probability(base)
    conditional = compute_conditional_inclusion_probabilities(base, prob_nonempty)

    assert conditional["exp_a"] == base["exp_a"] / prob_nonempty
    assert conditional["exp_b"] == base["exp_b"] / prob_nonempty


def test_full_evaluation_snapshot_is_explicit() -> None:
    snapshot = build_allocation_snapshot(
        population_root="/tmp/unused",
        experiments=["exp_a", "exp_b"],
        allocation_cfg={"enabled": False},
        seed=1,
        entity_id="org_full",
    )

    assert snapshot["sampling_design"] == "full_evaluation"
    assert snapshot["base_inclusion_prob"] == {"exp_a": 1.0, "exp_b": 1.0}
    assert snapshot["inclusion_prob"] == {"exp_a": 1.0, "exp_b": 1.0}
    assert snapshot["prob_nonempty"] == 1.0
    assert snapshot["selected_experiments"] == ["exp_a", "exp_b"]


def test_sample_poisson_no_longer_force_picks_argmax_after_empty_draw() -> None:
    picked = sample_experiments_poisson(
        experiments=["exp_a", "exp_b", "exp_c"],
        inclusion_prob={"exp_a": 0.05, "exp_b": 0.04, "exp_c": 0.03},
        seed=6,
    )

    assert picked == ["exp_b"]


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


def test_allocation_history_ignores_noncanonical_candidate_dirs(tmp_path: Path) -> None:
    pop_root = tmp_path / "populations"
    candidate_dir = pop_root / "gen_0000" / "cand_deadbeef"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "summary.json").write_text(
        json.dumps(
            {
                "candidate_id": "deadbeef",
                "aggregate_score": 9.9,
                "experiments": {
                    "exp_a": {"status": "ok", "exp_score": 9.0},
                    "exp_b": {"status": "ok", "exp_score": 1.0},
                },
            }
        ),
        encoding="utf-8",
    )

    organism_dir = pop_root / "gen_0001" / "island_gradient_methods" / "org_alive"
    organism_dir.mkdir(parents=True, exist_ok=True)
    (organism_dir / "summary.json").write_text(
        json.dumps(
            {
                "phase_results": {
                    "simple": {
                        "experiments": {
                            "exp_a": {"status": "ok", "exp_score": 1.0},
                            "exp_b": {"status": "ok", "exp_score": 3.0},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    snapshot = build_allocation_snapshot(
        population_root=str(pop_root),
        experiments=["exp_a", "exp_b"],
        allocation_cfg={
            "enabled": True,
            "method": "neyman",
            "history_window": 10,
            "sample_size": 1,
            "min_history_for_variance": 1,
            "std_floor": 1.0e-6,
            "fallback": "uniform",
            "costs": {"exp_a": 1.0, "exp_b": 1.0},
        },
        seed=7,
        entity_id="org_alive",
    )

    assert "pi" not in snapshot
    assert snapshot["sampling_design"] == "conditional_poisson_nonempty"
    assert "base_inclusion_prob" in snapshot
    assert "prob_nonempty" in snapshot
    assert (
        snapshot["inclusion_prob"]["exp_a"]
        == snapshot["base_inclusion_prob"]["exp_a"] / snapshot["prob_nonempty"]
    )
    assert (
        snapshot["inclusion_prob"]["exp_b"]
        == snapshot["base_inclusion_prob"]["exp_b"] / snapshot["prob_nonempty"]
    )
    assert snapshot["stats"]["exp_a"]["mean"] == 1.0
    assert snapshot["stats"]["exp_b"]["mean"] == 3.0
