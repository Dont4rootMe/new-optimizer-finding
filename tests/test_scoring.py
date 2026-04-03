"""Unit tests for canonical aggregate scoring behavior."""

from __future__ import annotations

from src.evolve.scoring import mean_score


def test_mean_score_all_ok() -> None:
    agg, status, per_exp = mean_score(
        eval_results={
            "exp_a": {
                "status": "ok",
                "objective_name": "train_loss",
                "objective_direction": "min",
                "objective_last": 0.8,
                "first_step_at_or_below_baseline": 10,
                "time_sec": 1.0,
                "steps": 10,
            },
            "exp_b": {
                "status": "ok",
                "objective_name": "train_loss",
                "objective_direction": "min",
                "objective_last": 0.5,
                "first_step_at_or_below_baseline": 12,
                "time_sec": 1.5,
                "steps": 12,
            },
        },
        selected_experiments=["exp_a", "exp_b"],
        experiment_cfgs={
            "exp_a": {"name": "exp_a", "normalization": {"eps": 1.0e-8}},
            "exp_b": {"name": "exp_b", "normalization": {"eps": 1.0e-8}},
        },
        baseline_profiles={
            "exp_a": {"objective_name": "train_loss", "objective_direction": "min", "objective_last": 1.0, "steps": 20},
            "exp_b": {"objective_name": "train_loss", "objective_direction": "min", "objective_last": 0.75, "steps": 18},
        },
        baseline_errors={},
        inclusion_prob={"exp_a": 1.0, "exp_b": 1.0},
        total_experiments=2,
        scoring_cfg={},
    )
    assert agg is not None
    assert abs(agg - 1.5192307692307692) < 1.0e-9
    assert status == "ok"
    assert per_exp["exp_a"]["status"] == "ok"
    assert per_exp["exp_a"]["exp_score"] is not None


def test_mean_score_partial() -> None:
    agg, status, per_exp = mean_score(
        eval_results={
            "exp_a": {
                "status": "ok",
                "objective_name": "train_loss",
                "objective_direction": "min",
                "objective_last": 0.45,
                "first_step_at_or_below_baseline": 5,
                "time_sec": 1.0,
                "steps": 5,
            },
            "exp_b": {"status": "failed", "error_msg": "boom"},
        },
        selected_experiments=["exp_a", "exp_b"],
        experiment_cfgs={
            "exp_a": {"name": "exp_a", "normalization": {"eps": 1.0e-8}},
            "exp_b": {"name": "exp_b", "normalization": {"eps": 1.0e-8}},
        },
        baseline_profiles={
            "exp_a": {"objective_name": "train_loss", "objective_direction": "min", "objective_last": 0.9, "steps": 10},
            "exp_b": {"objective_name": "train_loss", "objective_direction": "min", "objective_last": 1.0, "steps": 10},
        },
        baseline_errors={},
        inclusion_prob={"exp_a": 0.5, "exp_b": 0.5},
        total_experiments=2,
        scoring_cfg={},
    )
    assert agg is not None
    assert abs(agg - 2.0) < 1.0e-9
    assert status == "partial"
    assert per_exp["exp_b"]["status"] in {"failed", "timeout"}


def test_mean_score_uses_effective_conditional_inclusion_probabilities() -> None:
    agg, status, _ = mean_score(
        eval_results={
            "exp_a": {
                "status": "ok",
                "objective_name": "train_loss",
                "objective_direction": "min",
                "objective_last": 0.45,
                "first_step_at_or_below_baseline": 5,
                "time_sec": 1.0,
                "steps": 5,
            }
        },
        selected_experiments=["exp_a"],
        experiment_cfgs={
            "exp_a": {"name": "exp_a", "normalization": {"eps": 1.0e-8}},
        },
        baseline_profiles={
            "exp_a": {"objective_name": "train_loss", "objective_direction": "min", "objective_last": 0.9, "steps": 10},
        },
        baseline_errors={},
        inclusion_prob={"exp_a": 0.25},
        total_experiments=2,
        scoring_cfg={},
    )

    assert status == "ok"
    assert agg is not None
    assert abs(agg - 4.0) < 1.0e-9


def test_mean_score_rejects_non_positive_inclusion_probability() -> None:
    try:
        mean_score(
            eval_results={
                "exp_a": {
                    "status": "ok",
                    "objective_name": "train_loss",
                    "objective_direction": "min",
                    "objective_last": 0.45,
                    "first_step_at_or_below_baseline": 5,
                    "time_sec": 1.0,
                    "steps": 5,
                }
            },
            selected_experiments=["exp_a"],
            experiment_cfgs={
                "exp_a": {"name": "exp_a", "normalization": {"eps": 1.0e-8}},
            },
            baseline_profiles={
                "exp_a": {"objective_name": "train_loss", "objective_direction": "min", "objective_last": 0.9, "steps": 10},
            },
            baseline_errors={},
            inclusion_prob={"exp_a": 0.0},
            total_experiments=2,
            scoring_cfg={},
        )
    except ValueError as exc:
        assert "must be > 0" in str(exc)
    else:
        raise AssertionError("Expected mean_score() to reject non-positive inclusion probabilities.")


def test_mean_score_all_failed() -> None:
    agg, status, per_exp = mean_score(
        eval_results={
            "exp_a": {"status": "failed", "error_msg": "x"},
            "exp_b": {"status": "failed", "error_msg": "y"},
        },
        selected_experiments=["exp_a", "exp_b"],
        experiment_cfgs={
            "exp_a": {"name": "exp_a", "normalization": {"eps": 1.0e-8}},
            "exp_b": {"name": "exp_b", "normalization": {"eps": 1.0e-8}},
        },
        baseline_profiles={
            "exp_a": {"objective_name": "train_loss", "objective_direction": "min", "objective_last": 1.0, "steps": 10},
            "exp_b": {"objective_name": "train_loss", "objective_direction": "min", "objective_last": 1.0, "steps": 10},
        },
        baseline_errors={},
        inclusion_prob={"exp_a": 0.5, "exp_b": 0.5},
        total_experiments=2,
        scoring_cfg={},
    )
    assert agg is None
    assert status == "failed"
    assert set(per_exp.keys()) == {"exp_a", "exp_b"}
