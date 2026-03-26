"""Unit tests for aggregate scoring behavior."""

from __future__ import annotations

from src.evolve.scoring import mean_score


def test_mean_score_all_ok() -> None:
    agg, status, per_exp = mean_score(
        eval_results={
            "exp_a": {"status": "ok", "raw_metric": 0.8, "time_sec": 1.0, "steps": 10},
            "exp_b": {"status": "ok", "raw_metric": 0.6, "time_sec": 1.5, "steps": 12},
        },
        selected_experiments=["exp_a", "exp_b"],
        experiment_cfgs={
            "exp_a": {"primary_metric": {"direction": "max"}, "normalization": {"quality_ref": 1.0, "steps_ref": 10}},
            "exp_b": {"primary_metric": {"direction": "max"}, "normalization": {"quality_ref": 1.0, "steps_ref": 10}},
        },
        allocation_pi={"exp_a": 0.75, "exp_b": 0.25},
        scoring_cfg={"score_weights": {"quality": 0.7, "steps": 0.3}},
    )
    assert agg is not None
    assert abs(agg - 0.8125) < 1.0e-9
    assert status == "ok"
    assert per_exp["exp_a"]["status"] == "ok"
    assert per_exp["exp_a"]["exp_score"] is not None


def test_mean_score_partial() -> None:
    agg, status, per_exp = mean_score(
        eval_results={
            "exp_a": {"status": "ok", "raw_metric": 0.9, "time_sec": 1.0, "steps": 5},
            "exp_b": {"status": "failed", "error_msg": "boom"},
        },
        selected_experiments=["exp_a", "exp_b"],
        experiment_cfgs={
            "exp_a": {"primary_metric": {"direction": "max"}, "normalization": {"quality_ref": 1.0, "steps_ref": 10}},
            "exp_b": {"primary_metric": {"direction": "max"}, "normalization": {"quality_ref": 1.0, "steps_ref": 10}},
        },
        allocation_pi={"exp_a": 0.6, "exp_b": 0.4},
        scoring_cfg={"score_weights": {"quality": 0.7, "steps": 0.3}},
    )
    assert agg is not None
    assert abs(agg - 1.23) < 1.0e-9
    assert status == "partial"
    assert per_exp["exp_b"]["status"] in {"failed", "timeout"}


def test_mean_score_all_failed() -> None:
    agg, status, per_exp = mean_score(
        eval_results={
            "exp_a": {"status": "failed", "error_msg": "x"},
            "exp_b": {"status": "failed", "error_msg": "y"},
        },
        selected_experiments=["exp_a", "exp_b"],
        experiment_cfgs={
            "exp_a": {"primary_metric": {"direction": "max"}, "normalization": {"quality_ref": 1.0, "steps_ref": 10}},
            "exp_b": {"primary_metric": {"direction": "max"}, "normalization": {"quality_ref": 1.0, "steps_ref": 10}},
        },
        allocation_pi={"exp_a": 0.5, "exp_b": 0.5},
        scoring_cfg={"score_weights": {"quality": 0.7, "steps": 0.3}},
    )
    assert agg is None
    assert status == "failed"
    assert set(per_exp.keys()) == {"exp_a", "exp_b"}
