"""Unit tests for normalized experiment metric extraction."""

from __future__ import annotations

import math

from src.evolve.metrics_adapter import extract_metrics


def test_extract_metrics_direction_max() -> None:
    payload = {
        "status": "ok",
        "objective_name": "train_loss",
        "objective_direction": "min",
        "objective_last": 0.8,
        "first_step_at_or_below_baseline": 50,
        "steps": 50,
        "time_sec": 1.0,
    }
    exp_cfg = {
        "name": "exp_a",
        "normalization": {"eps": 1.0e-8},
    }
    baseline_profile = {"objective_name": "train_loss", "objective_direction": "min", "objective_last": 1.2, "steps": 100}

    out = extract_metrics(payload=payload, exp_cfg=exp_cfg, scoring_cfg={}, baseline_profile=baseline_profile)

    assert out["status"] == "ok"
    assert abs(float(out["quality_ratio"]) - 1.5) < 1.0e-9
    assert abs(float(out["steps_ratio"]) - 2.0) < 1.0e-9
    assert abs(float(out["exp_score"]) - (12.0 / 7.0)) < 1.0e-9


def test_extract_metrics_direction_min() -> None:
    payload = {
        "status": "ok",
        "objective_name": "train_loss",
        "objective_direction": "min",
        "objective_last": 30.0,
        "first_step_at_or_below_baseline": None,
        "steps": 100,
        "time_sec": 1.0,
    }
    exp_cfg = {
        "name": "exp_b",
        "normalization": {"eps": 1.0e-8},
    }
    baseline_profile = {"objective_name": "train_loss", "objective_direction": "min", "objective_last": 45.0, "steps": 100}

    out = extract_metrics(payload=payload, exp_cfg=exp_cfg, scoring_cfg={}, baseline_profile=baseline_profile)

    assert out["status"] == "ok"
    assert abs(float(out["quality_ratio"]) - 1.5) < 1.0e-9
    assert abs(float(out["steps_ratio"]) - 0.0) < 1.0e-9
    assert abs(float(out["exp_score"]) - 0.0) < 1.0e-9


def test_extract_metrics_baseline_error_fails() -> None:
    payload = {
        "status": "ok",
        "objective_name": "train_loss",
        "objective_direction": "min",
        "objective_last": 0.0,
        "steps": 0,
    }
    exp_cfg = {
        "name": "exp_c",
        "normalization": {"eps": 1.0e-6},
    }
    out = extract_metrics(
        payload=payload,
        exp_cfg=exp_cfg,
        scoring_cfg={},
        baseline_profile=None,
        baseline_error="missing baseline",
    )

    assert out["status"] == "failed"
    assert out["exp_score"] is None
    assert out["error_msg"] == "missing baseline"
