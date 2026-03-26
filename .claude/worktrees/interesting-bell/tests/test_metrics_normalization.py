"""Unit tests for normalized experiment metric extraction."""

from __future__ import annotations

import math

from src.evolve.metrics_adapter import extract_metrics


def test_extract_metrics_direction_max() -> None:
    payload = {"status": "ok", "raw_metric": 0.8, "steps": 50, "time_sec": 1.0}
    exp_cfg = {
        "primary_metric": {"direction": "max"},
        "normalization": {"quality_ref": 1.0, "steps_ref": 100.0, "eps": 1.0e-8},
    }
    scoring_cfg = {"score_weights": {"quality": 0.7, "steps": 0.3}}

    out = extract_metrics(payload=payload, exp_cfg=exp_cfg, scoring_cfg=scoring_cfg)

    assert out["status"] == "ok"
    assert abs(float(out["quality_ratio"]) - 0.8) < 1.0e-9
    assert abs(float(out["steps_ratio"]) - 2.0) < 1.0e-9
    assert abs(float(out["exp_score"]) - 1.16) < 1.0e-9


def test_extract_metrics_direction_min() -> None:
    payload = {"status": "ok", "raw_metric": 30.0, "steps": 100, "time_sec": 1.0}
    exp_cfg = {
        "primary_metric": {"direction": "min"},
        "normalization": {"quality_ref": 45.0, "steps_ref": 100.0, "eps": 1.0e-8},
    }
    scoring_cfg = {"score_weights": {"quality": 0.5, "steps": 0.5}}

    out = extract_metrics(payload=payload, exp_cfg=exp_cfg, scoring_cfg=scoring_cfg)

    assert out["status"] == "ok"
    assert abs(float(out["quality_ratio"]) - 1.5) < 1.0e-9
    assert abs(float(out["steps_ratio"]) - 1.0) < 1.0e-9
    assert abs(float(out["exp_score"]) - 1.25) < 1.0e-9


def test_extract_metrics_eps_guards_zero_values() -> None:
    payload = {"status": "ok", "raw_metric": 0.0, "steps": 0}
    exp_cfg = {
        "primary_metric": {"direction": "min"},
        "normalization": {"quality_ref": 2.0, "steps_ref": 10.0, "eps": 1.0e-6},
    }
    scoring_cfg = {"score_weights": {"quality": 1.0, "steps": 1.0}}

    out = extract_metrics(payload=payload, exp_cfg=exp_cfg, scoring_cfg=scoring_cfg)

    assert out["status"] == "ok"
    assert out["exp_score"] is not None
    assert float(out["quality_ratio"]) >= 0.0
    assert float(out["steps_ratio"]) >= 0.0
    assert math.isfinite(float(out["exp_score"]))
