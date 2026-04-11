"""Hydra-instantiated fake experiment evaluator for integration tests."""

from __future__ import annotations

from typing import Any


class FakeExperimentEvaluator:
    """Return deterministic score-bearing reports without touching implementation.py."""

    def __init__(self, **_: Any) -> None:
        pass

    def evaluate_organism(self, organism_dir: str | None, cfg) -> dict[str, Any]:
        del organism_dir
        score = 0.8 if str(cfg.name).endswith("a") else 0.6
        return {
            "status": "ok",
            "score": score,
            "objective_name": "train_loss",
            "objective_direction": "min",
            "objective_last": score,
            "objective_best": score,
            "objective_best_step": 5,
            "first_step_at_or_below_baseline": 5,
            "time_sec": 0.01,
            "steps": 5,
            "converged": True,
            "error_msg": None,
        }


class AlwaysFailExperimentEvaluator:
    """Return a deterministic failed report for seed/eval failure tests."""

    def __init__(self, **_: Any) -> None:
        pass

    def evaluate_organism(self, organism_dir: str | None, cfg) -> dict[str, Any]:
        del organism_dir, cfg
        return {
            "status": "failed",
            "score": None,
            "error_msg": "intentional evaluator failure",
        }
