"""ShinkaEvolve evaluator adapter for circle_packing_shinka / unit_square_26."""

from __future__ import annotations

from typing import Any

from src.baselines.shinka._evaluator import evaluate_with_host_experiment


def main(program_path: str, results_dir: str) -> dict[str, Any]:
    return evaluate_with_host_experiment(program_path=program_path, results_dir=results_dir)
