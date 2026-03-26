"""Neyman allocation helpers for partial benchmark evaluation."""

from __future__ import annotations

import math
import random
from statistics import mean, stdev
from typing import Any

from src.evolve.storage import load_recent_experiment_scores


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or math.isinf(out):
        return default
    return out


def compute_experiment_stats(
    history: dict[str, list[float]],
    experiments: list[str],
    costs: dict[str, Any],
    std_floor: float,
) -> dict[str, dict[str, float | int]]:
    """Compute per-experiment mean/std/count and normalized relative cost."""

    cleaned_costs: dict[str, float] = {}
    for exp_name in experiments:
        cost_value = _safe_float(costs.get(exp_name, 1.0), default=1.0)
        cleaned_costs[exp_name] = max(cost_value, std_floor)

    total_cost = sum(cleaned_costs.values())
    if total_cost <= 0:
        total_cost = float(len(experiments))
        cleaned_costs = {exp_name: 1.0 for exp_name in experiments}

    stats: dict[str, dict[str, float | int]] = {}
    for exp_name in experiments:
        values = history.get(exp_name, [])
        n = len(values)
        mean_value = float(mean(values)) if values else 0.0
        if n >= 2:
            std_value = float(stdev(values))
        else:
            std_value = std_floor

        stats[exp_name] = {
            "mean": mean_value,
            "std": max(std_value, std_floor),
            "n": int(n),
            "cost": cleaned_costs[exp_name] / total_cost,
        }

    return stats


def compute_neyman_pi(
    stats: dict[str, dict[str, float | int]],
    min_history_for_variance: int,
    std_floor: float,
    fallback: str,
) -> dict[str, float]:
    """Compute Neyman allocation probabilities with fallback behavior."""

    experiments = list(stats.keys())
    if not experiments:
        return {}

    if fallback == "uniform":
        if all(int(stats[exp_name].get("n", 0)) < min_history_for_variance for exp_name in experiments):
            uniform = 1.0 / float(len(experiments))
            return {exp_name: uniform for exp_name in experiments}

    raw_weights: dict[str, float] = {}
    for exp_name in experiments:
        std_value = max(_safe_float(stats[exp_name].get("std", std_floor), std_floor), std_floor)
        cost_value = max(_safe_float(stats[exp_name].get("cost", 1.0), 1.0), std_floor)
        raw_weights[exp_name] = std_value / math.sqrt(cost_value)

    total_weight = sum(raw_weights.values())
    if total_weight <= 0 or total_weight != total_weight:
        uniform = 1.0 / float(len(experiments))
        return {exp_name: uniform for exp_name in experiments}

    return {exp_name: raw_weights[exp_name] / total_weight for exp_name in experiments}


def sample_experiments_without_replacement(
    experiments: list[str],
    pi: dict[str, float],
    sample_size: int,
    seed: int,
) -> list[str]:
    """Weighted sampling without replacement from experiment names."""

    if not experiments:
        return []

    sample_size = max(1, min(sample_size, len(experiments)))
    if sample_size >= len(experiments):
        return list(experiments)

    rng = random.Random(seed)
    available = list(experiments)
    selected: list[str] = []

    while available and len(selected) < sample_size:
        weights = [max(_safe_float(pi.get(exp_name, 0.0), 0.0), 0.0) for exp_name in available]
        total = sum(weights)
        if total <= 0:
            pick = rng.choice(available)
        else:
            pick = rng.choices(population=available, weights=weights, k=1)[0]
        selected.append(pick)
        available.remove(pick)

    return selected


def build_allocation_snapshot(
    population_root: str,
    experiments: list[str],
    allocation_cfg: Any,
    seed: int,
    candidate_id: str,
) -> dict[str, Any]:
    """Build one allocation snapshot for a candidate."""

    if not experiments:
        raise ValueError("Allocation requires at least one experiment.")

    enabled = bool(allocation_cfg.get("enabled", True))
    method = str(allocation_cfg.get("method", "neyman")).lower()
    history_window = int(allocation_cfg.get("history_window", 100))
    sample_size = int(allocation_cfg.get("sample_size", len(experiments)))
    sample_size = max(1, min(sample_size, len(experiments)))

    std_floor = max(_safe_float(allocation_cfg.get("std_floor", 1.0e-6), 1.0e-6), 1.0e-12)
    min_history = int(allocation_cfg.get("min_history_for_variance", 3))
    fallback = str(allocation_cfg.get("fallback", "uniform")).lower()

    costs = allocation_cfg.get("costs", {})
    if not isinstance(costs, dict):
        costs = {}

    history = load_recent_experiment_scores(
        population_root=population_root,
        experiments=experiments,
        history_window=history_window,
    )

    stats = compute_experiment_stats(
        history=history,
        experiments=experiments,
        costs=costs,
        std_floor=std_floor,
    )

    if not enabled or sample_size >= len(experiments) or method != "neyman":
        uniform = 1.0 / float(len(experiments))
        pi = {exp_name: uniform for exp_name in experiments}
        selected_experiments = list(experiments)
    else:
        pi = compute_neyman_pi(
            stats=stats,
            min_history_for_variance=min_history,
            std_floor=std_floor,
            fallback=fallback,
        )
        deterministic_seed = int(seed) + sum(ord(ch) for ch in candidate_id)
        selected_experiments = sample_experiments_without_replacement(
            experiments=experiments,
            pi=pi,
            sample_size=sample_size,
            seed=deterministic_seed,
        )

    return {
        "method": method,
        "enabled": enabled,
        "history_window": history_window,
        "sample_size": sample_size,
        "pi": pi,
        "stats": stats,
        "selected_experiments": selected_experiments,
    }
