"""Canonical Neyman allocation helpers for honest subset evaluation."""

from __future__ import annotations

import math
import random
from statistics import mean, stdev
from typing import Any

from src.evolve.storage import load_recent_organism_experiment_scores


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
        std_value = float(stdev(values)) if n >= 2 else std_floor
        stats[exp_name] = {
            "mean": mean_value,
            "std": max(std_value, std_floor),
            "n": int(n),
            "cost": cleaned_costs[exp_name] / total_cost,
        }

    return stats


def compute_neyman_weights(
    stats: dict[str, dict[str, float | int]],
    min_history_for_variance: int,
    std_floor: float,
    fallback: str,
) -> dict[str, float]:
    """Compute normalized Neyman weights `w_i ∝ std_i / sqrt(cost_i)`."""

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


def compute_inclusion_probabilities(
    weights: dict[str, float],
    sample_size: int,
) -> dict[str, float]:
    """Compute Poisson-style inclusion probabilities `q_i = min(1, m * w_i)`."""

    clamped_sample_size = max(1, int(sample_size))
    return {
        exp_name: min(1.0, max(0.0, clamped_sample_size * float(weight)))
        for exp_name, weight in weights.items()
    }


def compute_nonempty_probability(inclusion_prob: dict[str, float]) -> float:
    """Compute `P(sample != emptyset)` for independent Bernoulli draws."""

    prob_all_skipped = 1.0
    for q_i in inclusion_prob.values():
        prob_all_skipped *= 1.0 - max(0.0, min(1.0, _safe_float(q_i, 0.0)))
    return max(0.0, min(1.0, 1.0 - prob_all_skipped))


def compute_conditional_inclusion_probabilities(
    inclusion_prob: dict[str, float],
    nonempty_probability: float,
) -> dict[str, float]:
    """Condition first-order inclusion probabilities on the sample being non-empty."""

    p_nonempty = _safe_float(nonempty_probability, 0.0)
    if p_nonempty <= 0.0:
        raise ValueError("Conditional Poisson sampling requires positive non-empty probability.")
    return {
        exp_name: max(0.0, min(1.0, _safe_float(q_i, 0.0) / p_nonempty))
        for exp_name, q_i in inclusion_prob.items()
    }


def sample_experiments_poisson(
    experiments: list[str],
    inclusion_prob: dict[str, float],
    seed: int,
) -> list[str]:
    """Sample from the Poisson design conditioned on a non-empty subset."""

    if not experiments:
        return []

    rng = random.Random(seed)
    while True:
        selected = [
            exp_name
            for exp_name in experiments
            if rng.random() < max(0.0, min(1.0, _safe_float(inclusion_prob.get(exp_name, 0.0), 0.0)))
        ]
        if selected:
            return selected


def build_allocation_snapshot(
    population_root: str,
    experiments: list[str],
    allocation_cfg: Any,
    seed: int,
    entity_id: str,
) -> dict[str, Any]:
    """Build an allocation snapshot with explicit Neyman weights and inclusion probabilities."""

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

    history = load_recent_organism_experiment_scores(
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

    if not enabled or method != "neyman" or sample_size >= len(experiments):
        uniform = 1.0 / float(len(experiments))
        weights = {exp_name: uniform for exp_name in experiments}
        base_inclusion_prob = {exp_name: 1.0 for exp_name in experiments}
        inclusion_prob = {exp_name: 1.0 for exp_name in experiments}
        sampling_design = "full_evaluation"
        prob_nonempty = 1.0
        selected_experiments = list(experiments)
    else:
        weights = compute_neyman_weights(
            stats=stats,
            min_history_for_variance=min_history,
            std_floor=std_floor,
            fallback=fallback,
        )
        base_inclusion_prob = compute_inclusion_probabilities(weights, sample_size=sample_size)
        prob_nonempty = compute_nonempty_probability(base_inclusion_prob)
        inclusion_prob = compute_conditional_inclusion_probabilities(
            base_inclusion_prob,
            nonempty_probability=prob_nonempty,
        )
        sampling_design = "conditional_poisson_nonempty"
        deterministic_seed = int(seed) + sum(ord(ch) for ch in entity_id)
        selected_experiments = sample_experiments_poisson(
            experiments=experiments,
            inclusion_prob=base_inclusion_prob,
            seed=deterministic_seed,
        )

    return {
        "method": method,
        "enabled": enabled,
        "history_window": history_window,
        "sample_size": sample_size,
        "sampling_design": sampling_design,
        "weights": weights,
        "base_inclusion_prob": base_inclusion_prob,
        "prob_nonempty": prob_nonempty,
        "inclusion_prob": inclusion_prob,
        "stats": stats,
        "selected_experiments": selected_experiments,
    }
