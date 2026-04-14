"""Canonical selection operators for organism-first evolution."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from collections import defaultdict
from statistics import median

from src.evolve.types import OrganismMeta


def uniform_select_organisms(
    population: list[OrganismMeta],
    k: int,
    rng: random.Random | None = None,
) -> list[OrganismMeta]:
    """Uniform parent sampling with replacement."""

    if not population or k <= 0:
        return []
    rng = rng or random.Random()
    return [rng.choice(population) for _ in range(k)]


def softmax_select_organisms(
    population: list[OrganismMeta],
    score_field: str = "simple_score",
    temperature: float = 1.0,
    k: int = 1,
    rng: random.Random | None = None,
) -> list[OrganismMeta]:
    """Softmax sampling over a score field with deterministic uniform fallback."""

    if not population or k <= 0:
        return []

    rng = rng or random.Random()
    weights = _softmax_weights(population, score_field=score_field, temperature=temperature)
    if weights is None:
        return uniform_select_organisms(population, k=k, rng=rng)
    return rng.choices(population=population, weights=weights, k=k)


def softmax_select_distinct_organisms(
    population: list[OrganismMeta],
    score_field: str = "simple_score",
    temperature: float = 1.0,
    k: int = 1,
    rng: random.Random | None = None,
) -> list[OrganismMeta]:
    """Softmax sampling without replacement."""

    if not population or k <= 0:
        return []

    rng = rng or random.Random()
    remaining = list(population)
    selected: list[OrganismMeta] = []
    target = min(k, len(remaining))
    while remaining and len(selected) < target:
        weights = _softmax_weights(remaining, score_field=score_field, temperature=temperature)
        if weights is None:
            pick = rng.choice(remaining)
        else:
            pick = rng.choices(population=remaining, weights=weights, k=1)[0]
        selected.append(pick)
        remaining = [organism for organism in remaining if organism.organism_id != pick.organism_id]
    return selected


def weighted_rule_select_organisms(
    population: list[OrganismMeta],
    *,
    parent_offspring_counts: Mapping[str, int] | None = None,
    score_field: str = "simple_score",
    weighted_rule_lambda: float = 1.0,
    k: int = 1,
    rng: random.Random | None = None,
) -> list[OrganismMeta]:
    """Weighted-rule sampling over fitness and prior parent usage."""

    if not population or k <= 0:
        return []

    rng = rng or random.Random()
    weights = _weighted_rule_weights(
        population,
        score_field=score_field,
        weighted_rule_lambda=weighted_rule_lambda,
        parent_offspring_counts=parent_offspring_counts,
    )
    if weights is None:
        return uniform_select_organisms(population, k=k, rng=rng)
    return rng.choices(population=population, weights=weights, k=k)


def weighted_rule_select_distinct_organisms(
    population: list[OrganismMeta],
    *,
    parent_offspring_counts: Mapping[str, int] | None = None,
    score_field: str = "simple_score",
    weighted_rule_lambda: float = 1.0,
    k: int = 1,
    rng: random.Random | None = None,
) -> list[OrganismMeta]:
    """Weighted-rule sampling without replacement."""

    if not population or k <= 0:
        return []

    rng = rng or random.Random()
    remaining = list(population)
    selected: list[OrganismMeta] = []
    target = min(k, len(remaining))
    while remaining and len(selected) < target:
        weights = _weighted_rule_weights(
            remaining,
            score_field=score_field,
            weighted_rule_lambda=weighted_rule_lambda,
            parent_offspring_counts=parent_offspring_counts,
        )
        if weights is None:
            pick = rng.choice(remaining)
        else:
            pick = rng.choices(population=remaining, weights=weights, k=1)[0]
        selected.append(pick)
        remaining = [organism for organism in remaining if organism.organism_id != pick.organism_id]
    return selected


def _softmax_weights(
    population: list[OrganismMeta],
    *,
    score_field: str,
    temperature: float,
) -> list[float] | None:
    temperature = max(float(temperature), 1.0e-8)

    raw_scores = [getattr(org, score_field) for org in population]
    finite_scores = [
        float(score) for score in raw_scores if score is not None and math.isfinite(float(score))
    ]
    if not finite_scores:
        return None

    max_score = max(finite_scores)
    weights: list[float] = []
    for score in raw_scores:
        if score is None:
            weights.append(0.0)
            continue
        try:
            normalized = (float(score) - max_score) / temperature
            weights.append(math.exp(normalized))
        except (TypeError, ValueError, OverflowError):
            weights.append(0.0)

    if sum(weights) <= 0:
        return None
    return weights


def _weighted_rule_weights(
    population: list[OrganismMeta],
    *,
    score_field: str,
    weighted_rule_lambda: float,
    parent_offspring_counts: Mapping[str, int] | None,
) -> list[float] | None:
    weighted_rule_lambda = max(float(weighted_rule_lambda), 1.0e-8)
    parent_offspring_counts = parent_offspring_counts or {}

    raw_scores = [getattr(org, score_field) for org in population]
    finite_scores: list[float] = []
    for score in raw_scores:
        if score is None:
            continue
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_score):
            finite_scores.append(numeric_score)

    if not finite_scores:
        return None

    alpha0 = median(finite_scores)
    weights: list[float] = []
    for organism, score in zip(population, raw_scores, strict=False):
        if score is None:
            weights.append(0.0)
            continue
        try:
            fitness = float(score)
        except (TypeError, ValueError):
            weights.append(0.0)
            continue
        if not math.isfinite(fitness):
            weights.append(0.0)
            continue

        offspring_count = max(0, int(parent_offspring_counts.get(organism.organism_id, 0)))
        selection_term = _sigmoid(weighted_rule_lambda * (fitness - alpha0))
        history_term = 1.0 / (1.0 + offspring_count)
        weights.append(selection_term * history_term)

    if sum(weights) <= 0:
        return None
    return weights


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def _group_by_island(population: list[OrganismMeta]) -> dict[str, list[OrganismMeta]]:
    grouped: dict[str, list[OrganismMeta]] = defaultdict(list)
    for organism in population:
        grouped[organism.island_id].append(organism)
    return dict(grouped)


def select_top_k_per_island(
    population: list[OrganismMeta],
    k: int,
    score_field: str = "simple_score",
) -> list[OrganismMeta]:
    """Select top-k organisms independently inside each island."""

    selected: list[OrganismMeta] = []
    for island_population in _group_by_island(population).values():
        island_scored = [
            organism for organism in island_population if getattr(organism, score_field) is not None
        ]
        island_scored.sort(key=lambda organism: getattr(organism, score_field), reverse=True)
        selected.extend(island_scored[: max(0, int(k))])
    return selected


def select_top_h_per_island(
    population: list[OrganismMeta],
    h: int,
    score_field: str = "hard_score",
) -> list[OrganismMeta]:
    """Select top-h organisms independently inside each island."""

    return select_top_k_per_island(population, k=h, score_field=score_field)
