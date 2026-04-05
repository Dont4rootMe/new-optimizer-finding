"""Canonical selection operators for organism-first evolution."""

from __future__ import annotations

import math
import random
from collections import defaultdict

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
