"""Selection operators for the evolutionary pipeline."""

from __future__ import annotations

import random
from typing import Any

from src.evolve.types import OrganismMeta


def tournament_select(
    population: list[OrganismMeta],
    k: int,
    tournament_size: int = 3,
    score_key: str = "score",
    rng: random.Random | None = None,
) -> list[OrganismMeta]:
    """Tournament selection: pick k winners from population.

    Each tournament draws tournament_size individuals and keeps the best.
    """
    if not population:
        return []
    rng = rng or random.Random()
    winners: list[OrganismMeta] = []
    for _ in range(k):
        contestants = rng.choices(population, k=min(tournament_size, len(population)))
        best = max(contestants, key=lambda org: getattr(org, score_key) or -float("inf"))
        winners.append(best)
    return winners


def elite_select(
    population: list[OrganismMeta],
    elite_count: int,
    score_key: str = "score",
) -> list[OrganismMeta]:
    """Elitist selection: top-N by score."""
    scored = [org for org in population if getattr(org, score_key) is not None]
    scored.sort(key=lambda org: getattr(org, score_key), reverse=True)
    return scored[:elite_count]


def select_parents_for_reproduction(
    survivors: list[OrganismMeta],
    num_offspring: int,
    mutation_rate: float = 0.7,
    tournament_size: int = 3,
    rng: random.Random | None = None,
) -> list[tuple[str, list[OrganismMeta]]]:
    """Build a reproduction plan: list of (operator_name, [parents]).

    mutation_rate fraction of offspring are produced by mutation (1 parent),
    the rest by crossover (2 parents).
    """
    if not survivors:
        return []

    rng = rng or random.Random()
    plan: list[tuple[str, list[OrganismMeta]]] = []

    for _ in range(num_offspring):
        if rng.random() < mutation_rate or len(survivors) < 2:
            parent = tournament_select(survivors, k=1, tournament_size=tournament_size, rng=rng)[0]
            plan.append(("mutation", [parent]))
        else:
            parent_a = tournament_select(survivors, k=1, tournament_size=tournament_size, rng=rng)[0]
            parent_b = tournament_select(survivors, k=1, tournament_size=tournament_size, rng=rng)[0]
            # Avoid self-crossover when possible
            if parent_a.organism_id == parent_b.organism_id and len(survivors) >= 2:
                others = [s for s in survivors if s.organism_id != parent_a.organism_id]
                if others:
                    parent_b = rng.choice(others)
            plan.append(("crossover", [parent_a, parent_b]))

    return plan
