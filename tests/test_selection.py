"""Tests for selection operators."""

from __future__ import annotations

import random

from src.evolve.selection import elite_select, select_parents_for_reproduction, tournament_select
from src.evolve.types import OrganismMeta


def _make_organism(org_id: str, score: float | None = None) -> OrganismMeta:
    return OrganismMeta(
        organism_id=org_id,
        generation=0,
        timestamp="2026-01-01T00:00:00Z",
        parent_ids=[],
        operator="seed",
        idea_dna=["test"],
        evolution_log=[],
        model_name="mock",
        prompt_hash="abc",
        seed=0,
        organism_dir="/tmp/test",
        optimizer_path="/tmp/test/optimizer.py",
        score=score,
        simple_score=score,
    )


def test_elite_select_returns_top_n() -> None:
    pop = [
        _make_organism("a", 0.9),
        _make_organism("b", 0.7),
        _make_organism("c", 0.8),
        _make_organism("d", None),
    ]
    elites = elite_select(pop, 2, score_key="score")
    assert len(elites) == 2
    assert elites[0].organism_id == "a"
    assert elites[1].organism_id == "c"


def test_elite_select_empty_population() -> None:
    elites = elite_select([], 3)
    assert elites == []


def test_tournament_select_returns_k() -> None:
    pop = [_make_organism(str(i), float(i) / 10) for i in range(10)]
    rng = random.Random(42)
    winners = tournament_select(pop, k=5, tournament_size=3, rng=rng)
    assert len(winners) == 5


def test_select_parents_for_reproduction() -> None:
    pop = [_make_organism(str(i), float(i) / 10) for i in range(5)]
    rng = random.Random(42)
    plan = select_parents_for_reproduction(
        pop, num_offspring=10, mutation_rate=0.6, tournament_size=2, rng=rng
    )
    assert len(plan) == 10
    mutations = sum(1 for op, _ in plan if op == "mutation")
    crossovers = sum(1 for op, _ in plan if op == "crossover")
    assert mutations + crossovers == 10

    for op_name, parents in plan:
        if op_name == "mutation":
            assert len(parents) == 1
        else:
            assert len(parents) == 2


def test_select_parents_no_crossover_with_one_survivor() -> None:
    pop = [_make_organism("solo", 0.5)]
    plan = select_parents_for_reproduction(pop, num_offspring=3, mutation_rate=0.3)
    # With only 1 survivor, all should be mutations
    for op_name, _ in plan:
        assert op_name == "mutation"
