"""Tests for canonical selection operators and explicit legacy helpers."""

from __future__ import annotations

import random
from pathlib import Path

import src.evolve.selection as canonical_selection
from src.evolve.legacy_selection import (
    elite_select,
    select_parents_for_reproduction,
    tournament_select,
)
from src.evolve.selection import (
    select_top_k_per_island,
    softmax_select_organisms,
    uniform_select_organisms,
)
from src.evolve.types import OrganismMeta


def _make_organism(org_id: str, score: float | None = None) -> OrganismMeta:
    org_dir = Path("/tmp/test") / f"org_{org_id}"
    return OrganismMeta(
        organism_id=org_id,
        island_id="gradient_methods",
        generation_created=0,
        current_generation_active=0,
        timestamp="2026-01-01T00:00:00Z",
        mother_id=None,
        father_id=None,
        operator="seed",
        genetic_code_path=str(org_dir / "genetic_code.md"),
        model_name="mock",
        prompt_hash="abc",
        seed=0,
        optimizer_path=str(org_dir / "optimizer.py"),
        lineage_path=str(org_dir / "lineage.json"),
        organism_dir=str(org_dir),
        simple_reward=score,
        hard_reward=score,
        selection_reward=score,
        genetic_code={"core_genes": ["test"], "interaction_notes": "", "compute_notes": ""},
        lineage=[],
    )


def test_elite_select_returns_top_n() -> None:
    pop = [
        _make_organism("a", 0.9),
        _make_organism("b", 0.7),
        _make_organism("c", 0.8),
        _make_organism("d", None),
    ]
    elites = elite_select(pop, 2, score_key="selection_reward")
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


def test_uniform_select_organisms_returns_requested_count() -> None:
    pop = [_make_organism(str(i), float(i)) for i in range(3)]
    picks = uniform_select_organisms(pop, k=8, rng=random.Random(42))
    assert len(picks) == 8
    assert {pick.organism_id for pick in picks}.issubset({"0", "1", "2"})


def test_softmax_select_biases_toward_higher_reward() -> None:
    pop = [
        _make_organism("low", 0.1),
        _make_organism("mid", 0.5),
        _make_organism("high", 1.0),
    ]
    picks = softmax_select_organisms(
        pop,
        score_field="selection_reward",
        temperature=0.2,
        k=200,
        rng=random.Random(0),
    )
    counts: dict[str, int] = {}
    for pick in picks:
        counts[pick.organism_id] = counts.get(pick.organism_id, 0) + 1
    assert counts["high"] > counts["mid"] > counts["low"]


def test_select_top_k_per_island_keeps_boundaries() -> None:
    pop = [
        _make_organism("a1", 0.9),
        _make_organism("a2", 0.4),
        _make_organism("b1", 0.8),
        _make_organism("b2", 0.2),
    ]
    pop[0].island_id = "island_a"
    pop[1].island_id = "island_a"
    pop[2].island_id = "island_b"
    pop[3].island_id = "island_b"

    selected = select_top_k_per_island(pop, k=1, score_field="simple_reward")
    assert {organism.organism_id for organism in selected} == {"a1", "b1"}


def test_canonical_selection_module_excludes_legacy_helpers() -> None:
    assert not hasattr(canonical_selection, "tournament_select")
    assert not hasattr(canonical_selection, "elite_select")
    assert not hasattr(canonical_selection, "select_parents_for_reproduction")
