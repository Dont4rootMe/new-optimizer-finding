"""Tests for canonical selection operators."""

from __future__ import annotations

import random
from pathlib import Path

import src.evolve.selection as canonical_selection
from src.evolve.selection import (
    select_top_k_per_island,
    softmax_select_distinct_organisms,
    softmax_select_organisms,
    uniform_select_organisms,
    weighted_rule_select_distinct_organisms,
    weighted_rule_select_organisms,
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
        implementation_path=str(org_dir / "implementation.py"),
        lineage_path=str(org_dir / "lineage.json"),
        organism_dir=str(org_dir),
        ancestor_ids=[],
        experiment_report_index={},
        simple_score=score,
        hard_score=score,
    )


def test_uniform_select_organisms_returns_requested_count() -> None:
    pop = [_make_organism(str(i), float(i)) for i in range(3)]
    picks = uniform_select_organisms(pop, k=8, rng=random.Random(42))
    assert len(picks) == 8
    assert {pick.organism_id for pick in picks}.issubset({"0", "1", "2"})


def test_softmax_select_biases_toward_higher_score() -> None:
    pop = [
        _make_organism("low", 0.1),
        _make_organism("mid", 0.5),
        _make_organism("high", 1.0),
    ]
    picks = softmax_select_organisms(
        pop,
        score_field="simple_score",
        temperature=0.2,
        k=200,
        rng=random.Random(0),
    )
    counts: dict[str, int] = {}
    for pick in picks:
        counts[pick.organism_id] = counts.get(pick.organism_id, 0) + 1
    assert counts["high"] > counts["mid"] > counts["low"]


def test_softmax_select_distinct_organisms_samples_without_replacement() -> None:
    pop = [
        _make_organism("low", 0.1),
        _make_organism("mid", 0.5),
        _make_organism("high", 1.0),
    ]
    picks = softmax_select_distinct_organisms(
        pop,
        score_field="simple_score",
        temperature=0.2,
        k=3,
        rng=random.Random(0),
    )
    assert len(picks) == 3
    assert len({pick.organism_id for pick in picks}) == 3


def test_weighted_rule_select_biases_toward_higher_score_when_counts_match() -> None:
    pop = [
        _make_organism("low", 0.1),
        _make_organism("mid", 0.5),
        _make_organism("high", 1.0),
    ]
    picks = weighted_rule_select_organisms(
        pop,
        parent_offspring_counts={},
        score_field="simple_score",
        weighted_rule_lambda=4.0,
        k=200,
        rng=random.Random(0),
    )
    counts: dict[str, int] = {}
    for pick in picks:
        counts[pick.organism_id] = counts.get(pick.organism_id, 0) + 1
    assert counts["high"] > counts["mid"] > counts["low"]


def test_weighted_rule_select_downweights_parent_with_many_offspring() -> None:
    pop = [
        _make_organism("leader", 1.0),
        _make_organism("challenger", 0.9),
    ]
    baseline = weighted_rule_select_organisms(
        pop,
        parent_offspring_counts={"leader": 0, "challenger": 0},
        score_field="simple_score",
        weighted_rule_lambda=6.0,
        k=300,
        rng=random.Random(0),
    )
    penalized = weighted_rule_select_organisms(
        pop,
        parent_offspring_counts={"leader": 8, "challenger": 0},
        score_field="simple_score",
        weighted_rule_lambda=6.0,
        k=300,
        rng=random.Random(0),
    )
    baseline_leader = sum(1 for pick in baseline if pick.organism_id == "leader")
    penalized_leader = sum(1 for pick in penalized if pick.organism_id == "leader")
    assert penalized_leader < baseline_leader


def test_weighted_rule_select_distinct_organisms_samples_without_replacement() -> None:
    pop = [
        _make_organism("low", 0.1),
        _make_organism("mid", 0.5),
        _make_organism("high", 1.0),
    ]
    picks = weighted_rule_select_distinct_organisms(
        pop,
        parent_offspring_counts={"high": 2},
        score_field="simple_score",
        weighted_rule_lambda=4.0,
        k=3,
        rng=random.Random(0),
    )
    assert len(picks) == 3
    assert len({pick.organism_id for pick in picks}) == 3


def test_weighted_rule_select_falls_back_to_uniform_when_scores_are_non_finite() -> None:
    pop = [
        _make_organism("none", None),
        _make_organism("nan", float("nan")),
        _make_organism("inf", float("inf")),
    ]
    expected = uniform_select_organisms(pop, k=20, rng=random.Random(123))
    actual = weighted_rule_select_organisms(
        pop,
        parent_offspring_counts={"none": 4},
        score_field="simple_score",
        weighted_rule_lambda=1.0,
        k=20,
        rng=random.Random(123),
    )
    assert [pick.organism_id for pick in actual] == [pick.organism_id for pick in expected]


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

    selected = select_top_k_per_island(pop, k=1, score_field="simple_score")
    assert {organism.organism_id for organism in selected} == {"a1", "b1"}


def test_canonical_selection_module_excludes_removed_helpers() -> None:
    assert not hasattr(canonical_selection, "tournament_select")
    assert not hasattr(canonical_selection, "elite_select")
    assert not hasattr(canonical_selection, "select_parents_for_reproduction")
