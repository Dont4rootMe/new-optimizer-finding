"""Tests for probabilistic mutation (trait deletion) of idea_dna."""

from __future__ import annotations

import random

from src.organisms.mutation import mutate_idea_dna


def test_mutate_deterministic_with_seed():
    """Fixed seed produces reproducible results."""
    dna = ["adaptive lr", "cosine schedule", "gradient clipping", "warmup"]

    surv1, rem1 = mutate_idea_dna(dna, q=0.3, rng=random.Random(42))
    surv2, rem2 = mutate_idea_dna(dna, q=0.3, rng=random.Random(42))
    assert surv1 == surv2
    assert rem1 == rem2


def test_mutate_q_0_keeps_all():
    """With q=0, no traits are deleted."""
    dna = ["a", "b", "c", "d"]
    surviving, removed = mutate_idea_dna(dna, q=0.0, rng=random.Random(0))
    assert surviving == dna
    assert removed == []


def test_mutate_q_1_removes_all_but_one():
    """With q=1, all would be removed, but at least one is kept."""
    dna = ["a", "b", "c", "d"]
    surviving, removed = mutate_idea_dna(dna, q=1.0, rng=random.Random(0))
    assert len(surviving) == 1
    assert len(removed) == 3
    # The rescued trait should not be in removed
    assert surviving[0] not in removed


def test_mutate_single_trait_survives():
    """A single trait always survives regardless of q."""
    dna = ["only_trait"]
    surviving, removed = mutate_idea_dna(dna, q=1.0, rng=random.Random(0))
    assert surviving == ["only_trait"]
    assert removed == []


def test_mutate_empty_dna():
    """Empty DNA stays empty."""
    surviving, removed = mutate_idea_dna([], q=0.5, rng=random.Random(0))
    assert surviving == []
    assert removed == []


def test_mutate_returns_partition():
    """Surviving + removed should equal the original DNA (as sets)."""
    dna = ["a", "b", "c", "d", "e"]
    surviving, removed = mutate_idea_dna(dna, q=0.4, rng=random.Random(42))
    assert set(surviving) | set(removed) == set(dna)
    assert set(surviving) & set(removed) == set()


def test_mutate_typical_case():
    """With q=0.3, roughly 30% of traits are removed over many runs."""
    dna = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]
    total_removed = 0
    runs = 1000
    for seed in range(runs):
        _, removed = mutate_idea_dna(dna, q=0.3, rng=random.Random(seed))
        total_removed += len(removed)

    avg_removed = total_removed / runs
    # Should be roughly 3 (30% of 10), allow some variance
    assert 2.0 < avg_removed < 4.0
