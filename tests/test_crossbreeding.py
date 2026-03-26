"""Tests for probabilistic crossbreeding of idea_dna."""

from __future__ import annotations

import random

from src.organisms.crossbreeding import crossbreed_idea_dna


def test_crossbreed_deterministic_with_seed():
    """Fixed seed produces reproducible results."""
    dna_a = ["adaptive lr", "cosine schedule", "gradient clipping"]
    dna_b = ["momentum scaling", "warmup phase", "weight decay"]

    result1 = crossbreed_idea_dna(dna_a, dna_b, p=0.7, rng=random.Random(42))
    result2 = crossbreed_idea_dna(dna_a, dna_b, p=0.7, rng=random.Random(42))
    assert result1 == result2


def test_crossbreed_p_1_only_dominant():
    """With p=1.0, all dominant traits are included, none from non-dominant."""
    dna_a = ["trait_a1", "trait_a2", "trait_a3"]
    dna_b = ["trait_b1", "trait_b2"]

    result = crossbreed_idea_dna(dna_a, dna_b, p=1.0, rng=random.Random(0))
    # p=1.0 => all from dominant included, 1-p=0 => none from non-dominant
    assert set(result) == set(dna_a)


def test_crossbreed_p_0_only_non_dominant():
    """With p=0.0, none from dominant, all from non-dominant."""
    dna_a = ["trait_a1", "trait_a2"]
    dna_b = ["trait_b1", "trait_b2", "trait_b3"]

    result = crossbreed_idea_dna(dna_a, dna_b, p=0.0, rng=random.Random(0))
    # p=0 => none from dominant, 1-p=1 => all from non-dominant
    assert set(result) == set(dna_b)


def test_crossbreed_guarantees_at_least_one_trait():
    """Even with extreme p, at least one trait survives."""
    dna_a = ["only_trait"]
    dna_b = []

    # p=0 would skip dominant, and non-dominant is empty
    result = crossbreed_idea_dna(dna_a, dna_b, p=0.0, rng=random.Random(0))
    assert len(result) >= 1


def test_crossbreed_deduplicates():
    """Duplicate traits (case-insensitive) are not repeated."""
    dna_a = ["Adaptive LR", "cosine schedule"]
    dna_b = ["adaptive lr", "warmup phase"]

    result = crossbreed_idea_dna(dna_a, dna_b, p=1.0, rng=random.Random(0))
    # "adaptive lr" (from b) should be skipped since "Adaptive LR" (from a) is already there
    lower_results = [r.lower() for r in result]
    assert len(lower_results) == len(set(lower_results))


def test_crossbreed_both_empty():
    """Two empty parents produce empty result (graceful)."""
    result = crossbreed_idea_dna([], [], p=0.5, rng=random.Random(0))
    assert result == []


def test_crossbreed_typical_case():
    """With p=0.7, result is a mix of both parents."""
    dna_a = ["a1", "a2", "a3", "a4", "a5"]
    dna_b = ["b1", "b2", "b3", "b4", "b5"]

    # Run many times and check we get traits from both
    all_results: set[str] = set()
    for seed in range(100):
        result = crossbreed_idea_dna(dna_a, dna_b, p=0.7, rng=random.Random(seed))
        all_results.update(result)

    # Over 100 runs, we should see traits from both parents
    a_traits = all_results & set(dna_a)
    b_traits = all_results & set(dna_b)
    assert len(a_traits) > 0
    assert len(b_traits) > 0
