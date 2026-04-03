"""Tests for probabilistic crossbreeding of gene pools."""

from __future__ import annotations

import random

from src.organisms.crossbreeding import merge_gene_pools


def test_crossbreed_deterministic_with_seed():
    """Fixed seed produces reproducible results."""
    genes_a = ["adaptive lr", "cosine schedule", "gradient clipping"]
    genes_b = ["momentum scaling", "warmup phase", "weight decay"]

    result1 = merge_gene_pools(genes_a, genes_b, inherit_probability=0.7, rng=random.Random(42))
    result2 = merge_gene_pools(genes_a, genes_b, inherit_probability=0.7, rng=random.Random(42))
    assert result1 == result2


def test_crossbreed_p_1_only_dominant():
    """With p=1.0, all dominant traits are included, none from non-dominant."""
    genes_a = ["trait_a1", "trait_a2", "trait_a3"]
    genes_b = ["trait_b1", "trait_b2"]

    result = merge_gene_pools(genes_a, genes_b, inherit_probability=1.0, rng=random.Random(0))
    # p=1.0 => all from dominant included, 1-p=0 => none from non-dominant
    assert set(result) == set(genes_a)


def test_crossbreed_p_0_only_non_dominant():
    """With p=0.0, none from dominant, all from non-dominant."""
    genes_a = ["trait_a1", "trait_a2"]
    genes_b = ["trait_b1", "trait_b2", "trait_b3"]

    result = merge_gene_pools(genes_a, genes_b, inherit_probability=0.0, rng=random.Random(0))
    # p=0 => none from dominant, 1-p=1 => all from non-dominant
    assert set(result) == set(genes_b)


def test_crossbreed_guarantees_at_least_one_trait():
    """Even with extreme p, at least one trait survives."""
    genes_a = ["only_trait"]
    genes_b = []

    # p=0 would skip dominant, and non-dominant is empty
    result = merge_gene_pools(genes_a, genes_b, inherit_probability=0.0, rng=random.Random(0))
    assert len(result) >= 1


def test_crossbreed_deduplicates():
    """Duplicate traits (case-insensitive) are not repeated."""
    genes_a = ["Adaptive LR", "cosine schedule"]
    genes_b = ["adaptive lr", "warmup phase"]

    result = merge_gene_pools(genes_a, genes_b, inherit_probability=1.0, rng=random.Random(0))
    # "adaptive lr" (from b) should be skipped since "Adaptive LR" (from a) is already there
    lower_results = [r.lower() for r in result]
    assert len(lower_results) == len(set(lower_results))


def test_crossbreed_both_empty():
    """Two empty parents produce empty result (graceful)."""
    result = merge_gene_pools([], [], inherit_probability=0.5, rng=random.Random(0))
    assert result == []


def test_crossbreed_typical_case():
    """With p=0.7, result is a mix of both parents."""
    genes_a = ["a1", "a2", "a3", "a4", "a5"]
    genes_b = ["b1", "b2", "b3", "b4", "b5"]

    # Run many times and check we get traits from both
    all_results: set[str] = set()
    for seed in range(100):
        result = merge_gene_pools(genes_a, genes_b, inherit_probability=0.7, rng=random.Random(seed))
        all_results.update(result)

    # Over 100 runs, we should see traits from both parents
    a_traits = all_results & set(genes_a)
    b_traits = all_results & set(genes_b)
    assert len(a_traits) > 0
    assert len(b_traits) > 0
