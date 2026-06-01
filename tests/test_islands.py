"""Tests for island synthesis helpers."""

from __future__ import annotations

import pytest

from src.evolve.islands import synthesize_islands_from_ids


def test_synthesize_islands_from_ids_builds_minimal_record() -> None:
    islands = synthesize_islands_from_ids(["gradient_methods", "second_order"])
    assert [island.island_id for island in islands] == ["gradient_methods", "second_order"]
    assert islands[0].name == "gradient methods"
    # The simplified Island carries only id + name; no description fields.
    assert not hasattr(islands[0], "description_text")
    assert not hasattr(islands[0], "description_path")


def test_synthesize_islands_from_ids_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one island_id"):
        synthesize_islands_from_ids([])


def test_synthesize_islands_from_ids_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="Duplicate island_id"):
        synthesize_islands_from_ids(["a", "a"])


def test_synthesize_islands_from_ids_rejects_blank_entries() -> None:
    with pytest.raises(ValueError, match="Empty island_id"):
        synthesize_islands_from_ids(["valid_one", ""])
