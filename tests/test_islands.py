"""Tests for island asset loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evolve.islands import load_islands


def test_load_islands_reads_multiple_files(tmp_path: Path) -> None:
    (tmp_path / "gradient_methods.txt").write_text("First-order ideas", encoding="utf-8")
    (tmp_path / "second_order.txt").write_text("Curvature-aware ideas", encoding="utf-8")

    islands = load_islands(tmp_path)

    assert [island.island_id for island in islands] == ["gradient_methods", "second_order"]
    assert islands[0].description_text == "First-order ideas"


def test_load_islands_rejects_empty_file(tmp_path: Path) -> None:
    (tmp_path / "broken.txt").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_islands(tmp_path)
