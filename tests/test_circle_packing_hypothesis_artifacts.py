"""Artifact IO tests for canonical circle-packing hypotheses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.circle_packing_shinka._runtime import genome_schema_v1 as schema
from src.evolve.types import OrganismMeta
from src.organisms.hypothesis_artifacts import read_canonical_genome, write_canonical_genome
from src.organisms.organism import read_organism_hypothesis_for_prompt
from tests.fixtures.circle_packing_genome import valid_circle_packing_genome


def test_write_read_roundtrip(tmp_path: Path) -> None:
    genome = valid_circle_packing_genome()
    write_canonical_genome(tmp_path, genome, schema)

    assert read_canonical_genome(tmp_path, schema) == genome


def test_write_canonical_genome_also_writes_genetic_code_markdown(tmp_path: Path) -> None:
    genome = valid_circle_packing_genome()
    write_canonical_genome(tmp_path, genome, schema)

    markdown = (tmp_path / "genetic_code.md").read_text(encoding="utf-8")
    assert markdown.startswith("## CORE_GENES\n- [layout]")
    assert "## CHANGE_DESCRIPTION\n" in markdown


def test_read_fails_cleanly_if_genome_json_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="genome"):
        read_canonical_genome(tmp_path, schema)


def test_read_fails_if_markdown_exists_but_genome_json_does_not(tmp_path: Path) -> None:
    (tmp_path / "genetic_code.md").write_text(
        "## CORE_GENES\n- [layout] Rendered only.\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="genome"):
        read_canonical_genome(tmp_path, schema)


def test_read_does_not_use_markdown_as_source_of_truth(tmp_path: Path) -> None:
    genome = valid_circle_packing_genome()
    write_canonical_genome(tmp_path, genome, schema)
    (tmp_path / "genetic_code.md").write_text(
        "## CORE_GENES\n- [layout] Tampered rendered markdown.\n",
        encoding="utf-8",
    )

    loaded = read_canonical_genome(tmp_path, schema)
    assert loaded["slots"]["layout"]["hypothesis"] == genome["slots"]["layout"]["hypothesis"]


def test_prompt_hypothesis_read_prefers_genome_over_markdown(tmp_path: Path) -> None:
    genome = valid_circle_packing_genome()
    write_canonical_genome(tmp_path, genome, schema)
    (tmp_path / "genetic_code.md").write_text(
        "## CORE_GENES\n- [layout] Tampered rendered markdown.\n",
        encoding="utf-8",
    )
    organism = OrganismMeta(
        organism_id=genome["organism_id"],
        island_id="symmetric_constructions",
        generation_created=0,
        current_generation_active=0,
        timestamp="2026-01-01T00:00:00Z",
        mother_id=None,
        father_id=None,
        operator="seed",
        genetic_code_path=str(tmp_path / "genetic_code.md"),
        implementation_path=str(tmp_path / "implementation.py"),
        lineage_path=str(tmp_path / "lineage.json"),
        organism_dir=str(tmp_path),
    )

    prompt_payload = read_organism_hypothesis_for_prompt(organism, schema_provider=schema)
    assert prompt_payload["core_genes"][0] == f"[layout] {genome['slots']['layout']['hypothesis']}"


def test_written_json_is_pretty_sorted_and_has_trailing_newline(tmp_path: Path) -> None:
    genome = valid_circle_packing_genome()
    write_canonical_genome(tmp_path, genome, schema)

    raw = (tmp_path / "genome.json").read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert json.loads(raw) == genome
    assert raw.splitlines()[1].startswith('  "global_hypothesis"')
