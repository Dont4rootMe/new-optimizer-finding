"""Rendering tests for circle_packing_shinka canonical genomes."""

from __future__ import annotations

from experiments.circle_packing_shinka._runtime import genome_schema_v1 as schema
from src.organisms.hypothesis_render import render_genetic_code_markdown
from tests.fixtures.circle_packing_genome import valid_circle_packing_genome


def test_valid_genome_renders_exact_four_sections() -> None:
    rendered = render_genetic_code_markdown(valid_circle_packing_genome(), schema)

    sections = [line for line in rendered.splitlines() if line.startswith("## ")]
    assert sections == [
        "## CORE_GENES",
        "## INTERACTION_NOTES",
        "## COMPUTE_NOTES",
        "## CHANGE_DESCRIPTION",
    ]


def test_core_genes_contains_eight_bullets_in_slot_order() -> None:
    genome = valid_circle_packing_genome()
    rendered = render_genetic_code_markdown(genome, schema)
    core_block = rendered.split("## INTERACTION_NOTES", maxsplit=1)[0]
    bullets = [line for line in core_block.splitlines() if line.startswith("- ")]

    assert len(bullets) == 8
    assert bullets == [
        f"- [{slot}] {genome['slots'][slot]['hypothesis']}"
        for slot in schema.SLOT_ORDER
    ]


def test_renderer_is_deterministic() -> None:
    genome = valid_circle_packing_genome()

    assert render_genetic_code_markdown(genome, schema) == render_genetic_code_markdown(genome, schema)


def test_rendered_change_description_is_source_verbatim() -> None:
    genome = valid_circle_packing_genome()
    rendered = render_genetic_code_markdown(genome, schema)

    assert rendered.rsplit("## CHANGE_DESCRIPTION\n", maxsplit=1)[1] == (
        genome["render_fields"]["change_description"] + "\n"
    )
