"""Parser tests for schema-driven genetic-code formats."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.organisms.genetic_code_format import (
    detect_genetic_code_format,
    parse_genetic_code_text,
    parse_genome_schema_text,
)

ROOT = Path(__file__).resolve().parents[1]

SCHEMA_SECTION_NAMES = (
    "INIT_GEOMETRY",
    "RADIUS_POLICY",
    "EXPANSION_POLICY",
    "CONFLICT_MODEL",
    "REPAIR_POLICY",
    "CONTROL_POLICY",
    "PARAMETERS",
    "OPTIONAL_CODE_SKETCH",
)
OPTIMIZER_SECTION_NAMES = (
    "STATE_REPRESENTATION",
    "GRADIENT_PROCESSING",
    "UPDATE_RULE",
    "PARAMETER_GROUP_POLICY",
    "STEP_CONTROL_POLICY",
    "STABILITY_POLICY",
    "PARAMETERS",
    "OPTIONAL_CODE_SKETCH",
)

SECTIONED_GENETIC_CODE = """## CORE_GENES
### INIT_GEOMETRY
- Start from a centered triangular scaffold trimmed to the square.

### RADIUS_POLICY
- Assign initial radii from local scaffold spacing.

### EXPANSION_POLICY
- Increase radii in small density-aware growth steps.

### CONFLICT_MODEL
- Treat pairwise overlap and boundary penetration as the two primary violations.

### REPAIR_POLICY
- Resolve the worst conflict first by local deterministic center shifts.

### CONTROL_POLICY
- Alternate growth and repair until no score-improving feasible update remains.

### PARAMETERS
- Use smaller corrective moves after repeated conflict recurrence.

### OPTIONAL_CODE_SKETCH
- None.

## INTERACTION_NOTES
This organism assumes the scaffold is already close to feasible and relies on deterministic local correction.

## COMPUTE_NOTES
The design implies a staged constructive search with repeated local feasibility restoration.

## CHANGE_DESCRIPTION
This organism tests whether a triangular scaffold combined with density-aware expansion and worst-conflict-first repair can improve score while preserving deterministic feasibility.
"""

LEGACY_GENETIC_CODE = """## CORE_GENES
- Adaptive first-moment tracking for noisy gradients
- Per-parameter normalization using a running second moment
- Late-training step shrinkage to stabilize convergence

## INTERACTION_NOTES
The moment and normalization signals should be coordinated.

## COMPUTE_NOTES
Only low-overhead tensor state is allowed.

## CHANGE_DESCRIPTION
Added a stable legacy flat gene list.
"""


def test_parse_genome_schema_text_accepts_valid_schema_file() -> None:
    schema_path = ROOT / "conf" / "experiments" / "circle_packing_shinka" / "prompts" / "shared" / "genome_schema.txt"
    sections = parse_genome_schema_text(schema_path.read_text(encoding="utf-8"))

    assert tuple(section.name for section in sections) == SCHEMA_SECTION_NAMES
    assert sections[0].description.startswith("This section defines how the organism proposes")


def test_parse_genome_schema_text_accepts_non_circle_schema_file() -> None:
    schema_path = ROOT / "conf" / "experiments" / "optimization_survey" / "prompts" / "shared" / "genome_schema.txt"
    sections = parse_genome_schema_text(schema_path.read_text(encoding="utf-8"))

    assert tuple(section.name for section in sections) == OPTIMIZER_SECTION_NAMES
    assert "optimizer state" in sections[0].description


def test_parse_genome_schema_text_rejects_duplicate_sections() -> None:
    text = "# INIT_GEOMETRY\nFirst.\n\n# INIT_GEOMETRY\nDuplicate."

    with pytest.raises(ValueError, match="duplicate"):
        parse_genome_schema_text(text)


def test_parse_genome_schema_text_rejects_malformed_header() -> None:
    with pytest.raises(ValueError, match="Malformed genome schema header"):
        parse_genome_schema_text("# Init Geometry\nBody.")


def test_parse_genome_schema_text_rejects_empty_schema() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_genome_schema_text(" \n\n")


def test_parse_genome_schema_text_rejects_body_before_first_header() -> None:
    with pytest.raises(ValueError, match="before the first section header"):
        parse_genome_schema_text("Body first.\n# INIT_GEOMETRY\nBody.")


def test_detect_genetic_code_format_returns_sectioned_for_sectioned_core() -> None:
    assert detect_genetic_code_format(SECTIONED_GENETIC_CODE) == "sectioned"


def test_detect_genetic_code_format_returns_legacy_for_flat_core() -> None:
    assert detect_genetic_code_format(LEGACY_GENETIC_CODE) == "legacy_flat"


def test_sectioned_parser_preserves_schema_section_order() -> None:
    parsed = parse_genetic_code_text(
        SECTIONED_GENETIC_CODE,
        expected_section_names=SCHEMA_SECTION_NAMES,
    )

    assert parsed.format_kind == "sectioned"
    assert parsed.core_gene_sections is not None
    assert tuple(section.name for section in parsed.core_gene_sections) == SCHEMA_SECTION_NAMES


def test_legacy_parser_preserves_flat_gene_entries() -> None:
    parsed = parse_genetic_code_text(LEGACY_GENETIC_CODE)

    assert parsed.format_kind == "legacy_flat"
    assert parsed.legacy_core_genes == (
        "Adaptive first-moment tracking for noisy gradients",
        "Per-parameter normalization using a running second moment",
        "Late-training step shrinkage to stabilize convergence",
    )


def test_legacy_parser_rejects_missing_change_description() -> None:
    text = LEGACY_GENETIC_CODE.replace(
        "\n## CHANGE_DESCRIPTION\nAdded a stable legacy flat gene list.\n",
        "",
    )

    with pytest.raises(ValueError, match="CHANGE_DESCRIPTION"):
        parse_genetic_code_text(text)


@pytest.mark.parametrize(
    "bad_entry",
    [
        "Bare prose without a bullet.",
        "* Star bullet is not canonical.",
        "1. Numbered item is not canonical.",
    ],
)
def test_legacy_parser_rejects_non_dash_bullet_gene_text(bad_entry: str) -> None:
    text = LEGACY_GENETIC_CODE.replace(
        "- Adaptive first-moment tracking for noisy gradients",
        bad_entry,
    )

    with pytest.raises(ValueError, match="non-bullet text"):
        parse_genetic_code_text(text)


def test_legacy_parser_accepts_dash_bullets_with_continuation_lines() -> None:
    text = LEGACY_GENETIC_CODE.replace(
        "- Adaptive first-moment tracking for noisy gradients",
        "- Adaptive first-moment tracking for noisy gradients:\n"
        "  keep one accumulator per parameter tensor",
    )

    parsed = parse_genetic_code_text(text)

    assert parsed.legacy_core_genes is not None
    assert parsed.legacy_core_genes[0] == (
        "Adaptive first-moment tracking for noisy gradients:\n"
        "keep one accumulator per parameter tensor"
    )


def test_sectioned_parser_attaches_continuation_lines_to_previous_bullet() -> None:
    text = SECTIONED_GENETIC_CODE.replace(
        "- Resolve the worst conflict first by local deterministic center shifts.",
        (
            "- Define a local repair patch for overlap resolution:\n"
            "  move the worse-conflicting circle along the separating direction,\n"
            "  then re-check its immediate neighbors only."
        ),
    )
    parsed = parse_genetic_code_text(text, expected_section_names=SCHEMA_SECTION_NAMES)

    assert parsed.core_gene_sections is not None
    repair_section = parsed.core_gene_sections[4]
    assert repair_section.entries[0].text == (
        "Define a local repair patch for overlap resolution:\n"
        "move the worse-conflicting circle along the separating direction,\n"
        "then re-check its immediate neighbors only."
    )


def test_sectioned_parser_uses_arbitrary_schema_section_names() -> None:
    core = "\n\n".join(
        f"### {name}\n- "
        + ("None." if name == OPTIMIZER_SECTION_NAMES[-1] else f"Optimizer idea for {name.lower()}.")
        for name in OPTIMIZER_SECTION_NAMES
    )
    text = (
        "## CORE_GENES\n"
        f"{core}\n\n"
        "## INTERACTION_NOTES\n"
        "Optimizer sections coordinate state, gradients, and updates.\n\n"
        "## COMPUTE_NOTES\n"
        "Single-pass tensor updates.\n\n"
        "## CHANGE_DESCRIPTION\n"
        "A sectioned optimizer fixture.\n"
    )

    parsed = parse_genetic_code_text(text, expected_section_names=OPTIMIZER_SECTION_NAMES)

    assert parsed.format_kind == "sectioned"
    assert parsed.core_gene_sections is not None
    assert tuple(section.name for section in parsed.core_gene_sections) == OPTIMIZER_SECTION_NAMES
