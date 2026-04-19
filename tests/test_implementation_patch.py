"""Tests for section-aligned implementation patch compilation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.organisms.implementation_patch import (
    ParsedImplementationPatch,
    assemble_implementation_from_patch,
    compute_changed_genome_sections,
    extract_region_bodies_from_source,
    parse_implementation_patch_response,
    parse_implementation_scaffold,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "conf" / "experiments" / "circle_packing_shinka" / "prompts" / "shared" / "template.txt").read_text(
    encoding="utf-8"
)
REGIONS = (
    "INIT_GEOMETRY",
    "RADIUS_POLICY",
    "EXPANSION_POLICY",
    "CONFLICT_MODEL",
    "REPAIR_POLICY",
    "CONTROL_POLICY",
    "PARAMETERS",
    "OPTIONAL_CODE_SKETCH",
)
OPTIMIZER_TEMPLATE = (
    ROOT / "conf" / "experiments" / "optimization_survey" / "prompts" / "shared" / "template.txt"
).read_text(encoding="utf-8")
OPTIMIZER_REGIONS = (
    "STATE_REPRESENTATION",
    "GRADIENT_PROCESSING",
    "UPDATE_RULE",
    "PARAMETER_GROUP_POLICY",
    "STEP_CONTROL_POLICY",
    "STABILITY_POLICY",
    "PARAMETERS",
    "OPTIONAL_CODE_SKETCH",
)


def _full_patch_response() -> str:
    pieces = ["## COMPILATION_MODE", "FULL"]
    for region in REGIONS:
        pieces.extend(("", f"## REGION {region}", f"    # body for {region}", "## END_REGION"))
    return "\n".join(pieces) + "\n"


def _patch_response(*regions: str) -> str:
    pieces = ["## COMPILATION_MODE", "PATCH"]
    for region in regions:
        pieces.extend(("", f"## REGION {region}", f"    # patched {region}", "## END_REGION"))
    return "\n".join(pieces) + "\n"


def _genetic_code(**overrides: str) -> str:
    entries = {
        "INIT_GEOMETRY": "Start from a centered scaffold.",
        "RADIUS_POLICY": "Use uniform radii.",
        "EXPANSION_POLICY": "Do not expand after initialization.",
        "CONFLICT_MODEL": "Track boundary and pairwise violations.",
        "REPAIR_POLICY": "Use deterministic local shifts.",
        "CONTROL_POLICY": "Run construction once.",
        "PARAMETERS": "Use radius 0.04.",
        "OPTIONAL_CODE_SKETCH": "None.",
    }
    entries.update(overrides)
    core = "\n\n".join(f"### {name}\n- {entries[name]}" for name in REGIONS)
    return (
        "## CORE_GENES\n"
        f"{core}\n\n"
        "## INTERACTION_NOTES\n"
        "Sections are coherent.\n\n"
        "## COMPUTE_NOTES\n"
        "Deterministic constructive code.\n\n"
        "## CHANGE_DESCRIPTION\n"
        "A sectioned test design.\n"
    )


def test_parse_implementation_scaffold_accepts_canonical_template() -> None:
    regions = parse_implementation_scaffold(TEMPLATE, expected_region_names=REGIONS)

    assert tuple(region.name for region in regions) == REGIONS
    assert regions[0].start_marker == "# === REGION: INIT_GEOMETRY ==="
    assert regions[-1].end_marker == "# === END_REGION: OPTIONAL_CODE_SKETCH ==="


def test_parse_implementation_scaffold_rejects_missing_region() -> None:
    broken = TEMPLATE.replace(
        "    # === REGION: PARAMETERS ===\n    # === END_REGION: PARAMETERS ===\n\n",
        "",
    )

    with pytest.raises(ValueError, match="expected"):
        parse_implementation_scaffold(broken, expected_region_names=REGIONS)


def test_parse_implementation_scaffold_rejects_duplicate_region() -> None:
    duplicate = TEMPLATE.replace(
        "    # === REGION: RADIUS_POLICY ===",
        "    # === REGION: INIT_GEOMETRY ===\n    # === END_REGION: INIT_GEOMETRY ===\n\n    # === REGION: RADIUS_POLICY ===",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        parse_implementation_scaffold(duplicate, expected_region_names=REGIONS)


def test_parse_implementation_scaffold_rejects_out_of_order_region() -> None:
    init_block = "    # === REGION: INIT_GEOMETRY ===\n    # === END_REGION: INIT_GEOMETRY ==="
    radius_block = "    # === REGION: RADIUS_POLICY ===\n    # === END_REGION: RADIUS_POLICY ==="
    out_of_order = (
        TEMPLATE.replace(init_block, "__RADIUS_BLOCK__")
        .replace(radius_block, init_block)
        .replace("__RADIUS_BLOCK__", radius_block)
    )

    with pytest.raises(ValueError, match="expected"):
        parse_implementation_scaffold(out_of_order, expected_region_names=REGIONS)


def test_parse_implementation_scaffold_rejects_mismatched_end_marker() -> None:
    broken = TEMPLATE.replace("END_REGION: INIT_GEOMETRY", "END_REGION: RADIUS_POLICY", 1)

    with pytest.raises(ValueError, match="Mismatched"):
        parse_implementation_scaffold(broken, expected_region_names=REGIONS)


def test_parse_implementation_scaffold_rejects_unexpected_extra_region() -> None:
    extra = TEMPLATE.replace(
        "    # === REGION: INIT_GEOMETRY ===",
        "    # === REGION: EXTRA_REGION ===\n    # === END_REGION: EXTRA_REGION ===\n\n    # === REGION: INIT_GEOMETRY ===",
    )

    with pytest.raises(ValueError, match="expected"):
        parse_implementation_scaffold(extra, expected_region_names=REGIONS)


def test_extract_region_bodies_preserves_whitespace_and_comments() -> None:
    source = TEMPLATE.replace(
        "    # === END_REGION: INIT_GEOMETRY ===",
        "    # keep two trailing spaces  \n\n    centers = []\n    # === END_REGION: INIT_GEOMETRY ===",
    )

    bodies = dict(extract_region_bodies_from_source(source, expected_region_names=REGIONS))

    assert bodies["INIT_GEOMETRY"] == "    # keep two trailing spaces  \n\n    centers = []\n"


def test_extract_region_bodies_rejects_non_scaffold_source() -> None:
    with pytest.raises(ValueError, match="expected"):
        extract_region_bodies_from_source("import numpy as np\n\ndef run_packing():\n    pass\n", expected_region_names=REGIONS)


def test_extract_region_bodies_rejects_missing_marker() -> None:
    broken = TEMPLATE.replace("    # === END_REGION: REPAIR_POLICY ===\n", "")

    with pytest.raises(ValueError, match="missing its end marker|Nested"):
        extract_region_bodies_from_source(broken, expected_region_names=REGIONS)


def test_compute_changed_genome_sections_identical_genomes_yields_empty_tuple() -> None:
    genome = _genetic_code()

    assert compute_changed_genome_sections(genome, genome, expected_section_names=REGIONS) == ()


def test_compute_changed_genome_sections_single_change() -> None:
    changed = compute_changed_genome_sections(
        _genetic_code(),
        _genetic_code(RADIUS_POLICY="Use non-uniform role-dependent radii."),
        expected_section_names=REGIONS,
    )

    assert changed == ("RADIUS_POLICY",)


def test_compute_changed_genome_sections_multiple_changes_preserve_order() -> None:
    changed = compute_changed_genome_sections(
        _genetic_code(),
        _genetic_code(REPAIR_POLICY="Use shrinking repair.", PARAMETERS="Use radius 0.035."),
        expected_section_names=REGIONS,
    )

    assert changed == ("REPAIR_POLICY", "PARAMETERS")


def test_compute_changed_genome_sections_normalizes_trailing_whitespace_only() -> None:
    maternal = _genetic_code(REPAIR_POLICY="Use deterministic local shifts.")
    child = _genetic_code(REPAIR_POLICY="Use deterministic local shifts.   ")

    assert compute_changed_genome_sections(maternal, child, expected_section_names=REGIONS) == ()


def test_compute_changed_genome_sections_does_not_use_fuzzy_equivalence() -> None:
    changed = compute_changed_genome_sections(
        _genetic_code(INIT_GEOMETRY="Use a triangular lattice."),
        _genetic_code(INIT_GEOMETRY="Use a triangular grid."),
        expected_section_names=REGIONS,
    )

    assert changed == ("INIT_GEOMETRY",)


def test_compute_changed_genome_sections_is_relative_to_maternal_base() -> None:
    mother = _genetic_code(RADIUS_POLICY="Use uniform radii.")
    child = _genetic_code(RADIUS_POLICY="Use father-style non-uniform radii.")

    assert compute_changed_genome_sections(mother, child, expected_section_names=REGIONS) == ("RADIUS_POLICY",)


def test_parse_implementation_patch_response_accepts_full_response() -> None:
    patch = parse_implementation_patch_response(
        _full_patch_response(),
        expected_mode="FULL",
        expected_region_names=REGIONS,
    )

    assert patch.compilation_mode == "FULL"
    assert tuple(name for name, _body in patch.region_bodies) == REGIONS


def test_parse_implementation_patch_response_accepts_patch_subset() -> None:
    patch = parse_implementation_patch_response(
        _patch_response("CONFLICT_MODEL", "REPAIR_POLICY"),
        expected_mode="PATCH",
        expected_region_names=("CONFLICT_MODEL", "REPAIR_POLICY"),
    )

    assert patch.compilation_mode == "PATCH"
    assert tuple(name for name, _body in patch.region_bodies) == ("CONFLICT_MODEL", "REPAIR_POLICY")


def test_parse_implementation_patch_response_rejects_wrong_mode() -> None:
    with pytest.raises(ValueError, match="must be PATCH"):
        parse_implementation_patch_response(
            _full_patch_response(),
            expected_mode="PATCH",
            expected_region_names=REGIONS,
        )


def test_parse_implementation_patch_response_rejects_missing_region() -> None:
    text = _full_patch_response().replace(
        "\n## REGION PARAMETERS\n    # body for PARAMETERS\n## END_REGION\n",
        "\n",
    )

    with pytest.raises(ValueError, match="expected"):
        parse_implementation_patch_response(
            text,
            expected_mode="FULL",
            expected_region_names=REGIONS,
        )


def test_parse_implementation_patch_response_rejects_extra_region() -> None:
    with pytest.raises(ValueError, match="expected"):
        parse_implementation_patch_response(
            _patch_response("RADIUS_POLICY", "REPAIR_POLICY"),
            expected_mode="PATCH",
            expected_region_names=("RADIUS_POLICY",),
        )


def test_parse_implementation_patch_response_rejects_out_of_order_region() -> None:
    with pytest.raises(ValueError, match="expected"):
        parse_implementation_patch_response(
            _patch_response("REPAIR_POLICY", "CONFLICT_MODEL"),
            expected_mode="PATCH",
            expected_region_names=("CONFLICT_MODEL", "REPAIR_POLICY"),
        )


def test_parse_implementation_patch_response_rejects_missing_end_region() -> None:
    text = "## COMPILATION_MODE\nPATCH\n\n## REGION RADIUS_POLICY\n    radii = np.ones(26)\n"

    with pytest.raises(ValueError, match="missing ## END_REGION"):
        parse_implementation_patch_response(
            text,
            expected_mode="PATCH",
            expected_region_names=("RADIUS_POLICY",),
        )


def test_parse_implementation_patch_response_rejects_unknown_region_name() -> None:
    with pytest.raises(ValueError, match="expected"):
        parse_implementation_patch_response(
            _patch_response("UNKNOWN_REGION"),
            expected_mode="PATCH",
            expected_region_names=("RADIUS_POLICY",),
        )


def test_assemble_implementation_from_full_patch_inserts_all_regions() -> None:
    patch = parse_implementation_patch_response(
        _full_patch_response(),
        expected_mode="FULL",
        expected_region_names=REGIONS,
    )

    source = assemble_implementation_from_patch(
        scaffold_text=TEMPLATE,
        patch=patch,
        expected_region_names=REGIONS,
    )

    assert "# === FIXED: DO NOT MODIFY ===" in source
    assert "    # body for INIT_GEOMETRY" in source
    assert "    # body for OPTIONAL_CODE_SKETCH" in source


def test_assemble_implementation_from_patch_preserves_unchanged_regions_byte_for_byte() -> None:
    base_patch = parse_implementation_patch_response(
        _full_patch_response(),
        expected_mode="FULL",
        expected_region_names=REGIONS,
    )
    base_source = assemble_implementation_from_patch(
        scaffold_text=TEMPLATE.replace("import numpy as np", "import numpy as np\n# maternal fixed comment"),
        patch=base_patch,
        expected_region_names=REGIONS,
    )
    radius_body = "    radii = np.full(26, 0.05, dtype=float)\n"
    patch = ParsedImplementationPatch(
        compilation_mode="PATCH",
        region_bodies=(("RADIUS_POLICY", radius_body),),
    )

    final_source = assemble_implementation_from_patch(
        scaffold_text=TEMPLATE,
        patch=patch,
        expected_region_names=REGIONS,
        base_source_text=base_source,
    )
    base_bodies = dict(extract_region_bodies_from_source(base_source, expected_region_names=REGIONS))
    final_bodies = dict(extract_region_bodies_from_source(final_source, expected_region_names=REGIONS))

    assert "# maternal fixed comment" in final_source
    assert final_bodies["RADIUS_POLICY"] == radius_body
    for region in REGIONS:
        if region != "RADIUS_POLICY":
            assert final_bodies[region] == base_bodies[region]


def test_assemble_implementation_from_patch_requires_base_source() -> None:
    patch = ParsedImplementationPatch(compilation_mode="PATCH", region_bodies=(("RADIUS_POLICY", "    pass\n"),))

    with pytest.raises(ValueError, match="requires base_source_text"):
        assemble_implementation_from_patch(
            scaffold_text=TEMPLATE,
            patch=patch,
            expected_region_names=REGIONS,
        )


def test_assemble_implementation_from_patch_rejects_non_scaffold_base() -> None:
    patch = ParsedImplementationPatch(compilation_mode="PATCH", region_bodies=(("RADIUS_POLICY", "    pass\n"),))

    with pytest.raises(ValueError, match="expected"):
        assemble_implementation_from_patch(
            scaffold_text=TEMPLATE,
            patch=patch,
            expected_region_names=REGIONS,
            base_source_text="def run_packing():\n    pass\n",
        )


def test_non_circle_scaffold_and_patch_preserve_unchanged_regions() -> None:
    regions = parse_implementation_scaffold(OPTIMIZER_TEMPLATE, expected_region_names=OPTIMIZER_REGIONS)
    assert tuple(region.name for region in regions) == OPTIMIZER_REGIONS

    full_patch = ParsedImplementationPatch(
        compilation_mode="FULL",
        region_bodies=tuple(
            (region, f"        # optimizer body for {region}\n")
            for region in OPTIMIZER_REGIONS
        ),
    )
    base_source = assemble_implementation_from_patch(
        scaffold_text=OPTIMIZER_TEMPLATE,
        patch=full_patch,
        expected_region_names=OPTIMIZER_REGIONS,
    )
    patched_body = "        # patched optimizer update\n"
    patch = ParsedImplementationPatch(
        compilation_mode="PATCH",
        region_bodies=(("UPDATE_RULE", patched_body),),
    )

    final_source = assemble_implementation_from_patch(
        scaffold_text=OPTIMIZER_TEMPLATE,
        patch=patch,
        expected_region_names=OPTIMIZER_REGIONS,
        base_source_text=base_source,
    )
    base_bodies = dict(extract_region_bodies_from_source(base_source, expected_region_names=OPTIMIZER_REGIONS))
    final_bodies = dict(extract_region_bodies_from_source(final_source, expected_region_names=OPTIMIZER_REGIONS))

    assert final_bodies["UPDATE_RULE"] == patched_body
    for region in OPTIMIZER_REGIONS:
        if region != "UPDATE_RULE":
            assert final_bodies[region] == base_bodies[region]
