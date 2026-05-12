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


def test_sectioned_parser_accepts_fenced_optional_code_sketch_block() -> None:
    text = SECTIONED_GENETIC_CODE.replace(
        "### OPTIONAL_CODE_SKETCH\n- None.",
        (
            "### OPTIONAL_CODE_SKETCH\n"
            "```python\n"
            "# local repair sketch\n"
            "def nudge(circle, direction):\n"
            "    return circle + 0.25 * direction\n"
            "```"
        ),
    )

    parsed = parse_genetic_code_text(text, expected_section_names=SCHEMA_SECTION_NAMES)

    assert parsed.core_gene_sections is not None
    optional = parsed.core_gene_sections[-1]
    assert optional.name == "OPTIONAL_CODE_SKETCH"
    assert optional.entries[0].text == (
        "```python\n"
        "# local repair sketch\n"
        "def nudge(circle, direction):\n"
        "    return circle + 0.25 * direction\n"
        "```"
    )


def test_sectioned_parser_accepts_bullet_wrapped_fenced_optional_code_sketch_block() -> None:
    """Real-world LLM output wraps fenced code blocks inside a bullet, e.g.

        - BFS for distance:
        - ```python
          def get_dist_map(...):
              ...
          ```

    The original parser only recognized fenced openers when the line started
    with a bare ``` (no bullet wrapper), so the closing ``` was misinterpreted
    as a fresh fenced opener and the parser raised
    ``unterminated fenced code block`` on rereads of the saved genetic_code.md.
    This regression test pins the relaxed behavior.
    """
    text = SECTIONED_GENETIC_CODE.replace(
        "### OPTIONAL_CODE_SKETCH\n- None.",
        (
            "### OPTIONAL_CODE_SKETCH\n"
            "- BFS for distance:\n"
            "- ```python\n"
            "  def get_dist_map(N, start_node):\n"
            "      dist = {start_node: 0}\n"
            "      return dist\n"
            "  ```"
        ),
    )

    parsed = parse_genetic_code_text(text, expected_section_names=SCHEMA_SECTION_NAMES)

    assert parsed.core_gene_sections is not None
    optional = parsed.core_gene_sections[-1]
    assert optional.name == "OPTIONAL_CODE_SKETCH"
    assert len(optional.entries) == 2
    assert optional.entries[0].text == "BFS for distance:"
    assert "def get_dist_map" in optional.entries[1].text
    assert optional.entries[1].text.strip().endswith("```")


AWTF_SECTION_NAMES = (
    "STATE_REPRESENTATION",
    "MACRO_STRATEGY",
    "CONSTRUCTION_POLICY",
    "LOCAL_REPAIR_POLICY",
    "OPTIONAL_CODE_SKETCH",
)

AWTF_GENETIC_CODE = """## CORE_GENES
### STATE_REPRESENTATION
- Quadrant labels are derived from `N // 2` for both axes.

### MACRO_STRATEGY
- The organism partitions the board into four quadrants linked by a central hub.

### CONSTRUCTION_POLICY
- Walls are placed around the hub except near the center.

### LOCAL_REPAIR_POLICY
- Stuck robots retry with individual moves once group commands stall.

### OPTIONAL_CODE_SKETCH
- None.

## INTERACTION_NOTES
The quadrants and hub combine to bound concurrent flows.

## COMPUTE_NOTES
Construction is `O(K * N^2)` worst case.

## CHANGE_DESCRIPTION
Tests quadrant routing with a central hub.
"""


@pytest.mark.parametrize(
    "section_name",
    ["STATE_REPRESENTATION", "CONSTRUCTION_POLICY", "LOCAL_REPAIR_POLICY"],
)
def test_sectioned_parser_accepts_fenced_block_in_code_bearing_subsection(section_name: str) -> None:
    """The 1516-organism post-mortem fix (P7) tolerated fenced code blocks only
    in `OPTIONAL_CODE_SKETCH`. The 491-organism awtf2025 run produced 102
    `failed_creation` cases (~20% of population) where the LLM emitted a fenced
    block inside `STATE_REPRESENTATION` (and occasionally `CONSTRUCTION_POLICY`
    or `LOCAL_REPAIR_POLICY`), parser raised "non-bullet text" and the organism
    died at creation. The schema explicitly permits code in those sections, so
    the parser tolerance now extends across all code-bearing subsections.
    """
    bullet_to_replace = {
        "STATE_REPRESENTATION": "- Quadrant labels are derived from `N // 2` for both axes.",
        "CONSTRUCTION_POLICY": "- Walls are placed around the hub except near the center.",
        "LOCAL_REPAIR_POLICY": "- Stuck robots retry with individual moves once group commands stall.",
    }[section_name]

    text = AWTF_GENETIC_CODE.replace(
        bullet_to_replace,
        bullet_to_replace
        + "\n```python\n"
        + "# scratch sketch the LLM emitted inline\n"
        + "def helper(n):\n"
        + "    return n // 2\n"
        + "```",
    )

    parsed = parse_genetic_code_text(text, expected_section_names=AWTF_SECTION_NAMES)

    assert parsed.core_gene_sections is not None
    target = next(section for section in parsed.core_gene_sections if section.name == section_name)
    assert any("def helper" in entry.text for entry in target.entries)


def test_sectioned_parser_accepts_bullet_wrapped_fenced_block_in_state_representation() -> None:
    """Real awtf2025 organisms emitted bullet-wrapped fenced blocks in
    `STATE_REPRESENTATION` (e.g. `- ```python ... ```` `). Mirrors the existing
    OPTIONAL_CODE_SKETCH regression test for symmetry.
    """
    text = AWTF_GENETIC_CODE.replace(
        "- Quadrant labels are derived from `N // 2` for both axes.",
        (
            "- Quadrant labels are derived from `N // 2` for both axes.\n"
            "- ```python\n"
            "  def get_quad(pos, n):\n"
            "      r, c = pos\n"
            "      return (r >= n // 2) * 2 + (c >= n // 2)\n"
            "  ```"
        ),
    )

    parsed = parse_genetic_code_text(text, expected_section_names=AWTF_SECTION_NAMES)

    assert parsed.core_gene_sections is not None
    state_section = parsed.core_gene_sections[0]
    assert state_section.name == "STATE_REPRESENTATION"
    assert len(state_section.entries) == 2
    assert "def get_quad" in state_section.entries[1].text


def test_sectioned_parser_still_rejects_fenced_block_in_macro_strategy() -> None:
    """MACRO_STRATEGY must remain plain-language bullets; fenced blocks there
    are a contract violation per the genome schema and remain a hard failure.
    """
    text = AWTF_GENETIC_CODE.replace(
        "- The organism partitions the board into four quadrants linked by a central hub.",
        (
            "- The organism partitions the board into four quadrants linked by a central hub.\n"
            "```python\n"
            "for q in range(4):\n"
            "    pass\n"
            "```"
        ),
    )

    with pytest.raises(ValueError, match="non-bullet text"):
        parse_genetic_code_text(text, expected_section_names=AWTF_SECTION_NAMES)


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


def test_sectioned_parser_round_trips_bullet_wrapped_fenced_code() -> None:
    """Regression for the gen_0002 poison-parent bug.

    Pipeline: the LLM emits an entry whose first line is ``- ```python`` at
    indent 0 — a bulleted fence opener. The parser collects everything up to
    the closing ``\\`\\`\\``` as a single entry. Then ``_render_gene_entry``
    re-emits the entry with ``- `` prefix on the head and ``  `` (2 spaces)
    prefix on every continuation, so the file on disk has ``- - ```python``
    followed by 2-space-indented body and a 2-space-indented closing fence.

    An earlier version of the parser fired its fence-opener detector on any
    line whose ``strip()`` started with ``\\`\\`\\``` — including the indented
    closing fence — which made the parser think a new fence was opening,
    swallow the next subsection's content, and finally raise
    ``unterminated fenced code block`` at EOF. Every descendant of the
    affected ``simple_complete`` parent died at parent-load with
    ``Canonical genetic code at <path> is malformed``.

    The fix gates fence-opener detection on ``not line.startswith(" ")``: an
    indented ``\\`\\`\\``` is always a continuation, never a fresh opener.
    """

    from src.evolve.storage import (
        _render_genetic_code,
        parse_genetic_code_text as parse_for_storage,
    )

    poison_parent = """## CORE_GENES
### STATE_REPRESENTATION
- Distance fields are computed for each robot via BFS from its target.
- - ```python
    def get_multi_source_bfs(sources, N, walls):
        dist = [[float('inf')] * N for _ in range(N)]
        return dist
    ```

### MACRO_STRATEGY
- Robots are grouped by target quadrant for synchronized movement.

### CONSTRUCTION_POLICY
- A central cross of added walls partitions the board.
- - ```python
      def select_group_dir(group_id, bfs_maps):
          return 'U'
      ```

### LOCAL_REPAIR_POLICY
- Individual escalation triggers when group movement stalls.
- - ```python
      if group_stagnant[g_id] > N:
          emit_op('i', robot, dir)
      ```

### OPTIONAL_CODE_SKETCH
- None.

## INTERACTION_NOTES
Group routing and stuck-robot escalation interact through stagnation counters.

## COMPUTE_NOTES
O(K * N^2) per generation for BFS and direction selection.

## CHANGE_DESCRIPTION
Quadrant-corridor partitioning with weighted group-BFS routing.
"""

    parsed_payload_1 = parse_for_storage(
        poison_parent, expected_section_names=AWTF_SECTION_NAMES
    )
    rendered = _render_genetic_code(parsed_payload_1)
    parsed_payload_2 = parse_for_storage(
        rendered, expected_section_names=AWTF_SECTION_NAMES
    )

    assert parsed_payload_1 == parsed_payload_2
    section_names = tuple(s["name"] for s in parsed_payload_2["core_gene_sections"])
    assert section_names == AWTF_SECTION_NAMES


def test_sectioned_parser_auto_completes_missing_optional_section() -> None:
    """Local Ollama models routinely drop the trailing
    ``### OPTIONAL_CODE_SKETCH`` even when the prompt insists on it. Of 31
    ``failed_creation`` cases in one 80-organism run, 18 were caused by the
    LLM emitting the first 4 required subsections in correct order and
    simply omitting the optional 5th. The parser now treats that single
    omission as if the LLM had written ``### OPTIONAL_CODE_SKETCH\\n- None.``
    so the design survives instead of consuming a retry budget.

    Any structural mismatch elsewhere (wrong order, wrong names, missing
    required subsection) must still raise — the auto-complete only fires
    when the actual subsections match ``expected_section_names[:-1]``
    exactly.
    """

    truncated = """## CORE_GENES
### STATE_REPRESENTATION
- Quadrant labels are derived from `N // 2` for both axes.

### MACRO_STRATEGY
- The organism partitions the board into four quadrants linked by a central hub.

### CONSTRUCTION_POLICY
- Walls are placed around the hub except near the center.

### LOCAL_REPAIR_POLICY
- Stuck robots retry with individual moves once group commands stall.

## INTERACTION_NOTES
The quadrants and hub combine to bound concurrent flows.

## COMPUTE_NOTES
Construction is `O(K * N^2)` worst case.

## CHANGE_DESCRIPTION
Tests quadrant routing with a central hub.
"""

    from src.evolve.storage import parse_genetic_code_text as parse_for_storage

    parsed = parse_for_storage(
        truncated, expected_section_names=AWTF_SECTION_NAMES
    )

    section_names = tuple(s["name"] for s in parsed["core_gene_sections"])
    assert section_names == AWTF_SECTION_NAMES
    optional = parsed["core_gene_sections"][-1]
    assert optional["name"] == "OPTIONAL_CODE_SKETCH"
    assert optional["entries"] == ["None."]


def test_sectioned_parser_does_not_auto_complete_when_required_subsection_missing() -> None:
    """The auto-complete must NOT fire for a required-subsection omission;
    those still need to fail loudly so the LLM retries.
    """

    missing_required = """## CORE_GENES
### STATE_REPRESENTATION
- Quadrant labels are derived from `N // 2`.

### MACRO_STRATEGY
- Four quadrants linked by a central hub.

### LOCAL_REPAIR_POLICY
- Stuck robots retry with individual moves.

## INTERACTION_NOTES
Body.

## COMPUTE_NOTES
Body.

## CHANGE_DESCRIPTION
Body.
"""

    with pytest.raises(ValueError, match="must match the genome schema exactly"):
        parse_genetic_code_text(
            missing_required, expected_section_names=AWTF_SECTION_NAMES
        )
