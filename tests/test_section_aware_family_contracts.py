"""Cross-family contracts for schema-driven section-aware evolution."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.evolve.manual_pipeline import load_manual_pipeline_context
from src.evolve.prompt_utils import load_prompt_bundle
from src.organisms.genetic_code_format import parse_genome_schema_text
from src.organisms.implementation_patch import (
    parse_implementation_scaffold,
    resolve_implementation_region_order,
)

ROOT = Path(__file__).resolve().parents[1]

FAMILIES = {
    "circle_packing_shinka": {
        "config": "config_circle_packing_shinka",
        "experiment": "unit_square_26",
        "sections": (
            "INIT_GEOMETRY",
            "RADIUS_POLICY",
            "EXPANSION_POLICY",
            "CONFLICT_MODEL",
            "REPAIR_POLICY",
            "CONTROL_POLICY",
            "PARAMETERS",
            "OPTIONAL_CODE_SKETCH",
        ),
        "implementation_regions": (
            "PARAMETERS",
            "INIT_GEOMETRY",
            "RADIUS_POLICY",
            "CONFLICT_MODEL",
            "REPAIR_POLICY",
            "EXPANSION_POLICY",
            "CONTROL_POLICY",
            "OPTIONAL_CODE_SKETCH",
        ),
    },
    "optimization_survey": {
        "config": "config_optimization_survey",
        "experiment": "xor_mlp",
        "sections": (
            "STATE_REPRESENTATION",
            "GRADIENT_PROCESSING",
            "UPDATE_RULE",
            "PARAMETER_GROUP_POLICY",
            "STEP_CONTROL_POLICY",
            "STABILITY_POLICY",
            "PARAMETERS",
            "OPTIONAL_CODE_SKETCH",
        ),
    },
    "awtf2025_heuristic": {
        "config": "config_awtf2025_heuristic",
        "experiment": "group_commands_and_wall_planning",
        "sections": (
            "STATE_REPRESENTATION",
            "MACRO_STRATEGY",
            "CONSTRUCTION_POLICY",
            "LOCAL_REPAIR_POLICY",
            "OPTIONAL_CODE_SKETCH",
        ),
    },
    "co_bench": {
        # ONE preset selects the active task via experiments.co_bench.CO_BENCH_TASK;
        # the default (no override) is TSP. The shared section schema / prompts are
        # task-agnostic, so any task exercises the same section-aware surface.
        "config": "config_co-bench",
        "experiment": "co_bench",
        # On disk the family lives under conf/experiments/co-bench (hyphen) while
        # the family/experiment key is co_bench (underscore).
        "dir": "co-bench",
        # CORE_GENES order (genome_schema.txt).
        "sections": (
            "CONSTRUCTION",
            "SEARCH_STRATEGY",
            "NEIGHBORHOOD",
            "FEASIBILITY_MODEL",
            "REPAIR_POLICY",
            "CONTROL_POLICY",
            "PARAMETERS",
            "OPTIONAL_CODE_SKETCH",
        ),
        # Scaffold execution order (shared/template.txt) differs from CORE_GENES
        # order (PARAMETERS executes first), like circle_packing_shinka.
        "implementation_regions": (
            "PARAMETERS",
            "CONSTRUCTION",
            "FEASIBILITY_MODEL",
            "NEIGHBORHOOD",
            "REPAIR_POLICY",
            "SEARCH_STRATEGY",
            "CONTROL_POLICY",
            "OPTIONAL_CODE_SKETCH",
        ),
    },
}


def _family_dir(family: str, spec: dict) -> str:
    """On-disk prompt directory for a family (defaults to the family key).

    Most families store prompts under conf/experiments/<family>/; co_bench is
    the exception (key co_bench, on-disk co-bench), so it sets spec["dir"].
    """

    return str(spec.get("dir", family))


def _compose(config_name: str):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        return compose(config_name=config_name)


def _prompt_path(relative_path: str) -> Path:
    return (ROOT / relative_path).resolve()


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_family_config_exposes_section_aware_surface(family: str) -> None:
    spec = FAMILIES[family]
    cfg = _compose(str(spec["config"]))

    family_dir = _family_dir(family, spec)
    assert cfg.evolver.prompts.genome_schema == f"conf/experiments/{family_dir}/prompts/shared/genome_schema.txt"
    assert cfg.evolver.reproduction.selection_score.mode == "weighted_sum"
    assert cfg.evolver.reproduction.selection_score.normalize_weights is True
    assert cfg.evolver.reproduction.selection_score.weights.simple_score == 1.0
    assert cfg.evolver.reproduction.selection_score.weights.inheritance_fitness == 0.0


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_family_schema_prompts_and_templates_are_section_aligned(family: str) -> None:
    spec = FAMILIES[family]
    expected_sections = tuple(spec["sections"])
    expected_regions = tuple(spec.get("implementation_regions", spec["sections"]))
    cfg = _compose(str(spec["config"]))
    bundle = load_prompt_bundle(cfg)

    schema_sections = parse_genome_schema_text(bundle.genome_schema)
    assert tuple(section.name for section in schema_sections) == expected_sections

    template_regions = resolve_implementation_region_order(
        bundle.implementation_template,
        expected_section_names=expected_regions,
    )
    assert template_regions == expected_regions

    raw_prompt_values = cfg.evolver.prompts
    generation_user_keys = ("seed_user", "mutation_user", "crossover_user")
    for key in generation_user_keys:
        text = _prompt_path(str(raw_prompt_values[key])).read_text(encoding="utf-8")
        assert "=== GENOME SECTION SCHEMA ===" in text
        assert "The schema below is authoritative for the structure and meaning of CORE_GENES." in text
        assert "{genome_schema}" in text

    for prompt in (bundle.seed_system, bundle.mutation_system, bundle.crossover_system):
        assert "## CORE_GENES" in prompt
        assert "## INTERACTION_NOTES" in prompt
        assert "## COMPUTE_NOTES" in prompt
        assert "## CHANGE_DESCRIPTION" in prompt
        for section_name in expected_sections:
            assert f"### {section_name}" in prompt

    for prompt in (bundle.mutation_novelty_system, bundle.crossover_novelty_system):
        assert "sectioned `## CORE_GENES`" in prompt
        assert "## NOVELTY_VERDICT" in prompt
        assert "## REJECTION_REASON" in prompt
        assert "## SECTIONS_AT_ISSUE" in prompt

    if family in ("circle_packing_shinka", "co_bench"):
        # co_bench shares the single-rewrite implementation contract: the LLM
        # returns the full implementation.py, not a region-marked patch.
        assert "Single rewrite contract:" in bundle.implementation_system
        assert "return ONLY the final full `implementation.py`" in bundle.implementation_system
        assert "treat it as the concrete parent program" in bundle.implementation_system
        assert "EVOLVE-BLOCK-START" in bundle.implementation_system
        assert "do not output a full `implementation.py`" not in bundle.implementation_system
    elif family == "awtf2025_heuristic":
        # The awtf2025_heuristic implementation prompt was rewritten after the
        # 426-organism atcoder run: small local models were tripping on the
        # multi-convention markup (## END_REGION: NAME and similar). The new
        # contract leads with a concrete artifact example and forbids the
        # decorated end marker outright.
        assert "## COMPILATION_MODE" in bundle.implementation_system
        assert "The first non-empty line of the answer is exactly `## COMPILATION_MODE`." in bundle.implementation_system
        assert "Every region closes with the bare line `## END_REGION`." in bundle.implementation_system
        assert "Do not write `## END_REGION: SECTION_NAME`." in bundle.implementation_system
        assert "Concrete minimal example for FULL mode" in bundle.implementation_system
    else:
        assert "## COMPILATION_MODE" in bundle.implementation_system
        assert "The first line of your answer must be `## COMPILATION_MODE`." in bundle.implementation_system
        assert "do not start with `REGION ...`" in bundle.implementation_system
        assert "Every `## REGION SECTION_NAME` block must be closed by `## END_REGION`" in bundle.implementation_system
        assert "do not output a full `implementation.py`" in bundle.implementation_system
        assert "Execution-order discipline" in bundle.implementation_system
    if family in ("circle_packing_shinka", "co_bench"):
        # Single-rewrite families do not negotiate a FULL/PATCH compilation mode.
        assert "=== COMPILATION MODE ===" not in bundle.implementation_user
    else:
        assert "=== COMPILATION MODE ===" in bundle.implementation_user
    assert "=== CHANGED_SECTIONS ===" in bundle.implementation_user
    assert "=== MATERNAL BASE GENETIC CODE ===" in bundle.implementation_user
    assert "=== MATERNAL BASE IMPLEMENTATION ===" in bundle.implementation_user
    if family == "awtf2025_heuristic":
        # The atcoder rewrite replaced the raw scaffold display with a structured
        # region-order list to keep the LLM from copying the # === REGION ===
        # markers verbatim into its artifact.
        assert "=== CANONICAL REGION ORDER ===" in bundle.implementation_user
        assert "=== PRE-DEFINED LOCAL VARIABLES ===" in bundle.implementation_user
    else:
        assert "=== CANONICAL IMPLEMENTATION SCAFFOLD ===" in bundle.implementation_user
    assert "Do NOT add commentary before or after the file" in bundle.repair_system
    assert "=== CANONICAL IMPLEMENTATION SCAFFOLD ===" in bundle.repair_user

    family_dir = _family_dir(family, spec)
    legacy_template = ROOT / "conf" / "experiments" / family_dir / "prompts" / "implementation" / "template.txt"
    if legacy_template.exists():
        legacy_text = legacy_template.read_text(encoding="utf-8")
        assert "EDITABLE:" not in legacy_text
        parse_implementation_scaffold(legacy_text, expected_region_names=expected_regions)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_manual_pipeline_loads_section_aware_prompt_bundle_for_each_family(family: str) -> None:
    spec = FAMILIES[family]

    context = load_manual_pipeline_context(
        config_name=str(spec["config"]),
        experiment_name=str(spec["experiment"]),
    )

    assert context.prompt_bundle.genome_schema
    assert context.prompt_bundle.implementation_template
