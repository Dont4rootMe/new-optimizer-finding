"""End-to-end coverage of the two-step design pipeline plumbing.

These tests check that builders return a populated DesignPromptBundle when
the family ships rationalization prompts, that the placeholder is correctly
substituted, and that single-call mode keeps working when rationalization
assets are absent.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from src.evolve.prompt_utils import (
    DesignPromptBundle,
    RATIONALIZATION_PLACEHOLDER,
    RATIONALIZATION_SINGLE_CALL_STUB,
    load_prompt_bundle,
)
from src.evolve.types import OrganismMeta
from src.organisms.crossbreeding import build_crossover_design_bundle
from src.organisms.mutation import build_mutation_design_bundle
from src.organisms.organism import save_organism_artifacts


ROOT = Path(__file__).resolve().parents[1]


def _compose_cfg(name: str):
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        return compose(config_name=name)


def _make_parent(tmp_path: Path, name: str, island_id: str) -> OrganismMeta:
    org_dir = tmp_path / name
    org_dir.mkdir(parents=True, exist_ok=True)
    (org_dir / "implementation.py").write_text(
        "from __future__ import annotations\n\n"
        "def solve_case(input_text: str) -> str:\n"
        "    return ''\n",
        encoding="utf-8",
    )
    organism = OrganismMeta(
        organism_id=name,
        island_id=island_id,
        generation_created=0,
        current_generation_active=0,
        timestamp="2026-01-01T00:00:00Z",
        mother_id=None,
        father_id=None,
        operator="seed",
        genetic_code_path=str(org_dir / "genetic_code.md"),
        implementation_path=str(org_dir / "implementation.py"),
        lineage_path=str(org_dir / "lineage.json"),
        organism_dir=str(org_dir),
        prompt_hash="abc",
        seed=123,
    )
    save_organism_artifacts(
        organism,
        genetic_code={
            "core_genes": [
                "Idea about partition structure",
                "Idea about deterministic local repair",
            ],
            "interaction_notes": "Reference parent notes.",
            "compute_notes": "Reference parent compute notes.",
        },
        lineage=[
            {
                "generation": 0,
                "operator": "seed",
                "mother_id": None,
                "father_id": None,
                "change_description": "Initial seed with no_walls + quadrant grouping.",
                "selected_simple_experiments": ["group_commands_and_wall_planning"],
                "selected_hard_experiments": [],
                "simple_score": -1000.0,
                "hard_score": None,
                "cross_island": False,
                "father_island_id": None,
            }
        ],
    )
    return organism


def test_awtf2025_mutation_bundle_carries_rationalization_prompts(tmp_path: Path) -> None:
    cfg = _compose_cfg("config_awtf2025_heuristic")
    bundle = load_prompt_bundle(cfg)
    assert bundle.supports_two_step_mutation
    parent = _make_parent(tmp_path, "parent_a", "macro_partitioning")

    design_bundle = build_mutation_design_bundle(
        parent=parent,
        prompts=bundle,
        family_id="awtf2025_heuristic",
    )

    assert isinstance(design_bundle, DesignPromptBundle)
    assert design_bundle.is_two_step
    assert design_bundle.rationalization_system
    assert design_bundle.rationalization_user
    assert "{parent_genetic_code}" not in design_bundle.rationalization_user
    # Step-2 user template still carries the literal {rationalization} placeholder.
    assert RATIONALIZATION_PLACEHOLDER in design_bundle.formalization_user_template
    # Regime hint is computed (lineage has one ancestor → "unclear" path).
    assert design_bundle.lineage_regime_hint  # non-empty string


def test_render_formalization_substitutes_step1_output_or_stub() -> None:
    template = (
        "before\n\n=== STEP 1 ===\n" + RATIONALIZATION_PLACEHOLDER + "\n=== END ===\nafter"
    )
    bundle = DesignPromptBundle(
        formalization_system="sys",
        formalization_user_template=template,
    )
    # Two-step path: real text gets substituted.
    sys_a, user_a = bundle.render_formalization(rationale_text="## SCORE_BEARING_CORE\nfoo")
    assert sys_a == "sys"
    assert "## SCORE_BEARING_CORE" in user_a
    assert RATIONALIZATION_PLACEHOLDER not in user_a

    # Single-call path: stub substituted.
    _, user_b = bundle.render_formalization(rationale_text=None)
    assert RATIONALIZATION_SINGLE_CALL_STUB in user_b


def test_circle_packing_crossover_bundle_supports_two_step(tmp_path: Path) -> None:
    cfg = _compose_cfg("config_circle_packing_shinka")
    bundle = load_prompt_bundle(cfg)
    assert bundle.supports_two_step_crossover
    mother = _make_parent(tmp_path, "mother_c", "symmetric_constructions")
    father = _make_parent(tmp_path, "father_c", "iterative_repair")

    design_bundle = build_crossover_design_bundle(
        mother=mother,
        father=father,
        prompts=bundle,
        family_id="circle_packing_shinka",
    )

    assert design_bundle.is_two_step
    assert "WHAT_TO_REMOVE" in design_bundle.rationalization_system
    assert "WHAT_TO_ADD_OR_INVENT" in design_bundle.rationalization_system
    assert RATIONALIZATION_PLACEHOLDER in design_bundle.formalization_user_template


def test_legacy_optimization_survey_bundle_is_single_call() -> None:
    cfg = _compose_cfg("config_optimization_survey")
    bundle = load_prompt_bundle(cfg)
    # optimization_survey hasn't migrated → no rationalization prompts shipped.
    assert not bundle.supports_two_step_mutation
    assert not bundle.supports_two_step_crossover
