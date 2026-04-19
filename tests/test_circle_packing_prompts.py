"""Prompt-level contract tests for circle_packing_shinka."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from src.evolve.prompt_utils import load_prompt_bundle
from src.evolve.types import OrganismMeta
from src.organisms.crossbreeding import _build_crossbreed_prompt
from src.organisms.mutation import _build_mutate_prompt
from src.organisms.organism import save_organism_artifacts

ROOT = Path(__file__).resolve().parents[1]


def _compose_cfg():
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        return compose(config_name="config_circle_packing_shinka")


def _make_parent(tmp_path: Path, name: str, island_id: str) -> OrganismMeta:
    org_dir = tmp_path / name
    org_dir.mkdir(parents=True, exist_ok=True)
    implementation_path = org_dir / "implementation.py"
    implementation_path.write_text(
        "import numpy as np\n\n"
        "def run_packing():\n"
        "    centers = np.zeros((26, 2), dtype=float)\n"
        "    radii = np.zeros(26, dtype=float)\n"
        "    return centers, radii, float(np.sum(radii))\n",
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
        implementation_path=str(implementation_path),
        lineage_path=str(org_dir / "lineage.json"),
        organism_dir=str(org_dir),
        prompt_hash="abc",
        seed=123,
    )
    save_organism_artifacts(
        organism,
        genetic_code={
            "core_genes": [
                "Idea about global organization of the packing process",
                "Idea about feasibility maintenance under geometric constraints",
                "Idea about deterministic improvement of candidate placements",
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
                "change_description": "Initial research direction.",
                "selected_simple_experiments": ["unit_square_26"],
                "selected_hard_experiments": [],
                "simple_score": 1.0,
                "hard_score": None,
                "cross_island": False,
                "father_island_id": None,
            }
        ],
    )
    return organism


def test_circle_packing_prompt_bundle_uses_typed_json_language() -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)

    assert "The canonical hypothesis is `genome.json`." in bundle.project_context
    assert "The design-stage output is a typed JSON design artifact." in bundle.project_context
    assert "Do not invent new major ideas at implementation time." in bundle.implementation_system
    assert "Return only valid JSON." in bundle.mutation_system
    assert "The slot assignment is the primary artifact." in bundle.crossover_system
    assert "Do not emit `CORE_GENES`, `INTERACTION_NOTES`, `COMPUTE_NOTES`, or `CHANGE_DESCRIPTION`" in bundle.seed_system
    assert "=== CANDIDATE CHILD SLOT ASSIGNMENTS ===" in bundle.mutation_novelty_user
    assert "=== PRIMARY PARENT SLOT ASSIGNMENTS ===" in bundle.crossover_novelty_user


def test_circle_packing_mutation_prompt_prioritizes_child_draft(tmp_path: Path) -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)
    parent = _make_parent(tmp_path, "parent_a", "symmetric_constructions")

    _, user_prompt = _build_mutate_prompt(
        inherited_genes=[
            "Child idea about structural organization",
            "Child idea about feasibility preservation",
            "Child idea about refinement logic",
        ],
        removed_genes=["Excluded idea about older organization"],
        parent=parent,
        prompts=bundle,
    )

    assert "=== CHILD SLOT DRAFT ===" in user_prompt
    assert "=== EXCLUDED MODULES ===" in user_prompt
    assert "Prior rejected attempts to address inside this typed JSON response:" in user_prompt
    assert "=== PARENT GENOME SUMMARY ===" in user_prompt
    assert "Preserve every slot that is not declared in changed_slots." in user_prompt
    assert "IMPLEMENTATION CODE" not in user_prompt
    assert user_prompt.index("=== CHILD SLOT DRAFT ===") < user_prompt.index(
        "=== PARENT SLOT ASSIGNMENTS ==="
    )


def test_circle_packing_crossover_prompt_prioritizes_child_draft(tmp_path: Path) -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)
    mother = _make_parent(tmp_path, "mother_a", "symmetric_constructions")
    father = _make_parent(tmp_path, "father_b", "iterative_repair")

    _, user_prompt = _build_crossbreed_prompt(
        inherited_genes=[
            "Merged child idea about overall construction",
            "Merged child idea about feasibility maintenance",
            "Merged child idea about refinement policy",
        ],
        mother=mother,
        father=father,
        prompts=bundle,
    )

    assert "=== CHILD SLOT DRAFT ===" in user_prompt
    assert "Prior rejected attempts to address inside this typed JSON response:" in user_prompt
    assert "=== PRIMARY PARENT SLOT ASSIGNMENTS ===" in user_prompt
    assert "=== SECONDARY PARENT SLOT ASSIGNMENTS ===" in user_prompt
    assert "typed recombination" in user_prompt
    assert "IMPLEMENTATION CODE" not in user_prompt
    assert user_prompt.index("=== CHILD SLOT DRAFT ===") < user_prompt.index(
        "=== PRIMARY PARENT SLOT ASSIGNMENTS ==="
    )


def test_circle_packing_mutation_prompt_renders_novelty_feedback(tmp_path: Path) -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)
    parent = _make_parent(tmp_path, "parent_feedback", "symmetric_constructions")

    _, user_prompt = _build_mutate_prompt(
        inherited_genes=[
            "Child idea about structural organization",
            "Child idea about feasibility preservation",
            "Child idea about refinement logic",
        ],
        removed_genes=["Excluded idea about older organization"],
        parent=parent,
        prompts=bundle,
        novelty_feedback=["The previous child only paraphrased the parent layout idea."],
    )

    assert "- The previous child only paraphrased the parent layout idea." in user_prompt


def test_circle_packing_prompts_avoid_solution_leading_lists() -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)
    root = ROOT / "conf" / "experiments" / "circle_packing_shinka" / "prompts"

    symmetric_text = (root / "islands" / "symmetric_constructions.txt").read_text(encoding="utf-8")
    iterative_text = (root / "islands" / "iterative_repair.txt").read_text(encoding="utf-8")

    banned = ("rings", "rows", "lattices", "greedy local moves")
    for token in banned:
        assert token not in bundle.seed_system.lower()
        assert token not in bundle.mutation_system.lower()
        assert token not in bundle.crossover_system.lower()
        assert token not in symmetric_text.lower()
        assert token not in iterative_text.lower()
