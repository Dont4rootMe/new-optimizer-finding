"""Prompt-level contract tests for circle_packing_shinka."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from src.evolve.operators import SeedOperator
from src.evolve.prompt_utils import load_prompt_bundle
from src.evolve.types import Island, OrganismMeta
from src.organisms.compatibility import (
    build_crossover_compatibility_prompt,
    build_mutation_compatibility_prompt,
    build_seed_compatibility_prompt,
)
from src.organisms.crossbreeding import _build_crossbreed_prompt
from src.organisms.mutation import _build_mutate_prompt
from src.organisms.organism import save_organism_artifacts

ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = ROOT / "conf" / "experiments" / "circle_packing_shinka" / "prompts"
SCHEMA_BLOCK = (
    "=== GENOME SECTION SCHEMA ===\n"
    "The schema below is authoritative for the structure and meaning of CORE_GENES.\n"
    "{genome_schema}"
)
SECTION_HEADINGS = (
    "### INIT_GEOMETRY",
    "### RADIUS_POLICY",
    "### EXPANSION_POLICY",
    "### CONFLICT_MODEL",
    "### REPAIR_POLICY",
    "### CONTROL_POLICY",
    "### PARAMETERS",
    "### OPTIONAL_CODE_SKETCH",
)


def _sectioned_core_genes_body() -> str:
    return "\n\n".join(
        f"{heading}\n- {'None.' if heading.endswith('OPTIONAL_CODE_SKETCH') else 'Section-local test idea.'}"
        for heading in SECTION_HEADINGS
    )


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


def test_circle_packing_prompt_bundle_uses_gene_centric_language() -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)

    assert "primary object is the organism's genetic code" in bundle.project_context
    assert "# INIT_GEOMETRY" in bundle.genome_schema
    assert "do not invent new major ideas at implementation time" in bundle.implementation_system
    assert "child genetic code draft" in bundle.mutation_system.lower()
    assert "child draft" in bundle.crossover_system.lower()
    assert "keep it essentially intact" in bundle.mutation_system
    assert "valid source of novelty" in bundle.mutation_novelty_user
    assert "preserves substantial material from both parents" in bundle.crossover_novelty_user
    assert "## COMPATIBILITY_VERDICT" in bundle.compatibility_seed_system
    assert "compatibility is not the same as novelty" in bundle.compatibility_mutation_system


def test_circle_packing_generation_prompt_files_are_section_schema_aware() -> None:
    prompt_files = {
        "seed_system": PROMPT_ROOT / "seed" / "system.txt",
        "seed_user": PROMPT_ROOT / "seed" / "user.txt",
        "mutation_system": PROMPT_ROOT / "mutation" / "system.txt",
        "mutation_user": PROMPT_ROOT / "mutation" / "user.txt",
        "crossover_system": PROMPT_ROOT / "crossover" / "system.txt",
        "crossover_user": PROMPT_ROOT / "crossover" / "user.txt",
    }
    prompts = {name: path.read_text(encoding="utf-8") for name, path in prompt_files.items()}

    for name, text in prompts.items():
        assert text.strip()
        if name.endswith("_user"):
            assert SCHEMA_BLOCK in text
        if name.endswith("_system"):
            assert "## CORE_GENES" in text
            assert "## INTERACTION_NOTES" in text
            assert "## COMPUTE_NOTES" in text
            assert "## CHANGE_DESCRIPTION" in text
            assert "full end-to-end implementation" in text
            assert "put code in `CHANGE_DESCRIPTION`" in text or "`CHANGE_DESCRIPTION` must remain plain-language" in text
            positions = [text.index(heading) for heading in SECTION_HEADINGS]
            assert positions == sorted(positions)


def test_circle_packing_generation_prompts_removed_flat_and_global_ban_contracts() -> None:
    texts = [
        (PROMPT_ROOT / "seed" / "system.txt").read_text(encoding="utf-8"),
        (PROMPT_ROOT / "mutation" / "system.txt").read_text(encoding="utf-8"),
        (PROMPT_ROOT / "crossover" / "system.txt").read_text(encoding="utf-8"),
        (PROMPT_ROOT / "shared" / "project_context.txt").read_text(encoding="utf-8"),
    ]
    combined = "\n".join(texts)

    assert "At least 3 bullets, at most 7" not in combined
    assert "Genes must NEVER contain" not in combined
    assert "Genes and CHANGE_DESCRIPTION are idea-level artifacts. They must NEVER contain" not in combined
    assert "Hardcoded numeric constants" not in combined
    assert "No Python code, no variable names, no hardcoded constants" not in combined
    assert "full end-to-end implementation" in combined
    assert "must not contain Python code, code-like snippets" in combined


def test_circle_packing_generation_prompt_rendering_includes_schema(tmp_path: Path) -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)
    parent = _make_parent(tmp_path, "parent_schema", "symmetric_constructions")
    secondary = _make_parent(tmp_path, "secondary_schema", "iterative_repair")
    island = Island(
        island_id="symmetric_constructions",
        name="Symmetric constructions",
        description_path="",
        description_text="Prefer coherent geometric constructions.",
    )

    _, seed_user = SeedOperator(island).build_prompts(bundle)
    _, mutation_user = _build_mutate_prompt(
        inherited_genes=["Child draft idea about local geometry"],
        removed_genes=["Excluded draft idea"],
        parent=parent,
        prompts=bundle,
    )
    _, crossover_user = _build_crossbreed_prompt(
        inherited_genes=["Merged child draft idea about local geometry"],
        mother=parent,
        father=secondary,
        prompts=bundle,
    )

    for rendered in (seed_user, mutation_user, crossover_user):
        assert "=== GENOME SECTION SCHEMA ===" in rendered
        assert "{genome_schema}" not in rendered
        assert "# INIT_GEOMETRY" in rendered
        assert "# OPTIONAL_CODE_SKETCH" in rendered


def test_circle_packing_validation_prompt_rendering_includes_schema(tmp_path: Path) -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)
    parent = _make_parent(tmp_path, "parent_validation", "symmetric_constructions")
    secondary = _make_parent(tmp_path, "secondary_validation", "iterative_repair")
    candidate_design = {
        "CORE_GENES": _sectioned_core_genes_body(),
        "INTERACTION_NOTES": "The sectioned ideas are mutually supportive.",
        "COMPUTE_NOTES": "The design uses deterministic staged construction.",
        "CHANGE_DESCRIPTION": "A sectioned validation fixture.",
    }

    rendered_prompts = [
        build_seed_compatibility_prompt(
            candidate_design=candidate_design,
            prompts=bundle,
            expected_core_gene_sections=tuple(heading[4:] for heading in SECTION_HEADINGS),
        )[1],
        build_mutation_compatibility_prompt(
            inherited_genes=["Child draft sectioned idea"],
            removed_genes=["Excluded idea"],
            parent=parent,
            candidate_design=candidate_design,
            prompts=bundle,
            expected_core_gene_sections=tuple(heading[4:] for heading in SECTION_HEADINGS),
        )[1],
        build_crossover_compatibility_prompt(
            inherited_genes=["Merged sectioned idea"],
            mother=parent,
            father=secondary,
            candidate_design=candidate_design,
            prompts=bundle,
            expected_core_gene_sections=tuple(heading[4:] for heading in SECTION_HEADINGS),
        )[1],
    ]

    for rendered in rendered_prompts:
        assert "=== GENOME SECTION SCHEMA ===" in rendered
        assert "{genome_schema}" not in rendered
        assert "# INIT_GEOMETRY" in rendered
        assert "=== CANDIDATE" in rendered


def test_circle_packing_validation_prompts_are_section_schema_aware() -> None:
    bundle = load_prompt_bundle(_compose_cfg())

    novelty_prompts = (
        bundle.mutation_novelty_system,
        bundle.mutation_novelty_user,
        bundle.crossover_novelty_system,
        bundle.crossover_novelty_user,
    )
    for prompt in novelty_prompts:
        assert "sectioned `## CORE_GENES`" in prompt or "=== GENOME SECTION SCHEMA ===" in prompt
    for prompt in (bundle.mutation_novelty_system, bundle.crossover_novelty_system):
        assert "not as a flat list" in prompt
    for prompt in (bundle.mutation_novelty_user, bundle.crossover_novelty_user):
        assert "=== GENOME SECTION SCHEMA ===" in prompt
        assert "{genome_schema}" in prompt
    for prompt in (bundle.mutation_novelty_system, bundle.crossover_novelty_system):
        assert "## NOVELTY_VERDICT" in prompt
        assert "## REJECTION_REASON" in prompt
        assert "## SECTIONS_AT_ISSUE" in prompt
        assert "Do not propose edits" in prompt

    compatibility_prompts = (
        bundle.compatibility_seed_system,
        bundle.compatibility_seed_user,
        bundle.compatibility_mutation_system,
        bundle.compatibility_mutation_user,
        bundle.compatibility_crossover_system,
        bundle.compatibility_crossover_user,
    )
    for prompt in compatibility_prompts:
        assert prompt.strip()
    for prompt in (
        bundle.compatibility_seed_user,
        bundle.compatibility_mutation_user,
        bundle.compatibility_crossover_user,
    ):
        assert "=== GENOME SECTION SCHEMA ===" in prompt
        assert "{genome_schema}" in prompt
    for prompt in (
        bundle.compatibility_seed_system,
        bundle.compatibility_mutation_system,
        bundle.compatibility_crossover_system,
    ):
        assert "## COMPATIBILITY_VERDICT" in prompt
        assert "## REJECTION_REASON" in prompt
        assert "## SECTIONS_AT_ISSUE" in prompt
        assert "Do not propose edits" in prompt
        assert "full end-to-end implementation" in prompt or "full algorithm implementation" in prompt
    assert "compatibility is not the same as novelty" in bundle.compatibility_mutation_system
    assert "compatibility is not the same as novelty" in bundle.compatibility_crossover_system
    assert "{candidate_genetic_code}" in bundle.compatibility_seed_user
    assert "{candidate_change_description}" in bundle.compatibility_seed_user
    for placeholder in (
        "{inherited_gene_pool}",
        "{removed_gene_pool}",
        "{parent_genetic_code}",
        "{candidate_genetic_code}",
        "{candidate_change_description}",
    ):
        assert placeholder in bundle.compatibility_mutation_user
    for placeholder in (
        "{inherited_gene_pool}",
        "{mother_genetic_code}",
        "{father_genetic_code}",
        "{candidate_genetic_code}",
        "{candidate_change_description}",
    ):
        assert placeholder in bundle.compatibility_crossover_user


def test_circle_packing_implementation_and_repair_prompts_do_not_require_genome_schema() -> None:
    bundle = load_prompt_bundle(_compose_cfg())

    non_generation_prompts = (
        bundle.implementation_system,
        bundle.implementation_user,
        bundle.implementation_template,
        bundle.repair_system,
        bundle.repair_user,
    )
    assert all("{genome_schema}" not in prompt for prompt in non_generation_prompts)


def test_circle_packing_implementation_prompt_uses_patch_artifact_contract() -> None:
    bundle = load_prompt_bundle(_compose_cfg())
    combined_prompt = "\n".join((bundle.implementation_system, bundle.implementation_user))

    assert "do not output a full `implementation.py`" in bundle.implementation_system
    assert "## COMPILATION_MODE" in bundle.implementation_system
    assert "do not invent new major ideas at implementation time" in combined_prompt
    assert "Return ONLY valid Python source code" not in combined_prompt
    assert "=== COMPILATION MODE ===" in bundle.implementation_user
    assert "=== CHANGED_SECTIONS ===" in bundle.implementation_user
    assert "=== MATERNAL BASE GENETIC CODE ===" in bundle.implementation_user
    assert "=== MATERNAL BASE IMPLEMENTATION ===" in bundle.implementation_user
    assert "=== CANONICAL IMPLEMENTATION SCAFFOLD ===" in bundle.implementation_user
    assert "RUN_PACKING_BODY" not in bundle.implementation_template
    for region in tuple(heading[4:] for heading in SECTION_HEADINGS):
        assert f"## REGION {region}" in bundle.implementation_system
        assert f"# === REGION: {region} ===" in bundle.implementation_template
        assert f"# === END_REGION: {region} ===" in bundle.implementation_template


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

    assert "=== CHILD GENETIC CODE DRAFT ===" in user_prompt
    assert "=== EXCLUDED IDEAS ===" in user_prompt
    assert "=== NOVELTY REJECTION FEEDBACK ===" in user_prompt
    assert "REFERENCE ONLY" in user_prompt
    assert "keep it mostly intact rather than rebuilding the parent" in user_prompt
    assert "IMPLEMENTATION CODE" not in user_prompt
    assert user_prompt.index("=== CHILD GENETIC CODE DRAFT ===") < user_prompt.index(
        "=== PARENT GENETIC CODE (REFERENCE ONLY) ==="
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

    assert "=== CHILD GENETIC CODE DRAFT ===" in user_prompt
    assert "=== NOVELTY REJECTION FEEDBACK ===" in user_prompt
    assert "REFERENCE ONLY" in user_prompt
    assert "keep it mostly intact instead of rebuilding one parent" in user_prompt
    assert "IMPLEMENTATION CODE" not in user_prompt
    assert user_prompt.index("=== CHILD GENETIC CODE DRAFT ===") < user_prompt.index(
        "=== PRIMARY PARENT GENETIC CODE (REFERENCE ONLY) ==="
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
