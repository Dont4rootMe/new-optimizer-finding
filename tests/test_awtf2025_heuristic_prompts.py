"""Prompt-level contract tests for awtf2025_heuristic.

Updated for the `removing-genetic-sampling` branch: random gene-pruning and
gene-merging steps are gone; the LLM receives the parent's full genome and is
asked to design the child directly. The test fixtures still exercise the
prompt builders, but no longer assert child-draft / excluded-ideas framing.
"""

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
        return compose(config_name="config_awtf2025_heuristic")


def _make_parent(tmp_path: Path, name: str, island_id: str) -> OrganismMeta:
    org_dir = tmp_path / name
    org_dir.mkdir(parents=True, exist_ok=True)
    implementation_path = org_dir / "implementation.py"
    implementation_path.write_text(
        "from __future__ import annotations\n\n"
        "def solve_case(input_text: str) -> str:\n"
        "    lines = [line.strip() for line in input_text.splitlines() if line.strip()]\n"
        "    n, k = map(int, lines[0].split())\n"
        "    vertical = ['0' * (n - 1) for _ in range(n)]\n"
        "    horizontal = ['0' * n for _ in range(n - 1)]\n"
        "    groups = [str(i) for i in range(k)]\n"
        "    return '\\n'.join(vertical + horizontal + groups)\n",
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
                "Idea about macro board structure and where synchronized flow should happen",
                "Idea about how groups are assigned relative to route compatibility",
                "Idea about deterministic local repair when congestion appears",
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
                "selected_simple_experiments": ["group_commands_and_wall_planning"],
                "selected_hard_experiments": [],
                "simple_score": -1234.0,
                "hard_score": None,
                "cross_island": False,
                "father_island_id": None,
            }
        ],
    )
    return organism


def test_awtf2025_prompt_bundle_uses_full_genome_design_language() -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)

    assert "primary object is the organism's genetic code" in bundle.project_context
    # The new mutation/crossover prompts hand the LLM the parent's full genome
    # and explicitly tell it the random pre-LLM sampling step is gone.
    assert "no pre-sampled child draft" in bundle.mutation_system.lower()
    assert "no pre-sampled child draft" in bundle.crossover_system.lower()
    assert "verbatim repetition of the entire parent genome is an invalid mutation" in bundle.mutation_system
    assert "verbatim cloning of the primary parent is an invalid crossover" in bundle.crossover_system
    # Lineage-aware diversification nudge.
    assert "regime convergence" in bundle.mutation_user.lower()


def test_awtf2025_mutation_prompt_works_from_parent_full_genome(tmp_path: Path) -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)
    parent = _make_parent(tmp_path, "parent_a", "macro_partitioning")

    _, user_prompt = _build_mutate_prompt(
        inherited_genes=[],
        removed_genes=[],
        parent=parent,
        prompts=bundle,
    )

    # New framing: parent genome is the only source of truth; no draft / no excluded pool.
    assert "=== PARENT GENETIC CODE ===" in user_prompt
    assert "=== PARENT LINEAGE SUMMARY ===" in user_prompt
    assert "=== NOVELTY REJECTION FEEDBACK ===" in user_prompt
    # The dropped placeholders must not appear anywhere in the rendered prompt.
    assert "CHILD GENETIC CODE DRAFT" not in user_prompt
    assert "EXCLUDED IDEAS" not in user_prompt
    assert "{inherited_gene_pool}" not in user_prompt
    assert "{removed_gene_pool}" not in user_prompt
    assert "IMPLEMENTATION CODE" not in user_prompt


def test_awtf2025_crossover_prompt_works_from_two_parent_full_genomes(tmp_path: Path) -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)
    mother = _make_parent(tmp_path, "mother_a", "macro_partitioning")
    father = _make_parent(tmp_path, "father_b", "staged_routing_repair")

    _, user_prompt = _build_crossbreed_prompt(
        inherited_genes=[],
        mother=mother,
        father=father,
        prompts=bundle,
    )

    assert "=== PRIMARY PARENT GENETIC CODE ===" in user_prompt
    assert "=== SECONDARY PARENT GENETIC CODE ===" in user_prompt
    assert "=== NOVELTY REJECTION FEEDBACK ===" in user_prompt
    assert "CHILD GENETIC CODE DRAFT" not in user_prompt
    assert "{inherited_gene_pool}" not in user_prompt
    assert "IMPLEMENTATION CODE" not in user_prompt


def test_awtf2025_mutation_prompt_renders_novelty_feedback(tmp_path: Path) -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)
    parent = _make_parent(tmp_path, "parent_feedback", "macro_partitioning")

    _, user_prompt = _build_mutate_prompt(
        inherited_genes=[],
        removed_genes=[],
        parent=parent,
        prompts=bundle,
        novelty_feedback=["The previous child only paraphrased the parent staging idea."],
    )

    assert "- The previous child only paraphrased the parent staging idea." in user_prompt


def test_awtf2025_implementation_prompt_includes_mechanic_audit_step() -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)

    # Task-4 audit hint must appear in the implementation system prompt and must
    # explicitly ask the LLM to walk CONSTRUCTION_POLICY and LOCAL_REPAIR_POLICY
    # mechanics and confirm an executable code path for each.
    impl_system = bundle.implementation_system
    assert "Mechanic-to-code audit" in impl_system
    assert "CONSTRUCTION_POLICY" in impl_system
    assert "LOCAL_REPAIR_POLICY" in impl_system


def test_awtf2025_prompts_avoid_solution_leading_lists() -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)

    banned = ("all robots in one group", "fixed spiral script", "single hardcoded output")
    for token in banned:
        assert token not in bundle.seed_system.lower()
        assert token not in bundle.mutation_system.lower()
        assert token not in bundle.crossover_system.lower()
