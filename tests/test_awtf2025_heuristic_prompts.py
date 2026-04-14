"""Prompt-level contract tests for awtf2025_heuristic."""

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


def test_awtf2025_prompt_bundle_uses_gene_centric_language() -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)

    assert "primary object is the organism's genetic code" in bundle.project_context
    assert "Do not invent new major ideas at implementation time." in bundle.implementation_system
    assert "child genetic code draft" in bundle.mutation_system.lower()
    assert "child draft" in bundle.crossover_system.lower()
    assert "keep it essentially intact" in bundle.mutation_system
    assert "valid source of novelty" in bundle.mutation_novelty_user
    assert "preserves substantial material from both parents" in bundle.crossover_novelty_user


def test_awtf2025_mutation_prompt_prioritizes_child_draft(tmp_path: Path) -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)
    parent = _make_parent(tmp_path, "parent_a", "macro_partitioning")

    _, user_prompt = _build_mutate_prompt(
        inherited_genes=[
            "Child idea about macro partition structure",
            "Child idea about synchronized group policy",
            "Child idea about congestion repair logic",
        ],
        removed_genes=["Excluded idea about older partitioning"],
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


def test_awtf2025_crossover_prompt_prioritizes_child_draft(tmp_path: Path) -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)
    mother = _make_parent(tmp_path, "mother_a", "macro_partitioning")
    father = _make_parent(tmp_path, "father_b", "staged_routing_repair")

    _, user_prompt = _build_crossbreed_prompt(
        inherited_genes=[
            "Merged child idea about board structure",
            "Merged child idea about group scheduling",
            "Merged child idea about targeted repair",
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


def test_awtf2025_mutation_prompt_renders_novelty_feedback(tmp_path: Path) -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)
    parent = _make_parent(tmp_path, "parent_feedback", "macro_partitioning")

    _, user_prompt = _build_mutate_prompt(
        inherited_genes=[
            "Child idea about macro partition structure",
            "Child idea about synchronized group policy",
            "Child idea about congestion repair logic",
        ],
        removed_genes=["Excluded idea about older partitioning"],
        parent=parent,
        prompts=bundle,
        novelty_feedback=["The previous child only paraphrased the parent staging idea."],
    )

    assert "- The previous child only paraphrased the parent staging idea." in user_prompt


def test_awtf2025_prompts_avoid_solution_leading_lists() -> None:
    cfg = _compose_cfg()
    bundle = load_prompt_bundle(cfg)
    root = ROOT / "conf" / "experiments" / "awtf2025_heuristic" / "prompts"

    macro_text = (root / "islands" / "macro_partitioning.txt").read_text(encoding="utf-8")
    repair_text = (root / "islands" / "staged_routing_repair.txt").read_text(encoding="utf-8")

    banned = ("all robots in one group", "fixed spiral script", "single hardcoded output")
    for token in banned:
        assert token not in bundle.seed_system.lower()
        assert token not in bundle.mutation_system.lower()
        assert token not in bundle.crossover_system.lower()
        assert token not in macro_text.lower()
        assert token not in repair_text.lower()
