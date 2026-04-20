"""Tests for config-driven prompt bundle loading."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from src.evolve.prompt_utils import compose_system_prompt, load_prompt_bundle

ROOT = Path(__file__).resolve().parents[1]


def test_load_prompt_bundle_from_optimization_survey_conf_assets() -> None:
    cfg = OmegaConf.create(
        {
            "evolver": {
                "prompts": {
                    "project_context": "conf/experiments/optimization_survey/prompts/shared/project_context.txt",
                    "genome_schema": "conf/experiments/optimization_survey/prompts/shared/genome_schema.txt",
                    "seed_system": "conf/experiments/optimization_survey/prompts/seed/system.txt",
                    "seed_user": "conf/experiments/optimization_survey/prompts/seed/user.txt",
                    "compatibility_seed_system": "conf/experiments/optimization_survey/prompts/compatibility/seed/system.txt",
                    "compatibility_seed_user": "conf/experiments/optimization_survey/prompts/compatibility/seed/user.txt",
                    "mutation_system": "conf/experiments/optimization_survey/prompts/mutation/system.txt",
                    "mutation_user": "conf/experiments/optimization_survey/prompts/mutation/user.txt",
                    "mutation_novelty_system": "conf/experiments/optimization_survey/prompts/novelty/mutation/system.txt",
                    "mutation_novelty_user": "conf/experiments/optimization_survey/prompts/novelty/mutation/user.txt",
                    "compatibility_mutation_system": "conf/experiments/optimization_survey/prompts/compatibility/mutation/system.txt",
                    "compatibility_mutation_user": "conf/experiments/optimization_survey/prompts/compatibility/mutation/user.txt",
                    "crossover_system": "conf/experiments/optimization_survey/prompts/crossover/system.txt",
                    "crossover_user": "conf/experiments/optimization_survey/prompts/crossover/user.txt",
                    "crossover_novelty_system": "conf/experiments/optimization_survey/prompts/novelty/crossover/system.txt",
                    "crossover_novelty_user": "conf/experiments/optimization_survey/prompts/novelty/crossover/user.txt",
                    "compatibility_crossover_system": "conf/experiments/optimization_survey/prompts/compatibility/crossover/system.txt",
                    "compatibility_crossover_user": "conf/experiments/optimization_survey/prompts/compatibility/crossover/user.txt",
                    "implementation_system": "conf/experiments/optimization_survey/prompts/implementation/system.txt",
                    "implementation_user": "conf/experiments/optimization_survey/prompts/implementation/user.txt",
                    "implementation_template": "conf/experiments/optimization_survey/prompts/shared/template.txt",
                    "repair_system": "conf/experiments/optimization_survey/prompts/repair/system.txt",
                    "repair_user": "conf/experiments/optimization_survey/prompts/repair/user.txt",
                }
            }
        }
    )

    bundle = load_prompt_bundle(cfg)

    assert "automated evolutionary search for novel optimizer organisms" in bundle.project_context
    assert "## CORE_GENES" in bundle.seed_system
    assert "## NOVELTY_VERDICT" in bundle.mutation_novelty_system
    assert "{island_description}" in bundle.seed_user
    assert "=== COMPILATION MODE ===" in bundle.implementation_user
    assert "## COMPILATION_MODE" in bundle.implementation_system
    assert "=== ERROR HISTORY ===" in bundle.repair_user
    assert "=== CANONICAL IMPLEMENTATION SCAFFOLD ===" in bundle.repair_user
    assert "Repair is a full-file rewrite, not a diff." in bundle.repair_system
    assert "Assign every local variable before first use" in bundle.repair_system
    assert "Execution-order discipline" in bundle.implementation_system
    assert "child-side draft already selected by evolution" in bundle.mutation_user
    assert "selected recombination is already coherent" in bundle.crossover_user
    assert "valid source of novelty" in bundle.mutation_novelty_user
    assert "preserves substantial material from both parents" in bundle.crossover_novelty_user
    assert "# STATE_REPRESENTATION" in bundle.genome_schema
    assert "## COMPATIBILITY_VERDICT" in bundle.compatibility_seed_system
    assert "compatibility is not the same as novelty" in bundle.compatibility_mutation_system
    assert "compatibility is not the same as novelty" in bundle.compatibility_crossover_system


def test_circle_packing_mutation_and_crossover_prompts_restate_structured_contract() -> None:
    cfg = OmegaConf.create(
        {
            "evolver": {
                "prompts": {
                    "project_context": "conf/experiments/circle_packing_shinka/prompts/shared/project_context.txt",
                    "genome_schema": "conf/experiments/circle_packing_shinka/prompts/shared/genome_schema.txt",
                    "seed_system": "conf/experiments/circle_packing_shinka/prompts/seed/system.txt",
                    "seed_user": "conf/experiments/circle_packing_shinka/prompts/seed/user.txt",
                    "compatibility_seed_system": "conf/experiments/circle_packing_shinka/prompts/compatibility/seed/system.txt",
                    "compatibility_seed_user": "conf/experiments/circle_packing_shinka/prompts/compatibility/seed/user.txt",
                    "mutation_system": "conf/experiments/circle_packing_shinka/prompts/mutation/system.txt",
                    "mutation_user": "conf/experiments/circle_packing_shinka/prompts/mutation/user.txt",
                    "mutation_novelty_system": "conf/experiments/circle_packing_shinka/prompts/novelty/mutation/system.txt",
                    "mutation_novelty_user": "conf/experiments/circle_packing_shinka/prompts/novelty/mutation/user.txt",
                    "compatibility_mutation_system": "conf/experiments/circle_packing_shinka/prompts/compatibility/mutation/system.txt",
                    "compatibility_mutation_user": "conf/experiments/circle_packing_shinka/prompts/compatibility/mutation/user.txt",
                    "crossover_system": "conf/experiments/circle_packing_shinka/prompts/crossover/system.txt",
                    "crossover_user": "conf/experiments/circle_packing_shinka/prompts/crossover/user.txt",
                    "crossover_novelty_system": "conf/experiments/circle_packing_shinka/prompts/novelty/crossover/system.txt",
                    "crossover_novelty_user": "conf/experiments/circle_packing_shinka/prompts/novelty/crossover/user.txt",
                    "compatibility_crossover_system": "conf/experiments/circle_packing_shinka/prompts/compatibility/crossover/system.txt",
                    "compatibility_crossover_user": "conf/experiments/circle_packing_shinka/prompts/compatibility/crossover/user.txt",
                    "implementation_system": "conf/experiments/circle_packing_shinka/prompts/implementation/system.txt",
                    "implementation_user": "conf/experiments/circle_packing_shinka/prompts/implementation/user.txt",
                    "implementation_template": "conf/experiments/circle_packing_shinka/prompts/shared/template.txt",
                    "repair_system": "conf/experiments/circle_packing_shinka/prompts/repair/system.txt",
                    "repair_user": "conf/experiments/circle_packing_shinka/prompts/repair/user.txt",
                }
            }
        }
    )

    bundle = load_prompt_bundle(cfg)

    for prompt in (bundle.mutation_system, bundle.crossover_system):
        assert "## Response format" in prompt
        assert "## CORE_GENES" in prompt
        assert "## CHANGE_DESCRIPTION" in prompt
    assert "## NOVELTY_VERDICT" in bundle.crossover_novelty_system
    assert "CURRENT IMPLEMENTATION.PY" in bundle.repair_user
    assert "=== CANONICAL IMPLEMENTATION SCAFFOLD ===" in bundle.repair_user
    assert "Region-preserving repair strategy" in bundle.repair_system
    assert "Execution-order discipline" in bundle.implementation_system
    assert "Validity preservation note" in bundle.implementation_user
    assert "smallest coherent module" in bundle.mutation_system
    assert "primary-parent-dominant organism" in bundle.crossover_user
    assert "valid source of novelty" in bundle.mutation_novelty_user
    assert "preserves substantial material from both parents" in bundle.crossover_novelty_user
    assert "## COMPATIBILITY_VERDICT" in bundle.compatibility_seed_system
    assert "validity beats score" not in bundle.implementation_system
    assert "feasibility safety pass" not in bundle.implementation_user
    assert "Repair is a full-file rewrite, not a diff." not in bundle.repair_system


def test_load_prompt_bundle_from_explicit_paths(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    files = {
        "shared/project_context.txt": "project context",
        "seed/system.txt": "seed system",
        "seed/user.txt": "seed user",
        "mutation/system.txt": "mutation system",
        "mutation/user.txt": "mutation user",
        "novelty/mutation/system.txt": "mutation novelty system",
        "novelty/mutation/user.txt": "mutation novelty user",
        "crossover/system.txt": "crossover system",
        "crossover/user.txt": "crossover user",
        "novelty/crossover/system.txt": "crossover novelty system",
        "novelty/crossover/user.txt": "crossover novelty user",
        "implementation/system.txt": "implementation system",
        "implementation/user.txt": "implementation user",
        "implementation/template.txt": "implementation template",
        "repair/system.txt": "repair system",
        "repair/user.txt": "repair user",
    }
    for relative_path, contents in files.items():
        target = prompts_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")

    cfg = OmegaConf.create(
        {
            "evolver": {
                "prompts": {
                    "project_context": str(prompts_dir / "shared" / "project_context.txt"),
                    "seed_system": str(prompts_dir / "seed" / "system.txt"),
                    "seed_user": str(prompts_dir / "seed" / "user.txt"),
                    "mutation_system": str(prompts_dir / "mutation" / "system.txt"),
                    "mutation_user": str(prompts_dir / "mutation" / "user.txt"),
                    "mutation_novelty_system": str(prompts_dir / "novelty" / "mutation" / "system.txt"),
                    "mutation_novelty_user": str(prompts_dir / "novelty" / "mutation" / "user.txt"),
                    "crossover_system": str(prompts_dir / "crossover" / "system.txt"),
                    "crossover_user": str(prompts_dir / "crossover" / "user.txt"),
                    "crossover_novelty_system": str(prompts_dir / "novelty" / "crossover" / "system.txt"),
                    "crossover_novelty_user": str(prompts_dir / "novelty" / "crossover" / "user.txt"),
                    "implementation_system": str(prompts_dir / "implementation" / "system.txt"),
                    "implementation_user": str(prompts_dir / "implementation" / "user.txt"),
                    "implementation_template": str(prompts_dir / "implementation" / "template.txt"),
                    "repair_system": str(prompts_dir / "repair" / "system.txt"),
                    "repair_user": str(prompts_dir / "repair" / "user.txt"),
                }
            }
        }
    )

    bundle = load_prompt_bundle(cfg)

    assert bundle.project_context == "project context"
    assert bundle.crossover_user == "crossover user"
    assert bundle.mutation_novelty_user == "mutation novelty user"
    assert bundle.implementation_template == "implementation template"
    assert bundle.repair_user == "repair user"
    assert bundle.genome_schema == ""
    assert bundle.compatibility_seed_system == ""
    assert bundle.compatibility_mutation_user == ""
    assert compose_system_prompt(bundle.project_context, bundle.seed_system) == "project context\n\nseed system"
