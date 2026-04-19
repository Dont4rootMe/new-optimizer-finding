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
                    "seed_system": "conf/experiments/optimization_survey/prompts/seed/system.txt",
                    "seed_user": "conf/experiments/optimization_survey/prompts/seed/user.txt",
                    "mutation_system": "conf/experiments/optimization_survey/prompts/mutation/system.txt",
                    "mutation_user": "conf/experiments/optimization_survey/prompts/mutation/user.txt",
                    "mutation_novelty_system": "conf/experiments/optimization_survey/prompts/novelty/mutation/system.txt",
                    "mutation_novelty_user": "conf/experiments/optimization_survey/prompts/novelty/mutation/user.txt",
                    "crossover_system": "conf/experiments/optimization_survey/prompts/crossover/system.txt",
                    "crossover_user": "conf/experiments/optimization_survey/prompts/crossover/user.txt",
                    "crossover_novelty_system": "conf/experiments/optimization_survey/prompts/novelty/crossover/system.txt",
                    "crossover_novelty_user": "conf/experiments/optimization_survey/prompts/novelty/crossover/user.txt",
                    "implementation_system": "conf/experiments/optimization_survey/prompts/implementation/system.txt",
                    "implementation_user": "conf/experiments/optimization_survey/prompts/implementation/user.txt",
                    "implementation_template": "conf/experiments/optimization_survey/prompts/implementation/template.txt",
                    "repair_system": "conf/experiments/optimization_survey/prompts/repair/system.txt",
                    "repair_user": "conf/experiments/optimization_survey/prompts/repair/user.txt",
                }
            }
        }
    )

    bundle = load_prompt_bundle(cfg)

    assert "automated evolutionary search for novel optimizers" in bundle.project_context
    assert "## CORE_GENES" in bundle.seed_system
    assert "## NOVELTY_VERDICT" in bundle.mutation_novelty_system
    assert "{island_description}" in bundle.seed_user
    assert "Return only the final `implementation.py` text." in bundle.implementation_user
    assert "=== ERROR HISTORY ===" in bundle.repair_user
    assert "already the evolved child draft" in bundle.mutation_user
    assert "faithful, coherent recombination is allowed" in bundle.crossover_user
    assert "valid source of novelty" in bundle.mutation_novelty_user
    assert "preserves substantial material from both parents" in bundle.crossover_novelty_user


def test_circle_packing_mutation_and_crossover_prompts_restate_structured_contract() -> None:
    cfg = OmegaConf.create(
        {
            "evolver": {
                "prompts": {
                    "project_context": "conf/experiments/circle_packing_shinka/prompts/shared/project_context.txt",
                    "seed_system": "conf/experiments/circle_packing_shinka/prompts/seed/system.txt",
                    "seed_user": "conf/experiments/circle_packing_shinka/prompts/seed/user.txt",
                    "mutation_system": "conf/experiments/circle_packing_shinka/prompts/mutation/system.txt",
                    "mutation_user": "conf/experiments/circle_packing_shinka/prompts/mutation/user.txt",
                    "mutation_novelty_system": "conf/experiments/circle_packing_shinka/prompts/novelty/mutation/system.txt",
                    "mutation_novelty_user": "conf/experiments/circle_packing_shinka/prompts/novelty/mutation/user.txt",
                    "crossover_system": "conf/experiments/circle_packing_shinka/prompts/crossover/system.txt",
                    "crossover_user": "conf/experiments/circle_packing_shinka/prompts/crossover/user.txt",
                    "crossover_novelty_system": "conf/experiments/circle_packing_shinka/prompts/novelty/crossover/system.txt",
                    "crossover_novelty_user": "conf/experiments/circle_packing_shinka/prompts/novelty/crossover/user.txt",
                    "implementation_system": "conf/experiments/circle_packing_shinka/prompts/implementation/system.txt",
                    "implementation_user": "conf/experiments/circle_packing_shinka/prompts/implementation/user.txt",
                    "implementation_template": "conf/experiments/circle_packing_shinka/prompts/implementation/template.txt",
                    "repair_system": "conf/experiments/circle_packing_shinka/prompts/repair/system.txt",
                    "repair_user": "conf/experiments/circle_packing_shinka/prompts/repair/user.txt",
                }
            }
        }
    )

    bundle = load_prompt_bundle(cfg)

    for prompt in (bundle.mutation_system, bundle.crossover_system):
        assert "Return only valid JSON." in prompt
        assert "The slot assignment is the primary artifact." in prompt
        assert "Use only provided module keys." in prompt
        assert "Do not emit `CORE_GENES`, `INTERACTION_NOTES`, `COMPUTE_NOTES`, or `CHANGE_DESCRIPTION`" in prompt
    assert "Return only valid JSON." in bundle.crossover_novelty_system
    assert "CURRENT IMPLEMENTATION.PY" in bundle.repair_user
    assert "=== CHILD SLOT DRAFT ===" in bundle.mutation_user
    assert "=== PRIMARY PARENT SLOT ASSIGNMENTS ===" in bundle.crossover_user
    assert "=== CANDIDATE CHILD SLOT ASSIGNMENTS ===" in bundle.mutation_novelty_user
    assert "=== SECONDARY PARENT SLOT ASSIGNMENTS ===" in bundle.crossover_novelty_user


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
    assert compose_system_prompt(bundle.project_context, bundle.seed_system) == "project context\n\nseed system"
