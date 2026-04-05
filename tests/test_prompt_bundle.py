"""Tests for config-driven prompt bundle loading."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from src.evolve.prompt_utils import compose_system_prompt, load_prompt_bundle

ROOT = Path(__file__).resolve().parents[1]


def test_load_prompt_bundle_from_default_conf_assets() -> None:
    cfg = OmegaConf.create(
        {
            "evolver": {
                "prompts": {
                    "project_context": "conf/experiments/optimization_survey/prompts/shared/project_context.txt",
                    "seed_system": "conf/experiments/optimization_survey/prompts/seed/system.txt",
                    "seed_user": "conf/experiments/optimization_survey/prompts/seed/user.txt",
                    "mutation_system": "conf/experiments/optimization_survey/prompts/mutation/system.txt",
                    "mutation_user": "conf/experiments/optimization_survey/prompts/mutation/user.txt",
                    "crossover_system": "conf/experiments/optimization_survey/prompts/crossover/system.txt",
                    "crossover_user": "conf/experiments/optimization_survey/prompts/crossover/user.txt",
                    "implementation_system": "conf/experiments/optimization_survey/prompts/implementation/system.txt",
                    "implementation_user": "conf/experiments/optimization_survey/prompts/implementation/user.txt",
                    "implementation_template": "conf/experiments/optimization_survey/prompts/implementation/template.txt",
                }
            }
        }
    )

    bundle = load_prompt_bundle(cfg)

    assert "automated evolutionary search for novel optimizers" in bundle.project_context
    assert "## CORE_GENES" in bundle.seed_system
    assert "{island_description}" in bundle.seed_user
    assert "Return only the final `implementation.py` text." in bundle.implementation_user


def test_load_prompt_bundle_from_explicit_paths(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    files = {
        "shared/project_context.txt": "project context",
        "seed/system.txt": "seed system",
        "seed/user.txt": "seed user",
        "mutation/system.txt": "mutation system",
        "mutation/user.txt": "mutation user",
        "crossover/system.txt": "crossover system",
        "crossover/user.txt": "crossover user",
        "implementation/system.txt": "implementation system",
        "implementation/user.txt": "implementation user",
        "implementation/template.txt": "implementation template",
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
                    "crossover_system": str(prompts_dir / "crossover" / "system.txt"),
                    "crossover_user": str(prompts_dir / "crossover" / "user.txt"),
                    "implementation_system": str(prompts_dir / "implementation" / "system.txt"),
                    "implementation_user": str(prompts_dir / "implementation" / "user.txt"),
                    "implementation_template": str(prompts_dir / "implementation" / "template.txt"),
                }
            }
        }
    )

    bundle = load_prompt_bundle(cfg)

    assert bundle.project_context == "project context"
    assert bundle.crossover_user == "crossover user"
    assert bundle.implementation_template == "implementation template"
    assert compose_system_prompt(bundle.project_context, bundle.seed_system) == "project context\n\nseed system"
