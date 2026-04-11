"""Unit tests for generic organism generator behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from api_platforms import LlmResponse
import src.evolve.operators as canonical_seed_operators
from src.evolve.generator import CandidateGenerator
from src.evolve.types import Island


def _cfg():
    return OmegaConf.create(
        {
            "seed": 123,
            "evolver": {
                "max_generation_attempts": 2,
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
                },
                "llm": {"route_weights": {"mock": 1.0}, "seed": 123},
            },
            "api_platforms": {"mock": {"_target_": "api_platforms.mock.platform.build_platform"}},
            "paths": {"api_platform_runtime_root": ".tmp_api_platform_runtime"},
        }
    )


def test_generator_loads_only_current_prompt_bundle_assets(monkeypatch) -> None:
    original_read_text = Path.read_text
    seen_paths: list[Path] = []

    def guarded_read_text(self: Path, *args, **kwargs):  # type: ignore[override]
        seen_paths.append(self)
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    generator = CandidateGenerator(_cfg())

    observed = {
        tuple(path.parts[-3:])
        for path in seen_paths
        if len(path.parts) >= 3 and "prompts" in path.parts
    }
    assert observed == {
        ("prompts", "shared", "project_context.txt"),
        ("prompts", "seed", "system.txt"),
        ("prompts", "seed", "user.txt"),
        ("prompts", "mutation", "system.txt"),
        ("prompts", "mutation", "user.txt"),
        ("prompts", "crossover", "system.txt"),
        ("prompts", "crossover", "user.txt"),
        ("prompts", "implementation", "system.txt"),
        ("prompts", "implementation", "user.txt"),
        ("prompts", "implementation", "template.txt"),
    }
    assert "## CORE_GENES" in generator.prompt_bundle.seed_system
    generator.close()


def test_canonical_generator_seeds_real_island_organism(tmp_path: Path) -> None:
    generator = CandidateGenerator(_cfg())
    island = Island(
        island_id="gradient_methods",
        name="gradient methods",
        description_path=str(tmp_path / "gradient_methods.txt"),
        description_text="First-order optimization heuristics.",
    )
    organism_dir = tmp_path / "org_seed"
    organism_dir.mkdir(parents=True, exist_ok=True)

    try:
        organism = generator.generate_seed_organism(
            island=island,
            organism_id="seed01",
            generation=0,
            organism_dir=organism_dir,
        )
    finally:
        generator.close()

    assert organism.operator == "seed"
    assert organism.island_id == "gradient_methods"
    assert organism.implementation_path == str(organism_dir / "implementation.py")
    assert (organism_dir / "implementation.py").exists()
    assert (organism_dir / "genetic_code.md").exists()
    assert (organism_dir / "lineage.json").exists()
    assert (organism_dir / "organism.json").exists()
    assert organism.llm_route_id == "mock"
    assert organism.llm_provider == "mock"
    assert organism.provider_model_id == "mock-model"
    organism_meta = json.loads((organism_dir / "organism.json").read_text(encoding="utf-8"))
    assert organism_meta["llm_route_id"] == "mock"
    assert organism_meta["llm_provider"] == "mock"
    assert organism_meta["provider_model_id"] == "mock-model"
    assert "design" in (organism_dir / "llm_request.json").read_text(encoding="utf-8")
    assert "implementation" in (organism_dir / "llm_request.json").read_text(encoding="utf-8")
    llm_request = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    llm_response = json.loads((organism_dir / "llm_response.json").read_text(encoding="utf-8"))
    assert llm_request["route_id"] == "mock"
    assert llm_request["design"]["route_id"] == "mock"
    assert llm_request["implementation"]["route_id"] == "mock"
    assert llm_response["route_id"] == "mock"
    assert llm_response["design"]["route_id"] == "mock"
    assert llm_response["implementation"]["route_id"] == "mock"


def test_canonical_seed_operator_module_excludes_prompt_only_mutation_and_crossover() -> None:
    assert hasattr(canonical_seed_operators, "SeedOperator")
    assert not hasattr(canonical_seed_operators, "MutationOperator")
    assert not hasattr(canonical_seed_operators, "CrossoverOperator")


def test_run_creation_stages_persists_design_exchange_before_parse_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    organism_dir = tmp_path / "org_partial"
    organism_dir.mkdir(parents=True, exist_ok=True)
    calls: list[str] = []

    def fake_generate(request):
        calls.append(request.stage)
        return LlmResponse(
            text=(
                "## CORE_GENES\n"
                "- valid enough gene one\n"
                "- valid enough gene two\n"
                "- valid enough gene three\n\n"
                "## INTERACTION_NOTES\n"
                "Partial design only.\n"
            ),
            route_id="mock",
            provider="mock",
            provider_model_id="mock-model",
            raw_request={"stage": request.stage},
            raw_response={"stage": request.stage},
            usage={},
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
        )

    monkeypatch.setattr(generator.registry, "generate", fake_generate)

    try:
        with pytest.raises(ValueError, match="COMPUTE_NOTES|CHANGE_DESCRIPTION"):
            generator.run_creation_stages(
                design_system_prompt="design system",
                design_user_prompt="design user",
                org_dir=organism_dir,
                organism_id="org_partial",
                generation=0,
            )
    finally:
        generator.close()

    assert calls == ["design"]
    llm_request = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    llm_response = json.loads((organism_dir / "llm_response.json").read_text(encoding="utf-8"))
    assert llm_request["design"]["request"] == {"stage": "design"}
    assert llm_response["design"]["response"] == {"stage": "design"}
    assert "implementation" not in llm_request
    assert "implementation" not in llm_response
