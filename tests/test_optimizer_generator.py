"""Unit tests for optimizer generator code validation."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

import src.evolve.operators as canonical_seed_operators
from src.evolve.generator import OptimizerGenerator
from src.evolve.types import Island


def _cfg():
    return OmegaConf.create(
        {
            "evolver": {
                "prompts": {
                    "project_context": "conf/prompts/shared/project_context.txt",
                    "seed_system": "conf/prompts/seed/system.txt",
                    "seed_user": "conf/prompts/seed/user.txt",
                    "mutation_system": "conf/prompts/mutation/system.txt",
                    "mutation_user": "conf/prompts/mutation/user.txt",
                    "crossover_system": "conf/prompts/crossover/system.txt",
                    "crossover_user": "conf/prompts/crossover/user.txt",
                },
                "max_generation_attempts": 2,
                "llm": {
                    "provider": "mock",
                    "model": "mock-model",
                    "temperature": 0.0,
                    "max_output_tokens": 512,
                    "reasoning_effort": None,
                    "seed": 123,
                    "fallback_to_chat_completions": True,
                },
            }
        }
    )


def test_validate_code_success() -> None:
    generator = OptimizerGenerator(_cfg())
    code = """
import torch.nn as nn

import torch

class DemoController:
    def __init__(self, model: nn.Module, max_steps: int):
        self.model = model
        self.max_steps = max_steps

    def step(self, weights, grads, activations, step_fn):
        del weights, grads, activations, step_fn

    def zero_grad(self, set_to_none=True):
        del set_to_none


def build_optimizer(model: nn.Module, max_steps: int):
    return DemoController(model, max_steps)
"""
    ok, error = generator._validate_code(code)
    assert ok
    assert error is None


def test_validate_code_syntax_error() -> None:
    generator = OptimizerGenerator(_cfg())
    code = "def build_optimizer(model, max_steps):\n    return [\n"
    ok, error = generator._validate_code(code)
    assert not ok
    assert error is not None


def test_validate_code_missing_builder() -> None:
    generator = OptimizerGenerator(_cfg())
    code = """
class SomethingElse:
    def __init__(self, model, max_steps):
        pass

    def step(self, weights, grads, activations, step_fn):
        pass

    def zero_grad(self, set_to_none=True):
        pass
"""
    ok, error = generator._validate_code(code)
    assert not ok
    assert error is not None


def test_validate_code_missing_controller_methods() -> None:
    generator = OptimizerGenerator(_cfg())
    code = """
class BadController:
    def __init__(self, model, max_steps):
        self.model = model
        self.max_steps = max_steps


def build_optimizer(model, max_steps):
    return BadController(model, max_steps)
"""
    ok, error = generator._validate_code(code)
    assert not ok
    assert error is not None


def test_validate_code_rejects_invalid_builder_signature() -> None:
    generator = OptimizerGenerator(_cfg())
    code = """
class InvalidBuilderController:
    def __init__(self, cfg):
        self.cfg = cfg

    def step(self, weights, grads, activations, step_fn):
        del weights, grads, activations, step_fn

    def zero_grad(self, set_to_none=True):
        del set_to_none


def build_optimizer(cfg):
    return InvalidBuilderController(cfg)
"""
    ok, error = generator._validate_code(code)
    assert not ok
    assert "build_optimizer" in str(error)


def test_validate_code_requires_step_fn_argument() -> None:
    generator = OptimizerGenerator(_cfg())
    code = """
class MissingStepFnController:
    def __init__(self, model, max_steps):
        self.model = model
        self.max_steps = max_steps

    def step(self, weights, grads, activations):
        del weights, grads, activations

    def zero_grad(self, set_to_none=True):
        del set_to_none


def build_optimizer(model, max_steps):
    return MissingStepFnController(model, max_steps)
"""
    ok, error = generator._validate_code(code)
    assert not ok
    assert "step(self, weights, grads, activations, step_fn)" in str(error)


def test_generator_loads_only_current_prompt_bundle_assets(monkeypatch) -> None:
    original_read_text = Path.read_text
    seen_paths: list[Path] = []

    def guarded_read_text(self: Path, *args, **kwargs):  # type: ignore[override]
        seen_paths.append(self)
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    generator = OptimizerGenerator(_cfg())

    observed = {
        tuple(path.parts[-2:])
        for path in seen_paths
        if len(path.parts) >= 2 and "prompts" in path.parts
    }
    assert observed == {
        ("shared", "project_context.txt"),
        ("seed", "system.txt"),
        ("seed", "user.txt"),
        ("mutation", "system.txt"),
        ("mutation", "user.txt"),
        ("crossover", "system.txt"),
        ("crossover", "user.txt"),
    }
    assert "## CORE_GENES" in generator.prompt_bundle.seed_system


def test_canonical_generator_seeds_real_island_organism(tmp_path: Path) -> None:
    generator = OptimizerGenerator(_cfg())
    island = Island(
        island_id="gradient_methods",
        name="gradient methods",
        description_path=str(tmp_path / "gradient_methods.txt"),
        description_text="First-order optimization heuristics.",
    )
    organism_dir = tmp_path / "org_seed"
    organism_dir.mkdir(parents=True, exist_ok=True)

    organism = generator.generate_seed_organism(
        island=island,
        organism_id="seed01",
        generation=0,
        organism_dir=organism_dir,
    )

    assert organism.operator == "seed"
    assert organism.island_id == "gradient_methods"
    assert (organism_dir / "optimizer.py").exists()
    assert (organism_dir / "genetic_code.md").exists()
    assert (organism_dir / "lineage.json").exists()


def test_canonical_seed_operator_module_excludes_prompt_only_mutation_and_crossover() -> None:
    assert hasattr(canonical_seed_operators, "SeedOperator")
    assert not hasattr(canonical_seed_operators, "MutationOperator")
    assert not hasattr(canonical_seed_operators, "CrossoverOperator")
