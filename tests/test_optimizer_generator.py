"""Unit tests for optimizer generator code validation."""

from __future__ import annotations

from omegaconf import OmegaConf

from src.evolve.generator import OptimizerGenerator


def _cfg():
    return OmegaConf.create(
        {
            "evolver": {
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
import torch

class DemoController:
    def __init__(self, cfg):
        self.cfg = cfg

    def initialize(self, named_parameters, cfg):
        self.params = [p for _, p in named_parameters]

    def step(self, weights, grads, activations):
        del weights, grads, activations

    def zero_grad(self, set_to_none=True):
        del set_to_none


def build_optimizer(cfg):
    return DemoController(cfg)
"""
    ok, error = generator._validate_code(code)
    assert ok
    assert error is None


def test_validate_code_syntax_error() -> None:
    generator = OptimizerGenerator(_cfg())
    code = "def build_optimizer(cfg):\n    return [\n"
    ok, error = generator._validate_code(code)
    assert not ok
    assert error is not None


def test_validate_code_missing_builder() -> None:
    generator = OptimizerGenerator(_cfg())
    code = """
class SomethingElse:
    def initialize(self, named_parameters, cfg):
        pass

    def step(self, weights, grads, activations):
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
    def __init__(self, cfg):
        self.cfg = cfg


def build_optimizer(cfg):
    return BadController(cfg)
"""
    ok, error = generator._validate_code(code)
    assert not ok
    assert error is not None


def test_mock_candidate_code_does_not_expect_framework_hparams() -> None:
    generator = OptimizerGenerator(_cfg())
    code = generator._mock_candidate_code(candidate_id="abcdef12", generation=0)

    assert "optimizer_kwargs" not in code
    assert "scheduler_cfg" not in code
    assert "max_steps" in code
