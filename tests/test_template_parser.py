"""Tests for optimizer template parser."""

from __future__ import annotations

from pathlib import Path

from src.evolve.template_parser import (
    extract_editable_sections,
    parse_llm_response,
    render_template,
    validate_rendered_code,
)

ROOT = Path(__file__).resolve().parents[1]


def _template_text() -> str:
    return (ROOT / "conf" / "prompts" / "implementation" / "template.txt").read_text(encoding="utf-8")


def test_render_and_extract_roundtrip() -> None:
    sections = {
        "IMPORTS": "import math",
        "INIT_BODY": "        self.model = model\n        self.max_steps = max_steps",
        "STEP_BODY": "        del weights, grads, activations, step_fn",
        "ZERO_GRAD_BODY": "        pass",
    }
    code = render_template(sections, optimizer_name="TestOpt", class_name="TestOpt")
    extracted = extract_editable_sections(code)

    assert "IMPORTS" in extracted
    assert "INIT_BODY" in extracted
    assert "STEP_BODY" in extracted
    assert "ZERO_GRAD_BODY" in extracted
    assert "import math" in extracted["IMPORTS"]
    assert "self.model = model" in extracted["INIT_BODY"]


def test_validate_rendered_code_valid() -> None:
    sections = {
        "IMPORTS": "import math",
        "INIT_BODY": "        self.model = model",
        "STEP_BODY": "        pass",
        "ZERO_GRAD_BODY": "        pass",
    }
    code = render_template(sections, optimizer_name="ValidOpt", class_name="ValidOpt")
    ok, error = validate_rendered_code(code)
    assert ok, f"Validation failed: {error}"


def test_validate_rendered_code_syntax_error() -> None:
    ok, error = validate_rendered_code("def foo(\n")
    assert not ok
    assert "Syntax error" in error


def test_validate_rendered_code_rejects_invalid_builder_signature() -> None:
    code = """
class InvalidBuilderOpt:
    def __init__(self, cfg):
        self.cfg = cfg

    def step(self, weights, grads, activations, step_fn):
        del weights, grads, activations, step_fn

    def zero_grad(self, set_to_none=True):
        del set_to_none


def build_optimizer(cfg):
    return InvalidBuilderOpt(cfg)
"""
    ok, error = validate_rendered_code(code)
    assert not ok
    assert "build_optimizer" in str(error)


def test_validate_rendered_code_rejects_missing_step_fn() -> None:
    code = """
class InvalidOpt:
    def __init__(self, model, max_steps):
        self.model = model
        self.max_steps = max_steps

    def step(self, weights, grads, activations):
        del weights, grads, activations

    def zero_grad(self, set_to_none=True):
        del set_to_none


def build_optimizer(model, max_steps):
    return InvalidOpt(model, max_steps)
"""
    ok, error = validate_rendered_code(code)
    assert not ok
    assert "step(self, weights, grads, activations, step_fn)" in str(error)


def test_parse_llm_response_sections() -> None:
    text = (
        "## CORE_GENES\n"
        "- momentum schedule\n"
        "- warmup controller\n"
        "- gradient clipping\n"
        "\n"
        "## INTERACTION_NOTES\n"
        "Momentum and warmup stay synchronized.\n"
        "\n"
        "## COMPUTE_NOTES\n"
        "No extra closure calls.\n"
        "\n"
        "## CHANGE_DESCRIPTION\n"
        "Added momentum with warmup phase.\n"
    )
    parsed = parse_llm_response(text)

    assert "momentum schedule" in parsed["CORE_GENES"]
    assert "Momentum and warmup" in parsed["INTERACTION_NOTES"]
    assert "No extra closure calls" in parsed["COMPUTE_NOTES"]
    assert "momentum" in parsed["CHANGE_DESCRIPTION"]


def test_validate_rendered_code_rejects_markdown_fences_with_template() -> None:
    ok, error = validate_rendered_code("```python\nprint('x')\n```", _template_text())
    assert not ok
    assert "code fences" in str(error)


def test_validate_rendered_code_rejects_missing_scaffold_markers() -> None:
    code = (
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Bad:\n"
        "    def __init__(self, model, max_steps):\n"
        "        self.model = model\n"
        "        self.max_steps = max_steps\n"
        "    def step(self, weights, grads, activations, step_fn):\n"
        "        del weights, grads, activations, step_fn\n"
        "    def zero_grad(self, set_to_none=True):\n"
        "        del set_to_none\n\n"
        "def build_optimizer(model, max_steps):\n"
        "    return Bad(model, max_steps)\n"
    )
    ok, error = validate_rendered_code(code, _template_text())
    assert not ok
    assert "editable section markers" in str(error) or "template scaffold" in str(error)
