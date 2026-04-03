"""Tests for optimizer template parser."""

from __future__ import annotations

from src.evolve.template_parser import (
    extract_editable_sections,
    parse_llm_response,
    render_template,
    validate_rendered_code,
)


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


def test_validate_rendered_code_rejects_legacy_builder_contract() -> None:
    code = """
class LegacyOpt:
    def __init__(self, cfg):
        self.cfg = cfg

    def step(self, weights, grads, activations, step_fn):
        del weights, grads, activations, step_fn

    def zero_grad(self, set_to_none=True):
        del set_to_none


def build_optimizer(cfg):
    return LegacyOpt(cfg)
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
        "\n"
        "## IMPORTS\n"
        "import math\n"
        "\n"
        "## INIT_BODY\n"
        "        self.model = model\n"
        "\n"
        "## STEP_BODY\n"
        "        pass\n"
        "\n"
        "## ZERO_GRAD_BODY\n"
        "        pass\n"
    )
    parsed = parse_llm_response(text)

    assert "momentum schedule" in parsed["CORE_GENES"]
    assert "Momentum and warmup" in parsed["INTERACTION_NOTES"]
    assert "No extra closure calls" in parsed["COMPUTE_NOTES"]
    assert "momentum" in parsed["CHANGE_DESCRIPTION"]
    assert "import math" in parsed["IMPORTS"]
    assert "self.model = model" in parsed["INIT_BODY"]
    assert "pass" in parsed["STEP_BODY"]
    assert "pass" in parsed["ZERO_GRAD_BODY"]
