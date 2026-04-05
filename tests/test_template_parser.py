"""Tests for generic structured response parsing."""

from __future__ import annotations

from pathlib import Path

from src.evolve.template_parser import parse_llm_response

ROOT = Path(__file__).resolve().parents[1]


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


def test_implementation_template_asset_exists() -> None:
    template_path = ROOT / "conf" / "experiments" / "optimization_survey" / "prompts" / "implementation" / "template.txt"
    assert template_path.exists()
    assert "{imports}" in template_path.read_text(encoding="utf-8")
