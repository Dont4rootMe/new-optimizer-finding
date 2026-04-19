"""Generated prompt context tests for circle_packing_shinka typed prompts."""

from __future__ import annotations

from pathlib import Path

from experiments.circle_packing_shinka._runtime import genome_schema_v1 as schema
from experiments.circle_packing_shinka._runtime import module_library_v1 as library
from experiments.circle_packing_shinka._runtime import prompt_context_v1 as context

ROOT = Path(__file__).resolve().parents[1]


def test_generated_prompt_context_is_deterministic() -> None:
    assert context.build_typed_hypothesis_prompt_context() == context.build_typed_hypothesis_prompt_context()
    assert context.build_seed_prompt_context() == context.build_seed_prompt_context()


def test_generated_prompt_context_includes_exact_slot_order() -> None:
    rendered = context.build_seed_prompt_context()
    for idx, slot in enumerate(schema.SLOT_ORDER, start=1):
        assert f"{idx}. {slot}" in rendered


def test_generated_prompt_context_includes_exact_allowed_module_keys() -> None:
    rendered = context.build_typed_hypothesis_prompt_context()
    for slot, module_keys in library.REQUIRED_MODULE_KEYS.items():
        assert f"{slot}:" in rendered
        for module_key in module_keys:
            assert module_key in rendered


def test_generated_prompt_context_includes_parameter_options() -> None:
    rendered = context.build_seed_prompt_context()

    assert "orientation_bias:categorical_token" in rendered
    assert "balanced" in rendered
    assert "fallback_shrink_enabled:boolean" in rendered


def test_generated_prompt_context_includes_compatibility_reminders() -> None:
    rendered = context.build_mutation_prompt_context()

    assert "HC-IF-001" in rendered
    assert "HC-LG-003" in rendered
    assert "FC-001" in rendered
    assert "functional_checks.json" in rendered


def test_prompt_files_do_not_hand_duplicate_library_tables() -> None:
    prompt_root = ROOT / "conf" / "experiments" / "circle_packing_shinka" / "prompts"
    static_prompt_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            prompt_root / "seed" / "user.txt",
            prompt_root / "mutation" / "user.txt",
            prompt_root / "crossover" / "user.txt",
            prompt_root / "novelty" / "mutation" / "user.txt",
            prompt_root / "novelty" / "crossover" / "user.txt",
        ]
    )

    assert "layout_triangular_lattice" not in static_prompt_text
    assert "growth_role_conditioned_growth" not in static_prompt_text
    assert "{typed_prompt_context}" in static_prompt_text


def test_prompt_files_contain_json_only_style_rules() -> None:
    prompt_root = ROOT / "conf" / "experiments" / "circle_packing_shinka" / "prompts"
    for path in [
        prompt_root / "seed" / "system.txt",
        prompt_root / "mutation" / "system.txt",
        prompt_root / "crossover" / "system.txt",
    ]:
        text = path.read_text(encoding="utf-8")
        assert "Return only valid JSON." in text
        assert "Do not use markdown fences." in text
        assert "Do not add commentary before or after the JSON object." in text
        assert "The slot assignment is the primary artifact." in text
        assert "Use only provided module keys." in text
        assert "Use only provided parameter names and allowed values." in text
        assert "Do not invent a new module." in text
        assert "Those human-readable fields will be rendered by the pipeline after validation." in text

