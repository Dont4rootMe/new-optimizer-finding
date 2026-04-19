"""Task-blind rendering for canonical hypothesis artifacts."""

from __future__ import annotations

from typing import Any


def _schema_validation_callable(schema_provider: Any):
    candidate = getattr(schema_provider, "validate_genome", None)
    if callable(candidate):
        return candidate

    for name in sorted(dir(schema_provider)):
        if name.startswith("validate_") and name.endswith("_genome_v1"):
            candidate = getattr(schema_provider, name)
            if callable(candidate):
                return candidate

    raise ValueError("Schema provider must expose validate_genome or a validate_*_genome_v1 function.")


def _dedupe_stable(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def render_genetic_code_markdown(genome: dict, schema_provider) -> str:
    """Render deterministic ``genetic_code.md`` from a validated canonical genome."""

    _schema_validation_callable(schema_provider)(genome)

    slot_order = list(getattr(schema_provider, "SLOT_ORDER"))
    slots = genome["slots"]
    render_fields = genome["render_fields"]

    core_lines = [
        f"- [{slot} | {slots[slot]['module_key']}] {slots[slot]['hypothesis']}"
        for slot in slot_order
    ]

    interaction_values: list[str] = []
    interaction_values.extend(render_fields["interaction_notes"])
    for slot in slot_order:
        interaction_values.extend(slots[slot]["assumptions"])
    for slot in slot_order:
        interaction_values.extend(slots[slot]["failure_modes"])
    interaction_lines = [f"- {value}" for value in _dedupe_stable(interaction_values)]

    compute_lines = [f"- {value}" for value in render_fields["compute_notes"]]

    return (
        "## CORE_GENES\n"
        + "\n".join(core_lines)
        + "\n\n"
        + "## INTERACTION_NOTES\n"
        + "\n".join(interaction_lines)
        + "\n\n"
        + "## COMPUTE_NOTES\n"
        + "\n".join(compute_lines)
        + "\n\n"
        + "## CHANGE_DESCRIPTION\n"
        + render_fields["change_description"]
        + "\n"
    )
