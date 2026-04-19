"""Canonical typed segmented genome schema for circle_packing_shinka v1."""

from __future__ import annotations

import re
from typing import Any


SCHEMA_NAME = "typed_segmented_genome"
SCHEMA_VERSION = "1.0"
TASK_FAMILY = "circle_packing_shinka"
TASK_NAME = "unit_square_26"
REPRESENTATION = "typed_segmented_hypothesis"

SLOT_ORDER = [
    "layout",
    "selection",
    "radius_init",
    "growth",
    "conflict",
    "repair",
    "boundary",
    "termination",
]
SLOT_ENUM = tuple(SLOT_ORDER)

SLOT_TO_MODULE_TYPE = {
    "layout": "layout_generator",
    "selection": "placement_order_policy",
    "radius_init": "initial_radius_policy",
    "growth": "radius_growth_policy",
    "conflict": "conflict_definition_policy",
    "repair": "feasibility_repair_policy",
    "boundary": "boundary_handling_policy",
    "termination": "stopping_policy",
}
MODULE_TYPE_ENUM = tuple(SLOT_TO_MODULE_TYPE.values())

LINKAGE_GROUPS = {
    "placement_block": ["layout", "selection", "radius_init"],
    "dynamics_block": ["growth", "conflict", "repair", "boundary"],
    "control_block": ["termination"],
}
LINKAGE_GROUP_ENUM = tuple(LINKAGE_GROUPS.keys())

STATE_VOCAB = {
    "candidate_centers",
    "ordered_centers",
    "current_radii",
    "proposed_radii",
    "conflict_report",
    "boundary_report",
    "packing_summary",
    "feasibility_status",
    "termination_signal",
}

SLOT_INTERFACE_REQUIREMENTS = {
    "layout": {
        "reads_any": [],
        "writes": ["candidate_centers"],
    },
    "selection": {
        "reads": ["candidate_centers"],
        "writes": ["ordered_centers"],
    },
    "radius_init": {
        "reads": ["ordered_centers"],
        "writes": ["current_radii"],
    },
    "growth": {
        "reads": ["current_radii"],
        "writes": ["proposed_radii"],
    },
    "conflict": {
        "reads": ["proposed_radii"],
        "writes": ["conflict_report"],
    },
    "repair": {
        "reads": ["conflict_report"],
        "writes": ["feasibility_status"],
    },
    "boundary": {
        "reads": ["proposed_radii"],
        "writes": ["boundary_report"],
    },
    "termination": {
        "reads_any": ["packing_summary", "feasibility_status", "boundary_report"],
        "writes": ["termination_signal"],
    },
}

VALUE_KIND_ENUM = {
    "categorical_token",
    "ordinal_token",
    "boolean",
    "relation_token",
}

GLOBAL_HYPOTHESIS_KEYS = {
    "title",
    "core_claim",
    "expected_advantage",
    "novelty_statement",
}
RENDER_FIELDS_KEYS = {
    "interaction_notes",
    "compute_notes",
    "change_description",
}
TOP_LEVEL_KEYS = {
    "schema_name",
    "schema_version",
    "task_family",
    "task_name",
    "representation",
    "organism_id",
    "slot_order",
    "linkage_groups",
    "global_hypothesis",
    "slots",
    "render_fields",
}
MODULE_KEYS = {
    "module_id",
    "slot",
    "module_type",
    "family",
    "hypothesis",
    "reads_state",
    "writes_state",
    "parameterization",
    "preconditions",
    "postconditions",
    "invariants",
    "assumptions",
    "expected_effects",
    "failure_modes",
    "compatibility_tags",
    "linkage_group",
}
PARAMETER_KEYS = {"name", "value_kind", "value"}

TEXT_DISCIPLINE_MODULE_LIST_FIELDS = {
    "assumptions",
    "expected_effects",
    "failure_modes",
}

_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_TOKEN_RE = re.compile(r"^[a-z][a-z_]*$")


def canonical_linkage_group_list() -> list[dict[str, list[str] | str]]:
    return [
        {"group_id": group_id, "slots": list(slots)}
        for group_id, slots in LINKAGE_GROUPS.items()
    ]


def linkage_group_for_slot(slot: str) -> str:
    for group_id, slots in LINKAGE_GROUPS.items():
        if slot in slots:
            return group_id
    raise ValueError(f"Unknown slot {slot!r}.")


def _fail(path: str, message: str) -> None:
    raise ValueError(f"{path}: {message}")


def _ensure_exact_keys(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    keys = set(value.keys())
    missing = sorted(expected.difference(keys))
    extra = sorted(keys.difference(expected))
    if missing:
        _fail(path, f"missing keys: {', '.join(missing)}")
    if extra:
        _fail(path, f"extra keys: {', '.join(extra)}")
    return value


def _ensure_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        _fail(path, "must be a string")
    if value.strip() != value or not value:
        _fail(path, "must be a non-empty stripped string")
    if "\n" in value or "\r" in value:
        _fail(path, "must be a single-line string")
    return value


def _ensure_string_list(value: Any, path: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        _fail(path, "must be a list")
    if not allow_empty and not value:
        _fail(path, "must be a non-empty list")
    out: list[str] = []
    for idx, item in enumerate(value):
        out.append(_ensure_str(item, f"{path}[{idx}]"))
    return out


def _check_text_discipline(text: str, path: str) -> None:
    lowered = text.lower()
    if "`" in text:
        _fail(path, "must not contain backticks or fenced code blocks")
    for banned in ("numpy", "def ", "import "):
        if banned in lowered:
            _fail(path, f"must not contain {banned!r}")


def _reject_numeric_json_values(value: Any, path: str = "$") -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        _fail(path, "numeric JSON values are forbidden in genome v1")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_numeric_json_values(child, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _reject_numeric_json_values(child, f"{path}[{idx}]")


def _reject_null_values(value: Any, path: str = "$") -> None:
    if value is None:
        _fail(path, "null values are forbidden in genome v1")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_null_values(child, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _reject_null_values(child, f"{path}[{idx}]")


def _validate_linkage_groups(value: Any) -> None:
    if value != canonical_linkage_group_list():
        _fail("linkage_groups", f"must be exactly {canonical_linkage_group_list()!r}")


def _validate_global_hypothesis(value: Any) -> None:
    payload = _ensure_exact_keys(value, GLOBAL_HYPOTHESIS_KEYS, "global_hypothesis")
    for key in sorted(GLOBAL_HYPOTHESIS_KEYS):
        text = _ensure_str(payload[key], f"global_hypothesis.{key}")
        _check_text_discipline(text, f"global_hypothesis.{key}")


def _validate_render_fields(value: Any) -> None:
    payload = _ensure_exact_keys(value, RENDER_FIELDS_KEYS, "render_fields")
    for key in ("interaction_notes", "compute_notes"):
        notes = _ensure_string_list(payload[key], f"render_fields.{key}")
        for idx, note in enumerate(notes):
            _check_text_discipline(note, f"render_fields.{key}[{idx}]")
    change_description = _ensure_str(payload["change_description"], "render_fields.change_description")
    _check_text_discipline(change_description, "render_fields.change_description")


def _validate_state_list(values: list[str], path: str) -> None:
    for value in values:
        if value not in STATE_VOCAB:
            _fail(path, f"illegal state symbol {value!r}")


def _validate_parameterization(value: Any, path: str) -> None:
    if not isinstance(value, list):
        _fail(path, "must be a list")
    seen_names: set[str] = set()
    for idx, item in enumerate(value):
        item_path = f"{path}[{idx}]"
        payload = _ensure_exact_keys(item, PARAMETER_KEYS, item_path)
        name = _ensure_str(payload["name"], f"{item_path}.name")
        if _SNAKE_CASE_RE.fullmatch(name) is None:
            _fail(f"{item_path}.name", "must be snake_case")
        if name in seen_names:
            _fail(f"{item_path}.name", f"duplicate parameter name {name!r}")
        seen_names.add(name)

        value_kind = _ensure_str(payload["value_kind"], f"{item_path}.value_kind")
        if value_kind not in VALUE_KIND_ENUM:
            _fail(f"{item_path}.value_kind", f"must be one of {sorted(VALUE_KIND_ENUM)!r}")

        parameter_value = payload["value"]
        if value_kind == "boolean":
            if not isinstance(parameter_value, bool):
                _fail(f"{item_path}.value", "must be a JSON boolean when value_kind is boolean")
        else:
            token = _ensure_str(parameter_value, f"{item_path}.value")
            if _TOKEN_RE.fullmatch(token) is None:
                _fail(f"{item_path}.value", "must be a lowercase non-numeric token")


def _validate_interface(module: dict[str, Any], slot: str, path: str) -> None:
    reads = module["reads_state"]
    writes = module["writes_state"]
    requirements = SLOT_INTERFACE_REQUIREMENTS[slot]

    for required in requirements.get("reads", []):
        if required not in reads:
            _fail(path, f"slot {slot!r} must read {required!r}")
    for required in requirements.get("writes", []):
        if required not in writes:
            _fail(path, f"slot {slot!r} must write {required!r}")
    reads_any = requirements.get("reads_any", [])
    if reads_any and not any(required in reads for required in reads_any):
        _fail(path, f"slot {slot!r} must read at least one of {reads_any!r}")


def _validate_module(value: Any, slot: str, seen_module_ids: set[str]) -> None:
    path = f"slots.{slot}"
    module = _ensure_exact_keys(value, MODULE_KEYS, path)

    module_id = _ensure_str(module["module_id"], f"{path}.module_id")
    if module_id in seen_module_ids:
        _fail(f"{path}.module_id", f"duplicate module_id {module_id!r}")
    seen_module_ids.add(module_id)

    declared_slot = _ensure_str(module["slot"], f"{path}.slot")
    if declared_slot != slot:
        _fail(f"{path}.slot", f"must match enclosing slot {slot!r}")

    module_type = _ensure_str(module["module_type"], f"{path}.module_type")
    expected_module_type = SLOT_TO_MODULE_TYPE[slot]
    if module_type != expected_module_type:
        _fail(f"{path}.module_type", f"expected {expected_module_type!r}, got {module_type!r}")

    _ensure_str(module["family"], f"{path}.family")
    hypothesis = _ensure_str(module["hypothesis"], f"{path}.hypothesis")
    _check_text_discipline(hypothesis, f"{path}.hypothesis")

    reads_state = _ensure_string_list(module["reads_state"], f"{path}.reads_state")
    writes_state = _ensure_string_list(module["writes_state"], f"{path}.writes_state")
    _validate_state_list(reads_state, f"{path}.reads_state")
    _validate_state_list(writes_state, f"{path}.writes_state")

    _validate_parameterization(module["parameterization"], f"{path}.parameterization")

    for key in (
        "preconditions",
        "postconditions",
        "invariants",
        "assumptions",
        "expected_effects",
        "failure_modes",
        "compatibility_tags",
    ):
        values = _ensure_string_list(module[key], f"{path}.{key}")
        if key in TEXT_DISCIPLINE_MODULE_LIST_FIELDS:
            for idx, text in enumerate(values):
                _check_text_discipline(text, f"{path}.{key}[{idx}]")

    linkage_group = _ensure_str(module["linkage_group"], f"{path}.linkage_group")
    expected_linkage_group = linkage_group_for_slot(slot)
    if linkage_group != expected_linkage_group:
        _fail(
            f"{path}.linkage_group",
            f"expected {expected_linkage_group!r}, got {linkage_group!r}",
        )

    _validate_interface(module, slot, path)


def _validate_slots(value: Any) -> None:
    if not isinstance(value, dict):
        _fail("slots", "must be an object")
    keys = set(value.keys())
    expected = set(SLOT_ORDER)
    missing = sorted(expected.difference(keys))
    extra = sorted(keys.difference(expected))
    if missing:
        _fail("slots", f"missing slots: {', '.join(missing)}")
    if extra:
        _fail("slots", f"extra slots: {', '.join(extra)}")

    seen_module_ids: set[str] = set()
    for slot in SLOT_ORDER:
        _validate_module(value[slot], slot, seen_module_ids)


def validate_circle_packing_genome_v1(genome: dict) -> None:
    """Validate one circle_packing_shinka typed segmented genome v1 payload."""

    if not isinstance(genome, dict):
        raise ValueError("genome must be a JSON object")

    _ensure_exact_keys(genome, TOP_LEVEL_KEYS, "$")
    _reject_null_values(genome)
    _reject_numeric_json_values(genome)

    expected_top_level_values = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "task_family": TASK_FAMILY,
        "task_name": TASK_NAME,
        "representation": REPRESENTATION,
    }
    for key, expected in expected_top_level_values.items():
        if genome.get(key) != expected:
            _fail(key, f"expected {expected!r}, got {genome.get(key)!r}")

    organism_id = _ensure_str(genome.get("organism_id"), "organism_id")
    if not organism_id:
        _fail("organism_id", "must be non-empty")

    if genome.get("slot_order") != SLOT_ORDER:
        _fail("slot_order", f"must be exactly {SLOT_ORDER!r}")

    _validate_linkage_groups(genome.get("linkage_groups"))
    _validate_global_hypothesis(genome.get("global_hypothesis"))
    _validate_render_fields(genome.get("render_fields"))
    _validate_slots(genome.get("slots"))


validate_genome = validate_circle_packing_genome_v1


def _section_lines(parsed: dict[str, str], key: str, fallback: str) -> list[str]:
    raw = str(parsed.get(key, "")).strip()
    if not raw:
        return [fallback]

    values: list[str] = []
    for line in raw.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("- "):
            cleaned = cleaned[2:].strip()
        if cleaned:
            values.append(cleaned)
    return values or [fallback]


def _pick(values: list[str], index: int) -> str:
    if not values:
        raise ValueError("Cannot select from empty design values.")
    return values[min(index, len(values) - 1)]


def _slot_state(slot: str) -> tuple[list[str], list[str]]:
    if slot == "layout":
        return ["packing_summary"], ["candidate_centers"]
    if slot == "selection":
        return ["candidate_centers"], ["ordered_centers"]
    if slot == "radius_init":
        return ["ordered_centers"], ["current_radii"]
    if slot == "growth":
        return ["current_radii"], ["proposed_radii"]
    if slot == "conflict":
        return ["proposed_radii"], ["conflict_report"]
    if slot == "repair":
        return ["conflict_report"], ["feasibility_status"]
    if slot == "boundary":
        return ["proposed_radii"], ["boundary_report"]
    if slot == "termination":
        return ["packing_summary", "feasibility_status", "boundary_report"], ["termination_signal"]
    raise ValueError(f"Unknown slot {slot!r}.")


def _module_from_design(
    *,
    organism_id: str,
    slot: str,
    hypothesis: str,
) -> dict[str, Any]:
    reads_state, writes_state = _slot_state(slot)
    linkage_group = linkage_group_for_slot(slot)
    return {
        "module_id": f"{organism_id}_{slot}",
        "slot": slot,
        "module_type": SLOT_TO_MODULE_TYPE[slot],
        "family": f"{linkage_group}_family",
        "hypothesis": hypothesis,
        "reads_state": reads_state,
        "writes_state": writes_state,
        "parameterization": [],
        "preconditions": [f"The {slot} module receives the declared abstract state inputs."],
        "postconditions": [f"The {slot} module updates its declared abstract state outputs."],
        "invariants": ["The module preserves a hypothesis-level geometric intent rather than raw code details."],
        "assumptions": [f"The {slot} policy remains compatible with the surrounding circle-packing construction."],
        "expected_effects": [f"The {slot} policy contributes to a feasible and coherent packing hypothesis."],
        "failure_modes": [f"The {slot} policy may underperform when upstream assumptions conflict with feasibility constraints."],
        "compatibility_tags": [f"{slot}_interface"],
        "linkage_group": linkage_group,
    }


def build_genome_from_design_response(
    parsed: dict[str, str],
    *,
    organism_id: str,
    task_name: str = TASK_NAME,
) -> dict[str, Any]:
    """Adapt the existing structured design sections into the v1 typed genome."""

    if task_name != TASK_NAME:
        raise ValueError(f"Circle-packing genome v1 only supports task_name={TASK_NAME!r}.")

    core_genes = _section_lines(
        parsed,
        "CORE_GENES",
        "Use a coherent geometric construction for circle packing.",
    )
    interaction_notes = _section_lines(
        parsed,
        "INTERACTION_NOTES",
        "The segmented policies should remain mutually consistent.",
    )
    compute_notes = _section_lines(
        parsed,
        "COMPUTE_NOTES",
        "The realization should keep the construction deterministic and lightweight.",
    )
    change_description = str(parsed.get("CHANGE_DESCRIPTION", "")).strip()
    if not change_description:
        raise ValueError("Circle-packing genome adapter requires CHANGE_DESCRIPTION.")

    placement_hypothesis = _pick(core_genes, 0)
    dynamics_hypothesis = _pick(core_genes, 1)
    control_hypothesis = _pick(core_genes, 2)
    slot_hypotheses = {
        "layout": placement_hypothesis,
        "selection": placement_hypothesis,
        "radius_init": placement_hypothesis,
        "growth": dynamics_hypothesis,
        "conflict": dynamics_hypothesis,
        "repair": dynamics_hypothesis,
        "boundary": dynamics_hypothesis,
        "termination": control_hypothesis,
    }

    genome = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "task_family": TASK_FAMILY,
        "task_name": TASK_NAME,
        "representation": REPRESENTATION,
        "organism_id": organism_id,
        "slot_order": list(SLOT_ORDER),
        "linkage_groups": canonical_linkage_group_list(),
        "global_hypothesis": {
            "title": "Segmented circle packing hypothesis",
            "core_claim": placement_hypothesis,
            "expected_advantage": dynamics_hypothesis,
            "novelty_statement": change_description,
        },
        "slots": {
            slot: _module_from_design(
                organism_id=organism_id,
                slot=slot,
                hypothesis=slot_hypotheses[slot],
            )
            for slot in SLOT_ORDER
        },
        "render_fields": {
            "interaction_notes": interaction_notes,
            "compute_notes": compute_notes,
            "change_description": change_description,
        },
    }
    validate_circle_packing_genome_v1(genome)
    return genome
