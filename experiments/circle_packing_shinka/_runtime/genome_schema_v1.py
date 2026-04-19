"""Canonical typed segmented genome schema for circle_packing_shinka v1."""

from __future__ import annotations

from typing import Any

from experiments.circle_packing_shinka._runtime import module_library_v1 as module_library


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
    "module_key",
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
    "inheritance_unit",
}
COMPACT_SLOT_ASSIGNMENT_KEYS = {"module_key", "module_id", "parameterization"}


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

    reads_state = _ensure_string_list(module["reads_state"], f"{path}.reads_state")
    writes_state = _ensure_string_list(module["writes_state"], f"{path}.writes_state")
    _validate_state_list(reads_state, f"{path}.reads_state")
    _validate_state_list(writes_state, f"{path}.writes_state")

    try:
        module_library.validate_module_instance(module)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    module_type = _ensure_str(module["module_type"], f"{path}.module_type")
    if module_type != SLOT_TO_MODULE_TYPE[slot]:
        _fail(f"{path}.module_type", f"expected {SLOT_TO_MODULE_TYPE[slot]!r}, got {module_type!r}")
    if module["linkage_group"] != linkage_group_for_slot(slot):
        _fail(f"{path}.linkage_group", f"expected {linkage_group_for_slot(slot)!r}")
    if module["inheritance_unit"] not in module_library.INHERITANCE_UNITS:
        _fail(f"{path}.inheritance_unit", "must be slot or linkage_block")

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


def _default_parameterization(module_key: str) -> list[dict[str, Any]]:
    return module_library.default_parameterization_for_module(module_key)


def materialize_module_instance(
    *,
    slot: str,
    module_key: str,
    module_id: str,
    parameterization: list[dict],
) -> dict:
    """Materialize one library-backed circle-packing module instance."""

    return module_library.materialize_module_instance(
        slot=slot,
        module_key=module_key,
        module_id=module_id,
        parameterization=parameterization,
    )


def _validate_compact_slot_assignments(slot_assignments: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(slot_assignments, dict):
        _fail("slot_assignments", "must be an object")
    keys = set(slot_assignments.keys())
    expected = set(SLOT_ORDER)
    missing = sorted(expected.difference(keys))
    extra = sorted(keys.difference(expected))
    if missing:
        _fail("slot_assignments", f"missing slots: {', '.join(missing)}")
    if extra:
        _fail("slot_assignments", f"extra slots: {', '.join(extra)}")

    validated: dict[str, dict[str, Any]] = {}
    for slot in SLOT_ORDER:
        assignment = _ensure_exact_keys(
            slot_assignments[slot],
            COMPACT_SLOT_ASSIGNMENT_KEYS,
            f"slot_assignments.{slot}",
        )
        module_key = _ensure_str(assignment["module_key"], f"slot_assignments.{slot}.module_key")
        module_id = _ensure_str(assignment["module_id"], f"slot_assignments.{slot}.module_id")
        parameterization = assignment["parameterization"]
        if not isinstance(parameterization, list):
            _fail(f"slot_assignments.{slot}.parameterization", "must be a list")
        validated[slot] = {
            "module_key": module_key,
            "module_id": module_id,
            "parameterization": parameterization,
        }
    return validated


def materialize_circle_packing_genome_v1(
    *,
    organism_id: str,
    global_hypothesis: dict,
    slot_assignments: dict,
    render_fields: dict,
) -> dict:
    """Materialize a full canonical genome from compact slot assignments."""

    _ensure_str(organism_id, "organism_id")
    _validate_global_hypothesis(global_hypothesis)
    _validate_render_fields(render_fields)
    assignments = _validate_compact_slot_assignments(slot_assignments)
    slots = {
        slot: module_library.materialize_module_instance(
            slot=slot,
            module_key=assignments[slot]["module_key"],
            module_id=assignments[slot]["module_id"],
            parameterization=assignments[slot]["parameterization"],
        )
        for slot in SLOT_ORDER
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
        "global_hypothesis": dict(global_hypothesis),
        "slots": slots,
        "render_fields": dict(render_fields),
    }
    validate_circle_packing_genome_v1(genome)
    return genome


def validate_materialized_slot_modules_against_library(genome: dict) -> None:
    """Validate that every materialized slot module is an exact library instance."""

    slots = genome.get("slots")
    if not isinstance(slots, dict):
        _fail("slots", "must be an object")
    for slot in SLOT_ORDER:
        if slot not in slots:
            _fail("slots", f"missing slot {slot!r}")
        module_library.validate_module_instance(slots[slot])


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
    slot_assignments = {
        "layout": {
            "module_key": "layout_triangular_lattice",
            "module_id": f"{organism_id}_layout",
            "parameterization": _default_parameterization("layout_triangular_lattice"),
        },
        "selection": {
            "module_key": "selection_center_outward",
            "module_id": f"{organism_id}_selection",
            "parameterization": _default_parameterization("selection_center_outward"),
        },
        "radius_init": {
            "module_key": "radius_init_lattice_derived",
            "module_id": f"{organism_id}_radius_init",
            "parameterization": _default_parameterization("radius_init_lattice_derived"),
        },
        "growth": {
            "module_key": "growth_density_scaled_additive",
            "module_id": f"{organism_id}_growth",
            "parameterization": _default_parameterization("growth_density_scaled_additive"),
        },
        "conflict": {
            "module_key": "conflict_overlap_plus_boundary_penetration",
            "module_id": f"{organism_id}_conflict",
            "parameterization": _default_parameterization("conflict_overlap_plus_boundary_penetration"),
        },
        "repair": {
            "module_key": "repair_pairwise_repulsion",
            "module_id": f"{organism_id}_repair",
            "parameterization": _default_parameterization("repair_pairwise_repulsion"),
        },
        "boundary": {
            "module_key": "boundary_repulsive_margin",
            "module_id": f"{organism_id}_boundary",
            "parameterization": _default_parameterization("boundary_repulsive_margin"),
        },
        "termination": {
            "module_key": "termination_no_violation_and_no_gain",
            "module_id": f"{organism_id}_termination",
            "parameterization": _default_parameterization("termination_no_violation_and_no_gain"),
        },
    }
    return materialize_circle_packing_genome_v1(
        organism_id=organism_id,
        global_hypothesis={
            "title": "Segmented circle packing hypothesis",
            "core_claim": placement_hypothesis,
            "expected_advantage": dynamics_hypothesis,
            "novelty_statement": change_description,
        },
        slot_assignments=slot_assignments,
        render_fields={
            "interaction_notes": interaction_notes,
            "compute_notes": compute_notes,
            "change_description": change_description,
        },
    )
