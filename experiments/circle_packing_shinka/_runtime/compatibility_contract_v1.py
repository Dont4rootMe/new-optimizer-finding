"""Compatibility contract for circle_packing_shinka genome/library v1."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from experiments.circle_packing_shinka._runtime import genome_schema_v1 as schema
from experiments.circle_packing_shinka._runtime import module_library_v1 as library


COMPATIBILITY_REPORT_SCHEMA_NAME = "circle_packing_compatibility_report"
FUNCTIONAL_CHECKS_SCHEMA_NAME = "circle_packing_functional_checks"
SCHEMA_VERSION = "1.0"
TASK_FAMILY = schema.TASK_FAMILY

SLOT_ORDER = tuple(schema.SLOT_ORDER)
SLOT_PAIRS_TO_SCORE = (
    ("layout", "radius_init"),
    ("growth", "conflict"),
    ("conflict", "repair"),
    ("repair", "boundary"),
    ("repair", "termination"),
)

STRUCTURED_PLACEMENT_LAYOUTS = {
    "layout_triangular_lattice",
    "layout_boundary_ring_interior_fill",
    "layout_reflective_symmetric_seed",
}
ROLE_SUPPORT_LAYOUTS = {
    "layout_boundary_ring_interior_fill",
    "layout_reflective_symmetric_seed",
}
ROLE_AWARE_MODULES = {
    "radius_init_role_based_boundary_vs_interior",
    "growth_role_conditioned_growth",
    "boundary_role_based_edge_protection",
}
ROLE_CONDITIONED_GROWTH_SUPPORT = ROLE_SUPPORT_LAYOUTS | {
    "radius_init_role_based_boundary_vs_interior",
    "boundary_role_based_edge_protection",
}
CONSTRAINT_GRAPH_REPAIRS = {
    "repair_sequential_worst_first",
    "repair_bilevel_positions_then_radii",
    "repair_shrink_on_persistent_failure",
}
BILEVEL_COMPATIBLE_CONFLICTS = {
    "conflict_worst_violation_first",
    "conflict_constraint_graph",
    "conflict_overlap_plus_boundary_penetration",
}
REPAIR_BUDGET_COMPATIBLE_REPAIRS = {
    "repair_sequential_worst_first",
    "repair_shrink_on_persistent_failure",
    "repair_bilevel_positions_then_radii",
}

MODULE_BASE_COMPATIBILITY = {
    "layout_triangular_lattice": 0.25,
    "layout_jittered_low_discrepancy": 0.0,
    "layout_boundary_ring_interior_fill": 0.25,
    "layout_reflective_symmetric_seed": 0.25,
    "selection_center_outward": 0.0,
    "selection_boundary_first": 0.0,
    "selection_uniform_scan_order": -0.25,
    "selection_conflict_priority_order": 0.25,
    "radius_init_uniform_small": -0.25,
    "radius_init_lattice_derived": 0.25,
    "radius_init_role_based_boundary_vs_interior": 0.25,
    "radius_init_local_clearance_based": 0.0,
    "growth_uniform_additive": 0.0,
    "growth_uniform_multiplicative": 0.0,
    "growth_density_scaled_additive": 0.25,
    "growth_role_conditioned_growth": 0.25,
    "conflict_pairwise_overlap": 0.0,
    "conflict_overlap_plus_boundary_penetration": 0.25,
    "conflict_worst_violation_first": 0.25,
    "conflict_constraint_graph": 0.25,
    "repair_pairwise_repulsion": 0.0,
    "repair_boundary_repulsion": 0.0,
    "repair_sequential_worst_first": 0.25,
    "repair_shrink_on_persistent_failure": -0.25,
    "repair_bilevel_positions_then_radii": 0.25,
    "boundary_hard_containment": 0.0,
    "boundary_repulsive_margin": 0.25,
    "boundary_edge_aware_inset": 0.25,
    "boundary_role_based_edge_protection": 0.25,
    "termination_fixed_stability": 0.0,
    "termination_no_violation_and_no_gain": 0.25,
    "termination_plateau_detection": 0.25,
    "termination_repair_budget_exhaustion": 0.0,
}

PAIRWISE_COMPATIBILITY = {
    ("layout", "layout_triangular_lattice", "radius_init", "radius_init_lattice_derived"): (
        0.5,
        "Lattice-derived radii preserve scaffold spacing from the triangular layout.",
    ),
    (
        "layout",
        "layout_boundary_ring_interior_fill",
        "radius_init",
        "radius_init_role_based_boundary_vs_interior",
    ): (
        0.5,
        "Boundary and interior radius roles align with the boundary ring layout.",
    ),
    (
        "layout",
        "layout_reflective_symmetric_seed",
        "radius_init",
        "radius_init_role_based_boundary_vs_interior",
    ): (
        0.25,
        "Role-based radii can preserve reflected boundary and interior duties.",
    ),
    ("layout", "layout_jittered_low_discrepancy", "radius_init", "radius_init_lattice_derived"): (
        -0.25,
        "Spacing-derived radii are less natural for a jittered low-discrepancy layout.",
    ),
    (
        "growth",
        "growth_density_scaled_additive",
        "conflict",
        "conflict_overlap_plus_boundary_penetration",
    ): (
        0.5,
        "Density-scaled growth benefits from conflict reports that include boundary pressure.",
    ),
    ("growth", "growth_role_conditioned_growth", "conflict", "conflict_constraint_graph"): (
        0.25,
        "Role-conditioned growth can use graph-structured conflict pressure.",
    ),
    ("growth", "growth_uniform_additive", "conflict", "conflict_constraint_graph"): (
        -0.25,
        "Uniform additive growth gives little structure to a graph conflict model.",
    ),
    ("conflict", "conflict_constraint_graph", "repair", "repair_pairwise_repulsion"): (
        -0.5,
        "Pairwise repair is too narrow for a constraint-graph conflict report.",
    ),
    ("conflict", "conflict_constraint_graph", "repair", "repair_sequential_worst_first"): (
        0.5,
        "Sequential repair can consume prioritized graph conflicts.",
    ),
    ("conflict", "conflict_constraint_graph", "repair", "repair_bilevel_positions_then_radii"): (
        0.5,
        "Bilevel repair can exploit graph-structured feasibility pressure.",
    ),
    (
        "conflict",
        "conflict_overlap_plus_boundary_penetration",
        "repair",
        "repair_boundary_repulsion",
    ): (
        0.25,
        "Boundary-aware conflict reports support boundary repulsion repair.",
    ),
    ("conflict", "conflict_worst_violation_first", "repair", "repair_sequential_worst_first"): (
        0.5,
        "Worst-first conflict reports align with sequential worst-first repair.",
    ),
    ("repair", "repair_boundary_repulsion", "boundary", "boundary_repulsive_margin"): (
        0.25,
        "Boundary repulsion repair and repulsive margin handling share edge pressure semantics.",
    ),
    ("repair", "repair_pairwise_repulsion", "boundary", "boundary_hard_containment"): (
        -0.25,
        "Pairwise-only repair can leave hard containment pressure isolated.",
    ),
    (
        "repair",
        "repair_bilevel_positions_then_radii",
        "boundary",
        "boundary_role_based_edge_protection",
    ): (
        0.25,
        "Bilevel repair can preserve role-based edge protection while adjusting radii.",
    ),
    (
        "repair",
        "repair_shrink_on_persistent_failure",
        "termination",
        "termination_repair_budget_exhaustion",
    ): (
        0.25,
        "Shrink fallback exposes persistent repair pressure to budget-based stopping.",
    ),
    ("repair", "repair_pairwise_repulsion", "termination", "termination_repair_budget_exhaustion"): (
        -0.5,
        "Pairwise-only repair does not expose a rich repair-budget concept.",
    ),
    (
        "repair",
        "repair_sequential_worst_first",
        "termination",
        "termination_repair_budget_exhaustion",
    ): (
        0.5,
        "Sequential repair gives budget-based stopping a concrete repair process.",
    ),
}

BLOCK_COMPATIBILITY = {
    "placement_block": {
        (
            "layout_triangular_lattice",
            "selection_center_outward",
            "radius_init_lattice_derived",
        ): (
            0.5,
            "The placement block keeps lattice geometry, ordering, and radii coupled.",
        ),
        (
            "layout_boundary_ring_interior_fill",
            "selection_boundary_first",
            "radius_init_role_based_boundary_vs_interior",
        ): (
            0.5,
            "The placement block consistently protects boundary and interior roles.",
        ),
        (
            "layout_jittered_low_discrepancy",
            "selection_uniform_scan_order",
            "radius_init_lattice_derived",
        ): (
            -0.5,
            "The placement block mixes irregular placement with lattice-derived radius assumptions.",
        ),
    },
    "dynamics_block": {
        (
            "growth_density_scaled_additive",
            "conflict_constraint_graph",
            "repair_sequential_worst_first",
            "boundary_repulsive_margin",
        ): (
            0.5,
            "The dynamics block pairs rich conflict structure with corrective repair and soft edges.",
        ),
        (
            "growth_role_conditioned_growth",
            "conflict_overlap_plus_boundary_penetration",
            "repair_bilevel_positions_then_radii",
            "boundary_role_based_edge_protection",
        ): (
            0.5,
            "The dynamics block preserves role-aware structure through growth, repair, and boundary handling.",
        ),
        (
            "growth_uniform_additive",
            "conflict_constraint_graph",
            "repair_pairwise_repulsion",
            "boundary_hard_containment",
        ): (
            -0.5,
            "The dynamics block combines a rich conflict graph with narrow repair and hard containment.",
        ),
    },
}

TERMINATION_INFORMATIVENESS_REQUIREMENTS = {
    "termination_fixed_stability": {
        "required_reads": set(),
        "any_reads": {"feasibility_status", "boundary_report", "packing_summary"},
        "compatible_repairs": None,
    },
    "termination_no_violation_and_no_gain": {
        "required_reads": {"feasibility_status", "packing_summary"},
        "any_reads": set(),
        "compatible_repairs": None,
    },
    "termination_plateau_detection": {
        "required_reads": {"packing_summary"},
        "any_reads": set(),
        "compatible_repairs": None,
    },
    "termination_repair_budget_exhaustion": {
        "required_reads": {"feasibility_status"},
        "any_reads": set(),
        "compatible_repairs": set(REPAIR_BUDGET_COMPATIBLE_REPAIRS),
    },
}

SELF_DEFEATING_SCORE_COMBINATIONS = {
    (
        "radius_init_uniform_small",
        "repair_shrink_on_persistent_failure",
        "termination_fixed_stability",
    ),
    (
        "radius_init_uniform_small",
        "repair_shrink_on_persistent_failure",
        "termination_repair_budget_exhaustion",
    ),
}

PARAMETER_INTERFACE_INVALIDITY = set()
INVARIANT_CONFLICT_ASSUMPTION = "preserve radii whenever conflicts persist"

_COMPATIBILITY_TOP_LEVEL_KEYS = {
    "schema_name",
    "schema_version",
    "task_family",
    "organism_id",
    "hard_compatibility",
    "soft_compatibility",
    "interface_report",
}
_FUNCTIONAL_TOP_LEVEL_KEYS = {
    "schema_name",
    "schema_version",
    "task_family",
    "organism_id",
    "checks_passed",
    "checks",
}
_HARD_VIOLATION_KEYS = {"rule_id", "severity", "scope", "slot", "other_slot", "message"}
_SOFT_COMPONENT_KEYS = {"component_id", "scope", "slot", "other_slot", "score", "message"}
_INTERFACE_VIOLATION_KEYS = {"rule_id", "slot", "message"}
_FUNCTIONAL_CHECK_KEYS = {"check_id", "status", "category", "message"}


def _fail(path: str, message: str) -> None:
    raise ValueError(f"{path}: {message}")


def _ensure_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    return value


def _ensure_exact_keys(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    payload = _ensure_dict(value, path)
    keys = set(payload.keys())
    missing = sorted(expected.difference(keys))
    extra = sorted(keys.difference(expected))
    if missing:
        _fail(path, f"missing keys: {', '.join(missing)}")
    if extra:
        _fail(path, f"extra keys: {', '.join(extra)}")
    return payload


def _ensure_str(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        _fail(path, "must be a string")
    if value.strip() != value:
        _fail(path, "must be stripped")
    if not allow_empty and not value:
        _fail(path, "must be non-empty")
    return value


def _ensure_str_list(value: Any, path: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list):
        _fail(path, "must be a list")
    if not allow_empty and not value:
        _fail(path, "must be non-empty")
    return [_ensure_str(item, f"{path}[{idx}]") for idx, item in enumerate(value)]


def _find_null(value: Any, path: str = "$") -> str | None:
    if value is None:
        return path
    if isinstance(value, dict):
        for key, child in value.items():
            found = _find_null(child, f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            found = _find_null(child, f"{path}[{idx}]")
            if found is not None:
                return found
    return None


def _module_key(genome: dict[str, Any], slot: str) -> str:
    module = genome.get("slots", {}).get(slot)
    if not isinstance(module, dict):
        return ""
    value = module.get("module_key")
    return value if isinstance(value, str) else ""


def _module_keys(genome: dict[str, Any]) -> dict[str, str]:
    return {slot: _module_key(genome, slot) for slot in SLOT_ORDER}


def _parameter_tuple(module: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    values = []
    parameterization = module.get("parameterization", [])
    if not isinstance(parameterization, list):
        return tuple()
    for entry in parameterization:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            values.append((entry["name"], entry.get("value")))
    return tuple(sorted(values))


def _validate_reportable_genome_shape(genome: dict[str, Any]) -> None:
    if not isinstance(genome, dict):
        raise ValueError("Compatibility input genome must be a JSON object.")
    null_path = _find_null(genome)
    if null_path is not None:
        raise ValueError(f"Compatibility input genome contains null at {null_path}.")
    _ensure_exact_keys(genome, schema.TOP_LEVEL_KEYS, "$")
    expected_values = {
        "schema_name": schema.SCHEMA_NAME,
        "schema_version": schema.SCHEMA_VERSION,
        "task_family": schema.TASK_FAMILY,
        "task_name": schema.TASK_NAME,
        "representation": schema.REPRESENTATION,
    }
    for key, expected in expected_values.items():
        if genome.get(key) != expected:
            _fail(key, f"expected {expected!r}, got {genome.get(key)!r}")
    _ensure_str(genome.get("organism_id"), "organism_id")
    if genome.get("slot_order") != list(SLOT_ORDER):
        _fail("slot_order", f"must be exactly {list(SLOT_ORDER)!r}")
    if genome.get("linkage_groups") != schema.canonical_linkage_group_list():
        _fail("linkage_groups", "must be canonical")

    slots = _ensure_dict(genome.get("slots"), "slots")
    for slot, module in slots.items():
        if slot not in SLOT_ORDER:
            continue
        payload = _ensure_exact_keys(module, library.MODULE_INSTANCE_KEYS, f"slots.{slot}")
        module_key = _ensure_str(payload["module_key"], f"slots.{slot}.module_key")
        try:
            library.get_module_definition(module_key)
        except ValueError as exc:
            raise ValueError(f"slots.{slot}.module_key: unknown module_key {module_key!r}") from exc
        _ensure_str(payload["module_id"], f"slots.{slot}.module_id")
        _ensure_str(payload["slot"], f"slots.{slot}.slot")
        _ensure_str(payload["module_type"], f"slots.{slot}.module_type")
        _ensure_str(payload["family"], f"slots.{slot}.family")
        _ensure_str(payload["hypothesis"], f"slots.{slot}.hypothesis")
        reads_state = _ensure_str_list(payload["reads_state"], f"slots.{slot}.reads_state", allow_empty=True)
        writes_state = _ensure_str_list(payload["writes_state"], f"slots.{slot}.writes_state", allow_empty=True)
        for state in reads_state:
            if state not in schema.STATE_VOCAB:
                _fail(f"slots.{slot}.reads_state", f"illegal state symbol {state!r}")
        for state in writes_state:
            if state not in schema.STATE_VOCAB:
                _fail(f"slots.{slot}.writes_state", f"illegal state symbol {state!r}")


def _violation(
    rule_id: str,
    scope: str,
    message: str,
    *,
    slot: str = "",
    other_slot: str = "",
) -> dict[str, str]:
    return {
        "rule_id": rule_id,
        "severity": "error",
        "scope": scope,
        "slot": slot,
        "other_slot": other_slot,
        "message": message,
    }


def _component(
    component_id: str,
    scope: str,
    score: float,
    message: str,
    *,
    slot: str = "",
    other_slot: str = "",
) -> dict[str, Any]:
    return {
        "component_id": component_id,
        "scope": scope,
        "slot": slot,
        "other_slot": other_slot,
        "score": float(score),
        "message": message,
    }


def _check_result(check_id: str, status: str, category: str, message: str) -> dict[str, str]:
    return {
        "check_id": check_id,
        "status": status,
        "category": category,
        "message": message,
    }


def _add_slot_validity_violations(genome: dict[str, Any], violations: list[dict[str, str]]) -> None:
    slots = genome.get("slots")
    if not isinstance(slots, dict):
        violations.append(_violation("HC-SLOT-001", "global", "Genome slots must be an object."))
        return

    slot_keys = set(slots.keys())
    missing = sorted(set(SLOT_ORDER).difference(slot_keys))
    extra = sorted(slot_keys.difference(SLOT_ORDER))
    for slot in missing:
        violations.append(_violation("HC-SLOT-001", "slot", f"Missing canonical slot {slot!r}.", slot=slot))
    for slot in extra:
        violations.append(_violation("HC-SLOT-001", "slot", f"Extra noncanonical slot {slot!r}.", slot=slot))

    seen_module_ids: set[str] = set()
    for slot in SLOT_ORDER:
        module = slots.get(slot)
        if not isinstance(module, dict):
            continue
        module_key = module["module_key"]
        definition = library.get_module_definition(module_key)
        if definition["slot"] != slot or module.get("slot") != slot:
            violations.append(
                _violation(
                    "HC-SLOT-001",
                    "slot",
                    f"Module {module_key!r} is not valid for slot {slot!r}.",
                    slot=slot,
                )
            )

        module_id = module.get("module_id")
        if isinstance(module_id, str):
            if module_id in seen_module_ids:
                violations.append(
                    _violation(
                        "HC-SLOT-004",
                        "global",
                        f"Duplicate module_id {module_id!r}.",
                        slot=slot,
                    )
                )
            seen_module_ids.add(module_id)

        try:
            expected = library.materialize_module_instance(
                slot=slot,
                module_key=module_key,
                module_id=module["module_id"],
                parameterization=module["parameterization"],
            )
        except ValueError as exc:
            violations.append(
                _violation(
                    "HC-SLOT-003",
                    "slot",
                    f"Invalid parameterization for {module_key!r}: {exc}",
                    slot=slot,
                )
            )
            continue

        controlled_fields = sorted(set(expected.keys()).difference({"module_id", "parameterization", "hypothesis"}))
        drifted = [field for field in controlled_fields if module.get(field) != expected[field]]
        if drifted:
            violations.append(
                _violation(
                    "HC-SLOT-002",
                    "slot",
                    "Library-controlled module metadata drifted: " + ", ".join(drifted),
                    slot=slot,
                )
            )


def _add_minimum_interface_violations(genome: dict[str, Any], violations: list[dict[str, str]]) -> None:
    for slot in SLOT_ORDER:
        module = genome.get("slots", {}).get(slot)
        if not isinstance(module, dict):
            continue
        reads = module.get("reads_state", [])
        writes = module.get("writes_state", [])
        requirements = schema.SLOT_INTERFACE_REQUIREMENTS[slot]
        for state in requirements.get("reads", []):
            if state not in reads:
                violations.append(
                    _violation(
                        "HC-IF-002",
                        "slot",
                        f"Slot {slot!r} must read required state {state!r}.",
                        slot=slot,
                    )
                )
        for state in requirements.get("writes", []):
            if state not in writes:
                violations.append(
                    _violation(
                        "HC-IF-002",
                        "slot",
                        f"Slot {slot!r} must write required state {state!r}.",
                        slot=slot,
                    )
                )
        reads_any = requirements.get("reads_any", [])
        if reads_any and not any(state in reads for state in reads_any):
            violations.append(
                _violation(
                    "HC-IF-002",
                    "slot",
                    f"Slot {slot!r} must read at least one of {reads_any!r}.",
                    slot=slot,
                )
            )


def _add_interface_chain_violations(genome: dict[str, Any], violations: list[dict[str, str]]) -> None:
    slots = genome.get("slots", {})
    writes_by_slot = {
        slot: set(slots.get(slot, {}).get("writes_state", []))
        if isinstance(slots.get(slot), dict)
        else set()
        for slot in SLOT_ORDER
    }
    produced_before: set[str] = set()
    for idx, slot in enumerate(SLOT_ORDER):
        module = slots.get(slot)
        if not isinstance(module, dict):
            continue
        reads = set(module.get("reads_state", []))
        writes = set(module.get("writes_state", []))
        if not reads and not writes:
            violations.append(
                _violation(
                    "HC-IF-004",
                    "slot",
                    f"Slot {slot!r} has an empty operational interface.",
                    slot=slot,
                )
            )
        for state in sorted(reads):
            later_writers = [
                later_slot
                for later_slot in SLOT_ORDER[idx + 1 :]
                if state in writes_by_slot[later_slot]
            ]
            if state in produced_before:
                continue
            if later_writers:
                violations.append(
                    _violation(
                        "HC-IF-003",
                        "slot",
                        f"Slot {slot!r} reads {state!r}, which is only produced by later slot {later_writers[0]!r}.",
                        slot=slot,
                    )
                )
            elif slot != "layout":
                violations.append(
                    _violation(
                        "HC-IF-001",
                        "slot",
                        f"Slot {slot!r} reads {state!r} without a producing predecessor.",
                        slot=slot,
                    )
                )
        produced_before.update(writes)


def _add_invariant_and_parameter_violations(genome: dict[str, Any], violations: list[dict[str, str]]) -> None:
    slots = genome.get("slots", {})
    boundary_key = _module_key(genome, "boundary")
    repair = slots.get("repair", {})
    repair_key = _module_key(genome, "repair")

    if boundary_key == "boundary_hard_containment":
        repair_tags = set(repair.get("compatibility_tags", [])) if isinstance(repair, dict) else set()
        if "unrestricted_outward_displacement" in repair_tags:
            violations.append(
                _violation(
                    "HC-IFC-003",
                    "pair",
                    "Hard containment is incompatible with unrestricted outward displacement repair.",
                    slot="repair",
                    other_slot="boundary",
                )
            )

    if repair_key == "repair_shrink_on_persistent_failure":
        texts: list[str] = []
        for module in slots.values():
            if not isinstance(module, dict):
                continue
            texts.extend(module.get("assumptions", []))
            texts.extend(module.get("invariants", []))
        if any(INVARIANT_CONFLICT_ASSUMPTION in text.lower() for text in texts):
            violations.append(
                _violation(
                    "HC-IFC-003",
                    "global",
                    "Shrink-on-failure repair conflicts with a preserve-radii persistence assumption.",
                )
            )

    module_keys = _module_keys(genome)
    for slot in SLOT_ORDER:
        module = slots.get(slot)
        if not isinstance(module, dict):
            continue
        for parameter_name, value in _parameter_tuple(module):
            gate = (module_keys[slot], parameter_name, value)
            if gate in PARAMETER_INTERFACE_INVALIDITY:
                violations.append(
                    _violation(
                        "HC-IFC-004",
                        "slot",
                        f"Parameter {parameter_name!r} invalidates the declared interface mode.",
                        slot=slot,
                    )
                )


def _add_linkage_and_termination_violations(genome: dict[str, Any], violations: list[dict[str, str]]) -> None:
    keys = _module_keys(genome)
    layout = keys["layout"]
    radius_init = keys["radius_init"]
    growth = keys["growth"]
    conflict = keys["conflict"]
    repair = keys["repair"]
    boundary = keys["boundary"]
    termination = keys["termination"]

    forbidden_radius_for_structured_layouts: set[str] = set()
    if layout in STRUCTURED_PLACEMENT_LAYOUTS and radius_init in forbidden_radius_for_structured_layouts:
        violations.append(
            _violation(
                "HC-LG-001",
                "linkage_group",
                "Structured placement layout conflicts with a radius initializer that rejects geometric roles.",
                slot="layout",
                other_slot="radius_init",
            )
        )

    if growth == "growth_role_conditioned_growth":
        support_modules = {layout, radius_init, boundary}
        if not support_modules.intersection(ROLE_CONDITIONED_GROWTH_SUPPORT):
            violations.append(
                _violation(
                    "HC-LG-002",
                    "linkage_group",
                    "Role-conditioned growth requires role-aware placement, initialization, or boundary support.",
                    slot="growth",
                    other_slot="dynamics_block",
                )
            )

    if conflict == "conflict_constraint_graph" and repair not in CONSTRAINT_GRAPH_REPAIRS:
        violations.append(
            _violation(
                "HC-LG-003",
                "linkage_group",
                "Constraint-graph conflict requires nontrivial repair support.",
                slot="conflict",
                other_slot="repair",
            )
        )

    if repair == "repair_bilevel_positions_then_radii" and conflict not in BILEVEL_COMPATIBLE_CONFLICTS:
        violations.append(
            _violation(
                "HC-LG-004",
                "linkage_group",
                "Bilevel repair requires an enriched conflict definition.",
                slot="repair",
                other_slot="conflict",
            )
        )

    if termination == "termination_repair_budget_exhaustion" and repair not in REPAIR_BUDGET_COMPATIBLE_REPAIRS:
        violations.append(
            _violation(
                "HC-TR-001",
                "pair",
                "Repair-budget termination requires repair modules with explicit repair progression.",
                slot="termination",
                other_slot="repair",
            )
        )


def evaluate_hard_compatibility_rules(genome: dict) -> list[dict]:
    """Return hard compatibility violations for a reportable circle-packing genome."""

    _validate_reportable_genome_shape(genome)
    violations: list[dict[str, str]] = []
    _add_slot_validity_violations(genome, violations)
    _add_interface_chain_violations(genome, violations)
    _add_minimum_interface_violations(genome, violations)
    _add_invariant_and_parameter_violations(genome, violations)
    _add_linkage_and_termination_violations(genome, violations)
    return violations


def build_interface_report(genome: dict) -> dict:
    """Build the first-class interface report for a reportable genome."""

    _validate_reportable_genome_shape(genome)
    state_reads: dict[str, list[str]] = {}
    state_writes: dict[str, list[str]] = {}
    for slot in SLOT_ORDER:
        module = genome.get("slots", {}).get(slot)
        if isinstance(module, dict):
            state_reads[slot] = list(module.get("reads_state", []))
            state_writes[slot] = list(module.get("writes_state", []))
        else:
            state_reads[slot] = []
            state_writes[slot] = []

    hard_violations = evaluate_hard_compatibility_rules(genome)
    interface_violations = [
        {
            "rule_id": violation["rule_id"],
            "slot": violation["slot"] or "global",
            "message": violation["message"],
        }
        for violation in hard_violations
        if violation["rule_id"].startswith("HC-IF")
    ]
    report = {
        "state_reads": state_reads,
        "state_writes": state_writes,
        "state_obligations_satisfied": not interface_violations,
        "violations": interface_violations,
    }
    _validate_interface_report(report)
    return report


def evaluate_soft_compatibility_components(genome: dict) -> list[dict]:
    """Return deterministic soft compatibility components."""

    _validate_reportable_genome_shape(genome)
    keys = _module_keys(genome)
    components: list[dict[str, Any]] = []

    for slot in SLOT_ORDER:
        module_key = keys[slot]
        score = MODULE_BASE_COMPATIBILITY.get(module_key, 0.0)
        components.append(
            _component(
                f"SC-SLOT-{slot}",
                "slot",
                score,
                f"Per-slot structural contribution for {module_key}.",
                slot=slot,
            )
        )

    for slot_a, slot_b in SLOT_PAIRS_TO_SCORE:
        module_a = keys[slot_a]
        module_b = keys[slot_b]
        rule = PAIRWISE_COMPATIBILITY.get((slot_a, module_a, slot_b, module_b))
        if rule is None:
            score = 0.0
            message = f"No explicit pairwise rule matched {module_a} with {module_b}."
        else:
            score, message = rule
        components.append(
            _component(
                f"SC-PAIR-{slot_a}-{slot_b}",
                "pair",
                score,
                message,
                slot=slot_a,
                other_slot=slot_b,
            )
        )

    for group_id in ("placement_block", "dynamics_block"):
        block_slots = schema.LINKAGE_GROUPS[group_id]
        block_keys = tuple(keys[slot] for slot in block_slots)
        rule = BLOCK_COMPATIBILITY[group_id].get(block_keys)
        if rule is None:
            score = 0.0
            message = f"No explicit block rule matched {group_id}."
        else:
            score, message = rule
        components.append(
            _component(
                f"SC-BLOCK-{group_id.replace('_block', '')}",
                "linkage_group",
                score,
                message,
                slot=group_id,
            )
        )

    if evaluate_hard_compatibility_rules(genome):
        components.append(
            _component(
                "SC-HARD-INCOMPATIBLE",
                "global",
                0.0,
                "Soft score is informational only because hard compatibility failed.",
            )
        )
    return components


def compute_soft_compatibility(genome: dict) -> dict:
    """Compute deterministic additive soft compatibility."""

    components = evaluate_soft_compatibility_components(genome)
    payload = {
        "total_score": float(sum(component["score"] for component in components)),
        "components": components,
    }
    _validate_soft_compatibility(payload)
    return payload


def build_circle_packing_compatibility_report(genome: dict) -> dict:
    """Build the canonical compatibility report for one genome."""

    _validate_reportable_genome_shape(genome)
    hard_violations = evaluate_hard_compatibility_rules(genome)
    report = {
        "schema_name": COMPATIBILITY_REPORT_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "task_family": TASK_FAMILY,
        "organism_id": genome["organism_id"],
        "hard_compatibility": {
            "is_compatible": not hard_violations,
            "violations": hard_violations,
        },
        "soft_compatibility": compute_soft_compatibility(genome),
        "interface_report": build_interface_report(genome),
    }
    validate_compatibility_report(report)
    return report


def validate_hard_compatibility(genome: dict) -> None:
    """Raise if hard compatibility fails."""

    report = build_circle_packing_compatibility_report(genome)
    violations = report["hard_compatibility"]["violations"]
    if violations:
        reasons = "; ".join(f"{item['rule_id']}: {item['message']}" for item in violations)
        raise ValueError(f"Hard compatibility failed for {genome['organism_id']}: {reasons}")


def _stage_chain_exists(genome: dict[str, Any]) -> bool:
    writes = {
        slot: set(genome.get("slots", {}).get(slot, {}).get("writes_state", []))
        for slot in SLOT_ORDER
    }
    reads = {
        slot: set(genome.get("slots", {}).get(slot, {}).get("reads_state", []))
        for slot in SLOT_ORDER
    }
    return (
        "candidate_centers" in writes["layout"]
        and "candidate_centers" in reads["selection"]
        and "ordered_centers" in writes["selection"]
        and "ordered_centers" in reads["radius_init"]
        and "current_radii" in writes["radius_init"]
        and "current_radii" in reads["growth"]
        and "proposed_radii" in writes["growth"]
        and "proposed_radii" in reads["conflict"]
        and "conflict_report" in writes["conflict"]
        and "conflict_report" in reads["repair"]
        and "feasibility_status" in writes["repair"]
        and "proposed_radii" in reads["boundary"]
        and "boundary_report" in writes["boundary"]
        and "termination_signal" in writes["termination"]
        and bool(reads["termination"].intersection({"packing_summary", "feasibility_status", "boundary_report"}))
    )


def _feasibility_path_exists(genome: dict[str, Any]) -> bool:
    slots = genome.get("slots", {})
    for slot in ("conflict", "repair", "boundary"):
        module = slots.get(slot, {})
        if not isinstance(module, dict):
            return False
        if not module.get("reads_state") or not module.get("writes_state"):
            return False
    return True


def _score_preservation_is_sane(genome: dict[str, Any]) -> bool:
    keys = _module_keys(genome)
    combination = (
        keys["radius_init"],
        keys["repair"],
        keys["termination"],
    )
    return combination not in SELF_DEFEATING_SCORE_COMBINATIONS


def _role_consistency_is_sane(genome: dict[str, Any]) -> bool:
    keys = _module_keys(genome)
    present_role_modules = {keys[slot] for slot in ("radius_init", "growth", "boundary")}.intersection(
        ROLE_AWARE_MODULES
    )
    if not present_role_modules:
        return True
    supporting_layout = keys["layout"] in ROLE_SUPPORT_LAYOUTS
    return supporting_layout or len(present_role_modules) >= 2


def _termination_is_informative(genome: dict[str, Any]) -> bool:
    keys = _module_keys(genome)
    termination_key = keys["termination"]
    if termination_key not in TERMINATION_INFORMATIVENESS_REQUIREMENTS:
        return False
    requirements = TERMINATION_INFORMATIVENESS_REQUIREMENTS[termination_key]
    termination = genome.get("slots", {}).get("termination")
    if not isinstance(termination, dict):
        return False
    reads = set(termination.get("reads_state", []))
    required_reads = requirements["required_reads"]
    any_reads = requirements["any_reads"]
    compatible_repairs = requirements["compatible_repairs"]

    if required_reads and not required_reads.issubset(reads):
        return False
    if any_reads and not reads.intersection(any_reads):
        return False
    if compatible_repairs is not None and keys["repair"] not in compatible_repairs:
        return False
    return True


def evaluate_functional_checks(genome: dict) -> list[dict]:
    """Return deterministic cheap functional checks for a reportable genome."""

    _validate_reportable_genome_shape(genome)
    checks = [
        _check_result(
            "FC-001",
            "passed" if _stage_chain_exists(genome) else "failed",
            "interface",
            "Executable stage chain exists."
            if _stage_chain_exists(genome)
            else "The canonical layout-to-termination stage chain is broken.",
        ),
        _check_result(
            "FC-002",
            "passed" if _feasibility_path_exists(genome) else "failed",
            "behavioral_proxy",
            "Feasibility path exists through conflict, repair, and boundary modules."
            if _feasibility_path_exists(genome)
            else "Conflict, repair, or boundary handling is operationally empty.",
        ),
        _check_result(
            "FC-003",
            "passed" if _score_preservation_is_sane(genome) else "failed",
            "behavioral_proxy",
            "Score-preservation proxy is sane."
            if _score_preservation_is_sane(genome)
            else "The genome combines conservative initialization, shrink repair, and conservative stopping.",
        ),
        _check_result(
            "FC-004",
            "passed" if _role_consistency_is_sane(genome) else "failed",
            "linkage",
            "Role-aware modules have support."
            if _role_consistency_is_sane(genome)
            else "A role-aware module appears without another role-aware module or supporting structured layout.",
        ),
        _check_result(
            "FC-005",
            "passed" if _termination_is_informative(genome) else "failed",
            "behavioral_proxy",
            "Termination module observes enough state for its stopping notion."
            if _termination_is_informative(genome)
            else "Termination module lacks the state or repair support required by its stopping notion.",
        ),
    ]
    return checks


def build_circle_packing_functional_checks(genome: dict) -> dict:
    """Build the canonical cheap functional check artifact."""

    _validate_reportable_genome_shape(genome)
    checks = evaluate_functional_checks(genome)
    report = {
        "schema_name": FUNCTIONAL_CHECKS_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "task_family": TASK_FAMILY,
        "organism_id": genome["organism_id"],
        "checks_passed": all(check["status"] == "passed" for check in checks),
        "checks": checks,
    }
    validate_functional_checks_report(report)
    return report


def _validate_hard_compatibility(payload: Any) -> None:
    data = _ensure_exact_keys(payload, {"is_compatible", "violations"}, "hard_compatibility")
    if not isinstance(data["is_compatible"], bool):
        _fail("hard_compatibility.is_compatible", "must be a boolean")
    if not isinstance(data["violations"], list):
        _fail("hard_compatibility.violations", "must be a list")
    for idx, violation in enumerate(data["violations"]):
        item = _ensure_exact_keys(violation, _HARD_VIOLATION_KEYS, f"hard_compatibility.violations[{idx}]")
        if item["severity"] != "error":
            _fail(f"hard_compatibility.violations[{idx}].severity", "must be 'error'")
        if item["scope"] not in {"global", "slot", "pair", "linkage_group"}:
            _fail(f"hard_compatibility.violations[{idx}].scope", "has invalid scope")
        _ensure_str(item["rule_id"], f"hard_compatibility.violations[{idx}].rule_id")
        _ensure_str(item["slot"], f"hard_compatibility.violations[{idx}].slot", allow_empty=True)
        _ensure_str(item["other_slot"], f"hard_compatibility.violations[{idx}].other_slot", allow_empty=True)
        _ensure_str(item["message"], f"hard_compatibility.violations[{idx}].message")
    if data["is_compatible"] != (len(data["violations"]) == 0):
        _fail("hard_compatibility.is_compatible", "must match absence of violations")


def _validate_soft_compatibility(payload: Any) -> None:
    data = _ensure_exact_keys(payload, {"total_score", "components"}, "soft_compatibility")
    if not isinstance(data["total_score"], (int, float)) or isinstance(data["total_score"], bool):
        _fail("soft_compatibility.total_score", "must be a number")
    if not isinstance(data["components"], list):
        _fail("soft_compatibility.components", "must be a list")
    for idx, component in enumerate(data["components"]):
        item = _ensure_exact_keys(component, _SOFT_COMPONENT_KEYS, f"soft_compatibility.components[{idx}]")
        if item["scope"] not in {"global", "slot", "pair", "linkage_group"}:
            _fail(f"soft_compatibility.components[{idx}].scope", "has invalid scope")
        if not isinstance(item["score"], (int, float)) or isinstance(item["score"], bool):
            _fail(f"soft_compatibility.components[{idx}].score", "must be a number")
        _ensure_str(item["component_id"], f"soft_compatibility.components[{idx}].component_id")
        _ensure_str(item["slot"], f"soft_compatibility.components[{idx}].slot", allow_empty=True)
        _ensure_str(item["other_slot"], f"soft_compatibility.components[{idx}].other_slot", allow_empty=True)
        _ensure_str(item["message"], f"soft_compatibility.components[{idx}].message")


def _validate_interface_report(payload: Any) -> None:
    data = _ensure_exact_keys(
        payload,
        {"state_reads", "state_writes", "state_obligations_satisfied", "violations"},
        "interface_report",
    )
    for key in ("state_reads", "state_writes"):
        mapping = _ensure_dict(data[key], f"interface_report.{key}")
        if list(mapping.keys()) != list(SLOT_ORDER):
            _fail(f"interface_report.{key}", "must contain canonical slot keys in order")
        for slot in SLOT_ORDER:
            _ensure_str_list(mapping[slot], f"interface_report.{key}.{slot}", allow_empty=True)
    if not isinstance(data["state_obligations_satisfied"], bool):
        _fail("interface_report.state_obligations_satisfied", "must be a boolean")
    if not isinstance(data["violations"], list):
        _fail("interface_report.violations", "must be a list")
    for idx, violation in enumerate(data["violations"]):
        item = _ensure_exact_keys(violation, _INTERFACE_VIOLATION_KEYS, f"interface_report.violations[{idx}]")
        _ensure_str(item["rule_id"], f"interface_report.violations[{idx}].rule_id")
        _ensure_str(item["slot"], f"interface_report.violations[{idx}].slot")
        _ensure_str(item["message"], f"interface_report.violations[{idx}].message")
    if data["state_obligations_satisfied"] != (len(data["violations"]) == 0):
        _fail("interface_report.state_obligations_satisfied", "must match absence of violations")


def validate_compatibility_report(report: dict) -> None:
    """Validate a compatibility report artifact shape."""

    if not isinstance(report, dict):
        raise ValueError("Compatibility report must be a JSON object.")
    null_path = _find_null(report)
    if null_path is not None:
        raise ValueError(f"Compatibility report contains null at {null_path}.")
    payload = _ensure_exact_keys(report, _COMPATIBILITY_TOP_LEVEL_KEYS, "$")
    if payload["schema_name"] != COMPATIBILITY_REPORT_SCHEMA_NAME:
        _fail("schema_name", f"expected {COMPATIBILITY_REPORT_SCHEMA_NAME!r}")
    if payload["schema_version"] != SCHEMA_VERSION:
        _fail("schema_version", f"expected {SCHEMA_VERSION!r}")
    if payload["task_family"] != TASK_FAMILY:
        _fail("task_family", f"expected {TASK_FAMILY!r}")
    _ensure_str(payload["organism_id"], "organism_id")
    _validate_hard_compatibility(payload["hard_compatibility"])
    _validate_soft_compatibility(payload["soft_compatibility"])
    _validate_interface_report(payload["interface_report"])


def validate_functional_checks_report(report: dict) -> None:
    """Validate a functional checks artifact shape."""

    if not isinstance(report, dict):
        raise ValueError("Functional checks report must be a JSON object.")
    null_path = _find_null(report)
    if null_path is not None:
        raise ValueError(f"Functional checks report contains null at {null_path}.")
    payload = _ensure_exact_keys(report, _FUNCTIONAL_TOP_LEVEL_KEYS, "$")
    if payload["schema_name"] != FUNCTIONAL_CHECKS_SCHEMA_NAME:
        _fail("schema_name", f"expected {FUNCTIONAL_CHECKS_SCHEMA_NAME!r}")
    if payload["schema_version"] != SCHEMA_VERSION:
        _fail("schema_version", f"expected {SCHEMA_VERSION!r}")
    if payload["task_family"] != TASK_FAMILY:
        _fail("task_family", f"expected {TASK_FAMILY!r}")
    _ensure_str(payload["organism_id"], "organism_id")
    if not isinstance(payload["checks_passed"], bool):
        _fail("checks_passed", "must be a boolean")
    if not isinstance(payload["checks"], list):
        _fail("checks", "must be a list")
    for idx, check in enumerate(payload["checks"]):
        item = _ensure_exact_keys(check, _FUNCTIONAL_CHECK_KEYS, f"checks[{idx}]")
        if item["status"] not in {"passed", "failed"}:
            _fail(f"checks[{idx}].status", "must be passed or failed")
        if item["category"] not in {"schema", "interface", "linkage", "behavioral_proxy"}:
            _fail(f"checks[{idx}].category", "has invalid category")
        _ensure_str(item["check_id"], f"checks[{idx}].check_id")
        _ensure_str(item["message"], f"checks[{idx}].message")
    if payload["checks_passed"] != all(check["status"] == "passed" for check in payload["checks"]):
        _fail("checks_passed", "must equal all statuses are passed")


def _validate_rule_tables() -> None:
    library_keys = {
        module_key
        for slot in SLOT_ORDER
        for module_key in library.REQUIRED_MODULE_KEYS[slot]
    }
    if set(MODULE_BASE_COMPATIBILITY.keys()) != library_keys:
        missing = sorted(library_keys.difference(MODULE_BASE_COMPATIBILITY))
        extra = sorted(set(MODULE_BASE_COMPATIBILITY).difference(library_keys))
        raise ValueError(f"MODULE_BASE_COMPATIBILITY keys drifted; missing={missing}, extra={extra}")
    for module_key, score in MODULE_BASE_COMPATIBILITY.items():
        if score not in {-0.25, 0.0, 0.25}:
            raise ValueError(f"Invalid per-slot compatibility score for {module_key!r}: {score!r}")
    for pair_key, (score, _message) in PAIRWISE_COMPATIBILITY.items():
        if score not in {-0.5, -0.25, 0.25, 0.5}:
            raise ValueError(f"Invalid pairwise compatibility score for {pair_key!r}: {score!r}")
        slot_a, module_a, slot_b, module_b = pair_key
        if (slot_a, slot_b) not in SLOT_PAIRS_TO_SCORE:
            raise ValueError(f"Pairwise rule uses unsupported slot pair {slot_a!r}, {slot_b!r}")
        library.validate_slot_assignment(slot_a, module_a)
        library.validate_slot_assignment(slot_b, module_b)
    for group_id, table in BLOCK_COMPATIBILITY.items():
        if group_id not in {"placement_block", "dynamics_block"}:
            raise ValueError(f"Unsupported block compatibility group {group_id!r}")
        for block_keys, (score, _message) in table.items():
            if score not in {-0.5, 0.0, 0.5}:
                raise ValueError(f"Invalid block compatibility score for {block_keys!r}: {score!r}")
            block_slots = schema.LINKAGE_GROUPS[group_id]
            if len(block_keys) != len(block_slots):
                raise ValueError(f"Block rule {group_id!r} has wrong key length.")
            for slot, module_key in zip(block_slots, block_keys, strict=True):
                library.validate_slot_assignment(slot, module_key)


_validate_rule_tables()
