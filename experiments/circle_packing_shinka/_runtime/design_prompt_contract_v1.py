"""Strict typed design prompt contract for circle_packing_shinka v1."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from experiments.circle_packing_shinka._runtime import compatibility_contract_v1 as compatibility
from experiments.circle_packing_shinka._runtime import genome_schema_v1 as schema
from experiments.circle_packing_shinka._runtime import module_library_v1 as library


DESIGN_SCHEMA_NAME = "circle_packing_typed_design_response"
NOVELTY_SCHEMA_NAME = "circle_packing_typed_novelty_verdict"
SCHEMA_VERSION = "1.0"
TASK_FAMILY = schema.TASK_FAMILY
TASK_NAME = schema.TASK_NAME

DESIGN_TOP_LEVEL_KEYS = {
    "schema_name",
    "schema_version",
    "task_family",
    "task_name",
    "operation",
    "global_hypothesis",
    "slot_assignments",
    "render_fields",
    "operator_metadata",
}
NOVELTY_TOP_LEVEL_KEYS = {
    "schema_name",
    "schema_version",
    "task_family",
    "task_name",
    "operation",
    "verdict",
    "supported_differences",
    "rejection_reasons",
}
GLOBAL_HYPOTHESIS_KEYS = {
    "title",
    "core_claim",
    "expected_advantage",
    "novelty_statement",
}
RENDER_FIELDS_KEYS = {"interaction_notes", "compute_notes", "change_description"}
SLOT_ASSIGNMENT_KEYS = {"module_key", "module_id", "parameterization"}
SEED_METADATA = {"seed_family": "new_seed", "design_mode": "typed_library_selection"}
MUTATION_METADATA_KEYS = {
    "mutation_scope",
    "changed_slots",
    "preserved_slots",
    "parent_reference",
    "change_rationale",
}
CROSSOVER_METADATA_KEYS = {
    "inheritance_mode",
    "slot_origins",
    "primary_slots",
    "secondary_slots",
    "change_rationale",
}
NOVELTY_DIFFERENCE_KEYS = {"kind", "slot", "message"}
NOVELTY_REJECTION_KEYS = {"reason_id", "message"}
NOVELTY_DIFFERENCE_KINDS = {
    "slot_change",
    "linkage_block_change",
    "recombination_structure",
    "parameter_change",
}


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


def _ensure_str_list(value: Any, path: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        _fail(path, "must be a list")
    if not allow_empty and not value:
        _fail(path, "must be non-empty")
    return [_ensure_str(item, f"{path}[{idx}]") for idx, item in enumerate(value)]


def _reject_nulls_and_numbers(value: Any, path: str = "$") -> None:
    if value is None:
        _fail(path, "null values are forbidden")
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        _fail(path, "numeric JSON values are forbidden")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nulls_and_numbers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _reject_nulls_and_numbers(child, f"{path}[{idx}]")


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    text = str(raw_text).strip()
    if not text:
        raise ValueError("Typed prompt response must not be empty.")
    if not text.startswith("{") or not text.endswith("}"):
        raise ValueError("Typed prompt response must be exactly one valid JSON object with no surrounding text.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Typed prompt response is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Typed prompt response must decode to a JSON object.")
    return payload


def _canonical_slot_set(values: list[str], path: str) -> set[str]:
    seen: set[str] = set()
    for value in values:
        if value not in schema.SLOT_ORDER:
            _fail(path, f"unknown slot {value!r}")
        if value in seen:
            _fail(path, f"duplicate slot {value!r}")
        seen.add(value)
    return seen


def _parameter_signature(parameterization: Any) -> tuple[tuple[str, str, Any], ...]:
    if not isinstance(parameterization, list):
        return tuple()
    out: list[tuple[str, str, Any]] = []
    for entry in parameterization:
        if isinstance(entry, dict):
            out.append((str(entry.get("name")), str(entry.get("value_kind")), entry.get("value")))
    return tuple(out)


def compact_slot_assignments_from_genome(genome: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the compact typed slot-assignment projection for one canonical genome."""

    schema.validate_circle_packing_genome_v1(genome)
    return {
        slot: {
            "module_key": genome["slots"][slot]["module_key"],
            "module_id": genome["slots"][slot]["module_id"],
            "parameterization": deepcopy(genome["slots"][slot]["parameterization"]),
        }
        for slot in schema.SLOT_ORDER
    }


def _validate_global_hypothesis(payload: Any) -> None:
    data = _ensure_exact_keys(payload, GLOBAL_HYPOTHESIS_KEYS, "global_hypothesis")
    for key in sorted(GLOBAL_HYPOTHESIS_KEYS):
        _ensure_str(data[key], f"global_hypothesis.{key}")


def _validate_render_fields(payload: Any) -> None:
    data = _ensure_exact_keys(payload, RENDER_FIELDS_KEYS, "render_fields")
    for key in ("interaction_notes", "compute_notes"):
        values = _ensure_str_list(data[key], f"render_fields.{key}")
        for idx, value in enumerate(values):
            _ensure_str(value, f"render_fields.{key}[{idx}]")
    _ensure_str(data["change_description"], "render_fields.change_description")


def _validate_slot_assignments(payload: Any) -> None:
    assignments = _ensure_dict(payload, "slot_assignments")
    if list(assignments.keys()) != schema.SLOT_ORDER:
        missing = sorted(set(schema.SLOT_ORDER).difference(assignments))
        extra = sorted(set(assignments).difference(schema.SLOT_ORDER))
        if missing:
            _fail("slot_assignments", f"missing slots: {', '.join(missing)}")
        if extra:
            _fail("slot_assignments", f"extra slots: {', '.join(extra)}")
        _fail("slot_assignments", "slots must be in canonical order")

    seen_module_ids: set[str] = set()
    for slot in schema.SLOT_ORDER:
        assignment = _ensure_exact_keys(assignments[slot], SLOT_ASSIGNMENT_KEYS, f"slot_assignments.{slot}")
        module_key = _ensure_str(assignment["module_key"], f"slot_assignments.{slot}.module_key")
        module_id = _ensure_str(assignment["module_id"], f"slot_assignments.{slot}.module_id")
        if module_id in seen_module_ids:
            _fail(f"slot_assignments.{slot}.module_id", f"duplicate module_id {module_id!r}")
        seen_module_ids.add(module_id)
        library.validate_slot_assignment(slot, module_key)
        library.materialize_module_instance(
            slot=slot,
            module_key=module_key,
            module_id=module_id,
            parameterization=assignment["parameterization"],
        )


def _validate_seed_metadata(payload: Any) -> None:
    data = _ensure_exact_keys(payload, set(SEED_METADATA.keys()), "operator_metadata")
    if data != SEED_METADATA:
        _fail("operator_metadata", f"seed metadata must be exactly {SEED_METADATA!r}")


def _validate_mutation_metadata(payload: Any) -> None:
    data = _ensure_exact_keys(payload, MUTATION_METADATA_KEYS, "operator_metadata")
    mutation_scope = _ensure_str(data["mutation_scope"], "operator_metadata.mutation_scope")
    if mutation_scope not in {"slot", "linkage_block"}:
        _fail("operator_metadata.mutation_scope", "must be slot or linkage_block")
    changed_slots = _ensure_str_list(data["changed_slots"], "operator_metadata.changed_slots")
    preserved_slots = _ensure_str_list(data["preserved_slots"], "operator_metadata.preserved_slots", allow_empty=True)
    changed = _canonical_slot_set(changed_slots, "operator_metadata.changed_slots")
    preserved = _canonical_slot_set(preserved_slots, "operator_metadata.preserved_slots")
    if changed.union(preserved) != set(schema.SLOT_ORDER) or changed.intersection(preserved):
        _fail("operator_metadata", "changed_slots and preserved_slots must partition the canonical slots")
    if mutation_scope == "slot" and len(changed_slots) != 1:
        _fail("operator_metadata.changed_slots", "slot mutation scope must change exactly one slot")
    if mutation_scope == "linkage_block" and list(changed_slots) not in schema.LINKAGE_GROUPS.values():
        _fail("operator_metadata.changed_slots", "linkage_block mutation scope must change exactly one linkage block")
    if data["parent_reference"] != "parent":
        _fail("operator_metadata.parent_reference", "must be exactly 'parent'")
    _ensure_str(data["change_rationale"], "operator_metadata.change_rationale")


def _validate_crossover_metadata(payload: Any) -> None:
    data = _ensure_exact_keys(payload, CROSSOVER_METADATA_KEYS, "operator_metadata")
    inheritance_mode = _ensure_str(data["inheritance_mode"], "operator_metadata.inheritance_mode")
    if inheritance_mode not in {"slotwise", "linkage_block"}:
        _fail("operator_metadata.inheritance_mode", "must be slotwise or linkage_block")
    slot_origins = _ensure_dict(data["slot_origins"], "operator_metadata.slot_origins")
    if list(slot_origins.keys()) != schema.SLOT_ORDER:
        _fail("operator_metadata.slot_origins", "must cover canonical slots in order")
    for slot in schema.SLOT_ORDER:
        origin = _ensure_str(slot_origins[slot], f"operator_metadata.slot_origins.{slot}")
        if origin not in {"primary", "secondary"}:
            _fail(f"operator_metadata.slot_origins.{slot}", "must be primary or secondary")
    primary_slots = _ensure_str_list(data["primary_slots"], "operator_metadata.primary_slots", allow_empty=True)
    secondary_slots = _ensure_str_list(data["secondary_slots"], "operator_metadata.secondary_slots", allow_empty=True)
    primary = _canonical_slot_set(primary_slots, "operator_metadata.primary_slots")
    secondary = _canonical_slot_set(secondary_slots, "operator_metadata.secondary_slots")
    if primary.union(secondary) != set(schema.SLOT_ORDER) or primary.intersection(secondary):
        _fail("operator_metadata", "primary_slots and secondary_slots must partition the canonical slots")
    expected_primary = {slot for slot, origin in slot_origins.items() if origin == "primary"}
    expected_secondary = {slot for slot, origin in slot_origins.items() if origin == "secondary"}
    if primary != expected_primary or secondary != expected_secondary:
        _fail("operator_metadata", "primary_slots and secondary_slots must match slot_origins")
    if inheritance_mode == "linkage_block":
        for slots, path in ((primary_slots, "primary_slots"), (secondary_slots, "secondary_slots")):
            if not _is_union_of_linkage_blocks(set(slots)):
                _fail(f"operator_metadata.{path}", "linkage_block inheritance must use full linkage blocks")
    _ensure_str(data["change_rationale"], "operator_metadata.change_rationale")


def _is_union_of_linkage_blocks(slots: set[str]) -> bool:
    remaining = set(slots)
    for block_slots in schema.LINKAGE_GROUPS.values():
        block = set(block_slots)
        if block.issubset(remaining):
            remaining.difference_update(block)
        elif block.intersection(remaining):
            return False
    return not remaining


def _validate_design_structure(response: dict[str, Any], operation: str) -> None:
    _reject_nulls_and_numbers(response)
    data = _ensure_exact_keys(response, DESIGN_TOP_LEVEL_KEYS, "$")
    expected = {
        "schema_name": DESIGN_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "task_family": TASK_FAMILY,
        "task_name": TASK_NAME,
        "operation": operation,
    }
    for key, value in expected.items():
        if data[key] != value:
            _fail(key, f"expected {value!r}, got {data[key]!r}")
    _validate_global_hypothesis(data["global_hypothesis"])
    _validate_slot_assignments(data["slot_assignments"])
    _validate_render_fields(data["render_fields"])
    if operation == "seed":
        _validate_seed_metadata(data["operator_metadata"])
    elif operation == "mutation":
        _validate_mutation_metadata(data["operator_metadata"])
    elif operation == "crossover":
        _validate_crossover_metadata(data["operator_metadata"])
    else:
        raise ValueError(f"Unsupported design operation {operation!r}.")


def _materialize_and_gate(organism_id: str, response: dict[str, Any]) -> dict[str, Any]:
    genome = schema.materialize_circle_packing_genome_v1(
        organism_id=organism_id,
        global_hypothesis=response["global_hypothesis"],
        slot_assignments=response["slot_assignments"],
        render_fields=response["render_fields"],
    )
    compatibility_report = compatibility.build_circle_packing_compatibility_report(genome)
    if not compatibility_report["hard_compatibility"]["is_compatible"]:
        reasons = "; ".join(
            f"{violation['rule_id']}: {violation['message']}"
            for violation in compatibility_report["hard_compatibility"]["violations"]
        )
        raise ValueError(f"Design response failed hard compatibility: {reasons}")
    checks = compatibility.build_circle_packing_functional_checks(genome)
    if not checks["checks_passed"]:
        reasons = "; ".join(
            f"{check['check_id']}: {check['message']}"
            for check in checks["checks"]
            if check["status"] == "failed"
        )
        raise ValueError(f"Design response failed functional checks: {reasons}")
    return genome


def validate_design_response(response: dict, operation: str) -> None:
    """Validate a strict typed seed/mutation/crossover design response."""

    _validate_design_structure(response, operation)
    _materialize_and_gate("__design_validation__", response)


def validate_mutation_response_against_parent(response: dict, parent_genome: dict) -> None:
    """Validate mutation metadata and slot preservation against the parent genome."""

    schema.validate_circle_packing_genome_v1(parent_genome)
    _validate_design_structure(response, "mutation")
    parent_assignments = compact_slot_assignments_from_genome(parent_genome)
    changed_slots = set(response["operator_metadata"]["changed_slots"])
    preserved_slots = set(response["operator_metadata"]["preserved_slots"])
    actual_changed = {
        slot
        for slot in schema.SLOT_ORDER
        if _assignment_signature(response["slot_assignments"][slot]) != _assignment_signature(parent_assignments[slot])
    }
    if actual_changed != changed_slots:
        _fail("operator_metadata.changed_slots", f"declared {sorted(changed_slots)!r} but actual changed slots are {sorted(actual_changed)!r}")
    for slot in preserved_slots:
        if _assignment_signature(response["slot_assignments"][slot]) != _assignment_signature(parent_assignments[slot]):
            _fail("operator_metadata.preserved_slots", f"slot {slot!r} differs from parent despite preservation claim")
    _materialize_and_gate("__design_validation__", response)


def validate_crossover_response_against_parents(
    response: dict,
    *,
    primary_parent_genome: dict,
    secondary_parent_genome: dict,
) -> None:
    """Validate crossover slot origins against both parent genomes."""

    schema.validate_circle_packing_genome_v1(primary_parent_genome)
    schema.validate_circle_packing_genome_v1(secondary_parent_genome)
    _validate_design_structure(response, "crossover")
    primary = compact_slot_assignments_from_genome(primary_parent_genome)
    secondary = compact_slot_assignments_from_genome(secondary_parent_genome)
    for slot in schema.SLOT_ORDER:
        origin = response["operator_metadata"]["slot_origins"][slot]
        source = primary if origin == "primary" else secondary
        candidate = response["slot_assignments"][slot]
        if candidate["module_key"] != source[slot]["module_key"]:
            _fail("operator_metadata.slot_origins", f"slot {slot!r} does not use the declared {origin} parent module")
        if _parameter_signature(candidate["parameterization"]) != _parameter_signature(source[slot]["parameterization"]):
            _fail("operator_metadata.slot_origins", f"slot {slot!r} does not use the declared {origin} parent parameters")
    _materialize_and_gate("__design_validation__", response)


def _assignment_signature(assignment: dict[str, Any]) -> tuple[str, tuple[tuple[str, str, Any], ...]]:
    return (
        str(assignment.get("module_key")),
        _parameter_signature(assignment.get("parameterization")),
    )


def materialize_genome_from_design_response(
    *,
    organism_id: str,
    response: dict,
) -> dict:
    """Materialize and gate a canonical genome from a typed design response."""

    operation = _ensure_str(response.get("operation"), "operation")
    _validate_design_structure(response, operation)
    return _materialize_and_gate(organism_id, response)


def parse_seed_design_response(raw_text: str, *, organism_id: str = "__design_validation__") -> dict:
    response = _parse_json_object(raw_text)
    validate_design_response(response, "seed")
    _materialize_and_gate(organism_id, response)
    return response


def parse_mutation_design_response(
    raw_text: str,
    *,
    parent_genome: dict | None = None,
    organism_id: str = "__design_validation__",
) -> dict:
    response = _parse_json_object(raw_text)
    if parent_genome is None:
        validate_design_response(response, "mutation")
    else:
        validate_mutation_response_against_parent(response, parent_genome)
    _materialize_and_gate(organism_id, response)
    return response


def parse_crossover_design_response(
    raw_text: str,
    *,
    primary_parent_genome: dict | None = None,
    secondary_parent_genome: dict | None = None,
    organism_id: str = "__design_validation__",
) -> dict:
    response = _parse_json_object(raw_text)
    if primary_parent_genome is None or secondary_parent_genome is None:
        validate_design_response(response, "crossover")
    else:
        validate_crossover_response_against_parents(
            response,
            primary_parent_genome=primary_parent_genome,
            secondary_parent_genome=secondary_parent_genome,
        )
    _materialize_and_gate(organism_id, response)
    return response


def _validate_novelty_shape(response: dict[str, Any], operation: str) -> None:
    _reject_nulls_and_numbers(response)
    data = _ensure_exact_keys(response, NOVELTY_TOP_LEVEL_KEYS, "$")
    expected = {
        "schema_name": NOVELTY_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "task_family": TASK_FAMILY,
        "task_name": TASK_NAME,
        "operation": operation,
    }
    for key, value in expected.items():
        if data[key] != value:
            _fail(key, f"expected {value!r}, got {data[key]!r}")
    verdict = _ensure_str(data["verdict"], "verdict")
    if verdict not in {"NOVELTY_ACCEPTED", "NOVELTY_REJECTED"}:
        _fail("verdict", "must be NOVELTY_ACCEPTED or NOVELTY_REJECTED")
    differences = data["supported_differences"]
    if not isinstance(differences, list):
        _fail("supported_differences", "must be a list")
    for idx, difference in enumerate(differences):
        item = _ensure_exact_keys(difference, NOVELTY_DIFFERENCE_KEYS, f"supported_differences[{idx}]")
        kind = _ensure_str(item["kind"], f"supported_differences[{idx}].kind")
        if kind not in NOVELTY_DIFFERENCE_KINDS:
            _fail(f"supported_differences[{idx}].kind", "has invalid novelty difference kind")
        slot = _ensure_str(item["slot"], f"supported_differences[{idx}].slot", allow_empty=True)
        if slot and slot not in schema.SLOT_ORDER:
            _fail(f"supported_differences[{idx}].slot", f"unknown slot {slot!r}")
        _ensure_str(item["message"], f"supported_differences[{idx}].message")
    rejections = data["rejection_reasons"]
    if not isinstance(rejections, list):
        _fail("rejection_reasons", "must be a list")
    for idx, reason in enumerate(rejections):
        item = _ensure_exact_keys(reason, NOVELTY_REJECTION_KEYS, f"rejection_reasons[{idx}]")
        _ensure_str(item["reason_id"], f"rejection_reasons[{idx}].reason_id")
        _ensure_str(item["message"], f"rejection_reasons[{idx}].message")
    if verdict == "NOVELTY_ACCEPTED":
        if rejections:
            _fail("rejection_reasons", "accepted novelty must have an empty rejection list")
        if not differences:
            _fail("supported_differences", "accepted novelty must include structural differences")
    if verdict == "NOVELTY_REJECTED" and not rejections:
        _fail("rejection_reasons", "rejected novelty must include at least one rejection reason")


def validate_novelty_response(response: dict, operation: str) -> None:
    """Validate a strict typed novelty verdict response."""

    _validate_novelty_shape(response, operation)


def _mutation_structural_difference_support(parent_genome: dict, candidate_response: dict) -> set[tuple[str, str]]:
    parent = compact_slot_assignments_from_genome(parent_genome)
    candidate = candidate_response["slot_assignments"]
    changed_slots = [
        slot
        for slot in schema.SLOT_ORDER
        if _assignment_signature(candidate[slot]) != _assignment_signature(parent[slot])
    ]
    supported: set[tuple[str, str]] = set()
    for slot in changed_slots:
        if candidate[slot]["module_key"] != parent[slot]["module_key"]:
            supported.add(("slot_change", slot))
        if _parameter_signature(candidate[slot]["parameterization"]) != _parameter_signature(parent[slot]["parameterization"]):
            supported.add(("parameter_change", slot))
    if changed_slots in schema.LINKAGE_GROUPS.values():
        for slot in changed_slots:
            supported.add(("linkage_block_change", slot))
    return supported


def validate_mutation_novelty_against_structure(
    response: dict,
    *,
    parent_genome: dict,
    candidate_response: dict,
) -> None:
    validate_novelty_response(response, "mutation")
    if response["verdict"] == "NOVELTY_REJECTED":
        return
    supported = _mutation_structural_difference_support(parent_genome, candidate_response)
    for difference in response["supported_differences"]:
        key = (difference["kind"], difference["slot"])
        if key not in supported and difference["kind"] != "linkage_block_change":
            _fail("supported_differences", f"unsupported mutation novelty claim {key!r}")
    if not supported:
        _fail("supported_differences", "accepted mutation novelty is unsupported by slot or parameter differences")


def validate_crossover_novelty_against_structure(
    response: dict,
    *,
    primary_parent_genome: dict,
    secondary_parent_genome: dict,
    candidate_response: dict,
) -> None:
    validate_novelty_response(response, "crossover")
    if response["verdict"] == "NOVELTY_REJECTED":
        return
    primary = compact_slot_assignments_from_genome(primary_parent_genome)
    secondary = compact_slot_assignments_from_genome(secondary_parent_genome)
    candidate = candidate_response["slot_assignments"]
    primary_equal = all(_assignment_signature(candidate[slot]) == _assignment_signature(primary[slot]) for slot in schema.SLOT_ORDER)
    secondary_equal = all(_assignment_signature(candidate[slot]) == _assignment_signature(secondary[slot]) for slot in schema.SLOT_ORDER)
    primary_slots = set(candidate_response["operator_metadata"]["primary_slots"])
    secondary_slots = set(candidate_response["operator_metadata"]["secondary_slots"])
    if primary_equal or secondary_equal or not primary_slots or not secondary_slots:
        _fail("supported_differences", "accepted crossover novelty must be a nontrivial mixed-parent structure")
    supported = {("recombination_structure", "")}
    for slot in secondary_slots:
        supported.add(("slot_change", slot))
    for difference in response["supported_differences"]:
        key = (difference["kind"], difference["slot"])
        if key not in supported:
            _fail("supported_differences", f"unsupported crossover novelty claim {key!r}")


def parse_mutation_novelty_response(
    raw_text: str,
    *,
    parent_genome: dict | None = None,
    candidate_response: dict | None = None,
) -> dict:
    response = _parse_json_object(raw_text)
    if parent_genome is None or candidate_response is None:
        validate_novelty_response(response, "mutation")
    else:
        validate_mutation_novelty_against_structure(
            response,
            parent_genome=parent_genome,
            candidate_response=candidate_response,
        )
    return response


def parse_crossover_novelty_response(
    raw_text: str,
    *,
    primary_parent_genome: dict | None = None,
    secondary_parent_genome: dict | None = None,
    candidate_response: dict | None = None,
) -> dict:
    response = _parse_json_object(raw_text)
    if primary_parent_genome is None or secondary_parent_genome is None or candidate_response is None:
        validate_novelty_response(response, "crossover")
    else:
        validate_crossover_novelty_against_structure(
            response,
            primary_parent_genome=primary_parent_genome,
            secondary_parent_genome=secondary_parent_genome,
            candidate_response=candidate_response,
        )
    return response
