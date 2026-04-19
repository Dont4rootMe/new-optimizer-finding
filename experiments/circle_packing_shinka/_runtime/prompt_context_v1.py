"""Generated prompt context for circle_packing_shinka typed genome v1."""

from __future__ import annotations

import json
from typing import Any

from experiments.circle_packing_shinka._runtime import compatibility_contract_v1 as compatibility
from experiments.circle_packing_shinka._runtime import design_prompt_contract_v1 as design_contract
from experiments.circle_packing_shinka._runtime import genome_schema_v1 as schema
from experiments.circle_packing_shinka._runtime import module_library_v1 as library


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _section(title: str, body: str) -> str:
    return f"=== {title} ===\n{body.strip()}"


def _slot_order_digest() -> str:
    return "\n".join(f"{idx}. {slot}" for idx, slot in enumerate(schema.SLOT_ORDER, start=1))


def _module_library_digest() -> str:
    lines: list[str] = []
    module_library = library.get_circle_packing_module_library_v1()
    for slot in schema.SLOT_ORDER:
        lines.append(f"{slot}:")
        for module_key, definition in module_library[slot].items():
            lines.append(
                "- "
                f"{module_key} | type={definition['module_type']} | "
                f"family={definition['family']} | linkage={definition['linkage_group']} | "
                f"inheritance={definition['inheritance_unit']}"
            )
    return "\n".join(lines)


def _parameter_options_digest() -> str:
    lines: list[str] = []
    module_library = library.get_circle_packing_module_library_v1()
    for slot in schema.SLOT_ORDER:
        lines.append(f"{slot}:")
        for module_key, definition in module_library[slot].items():
            params = definition["allowed_parameters"]
            if not params:
                lines.append(f"- {module_key}: no parameters")
                continue
            rendered_params = []
            for param in params:
                rendered_params.append(
                    f"{param['name']}:{param['value_kind']} in {_json(param['allowed_values'])}"
                )
            lines.append(f"- {module_key}: " + "; ".join(rendered_params))
    return "\n".join(lines)


def _hard_compatibility_digest() -> str:
    return "\n".join(
        [
            "Hard compatibility is fail-closed.",
            "HC-IF-001: every read must be produced by an earlier slot.",
            "HC-IF-002: canonical minimum interface obligations must hold.",
            "HC-IF-003: no slot may read state produced only by a later slot.",
            "HC-IF-004: no module may have empty reads and empty writes simultaneously.",
            "HC-LG-002: growth_role_conditioned_growth needs role-aware support.",
            "HC-LG-003: conflict_constraint_graph needs nontrivial repair.",
            "HC-LG-004: repair_bilevel_positions_then_radii needs enriched conflict.",
            "HC-TR-001: termination_repair_budget_exhaustion needs repair-budget-compatible repair.",
            "Use only module combinations that can pass compatibility_report.json.",
        ]
    )


def _functional_check_digest() -> str:
    return "\n".join(
        [
            "FC-001: executable stage chain exists from layout through termination.",
            "FC-002: conflict, repair, and boundary provide a feasibility path.",
            "FC-003: avoid conservative start plus shrink repair plus conservative stopping.",
            "FC-004: role-aware modules need role-aware or structured support.",
            "FC-005: termination must observe enough state for its stopping notion.",
            "Use only designs that can pass functional_checks.json.",
        ]
    )


def _design_output_contract(operation: str) -> str:
    base = {
        "schema_name": design_contract.DESIGN_SCHEMA_NAME,
        "schema_version": design_contract.SCHEMA_VERSION,
        "task_family": design_contract.TASK_FAMILY,
        "task_name": design_contract.TASK_NAME,
        "operation": operation,
        "global_hypothesis": {
            "title": "string",
            "core_claim": "string",
            "expected_advantage": "string",
            "novelty_statement": "string",
        },
        "slot_assignments": {
            slot: {
                "module_key": "library_module_key",
                "module_id": f"{slot}_module_id",
                "parameterization": [
                    {"name": "allowed_parameter_name", "value_kind": "allowed_kind", "value": "allowed_value"}
                ],
            }
            for slot in schema.SLOT_ORDER
        },
        "render_fields": {
            "interaction_notes": ["string", "string"],
            "compute_notes": ["string", "string"],
            "change_description": "string",
        },
        "operator_metadata": _operator_metadata_contract(operation),
    }
    return (
        "Return only valid JSON.\n"
        "Do not use markdown fences.\n"
        "Do not add commentary before or after the JSON object.\n"
        "Do not emit CORE_GENES, INTERACTION_NOTES, COMPUTE_NOTES, or CHANGE_DESCRIPTION as markdown sections.\n"
        "Those human-readable fields will be rendered by the pipeline after validation.\n\n"
        + _json(base)
    )


def _operator_metadata_contract(operation: str) -> dict[str, Any]:
    if operation == "seed":
        return dict(design_contract.SEED_METADATA)
    if operation == "mutation":
        return {
            "mutation_scope": "slot | linkage_block",
            "changed_slots": ["slot_name"],
            "preserved_slots": ["slot_name"],
            "parent_reference": "parent",
            "change_rationale": "string",
        }
    if operation == "crossover":
        return {
            "inheritance_mode": "slotwise | linkage_block",
            "slot_origins": {slot: "primary | secondary" for slot in schema.SLOT_ORDER},
            "primary_slots": ["slot_name"],
            "secondary_slots": ["slot_name"],
            "change_rationale": "string",
        }
    raise ValueError(f"Unsupported operation {operation!r}.")


def _novelty_output_contract(operation: str) -> str:
    return (
        "Return only valid JSON.\n"
        "Do not use markdown fences.\n"
        "Do not add commentary before or after the JSON object.\n"
        "Judge structural typed novelty, not prose wording.\n\n"
        + _json(
            {
                "schema_name": design_contract.NOVELTY_SCHEMA_NAME,
                "schema_version": design_contract.SCHEMA_VERSION,
                "task_family": design_contract.TASK_FAMILY,
                "task_name": design_contract.TASK_NAME,
                "operation": operation,
                "verdict": "NOVELTY_ACCEPTED | NOVELTY_REJECTED",
                "supported_differences": [
                    {
                        "kind": "slot_change | linkage_block_change | recombination_structure | parameter_change",
                        "slot": "slot_name_or_empty_string",
                        "message": "string",
                    }
                ],
                "rejection_reasons": [
                    {
                        "reason_id": "string",
                        "message": "string",
                    }
                ],
            }
        )
    )


def build_typed_hypothesis_prompt_context() -> str:
    """Return a deterministic generated context for typed hypothesis prompts."""

    return "\n\n".join(
        [
            _section("SLOT ORDER", _slot_order_digest()),
            _section("MODULE LIBRARY DIGEST", _module_library_digest()),
            _section("PARAMETER OPTIONS DIGEST", _parameter_options_digest()),
            _section("HARD COMPATIBILITY DIGEST", _hard_compatibility_digest()),
            _section("FUNCTIONAL CHECK DIGEST", _functional_check_digest()),
        ]
    )


def build_seed_prompt_context() -> str:
    """Return generated seed prompt sections after the task section."""

    return "\n\n".join(
        [
            _section("SLOT ORDER", _slot_order_digest()),
            _section("MODULE LIBRARY DIGEST", _module_library_digest()),
            _section("PARAMETER OPTIONS DIGEST", _parameter_options_digest()),
            _section("HARD COMPATIBILITY DIGEST", _hard_compatibility_digest()),
            _section("FUNCTIONAL CHECK DIGEST", _functional_check_digest()),
            _section("OUTPUT CONTRACT", _design_output_contract("seed")),
        ]
    )


def build_mutation_prompt_context() -> str:
    """Return generated mutation prompt sections after parent context."""

    return "\n\n".join(
        [
            _section("MODULE LIBRARY DIGEST", _module_library_digest()),
            _section("HARD COMPATIBILITY DIGEST", _hard_compatibility_digest()),
            _section("FUNCTIONAL CHECK DIGEST", _functional_check_digest()),
            _section(
                "TASK",
                "Edit the typed child slot draft into a full mutation response. "
                "Preserve all unchanged slots exactly. Change one slot or one full linkage block only. "
                "Return the final full slot assignment, not a diff-only artifact.",
            ),
            _section("OUTPUT CONTRACT", _design_output_contract("mutation")),
        ]
    )


def build_crossover_prompt_context() -> str:
    """Return generated crossover prompt sections after parent context."""

    return "\n\n".join(
        [
            _section("MODULE LIBRARY DIGEST", _module_library_digest()),
            _section("HARD COMPATIBILITY DIGEST", _hard_compatibility_digest()),
            _section("FUNCTIONAL CHECK DIGEST", _functional_check_digest()),
            _section(
                "TASK",
                "Produce a typed recombination, not a prose merge. Every slot must come from primary or secondary, "
                "slot_origins must match the chosen source, and no module may appear unless it exists in the "
                "designated source parent for that slot.",
            ),
            _section("OUTPUT CONTRACT", _design_output_contract("crossover")),
        ]
    )


def build_novelty_prompt_context() -> str:
    """Return generated typed novelty criteria and output contract."""

    criteria = "\n".join(
        [
            "Mutation novelty may be supported only by slot changes, linkage-block changes, allowed parameter changes, or faithful preservation of a structurally distinct selected child draft.",
            "Crossover novelty may be supported only by nontrivial mixed-parent recombination, full linkage-block inheritance from both parents, or a child not substantively equal to either parent.",
            "Reject unsupported rhetorical novelty claims.",
        ]
    )
    return "\n\n".join(
        [
            _section("TYPED NOVELTY CRITERIA", criteria),
            _section("OUTPUT CONTRACT", _novelty_output_contract("mutation | crossover")),
        ]
    )


def build_mutation_novelty_prompt_context() -> str:
    return "\n\n".join(
        [
            _section(
                "TASK",
                "Evaluate typed structural novelty relative to the parent. Do not judge prose quality.",
            ),
            _section("OUTPUT CONTRACT", _novelty_output_contract("mutation")),
        ]
    )


def build_crossover_novelty_prompt_context() -> str:
    return "\n\n".join(
        [
            _section(
                "TASK",
                "Evaluate whether the child is a nontrivial typed recombination of the parents. Do not judge prose quality.",
            ),
            _section("OUTPUT CONTRACT", _novelty_output_contract("crossover")),
        ]
    )


def format_slot_assignments_for_prompt(genome: dict[str, Any]) -> str:
    """Format compact slot assignments from a validated genome."""

    return _json(design_contract.compact_slot_assignments_from_genome(genome))


def format_genome_summary_for_prompt(genome: dict[str, Any]) -> str:
    """Format a compact hypothesis summary for typed prompts."""

    schema.validate_circle_packing_genome_v1(genome)
    modules = {
        slot: genome["slots"][slot]["module_key"]
        for slot in schema.SLOT_ORDER
    }
    return _json(
        {
            "organism_id": genome["organism_id"],
            "global_hypothesis": genome["global_hypothesis"],
            "modules": modules,
            "change_description": genome["render_fields"]["change_description"],
        }
    )


def format_excluded_modules_for_prompt(removed_gene_pool: list[str], genome: dict[str, Any]) -> str:
    """Format excluded mutation context without treating prose as authoritative."""

    schema.validate_circle_packing_genome_v1(genome)
    return _json(
        {
            "sampling_context": list(removed_gene_pool),
            "note": "These strings are sampling hints only; the canonical parent slot assignments are authoritative.",
        }
    )

