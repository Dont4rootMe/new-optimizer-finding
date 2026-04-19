"""Shared fixtures for circle-packing genome tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from experiments.circle_packing_shinka._runtime import genome_schema_v1 as schema


_HYPOTHESES = {
    "layout": "Use a stable geometric scaffold as the initial support.",
    "selection": "Order candidate centers by their expected contribution to balanced density.",
    "radius_init": "Start from conservative radii that preserve room for coordinated growth.",
    "growth": "Grow radii through a balanced policy that keeps local pressure visible.",
    "conflict": "Represent overlap pressure as a declared feasibility signal.",
    "repair": "Resolve infeasibility while preserving the intended geometric scaffold.",
    "boundary": "Treat square boundary pressure as a separate feasibility signal.",
    "termination": "Stop when feasibility and packing quality have stabilized.",
}


def _state_for_slot(slot: str) -> tuple[list[str], list[str]]:
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


def valid_circle_packing_genome(*, organism_id: str = "org_alpha") -> dict[str, Any]:
    slots: dict[str, Any] = {}
    for slot in schema.SLOT_ORDER:
        reads_state, writes_state = _state_for_slot(slot)
        linkage_group = schema.linkage_group_for_slot(slot)
        slots[slot] = {
            "module_id": f"{organism_id}_{slot}",
            "slot": slot,
            "module_type": schema.SLOT_TO_MODULE_TYPE[slot],
            "family": f"{linkage_group}_family",
            "hypothesis": _HYPOTHESES[slot],
            "reads_state": reads_state,
            "writes_state": writes_state,
            "parameterization": [],
            "preconditions": [f"The {slot} module receives its declared abstract inputs."],
            "postconditions": [f"The {slot} module writes its declared abstract outputs."],
            "invariants": ["The module preserves hypothesis-level geometric intent."],
            "assumptions": [f"The {slot} policy remains compatible with neighboring slots."],
            "expected_effects": [f"The {slot} policy improves coherent feasible packing behavior."],
            "failure_modes": [f"The {slot} policy can fail when geometric pressure is misclassified."],
            "compatibility_tags": [f"{slot}_interface"],
            "linkage_group": linkage_group,
        }

    genome = {
        "schema_name": schema.SCHEMA_NAME,
        "schema_version": schema.SCHEMA_VERSION,
        "task_family": schema.TASK_FAMILY,
        "task_name": schema.TASK_NAME,
        "representation": schema.REPRESENTATION,
        "organism_id": organism_id,
        "slot_order": list(schema.SLOT_ORDER),
        "linkage_groups": schema.canonical_linkage_group_list(),
        "global_hypothesis": {
            "title": "Segmented circle packing hypothesis",
            "core_claim": "A segmented geometric construction can preserve feasibility while improving density.",
            "expected_advantage": "Declared interfaces make placement and repair responsibilities explicit.",
            "novelty_statement": "The organism fixes a typed segmented hypothesis before implementation.",
        },
        "slots": slots,
        "render_fields": {
            "interaction_notes": [
                "Placement and dynamics modules exchange only declared abstract state.",
                "Repair decisions preserve the declared layout intent.",
            ],
            "compute_notes": [
                "The realization should remain deterministic and lightweight.",
                "The implementation should avoid unnecessary search loops.",
            ],
            "change_description": "The organism fixes a typed segmented hypothesis before implementation.",
        },
    }
    return deepcopy(genome)
