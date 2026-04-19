"""Shared fixtures for circle-packing genome tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from experiments.circle_packing_shinka._runtime import genome_schema_v1 as schema
from experiments.circle_packing_shinka._runtime import module_library_v1 as library


def valid_global_hypothesis() -> dict[str, str]:
    return {
        "title": "Segmented circle packing hypothesis",
        "core_claim": "A library-backed geometric construction can preserve feasibility while improving density.",
        "expected_advantage": "Declared modules make placement and repair responsibilities explicit.",
        "novelty_statement": "The organism fixes a typed module composition before implementation.",
    }


def valid_render_fields() -> dict[str, Any]:
    return {
        "interaction_notes": [
            "Placement and dynamics modules exchange only declared abstract state.",
            "Repair decisions preserve the declared layout intent.",
        ],
        "compute_notes": [
            "The realization should remain deterministic and lightweight.",
            "The implementation should avoid unnecessary search loops.",
        ],
        "change_description": "The organism fixes a typed module composition before implementation.",
    }


def valid_slot_assignments(*, organism_id: str = "org_alpha") -> dict[str, Any]:
    assignments = {
        "layout": "layout_triangular_lattice",
        "selection": "selection_center_outward",
        "radius_init": "radius_init_lattice_derived",
        "growth": "growth_density_scaled_additive",
        "conflict": "conflict_overlap_plus_boundary_penetration",
        "repair": "repair_pairwise_repulsion",
        "boundary": "boundary_repulsive_margin",
        "termination": "termination_no_violation_and_no_gain",
    }
    return {
        slot: {
            "module_key": module_key,
            "module_id": f"{organism_id}_{slot}",
            "parameterization": library.default_parameterization_for_module(module_key),
        }
        for slot, module_key in assignments.items()
    }


def structured_role_slot_assignments(*, organism_id: str = "org_role") -> dict[str, Any]:
    assignments = {
        "layout": "layout_boundary_ring_interior_fill",
        "selection": "selection_boundary_first",
        "radius_init": "radius_init_role_based_boundary_vs_interior",
        "growth": "growth_role_conditioned_growth",
        "conflict": "conflict_overlap_plus_boundary_penetration",
        "repair": "repair_bilevel_positions_then_radii",
        "boundary": "boundary_role_based_edge_protection",
        "termination": "termination_no_violation_and_no_gain",
    }
    return {
        slot: {
            "module_key": module_key,
            "module_id": f"{organism_id}_{slot}",
            "parameterization": library.default_parameterization_for_module(module_key),
        }
        for slot, module_key in assignments.items()
    }


def corrective_slot_assignments(*, organism_id: str = "org_corrective") -> dict[str, Any]:
    assignments = {
        "layout": "layout_jittered_low_discrepancy",
        "selection": "selection_conflict_priority_order",
        "radius_init": "radius_init_local_clearance_based",
        "growth": "growth_density_scaled_additive",
        "conflict": "conflict_constraint_graph",
        "repair": "repair_sequential_worst_first",
        "boundary": "boundary_repulsive_margin",
        "termination": "termination_repair_budget_exhaustion",
    }
    return {
        slot: {
            "module_key": module_key,
            "module_id": f"{organism_id}_{slot}",
            "parameterization": library.default_parameterization_for_module(module_key),
        }
        for slot, module_key in assignments.items()
    }


def valid_circle_packing_genome(*, organism_id: str = "org_alpha") -> dict[str, Any]:
    genome = schema.materialize_circle_packing_genome_v1(
        organism_id=organism_id,
        global_hypothesis=valid_global_hypothesis(),
        slot_assignments=valid_slot_assignments(organism_id=organism_id),
        render_fields=valid_render_fields(),
    )
    return deepcopy(genome)


def structured_role_genome(*, organism_id: str = "org_role") -> dict[str, Any]:
    genome = schema.materialize_circle_packing_genome_v1(
        organism_id=organism_id,
        global_hypothesis=valid_global_hypothesis(),
        slot_assignments=structured_role_slot_assignments(organism_id=organism_id),
        render_fields=valid_render_fields(),
    )
    return deepcopy(genome)


def corrective_genome(*, organism_id: str = "org_corrective") -> dict[str, Any]:
    genome = schema.materialize_circle_packing_genome_v1(
        organism_id=organism_id,
        global_hypothesis=valid_global_hypothesis(),
        slot_assignments=corrective_slot_assignments(organism_id=organism_id),
        render_fields=valid_render_fields(),
    )
    return deepcopy(genome)
