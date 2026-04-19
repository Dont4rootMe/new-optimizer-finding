"""Closed canonical module library for circle_packing_shinka genome v1."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any


LIBRARY_NAME = "circle_packing_module_library"
LIBRARY_VERSION = "1.0"
TASK_FAMILY = "circle_packing_shinka"
TASK_NAME = "unit_square_26"

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

LINKAGE_GROUPS = {
    "placement_block": ["layout", "selection", "radius_init"],
    "dynamics_block": ["growth", "conflict", "repair", "boundary"],
    "control_block": ["termination"],
}

STATE_BY_SLOT = {
    "layout": {
        "reads_state": ["packing_summary"],
        "writes_state": ["candidate_centers"],
    },
    "selection": {
        "reads_state": ["candidate_centers"],
        "writes_state": ["ordered_centers"],
    },
    "radius_init": {
        "reads_state": ["ordered_centers"],
        "writes_state": ["current_radii"],
    },
    "growth": {
        "reads_state": ["current_radii"],
        "writes_state": ["proposed_radii"],
    },
    "conflict": {
        "reads_state": ["proposed_radii"],
        "writes_state": ["conflict_report"],
    },
    "repair": {
        "reads_state": ["conflict_report"],
        "writes_state": ["feasibility_status"],
    },
    "boundary": {
        "reads_state": ["proposed_radii"],
        "writes_state": ["boundary_report"],
    },
    "termination": {
        "reads_state": ["packing_summary", "feasibility_status", "boundary_report"],
        "writes_state": ["termination_signal"],
    },
}

REQUIRED_MODULE_KEYS = {
    "layout": [
        "layout_triangular_lattice",
        "layout_jittered_low_discrepancy",
        "layout_boundary_ring_interior_fill",
        "layout_reflective_symmetric_seed",
    ],
    "selection": [
        "selection_center_outward",
        "selection_boundary_first",
        "selection_uniform_scan_order",
        "selection_conflict_priority_order",
    ],
    "radius_init": [
        "radius_init_uniform_small",
        "radius_init_lattice_derived",
        "radius_init_role_based_boundary_vs_interior",
        "radius_init_local_clearance_based",
    ],
    "growth": [
        "growth_uniform_additive",
        "growth_uniform_multiplicative",
        "growth_density_scaled_additive",
        "growth_role_conditioned_growth",
    ],
    "conflict": [
        "conflict_pairwise_overlap",
        "conflict_overlap_plus_boundary_penetration",
        "conflict_worst_violation_first",
        "conflict_constraint_graph",
    ],
    "repair": [
        "repair_pairwise_repulsion",
        "repair_boundary_repulsion",
        "repair_sequential_worst_first",
        "repair_shrink_on_persistent_failure",
        "repair_bilevel_positions_then_radii",
    ],
    "boundary": [
        "boundary_hard_containment",
        "boundary_repulsive_margin",
        "boundary_edge_aware_inset",
        "boundary_role_based_edge_protection",
    ],
    "termination": [
        "termination_fixed_stability",
        "termination_no_violation_and_no_gain",
        "termination_plateau_detection",
        "termination_repair_budget_exhaustion",
    ],
}

LINKAGE_BLOCK_MODULE_KEYS = {
    "layout_triangular_lattice",
    "layout_boundary_ring_interior_fill",
    "radius_init_lattice_derived",
    "radius_init_role_based_boundary_vs_interior",
    "growth_role_conditioned_growth",
    "boundary_role_based_edge_protection",
    "repair_bilevel_positions_then_radii",
    "conflict_constraint_graph",
}

INHERITANCE_UNITS = {"slot", "linkage_block"}
VALUE_KINDS = {"categorical_token", "ordinal_token", "boolean", "relation_token"}

MODULE_DEFINITION_KEYS = {
    "module_key",
    "slot",
    "module_type",
    "family",
    "display_name",
    "hypothesis_template",
    "reads_state",
    "writes_state",
    "required_preconditions",
    "guaranteed_postconditions",
    "invariants",
    "assumptions",
    "expected_effects",
    "failure_modes",
    "compatibility_tags",
    "allowed_parameters",
    "linkage_group",
    "inheritance_unit",
}

PARAMETER_SCHEMA_KEYS = {"name", "value_kind", "allowed_values", "required"}
PARAMETER_INSTANCE_KEYS = {"name", "value_kind", "value"}
MODULE_INSTANCE_KEYS = {
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

_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_TOKEN_RE = re.compile(r"^[a-z][a-z_]*$")


def _slot_linkage_group(slot: str) -> str:
    for group_id, slots in LINKAGE_GROUPS.items():
        if slot in slots:
            return group_id
    raise ValueError(f"Unknown slot {slot!r}.")


def _param(
    name: str,
    value_kind: str,
    allowed_values: list[str | bool],
    *,
    required: bool = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "value_kind": value_kind,
        "allowed_values": list(allowed_values),
        "required": required,
    }


def _module(
    *,
    module_key: str,
    slot: str,
    family: str,
    display_name: str,
    hypothesis_template: str,
    required_preconditions: list[str],
    guaranteed_postconditions: list[str],
    invariants: list[str],
    assumptions: list[str],
    expected_effects: list[str],
    failure_modes: list[str],
    compatibility_tags: list[str],
    allowed_parameters: list[dict[str, Any]],
) -> dict[str, Any]:
    inheritance_unit = "linkage_block" if module_key in LINKAGE_BLOCK_MODULE_KEYS else "slot"
    state = STATE_BY_SLOT[slot]
    return {
        "module_key": module_key,
        "slot": slot,
        "module_type": SLOT_TO_MODULE_TYPE[slot],
        "family": family,
        "display_name": display_name,
        "hypothesis_template": hypothesis_template,
        "reads_state": list(state["reads_state"]),
        "writes_state": list(state["writes_state"]),
        "required_preconditions": list(required_preconditions),
        "guaranteed_postconditions": list(guaranteed_postconditions),
        "invariants": list(invariants),
        "assumptions": list(assumptions),
        "expected_effects": list(expected_effects),
        "failure_modes": list(failure_modes),
        "compatibility_tags": list(compatibility_tags),
        "allowed_parameters": deepcopy(allowed_parameters),
        "linkage_group": _slot_linkage_group(slot),
        "inheritance_unit": inheritance_unit,
    }


MODULE_LIBRARY = {
    "layout": {
        "layout_triangular_lattice": _module(
            module_key="layout_triangular_lattice",
            slot="layout",
            family="triangular_lattice_layout",
            display_name="Triangular Lattice Layout",
            hypothesis_template="Use a triangular lattice scaffold as the initial geometric support.",
            required_preconditions=["Packing intent is represented as abstract placement state."],
            guaranteed_postconditions=["Candidate centers follow a repeating triangular scaffold."],
            invariants=["The scaffold keeps neighboring candidates evenly staggered."],
            assumptions=["Regular staggered geometry creates strong baseline spacing."],
            expected_effects=["Candidate centers begin near a dense and orderly arrangement."],
            failure_modes=["Regular structure can leave useful asymmetric gaps unused."],
            compatibility_tags=["regular_spacing", "staggered_geometry", "placement_block"],
            allowed_parameters=[
                _param("orientation_bias", "categorical_token", ["balanced", "interior_favoring", "boundary_favoring"]),
            ],
        ),
        "layout_jittered_low_discrepancy": _module(
            module_key="layout_jittered_low_discrepancy",
            slot="layout",
            family="low_discrepancy_layout",
            display_name="Jittered Low Discrepancy Layout",
            hypothesis_template="Use a low discrepancy scaffold with controlled jitter to avoid rigid symmetry traps.",
            required_preconditions=["Placement state can represent broadly distributed candidates."],
            guaranteed_postconditions=["Candidate centers cover the square with mild irregularity."],
            invariants=["Jitter remains hypothesis-level and does not encode raw random search."],
            assumptions=["Mild irregularity can expose placements hidden by regular grids."],
            expected_effects=["Candidate centers retain coverage while reducing repeated local conflicts."],
            failure_modes=["Too much irregularity can weaken coordinated packing structure."],
            compatibility_tags=["irregular_spacing", "coverage_layout", "placement_block"],
            allowed_parameters=[
                _param("jitter_style", "categorical_token", ["balanced", "spread_preserving", "conflict_escaping"]),
            ],
        ),
        "layout_boundary_ring_interior_fill": _module(
            module_key="layout_boundary_ring_interior_fill",
            slot="layout",
            family="boundary_ring_layout",
            display_name="Boundary Ring Interior Fill Layout",
            hypothesis_template="Place a protected boundary ring before filling the interior with compatible support.",
            required_preconditions=["Placement state distinguishes boundary pressure from interior support."],
            guaranteed_postconditions=["Candidate centers include boundary roles and interior roles."],
            invariants=["Boundary structure remains coupled to interior fill."],
            assumptions=["Protecting edge positions early can stabilize later radius growth."],
            expected_effects=["Candidate centers balance edge utilization with interior density."],
            failure_modes=["Boundary commitment can crowd the interior if roles are misbalanced."],
            compatibility_tags=["boundary_roles", "interior_fill", "placement_block"],
            allowed_parameters=[
                _param("ring_priority", "categorical_token", ["boundary_favoring", "interior_favoring", "balanced"]),
            ],
        ),
        "layout_reflective_symmetric_seed": _module(
            module_key="layout_reflective_symmetric_seed",
            slot="layout",
            family="reflective_symmetry_layout",
            display_name="Reflective Symmetric Seed Layout",
            hypothesis_template="Use reflected placement motifs to preserve symmetry while allowing interior variation.",
            required_preconditions=["Placement state can express mirrored candidate roles."],
            guaranteed_postconditions=["Candidate centers contain reflected structural motifs."],
            invariants=["Reflected roles remain paired at the hypothesis level."],
            assumptions=["Symmetric motifs reduce avoidable imbalance in the packing."],
            expected_effects=["Candidate centers start with coordinated mirror structure."],
            failure_modes=["Symmetry can resist beneficial asymmetric local improvements."],
            compatibility_tags=["reflective_symmetry", "paired_roles", "placement_block"],
            allowed_parameters=[
                _param("symmetry_mode", "categorical_token", ["mirror_balanced", "diagonal_reflective", "paired_reflective"]),
            ],
        ),
    },
    "selection": {
        "selection_center_outward": _module(
            module_key="selection_center_outward",
            slot="selection",
            family="center_outward_selection",
            display_name="Center Outward Selection",
            hypothesis_template="Order placements from interior support outward so central density anchors later choices.",
            required_preconditions=["Candidate centers are available as placement options."],
            guaranteed_postconditions=["Ordered centers prioritize interior support before outer roles."],
            invariants=["Ordering remains deterministic for the same candidate state."],
            assumptions=["Interior anchoring helps prevent late central crowding."],
            expected_effects=["Placement order stabilizes the densest region before resolving edges."],
            failure_modes=["Center emphasis can delay important boundary commitments."],
            compatibility_tags=["center_priority", "ordered_centers", "placement_block"],
            allowed_parameters=[
                _param("priority_bias", "categorical_token", ["center_first", "balanced", "density_first"]),
            ],
        ),
        "selection_boundary_first": _module(
            module_key="selection_boundary_first",
            slot="selection",
            family="boundary_first_selection",
            display_name="Boundary First Selection",
            hypothesis_template="Order boundary-sensitive candidates early so edge feasibility shapes the packing.",
            required_preconditions=["Candidate centers expose boundary-sensitive roles."],
            guaranteed_postconditions=["Ordered centers prioritize boundary roles before interior fill."],
            invariants=["Boundary priority remains explicit in ordering metadata."],
            assumptions=["Early edge commitments reduce later containment conflicts."],
            expected_effects=["Placement order protects edge capacity before interior pressure rises."],
            failure_modes=["Boundary priority can overconstrain interior density."],
            compatibility_tags=["boundary_priority", "edge_roles", "placement_block"],
            allowed_parameters=[
                _param("boundary_priority", "categorical_token", ["edge_first", "corner_first", "balanced"]),
            ],
        ),
        "selection_uniform_scan_order": _module(
            module_key="selection_uniform_scan_order",
            slot="selection",
            family="uniform_scan_selection",
            display_name="Uniform Scan Order Selection",
            hypothesis_template="Order candidates through a uniform scan that avoids adaptive placement bias.",
            required_preconditions=["Candidate centers can be traversed deterministically."],
            guaranteed_postconditions=["Ordered centers follow a stable scan pattern."],
            invariants=["Selection does not depend on conflict feedback."],
            assumptions=["A neutral scan preserves the layout hypothesis without early distortion."],
            expected_effects=["Placement order remains predictable and easy to audit."],
            failure_modes=["Neutral ordering may ignore obvious high-pressure conflicts."],
            compatibility_tags=["neutral_order", "stable_scan", "placement_block"],
            allowed_parameters=[
                _param("scan_style", "categorical_token", ["row_major", "column_major", "alternating"]),
            ],
        ),
        "selection_conflict_priority_order": _module(
            module_key="selection_conflict_priority_order",
            slot="selection",
            family="conflict_priority_selection",
            display_name="Conflict Priority Order Selection",
            hypothesis_template="Order candidates by anticipated conflict pressure so fragile placements resolve early.",
            required_preconditions=["Candidate centers can be scored by abstract pressure signals."],
            guaranteed_postconditions=["Ordered centers prioritize high-pressure candidates."],
            invariants=["Conflict priority uses declared abstract state rather than implementation details."],
            assumptions=["Resolving fragile positions early improves downstream feasibility."],
            expected_effects=["Placement order focuses attention on candidates likely to constrain radii."],
            failure_modes=["Pressure estimates can overfit early conflicts and harm global balance."],
            compatibility_tags=["pressure_priority", "conflict_aware_order", "placement_block"],
            allowed_parameters=[
                _param("conflict_signal", "categorical_token", ["overlap_first", "boundary_first", "combined_pressure"]),
            ],
        ),
    },
    "radius_init": {
        "radius_init_uniform_small": _module(
            module_key="radius_init_uniform_small",
            slot="radius_init",
            family="uniform_small_radius_init",
            display_name="Uniform Small Radius Initialization",
            hypothesis_template="Start all ordered centers with conservative uniform radii before later growth.",
            required_preconditions=["Ordered centers are available for radius assignment."],
            guaranteed_postconditions=["Current radii begin with a shared conservative role."],
            invariants=["Initial radius policy does not encode raw numeric constants."],
            assumptions=["A cautious common start reduces early infeasibility."],
            expected_effects=["Radius growth begins from a stable feasible baseline."],
            failure_modes=["Uniform starts can waste known boundary or interior differences."],
            compatibility_tags=["uniform_radius", "conservative_start", "placement_block"],
            allowed_parameters=[
                _param("initial_scale", "ordinal_token", ["low", "medium", "high"]),
            ],
        ),
        "radius_init_lattice_derived": _module(
            module_key="radius_init_lattice_derived",
            slot="radius_init",
            family="lattice_derived_radius_init",
            display_name="Lattice Derived Radius Initialization",
            hypothesis_template="Derive initial radii from scaffold spacing so geometry and size begin coupled.",
            required_preconditions=["Ordered centers preserve enough scaffold spacing information."],
            guaranteed_postconditions=["Current radii reflect relative spacing in the placement scaffold."],
            invariants=["Radius scale remains linked to layout geometry."],
            assumptions=["Spacing-derived radii reduce mismatch between placement and growth."],
            expected_effects=["Radius initialization starts closer to the intended dense arrangement."],
            failure_modes=["Spacing assumptions can be wrong after irregular ordering."],
            compatibility_tags=["spacing_coupled", "layout_radius_coupling", "placement_block"],
            allowed_parameters=[
                _param("derivation_bias", "categorical_token", ["spacing_limited", "density_scaled", "balanced"]),
            ],
        ),
        "radius_init_role_based_boundary_vs_interior": _module(
            module_key="radius_init_role_based_boundary_vs_interior",
            slot="radius_init",
            family="role_based_radius_init",
            display_name="Role Based Boundary Interior Radius Initialization",
            hypothesis_template="Assign initial radii by boundary and interior roles so edge pressure is explicit.",
            required_preconditions=["Ordered centers carry boundary and interior role information."],
            guaranteed_postconditions=["Current radii distinguish boundary roles from interior roles."],
            invariants=["Role-based radius differences remain abstract and finite."],
            assumptions=["Boundary and interior circles need different starting pressure."],
            expected_effects=["Radius initialization supports later role-conditioned feasibility handling."],
            failure_modes=["Wrong role assignment can lock in poor radius priorities."],
            compatibility_tags=["boundary_roles", "role_based_radius", "placement_block"],
            allowed_parameters=[
                _param("boundary_relation", "relation_token", ["boundary_smaller", "balanced", "interior_smaller"]),
            ],
        ),
        "radius_init_local_clearance_based": _module(
            module_key="radius_init_local_clearance_based",
            slot="radius_init",
            family="local_clearance_radius_init",
            display_name="Local Clearance Based Radius Initialization",
            hypothesis_template="Assign initial radii from local clearance estimates to reflect nearby crowding.",
            required_preconditions=["Ordered centers expose local neighborhood pressure."],
            guaranteed_postconditions=["Current radii vary according to abstract clearance."],
            invariants=["Clearance estimates remain hypothesis-level declarations."],
            assumptions=["Local clearance is a useful proxy for safe initial size."],
            expected_effects=["Radius initialization adapts to crowded and open regions."],
            failure_modes=["Local clearance can miss longer range packing constraints."],
            compatibility_tags=["clearance_based", "local_density", "placement_block"],
            allowed_parameters=[
                _param("clearance_bias", "categorical_token", ["conservative", "balanced", "aggressive"]),
            ],
        ),
    },
    "growth": {
        "growth_uniform_additive": _module(
            module_key="growth_uniform_additive",
            slot="growth",
            family="uniform_additive_growth",
            display_name="Uniform Additive Growth",
            hypothesis_template="Grow all current radii through a shared additive pressure policy.",
            required_preconditions=["Current radii are available for proposed growth."],
            guaranteed_postconditions=["Proposed radii increase through a common additive rule."],
            invariants=["Growth pressure is shared across roles."],
            assumptions=["Uniform additive growth keeps competition simple and auditable."],
            expected_effects=["Proposed radii advance evenly before conflict handling."],
            failure_modes=["Uniform growth can ignore local density differences."],
            compatibility_tags=["uniform_growth", "additive_pressure", "dynamics_block"],
            allowed_parameters=[
                _param("adaptivity_level", "ordinal_token", ["low", "medium", "high"]),
            ],
        ),
        "growth_uniform_multiplicative": _module(
            module_key="growth_uniform_multiplicative",
            slot="growth",
            family="uniform_multiplicative_growth",
            display_name="Uniform Multiplicative Growth",
            hypothesis_template="Grow all current radii through a shared multiplicative pressure policy.",
            required_preconditions=["Current radii are available for proportional growth."],
            guaranteed_postconditions=["Proposed radii preserve relative size relationships."],
            invariants=["Relative radius ordering remains stable during growth."],
            assumptions=["Proportional growth respects prior radius commitments."],
            expected_effects=["Proposed radii scale without erasing initial role differences."],
            failure_modes=["Proportional growth can amplify early radius mistakes."],
            compatibility_tags=["uniform_growth", "multiplicative_pressure", "dynamics_block"],
            allowed_parameters=[
                _param("growth_temper", "ordinal_token", ["low", "medium", "high"]),
            ],
        ),
        "growth_density_scaled_additive": _module(
            module_key="growth_density_scaled_additive",
            slot="growth",
            family="density_scaled_growth",
            display_name="Density Scaled Additive Growth",
            hypothesis_template="Grow radii with additive pressure scaled by local density signals.",
            required_preconditions=["Current radii can be related to local density pressure."],
            guaranteed_postconditions=["Proposed radii reflect density-sensitive growth."],
            invariants=["Density scaling remains tied to abstract state."],
            assumptions=["Sparse areas can accept stronger growth than crowded areas."],
            expected_effects=["Proposed radii allocate growth where space is more plausible."],
            failure_modes=["Density signals can understate hidden conflicts."],
            compatibility_tags=["density_scaled", "adaptive_growth", "dynamics_block"],
            allowed_parameters=[
                _param("density_response", "categorical_token", ["density_scaled", "balanced", "sparse_boosted"]),
            ],
        ),
        "growth_role_conditioned_growth": _module(
            module_key="growth_role_conditioned_growth",
            slot="growth",
            family="role_conditioned_growth",
            display_name="Role Conditioned Growth",
            hypothesis_template="Grow radii according to boundary and interior roles so structural duties remain coupled.",
            required_preconditions=["Current radii preserve role information from placement."],
            guaranteed_postconditions=["Proposed radii express role-conditioned growth pressure."],
            invariants=["Role-conditioned growth remains coordinated with placement roles."],
            assumptions=["Boundary and interior roles benefit from different growth pressure."],
            expected_effects=["Proposed radii respect structural roles across the dynamics block."],
            failure_modes=["Role pressure can reinforce a poor early role assignment."],
            compatibility_tags=["role_conditioned", "boundary_roles", "dynamics_block"],
            allowed_parameters=[
                _param("role_relation", "relation_token", ["boundary_slower", "balanced", "interior_slower"]),
            ],
        ),
    },
    "conflict": {
        "conflict_pairwise_overlap": _module(
            module_key="conflict_pairwise_overlap",
            slot="conflict",
            family="pairwise_overlap_conflict",
            display_name="Pairwise Overlap Conflict",
            hypothesis_template="Define conflicts through pairwise overlap pressure among proposed radii.",
            required_preconditions=["Proposed radii are available for feasibility checking."],
            guaranteed_postconditions=["Conflict report describes pairwise overlap pressure."],
            invariants=["Conflict semantics stay focused on circle interaction pressure."],
            assumptions=["Pairwise overlap is the primary signal limiting feasible radius growth."],
            expected_effects=["Conflict reports isolate direct circle collisions."],
            failure_modes=["Pairwise focus can miss boundary-driven infeasibility."],
            compatibility_tags=["pairwise_overlap", "direct_conflict", "dynamics_block"],
            allowed_parameters=[
                _param("severity_mode", "categorical_token", ["total_overlap", "worst_overlap", "balanced"]),
            ],
        ),
        "conflict_overlap_plus_boundary_penetration": _module(
            module_key="conflict_overlap_plus_boundary_penetration",
            slot="conflict",
            family="overlap_boundary_conflict",
            display_name="Overlap Plus Boundary Penetration Conflict",
            hypothesis_template="Define conflicts through both pairwise overlap and boundary penetration pressure.",
            required_preconditions=["Proposed radii can be checked against circles and square boundaries."],
            guaranteed_postconditions=["Conflict report includes overlap pressure and boundary pressure."],
            invariants=["Boundary pressure is first-class in conflict semantics."],
            assumptions=["Joint conflict signals reduce disagreement between repair and boundary handling."],
            expected_effects=["Conflict reports expose circle and edge violations together."],
            failure_modes=["Combining pressures can obscure which repair action matters most."],
            compatibility_tags=["pairwise_overlap", "boundary_pressure", "dynamics_block"],
            allowed_parameters=[
                _param("violation_balance", "relation_token", ["boundary_stronger", "pairwise_stronger", "balanced"]),
            ],
        ),
        "conflict_worst_violation_first": _module(
            module_key="conflict_worst_violation_first",
            slot="conflict",
            family="worst_violation_conflict",
            display_name="Worst Violation First Conflict",
            hypothesis_template="Define conflicts by emphasizing the worst current violation before aggregate pressure.",
            required_preconditions=["Proposed radii expose comparable violation pressure."],
            guaranteed_postconditions=["Conflict report highlights the dominant violation."],
            invariants=["Worst-violation identity remains explicit."],
            assumptions=["Repair benefits from knowing the most restrictive failure first."],
            expected_effects=["Conflict reports guide sequential repair toward the sharpest blocker."],
            failure_modes=["Worst-first focus can neglect many smaller coordinated conflicts."],
            compatibility_tags=["worst_first", "priority_conflict", "dynamics_block"],
            allowed_parameters=[
                _param("priority_signal", "categorical_token", ["largest_overlap", "largest_boundary", "balanced"]),
            ],
        ),
        "conflict_constraint_graph": _module(
            module_key="conflict_constraint_graph",
            slot="conflict",
            family="constraint_graph_conflict",
            display_name="Constraint Graph Conflict",
            hypothesis_template="Represent conflicts as a constraint graph linking mutually restrictive circles and edges.",
            required_preconditions=["Proposed radii can be related through abstract constraint neighborhoods."],
            guaranteed_postconditions=["Conflict report exposes graph-structured feasibility pressure."],
            invariants=["Conflict graph semantics remain coupled to repair structure."],
            assumptions=["Graph structure captures coordinated conflicts better than isolated violations."],
            expected_effects=["Conflict reports support block-aware repair decisions."],
            failure_modes=["Graph structure can overemphasize noisy weak constraints."],
            compatibility_tags=["constraint_graph", "block_conflict", "dynamics_block"],
            allowed_parameters=[
                _param("graph_scope", "categorical_token", ["local_neighbors", "global_pressure", "balanced"]),
            ],
        ),
    },
    "repair": {
        "repair_pairwise_repulsion": _module(
            module_key="repair_pairwise_repulsion",
            slot="repair",
            family="pairwise_repulsion_repair",
            display_name="Pairwise Repulsion Repair",
            hypothesis_template="Repair infeasibility by separating pairwise overlaps while preserving the layout intent.",
            required_preconditions=["Conflict report includes pairwise pressure."],
            guaranteed_postconditions=["Feasibility status reflects pairwise repulsion repair attempts."],
            invariants=["Repair remains local to direct overlap pressure."],
            assumptions=["Separating overlapping neighbors improves feasibility without redesigning placement."],
            expected_effects=["Feasibility status improves through direct circle separation."],
            failure_modes=["Pairwise repair can move pressure into boundary constraints."],
            compatibility_tags=["pairwise_repair", "local_repulsion", "dynamics_block"],
            allowed_parameters=[
                _param("repair_strength", "ordinal_token", ["low", "medium", "high"]),
            ],
        ),
        "repair_boundary_repulsion": _module(
            module_key="repair_boundary_repulsion",
            slot="repair",
            family="boundary_repulsion_repair",
            display_name="Boundary Repulsion Repair",
            hypothesis_template="Repair infeasibility by pushing boundary pressure inward before resolving residual conflicts.",
            required_preconditions=["Conflict report exposes boundary-related pressure."],
            guaranteed_postconditions=["Feasibility status reflects boundary repulsion repair attempts."],
            invariants=["Boundary correction is treated as a declared repair action."],
            assumptions=["Boundary pressure should be relieved before interior refinements settle."],
            expected_effects=["Feasibility status improves by protecting containment first."],
            failure_modes=["Inward pressure can create new interior overlap."],
            compatibility_tags=["boundary_repair", "edge_pressure", "dynamics_block"],
            allowed_parameters=[
                _param("boundary_strength", "ordinal_token", ["low", "medium", "high"]),
            ],
        ),
        "repair_sequential_worst_first": _module(
            module_key="repair_sequential_worst_first",
            slot="repair",
            family="sequential_worst_first_repair",
            display_name="Sequential Worst First Repair",
            hypothesis_template="Repair infeasibility by repeatedly addressing the dominant reported violation.",
            required_preconditions=["Conflict report identifies prioritized violation pressure."],
            guaranteed_postconditions=["Feasibility status reflects sequential worst-first repair."],
            invariants=["Repair ordering follows declared conflict priority."],
            assumptions=["Dominant blockers should be resolved before diffuse fine tuning."],
            expected_effects=["Feasibility status improves through focused correction of sharp violations."],
            failure_modes=["Sequential focus can cycle when violations trade places."],
            compatibility_tags=["worst_first_repair", "sequential_repair", "dynamics_block"],
            allowed_parameters=[
                _param("retry_policy", "categorical_token", ["persistent_first", "recent_first", "balanced"]),
            ],
        ),
        "repair_shrink_on_persistent_failure": _module(
            module_key="repair_shrink_on_persistent_failure",
            slot="repair",
            family="shrink_failure_repair",
            display_name="Shrink On Persistent Failure Repair",
            hypothesis_template="Repair persistent infeasibility by shrinking resistant radii after movement fails.",
            required_preconditions=["Conflict report can distinguish persistent infeasibility."],
            guaranteed_postconditions=["Feasibility status records shrink-based recovery when needed."],
            invariants=["Shrink repair is a fallback rather than the primary growth policy."],
            assumptions=["Some conflicts are better resolved by conceding radius than by moving centers."],
            expected_effects=["Feasibility status can recover from stubborn local conflicts."],
            failure_modes=["Excessive shrink recovery can sacrifice packing quality."],
            compatibility_tags=["fallback_shrink", "persistent_failure", "dynamics_block"],
            allowed_parameters=[
                _param("fallback_shrink_enabled", "boolean", [False, True]),
            ],
        ),
        "repair_bilevel_positions_then_radii": _module(
            module_key="repair_bilevel_positions_then_radii",
            slot="repair",
            family="bilevel_repair",
            display_name="Bilevel Positions Then Radii Repair",
            hypothesis_template="Repair infeasibility as a bilevel process that stabilizes positions before radius concessions.",
            required_preconditions=["Conflict report supports both positional and radius pressure."],
            guaranteed_postconditions=["Feasibility status reflects coordinated position and radius repair."],
            invariants=["Position repair and radius repair remain linked in the dynamics block."],
            assumptions=["Separating positional correction from radius concession improves structural consistency."],
            expected_effects=["Feasibility status improves without immediately sacrificing radius quality."],
            failure_modes=["Bilevel repair can delay necessary radius concessions."],
            compatibility_tags=["bilevel_repair", "position_radius_coupling", "dynamics_block"],
            allowed_parameters=[
                _param("phase_order", "relation_token", ["positions_before_radii", "balanced", "radii_before_positions"]),
            ],
        ),
    },
    "boundary": {
        "boundary_hard_containment": _module(
            module_key="boundary_hard_containment",
            slot="boundary",
            family="hard_containment_boundary",
            display_name="Hard Containment Boundary",
            hypothesis_template="Handle boundaries as hard containment constraints that proposed radii must satisfy.",
            required_preconditions=["Proposed radii are available for boundary checking."],
            guaranteed_postconditions=["Boundary report declares containment feasibility."],
            invariants=["Containment remains a hard structural requirement."],
            assumptions=["Strict containment prevents boundary errors from contaminating other signals."],
            expected_effects=["Boundary report gives a clear feasibility gate."],
            failure_modes=["Hard containment can be brittle near tight edge configurations."],
            compatibility_tags=["hard_containment", "boundary_report", "dynamics_block"],
            allowed_parameters=[
                _param("containment_style", "categorical_token", ["strict_barrier", "rejecting_barrier", "balanced_barrier"]),
            ],
        ),
        "boundary_repulsive_margin": _module(
            module_key="boundary_repulsive_margin",
            slot="boundary",
            family="repulsive_margin_boundary",
            display_name="Repulsive Margin Boundary",
            hypothesis_template="Handle boundaries as repulsive margins that discourage edge penetration before failure.",
            required_preconditions=["Proposed radii can express proximity to square edges."],
            guaranteed_postconditions=["Boundary report includes margin pressure."],
            invariants=["Boundary pressure is continuous at the hypothesis level."],
            assumptions=["Soft margin pressure can prevent late hard containment failures."],
            expected_effects=["Boundary report guides repair before edge violations dominate."],
            failure_modes=["Margin pressure can reduce useful edge utilization."],
            compatibility_tags=["repulsive_margin", "soft_boundary", "dynamics_block"],
            allowed_parameters=[
                _param("margin_strength", "ordinal_token", ["low", "medium", "high"]),
            ],
        ),
        "boundary_edge_aware_inset": _module(
            module_key="boundary_edge_aware_inset",
            slot="boundary",
            family="edge_aware_inset_boundary",
            display_name="Edge Aware Inset Boundary",
            hypothesis_template="Handle boundaries by favoring inset behavior that protects edge and corner feasibility.",
            required_preconditions=["Proposed radii can distinguish edge and corner pressure."],
            guaranteed_postconditions=["Boundary report distinguishes edge-aware containment pressure."],
            invariants=["Edge and corner pressure remain separately visible."],
            assumptions=["Inset awareness reduces accidental boundary crowding."],
            expected_effects=["Boundary report supports targeted edge-safe repair."],
            failure_modes=["Inset bias can leave too much unused boundary capacity."],
            compatibility_tags=["edge_aware", "inset_boundary", "dynamics_block"],
            allowed_parameters=[
                _param("inset_bias", "categorical_token", ["corner_protecting", "edge_protecting", "balanced"]),
            ],
        ),
        "boundary_role_based_edge_protection": _module(
            module_key="boundary_role_based_edge_protection",
            slot="boundary",
            family="role_based_edge_boundary",
            display_name="Role Based Edge Protection Boundary",
            hypothesis_template="Handle boundaries according to explicit edge roles inherited from the placement structure.",
            required_preconditions=["Proposed radii preserve boundary role information."],
            guaranteed_postconditions=["Boundary report expresses role-based edge protection."],
            invariants=["Boundary handling remains coupled to role-based placement assumptions."],
            assumptions=["Role-aware edge protection keeps boundary and repair semantics aligned."],
            expected_effects=["Boundary report protects edge roles without flattening interior behavior."],
            failure_modes=["Wrong role protection can preserve poor boundary commitments."],
            compatibility_tags=["boundary_roles", "edge_protection", "dynamics_block"],
            allowed_parameters=[
                _param("edge_role_relation", "relation_token", ["boundary_stronger", "balanced", "interior_stronger"]),
            ],
        ),
    },
    "termination": {
        "termination_fixed_stability": _module(
            module_key="termination_fixed_stability",
            slot="termination",
            family="fixed_stability_termination",
            display_name="Fixed Stability Termination",
            hypothesis_template="Stop when the packing has maintained stable feasibility under the declared control signal.",
            required_preconditions=["Packing summary or feasibility status is available for control."],
            guaranteed_postconditions=["Termination signal reflects a stable stopping decision."],
            invariants=["Stopping logic remains deterministic and audit-friendly."],
            assumptions=["Stable feasibility is enough to end constructive refinement."],
            expected_effects=["Termination avoids unnecessary changes after stability emerges."],
            failure_modes=["Fixed stability can stop before subtle improvements appear."],
            compatibility_tags=["stability_stop", "control_block"],
            allowed_parameters=[
                _param("stability_patience", "ordinal_token", ["low", "medium", "high"]),
            ],
        ),
        "termination_no_violation_and_no_gain": _module(
            module_key="termination_no_violation_and_no_gain",
            slot="termination",
            family="no_violation_no_gain_termination",
            display_name="No Violation And No Gain Termination",
            hypothesis_template="Stop when feasibility has no reported violation and recent packing gains have faded.",
            required_preconditions=["Feasibility status and packing summary can both inform control."],
            guaranteed_postconditions=["Termination signal combines feasibility and gain stagnation."],
            invariants=["Stopping depends on both safety and improvement signals."],
            assumptions=["A feasible packing with fading gains should preserve its current structure."],
            expected_effects=["Termination prevents over-repair after gains disappear."],
            failure_modes=["Gain fading can be mistaken for true convergence."],
            compatibility_tags=["feasibility_stop", "gain_stop", "control_block"],
            allowed_parameters=[
                _param("gain_policy", "categorical_token", ["feasibility_first", "gain_first", "balanced"]),
            ],
        ),
        "termination_plateau_detection": _module(
            module_key="termination_plateau_detection",
            slot="termination",
            family="plateau_detection_termination",
            display_name="Plateau Detection Termination",
            hypothesis_template="Stop when packing improvement enters a plateau under declared feasibility constraints.",
            required_preconditions=["Packing summary exposes abstract progress information."],
            guaranteed_postconditions=["Termination signal reflects plateau-based stopping."],
            invariants=["Plateau detection does not depend on hidden implementation counters."],
            assumptions=["Plateaus identify when further refinement is unlikely to pay off."],
            expected_effects=["Termination focuses effort before improvement stalls completely."],
            failure_modes=["Plateau detection can miss delayed improvements."],
            compatibility_tags=["plateau_stop", "progress_sensitive", "control_block"],
            allowed_parameters=[
                _param("plateau_sensitivity", "ordinal_token", ["low", "medium", "high"]),
            ],
        ),
        "termination_repair_budget_exhaustion": _module(
            module_key="termination_repair_budget_exhaustion",
            slot="termination",
            family="repair_budget_termination",
            display_name="Repair Budget Exhaustion Termination",
            hypothesis_template="Stop when repair effort is exhausted relative to remaining feasibility pressure.",
            required_preconditions=["Feasibility status can indicate unresolved repair pressure."],
            guaranteed_postconditions=["Termination signal reflects repair-budget exhaustion."],
            invariants=["Budget semantics remain abstract and not tied to raw loop counts."],
            assumptions=["Persistent repair pressure can signal diminishing returns."],
            expected_effects=["Termination avoids damaging churn when repair no longer helps."],
            failure_modes=["Budget exhaustion can accept an under-repaired packing."],
            compatibility_tags=["repair_budget_stop", "control_block"],
            allowed_parameters=[
                _param("budget_policy", "categorical_token", ["conservative", "balanced", "aggressive"]),
            ],
        ),
    },
}


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
    lowered = value.lower()
    if "`" in value:
        _fail(path, "must not contain backticks")
    for banned in ("numpy", "def ", "import "):
        if banned in lowered:
            _fail(path, f"must not contain {banned!r}")
    return value


def _ensure_string_list(value: Any, path: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        _fail(path, "must be a list")
    if not allow_empty and not value:
        _fail(path, "must be a non-empty list")
    return [_ensure_str(item, f"{path}[{idx}]") for idx, item in enumerate(value)]


def _reject_numeric_json_values(value: Any, path: str = "$") -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        _fail(path, "numeric JSON values are forbidden")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_numeric_json_values(child, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _reject_numeric_json_values(child, f"{path}[{idx}]")


def _validate_parameter_schema(parameters: Any, path: str) -> None:
    if not isinstance(parameters, list):
        _fail(path, "must be a list")
    seen_names: set[str] = set()
    for idx, entry in enumerate(parameters):
        entry_path = f"{path}[{idx}]"
        payload = _ensure_exact_keys(entry, PARAMETER_SCHEMA_KEYS, entry_path)
        name = _ensure_str(payload["name"], f"{entry_path}.name")
        if _SNAKE_CASE_RE.fullmatch(name) is None:
            _fail(f"{entry_path}.name", "must be snake_case")
        if name in seen_names:
            _fail(f"{entry_path}.name", f"duplicate parameter name {name!r}")
        seen_names.add(name)

        value_kind = _ensure_str(payload["value_kind"], f"{entry_path}.value_kind")
        if value_kind not in VALUE_KINDS:
            _fail(f"{entry_path}.value_kind", f"must be one of {sorted(VALUE_KINDS)!r}")

        allowed_values = payload["allowed_values"]
        if not isinstance(allowed_values, list) or not allowed_values:
            _fail(f"{entry_path}.allowed_values", "must be a non-empty list")
        for value_idx, allowed_value in enumerate(allowed_values):
            allowed_path = f"{entry_path}.allowed_values[{value_idx}]"
            if value_kind == "boolean":
                if not isinstance(allowed_value, bool):
                    _fail(allowed_path, "must be a boolean for boolean parameters")
            else:
                token = _ensure_str(allowed_value, allowed_path)
                if _TOKEN_RE.fullmatch(token) is None:
                    _fail(allowed_path, "must be a lowercase token")

        if not isinstance(payload["required"], bool):
            _fail(f"{entry_path}.required", "must be a boolean")


def _validate_library_definition(definition: Any, *, slot: str, module_key: str) -> None:
    path = f"MODULE_LIBRARY.{slot}.{module_key}"
    payload = _ensure_exact_keys(definition, MODULE_DEFINITION_KEYS, path)
    if payload["module_key"] != module_key:
        _fail(f"{path}.module_key", f"must equal {module_key!r}")
    if payload["slot"] != slot:
        _fail(f"{path}.slot", f"must equal {slot!r}")
    if payload["module_type"] != SLOT_TO_MODULE_TYPE[slot]:
        _fail(f"{path}.module_type", f"must equal {SLOT_TO_MODULE_TYPE[slot]!r}")
    if payload["linkage_group"] != _slot_linkage_group(slot):
        _fail(f"{path}.linkage_group", f"must equal {_slot_linkage_group(slot)!r}")

    expected_inheritance = "linkage_block" if module_key in LINKAGE_BLOCK_MODULE_KEYS else "slot"
    if payload["inheritance_unit"] != expected_inheritance:
        _fail(f"{path}.inheritance_unit", f"must equal {expected_inheritance!r}")

    for key in (
        "module_key",
        "slot",
        "module_type",
        "family",
        "display_name",
        "hypothesis_template",
        "linkage_group",
        "inheritance_unit",
    ):
        _ensure_str(payload[key], f"{path}.{key}")

    for key in (
        "reads_state",
        "writes_state",
        "required_preconditions",
        "guaranteed_postconditions",
        "invariants",
        "assumptions",
        "expected_effects",
        "failure_modes",
        "compatibility_tags",
    ):
        _ensure_string_list(payload[key], f"{path}.{key}")

    if payload["reads_state"] != STATE_BY_SLOT[slot]["reads_state"]:
        _fail(f"{path}.reads_state", "must match canonical slot reads")
    if payload["writes_state"] != STATE_BY_SLOT[slot]["writes_state"]:
        _fail(f"{path}.writes_state", "must match canonical slot writes")

    _validate_parameter_schema(payload["allowed_parameters"], f"{path}.allowed_parameters")
    _reject_numeric_json_values(payload, path)


def _build_module_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    display_names: set[str] = set()
    hypothesis_templates: set[str] = set()
    if list(MODULE_LIBRARY.keys()) != SLOT_ORDER:
        raise ValueError("MODULE_LIBRARY slot order is not canonical.")
    for slot in SLOT_ORDER:
        slot_modules = MODULE_LIBRARY[slot]
        if list(slot_modules.keys()) != REQUIRED_MODULE_KEYS[slot]:
            raise ValueError(f"MODULE_LIBRARY.{slot} module keys are not canonical.")
        for module_key, definition in slot_modules.items():
            _validate_library_definition(definition, slot=slot, module_key=module_key)
            if module_key in index:
                raise ValueError(f"Duplicate module_key {module_key!r} in module library.")
            display_name = definition["display_name"]
            if display_name in display_names:
                raise ValueError(f"Duplicate display_name {display_name!r} in module library.")
            display_names.add(display_name)
            hypothesis_template = definition["hypothesis_template"]
            if hypothesis_template in hypothesis_templates:
                raise ValueError(
                    f"Duplicate hypothesis_template for module_key {module_key!r}; aliases are not allowed."
                )
            hypothesis_templates.add(hypothesis_template)
            index[module_key] = definition
    return index


MODULE_INDEX = _build_module_index()


def get_circle_packing_module_library_v1() -> dict:
    """Return a defensive copy of the closed v1 module library."""

    return deepcopy(MODULE_LIBRARY)


def get_allowed_modules_for_slot(slot: str) -> dict:
    """Return allowed module definitions for one canonical slot."""

    if slot not in MODULE_LIBRARY:
        raise ValueError(f"Unknown slot {slot!r}.")
    return deepcopy(MODULE_LIBRARY[slot])


def get_module_definition(module_key: str) -> dict:
    """Return the canonical definition for one module key."""

    if module_key not in MODULE_INDEX:
        raise ValueError(f"Unknown module_key {module_key!r}.")
    return deepcopy(MODULE_INDEX[module_key])


def validate_slot_assignment(slot: str, module_key: str) -> None:
    """Validate that a module key belongs to the requested slot."""

    if slot not in MODULE_LIBRARY:
        raise ValueError(f"Unknown slot {slot!r}.")
    definition = get_module_definition(module_key)
    if definition["slot"] != slot:
        raise ValueError(
            f"Module {module_key!r} belongs to slot {definition['slot']!r}, not slot {slot!r}."
        )


def _validate_parameterization(
    parameterization: Any,
    definition: dict[str, Any],
    *,
    path: str,
) -> list[dict[str, Any]]:
    if not isinstance(parameterization, list):
        _fail(path, "must be a list")

    allowed_by_name = {
        param["name"]: param
        for param in definition["allowed_parameters"]
    }
    values_by_name: dict[str, dict[str, Any]] = {}
    for idx, entry in enumerate(parameterization):
        entry_path = f"{path}[{idx}]"
        payload = _ensure_exact_keys(entry, PARAMETER_INSTANCE_KEYS, entry_path)
        name = _ensure_str(payload["name"], f"{entry_path}.name")
        if name in values_by_name:
            _fail(f"{entry_path}.name", f"duplicate parameter name {name!r}")
        if name not in allowed_by_name:
            _fail(f"{entry_path}.name", f"unexpected parameter name {name!r}")
        schema = allowed_by_name[name]
        value_kind = _ensure_str(payload["value_kind"], f"{entry_path}.value_kind")
        if value_kind != schema["value_kind"]:
            _fail(
                f"{entry_path}.value_kind",
                f"expected {schema['value_kind']!r}, got {value_kind!r}",
            )
        value = payload["value"]
        if value_kind == "boolean":
            if not isinstance(value, bool):
                _fail(f"{entry_path}.value", "must be a boolean")
        else:
            value = _ensure_str(value, f"{entry_path}.value")
        if value not in schema["allowed_values"]:
            _fail(
                f"{entry_path}.value",
                f"must be one of {schema['allowed_values']!r}",
            )
        values_by_name[name] = {"name": name, "value_kind": value_kind, "value": value}

    missing = [
        param["name"]
        for param in definition["allowed_parameters"]
        if param["required"] and param["name"] not in values_by_name
    ]
    if missing:
        _fail(path, f"missing required parameters: {', '.join(missing)}")

    expected_names = {param["name"] for param in definition["allowed_parameters"]}
    extra = sorted(set(values_by_name.keys()).difference(expected_names))
    if extra:
        _fail(path, f"extra parameters: {', '.join(extra)}")

    return [
        deepcopy(values_by_name[param["name"]])
        for param in definition["allowed_parameters"]
        if param["name"] in values_by_name
    ]


def materialize_module_instance(
    *,
    slot: str,
    module_key: str,
    module_id: str,
    parameterization: list[dict],
) -> dict:
    """Build a fully materialized genome module instance from a library key."""

    validate_slot_assignment(slot, module_key)
    definition = get_module_definition(module_key)
    canonical_parameters = _validate_parameterization(
        parameterization,
        definition,
        path=f"module_instance.{module_key}.parameterization",
    )
    module_id_text = _ensure_str(module_id, f"module_instance.{module_key}.module_id")

    return {
        "module_id": module_id_text,
        "module_key": definition["module_key"],
        "slot": definition["slot"],
        "module_type": definition["module_type"],
        "family": definition["family"],
        "hypothesis": definition["hypothesis_template"],
        "reads_state": deepcopy(definition["reads_state"]),
        "writes_state": deepcopy(definition["writes_state"]),
        "parameterization": canonical_parameters,
        "preconditions": deepcopy(definition["required_preconditions"]),
        "postconditions": deepcopy(definition["guaranteed_postconditions"]),
        "invariants": deepcopy(definition["invariants"]),
        "assumptions": deepcopy(definition["assumptions"]),
        "expected_effects": deepcopy(definition["expected_effects"]),
        "failure_modes": deepcopy(definition["failure_modes"]),
        "compatibility_tags": deepcopy(definition["compatibility_tags"]),
        "linkage_group": definition["linkage_group"],
        "inheritance_unit": definition["inheritance_unit"],
    }


def validate_module_instance(module_instance: dict) -> None:
    """Validate one fully materialized genome module instance against the library."""

    payload = _ensure_exact_keys(module_instance, MODULE_INSTANCE_KEYS, "module_instance")
    module_key = _ensure_str(payload["module_key"], "module_instance.module_key")
    definition = get_module_definition(module_key)
    slot = _ensure_str(payload["slot"], "module_instance.slot")
    validate_slot_assignment(slot, module_key)

    module_id = _ensure_str(payload["module_id"], "module_instance.module_id")
    expected = materialize_module_instance(
        slot=slot,
        module_key=module_key,
        module_id=module_id,
        parameterization=payload["parameterization"],
    )
    if payload != expected:
        drifted = [
            key
            for key in sorted(MODULE_INSTANCE_KEYS)
            if payload.get(key) != expected.get(key)
        ]
        raise ValueError(
            "module_instance: library-controlled fields do not match canonical "
            f"definition for {module_key!r}: {', '.join(drifted)}"
        )
    if definition["slot"] != slot:
        raise ValueError(f"module_instance.slot: expected {definition['slot']!r}, got {slot!r}")


def get_module_inheritance_unit(module_key: str) -> str:
    """Return the inheritance unit label for one module key."""

    return get_module_definition(module_key)["inheritance_unit"]


def get_module_linkage_group(module_key: str) -> str:
    """Return the linkage group for one module key."""

    return get_module_definition(module_key)["linkage_group"]


def get_slot_linkage_group(slot: str) -> str:
    """Return the canonical linkage group for one slot."""

    if slot not in MODULE_LIBRARY:
        raise ValueError(f"Unknown slot {slot!r}.")
    return _slot_linkage_group(slot)


def default_parameterization_for_module(module_key: str) -> list[dict[str, Any]]:
    """Return deterministic default parameters for internal canonical adapters."""

    definition = get_module_definition(module_key)
    return [
        {
            "name": param["name"],
            "value_kind": param["value_kind"],
            "value": param["allowed_values"][0],
        }
        for param in definition["allowed_parameters"]
        if param["required"]
    ]
