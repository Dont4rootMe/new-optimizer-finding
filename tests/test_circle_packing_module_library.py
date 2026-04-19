"""Tests for the closed circle_packing_shinka module library."""

from __future__ import annotations

import pytest

from experiments.circle_packing_shinka._runtime import genome_schema_v1 as schema
from experiments.circle_packing_shinka._runtime import module_library_v1 as library


EXPECTED_MODULE_KEYS = {
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


def test_every_slot_has_exact_required_module_keys_and_no_extras() -> None:
    module_library = library.get_circle_packing_module_library_v1()

    assert list(module_library.keys()) == schema.SLOT_ORDER
    assert {slot: list(modules.keys()) for slot, modules in module_library.items()} == EXPECTED_MODULE_KEYS


def test_every_module_key_is_globally_unique() -> None:
    module_library = library.get_circle_packing_module_library_v1()
    keys = [
        module_key
        for slot_modules in module_library.values()
        for module_key in slot_modules
    ]

    assert len(keys) == len(set(keys)) == 33


def test_every_module_definition_has_required_fields_and_matches_slot_bucket() -> None:
    module_library = library.get_circle_packing_module_library_v1()

    for slot, slot_modules in module_library.items():
        for module_key, definition in slot_modules.items():
            assert set(definition.keys()) == library.MODULE_DEFINITION_KEYS
            assert definition["module_key"] == module_key
            assert definition["slot"] == slot
            assert definition["module_type"] == schema.SLOT_TO_MODULE_TYPE[slot]
            assert definition["linkage_group"] == schema.linkage_group_for_slot(slot)
            assert definition["inheritance_unit"] in library.INHERITANCE_UNITS
            assert isinstance(definition["allowed_parameters"], list)


def test_lookup_by_exact_module_key_works() -> None:
    definition = library.get_module_definition("layout_triangular_lattice")

    assert definition["module_key"] == "layout_triangular_lattice"
    assert definition["slot"] == "layout"


def test_lookup_of_unknown_module_key_fails() -> None:
    with pytest.raises(ValueError, match="Unknown module_key"):
        library.get_module_definition("layout_triangular_alias")


def test_all_required_inheritance_metadata_exists_and_is_valid() -> None:
    for slot_modules in library.get_circle_packing_module_library_v1().values():
        for module_key, definition in slot_modules.items():
            assert library.get_module_inheritance_unit(module_key) == definition["inheritance_unit"]
            assert library.get_module_linkage_group(module_key) == definition["linkage_group"]
            assert definition["inheritance_unit"] in {"slot", "linkage_block"}
