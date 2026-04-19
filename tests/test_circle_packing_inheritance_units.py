"""Tests for circle-packing inheritance and linkage metadata."""

from __future__ import annotations

from experiments.circle_packing_shinka._runtime import genome_schema_v1 as schema
from experiments.circle_packing_shinka._runtime import module_library_v1 as library
from tests.fixtures.circle_packing_genome import valid_circle_packing_genome


EXPECTED_SLOT_LINKAGE = {
    "layout": "placement_block",
    "selection": "placement_block",
    "radius_init": "placement_block",
    "growth": "dynamics_block",
    "conflict": "dynamics_block",
    "repair": "dynamics_block",
    "boundary": "dynamics_block",
    "termination": "control_block",
}

EXPECTED_LINKAGE_BLOCK_MODULES = {
    "layout_triangular_lattice",
    "layout_boundary_ring_interior_fill",
    "radius_init_lattice_derived",
    "radius_init_role_based_boundary_vs_interior",
    "growth_role_conditioned_growth",
    "boundary_role_based_edge_protection",
    "repair_bilevel_positions_then_radii",
    "conflict_constraint_graph",
}


def test_every_slot_maps_to_correct_canonical_linkage_group() -> None:
    assert {
        slot: library.get_slot_linkage_group(slot)
        for slot in schema.SLOT_ORDER
    } == EXPECTED_SLOT_LINKAGE


def test_every_module_reports_correct_linkage_group() -> None:
    for slot, slot_modules in library.get_circle_packing_module_library_v1().items():
        for module_key in slot_modules:
            assert library.get_module_linkage_group(module_key) == EXPECTED_SLOT_LINKAGE[slot]


def test_every_module_reports_valid_inheritance_unit() -> None:
    for slot_modules in library.get_circle_packing_module_library_v1().values():
        for module_key in slot_modules:
            assert library.get_module_inheritance_unit(module_key) in {"slot", "linkage_block"}


def test_linkage_block_modules_are_exactly_mandated_set() -> None:
    actual = {
        module_key
        for slot_modules in library.get_circle_packing_module_library_v1().values()
        for module_key in slot_modules
        if library.get_module_inheritance_unit(module_key) == "linkage_block"
    }

    assert actual == EXPECTED_LINKAGE_BLOCK_MODULES


def test_all_other_modules_are_slot_inheritance_units() -> None:
    for slot_modules in library.get_circle_packing_module_library_v1().values():
        for module_key in slot_modules:
            if module_key in EXPECTED_LINKAGE_BLOCK_MODULES:
                continue
            assert library.get_module_inheritance_unit(module_key) == "slot"


def test_inheritance_metadata_is_preserved_in_materialized_genome() -> None:
    genome = valid_circle_packing_genome()

    for slot, module in genome["slots"].items():
        module_key = module["module_key"]
        assert module["linkage_group"] == EXPECTED_SLOT_LINKAGE[slot]
        assert module["inheritance_unit"] == library.get_module_inheritance_unit(module_key)
