"""Typed design prompt contract tests for circle_packing_shinka."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from experiments.circle_packing_shinka._runtime import design_prompt_contract_v1 as contract
from experiments.circle_packing_shinka._runtime import module_library_v1 as library
from tests.fixtures.circle_packing_genome import corrective_genome, valid_circle_packing_genome


def _assignment(slot: str, module_key: str, organism_id: str = "child") -> dict:
    return {
        "module_key": module_key,
        "module_id": f"{organism_id}_{slot}",
        "parameterization": library.default_parameterization_for_module(module_key),
    }


def _base_response(operation: str = "seed", organism_id: str = "child") -> dict:
    assignments = {
        "layout": _assignment("layout", "layout_triangular_lattice", organism_id),
        "selection": _assignment("selection", "selection_center_outward", organism_id),
        "radius_init": _assignment("radius_init", "radius_init_lattice_derived", organism_id),
        "growth": _assignment("growth", "growth_density_scaled_additive", organism_id),
        "conflict": _assignment("conflict", "conflict_overlap_plus_boundary_penetration", organism_id),
        "repair": _assignment("repair", "repair_pairwise_repulsion", organism_id),
        "boundary": _assignment("boundary", "boundary_repulsive_margin", organism_id),
        "termination": _assignment("termination", "termination_no_violation_and_no_gain", organism_id),
    }
    response = {
        "schema_name": "circle_packing_typed_design_response",
        "schema_version": "1.0",
        "task_family": "circle_packing_shinka",
        "task_name": "unit_square_26",
        "operation": operation,
        "global_hypothesis": {
            "title": "Typed hypothesis",
            "core_claim": "A typed scaffold coordinates placement and feasibility.",
            "expected_advantage": "The selected modules keep responsibilities explicit.",
            "novelty_statement": "The child is represented as typed slot assignments.",
        },
        "slot_assignments": assignments,
        "render_fields": {
            "interaction_notes": ["Typed modules exchange declared state.", "Repair follows conflict reports."],
            "compute_notes": ["Keep the realization deterministic.", "Avoid unnecessary search loops."],
            "change_description": "The child is represented as typed slot assignments.",
        },
        "operator_metadata": {
            "seed_family": "new_seed",
            "design_mode": "typed_library_selection",
        },
    }
    if operation == "mutation":
        response["operator_metadata"] = {
            "mutation_scope": "slot",
            "changed_slots": ["growth"],
            "preserved_slots": [
                "layout",
                "selection",
                "radius_init",
                "conflict",
                "repair",
                "boundary",
                "termination",
            ],
            "parent_reference": "parent",
            "change_rationale": "The child changes growth pressure only.",
        }
        response["slot_assignments"]["growth"] = _assignment("growth", "growth_uniform_multiplicative", organism_id)
    if operation == "crossover":
        response["operator_metadata"] = {
            "inheritance_mode": "slotwise",
            "slot_origins": {
                "layout": "primary",
                "selection": "primary",
                "radius_init": "primary",
                "growth": "secondary",
                "conflict": "secondary",
                "repair": "secondary",
                "boundary": "secondary",
                "termination": "primary",
            },
            "primary_slots": ["layout", "selection", "radius_init", "termination"],
            "secondary_slots": ["growth", "conflict", "repair", "boundary"],
            "change_rationale": "The child combines primary placement with secondary dynamics.",
        }
        secondary = contract.compact_slot_assignments_from_genome(corrective_genome(organism_id="secondary"))
        for slot in ("growth", "conflict", "repair", "boundary"):
            response["slot_assignments"][slot] = deepcopy(secondary[slot])
            response["slot_assignments"][slot]["module_id"] = f"{organism_id}_{slot}"
    return response


def _raw(response: dict) -> str:
    return json.dumps(response, indent=2)


def test_valid_seed_json_parses_and_validates() -> None:
    response = contract.parse_seed_design_response(_raw(_base_response("seed")))
    assert response["operation"] == "seed"


def test_valid_mutation_json_parses_and_validates_against_parent() -> None:
    parent = valid_circle_packing_genome(organism_id="parent")
    response = contract.parse_mutation_design_response(
        _raw(_base_response("mutation")),
        parent_genome=parent,
    )
    assert response["operator_metadata"]["changed_slots"] == ["growth"]


def test_valid_crossover_json_parses_and_validates_against_parents() -> None:
    response = contract.parse_crossover_design_response(
        _raw(_base_response("crossover")),
        primary_parent_genome=valid_circle_packing_genome(organism_id="primary"),
        secondary_parent_genome=corrective_genome(organism_id="secondary"),
    )
    assert response["operator_metadata"]["secondary_slots"] == ["growth", "conflict", "repair", "boundary"]


def test_malformed_json_fails() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        contract.parse_seed_design_response("{not json")


def test_extra_top_level_key_fails() -> None:
    response = _base_response("seed")
    response["extra"] = "forbidden"
    with pytest.raises(ValueError, match="extra keys"):
        contract.parse_seed_design_response(_raw(response))


def test_missing_top_level_key_fails() -> None:
    response = _base_response("seed")
    del response["render_fields"]
    with pytest.raises(ValueError, match="missing keys"):
        contract.parse_seed_design_response(_raw(response))


def test_illegal_module_key_fails() -> None:
    response = _base_response("seed")
    response["slot_assignments"]["layout"]["module_key"] = "layout_not_real"
    with pytest.raises(ValueError, match="Unknown module_key|belongs"):
        contract.parse_seed_design_response(_raw(response))


def test_illegal_parameter_value_fails() -> None:
    response = _base_response("seed")
    response["slot_assignments"]["layout"]["parameterization"][0]["value"] = "not_allowed"
    with pytest.raises(ValueError, match="must be one of"):
        contract.parse_seed_design_response(_raw(response))


def test_mutation_with_wrong_changed_slots_fails() -> None:
    response = _base_response("mutation")
    response["operator_metadata"]["changed_slots"] = ["repair"]
    response["operator_metadata"]["preserved_slots"] = [
        "layout",
        "selection",
        "radius_init",
        "growth",
        "conflict",
        "boundary",
        "termination",
    ]
    with pytest.raises(ValueError, match="actual changed slots"):
        contract.parse_mutation_design_response(
            _raw(response),
            parent_genome=valid_circle_packing_genome(organism_id="parent"),
        )


def test_mutation_that_changes_undeclared_slots_fails() -> None:
    response = _base_response("mutation")
    response["slot_assignments"]["repair"] = _assignment("repair", "repair_boundary_repulsion")
    with pytest.raises(ValueError, match="actual changed slots"):
        contract.parse_mutation_design_response(
            _raw(response),
            parent_genome=valid_circle_packing_genome(organism_id="parent"),
        )


def test_crossover_with_non_parent_module_fails() -> None:
    response = _base_response("crossover")
    response["slot_assignments"]["growth"] = _assignment("growth", "growth_role_conditioned_growth")
    with pytest.raises(ValueError, match="declared secondary parent module"):
        contract.parse_crossover_design_response(
            _raw(response),
            primary_parent_genome=valid_circle_packing_genome(organism_id="primary"),
            secondary_parent_genome=corrective_genome(organism_id="secondary"),
        )


def test_crossover_with_inconsistent_slot_origins_fails() -> None:
    response = _base_response("crossover")
    response["operator_metadata"]["slot_origins"]["growth"] = "primary"
    with pytest.raises(ValueError, match="primary_slots and secondary_slots must match"):
        contract.parse_crossover_design_response(_raw(response))


def test_design_response_fails_if_hard_compatibility_fails() -> None:
    response = _base_response("seed")
    response["slot_assignments"]["growth"] = _assignment("growth", "growth_role_conditioned_growth")
    with pytest.raises(ValueError, match="hard compatibility"):
        contract.parse_seed_design_response(_raw(response))


def test_design_response_fails_if_functional_checks_fail() -> None:
    response = _base_response("seed")
    response["slot_assignments"]["radius_init"] = _assignment("radius_init", "radius_init_uniform_small")
    response["slot_assignments"]["repair"] = _assignment("repair", "repair_shrink_on_persistent_failure")
    response["slot_assignments"]["termination"] = _assignment("termination", "termination_fixed_stability")
    with pytest.raises(ValueError, match="functional checks"):
        contract.parse_seed_design_response(_raw(response))

