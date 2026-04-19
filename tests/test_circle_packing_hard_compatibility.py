"""Hard compatibility tests for circle_packing_shinka genome v1."""

from __future__ import annotations

from experiments.circle_packing_shinka._runtime import compatibility_contract_v1 as compatibility
from experiments.circle_packing_shinka._runtime import genome_schema_v1 as schema
from experiments.circle_packing_shinka._runtime import module_library_v1 as library
from tests.fixtures.circle_packing_genome import (
    corrective_genome,
    structured_role_genome,
    valid_circle_packing_genome,
)


def _replace_module(genome: dict, slot: str, module_key: str) -> None:
    genome["slots"][slot] = schema.materialize_module_instance(
        slot=slot,
        module_key=module_key,
        module_id=f"{genome['organism_id']}_{slot}_replacement",
        parameterization=library.default_parameterization_for_module(module_key),
    )


def _rule_ids(genome: dict) -> set[str]:
    report = compatibility.build_circle_packing_compatibility_report(genome)
    return {
        violation["rule_id"]
        for violation in report["hard_compatibility"]["violations"]
    }


def test_representative_valid_genomes_pass_hard_compatibility() -> None:
    for genome in (structured_role_genome(), corrective_genome()):
        compatibility.validate_hard_compatibility(genome)
        report = compatibility.build_circle_packing_compatibility_report(genome)
        assert report["hard_compatibility"]["is_compatible"] is True
        assert report["hard_compatibility"]["violations"] == []


def test_missing_predecessor_state_production_fails_hc_if_001() -> None:
    genome = valid_circle_packing_genome()
    genome["slots"]["layout"]["writes_state"] = []

    assert "HC-IF-001" in _rule_ids(genome)


def test_wrong_minimum_interface_chain_fails_hc_if_002() -> None:
    genome = valid_circle_packing_genome()
    genome["slots"]["growth"]["reads_state"] = []

    assert "HC-IF-002" in _rule_ids(genome)


def test_backward_dependency_fails_hc_if_003() -> None:
    genome = valid_circle_packing_genome()
    genome["slots"]["selection"]["reads_state"] = ["current_radii"]

    assert "HC-IF-003" in _rule_ids(genome)


def test_empty_operational_interface_fails_hc_if_004() -> None:
    genome = valid_circle_packing_genome()
    genome["slots"]["growth"]["reads_state"] = []
    genome["slots"]["growth"]["writes_state"] = []

    assert "HC-IF-004" in _rule_ids(genome)


def test_role_conditioned_growth_without_support_fails_hc_lg_002() -> None:
    genome = valid_circle_packing_genome()
    _replace_module(genome, "growth", "growth_role_conditioned_growth")

    assert "HC-LG-002" in _rule_ids(genome)


def test_constraint_graph_with_incompatible_repair_fails_hc_lg_003() -> None:
    genome = valid_circle_packing_genome()
    _replace_module(genome, "conflict", "conflict_constraint_graph")
    _replace_module(genome, "repair", "repair_pairwise_repulsion")

    assert "HC-LG-003" in _rule_ids(genome)


def test_bilevel_repair_with_incompatible_conflict_fails_hc_lg_004() -> None:
    genome = valid_circle_packing_genome()
    _replace_module(genome, "conflict", "conflict_pairwise_overlap")
    _replace_module(genome, "repair", "repair_bilevel_positions_then_radii")

    assert "HC-LG-004" in _rule_ids(genome)


def test_repair_budget_termination_with_incompatible_repair_fails_hc_tr_001() -> None:
    genome = valid_circle_packing_genome()
    _replace_module(genome, "repair", "repair_pairwise_repulsion")
    _replace_module(genome, "termination", "termination_repair_budget_exhaustion")

    assert "HC-TR-001" in _rule_ids(genome)

