"""Soft compatibility tests for circle_packing_shinka genome v1."""

from __future__ import annotations

from experiments.circle_packing_shinka._runtime import compatibility_contract_v1 as compatibility
from experiments.circle_packing_shinka._runtime import genome_schema_v1 as schema
from experiments.circle_packing_shinka._runtime import module_library_v1 as library
from tests.fixtures.circle_packing_genome import (
    corrective_genome,
    valid_circle_packing_genome,
)


def _replace_module(genome: dict, slot: str, module_key: str) -> None:
    genome["slots"][slot] = schema.materialize_module_instance(
        slot=slot,
        module_key=module_key,
        module_id=f"{genome['organism_id']}_{slot}_replacement",
        parameterization=library.default_parameterization_for_module(module_key),
    )


def _component(report: dict, component_id: str) -> dict:
    matches = [
        component
        for component in report["components"]
        if component["component_id"] == component_id
    ]
    assert len(matches) == 1
    return matches[0]


def test_soft_score_is_deterministic() -> None:
    genome = valid_circle_packing_genome()

    assert compatibility.compute_soft_compatibility(genome) == compatibility.compute_soft_compatibility(genome)


def test_hard_compatible_genomes_get_detailed_component_list() -> None:
    genome = corrective_genome()
    report = compatibility.build_circle_packing_compatibility_report(genome)

    assert report["hard_compatibility"]["is_compatible"] is True
    component_ids = {
        component["component_id"]
        for component in report["soft_compatibility"]["components"]
    }
    assert "SC-SLOT-layout" in component_ids
    assert "SC-PAIR-conflict-repair" in component_ids
    assert "SC-BLOCK-dynamics" in component_ids


def test_hard_incompatible_genomes_still_get_informational_soft_score() -> None:
    genome = valid_circle_packing_genome()
    _replace_module(genome, "growth", "growth_role_conditioned_growth")

    report = compatibility.build_circle_packing_compatibility_report(genome)
    assert report["hard_compatibility"]["is_compatible"] is False
    assert any(
        component["component_id"] == "SC-HARD-INCOMPATIBLE"
        for component in report["soft_compatibility"]["components"]
    )


def test_explicit_pairwise_bonus_is_applied() -> None:
    genome = valid_circle_packing_genome()
    report = compatibility.compute_soft_compatibility(genome)

    assert _component(report, "SC-PAIR-layout-radius_init")["score"] == 0.5


def test_explicit_pairwise_penalty_is_applied() -> None:
    genome = valid_circle_packing_genome()
    _replace_module(genome, "layout", "layout_jittered_low_discrepancy")

    report = compatibility.compute_soft_compatibility(genome)

    assert _component(report, "SC-PAIR-layout-radius_init")["score"] == -0.25


def test_block_bonus_and_penalty_are_applied() -> None:
    bonus_report = compatibility.compute_soft_compatibility(corrective_genome())
    assert _component(bonus_report, "SC-BLOCK-dynamics")["score"] == 0.5

    penalty_genome = valid_circle_packing_genome()
    _replace_module(penalty_genome, "growth", "growth_uniform_additive")
    _replace_module(penalty_genome, "conflict", "conflict_constraint_graph")
    _replace_module(penalty_genome, "repair", "repair_pairwise_repulsion")
    _replace_module(penalty_genome, "boundary", "boundary_hard_containment")

    penalty_report = compatibility.compute_soft_compatibility(penalty_genome)

    assert _component(penalty_report, "SC-BLOCK-dynamics")["score"] == -0.5

