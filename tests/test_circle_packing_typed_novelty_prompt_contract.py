"""Typed novelty prompt contract tests for circle_packing_shinka."""

from __future__ import annotations

import json

import pytest

from experiments.circle_packing_shinka._runtime import design_prompt_contract_v1 as contract
from tests.fixtures.circle_packing_genome import corrective_genome, valid_circle_packing_genome
from tests.test_circle_packing_design_prompt_contract import _base_response, _raw


def _novelty_response(
    operation: str,
    *,
    verdict: str = "NOVELTY_ACCEPTED",
    supported_differences: list[dict] | None = None,
    rejection_reasons: list[dict] | None = None,
) -> dict:
    if supported_differences is None:
        supported_differences = [
            {
                "kind": "slot_change",
                "slot": "growth",
                "message": "The child changes the growth slot.",
            }
        ]
    if rejection_reasons is None:
        rejection_reasons = []
    return {
        "schema_name": "circle_packing_typed_novelty_verdict",
        "schema_version": "1.0",
        "task_family": "circle_packing_shinka",
        "task_name": "unit_square_26",
        "operation": operation,
        "verdict": verdict,
        "supported_differences": supported_differences,
        "rejection_reasons": rejection_reasons,
    }


def test_valid_mutation_novelty_json_parses() -> None:
    candidate = _base_response("mutation")
    response = contract.parse_mutation_novelty_response(
        json.dumps(_novelty_response("mutation")),
        parent_genome=valid_circle_packing_genome(organism_id="parent"),
        candidate_response=candidate,
    )

    assert response["verdict"] == "NOVELTY_ACCEPTED"


def test_valid_crossover_novelty_json_parses() -> None:
    candidate = _base_response("crossover")
    response = contract.parse_crossover_novelty_response(
        json.dumps(
            _novelty_response(
                "crossover",
                supported_differences=[
                    {
                        "kind": "recombination_structure",
                        "slot": "",
                        "message": "The child uses material from both parents.",
                    }
                ],
            )
        ),
        primary_parent_genome=valid_circle_packing_genome(organism_id="primary"),
        secondary_parent_genome=corrective_genome(organism_id="secondary"),
        candidate_response=candidate,
    )

    assert response["verdict"] == "NOVELTY_ACCEPTED"


def test_accepted_novelty_with_empty_rejection_list_passes() -> None:
    contract.validate_novelty_response(_novelty_response("mutation"), "mutation")


def test_rejected_novelty_with_non_empty_rejection_list_passes() -> None:
    response = _novelty_response(
        "mutation",
        verdict="NOVELTY_REJECTED",
        supported_differences=[],
        rejection_reasons=[{"reason_id": "same_parent", "message": "No typed slot changed."}],
    )

    contract.validate_novelty_response(response, "mutation")


def test_accepted_novelty_with_unsupported_empty_differences_fails() -> None:
    response = _novelty_response("mutation", supported_differences=[])

    with pytest.raises(ValueError, match="structural differences"):
        contract.validate_novelty_response(response, "mutation")


def test_novelty_verdict_claiming_unsupported_structure_fails() -> None:
    parent = valid_circle_packing_genome(organism_id="parent")
    candidate = _base_response("mutation")
    candidate["slot_assignments"] = contract.compact_slot_assignments_from_genome(parent)
    candidate["operator_metadata"]["changed_slots"] = ["growth"]
    candidate["operator_metadata"]["preserved_slots"] = [
        "layout",
        "selection",
        "radius_init",
        "conflict",
        "repair",
        "boundary",
        "termination",
    ]

    response = _novelty_response("mutation")
    with pytest.raises(ValueError, match="unsupported"):
        contract.parse_mutation_novelty_response(
            json.dumps(response),
            parent_genome=parent,
            candidate_response=candidate,
        )


def test_markdown_novelty_verdict_fails_strict_json_parser() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        contract.parse_mutation_novelty_response("## NOVELTY_VERDICT\nNOVELTY_ACCEPTED\n")

