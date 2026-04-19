"""Tests for compact slot assignments and materialized slot decomposition."""

from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.circle_packing_shinka._runtime import genome_schema_v1 as schema
from experiments.circle_packing_shinka._runtime import module_library_v1 as library
from src.organisms.hypothesis_artifacts import validate_canonical_genome
from tests.fixtures.circle_packing_genome import (
    valid_global_hypothesis,
    valid_render_fields,
    valid_slot_assignments,
)


def _materialize(assignments: dict) -> dict:
    return schema.materialize_circle_packing_genome_v1(
        organism_id="org_alpha",
        global_hypothesis=valid_global_hypothesis(),
        slot_assignments=assignments,
        render_fields=valid_render_fields(),
    )


def test_compact_valid_slot_assignment_materializes_into_valid_full_genome() -> None:
    genome = _materialize(valid_slot_assignments())

    validate_canonical_genome(genome, schema)
    assert list(genome["slots"].keys()) == schema.SLOT_ORDER
    assert genome["slots"]["layout"]["module_key"] == "layout_triangular_lattice"
    assert genome["slots"]["layout"]["family"] == "triangular_lattice_layout"


def test_unknown_module_key_fails() -> None:
    assignments = valid_slot_assignments()
    assignments["layout"]["module_key"] = "layout_not_in_library"

    with pytest.raises(ValueError, match="Unknown module_key"):
        _materialize(assignments)


def test_valid_module_key_in_wrong_slot_fails() -> None:
    assignments = valid_slot_assignments()
    assignments["selection"]["module_key"] = "layout_triangular_lattice"

    with pytest.raises(ValueError, match="belongs to slot 'layout'"):
        _materialize(assignments)


def test_missing_slot_in_compact_assignment_fails() -> None:
    assignments = valid_slot_assignments()
    del assignments["repair"]

    with pytest.raises(ValueError, match="missing slots: repair"):
        _materialize(assignments)


def test_extra_slot_in_compact_assignment_fails() -> None:
    assignments = valid_slot_assignments()
    assignments["extra"] = deepcopy(assignments["layout"])

    with pytest.raises(ValueError, match="extra slots: extra"):
        _materialize(assignments)


def test_malformed_parameterization_fails() -> None:
    assignments = valid_slot_assignments()
    assignments["layout"]["parameterization"] = "not a list"

    with pytest.raises(ValueError, match="parameterization"):
        _materialize(assignments)


def test_illegal_parameter_value_fails() -> None:
    assignments = valid_slot_assignments()
    assignments["layout"]["parameterization"][0]["value"] = "not_allowed"

    with pytest.raises(ValueError, match="must be one of"):
        _materialize(assignments)


def test_missing_required_parameter_fails() -> None:
    assignments = valid_slot_assignments()
    assignments["layout"]["parameterization"] = []

    with pytest.raises(ValueError, match="missing required parameters"):
        _materialize(assignments)


def test_extra_parameter_fails() -> None:
    assignments = valid_slot_assignments()
    assignments["layout"]["parameterization"].append(
        {
            "name": "unexpected_bias",
            "value_kind": "categorical_token",
            "value": "balanced",
        }
    )

    with pytest.raises(ValueError, match="unexpected parameter name"):
        _materialize(assignments)


def test_materialized_metadata_drift_fails_validation() -> None:
    genome = _materialize(valid_slot_assignments())
    genome["slots"]["layout"]["family"] = "invented_family"

    with pytest.raises(ValueError, match="library-controlled fields"):
        validate_canonical_genome(genome, schema)


def test_materialized_instance_matches_library_definition() -> None:
    genome = _materialize(valid_slot_assignments())
    definition = library.get_module_definition(genome["slots"]["layout"]["module_key"])

    assert genome["slots"]["layout"]["preconditions"] == definition["required_preconditions"]
    assert genome["slots"]["layout"]["postconditions"] == definition["guaranteed_postconditions"]
    assert genome["slots"]["layout"]["hypothesis"] == definition["hypothesis_template"]
