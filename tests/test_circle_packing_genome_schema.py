"""Schema tests for the circle_packing_shinka typed segmented genome."""

from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.circle_packing_shinka._runtime import genome_schema_v1 as schema
from src.organisms.hypothesis_artifacts import validate_canonical_genome
from tests.fixtures.circle_packing_genome import valid_circle_packing_genome


def _assert_invalid(genome: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_canonical_genome(genome, schema)


def test_fully_valid_minimal_genome_passes() -> None:
    validate_canonical_genome(valid_circle_packing_genome(), schema)


def test_missing_slot_fails() -> None:
    genome = valid_circle_packing_genome()
    del genome["slots"]["repair"]

    _assert_invalid(genome, "missing slots: repair")


def test_extra_slot_fails() -> None:
    genome = valid_circle_packing_genome()
    genome["slots"]["extra"] = deepcopy(genome["slots"]["layout"])
    genome["slots"]["extra"]["slot"] = "extra"

    _assert_invalid(genome, "extra slots: extra")


def test_wrong_module_type_for_slot_fails() -> None:
    genome = valid_circle_packing_genome()
    genome["slots"]["layout"]["module_type"] = "radius_growth_policy"

    _assert_invalid(genome, "module_type")


def test_wrong_linkage_group_fails() -> None:
    genome = valid_circle_packing_genome()
    genome["slots"]["layout"]["linkage_group"] = "dynamics_block"

    _assert_invalid(genome, "linkage_group")


def test_invalid_state_symbol_fails() -> None:
    genome = valid_circle_packing_genome()
    genome["slots"]["growth"]["reads_state"] = ["private_radius_buffer"]

    _assert_invalid(genome, "illegal state symbol")


def test_duplicate_module_id_fails() -> None:
    genome = valid_circle_packing_genome()
    genome["slots"]["selection"]["module_id"] = genome["slots"]["layout"]["module_id"]

    _assert_invalid(genome, "duplicate module_id")


def test_non_boolean_boolean_parameter_fails() -> None:
    genome = valid_circle_packing_genome()
    genome["slots"]["layout"]["parameterization"] = [
        {
            "name": "use_boundary_bias",
            "value_kind": "boolean",
            "value": "true",
        }
    ]

    _assert_invalid(genome, "JSON boolean")


def test_numeric_parameter_value_fails() -> None:
    genome = valid_circle_packing_genome()
    genome["slots"]["layout"]["parameterization"] = [
        {
            "name": "aggression",
            "value_kind": "ordinal_token",
            "value": 1,
        }
    ]

    _assert_invalid(genome, "numeric JSON values")


def test_extra_top_level_key_fails() -> None:
    genome = valid_circle_packing_genome()
    genome["legacy_genetic_code"] = "not allowed"

    _assert_invalid(genome, "extra top-level keys|extra keys")
