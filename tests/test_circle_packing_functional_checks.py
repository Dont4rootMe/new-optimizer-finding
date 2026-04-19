"""Functional check tests for circle_packing_shinka genome v1."""

from __future__ import annotations

from pathlib import Path

from experiments.circle_packing_shinka._runtime import compatibility_contract_v1 as compatibility
from experiments.circle_packing_shinka._runtime import genome_schema_v1 as schema
from experiments.circle_packing_shinka._runtime import module_library_v1 as library
from src.organisms.hypothesis_artifacts import (
    read_compatibility_report,
    read_functional_checks_report,
    write_compatibility_report,
    write_functional_checks_report,
)
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


def _check_statuses(report: dict) -> dict[str, str]:
    return {
        check["check_id"]: check["status"]
        for check in report["checks"]
    }


def test_representative_valid_genomes_pass_all_functional_checks() -> None:
    for genome in (structured_role_genome(), corrective_genome()):
        report = compatibility.build_circle_packing_functional_checks(genome)
        assert report["checks_passed"] is True
        assert _check_statuses(report) == {
            "FC-001": "passed",
            "FC-002": "passed",
            "FC-003": "passed",
            "FC-004": "passed",
            "FC-005": "passed",
        }


def test_missing_executable_stage_chain_fails_fc_001() -> None:
    genome = valid_circle_packing_genome()
    genome["slots"]["selection"]["writes_state"] = []

    report = compatibility.build_circle_packing_functional_checks(genome)

    assert _check_statuses(report)["FC-001"] == "failed"
    assert report["checks_passed"] is False


def test_missing_feasibility_path_fails_fc_002() -> None:
    genome = valid_circle_packing_genome()
    genome["slots"]["repair"]["reads_state"] = []
    genome["slots"]["repair"]["writes_state"] = []

    report = compatibility.build_circle_packing_functional_checks(genome)

    assert _check_statuses(report)["FC-002"] == "failed"


def test_self_defeating_conservative_combination_fails_fc_003() -> None:
    genome = valid_circle_packing_genome()
    _replace_module(genome, "radius_init", "radius_init_uniform_small")
    _replace_module(genome, "repair", "repair_shrink_on_persistent_failure")
    _replace_module(genome, "termination", "termination_fixed_stability")

    report = compatibility.build_circle_packing_functional_checks(genome)

    assert _check_statuses(report)["FC-003"] == "failed"


def test_isolated_role_aware_module_fails_fc_004() -> None:
    genome = valid_circle_packing_genome()
    _replace_module(genome, "radius_init", "radius_init_role_based_boundary_vs_interior")

    report = compatibility.build_circle_packing_functional_checks(genome)

    assert _check_statuses(report)["FC-004"] == "failed"


def test_uninformative_termination_fails_fc_005() -> None:
    genome = valid_circle_packing_genome()
    _replace_module(genome, "termination", "termination_plateau_detection")
    genome["slots"]["termination"]["reads_state"] = ["feasibility_status"]

    report = compatibility.build_circle_packing_functional_checks(genome)

    assert _check_statuses(report)["FC-005"] == "failed"


def test_stage_three_artifact_io_roundtrips_deterministically(tmp_path: Path) -> None:
    genome = valid_circle_packing_genome()
    compatibility_report = compatibility.build_circle_packing_compatibility_report(genome)
    functional_checks = compatibility.build_circle_packing_functional_checks(genome)

    write_compatibility_report(tmp_path, compatibility_report)
    write_functional_checks_report(tmp_path, functional_checks)

    assert read_compatibility_report(tmp_path) == compatibility_report
    assert read_functional_checks_report(tmp_path) == functional_checks
    assert (tmp_path / "compatibility_report.json").read_text(encoding="utf-8").endswith("\n")
    assert (tmp_path / "functional_checks.json").read_text(encoding="utf-8").endswith("\n")

