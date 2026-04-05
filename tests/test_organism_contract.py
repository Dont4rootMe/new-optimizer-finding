"""Generic organism contract and lineage tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evolve.storage import read_genetic_code, read_lineage, read_organism_meta, write_json
from src.organisms.organism import (
    build_organism_from_response,
    format_lineage_summary,
    update_latest_lineage_entry,
)


def _implementation_code() -> str:
    return (
        "import torch.nn as nn\n\n"
        "OPTIMIZER_NAME = 'TestOpt'\n\n"
        "class TestOpt:\n"
        "    def __init__(self, model: nn.Module, max_steps: int) -> None:\n"
        "        self.model = model\n"
        "        self.max_steps = max_steps\n\n"
        "    def step(self, weights, grads, activations, step_fn) -> None:\n"
        "        del weights, grads, activations, step_fn\n\n"
        "    def zero_grad(self, set_to_none: bool = True) -> None:\n"
        "        del set_to_none\n\n"
        "def build_optimizer(model: nn.Module, max_steps: int):\n"
        "    return TestOpt(model, max_steps)\n"
    )


def _base_parsed() -> dict[str, str]:
    return {
        "CORE_GENES": "- adaptive momentum\n- warmup schedule\n- gradient clipping",
        "INTERACTION_NOTES": "Momentum and warmup are coordinated.",
        "COMPUTE_NOTES": "No extra step_fn calls.",
        "CHANGE_DESCRIPTION": "Initial organism build.",
    }


def _build(tmp_path: Path, parsed: dict[str, str], **overrides):
    org_dir = tmp_path / "org"
    org_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "parsed": parsed,
        "implementation_code": _implementation_code(),
        "organism_id": "org01",
        "island_id": "gradient_methods",
        "generation": 0,
        "mother_id": None,
        "father_id": None,
        "operator": "seed",
        "org_dir": org_dir,
        "llm_route_id": "mock",
        "llm_provider": "mock",
        "provider_model_id": "mock-model",
        "prompt_hash": "abc",
        "seed": 123,
        "timestamp": "2026-01-01T00:00:00Z",
        "parent_lineage": [],
        "ancestor_ids": [],
    }
    kwargs.update(overrides)
    return build_organism_from_response(**kwargs)


def test_build_organism_rejects_too_thin_core_genes(tmp_path: Path) -> None:
    parsed = _base_parsed()
    parsed["CORE_GENES"] = "- momentum\n- warmup\n- clipping"

    with pytest.raises(ValueError, match="too thin|at least 3"):
        _build(tmp_path, parsed)


def test_build_organism_rejects_missing_sections(tmp_path: Path) -> None:
    parsed = _base_parsed()
    del parsed["INTERACTION_NOTES"]

    with pytest.raises(ValueError, match="INTERACTION_NOTES"):
        _build(tmp_path, parsed)


def test_build_organism_rejects_empty_implementation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty implementation.py"):
        _build(tmp_path, _base_parsed(), implementation_code="   \n")


def test_new_lineage_write_omits_diff_summary_and_summary_mentions_cross_island(tmp_path: Path) -> None:
    org = _build(
        tmp_path,
        _base_parsed(),
        mother_id="mother01",
        father_id="father01",
        operator="crossover",
        cross_island=True,
        father_island_id="second_order",
    )
    update_latest_lineage_entry(
        org,
        phase="simple",
        phase_score=1.25,
        selected_experiments=["simple_a"],
    )

    lineage = read_lineage(Path(org.lineage_path))
    latest = lineage[-1]
    assert "gene_diff_summary" not in latest
    assert latest["cross_island"] is True
    assert latest["father_island_id"] == "second_order"
    assert latest["simple_score"] == 1.25

    summary = format_lineage_summary(lineage)
    assert "cross_island=true:second_order" in summary
    assert "change=Initial organism build." in summary


def test_old_lineage_with_legacy_fields_still_loads(tmp_path: Path) -> None:
    org_dir = tmp_path / "org_old"
    org_dir.mkdir(parents=True, exist_ok=True)
    lineage_path = org_dir / "lineage.json"
    lineage_path.write_text(
        json.dumps(
            [
                {
                    "generation": 0,
                    "operator": "seed",
                    "mother_id": None,
                    "father_id": None,
                    "change_description": "historical",
                    "gene_diff_summary": "historical",
                    "aggregate_score": 1.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    lineage = read_lineage(lineage_path)
    assert lineage[0]["aggregate_score"] == 1.0


def test_canonical_genetic_code_read_rejects_malformed_sections(tmp_path: Path) -> None:
    org_dir = tmp_path / "org_malformed_genes"
    org_dir.mkdir(parents=True, exist_ok=True)
    path = org_dir / "genetic_code.md"
    path.write_text("adaptive momentum; warmup schedule; gradient clipping", encoding="utf-8")

    with pytest.raises(ValueError, match="required sections"):
        read_genetic_code(path)


def test_canonical_organism_meta_read_rejects_missing_canonical_artifacts(tmp_path: Path) -> None:
    org_dir = tmp_path / "org_meta"
    org_dir.mkdir(parents=True, exist_ok=True)
    (org_dir / "implementation.py").write_text(_implementation_code(), encoding="utf-8")
    write_json(
        org_dir / "organism.json",
        {
            "organism_id": "org_meta",
            "island_id": "gradient_methods",
            "generation_created": 0,
            "current_generation_active": 0,
            "timestamp": "2026-01-01T00:00:00Z",
            "mother_id": None,
            "father_id": None,
            "operator": "seed",
            "genetic_code_path": str(org_dir / "genetic_code.md"),
            "implementation_path": str(org_dir / "implementation.py"),
            "lineage_path": str(org_dir / "lineage.json"),
            "organism_dir": str(org_dir),
            "ancestor_ids": [],
            "experiment_report_index": {},
            "status": "pending",
            "llm_route_id": "mock",
            "llm_provider": "mock",
            "provider_model_id": "mock-model",
            "prompt_hash": "abc",
            "seed": 123,
        },
    )

    with pytest.raises(FileNotFoundError, match="genetic code"):
        read_organism_meta(org_dir)


def test_canonical_organism_meta_read_rejects_noncanonical_meta_shape(tmp_path: Path) -> None:
    org_dir = tmp_path / "org_noncanonical_meta"
    org_dir.mkdir(parents=True, exist_ok=True)
    (org_dir / "implementation.py").write_text(_implementation_code(), encoding="utf-8")
    write_json(
        org_dir / "organism.json",
        {
            "organism_id": "org_noncanonical_meta",
            "generation": 0,
            "current_generation_active": 0,
            "timestamp": "2026-01-01T00:00:00Z",
            "operator": "seed",
            "genetic_code_path": str(org_dir / "genetic_code.md"),
            "implementation_path": str(org_dir / "implementation.py"),
            "lineage_path": str(org_dir / "lineage.json"),
            "organism_dir": str(org_dir),
            "status": "pending",
            "llm_route_id": "mock",
            "llm_provider": "mock",
            "provider_model_id": "mock-model",
            "prompt_hash": "abc",
            "seed": 123,
        },
    )

    with pytest.raises(ValueError, match="generation_created|island_id"):
        read_organism_meta(org_dir)
