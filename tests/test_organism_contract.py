"""Canonical organism contract and lineage tests."""

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


def _base_parsed() -> dict[str, str]:
    return {
        "CORE_GENES": "- adaptive momentum\n- warmup schedule\n- gradient clipping",
        "INTERACTION_NOTES": "Momentum and warmup are coordinated.",
        "COMPUTE_NOTES": "No extra step_fn calls.",
        "CHANGE_DESCRIPTION": "Initial organism build.",
        "IMPORTS": "import math",
        "INIT_BODY": "self.model = model\nself.max_steps = max_steps",
        "STEP_BODY": "del weights, grads, activations, step_fn",
        "ZERO_GRAD_BODY": "pass",
    }


def _build(tmp_path: Path, parsed: dict[str, str], **overrides):
    org_dir = tmp_path / "org"
    org_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "parsed": parsed,
        "organism_id": "org01",
        "island_id": "gradient_methods",
        "generation": 0,
        "mother_id": None,
        "father_id": None,
        "operator": "seed",
        "org_dir": org_dir,
        "model_name": "mock-model",
        "prompt_hash": "abc",
        "seed": 123,
        "timestamp": "2026-01-01T00:00:00Z",
        "parent_lineage": [],
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


def test_build_organism_rejects_noncanonical_gene_section_response(tmp_path: Path) -> None:
    parsed = {
        "GENE_POOL": "momentum; warmup; clipping",
        "CHANGE_DESCRIPTION": "noncanonical response",
        "IMPORTS": "import math",
        "INIT_BODY": "self.model = model",
        "STEP_BODY": "pass",
        "ZERO_GRAD_BODY": "pass",
    }

    with pytest.raises(ValueError, match="CORE_GENES"):
        _build(tmp_path, parsed)


def test_new_lineage_write_omits_aggregate_score_and_summary_mentions_cross_island(tmp_path: Path) -> None:
    org = _build(
        tmp_path,
        _base_parsed(),
        mother_id="mother01",
        father_id="father01",
        operator="crossover",
        cross_island=True,
        father_island_id="second_order",
        gene_diff_summary="Maternal genes preserved: adaptive momentum. Paternal genes introduced: diagonal preconditioning. Major rewrites: (none).",
    )
    update_latest_lineage_entry(
        org,
        phase="simple",
        phase_score=1.25,
        selected_experiments=["simple_a"],
    )

    lineage = read_lineage(Path(org.lineage_path))
    latest = lineage[-1]
    assert "aggregate_score" not in latest
    assert latest["cross_island"] is True
    assert latest["father_island_id"] == "second_order"

    summary = format_lineage_summary(lineage)
    assert "cross_island=true:second_order" in summary


def test_old_lineage_with_aggregate_score_still_loads(tmp_path: Path) -> None:
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


def test_canonical_genetic_code_read_requires_genetic_code_md(tmp_path: Path) -> None:
    org_dir = tmp_path / "org_missing_genes"
    org_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="genetic code"):
        read_genetic_code(org_dir / "genetic_code.md")


def test_canonical_lineage_read_requires_lineage_json(tmp_path: Path) -> None:
    org_dir = tmp_path / "org_missing_lineage"
    org_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="lineage"):
        read_lineage(org_dir / "lineage.json")


def test_canonical_organism_meta_read_rejects_missing_canonical_artifacts(tmp_path: Path) -> None:
    org_dir = tmp_path / "org_meta"
    org_dir.mkdir(parents=True, exist_ok=True)
    (org_dir / "optimizer.py").write_text("def build_optimizer(model, max_steps):\n    return None\n", encoding="utf-8")
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
            "optimizer_path": str(org_dir / "optimizer.py"),
            "lineage_path": str(org_dir / "lineage.json"),
            "organism_dir": str(org_dir),
            "status": "pending",
            "model_name": "mock-model",
            "prompt_hash": "abc",
            "seed": 123,
        },
    )

    with pytest.raises(FileNotFoundError, match="genetic code"):
        read_organism_meta(org_dir)


def test_canonical_organism_meta_read_rejects_noncanonical_meta_shape(tmp_path: Path) -> None:
    org_dir = tmp_path / "org_noncanonical_meta"
    org_dir.mkdir(parents=True, exist_ok=True)
    (org_dir / "optimizer.py").write_text("def build_optimizer(model, max_steps):\n    return None\n", encoding="utf-8")
    write_json(
        org_dir / "organism.json",
        {
            "organism_id": "org_noncanonical_meta",
            "generation": 0,
            "current_generation_active": 0,
            "timestamp": "2026-01-01T00:00:00Z",
            "operator": "seed",
            "genetic_code_path": str(org_dir / "genetic_code.md"),
            "optimizer_path": str(org_dir / "optimizer.py"),
            "lineage_path": str(org_dir / "lineage.json"),
            "organism_dir": str(org_dir),
            "status": "pending",
            "model_name": "mock-model",
            "prompt_hash": "abc",
            "seed": 123,
        },
    )

    with pytest.raises(ValueError, match="generation_created|island_id"):
        read_organism_meta(org_dir)
