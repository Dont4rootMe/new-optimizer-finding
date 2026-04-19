"""Notebook helper tests for manual prompting and evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evolve.manual_pipeline import (
    build_manual_crossover_prompts,
    build_manual_implementation_prompts,
    build_manual_mutation_prompts,
    load_manual_pipeline_context,
    run_manual_simple_evaluation,
)
from src.evolve.storage import parse_genetic_code_text

ROOT = Path(__file__).resolve().parents[1]

_GENETIC_CODE = """## CORE_GENES
- Adaptive first-moment tracking for noisy gradients
- Per-parameter normalization using a running second moment
- Late-training step shrinkage to stabilize convergence

## INTERACTION_NOTES
The moment and normalization signals should be coordinated.

## COMPUTE_NOTES
Only low-overhead tensor state is allowed.
"""

_LINEAGE = """[
  {
    "generation": 3,
    "operator": "mutation",
    "change_description": "Added late-training decay after a strong early-learning parent.",
    "simple_score": 1.2,
    "hard_score": null,
    "cross_island": false,
    "father_island_id": null
  }
]"""

_CIRCLE_PACKING_CODE = """import numpy as np

def run_packing():
    centers = np.asarray([
        [0.10, 0.15], [0.24, 0.15], [0.38, 0.15], [0.52, 0.15], [0.66, 0.15], [0.80, 0.15], [0.94, 0.15],
        [0.17, 0.33], [0.31, 0.33], [0.45, 0.33], [0.59, 0.33], [0.73, 0.33], [0.87, 0.33],
        [0.10, 0.51], [0.24, 0.51], [0.38, 0.51], [0.52, 0.51], [0.66, 0.51], [0.80, 0.51], [0.94, 0.51],
        [0.17, 0.69], [0.31, 0.69], [0.45, 0.69], [0.59, 0.69], [0.73, 0.69], [0.87, 0.69],
    ], dtype=float)
    radii = np.full(26, 0.04, dtype=float)
    reported_sum = float(np.sum(radii))
    return centers, radii, reported_sum
"""


def test_parse_genetic_code_text_accepts_markdown_payload() -> None:
    payload = parse_genetic_code_text(_GENETIC_CODE)

    assert payload["core_genes"][0] == "Adaptive first-moment tracking for noisy gradients"
    assert payload["interaction_notes"] == "The moment and normalization signals should be coordinated."
    assert payload["compute_notes"] == "Only low-overhead tensor state is allowed."


def test_manual_pipeline_requires_explicit_experiment_for_multi_experiment_config() -> None:
    with pytest.raises(ValueError, match="multiple experiments"):
        load_manual_pipeline_context(config_name="optimization_survey")


def test_manual_pipeline_accepts_short_config_name_alias() -> None:
    context = load_manual_pipeline_context(
        config_name="circle_packing_shinka",
        experiment_name="unit_square_26",
    )

    assert context.config_name == "config_circle_packing_shinka"
    assert context.experiment_name == "unit_square_26"


def test_manual_prompt_helpers_build_expected_sections() -> None:
    context = load_manual_pipeline_context(
        config_name="config_optimization_survey",
        experiment_name="xor_mlp",
    )

    crossover = build_manual_crossover_prompts(
        context,
        mother_genetic_code_text=_GENETIC_CODE,
        father_genetic_code_text=_GENETIC_CODE.replace("late-training", "oscillation-aware"),
        mother_lineage_json=_LINEAGE,
        father_lineage_json="[]",
        seed=7,
    )
    mutation = build_manual_mutation_prompts(
        context,
        parent_genetic_code_text=_GENETIC_CODE,
        parent_lineage_json=_LINEAGE,
        seed=7,
    )
    implementation = build_manual_implementation_prompts(
        context,
        organism_genetic_code_text=_GENETIC_CODE,
        novelty_summary="Combined stable normalization with a more aggressive late-stage decay.",
    )

    assert crossover["child_gene_pool"]
    assert "## CHANGE_DESCRIPTION" in crossover["system_prompt"]
    assert "=== PRIMARY PARENT GENETIC CODE (REFERENCE ONLY) ===" in crossover["user_prompt"]
    assert "=== SECONDARY PARENT GENETIC CODE (REFERENCE ONLY) ===" in crossover["user_prompt"]
    assert mutation["inherited_genes"]
    assert "=== EXCLUDED IDEAS ===" in mutation["user_prompt"]
    assert "=== ORGANISM CHANGE_DESCRIPTION ===" in implementation["user_prompt"]
    assert "=== COMPILATION MODE ===\nFULL" in implementation["user_prompt"]
    assert "Combined stable normalization" in implementation["user_prompt"]


def test_manual_simple_evaluation_runs_circle_packing_candidate(tmp_path: Path) -> None:
    implementation_path = tmp_path / "implementation.py"
    implementation_path.write_text(_CIRCLE_PACKING_CODE, encoding="utf-8")

    context = load_manual_pipeline_context(
        config_name="config_circle_packing_shinka",
        overrides=[
            f"paths.population_root={tmp_path / 'populations'}",
            f"paths.stats_root={tmp_path / 'stats'}",
            f"paths.data_root={tmp_path / 'data'}",
            f"paths.runs_root={tmp_path / 'runs'}",
            f"paths.api_platform_runtime_root={tmp_path / '.api_platform_runtime'}",
        ],
    )
    report = run_manual_simple_evaluation(
        context,
        implementation_path=implementation_path,
        mode="smoke",
        device="cpu",
        precision="fp32",
    )

    assert report["status"] == "ok"
    assert report["score"] == pytest.approx(1.04)
    assert report["manual_experiment_name"] == "unit_square_26"
    assert report["source_implementation_path"] == str(implementation_path.resolve())
