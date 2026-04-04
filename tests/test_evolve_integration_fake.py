"""Canonical organism-first integration tests with the fake evaluator."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from omegaconf import OmegaConf

from src.evolve.evolution_loop import EvolutionLoop
from src.evolve.storage import read_json


def _write_baseline(stats_root: Path, exp_name: str, objective_last: float = 1.0, steps: int = 10) -> None:
    exp_dir = stats_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "baseline.json").write_text(
        json.dumps(
            {
                "objective_name": "train_loss",
                "objective_direction": "min",
                "objective_last": objective_last,
                "steps": steps,
            }
        ),
        encoding="utf-8",
    )


def _canonical_cfg(tmp_path: Path, *, max_generations: int, resume: bool) -> object:
    pop_root = tmp_path / "populations"
    stats_root = tmp_path / "stats"
    islands_dir = tmp_path / "islands"
    islands_dir.mkdir(parents=True, exist_ok=True)
    (islands_dir / "gradient_methods.txt").write_text("Favor robust first-order methods.", encoding="utf-8")
    (islands_dir / "second_order.txt").write_text("Favor curvature-aware preconditioning.", encoding="utf-8")

    for exp_name in ("simple_a", "hard_b"):
        _write_baseline(stats_root, exp_name)

    return OmegaConf.create(
        {
            "mode": "evolve",
            "seed": 7,
            "deterministic": False,
            "precision": "fp32",
            "num_workers": 0,
            "safety": {
                "detect_nan": True,
                "abort_on_nan": True,
                "grad_clip_norm": None,
                "log_grad_norm": True,
            },
            "resources": {"num_gpus": 1, "gpu_ids": [0]},
            "paths": {
                "population_root": str(pop_root),
                "data_root": str(tmp_path / "data"),
                "stats_root": str(stats_root),
                "runs_root": str(tmp_path / "runs"),
                "optimizer_guesses_root": str(tmp_path / "optimizer_guesses"),
                "optimizer_results_dirname": "results",
            },
            "experiments": {
                "simple_a": {
                    "enabled": True,
                    "primary_metric": {"direction": "min"},
                    "normalization": {"eps": 1.0e-8},
                },
                "hard_b": {
                    "enabled": True,
                    "primary_metric": {"direction": "min"},
                    "normalization": {"eps": 1.0e-8},
                },
            },
            "evolver": {
                "enabled": True,
                "generation": 0,
                "resume": resume,
                "max_generations": max_generations,
                "fail_fast": False,
                "max_generation_attempts": 1,
                "eval_entrypoint_module": "tests.fixtures.fake_eval",
                "timeout_sec_per_eval": 60,
                "max_retries_per_eval": 0,
                "max_evaluation_jobs": 1,
                "islands": {
                    "dir": str(islands_dir),
                    "seed_organisms_per_island": 1,
                    "max_organisms_per_island": 1,
                },
                "reproduction": {
                    "offspring_per_generation": 2,
                    "operator_selection_strategy": "deterministic",
                    "operator_weights": {
                        "within_island_crossover": 0.0,
                        "inter_island_crossover": 1.0,
                        "mutation": 0.0,
                    },
                    "island_sampling": {
                        "within_island_crossover": "unified",
                        "inter_island_crossover": "unified",
                        "mutation": "unified",
                    },
                },
                "operators": {
                    "mutation": {
                        "gene_removal_probability": 0.2,
                        "parent_selection_softmax_temperature": 1.0,
                    },
                    "crossover": {
                        "primary_parent_gene_inheritance_probability": 0.7,
                        "parent_selection_softmax_temperature": 1.0,
                    },
                },
                "phases": {
                    "simple": {
                        "eval_mode": "smoke",
                        "timeout_sec_per_eval": 60,
                        "top_k_per_island": 1,
                        "experiments": ["simple_a"],
                        "allocation": {"enabled": False},
                    },
                    "great_filter": {
                        "enabled": True,
                        "interval_generations": 1,
                        "eval_mode": "full",
                        "timeout_sec_per_eval": 60,
                        "top_h_per_island": 1,
                        "experiments": ["hard_b"],
                        "allocation": {"enabled": False},
                    },
                },
                "llm": {
                    "provider": "mock",
                    "model": "mock-model",
                    "temperature": 0.0,
                    "max_output_tokens": 512,
                    "reasoning_effort": None,
                    "seed": 123,
                    "fallback_to_chat_completions": True,
                },
            },
        }
    )


def test_canonical_organism_first_pipeline_fake(tmp_path: Path) -> None:
    cfg = _canonical_cfg(tmp_path, max_generations=1, resume=False)

    summary = asyncio.run(EvolutionLoop(cfg).run())

    assert summary["total_generations"] == 1
    assert summary["active_population_size"] == 2

    pop_root = Path(str(cfg.paths.population_root))
    manifest = read_json(pop_root / "population_manifest.json")
    assert manifest["generation"] == 1
    assert len(manifest["active_organisms"]) == 2

    active_dirs = [Path(entry["organism_dir"]) for entry in manifest["active_organisms"]]
    for org_dir in active_dirs:
        assert org_dir.exists()
        assert (org_dir / "optimizer.py").exists()
        assert (org_dir / "genetic_code.md").exists()
        assert (org_dir / "lineage.json").exists()
        assert (org_dir / "summary.json").exists()

        summary_payload = read_json(org_dir / "summary.json")
        phase_results = summary_payload["phase_results"]
        assert list(phase_results["simple"]["selected_experiments"]) == ["simple_a"]
        assert list(phase_results["hard"]["selected_experiments"]) == ["hard_b"]
        assert phase_results["simple"]["allocation_snapshot"]["inclusion_prob"]["simple_a"] == 1.0
        assert phase_results["hard"]["allocation_snapshot"]["inclusion_prob"]["hard_b"] == 1.0
        assert summary_payload["simple_reward"] is not None
        assert summary_payload["hard_reward"] is not None


def test_canonical_multigeneration_resume_keeps_manifest_and_island_boundaries(tmp_path: Path) -> None:
    cfg_first = _canonical_cfg(tmp_path, max_generations=1, resume=False)
    asyncio.run(EvolutionLoop(cfg_first).run())

    cfg_resume = _canonical_cfg(tmp_path, max_generations=2, resume=True)
    summary = asyncio.run(EvolutionLoop(cfg_resume).run())

    assert summary["total_generations"] == 2

    pop_root = Path(str(cfg_resume.paths.population_root))
    manifest = read_json(pop_root / "population_manifest.json")
    assert manifest["generation"] == 2
    assert {entry["island_id"] for entry in manifest["active_organisms"]} == {
        "gradient_methods",
        "second_order",
    }
    for entry in manifest["active_organisms"]:
        assert "/island_" in entry["organism_dir"]
