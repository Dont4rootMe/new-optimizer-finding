"""Canonical organism-first integration tests with the fake evaluator."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.evolve.evolution_loop import EvolutionLoop
from src.evolve.storage import read_json


def _write_baseline(stats_root: Path, exp_name: str, objective_last: float = 1.0, steps: int = 10) -> None:
    exp_dir = stats_root / "optimization_survey" / exp_name
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
            "resources": {"evaluation": {"gpu_ranks": [0], "cpu_parallel_jobs": 1}},
            "paths": {
                "population_root": str(pop_root),
                "data_root": str(tmp_path / "data"),
                "stats_root": str(stats_root),
                "runs_root": str(tmp_path / "runs"),
                "candidate_results_dirname": "results",
                "api_platform_runtime_root": str(tmp_path / ".api_platform_runtime"),
            },
            "api_platforms": {"mock": {"_target_": "api_platforms.mock.platform.build_platform"}},
            "experiments": {
                "simple_a": {
                    "enabled": True,
                    "name": "simple_a",
                    "need_cuda": False,
                    "_target_": "tests.fixtures.fake_runner.FakeExperimentEvaluator",
                    "baseline": {"profile_path": str(stats_root / "optimization_survey" / "simple_a" / "baseline.json")},
                    "normalization": {"eps": 1.0e-8},
                    "compute": {"device": "cpu", "precision": "fp32", "num_workers": 0, "smoke_steps": 5, "max_steps": 5},
                },
                "hard_b": {
                    "enabled": True,
                    "name": "hard_b",
                    "need_cuda": False,
                    "_target_": "tests.fixtures.fake_runner.FakeExperimentEvaluator",
                    "baseline": {"profile_path": str(stats_root / "optimization_survey" / "hard_b" / "baseline.json")},
                    "normalization": {"eps": 1.0e-8},
                    "compute": {"device": "cpu", "precision": "fp32", "num_workers": 0, "smoke_steps": 5, "max_steps": 5},
                },
            },
            "evolver": {
                "resume": resume,
                "max_generations": max_generations,
                "max_retries_per_eval": 0,
                "creation": {
                    "max_attempts_per_organism": 1,
                    "max_parallel_organisms": 1,
                },
                "islands": {
                    "dir": str(islands_dir),
                    "seed_organisms_per_island": 1,
                    "max_organisms_per_island": 1,
                },
                "prompts": {
                    "project_context": "conf/experiments/optimization_survey/prompts/shared/project_context.txt",
                    "seed_system": "conf/experiments/optimization_survey/prompts/seed/system.txt",
                    "seed_user": "conf/experiments/optimization_survey/prompts/seed/user.txt",
                    "mutation_system": "conf/experiments/optimization_survey/prompts/mutation/system.txt",
                    "mutation_user": "conf/experiments/optimization_survey/prompts/mutation/user.txt",
                    "crossover_system": "conf/experiments/optimization_survey/prompts/crossover/system.txt",
                    "crossover_user": "conf/experiments/optimization_survey/prompts/crossover/user.txt",
                    "implementation_system": "conf/experiments/optimization_survey/prompts/implementation/system.txt",
                    "implementation_user": "conf/experiments/optimization_survey/prompts/implementation/user.txt",
                    "implementation_template": "conf/experiments/optimization_survey/prompts/implementation/template.txt",
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
                "llm": {"route_weights": {"mock": 1.0}, "seed": 123},
            },
        }
    )


def test_canonical_organism_first_pipeline_fake(tmp_path: Path) -> None:
    cfg = _canonical_cfg(tmp_path, max_generations=1, resume=False)
    seed_summary = asyncio.run(EvolutionLoop(cfg).seed_population())
    assert seed_summary["total_generations"] == 0
    assert seed_summary["active_population_size"] == 2

    summary = asyncio.run(EvolutionLoop(cfg).run())

    assert summary["total_generations"] == 1
    assert summary["active_population_size"] == 2

    pop_root = Path(str(cfg.paths.population_root))
    state = read_json(pop_root / "population_state.json")
    assert state["current_generation"] == 1
    assert len(state["active_organisms"]) == 2

    active_dirs = [Path(entry["organism_dir"]) for entry in state["active_organisms"]]
    for org_dir in active_dirs:
        assert org_dir.exists()
        assert (org_dir / "implementation.py").exists()
        assert (org_dir / "genetic_code.md").exists()
        assert (org_dir / "lineage.json").exists()
        assert (org_dir / "summary.json").exists()

        summary_payload = read_json(org_dir / "summary.json")
        phase_results = summary_payload["phase_results"]
        assert list(phase_results["simple"]["selected_experiments"]) == ["simple_a"]
        assert list(phase_results["hard"]["selected_experiments"]) == ["hard_b"]
        assert phase_results["simple"]["allocation_snapshot"]["inclusion_prob"]["simple_a"] == 1.0
        assert phase_results["hard"]["allocation_snapshot"]["inclusion_prob"]["hard_b"] == 1.0
        assert summary_payload["simple_score"] is not None
        assert summary_payload["hard_score"] is not None


def test_seed_population_writes_generation_zero_state(tmp_path: Path) -> None:
    cfg = _canonical_cfg(tmp_path, max_generations=1, resume=False)

    summary = asyncio.run(EvolutionLoop(cfg).seed_population())

    assert summary["total_generations"] == 0
    assert summary["active_population_size"] == 2

    pop_root = Path(str(cfg.paths.population_root))
    state = read_json(pop_root / "population_state.json")
    assert state["current_generation"] == 0
    assert state["inflight_seed"] is None
    assert len(state["active_organisms"]) == 2


def test_seed_population_raises_when_all_simple_evals_fail(tmp_path: Path) -> None:
    cfg = _canonical_cfg(tmp_path, max_generations=1, resume=False)
    cfg.experiments.simple_a._target_ = "tests.fixtures.fake_runner.AlwaysFailExperimentEvaluator"

    with pytest.raises(RuntimeError, match="0 active organisms after simple evaluation"):
        asyncio.run(EvolutionLoop(cfg).seed_population())

    pop_root = Path(str(cfg.paths.population_root))
    state = read_json(pop_root / "population_state.json")
    assert state["active_organisms"] == []
    assert state["inflight_seed"] is not None
    assert state["inflight_seed"]["completed"] is True
    assert state["inflight_seed"]["failed"] is True


def test_seed_population_resumes_inflight_seed_plan(tmp_path: Path) -> None:
    cfg = _canonical_cfg(tmp_path, max_generations=1, resume=True)
    loop = EvolutionLoop(cfg)
    loop.generation = 0
    planned = loop._plan_seed_population()
    loop._save_state(finalized_generation=0, inflight_seed=loop._serialize_inflight_seed(planned), inflight_generation=None)

    summary = asyncio.run(EvolutionLoop(cfg).seed_population())

    assert summary["total_generations"] == 0
    pop_root = Path(str(cfg.paths.population_root))
    state = read_json(pop_root / "population_state.json")
    assert state["inflight_seed"] is None
    assert len(state["active_organisms"]) == 2


def test_run_requires_seeded_population(tmp_path: Path) -> None:
    cfg = _canonical_cfg(tmp_path, max_generations=1, resume=False)

    try:
        asyncio.run(EvolutionLoop(cfg).run())
    except FileNotFoundError as exc:
        assert "seed_population.sh" in str(exc)
    else:
        raise AssertionError("EvolutionLoop.run() should require a pre-seeded population state")


def test_run_rejects_inflight_seed_state(tmp_path: Path) -> None:
    cfg = _canonical_cfg(tmp_path, max_generations=1, resume=True)
    loop = EvolutionLoop(cfg)
    loop.generation = 0
    planned = loop._plan_seed_population()
    loop._save_state(finalized_generation=0, inflight_seed=loop._serialize_inflight_seed(planned), inflight_generation=None)

    try:
        asyncio.run(EvolutionLoop(cfg).run())
    except RuntimeError as exc:
        assert "seed_population.sh" in str(exc)
    else:
        raise AssertionError("EvolutionLoop.run() should reject unfinished inflight_seed state")


def test_canonical_multigeneration_resume_keeps_population_state_and_island_boundaries(tmp_path: Path) -> None:
    cfg_first = _canonical_cfg(tmp_path, max_generations=1, resume=False)
    asyncio.run(EvolutionLoop(cfg_first).seed_population())
    asyncio.run(EvolutionLoop(cfg_first).run())

    cfg_resume = _canonical_cfg(tmp_path, max_generations=2, resume=True)
    summary = asyncio.run(EvolutionLoop(cfg_resume).run())

    assert summary["total_generations"] == 2

    pop_root = Path(str(cfg_resume.paths.population_root))
    state = read_json(pop_root / "population_state.json")
    assert state["current_generation"] == 2
    assert {entry["island_id"] for entry in state["active_organisms"]} == {
        "gradient_methods",
        "second_order",
    }
    for entry in state["active_organisms"]:
        assert "/island_" in entry["organism_dir"]
