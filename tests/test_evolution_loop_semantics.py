"""Canonical evolution-loop semantics tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.evolve.evolution_loop import EvolutionLoop
from src.evolve.types import OrganismMeta
from src.organisms.organism import save_organism_artifacts


def _cfg(tmp_path: Path, **overrides) -> object:
    islands_dir = tmp_path / "islands"
    islands_dir.mkdir(parents=True, exist_ok=True)
    (islands_dir / "gradient_methods.txt").write_text("First-order ideas", encoding="utf-8")
    (islands_dir / "second_order.txt").write_text("Curvature-aware ideas", encoding="utf-8")

    payload = {
        "seed": 123,
        "precision": "fp32",
        "paths": {
            "population_root": str(tmp_path / "populations"),
            "stats_root": str(tmp_path / "stats"),
        },
        "resources": {"num_gpus": 1, "gpu_ids": [0]},
        "experiments": {
            "simple_a": {"enabled": True, "primary_metric": {"direction": "min"}, "normalization": {"eps": 1.0e-8}},
            "hard_b": {"enabled": True, "primary_metric": {"direction": "min"}, "normalization": {"eps": 1.0e-8}},
        },
        "evolver": {
            "generation": 0,
            "resume": False,
            "enabled": True,
            "max_generations": 1,
            "eval_entrypoint_module": "tests.fixtures.fake_eval",
            "islands": {
                "dir": str(islands_dir),
                "organisms_per_island": 3,
                "inter_island_crossover_rate": 0.25,
            },
            "operators": {
                "mutation": {"probability": 0.5, "gene_delete_probability": 0.2},
                "crossover": {
                    "inherit_gene_probability_from_mother": 0.7,
                    "softmax_temperature": 1.0,
                },
            },
            "phases": {
                "simple": {
                    "eval_mode": "smoke",
                    "timeout_sec_per_eval": 60,
                    "top_k_per_island": 2,
                    "experiments": ["simple_a"],
                    "allocation": {"enabled": False},
                },
                "great_filter": {
                    "enabled": True,
                    "interval_generations": 1,
                    "eval_mode": "full",
                    "timeout_sec_per_eval": 120,
                    "top_h_per_island": 2,
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

    cfg = OmegaConf.create(payload)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def _make_organism(tmp_path: Path, org_id: str, island_id: str, score: float = 1.0) -> OrganismMeta:
    org_dir = tmp_path / island_id / f"org_{org_id}"
    org_dir.mkdir(parents=True, exist_ok=True)
    (org_dir / "optimizer.py").write_text(
        "import torch.nn as nn\n\n"
        "class Dummy:\n"
        "    def __init__(self, model: nn.Module, max_steps: int):\n"
        "        self.model = model\n"
        "    def step(self, weights, grads, activations, step_fn):\n"
        "        del weights, grads, activations, step_fn\n"
        "    def zero_grad(self, set_to_none=True):\n"
        "        del set_to_none\n\n"
        "def build_optimizer(model: nn.Module, max_steps: int):\n"
        "    return Dummy(model, max_steps)\n",
        encoding="utf-8",
    )
    organism = OrganismMeta(
        organism_id=org_id,
        island_id=island_id,
        generation_created=0,
        current_generation_active=0,
        timestamp="2026-01-01T00:00:00Z",
        mother_id=None,
        father_id=None,
        operator="seed",
        genetic_code_path=str(org_dir / "genetic_code.md"),
        optimizer_path=str(org_dir / "optimizer.py"),
        lineage_path=str(org_dir / "lineage.json"),
        organism_dir=str(org_dir),
        simple_reward=score,
        selection_reward=score,
        genetic_code={
            "core_genes": ["adaptive momentum", "warmup schedule", "gradient clipping"],
            "interaction_notes": "notes",
            "compute_notes": "compute",
        },
        lineage=[
            {
                "generation": 0,
                "operator": "seed",
                "mother_id": None,
                "father_id": None,
                "change_description": "Initial creation",
                "gene_diff_summary": "adaptive momentum",
                "selected_simple_experiments": [],
                "selected_hard_experiments": [],
                "simple_score": score,
                "hard_score": None,
                "cross_island": False,
                "father_island_id": None,
            }
        ],
    )
    save_organism_artifacts(organism)
    return organism


def test_inter_island_crossover_probability_matches_configured_rate(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    loop = EvolutionLoop(cfg)
    mother = _make_organism(tmp_path, "mother", "gradient_methods")
    local = _make_organism(tmp_path, "local", "gradient_methods")
    foreign = _make_organism(tmp_path, "foreign", "second_order")
    active = {"gradient_methods": [mother, local], "second_order": [foreign]}

    foreign_count = 0
    draws = 5000
    for _ in range(draws):
        pool = loop._select_father_pool(mother=mother, active_by_island=active)
        assert pool
        if all(candidate.island_id != mother.island_id for candidate in pool):
            foreign_count += 1

    observed = foreign_count / draws
    assert abs(observed - 0.25) < 0.03


def test_high_mutation_probability_produces_mutation_only(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    cfg.evolver.operators.mutation.probability = 1.0
    loop = EvolutionLoop(cfg)
    island = loop.islands_by_id["gradient_methods"]
    parent_a = _make_organism(tmp_path, "a", "gradient_methods")
    parent_b = _make_organism(tmp_path, "b", "gradient_methods")
    active = {"gradient_methods": [parent_a, parent_b], "second_order": []}
    calls = {"mutation": 0, "crossover": 0}

    def fake_mutation(*args, **kwargs):
        del args, kwargs
        calls["mutation"] += 1
        return _make_organism(tmp_path, f"mut{calls['mutation']}", "gradient_methods")

    def fake_crossover(*args, **kwargs):
        del args, kwargs
        calls["crossover"] += 1
        return _make_organism(tmp_path, f"cross{calls['crossover']}", "gradient_methods")

    monkeypatch.setattr("src.organisms.mutation.MutationOperator.produce", fake_mutation)
    monkeypatch.setattr("src.organisms.crossbreeding.CrossbreedingOperator.produce", fake_crossover)

    offspring = asyncio.run(loop._reproduce_for_island(island, active))

    assert offspring
    assert calls["mutation"] > 0
    assert calls["crossover"] == 0


def test_low_mutation_probability_allows_crossover(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    cfg.evolver.operators.mutation.probability = 0.0
    loop = EvolutionLoop(cfg)
    island = loop.islands_by_id["gradient_methods"]
    parent_a = _make_organism(tmp_path, "a", "gradient_methods")
    parent_b = _make_organism(tmp_path, "b", "gradient_methods")
    active = {"gradient_methods": [parent_a, parent_b], "second_order": []}
    calls = {"mutation": 0, "crossover": 0}

    def fake_mutation(*args, **kwargs):
        del args, kwargs
        calls["mutation"] += 1
        return _make_organism(tmp_path, f"mut{calls['mutation']}", "gradient_methods")

    def fake_crossover(*args, **kwargs):
        del args, kwargs
        calls["crossover"] += 1
        return _make_organism(tmp_path, f"cross{calls['crossover']}", "gradient_methods")

    monkeypatch.setattr("src.organisms.mutation.MutationOperator.produce", fake_mutation)
    monkeypatch.setattr("src.organisms.crossbreeding.CrossbreedingOperator.produce", fake_crossover)

    offspring = asyncio.run(loop._reproduce_for_island(island, active))

    assert offspring
    assert calls["crossover"] > 0


def test_phase_requests_default_to_smoke_and_full_and_preserve_simple_reward(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    loop = EvolutionLoop(cfg)
    organism = _make_organism(tmp_path, "org1", "gradient_methods")
    requests_seen = []

    async def fake_evaluate(_self, requests):
        requests_seen.extend(requests)
        summaries = []
        for request in requests:
            reward = 1.1 if request.phase == "simple" else 2.2
            summaries.append(
                type(
                    "Summary",
                    (),
                    {
                        "organism_id": request.organism_id,
                        "phase": request.phase,
                        "aggregate_score": reward,
                        "per_experiment": {request.experiments[0]: {"status": "ok", "exp_score": reward}},
                        "selected_experiments": list(request.experiments),
                        "allocation_snapshot": {
                            "method": "uniform",
                            "enabled": False,
                            "history_window": 0,
                            "sample_size": len(request.experiments),
                            "weights": {name: 1.0 for name in request.experiments},
                            "inclusion_prob": {name: 1.0 for name in request.experiments},
                            "stats": {},
                            "selected_experiments": list(request.experiments),
                        },
                        "status": "ok",
                        "created_at": request.created_at,
                        "eval_finished_at": request.created_at,
                        "error_msg": None,
                    },
                )()
            )
        return summaries

    monkeypatch.setattr("src.evolve.orchestrator.EvolverOrchestrator.evaluate_organisms", fake_evaluate)

    asyncio.run(loop._evaluate_phase([organism], phase="simple"))
    simple_reward = organism.simple_reward
    asyncio.run(loop._evaluate_phase([organism], phase="hard"))

    assert [request.phase for request in requests_seen] == ["simple", "hard"]
    assert requests_seen[0].eval_mode == "smoke"
    assert requests_seen[1].eval_mode == "full"
    assert requests_seen[0].experiments == ["simple_a"]
    assert requests_seen[1].experiments == ["hard_b"]
    assert organism.simple_reward == simple_reward
    assert organism.hard_reward == 2.2
    assert organism.selection_reward == 2.2


def test_invalid_top_h_above_top_k_is_rejected(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.evolver.phases.simple.top_k_per_island = 1
    cfg.evolver.phases.great_filter.top_h_per_island = 2

    with pytest.raises(ValueError, match="top_h_per_island"):
        EvolutionLoop(cfg)


def test_canonical_loop_requires_islands_dir(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    del cfg.evolver.islands["dir"]

    with pytest.raises(ValueError, match="evolver.islands.dir"):
        EvolutionLoop(cfg)


def test_canonical_loop_rejects_outdated_operator_fallbacks(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    del cfg.evolver.operators.mutation["probability"]
    cfg.evolution = {"mutation_rate": 0.9, "mutation_q": 0.1, "crossover_p": 0.3}

    with pytest.raises(ValueError, match="evolver.operators.mutation.probability"):
        EvolutionLoop(cfg)


def test_canonical_loop_rejects_outdated_phase_fallbacks(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    del cfg.evolver.phases["simple"]
    del cfg.evolver.phases["great_filter"]
    cfg.evaluation = {
        "simple_experiments": ["simple_a"],
        "hard_experiments": ["hard_b"],
        "simple_allocation": {"enabled": False},
        "hard_allocation": {"enabled": False},
    }

    with pytest.raises(ValueError, match="evolver.phases.simple"):
        EvolutionLoop(cfg)


def test_canonical_loop_rejects_top_level_timeout_fallback(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    del cfg.evolver.phases.simple["timeout_sec_per_eval"]
    cfg.evolver["timeout_sec_per_eval"] = 999
    loop = EvolutionLoop(cfg)

    with pytest.raises(ValueError, match="evolver.phases.simple.timeout_sec_per_eval"):
        loop._phase_timeout_sec("simple")


def test_canonical_run_rejects_outdated_max_generations_fallback(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    del cfg.evolver["max_generations"]
    cfg["evolution"] = {"max_generations": 2}
    loop = EvolutionLoop(cfg)

    with pytest.raises(ValueError, match="evolver.max_generations"):
        asyncio.run(loop.run())
