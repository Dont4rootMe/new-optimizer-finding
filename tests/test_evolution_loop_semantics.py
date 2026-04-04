"""Canonical evolution-loop semantics tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.evolve.evolution_loop import EvolutionLoop
from src.evolve.template_parser import render_template
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
                "seed_organisms_per_island": 3,
                "max_organisms_per_island": 3,
            },
            "reproduction": {
                "offspring_per_generation": 2,
                "operator_selection_strategy": "deterministic",
                "operator_weights": {
                    "within_island_crossover": 1.0,
                    "inter_island_crossover": 1.0,
                    "mutation": 1.0,
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
        render_template(
            {
                "IMPORTS": "import math",
                "INIT_BODY": "        self.model = model\n        self.max_steps = max_steps",
                "STEP_BODY": "        del weights, grads, activations, step_fn",
                "ZERO_GRAD_BODY": "        del set_to_none",
            },
            optimizer_name="Dummy",
            class_name="Dummy",
        ),
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


def test_inter_island_sampling_is_unified_and_distinct(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    loop = EvolutionLoop(cfg)
    primary_counts = {"gradient_methods": 0, "second_order": 0}
    draws = 5000
    for _ in range(draws):
        primary_island, secondary_island = loop._sample_island_ids(
            route="inter_island_crossover",
            candidate_island_ids=["gradient_methods", "second_order"],
            count=2,
            distinct=True,
        )
        assert primary_island != secondary_island
        primary_counts[primary_island] += 1

    observed = primary_counts["gradient_methods"] / draws
    assert abs(observed - 0.5) < 0.03


def test_deterministic_operator_selection_allocates_exact_counts(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.evolver.reproduction.operator_weights.within_island_crossover = 2.0
    cfg.evolver.reproduction.operator_weights.inter_island_crossover = 0.0
    cfg.evolver.reproduction.operator_weights.mutation = 6.0
    cfg.evolver.reproduction.offspring_per_generation = 8
    loop = EvolutionLoop(cfg)
    parent_a = _make_organism(tmp_path, "a", "gradient_methods")
    parent_b = _make_organism(tmp_path, "b", "gradient_methods")
    active = {"gradient_methods": [parent_a, parent_b], "second_order": []}
    assert loop._planned_reproduction_routes(active) == [
        "within_island_crossover",
        "within_island_crossover",
        "mutation",
        "mutation",
        "mutation",
        "mutation",
        "mutation",
        "mutation",
    ]


def test_random_operator_selection_samples_categorical_weights(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.evolver.reproduction.operator_selection_strategy = "random"
    cfg.evolver.reproduction.operator_weights.within_island_crossover = 2.0
    cfg.evolver.reproduction.operator_weights.inter_island_crossover = 0.0
    cfg.evolver.reproduction.operator_weights.mutation = 6.0
    loop = EvolutionLoop(cfg)
    draws = 5000
    counts = {"within_island_crossover": 0, "mutation": 0}
    available_routes = ["within_island_crossover", "mutation"]

    for _ in range(draws):
        counts[loop._sample_reproduction_route(available_routes)] += 1

    observed = counts["mutation"] / draws
    assert abs(observed - 0.75) < 0.03


def test_mutation_weight_can_force_mutation_only(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    cfg.evolver.reproduction.operator_weights.within_island_crossover = 0.0
    cfg.evolver.reproduction.operator_weights.inter_island_crossover = 0.0
    cfg.evolver.reproduction.operator_weights.mutation = 5.0
    cfg.evolver.reproduction.offspring_per_generation = 3
    loop = EvolutionLoop(cfg)
    parent_a = _make_organism(tmp_path, "a", "gradient_methods")
    parent_b = _make_organism(tmp_path, "b", "gradient_methods")
    active = {"gradient_methods": [parent_a, parent_b], "second_order": []}
    calls = {"mutation": 0, "within": 0, "inter": 0}

    def fake_mutation(**kwargs):
        del kwargs
        calls["mutation"] += 1
        return _make_organism(tmp_path, f"mut{calls['mutation']}", "gradient_methods")

    def fake_within(**kwargs):
        del kwargs
        calls["within"] += 1
        return _make_organism(tmp_path, f"within{calls['within']}", "gradient_methods")

    def fake_inter(**kwargs):
        del kwargs
        calls["inter"] += 1
        return _make_organism(tmp_path, f"inter{calls['inter']}", "gradient_methods")

    monkeypatch.setattr(loop, "_create_mutation_offspring", fake_mutation)
    monkeypatch.setattr(loop, "_create_within_island_crossover_offspring", fake_within)
    monkeypatch.setattr(loop, "_create_inter_island_crossover_offspring", fake_inter)

    offspring = loop._produce_offspring(active)

    assert len(offspring) == 3
    assert calls["mutation"] == 3
    assert calls["within"] == 0
    assert calls["inter"] == 0


def test_inter_island_weight_can_force_inter_island_crossover_only(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    cfg.evolver.reproduction.operator_weights.within_island_crossover = 0.0
    cfg.evolver.reproduction.operator_weights.inter_island_crossover = 5.0
    cfg.evolver.reproduction.operator_weights.mutation = 0.0
    cfg.evolver.reproduction.offspring_per_generation = 2
    loop = EvolutionLoop(cfg)
    parent_a = _make_organism(tmp_path, "a", "gradient_methods")
    parent_b = _make_organism(tmp_path, "b", "second_order")
    active = {"gradient_methods": [parent_a], "second_order": [parent_b]}
    calls = {"mutation": 0, "within": 0, "inter": 0}

    def fake_mutation(**kwargs):
        del kwargs
        calls["mutation"] += 1
        return _make_organism(tmp_path, f"mut{calls['mutation']}", "gradient_methods")

    def fake_within(**kwargs):
        del kwargs
        calls["within"] += 1
        return _make_organism(tmp_path, f"within{calls['within']}", "gradient_methods")

    def fake_inter(**kwargs):
        del kwargs
        calls["inter"] += 1
        return _make_organism(tmp_path, f"inter{calls['inter']}", "gradient_methods")

    monkeypatch.setattr(loop, "_create_mutation_offspring", fake_mutation)
    monkeypatch.setattr(loop, "_create_within_island_crossover_offspring", fake_within)
    monkeypatch.setattr(loop, "_create_inter_island_crossover_offspring", fake_inter)

    offspring = loop._produce_offspring(active)

    assert len(offspring) == 2
    assert calls["mutation"] == 0
    assert calls["within"] == 0
    assert calls["inter"] == 2


def test_mutation_parent_sampling_uses_softmax(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    cfg.evolver.reproduction.operator_weights.within_island_crossover = 0.0
    cfg.evolver.reproduction.operator_weights.inter_island_crossover = 0.0
    cfg.evolver.reproduction.operator_weights.mutation = 1.0
    cfg.evolver.reproduction.offspring_per_generation = 200
    cfg.evolver.operators.mutation.parent_selection_softmax_temperature = 0.1
    loop = EvolutionLoop(cfg)
    high = _make_organism(tmp_path, "high", "gradient_methods", score=10.0)
    low = _make_organism(tmp_path, "low", "gradient_methods", score=0.1)
    active = {"gradient_methods": [high, low], "second_order": []}
    parent_counts = {"high": 0, "low": 0}

    def fake_mutation_produce(_self, *, parent, organism_id, generation, org_dir, generator):
        del _self, organism_id, generation, org_dir, generator
        parent_counts[parent.organism_id] += 1
        return _make_organism(
            tmp_path,
            f"child_{parent_counts[parent.organism_id]}_{parent.organism_id}",
            "gradient_methods",
            score=parent.simple_reward or 1.0,
        )

    monkeypatch.setattr("src.organisms.mutation.MutationOperator.produce", fake_mutation_produce)

    offspring = loop._produce_offspring(active)

    assert len(offspring) == 200
    assert parent_counts["high"] > parent_counts["low"]


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


def test_invalid_top_h_above_top_k_is_rejected(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.evolver.phases.simple.top_k_per_island = 1
    cfg.evolver.phases.great_filter.top_h_per_island = 2

    with pytest.raises(ValueError, match="top_h_per_island"):
        EvolutionLoop(cfg)


def test_invalid_top_k_above_max_is_rejected(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.evolver.islands.max_organisms_per_island = 1
    cfg.evolver.phases.simple.top_k_per_island = 2

    with pytest.raises(ValueError, match="max_organisms_per_island"):
        EvolutionLoop(cfg)


def test_canonical_loop_requires_islands_dir(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    del cfg.evolver.islands["dir"]

    with pytest.raises(ValueError, match="evolver.islands.dir"):
        EvolutionLoop(cfg)


def test_canonical_loop_rejects_outdated_operator_fallbacks(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    del cfg.evolver.operators.mutation["gene_removal_probability"]
    cfg.evolution = {"mutation_q": 0.1, "crossover_p": 0.3}

    with pytest.raises(ValueError, match="evolver.operators.mutation.gene_removal_probability"):
        EvolutionLoop(cfg)


def test_invalid_operator_selection_strategy_is_rejected(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.evolver.reproduction.operator_selection_strategy = "unknown"

    with pytest.raises(ValueError, match="operator_selection_strategy"):
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
