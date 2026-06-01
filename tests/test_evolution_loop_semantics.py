"""Evolution-loop semantics tests."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.evolve.evolution_loop import EvolutionLoop
from src.evolve.orchestrator import DEFAULT_EVAL_ENTRYPOINT_MODULE, EvolverOrchestrator
from src.evolve.types import OrganismMeta, PlannedOrganismCreation
from src.organisms.organism import save_organism_artifacts


def _cfg(tmp_path: Path, **overrides) -> object:
    seed_program_path = tmp_path / "_baseline.py"
    seed_program_path.write_text(
        "import numpy as np\n\ndef run_optimizer(*args, **kwargs):\n    return {}\n",
        encoding="utf-8",
    )

    payload = {
        "seed": 123,
        "precision": "fp32",
        "deterministic": False,
        "num_workers": 0,
        "paths": {
            "population_root": str(tmp_path / "populations"),
            "stats_root": str(tmp_path / "stats"),
            "data_root": str(tmp_path / "data"),
            "runs_root": str(tmp_path / "runs"),
        },
        "resources": {"evaluation": {"gpu_ranks": [0], "cpu_parallel_jobs": 1}},
        "experiments": {
            "simple_a": {
                "enabled": True,
                "need_cuda": False,
                "normalization": {"eps": 1.0e-8},
                "compute": {"device": "cpu"},
            },
            "hard_b": {
                "enabled": True,
                "need_cuda": False,
                "normalization": {"eps": 1.0e-8},
                "compute": {"device": "cpu"},
            },
        },
        "evolver": {
            "resume": False,
            "max_generations": 1,
            "max_organism_creations": False,
            "_eval_entrypoint_module": "tests.fixtures.fake_eval",
            "creation": {
                "max_attempts_to_create_organism": 1,
                "max_attempts_to_repair_organism_after_error": 1,
                "max_attempts_to_regenerate_organism_after_novelty_rejection": 1,
                "max_parallel_organisms": 1,
            },
            "islands": {
                "mode": "from_seed",
                "seed_program_path": str(seed_program_path),
                "island_ids": ["gradient_methods", "second_order"],
                "seeds_per_island": 3,
                "max_organisms_per_island": 3,
            },
            "prompts": {
                "project_context": "conf/experiments/optimization_survey/prompts/shared/project_context.txt",
                "genome_schema": "conf/experiments/optimization_survey/prompts/shared/genome_schema.txt",
                "seed_system": "conf/experiments/optimization_survey/prompts/seed/system.txt",
                "seed_user": "conf/experiments/optimization_survey/prompts/seed/user.txt",
                "mutation_system": "conf/experiments/optimization_survey/prompts/mutation/system.txt",
                "mutation_user": "conf/experiments/optimization_survey/prompts/mutation/user.txt",
                "mutation_novelty_system": "conf/experiments/optimization_survey/prompts/novelty/mutation/system.txt",
                "mutation_novelty_user": "conf/experiments/optimization_survey/prompts/novelty/mutation/user.txt",
                "crossover_system": "conf/experiments/optimization_survey/prompts/crossover/system.txt",
                "crossover_user": "conf/experiments/optimization_survey/prompts/crossover/user.txt",
                "crossover_novelty_system": "conf/experiments/optimization_survey/prompts/novelty/crossover/system.txt",
                "crossover_novelty_user": "conf/experiments/optimization_survey/prompts/novelty/crossover/user.txt",
                "implementation_system": "conf/experiments/optimization_survey/prompts/implementation/system.txt",
                "implementation_user": "conf/experiments/optimization_survey/prompts/implementation/user.txt",
                "implementation_template": "conf/experiments/optimization_survey/prompts/shared/template.txt",
                "repair_system": "conf/experiments/optimization_survey/prompts/repair/system.txt",
                "repair_user": "conf/experiments/optimization_survey/prompts/repair/user.txt",
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
                "species_sampling": {
                    "strategy": "weighted_rule",
                    "weighted_rule_lambda": 1.0,
                    "mutation_softmax_temperature": 1.0,
                    "within_island_crossover_softmax_temperature": 1.0,
                    "inter_island_crossover_softmax_temperature": 1.0,
                },
            },
            "operators": {
                "mutation": {
                    "gene_removal_probability": 0.2,
                },
                "crossover": {
                    "primary_parent_gene_inheritance_probability": 0.7,
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
                    "timeout_sec_per_eval": 120,
                    "top_h_per_island": 2,
                    "experiments": ["hard_b"],
                    "allocation": {"enabled": False},
                },
            },
            "llm": {"route_weights": {"mock": 1.0}, "seed": 123},
        },
        "api_platforms": {"mock": {"_target_": "api_platforms.mock.platform.build_platform"}},
    }

    cfg = OmegaConf.create(payload)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def _implementation_code() -> str:
    return (
        "import torch.nn as nn\n\n"
        "OPTIMIZER_NAME = 'Dummy'\n\n"
        "class Dummy:\n"
        "    def __init__(self, model: nn.Module, max_steps: int) -> None:\n"
        "        self.model = model\n"
        "        self.max_steps = max_steps\n\n"
        "    def step(self, weights, grads, activations, step_fn) -> None:\n"
        "        del weights, grads, activations, step_fn\n\n"
        "    def zero_grad(self, set_to_none: bool = True) -> None:\n"
        "        del set_to_none\n\n"
        "def build_optimizer(model: nn.Module, max_steps: int):\n"
        "    return Dummy(model, max_steps)\n"
    )


def _make_organism(tmp_path: Path, org_id: str, island_id: str, score: float = 1.0) -> OrganismMeta:
    org_dir = tmp_path / island_id / f"org_{org_id}"
    org_dir.mkdir(parents=True, exist_ok=True)
    (org_dir / "implementation.py").write_text(_implementation_code(), encoding="utf-8")
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
        implementation_path=str(org_dir / "implementation.py"),
        lineage_path=str(org_dir / "lineage.json"),
        organism_dir=str(org_dir),
        ancestor_ids=[],
        experiment_report_index={},
        simple_score=score,
    )
    save_organism_artifacts(
        organism,
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
    return organism


def test_inter_island_sampling_is_unified_and_distinct(tmp_path: Path) -> None:
    loop = EvolutionLoop(_cfg(tmp_path))
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


def test_orchestrator_defaults_to_run_one_and_reads_queue_sizes(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        resources={"evaluation": {"gpu_ranks": [2, 4], "cpu_parallel_jobs": 3}},
        evolver={"_eval_entrypoint_module": None},
    )
    orchestrator = EvolverOrchestrator(cfg)
    try:
        assert orchestrator.eval_entrypoint_module == DEFAULT_EVAL_ENTRYPOINT_MODULE
        assert orchestrator.gpu_ranks == [2, 4]
        assert orchestrator.cpu_parallel_jobs == 3
    finally:
        orchestrator.close()


def test_orchestrator_rejects_grouped_route_gpu_overlap_with_evaluation_pool(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        resources={"evaluation": {"gpu_ranks": [4], "cpu_parallel_jobs": 3}},
        api_platforms={
            "ollama_qwen35_122b": {
                "_target_": "api_platforms.ollama_qwen35_122b.platform.build_platform",
                "base_url": "http://127.0.0.1:12434/api",
                "gpu_ranks": [[0, 1, 2], [3, 4, 5]],
                "max_concurrency": 3,
            }
        },
        evolver={"llm": {"route_weights": {"ollama_qwen35_122b": 1.0}, "seed": 123}},
    )

    with pytest.raises(ValueError, match=r"overlapping ranks: \[4\]"):
        EvolverOrchestrator(cfg)


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

    offspring = asyncio.run(loop._produce_offspring(active))

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

    offspring = asyncio.run(loop._produce_offspring(active))

    assert len(offspring) == 2
    assert calls["mutation"] == 0
    assert calls["within"] == 0
    assert calls["inter"] == 2


def test_weighted_rule_parent_counts_accumulate_within_generation(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    cfg.evolver.reproduction.operator_weights.within_island_crossover = 0.0
    cfg.evolver.reproduction.operator_weights.inter_island_crossover = 0.0
    cfg.evolver.reproduction.operator_weights.mutation = 1.0
    cfg.evolver.reproduction.offspring_per_generation = 3
    cfg.evolver.reproduction.species_sampling.strategy = "weighted_rule"
    loop = EvolutionLoop(cfg)
    parent_a = _make_organism(tmp_path, "a", "gradient_methods")
    parent_b = _make_organism(tmp_path, "b", "gradient_methods")
    active = {"gradient_methods": [parent_a, parent_b], "second_order": []}
    snapshots: list[dict[str, int]] = []

    def fake_weighted_rule_select_organisms(
        population,
        *,
        parent_offspring_counts,
        score_field,
        weighted_rule_lambda,
        k,
        rng,
    ):
        del population, score_field, weighted_rule_lambda, k, rng
        snapshots.append(dict(parent_offspring_counts))
        return [parent_a]

    monkeypatch.setattr(
        "src.evolve.evolution_loop.weighted_rule_select_organisms",
        fake_weighted_rule_select_organisms,
    )

    planned = loop._plan_offspring_generation(active)

    assert len(planned) == 3
    assert [plan.mother_id for plan in planned] == ["a", "a", "a"]
    assert snapshots == [{}, {"a": 1}, {"a": 2}]


def test_seed_planning_caps_total_organism_creations(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        evolver={
            "max_organism_creations": 3,
            "islands": {
                "seeds_per_island": 2,
            },
        },
    )
    loop = EvolutionLoop(cfg)

    planned = loop._plan_seed_population()

    assert len(planned) == 3
    assert loop._organism_creation_attempt_count() == 3


def test_evolution_loop_requires_at_least_one_stop_limit(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        evolver={
            "max_generations": False,
            "max_organism_creations": False,
        },
    )

    with pytest.raises(ValueError, match="max_generations or evolver.max_organism_creations"):
        EvolutionLoop(cfg)


def test_offspring_planning_uses_remaining_total_organism_creation_budget(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        evolver={
            "max_organism_creations": 5,
            "islands": {
                "seeds_per_island": 2,
            },
            "reproduction": {
                "offspring_per_generation": 5,
                "operator_weights": {
                    "within_island_crossover": 0.0,
                    "inter_island_crossover": 0.0,
                    "mutation": 1.0,
                },
            },
        },
    )
    loop = EvolutionLoop(cfg)
    seed_plans = loop._plan_seed_population()
    parent = _make_organism(tmp_path, "parent", "gradient_methods")
    active = {"gradient_methods": [parent], "second_order": []}
    loop.generation = 1

    planned = loop._plan_offspring_generation(active)

    assert len(seed_plans) == 4
    assert len(planned) == 1
    assert loop._organism_creation_attempt_count() == 5


def test_softmax_species_sampling_uses_route_specific_temperatures(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    cfg.evolver.reproduction.offspring_per_generation = 3
    cfg.evolver.reproduction.species_sampling.strategy = "softmax"
    cfg.evolver.reproduction.species_sampling.mutation_softmax_temperature = 0.11
    cfg.evolver.reproduction.species_sampling.within_island_crossover_softmax_temperature = 0.22
    cfg.evolver.reproduction.species_sampling.inter_island_crossover_softmax_temperature = 0.33
    loop = EvolutionLoop(cfg)
    gradient_a = _make_organism(tmp_path, "a", "gradient_methods")
    gradient_b = _make_organism(tmp_path, "b", "gradient_methods")
    second_a = _make_organism(tmp_path, "c", "second_order")
    second_b = _make_organism(tmp_path, "d", "second_order")
    active = {
        "gradient_methods": [gradient_a, gradient_b],
        "second_order": [second_a, second_b],
    }
    calls: list[tuple[str, float, int]] = []

    def fake_softmax_select_organisms(population, *, score_field, temperature, k, rng):
        del score_field, rng
        calls.append(("single", temperature, k))
        return list(population[:k])

    def fake_softmax_select_distinct_organisms(population, *, score_field, temperature, k, rng):
        del score_field, rng
        calls.append(("distinct", temperature, k))
        return list(population[:k])

    monkeypatch.setattr(
        "src.evolve.evolution_loop.softmax_select_organisms",
        fake_softmax_select_organisms,
    )
    monkeypatch.setattr(
        "src.evolve.evolution_loop.softmax_select_distinct_organisms",
        fake_softmax_select_distinct_organisms,
    )

    planned = loop._plan_offspring_generation(active)

    assert len(planned) == 3
    assert calls == [
        ("distinct", 0.22, 2),
        ("single", 0.33, 1),
        ("single", 0.33, 1),
        ("single", 0.11, 1),
    ]


def _make_seed_plan(organism_id: str, island_id: str) -> object:
    """Minimal `PlannedOrganismCreation` stub for `_assign_batch_routes` tests.

    The method only reads `organism_id`, so any object exposing that attribute
    works and we avoid pulling in the full `PlannedOrganismCreation` ctor here.
    """

    class _Plan:
        pass

    plan = _Plan()
    plan.organism_id = organism_id
    plan.island_id = island_id
    return plan


def test_assign_batch_routes_spreads_two_plans_across_two_equal_routes(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    loop = EvolutionLoop(cfg)
    loop.generator.route_weights = {"route_a": 1.0, "route_b": 1.0}
    plans = [
        _make_seed_plan("org-aaaa", "gradient_methods"),
        _make_seed_plan("org-bbbb", "second_order"),
    ]
    assignment = loop._assign_batch_routes(plans)
    assert set(assignment.values()) == {"route_a", "route_b"}
    # Deterministic: same inputs -> same mapping
    assert assignment == loop._assign_batch_routes(plans)


def test_assign_batch_routes_returns_empty_for_single_positive_route(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    loop = EvolutionLoop(cfg)
    loop.generator.route_weights = {"only_route": 1.0, "zero_route": 0.0}
    plans = [_make_seed_plan(f"org-{i}", "gradient_methods") for i in range(4)]
    assert loop._assign_batch_routes(plans) == {}


def test_assign_batch_routes_respects_weight_proportions(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    loop = EvolutionLoop(cfg)
    loop.generator.route_weights = {"heavy": 3.0, "light": 1.0}
    plans = [_make_seed_plan(f"org-{i:02d}", "gradient_methods") for i in range(8)]
    assignment = loop._assign_batch_routes(plans)
    counts = {"heavy": 0, "light": 0}
    for route in assignment.values():
        counts[route] += 1
    # Hamilton apportionment for 8 plans with 3:1 weights => 6:2
    assert counts == {"heavy": 6, "light": 2}


def test_max_parallel_organisms_bounds_seed_parallelism(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(
        tmp_path,
        evolver={
            "creation": {
                "max_parallel_organisms": 2,
            },
            "islands": {
                "seeds_per_island": 4,
            },
        },
    )
    loop = EvolutionLoop(cfg)
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_materialize_seed_from_file(*, island, organism_id, generation, organism_dir, source_path, pipeline_state_callback=None):
        del island, organism_id, generation, organism_dir, source_path, pipeline_state_callback
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.1)
        with lock:
            active -= 1
        return _make_organism(tmp_path, f"seed_{time.time_ns()}", "gradient_methods")

    monkeypatch.setattr(loop.generator, "materialize_seed_from_file", fake_materialize_seed_from_file)

    asyncio.run(loop._seed_initial_population())

    assert max_active <= 2


def _write_token_organism(
    population_root: Path,
    *,
    generation: int,
    island_id: str,
    organism_id: str,
    tokens_by_route: dict[str, int],
) -> None:
    """Write a minimal organism.json carrying ``token_usage`` under the glob layout.

    ``_tokens_spent_per_route`` only reads the ``token_usage`` field, so a
    minimal payload is enough to exercise the per-model token-budget stop.
    """

    org_dir = population_root / f"gen_{generation:04d}" / f"island_{island_id}" / f"org_{organism_id}"
    org_dir.mkdir(parents=True, exist_ok=True)
    usage = {
        route: {
            "prompt_tokens": 0,
            "completion_tokens": int(total),
            "total_tokens": int(total),
            "calls": 1,
        }
        for route, total in tokens_by_route.items()
    }
    (org_dir / "organism.json").write_text(
        json.dumps({"organism_id": organism_id, "token_usage": usage}),
        encoding="utf-8",
    )


def test_max_tokens_per_model_parses_false_and_int(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        evolver={
            "max_tokens_per_model": {
                "ollama_gemma4_31b": False,
                "ollama_qwen35_122b": 500000,
                "disabled_minus_one": -1,
            }
        },
    )
    loop = EvolutionLoop(cfg)
    # false / -1 drop out; only the real integer cap survives.
    assert loop.max_tokens_per_model == {"ollama_qwen35_122b": 500000}


def test_max_tokens_per_model_rejects_true(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, evolver={"max_tokens_per_model": {"ollama_gemma4_31b": True}})
    with pytest.raises(ValueError, match="max_tokens_per_model"):
        EvolutionLoop(cfg)


def test_token_budget_absent_is_inert(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, evolver={"max_tokens_per_model": {"ollama_gemma4_31b": False}})
    loop = EvolutionLoop(cfg)
    _write_token_organism(
        loop.population_root,
        generation=1,
        island_id="gradient_methods",
        organism_id="a",
        tokens_by_route={"ollama_gemma4_31b": 10**9},
    )
    # All-false config => no cap installed, never stops on tokens.
    assert loop.max_tokens_per_model == {}
    assert loop._token_budget_stop_reason() is None
    assert loop._stop_reason_before_generation(loop.generation + 1) is None


def test_token_budget_stops_when_route_exceeds(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        evolver={
            "max_generations": 100,
            "max_tokens_per_model": {
                "ollama_gemma4_31b": 1000,
                "ollama_qwen35_122b": False,
            },
        },
    )
    loop = EvolutionLoop(cfg)
    # Nothing spent yet -> no stop.
    assert loop._token_budget_stop_reason() is None

    _write_token_organism(
        loop.population_root,
        generation=1,
        island_id="gradient_methods",
        organism_id="a",
        tokens_by_route={"ollama_gemma4_31b": 600, "ollama_qwen35_122b": 9_999_999},
    )
    _write_token_organism(
        loop.population_root,
        generation=1,
        island_id="gradient_methods",
        organism_id="b",
        tokens_by_route={"ollama_gemma4_31b": 500},
    )

    # gemma total = 1100 >= 1000 -> stop; qwen is uncapped (false) so ignored.
    reason = loop._token_budget_stop_reason()
    assert reason is not None
    assert "ollama_gemma4_31b" in reason
    assert "ollama_qwen35_122b" not in reason
    # The generation-boundary stop check surfaces the same reason.
    assert loop._stop_reason_before_generation(loop.generation + 1) == reason


def test_tokens_spent_per_route_sums_across_organisms(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, evolver={"max_tokens_per_model": {"ollama_gemma4_31b": 10**9}})
    loop = EvolutionLoop(cfg)
    _write_token_organism(
        loop.population_root,
        generation=1,
        island_id="gradient_methods",
        organism_id="a",
        tokens_by_route={"ollama_gemma4_31b": 100, "ollama_qwen35_122b": 200},
    )
    _write_token_organism(
        loop.population_root,
        generation=2,
        island_id="second_order",
        organism_id="c",
        tokens_by_route={"ollama_gemma4_31b": 50},
    )
    spent = loop._tokens_spent_per_route()
    assert spent == {"ollama_gemma4_31b": 150, "ollama_qwen35_122b": 200}
