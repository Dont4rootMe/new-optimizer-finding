"""Evolution-loop semantics tests."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

from omegaconf import OmegaConf

from src.evolve.evolution_loop import EvolutionLoop
from src.evolve.orchestrator import DEFAULT_EVAL_ENTRYPOINT_MODULE, EvolverOrchestrator
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
            "_eval_entrypoint_module": "tests.fixtures.fake_eval",
            "creation": {
                "max_attempts_to_create_organism": 1,
                "max_attempts_to_repair_organism_after_error": 1,
                "max_parallel_organisms": 1,
            },
            "islands": {
                "dir": str(islands_dir),
                "seed_organisms_per_island": 3,
                "max_organisms_per_island": 3,
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
                "seed_organisms_per_island": 4,
            },
        },
    )
    loop = EvolutionLoop(cfg)
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_generate_seed_organism(*, island, organism_id, generation, organism_dir):
        del island, organism_id, generation, organism_dir
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.1)
        with lock:
            active -= 1
        return _make_organism(tmp_path, f"seed_{time.time_ns()}", "gradient_methods")

    monkeypatch.setattr(loop.generator, "generate_seed_organism", fake_generate_seed_organism)

    asyncio.run(loop._seed_initial_population())

    assert max_active <= 2
