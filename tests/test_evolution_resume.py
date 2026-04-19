"""Tests for population-state-driven evolution resume."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.evolve.evolution_loop import EvolutionLoop
from src.evolve.storage import (
    generation_dir,
    organism_dir,
    read_population_state,
    write_json,
    write_population_state,
)
from src.evolve.types import OrganismMeta
from src.organisms.organism import save_organism_artifacts


def _make_cfg(tmp_path: Path) -> object:
    islands_dir = tmp_path / "islands"
    islands_dir.mkdir(parents=True, exist_ok=True)
    (islands_dir / "gradient_methods.txt").write_text("First-order methods", encoding="utf-8")
    return OmegaConf.create(
        {
            "seed": 123,
            "precision": "fp32",
            "deterministic": False,
            "num_workers": 0,
            "paths": {
                "population_root": str(tmp_path / "populations"),
                "stats_root": str(tmp_path / "stats"),
                "data_root": str(tmp_path / "data"),
                "runs_root": str(tmp_path / "runs"),
                "api_platform_runtime_root": str(tmp_path / ".api_platform_runtime"),
            },
            "resources": {"evaluation": {"gpu_ranks": [0], "cpu_parallel_jobs": 1}},
            "experiments": {},
            "api_platforms": {"mock": {"_target_": "api_platforms.mock.platform.build_platform"}},
            "evolver": {
                "resume": True,
                "max_generations": 3,
                "creation": {
                    "max_attempts_to_create_organism": 1,
                    "max_attempts_to_repair_organism_after_error": 1,
                    "max_attempts_to_regenerate_organism_after_novelty_rejection": 1,
                    "max_parallel_organisms": 1,
                },
                "islands": {
                    "dir": str(islands_dir),
                    "seed_organisms_per_island": 2,
                    "max_organisms_per_island": 2,
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
                    "offspring_per_generation": 1,
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
                        "experiments": [],
                        "allocation": {},
                    },
                    "great_filter": {
                        "enabled": False,
                        "interval_generations": 5,
                        "eval_mode": "full",
                        "timeout_sec_per_eval": 120,
                        "top_h_per_island": 2,
                        "experiments": [],
                        "allocation": {},
                    },
                },
                "llm": {"route_weights": {"mock": 1.0}, "seed": 123},
            },
        }
    )


def _write_organism(pop_root: Path, generation: int, organism_id: str) -> OrganismMeta:
    gen_dir = generation_dir(pop_root, generation)
    org_dir = organism_dir(gen_dir, organism_id, island_id="gradient_methods")
    (org_dir / "implementation.py").write_text(
        "import torch.nn as nn\n\n"
        "class Dummy:\n"
        "    def __init__(self, model: nn.Module, max_steps: int):\n"
        "        self.model = model\n"
        "        self.max_steps = max_steps\n"
        "    def step(self, weights, grads, activations, step_fn):\n"
        "        del weights, grads, activations, step_fn\n"
        "    def zero_grad(self, set_to_none=True):\n"
        "        del set_to_none\n\n"
        "def build_optimizer(model: nn.Module, max_steps: int):\n"
        "    return Dummy(model, max_steps)\n",
        encoding="utf-8",
    )

    organism = OrganismMeta(
        organism_id=organism_id,
        island_id="gradient_methods",
        generation_created=generation,
        current_generation_active=3,
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
        simple_score=0.8,
    )
    save_organism_artifacts(
        organism,
        genetic_code={
            "core_genes": [
                f"adaptive rule {organism_id}",
                "warmup schedule",
                "gradient clipping",
            ],
            "interaction_notes": "Coordinate warmup and adaptive updates.",
            "compute_notes": "Keep controller cheap during fake resume tests.",
        },
        lineage=[],
    )
    return organism


def test_restore_population_from_state_reads_older_generation_survivor(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    pop_root = Path(str(cfg.paths.population_root))
    pop_root.mkdir(parents=True, exist_ok=True)

    older = _write_organism(pop_root, generation=0, organism_id="older")
    newer = _write_organism(pop_root, generation=3, organism_id="newer")
    write_population_state(pop_root, generation=3, organisms=[older, newer])

    loop = EvolutionLoop(cfg)
    restored = loop._restore_population_from_state(generation=3)

    assert [organism.organism_id for organism in restored] == ["older", "newer"]
    assert restored[0].generation_created == 0
    assert restored[1].generation_created == 3


def test_population_state_duplicate_entry_is_rejected(tmp_path: Path) -> None:
    pop_root = tmp_path / "populations"
    pop_root.mkdir(parents=True, exist_ok=True)
    write_json(
        pop_root / "population_state.json",
        {
            "current_generation": 1,
            "active_organisms": [
                {
                    "organism_id": "dup",
                    "island_id": "gradient_methods",
                    "organism_dir": str(pop_root / "gen_0001" / "island_gradient_methods" / "org_dup"),
                    "generation_created": 0,
                    "current_generation_active": 1,
                },
                {
                    "organism_id": "dup",
                    "island_id": "gradient_methods",
                    "organism_dir": str(pop_root / "gen_0001" / "island_gradient_methods" / "org_dup2"),
                    "generation_created": 1,
                    "current_generation_active": 1,
                },
            ],
        },
    )

    with pytest.raises(ValueError, match="duplicate organism_id"):
        read_population_state(pop_root)


def test_population_state_malformed_entry_is_rejected(tmp_path: Path) -> None:
    pop_root = tmp_path / "populations"
    pop_root.mkdir(parents=True, exist_ok=True)
    write_json(
        pop_root / "population_state.json",
        {
            "current_generation": 1,
            "active_organisms": [
                {
                    "organism_id": "broken",
                    "organism_dir": 123,
                    "generation_created": "bad",
                    "current_generation_active": 1,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="island_id|organism_dir|generation_created"):
        read_population_state(pop_root)


def test_canonical_resume_requires_population_state(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    loop = EvolutionLoop(cfg)

    with pytest.raises(FileNotFoundError, match="population_state.json"):
        loop._restore_population_from_state(generation=3)


def test_population_state_missing_organism_dir_fails_resume(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    pop_root = Path(str(cfg.paths.population_root))
    pop_root.mkdir(parents=True, exist_ok=True)
    write_population_state(
        pop_root,
        generation=3,
        organisms=[
            {
                "organism_id": "ghost",
                "island_id": "gradient_methods",
                "organism_dir": str(pop_root / "gen_0003" / "island_gradient_methods" / "org_ghost"),
                "generation_created": 1,
                "current_generation_active": 3,
            }
        ],
    )

    loop = EvolutionLoop(cfg)
    with pytest.raises(FileNotFoundError, match="missing organism dir"):
        loop._restore_population_from_state(generation=3)


def test_population_state_missing_genetic_code_fails_resume(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    pop_root = Path(str(cfg.paths.population_root))
    pop_root.mkdir(parents=True, exist_ok=True)

    organism = _write_organism(pop_root, generation=3, organism_id="broken_genes")
    Path(organism.genetic_code_path).unlink()
    write_population_state(pop_root, generation=3, organisms=[organism])

    loop = EvolutionLoop(cfg)
    with pytest.raises(FileNotFoundError, match="genetic code"):
        loop._restore_population_from_state(generation=3)


def test_population_state_missing_lineage_fails_resume(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    pop_root = Path(str(cfg.paths.population_root))
    pop_root.mkdir(parents=True, exist_ok=True)

    organism = _write_organism(pop_root, generation=3, organism_id="broken_lineage")
    Path(organism.lineage_path).unlink()
    write_population_state(pop_root, generation=3, organisms=[organism])

    loop = EvolutionLoop(cfg)
    with pytest.raises(FileNotFoundError, match="lineage"):
        loop._restore_population_from_state(generation=3)
