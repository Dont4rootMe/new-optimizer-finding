"""Multi-generation evolution loop with genetic operators and Great Filter."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.evolve.generator import OptimizerGenerator
from src.evolve.operators import SeedOperator
from src.evolve.selection import elite_select, select_parents_for_reproduction
from src.organisms.crossbreeding import CrossbreedingOperator
from src.organisms.mutation import MutationOperator as ProbMutationOperator
from src.evolve.storage import (
    generation_dir,
    load_population,
    organism_dir,
    organism_meta_path,
    read_json,
    utc_now_iso,
    write_json,
)
from src.evolve.types import OrganismMeta

LOGGER = logging.getLogger(__name__)

_STATE_FILE = "evolution_state.json"


def _dict_to_organism(d: dict[str, Any]) -> OrganismMeta:
    """Reconstruct OrganismMeta from a dict (loaded from JSON)."""
    return OrganismMeta(
        organism_id=d["organism_id"],
        generation=d["generation"],
        timestamp=d["timestamp"],
        parent_ids=d.get("parent_ids", []),
        operator=d.get("operator", "seed"),
        idea_dna=d.get("idea_dna", []),
        evolution_log=d.get("evolution_log", []),
        model_name=d.get("model_name", ""),
        prompt_hash=d.get("prompt_hash", ""),
        seed=d.get("seed", 0),
        organism_dir=d.get("organism_dir", ""),
        optimizer_path=d.get("optimizer_path", ""),
        score=d.get("score"),
        simple_score=d.get("simple_score"),
        hard_score=d.get("hard_score"),
        status=d.get("status", "pending"),
    )


class EvolutionLoop:
    """Runs multi-generation evolutionary optimization."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.evolver_cfg = cfg.evolver
        self.evo_cfg = cfg.evolver.evolution
        self.eval_cfg = cfg.evolver.evaluation

        self.population_root = Path(str(cfg.paths.population_root)).expanduser().resolve()
        self.population: list[OrganismMeta] = []
        self.generation = 0

        self.generator = OptimizerGenerator(cfg)

        self.simple_experiments: list[str] = list(self.eval_cfg.simple_experiments)
        self.hard_experiments: list[str] = list(self.eval_cfg.hard_experiments)

        self.rng = random.Random(int(cfg.seed))

    def _state_path(self) -> Path:
        return self.population_root / _STATE_FILE

    def _save_state(self) -> None:
        """Persist evolution state for resume."""
        state = {
            "current_generation": self.generation,
            "population_organism_ids": [org.organism_id for org in self.population],
            "best_score": max(
                (org.score for org in self.population if org.score is not None),
                default=None,
            ),
            "best_organism_id": max(
                self.population,
                key=lambda o: o.score if o.score is not None else -float("inf"),
            ).organism_id if self.population else None,
            "timestamp": utc_now_iso(),
        }
        write_json(self._state_path(), state)

    def _load_state(self) -> dict[str, Any] | None:
        """Load evolution state if exists."""
        path = self._state_path()
        if path.exists():
            return read_json(path)
        return None

    async def _seed_population(self, count: int) -> list[OrganismMeta]:
        """Create initial population from scratch."""
        organisms: list[OrganismMeta] = []
        seed_op = SeedOperator()
        gen_dir = generation_dir(self.population_root, self.generation)

        for i in range(count):
            org_id = uuid.uuid4().hex
            org_dir = organism_dir(gen_dir, org_id)
            try:
                org = self.generator.generate_organism(
                    operator=seed_op,
                    parents=[],
                    organism_id=org_id,
                    generation=self.generation,
                    organism_dir=org_dir,
                )
                organisms.append(org)
                LOGGER.info("Seeded organism %s (%d/%d)", org_id[:8], i + 1, count)
            except Exception as exc:
                LOGGER.error("Failed to seed organism %d: %s", i, exc)

        return organisms

    async def _reproduce(
        self, plan: list[tuple[str, list[OrganismMeta]]]
    ) -> list[OrganismMeta]:
        """Execute reproduction plan using probabilistic genetic operators."""
        offspring: list[OrganismMeta] = []
        gen_dir = generation_dir(self.population_root, self.generation)

        crossover_p = float(self.evo_cfg.get("crossover_p", 0.7))
        mutation_q = float(self.evo_cfg.get("mutation_q", 0.2))

        mutation_op = ProbMutationOperator(q=mutation_q, seed=self.rng.randint(0, 2**31))
        crossover_op = CrossbreedingOperator(p=crossover_p, seed=self.rng.randint(0, 2**31))

        for op_name, parents in plan:
            org_id = uuid.uuid4().hex
            org_dir = organism_dir(gen_dir, org_id)

            try:
                if op_name == "mutation":
                    org = mutation_op.produce(
                        parent=parents[0],
                        organism_id=org_id,
                        generation=self.generation,
                        org_dir=org_dir,
                        generator=self.generator,
                        prompts_dir=self.generator.prompts_dir,
                    )
                elif op_name == "crossover":
                    # First parent is dominant (higher score)
                    dominant, non_dominant = parents[0], parents[1]
                    if (non_dominant.score or 0) > (dominant.score or 0):
                        dominant, non_dominant = non_dominant, dominant
                    org = crossover_op.produce(
                        dominant=dominant,
                        non_dominant=non_dominant,
                        organism_id=org_id,
                        generation=self.generation,
                        org_dir=org_dir,
                        generator=self.generator,
                        prompts_dir=self.generator.prompts_dir,
                    )
                else:
                    raise ValueError(f"Unknown operator: {op_name}")

                offspring.append(org)
                LOGGER.info(
                    "%s -> organism %s (parents: %s)",
                    op_name,
                    org_id[:8],
                    [p.organism_id[:8] for p in parents],
                )
            except Exception as exc:
                LOGGER.error("Failed %s: %s", op_name, exc)

        return offspring

    async def _evaluate_organisms(
        self,
        organisms: list[OrganismMeta],
        experiments: list[str],
        score_key: str = "simple_score",
    ) -> None:
        """Evaluate organisms using the orchestrator.

        This delegates to EvolverOrchestrator for GPU-based evaluation.
        For now, organisms get placeholder scores; the full integration
        with the orchestrator's eval pipeline happens in Phase 6.2.
        """
        from src.evolve.orchestrator import EvolverOrchestrator

        # Build a temporary config with the right eval_experiments
        cfg_override = OmegaConf.create({
            "evolver": {
                "eval_experiments": experiments,
                "num_candidates": 0,  # no new generation, just eval
            }
        })
        eval_cfg = OmegaConf.merge(self.cfg, cfg_override)

        orchestrator = EvolverOrchestrator(eval_cfg)

        # Submit organisms as candidates for evaluation
        for org in organisms:
            cand_dir = Path(org.organism_dir)
            allocation_snapshot = orchestrator._build_candidate_allocation(org.organism_id)
            selected = list(allocation_snapshot.get("selected_experiments", experiments))

            orchestrator._register_candidate(
                candidate_id=org.organism_id,
                candidate_dir_path=cand_dir,
                created_at=org.timestamp,
                selected_experiments=selected,
                allocation_snapshot=allocation_snapshot,
            )

        result = await orchestrator.run()

        # Map scores back to organisms
        for summary in result.get("candidate_summaries", []):
            cand_id = summary.get("candidate_id")
            score = summary.get("aggregate_score")
            for org in organisms:
                if org.organism_id == cand_id:
                    setattr(org, score_key, score)
                    if org.score is None or (score is not None and score > org.score):
                        org.score = score
                    org.status = "evaluated"
                    # Update on disk
                    write_json(
                        organism_meta_path(Path(org.organism_dir)),
                        org.to_dict(),
                    )
                    break

    async def run(self) -> dict[str, Any]:
        """Execute the full multi-generation evolution."""
        max_generations = int(self.evo_cfg.max_generations)
        population_size = int(self.evo_cfg.population_size)
        elite_count = int(self.evo_cfg.elite_count)
        mutation_rate = float(self.evo_cfg.mutation_rate)
        tournament_size = int(self.evo_cfg.tournament_size)
        great_filter_interval = int(self.evo_cfg.great_filter_interval)

        # Check for resume
        state = self._load_state()
        if state is not None:
            self.generation = state["current_generation"]
            LOGGER.info("Resuming from generation %d", self.generation)
            pop_dicts = load_population(self.population_root, self.generation)
            self.population = [_dict_to_organism(d) for d in pop_dicts]
        else:
            # Generation 0: seed
            LOGGER.info("Starting evolution: seeding %d organisms", population_size)
            self.population = await self._seed_population(population_size)
            await self._evaluate_organisms(
                self.population, self.simple_experiments, score_key="simple_score"
            )
            self._save_state()

        for gen in range(self.generation + 1, max_generations + 1):
            self.generation = gen
            is_great_filter = (gen % great_filter_interval == 0)

            LOGGER.info(
                "=== Generation %d%s ===",
                gen,
                " [GREAT FILTER]" if is_great_filter else "",
            )

            # 1. Elitist selection
            survivors = elite_select(
                self.population, elite_count, score_key="simple_score"
            )
            LOGGER.info(
                "Selected %d elites (best score: %s)",
                len(survivors),
                survivors[0].score if survivors else "N/A",
            )

            # 2. Reproduction
            num_offspring = population_size - len(survivors)
            if num_offspring > 0 and survivors:
                reproduction_plan = select_parents_for_reproduction(
                    survivors,
                    num_offspring=num_offspring,
                    mutation_rate=mutation_rate,
                    tournament_size=tournament_size,
                    rng=self.rng,
                )
                offspring = await self._reproduce(reproduction_plan)
            else:
                offspring = []

            # 3. New population
            self.population = list(survivors) + offspring

            # Fill gaps if some offspring failed
            if len(self.population) < population_size:
                gap = population_size - len(self.population)
                LOGGER.info("Filling %d gaps with seed organisms", gap)
                seeds = await self._seed_population(gap)
                self.population.extend(seeds)

            # 4. Evaluate on simple tasks
            unevaluated = [o for o in self.population if o.simple_score is None]
            if unevaluated:
                await self._evaluate_organisms(
                    unevaluated, self.simple_experiments, score_key="simple_score"
                )

            # 5. Great Filter
            if is_great_filter:
                LOGGER.info("Running Great Filter on %d organisms", len(self.population))
                all_experiments = self.simple_experiments + self.hard_experiments
                await self._evaluate_organisms(
                    self.population, all_experiments, score_key="hard_score"
                )

                # Strict selection by hard_score
                self.population = elite_select(
                    self.population, elite_count, score_key="hard_score"
                )
                LOGGER.info(
                    "Great Filter: %d survivors",
                    len(self.population),
                )

                # Replenish with seeds
                if len(self.population) < population_size:
                    gap = population_size - len(self.population)
                    new_seeds = await self._seed_population(gap)
                    self.population.extend(new_seeds)

            # 6. Save state
            self._save_state()

        # Final summary
        best = max(
            self.population,
            key=lambda o: o.score if o.score is not None else -float("inf"),
        ) if self.population else None

        summary = {
            "total_generations": self.generation,
            "final_population_size": len(self.population),
            "best_organism_id": best.organism_id if best else None,
            "best_score": best.score if best else None,
            "best_optimizer_path": best.optimizer_path if best else None,
        }
        LOGGER.info("Evolution complete: %s", summary)
        return summary
