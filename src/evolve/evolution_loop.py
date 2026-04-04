"""Multi-generation organism-first evolution loop with island-aware selection."""

from __future__ import annotations

import logging
import random
import uuid
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.evolve.generator import OptimizerGenerator
from src.evolve.islands import load_islands
from src.evolve.orchestrator import EvolverOrchestrator
from src.evolve.selection import (
    select_top_h_per_island,
    select_top_k_per_island,
    softmax_select_distinct_organisms,
    softmax_select_organisms,
)
from src.evolve.storage import (
    generation_dir,
    organism_dir,
    read_json,
    read_organism_meta,
    read_organism_summary,
    read_population_manifest,
    utc_now_iso,
    write_json,
    write_organism_meta,
    write_organism_summary,
    write_population_manifest,
)
from src.evolve.types import Island, OrganismEvaluationRequest, OrganismMeta
from src.organisms.crossbreeding import CrossbreedingOperator
from src.organisms.mutation import MutationOperator
from src.organisms.organism import save_organism_artifacts, update_latest_lineage_entry

LOGGER = logging.getLogger(__name__)

_STATE_FILE = "evolution_state.json"
_MISSING = object()
_ROUTE_WITHIN_ISLAND_CROSSOVER = "within_island_crossover"
_ROUTE_INTER_ISLAND_CROSSOVER = "inter_island_crossover"
_ROUTE_MUTATION = "mutation"
_ROUTE_ORDER = (
    _ROUTE_WITHIN_ISLAND_CROSSOVER,
    _ROUTE_INTER_ISLAND_CROSSOVER,
    _ROUTE_MUTATION,
)


class EvolutionLoop:
    """Runs multi-generation island-aware optimizer evolution."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.evolver_cfg = cfg.evolver
        self.population_root = Path(str(cfg.paths.population_root)).expanduser().resolve()
        self.population: list[OrganismMeta] = []
        self.generation = 0

        self.generator = OptimizerGenerator(cfg)
        self.rng = random.Random(int(cfg.seed))

        self.islands = self._load_islands()
        self.islands_by_id = {island.island_id: island for island in self.islands}

        self.seed_organisms_per_island = self._seed_organisms_per_island()
        self.max_organisms_per_island = self._max_organisms_per_island()
        self.offspring_per_generation = self._offspring_per_generation()
        self.operator_selection_strategy = self._operator_selection_strategy()
        self.reproduction_operator_weights = self._reproduction_operator_weights()
        self.reproduction_island_sampling = self._reproduction_island_sampling()
        self.mutation_gene_removal_probability = self._mutation_gene_removal_probability()
        self.mutation_parent_selection_softmax_temperature = self._mutation_parent_selection_softmax_temperature()
        self.crossover_primary_parent_gene_inheritance_probability = (
            self._crossover_primary_parent_gene_inheritance_probability()
        )
        self.crossover_parent_selection_softmax_temperature = (
            self._crossover_parent_selection_softmax_temperature()
        )
        self._validate_phase_selection_bounds()

    def _require_cfg_value(self, path: str) -> Any:
        value = OmegaConf.select(self.cfg, path)
        if value is None:
            raise ValueError(f"Canonical evolve config must define {path}")
        return value

    def _load_islands(self) -> list[Island]:
        islands_dir = str(self._require_cfg_value("evolver.islands.dir")).strip()
        if not islands_dir:
            raise ValueError("Canonical evolve config must define evolver.islands.dir")
        return load_islands(islands_dir)

    def _seed_organisms_per_island(self) -> int:
        return int(self._require_cfg_value("evolver.islands.seed_organisms_per_island"))

    def _max_organisms_per_island(self) -> int:
        return int(self._require_cfg_value("evolver.islands.max_organisms_per_island"))

    def _offspring_per_generation(self) -> int:
        return int(self._require_cfg_value("evolver.reproduction.offspring_per_generation"))

    def _operator_selection_strategy(self) -> str:
        return str(self._require_cfg_value("evolver.reproduction.operator_selection_strategy"))

    def _reproduction_operator_weights(self) -> dict[str, float]:
        return {
            _ROUTE_WITHIN_ISLAND_CROSSOVER: float(
                self._require_cfg_value("evolver.reproduction.operator_weights.within_island_crossover")
            ),
            _ROUTE_INTER_ISLAND_CROSSOVER: float(
                self._require_cfg_value("evolver.reproduction.operator_weights.inter_island_crossover")
            ),
            _ROUTE_MUTATION: float(self._require_cfg_value("evolver.reproduction.operator_weights.mutation")),
        }

    def _reproduction_island_sampling(self) -> dict[str, str]:
        return {
            _ROUTE_WITHIN_ISLAND_CROSSOVER: str(
                self._require_cfg_value("evolver.reproduction.island_sampling.within_island_crossover")
            ),
            _ROUTE_INTER_ISLAND_CROSSOVER: str(
                self._require_cfg_value("evolver.reproduction.island_sampling.inter_island_crossover")
            ),
            _ROUTE_MUTATION: str(self._require_cfg_value("evolver.reproduction.island_sampling.mutation")),
        }

    def _mutation_gene_removal_probability(self) -> float:
        return float(self._require_cfg_value("evolver.operators.mutation.gene_removal_probability"))

    def _mutation_parent_selection_softmax_temperature(self) -> float:
        return float(self._require_cfg_value("evolver.operators.mutation.parent_selection_softmax_temperature"))

    def _crossover_primary_parent_gene_inheritance_probability(self) -> float:
        return float(
            self._require_cfg_value("evolver.operators.crossover.primary_parent_gene_inheritance_probability")
        )

    def _crossover_parent_selection_softmax_temperature(self) -> float:
        return float(self._require_cfg_value("evolver.operators.crossover.parent_selection_softmax_temperature"))

    def _phase_cfg(self, phase: str) -> dict[str, Any]:
        payload = OmegaConf.select(self.cfg, f"evolver.phases.{phase}")
        if payload is None:
            raise ValueError(f"Canonical evolve config must define evolver.phases.{phase}")
        resolved = OmegaConf.to_container(payload, resolve=True)
        if not isinstance(resolved, dict):
            raise ValueError(f"Canonical evolve config block evolver.phases.{phase} must be a mapping")
        return resolved

    def _phase_value(self, phase: str, key: str, default: Any = _MISSING) -> Any:
        cfg = self._phase_cfg("great_filter" if phase == "hard" else phase)
        if key in cfg and cfg[key] is not None:
            return cfg[key]
        if default is not _MISSING:
            return default
        phase_name = "great_filter" if phase == "hard" else phase
        raise ValueError(f"Canonical evolve config must define evolver.phases.{phase_name}.{key}")

    def _phase_experiments(self, phase: str) -> list[str]:
        return [str(name) for name in self._phase_value(phase, "experiments")]

    def _phase_allocation_cfg(self, phase: str) -> dict[str, Any]:
        allocation = self._phase_value(phase, "allocation")
        return allocation if isinstance(allocation, dict) else {}

    def _phase_eval_mode(self, phase: str) -> str:
        return str(self._phase_value(phase, "eval_mode"))

    def _phase_timeout_sec(self, phase: str) -> int:
        return int(self._phase_value(phase, "timeout_sec_per_eval"))

    def _simple_top_k(self) -> int:
        return int(self._phase_value("simple", "top_k_per_island"))

    def _great_filter_top_h(self) -> int:
        return int(self._phase_value("hard", "top_h_per_island"))

    def _great_filter_enabled(self) -> bool:
        return bool(self._phase_value("hard", "enabled"))

    def _great_filter_interval(self) -> int:
        return max(1, int(self._phase_value("hard", "interval_generations")))

    def _validate_phase_selection_bounds(self) -> None:
        top_k = self._simple_top_k()
        top_h = self._great_filter_top_h()
        if top_k > self.max_organisms_per_island:
            raise ValueError(
                "Invalid evolve config: evolver.phases.simple.top_k_per_island "
                "must be <= evolver.islands.max_organisms_per_island"
            )
        if top_h > top_k:
            raise ValueError(
                "Invalid evolve config: evolver.phases.great_filter.top_h_per_island "
                "must be <= evolver.phases.simple.top_k_per_island"
            )
        if self.max_organisms_per_island <= 0:
            raise ValueError("Invalid evolve config: evolver.islands.max_organisms_per_island must be > 0")
        if self.seed_organisms_per_island < 0:
            raise ValueError("Invalid evolve config: evolver.islands.seed_organisms_per_island must be >= 0")
        if self.offspring_per_generation < 0:
            raise ValueError("Invalid evolve config: evolver.reproduction.offspring_per_generation must be >= 0")
        for route, weight in self.reproduction_operator_weights.items():
            if weight < 0:
                raise ValueError(
                    "Invalid evolve config: evolver.reproduction.operator_weights."
                    f"{route} must be >= 0"
                )
        if self.offspring_per_generation > 0 and sum(self.reproduction_operator_weights.values()) <= 0:
            raise ValueError(
                "Invalid evolve config: evolver.reproduction.operator_weights must sum to > 0 "
                "when offspring_per_generation is positive"
            )
        if self.operator_selection_strategy not in {"deterministic", "random"}:
            raise ValueError(
                "Invalid evolve config: evolver.reproduction.operator_selection_strategy "
                "must be 'deterministic' or 'random'"
            )

    def _should_run_great_filter(self, generation: int) -> bool:
        return self._great_filter_enabled() and generation > 0 and generation % self._great_filter_interval() == 0

    def _state_path(self) -> Path:
        return self.population_root / _STATE_FILE

    def _best_active_organism(self) -> OrganismMeta | None:
        if not self.population:
            return None
        return max(
            self.population,
            key=lambda organism: organism.simple_reward if organism.simple_reward is not None else -float("inf"),
        )

    def _save_state(self) -> None:
        for organism in self.population:
            organism.current_generation_active = self.generation
            write_organism_meta(organism)

        best = self._best_active_organism()
        state = {
            "current_generation": self.generation,
            "active_population_size": len(self.population),
            "best_organism_id": best.organism_id if best is not None else None,
            "best_simple_reward": best.simple_reward if best is not None else None,
            "timestamp": utc_now_iso(),
        }
        write_json(self._state_path(), state)
        write_population_manifest(self.population_root, self.generation, self.population)

    def _load_state(self) -> dict[str, Any] | None:
        if self._state_path().exists():
            payload = read_json(self._state_path())
            if isinstance(payload, dict):
                return payload
        return None

    def _restore_population_from_manifest(self, generation: int) -> list[OrganismMeta]:
        manifest = read_population_manifest(self.population_root)
        if manifest is None:
            raise FileNotFoundError("Population manifest is required for canonical resume.")

        manifest_generation = int(manifest.get("generation", generation))
        if manifest_generation != generation:
            LOGGER.warning(
                "Evolution state points to generation %d but manifest points to generation %d. "
                "Using manifest generation.",
                generation,
                manifest_generation,
            )
            self.generation = manifest_generation

        organisms: list[OrganismMeta] = []
        for entry in manifest.get("active_organisms", []):
            organism_dir_path = Path(str(entry["organism_dir"]))
            if not organism_dir_path.is_absolute():
                organism_dir_path = (self.population_root / organism_dir_path).resolve()
            if not organism_dir_path.exists():
                raise FileNotFoundError(
                    f"Active population manifest points to missing organism dir: {organism_dir_path}"
                )
            organism = read_organism_meta(organism_dir_path)
            organism.current_generation_active = int(entry.get("current_generation_active", self.generation))
            organisms.append(organism)
        return organisms

    async def _seed_island(self, island: Island, count: int) -> list[OrganismMeta]:
        organisms: list[OrganismMeta] = []
        if count <= 0:
            return organisms

        gen_dir = generation_dir(self.population_root, self.generation)
        for _ in range(count):
            organism_id = uuid.uuid4().hex
            org_dir = organism_dir(gen_dir, organism_id, island_id=island.island_id)
            organism = self.generator.generate_seed_organism(
                island=island,
                organism_id=organism_id,
                generation=self.generation,
                organism_dir=org_dir,
            )
            organisms.append(organism)
        return organisms

    async def _seed_initial_population(self) -> list[OrganismMeta]:
        newborns: list[OrganismMeta] = []
        for island in self.islands:
            newborns.extend(await self._seed_island(island, self.seed_organisms_per_island))
        return newborns

    def _group_by_island(self, organisms: list[OrganismMeta]) -> dict[str, list[OrganismMeta]]:
        grouped: dict[str, list[OrganismMeta]] = {island.island_id: [] for island in self.islands}
        for organism in organisms:
            grouped.setdefault(organism.island_id, []).append(organism)
        return grouped

    def _eligible_island_ids_for_route(
        self,
        route: str,
        active_by_island: dict[str, list[OrganismMeta]],
    ) -> list[str]:
        if route == _ROUTE_WITHIN_ISLAND_CROSSOVER:
            return [island_id for island_id, pool in active_by_island.items() if len(pool) >= 2]
        if route in {_ROUTE_INTER_ISLAND_CROSSOVER, _ROUTE_MUTATION}:
            return [island_id for island_id, pool in active_by_island.items() if pool]
        raise ValueError(f"Unsupported reproduction route '{route}'")

    def _available_reproduction_routes(
        self,
        active_by_island: dict[str, list[OrganismMeta]],
    ) -> list[str]:
        available: list[str] = []
        within_island_candidates = self._eligible_island_ids_for_route(_ROUTE_WITHIN_ISLAND_CROSSOVER, active_by_island)
        if within_island_candidates:
            available.append(_ROUTE_WITHIN_ISLAND_CROSSOVER)

        inter_island_candidates = self._eligible_island_ids_for_route(_ROUTE_INTER_ISLAND_CROSSOVER, active_by_island)
        if len(inter_island_candidates) >= 2:
            available.append(_ROUTE_INTER_ISLAND_CROSSOVER)

        mutation_candidates = self._eligible_island_ids_for_route(_ROUTE_MUTATION, active_by_island)
        if mutation_candidates:
            available.append(_ROUTE_MUTATION)

        return available

    def _sample_reproduction_route(
        self,
        available_routes: list[str],
    ) -> str:
        if not available_routes:
            raise ValueError(
                "Cannot produce offspring: no reproduction routes are available for the current active population."
            )

        weights = [self.reproduction_operator_weights[route] for route in available_routes]
        return self.rng.choices(population=available_routes, weights=weights, k=1)[0]

    def _planned_reproduction_routes(
        self,
        active_by_island: dict[str, list[OrganismMeta]],
    ) -> list[str]:
        available_routes = [
            route
            for route in self._available_reproduction_routes(active_by_island)
            if self.reproduction_operator_weights[route] > 0
        ]
        if not available_routes:
            raise ValueError(
                "Cannot produce offspring: no reproduction routes are available for the current active population."
            )

        if self.operator_selection_strategy == "random":
            return [self._sample_reproduction_route(available_routes) for _ in range(self.offspring_per_generation)]
        if self.operator_selection_strategy == "deterministic":
            return self._deterministic_reproduction_plan(available_routes)
        raise ValueError(
            f"Unsupported operator selection strategy '{self.operator_selection_strategy}'."
        )

    def _deterministic_reproduction_plan(self, available_routes: list[str]) -> list[str]:
        total_weight = sum(self.reproduction_operator_weights[route] for route in available_routes)
        if total_weight <= 0:
            raise ValueError("Deterministic reproduction planning requires positive total route weight.")

        exact_counts = {
            route: self.offspring_per_generation * (self.reproduction_operator_weights[route] / total_weight)
            for route in available_routes
        }
        base_counts = {route: int(exact_counts[route]) for route in available_routes}
        assigned = sum(base_counts.values())
        remaining = self.offspring_per_generation - assigned

        ranked_routes = sorted(
            available_routes,
            key=lambda route: (
                -(exact_counts[route] - base_counts[route]),
                _ROUTE_ORDER.index(route),
            ),
        )
        for route in ranked_routes[:remaining]:
            base_counts[route] += 1

        plan: list[str] = []
        for route in _ROUTE_ORDER:
            if route in base_counts:
                plan.extend([route] * base_counts[route])
        return plan

    def _sample_island_ids(
        self,
        *,
        route: str,
        candidate_island_ids: list[str],
        count: int,
        distinct: bool,
    ) -> list[str]:
        strategy = self.reproduction_island_sampling[route]
        if strategy == "unified":
            if not candidate_island_ids or count <= 0:
                return []
            if distinct:
                if len(candidate_island_ids) < count:
                    raise ValueError(
                        f"Route '{route}' requires {count} distinct islands, got {len(candidate_island_ids)}."
                    )
                return list(self.rng.sample(candidate_island_ids, k=count))
            return [self.rng.choice(candidate_island_ids) for _ in range(count)]
        raise ValueError(
            f"Unsupported island sampling strategy '{strategy}' for reproduction route '{route}'."
        )

    def _newborn_org_dir(
        self,
        *,
        gen_dir: Path,
        island_id: str,
    ) -> tuple[str, Path]:
        organism_id = uuid.uuid4().hex
        return organism_id, organism_dir(gen_dir, organism_id, island_id=island_id)

    def _create_mutation_offspring(
        self,
        *,
        active_by_island: dict[str, list[OrganismMeta]],
        mutation_operator: MutationOperator,
        gen_dir: Path,
    ) -> OrganismMeta:
        candidate_islands = self._eligible_island_ids_for_route(_ROUTE_MUTATION, active_by_island)
        island_id = self._sample_island_ids(
            route=_ROUTE_MUTATION,
            candidate_island_ids=candidate_islands,
            count=1,
            distinct=False,
        )[0]
        parent = softmax_select_organisms(
            active_by_island[island_id],
            score_field="simple_reward",
            temperature=self.mutation_parent_selection_softmax_temperature,
            k=1,
            rng=self.rng,
        )[0]
        organism_id, org_dir = self._newborn_org_dir(gen_dir=gen_dir, island_id=island_id)
        return mutation_operator.produce(
            parent=parent,
            organism_id=organism_id,
            generation=self.generation,
            org_dir=org_dir,
            generator=self.generator,
        )

    def _create_within_island_crossover_offspring(
        self,
        *,
        active_by_island: dict[str, list[OrganismMeta]],
        crossover_operator: CrossbreedingOperator,
        gen_dir: Path,
    ) -> OrganismMeta:
        candidate_islands = self._eligible_island_ids_for_route(_ROUTE_WITHIN_ISLAND_CROSSOVER, active_by_island)
        island_id = self._sample_island_ids(
            route=_ROUTE_WITHIN_ISLAND_CROSSOVER,
            candidate_island_ids=candidate_islands,
            count=1,
            distinct=False,
        )[0]
        primary_parent, secondary_parent = softmax_select_distinct_organisms(
            active_by_island[island_id],
            score_field="simple_reward",
            temperature=self.crossover_parent_selection_softmax_temperature,
            k=2,
            rng=self.rng,
        )
        if primary_parent.organism_id == secondary_parent.organism_id:
            raise ValueError("Within-island crossover sampled duplicate parents; expected distinct organisms.")
        organism_id, org_dir = self._newborn_org_dir(gen_dir=gen_dir, island_id=island_id)
        return crossover_operator.produce(
            mother=primary_parent,
            father=secondary_parent,
            organism_id=organism_id,
            generation=self.generation,
            org_dir=org_dir,
            generator=self.generator,
        )

    def _create_inter_island_crossover_offspring(
        self,
        *,
        active_by_island: dict[str, list[OrganismMeta]],
        crossover_operator: CrossbreedingOperator,
        gen_dir: Path,
    ) -> OrganismMeta:
        candidate_islands = self._eligible_island_ids_for_route(_ROUTE_INTER_ISLAND_CROSSOVER, active_by_island)
        primary_island_id, secondary_island_id = self._sample_island_ids(
            route=_ROUTE_INTER_ISLAND_CROSSOVER,
            candidate_island_ids=candidate_islands,
            count=2,
            distinct=True,
        )
        primary_parent = softmax_select_organisms(
            active_by_island[primary_island_id],
            score_field="simple_reward",
            temperature=self.crossover_parent_selection_softmax_temperature,
            k=1,
            rng=self.rng,
        )[0]
        secondary_parent = softmax_select_organisms(
            active_by_island[secondary_island_id],
            score_field="simple_reward",
            temperature=self.crossover_parent_selection_softmax_temperature,
            k=1,
            rng=self.rng,
        )[0]
        organism_id, org_dir = self._newborn_org_dir(gen_dir=gen_dir, island_id=primary_island_id)
        return crossover_operator.produce(
            mother=primary_parent,
            father=secondary_parent,
            organism_id=organism_id,
            generation=self.generation,
            org_dir=org_dir,
            generator=self.generator,
        )

    def _produce_offspring(
        self,
        active_by_island: dict[str, list[OrganismMeta]],
    ) -> list[OrganismMeta]:
        if self.offspring_per_generation <= 0:
            return []

        gen_dir = generation_dir(self.population_root, self.generation)
        mutation_operator = MutationOperator(
            q=self.mutation_gene_removal_probability,
            seed=self.rng.randint(0, 2**31 - 1),
        )
        crossover_operator = CrossbreedingOperator(
            p=self.crossover_primary_parent_gene_inheritance_probability,
            seed=self.rng.randint(0, 2**31 - 1),
        )
        offspring: list[OrganismMeta] = []
        for route in self._planned_reproduction_routes(active_by_island):
            if route == _ROUTE_MUTATION:
                child = self._create_mutation_offspring(
                    active_by_island=active_by_island,
                    mutation_operator=mutation_operator,
                    gen_dir=gen_dir,
                )
            elif route == _ROUTE_WITHIN_ISLAND_CROSSOVER:
                child = self._create_within_island_crossover_offspring(
                    active_by_island=active_by_island,
                    crossover_operator=crossover_operator,
                    gen_dir=gen_dir,
                )
            elif route == _ROUTE_INTER_ISLAND_CROSSOVER:
                child = self._create_inter_island_crossover_offspring(
                    active_by_island=active_by_island,
                    crossover_operator=crossover_operator,
                    gen_dir=gen_dir,
                )
            else:
                raise ValueError(f"Unsupported reproduction route '{route}'")
            offspring.append(child)

        return offspring

    async def _evaluate_phase(
        self,
        organisms: list[OrganismMeta],
        *,
        phase: str,
    ) -> None:
        if not organisms:
            return

        experiments = self._phase_experiments(phase)
        if not experiments:
            return

        allocation_cfg = self._phase_allocation_cfg(phase)
        orchestrator = EvolverOrchestrator(self.cfg)
        requests = [
            OrganismEvaluationRequest(
                organism_id=organism.organism_id,
                organism_dir=organism.organism_dir,
                phase=phase,
                experiments=experiments,
                allocation_cfg=allocation_cfg,
                eval_mode=self._phase_eval_mode(phase),
                timeout_sec=self._phase_timeout_sec(phase),
                created_at=organism.timestamp,
            )
            for organism in organisms
        ]
        summaries = await orchestrator.evaluate_organisms(requests)
        summary_by_id = {summary.organism_id: summary for summary in summaries}

        for organism in organisms:
            summary = summary_by_id.get(organism.organism_id)
            if summary is None:
                continue

            if phase == "simple":
                organism.simple_reward = summary.aggregate_score
            elif phase == "hard":
                organism.hard_reward = summary.aggregate_score
            else:
                raise ValueError(f"Unsupported phase '{phase}'")

            organism.status = "evaluated" if summary.status in {"ok", "partial"} else "pending"
            update_latest_lineage_entry(
                organism,
                phase=phase,
                phase_score=summary.aggregate_score,
                selected_experiments=summary.selected_experiments,
            )
            save_organism_artifacts(organism)
            write_organism_meta(organism)
            self._write_phase_summary(organism, summary)

    def _write_phase_summary(
        self,
        organism: OrganismMeta,
        summary: Any,
    ) -> None:
        existing = read_organism_summary(organism.organism_dir) or {}
        phase_results = existing.get("phase_results", {})
        if not isinstance(phase_results, dict):
            phase_results = {}

        phase_results[summary.phase] = {
            "aggregate_score": summary.aggregate_score,
            "experiments": summary.per_experiment,
            "selected_experiments": summary.selected_experiments,
            "allocation_snapshot": summary.allocation_snapshot,
            "status": summary.status,
            "created_at": summary.created_at,
            "eval_finished_at": summary.eval_finished_at,
            "error_msg": summary.error_msg,
        }

        payload = {
            "organism_id": organism.organism_id,
            "island_id": organism.island_id,
            "generation_created": organism.generation_created,
            "current_generation_active": organism.current_generation_active,
            "operator": organism.operator,
            "status": organism.status,
            "simple_reward": organism.simple_reward,
            "hard_reward": organism.hard_reward,
            "phase_results": phase_results,
        }
        write_organism_summary(organism.organism_dir, payload)

    def _mark_eliminated(
        self,
        pool: list[OrganismMeta],
        survivors: list[OrganismMeta],
    ) -> None:
        survivor_ids = {organism.organism_id for organism in survivors}
        for organism in pool:
            if organism.organism_id in survivor_ids:
                continue
            organism.status = "eliminated"
            write_organism_meta(organism)

    async def run(self) -> dict[str, Any]:
        max_generations = int(self._require_cfg_value("evolver.max_generations"))

        state = self._load_state()
        if state is not None and bool(self.evolver_cfg.get("resume", True)):
            self.generation = int(state.get("current_generation", 0))
            self.population = self._restore_population_from_manifest(self.generation)
        else:
            self.generation = 0
            newborns = await self._seed_initial_population()
            await self._evaluate_phase(newborns, phase="simple")
            self.population = select_top_k_per_island(newborns, self._simple_top_k(), score_field="simple_reward")
            self._mark_eliminated(newborns, self.population)
            self._save_state()

        for generation in range(self.generation + 1, max_generations + 1):
            self.generation = generation
            active_by_island = self._group_by_island(self.population)
            offspring = self._produce_offspring(active_by_island)

            await self._evaluate_phase(offspring, phase="simple")
            candidate_pool = list(self.population) + offspring
            simple_survivors = select_top_k_per_island(
                candidate_pool,
                self._simple_top_k(),
                score_field="simple_reward",
            )
            self._mark_eliminated(candidate_pool, simple_survivors)

            if self._should_run_great_filter(generation):
                await self._evaluate_phase(simple_survivors, phase="hard")
                self.population = select_top_h_per_island(
                    simple_survivors,
                    self._great_filter_top_h(),
                    score_field="hard_reward",
                )
                self._mark_eliminated(simple_survivors, self.population)
            else:
                for organism in simple_survivors:
                    organism.current_generation_active = self.generation
                    write_organism_meta(organism)
                self.population = simple_survivors

            self._save_state()

        best = self._best_active_organism()
        return {
            "total_generations": self.generation,
            "active_population_size": len(self.population),
            "best_organism_id": best.organism_id if best is not None else None,
            "best_simple_reward": best.simple_reward if best is not None else None,
            "best_optimizer_path": best.optimizer_path if best is not None else None,
        }
