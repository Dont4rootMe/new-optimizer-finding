"""Multi-generation organism-first evolution loop with island-aware selection."""

from __future__ import annotations

import asyncio
import logging
import random
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from api_platforms import ApiPlatformRegistry
from src.evolve.generator import CandidateGenerator
from src.evolve.islands import load_islands
from src.evolve.orchestrator import EvolverOrchestrator
from src.evolve.selection import (
    select_top_h_per_island,
    select_top_k_per_island,
    softmax_select_distinct_organisms,
    softmax_select_organisms,
    weighted_rule_select_distinct_organisms,
    weighted_rule_select_organisms,
)
from src.evolve.storage import (
    generation_dir,
    genetic_code_path,
    implementation_path,
    lineage_path,
    organism_dir,
    phase_result_path,
    read_json,
    read_organism_meta,
    read_organism_summary,
    read_population_state,
    utc_now_iso,
    write_organism_meta,
    write_organism_summary,
    write_population_state,
)
from src.evolve.types import (
    Island,
    OrganismEvaluationRequest,
    OrganismMeta,
    PlannedOrganismCreation,
    PlannedPhaseEvaluation,
)
from src.evolve.visualization import render_evolution_overview
from src.organisms.crossbreeding import CrossbreedingOperator
from src.organisms.hypothesis_artifacts import read_canonical_genome
from src.organisms.mutation import MutationOperator
from src.organisms.organism import update_latest_lineage_entry

LOGGER = logging.getLogger(__name__)


def _announce(message: str) -> None:
    """Emit a progress line that is guaranteed to reach the user's terminal.

    The creation pipeline runs inside worker threads spawned by Hydra-managed
    entrypoints. Hydra reconfigures root logging at import time, which at least
    historically swallowed per-module LOGGER output on stderr (see the comment
    on `_record_creation_event`). We therefore write directly to stderr with an
    explicit flush in addition to calling LOGGER.info, so that "starting
    sampling generation N" style milestones surface live even when the logging
    tree is uncooperative.
    """

    LOGGER.info(message)
    try:
        print(f"[evolve] {message}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        pass


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
    """Runs multi-generation island-aware organism evolution."""

    def __init__(self, cfg: DictConfig, llm_registry: ApiPlatformRegistry | None = None) -> None:
        self.cfg = cfg
        self.evolver_cfg = cfg.evolver
        self.population_root = Path(str(cfg.paths.population_root)).expanduser().resolve()
        self.population: list[OrganismMeta] = []
        self.generation = 0

        self.llm_registry = llm_registry or ApiPlatformRegistry(cfg)
        self._owns_llm_registry = llm_registry is None
        self.generator = CandidateGenerator(cfg, llm_registry=self.llm_registry)
        self.hypothesis_schema_provider = self.generator.hypothesis_schema_provider
        self.orchestrator = EvolverOrchestrator(cfg, repair_callback=self._repair_organism_after_eval_error)
        self.rng = random.Random(int(cfg.seed))
        self.max_parallel_organisms = self._max_parallel_organisms()

        self.islands = self._load_islands()
        self.islands_by_id = {island.island_id: island for island in self.islands}

        self.seed_organisms_per_island = self._seed_organisms_per_island()
        self.max_organisms_per_island = self._max_organisms_per_island()
        self.offspring_per_generation = self._offspring_per_generation()
        self.operator_selection_strategy = self._operator_selection_strategy()
        self.reproduction_operator_weights = self._reproduction_operator_weights()
        self.reproduction_island_sampling = self._reproduction_island_sampling()
        self.species_sampling_strategy = self._species_sampling_strategy()
        self.species_sampling_weighted_rule_lambda = self._species_sampling_weighted_rule_lambda()
        self.mutation_softmax_temperature = self._mutation_softmax_temperature()
        self.within_island_crossover_softmax_temperature = self._within_island_crossover_softmax_temperature()
        self.inter_island_crossover_softmax_temperature = self._inter_island_crossover_softmax_temperature()
        self.mutation_gene_removal_probability = self._mutation_gene_removal_probability()
        self.crossover_primary_parent_gene_inheritance_probability = (
            self._crossover_primary_parent_gene_inheritance_probability()
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

    def _max_parallel_organisms(self) -> int:
        return max(1, int(self._require_cfg_value("evolver.creation.max_parallel_organisms")))

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

    def _species_sampling_value(self, key: str) -> Any:
        return self._require_cfg_value(f"evolver.reproduction.species_sampling.{key}")

    def _species_sampling_strategy(self) -> str:
        return str(self._species_sampling_value("strategy"))

    def _species_sampling_weighted_rule_lambda(self) -> float:
        return float(self._species_sampling_value("weighted_rule_lambda"))

    def _mutation_softmax_temperature(self) -> float:
        return float(self._species_sampling_value("mutation_softmax_temperature"))

    def _within_island_crossover_softmax_temperature(self) -> float:
        return float(self._species_sampling_value("within_island_crossover_softmax_temperature"))

    def _inter_island_crossover_softmax_temperature(self) -> float:
        return float(self._species_sampling_value("inter_island_crossover_softmax_temperature"))

    def _mutation_gene_removal_probability(self) -> float:
        return float(self._require_cfg_value("evolver.operators.mutation.gene_removal_probability"))

    def _crossover_primary_parent_gene_inheritance_probability(self) -> float:
        return float(
            self._require_cfg_value("evolver.operators.crossover.primary_parent_gene_inheritance_probability")
        )

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

    def _great_filter_top_h(self) -> int:
        return int(self._phase_value("hard", "top_h_per_island"))

    def _great_filter_enabled(self) -> bool:
        return bool(self._phase_value("hard", "enabled"))

    def _great_filter_interval(self) -> int:
        return max(1, int(self._phase_value("hard", "interval_generations")))

    def _validate_phase_selection_bounds(self) -> None:
        top_h = self._great_filter_top_h()
        if top_h > self.max_organisms_per_island:
            raise ValueError(
                "Invalid evolve config: evolver.phases.great_filter.top_h_per_island "
                "must be <= evolver.islands.max_organisms_per_island"
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
        if self.species_sampling_strategy not in {"softmax", "weighted_rule"}:
            raise ValueError(
                "Invalid evolve config: evolver.reproduction.species_sampling.strategy "
                "must be 'softmax' or 'weighted_rule'"
            )
        if self.species_sampling_weighted_rule_lambda <= 0:
            raise ValueError(
                "Invalid evolve config: evolver.reproduction.species_sampling.weighted_rule_lambda must be > 0"
            )
        for key, temperature in (
            ("mutation_softmax_temperature", self.mutation_softmax_temperature),
            (
                "within_island_crossover_softmax_temperature",
                self.within_island_crossover_softmax_temperature,
            ),
            (
                "inter_island_crossover_softmax_temperature",
                self.inter_island_crossover_softmax_temperature,
            ),
        ):
            if temperature <= 0:
                raise ValueError(
                    "Invalid evolve config: evolver.reproduction.species_sampling."
                    f"{key} must be > 0"
                )

    def _should_run_great_filter(self, generation: int) -> bool:
        return self._great_filter_enabled() and generation > 0 and generation % self._great_filter_interval() == 0

    def _best_active_organism(self) -> OrganismMeta | None:
        if not self.population:
            return None
        return max(
            self.population,
            key=lambda organism: organism.simple_score if organism.simple_score is not None else -float("inf"),
        )

    def _save_state(
        self,
        *,
        finalized_generation: int | None = None,
        inflight_seed: dict[str, Any] | None = None,
        inflight_generation: dict[str, Any] | None = None,
    ) -> None:
        active_generation = self.generation if finalized_generation is None else finalized_generation
        for organism in self.population:
            organism.current_generation_active = active_generation
            write_organism_meta(organism)

        best = self._best_active_organism()
        write_population_state(
            self.population_root,
            active_generation,
            self.population,
            best_organism_id=best.organism_id if best is not None else None,
            best_simple_score=best.simple_score if best is not None else None,
            inflight_seed=inflight_seed,
            inflight_generation=inflight_generation,
        )

    def _render_progress_snapshot(self) -> None:
        try:
            out_path = render_evolution_overview(self.population_root)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to render evolution overview for %s", self.population_root)
            return
        if out_path is not None:
            LOGGER.info("Updated evolution overview at %s", out_path)

    def _load_state(self) -> dict[str, Any] | None:
        payload = read_population_state(self.population_root)
        if isinstance(payload, dict):
            return payload
        return None

    def _restore_population_from_state(self, generation: int) -> list[OrganismMeta]:
        state = read_population_state(self.population_root)
        if state is None:
            raise FileNotFoundError("population_state.json is required for canonical resume.")

        state_generation = int(state.get("current_generation", generation))
        if state_generation != generation:
            LOGGER.warning(
                "Population state points to generation %d while requested generation is %d. "
                "Using population state generation.",
                state_generation,
                generation,
            )
            self.generation = state_generation

        organisms: list[OrganismMeta] = []
        for entry in state.get("active_organisms", []):
            organism_dir_path = Path(str(entry["organism_dir"]))
            if not organism_dir_path.is_absolute():
                organism_dir_path = (self.population_root / organism_dir_path).resolve()
            if not organism_dir_path.exists():
                raise FileNotFoundError(
                    f"population_state.json points to missing organism dir: {organism_dir_path}"
                )
            organism = read_organism_meta(organism_dir_path)
            self._validate_canonical_hypothesis_on_resume(organism)
            organism.current_generation_active = int(entry.get("current_generation_active", self.generation))
            organisms.append(organism)
        return organisms

    def _validate_canonical_hypothesis_on_resume(self, organism: OrganismMeta) -> None:
        if self.hypothesis_schema_provider is None:
            return
        try:
            read_canonical_genome(Path(organism.organism_dir), self.hypothesis_schema_provider)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Canonical resume requires genome.json for organism {organism.organism_id}; "
                "legacy circle-packing organisms without genome.json are not supported by this path."
            ) from exc

    async def _seed_initial_population(self) -> list[OrganismMeta]:
        """Compatibility helper that uses the shared queued seed pipeline."""

        return await self._execute_planned_creations(self._plan_seed_population(), state_kind="seed")

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

    def _select_parent_organisms(
        self,
        population: list[OrganismMeta],
        *,
        k: int,
        distinct: bool,
        softmax_temperature: float,
        parent_offspring_counts: dict[str, int],
    ) -> list[OrganismMeta]:
        if self.species_sampling_strategy == "softmax":
            if distinct:
                return softmax_select_distinct_organisms(
                    population,
                    score_field="simple_score",
                    temperature=softmax_temperature,
                    k=k,
                    rng=self.rng,
                )
            return softmax_select_organisms(
                population,
                score_field="simple_score",
                temperature=softmax_temperature,
                k=k,
                rng=self.rng,
            )
        if self.species_sampling_strategy == "weighted_rule":
            if distinct:
                return weighted_rule_select_distinct_organisms(
                    population,
                    parent_offspring_counts=parent_offspring_counts,
                    score_field="simple_score",
                    weighted_rule_lambda=self.species_sampling_weighted_rule_lambda,
                    k=k,
                    rng=self.rng,
                )
            return weighted_rule_select_organisms(
                population,
                parent_offspring_counts=parent_offspring_counts,
                score_field="simple_score",
                weighted_rule_lambda=self.species_sampling_weighted_rule_lambda,
                k=k,
                rng=self.rng,
            )
        raise ValueError(f"Unsupported species sampling strategy '{self.species_sampling_strategy}'.")

    @staticmethod
    def _increment_parent_offspring_counts(
        parent_offspring_counts: dict[str, int],
        *parents: OrganismMeta,
    ) -> None:
        for parent in parents:
            parent_offspring_counts[parent.organism_id] = parent_offspring_counts.get(parent.organism_id, 0) + 1

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
        parent: OrganismMeta,
        organism_id: str,
        org_dir: Path,
        operator_seed: int,
    ) -> OrganismMeta:
        mutation_operator = MutationOperator(
            q=self.mutation_gene_removal_probability,
            seed=operator_seed,
        )
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
        mother: OrganismMeta,
        father: OrganismMeta,
        organism_id: str,
        org_dir: Path,
        operator_seed: int,
    ) -> OrganismMeta:
        crossover_operator = CrossbreedingOperator(
            p=self.crossover_primary_parent_gene_inheritance_probability,
            seed=operator_seed,
        )
        return crossover_operator.produce(
            mother=mother,
            father=father,
            organism_id=organism_id,
            generation=self.generation,
            org_dir=org_dir,
            generator=self.generator,
        )

    def _create_inter_island_crossover_offspring(
        self,
        *,
        mother: OrganismMeta,
        father: OrganismMeta,
        organism_id: str,
        org_dir: Path,
        operator_seed: int,
    ) -> OrganismMeta:
        crossover_operator = CrossbreedingOperator(
            p=self.crossover_primary_parent_gene_inheritance_probability,
            seed=operator_seed,
        )
        return crossover_operator.produce(
            mother=mother,
            father=father,
            organism_id=organism_id,
            generation=self.generation,
            org_dir=org_dir,
            generator=self.generator,
        )

    def _simple_evaluation_request(
        self,
        *,
        organism_id: str,
        organism_dir_path: Path,
        created_at: str,
    ) -> OrganismEvaluationRequest:
        return OrganismEvaluationRequest(
            organism_id=organism_id,
            organism_dir=str(organism_dir_path),
            phase="simple",
            experiments=self._phase_experiments("simple"),
            allocation_cfg=self._phase_allocation_cfg("simple"),
            eval_mode=self._phase_eval_mode("simple"),
            timeout_sec=self._phase_timeout_sec("simple"),
            created_at=created_at,
        )

    def _record_creation_event(
        self,
        plan: PlannedOrganismCreation,
        *,
        event: str,
        message: str,
        include_traceback: bool = False,
    ) -> None:
        """Append a creation-pipeline event to `org_dir/logs/creation.err`.

        This file is the survivable ground-truth for diagnosing organism
        creation hangs/crashes: it is written synchronously on disk at every
        transition (`started`, `completed`, `failed`) so even a hard SIGKILL
        leaves enough context to understand where the pipeline stopped. It
        intentionally bypasses the Python `logging` framework because Hydra
        reconfigures root handlers at import time and swallows per-module
        logger output from long-running creation threads.
        """

        org_dir = Path(plan.organism_dir)
        log_dir = org_dir / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        log_path = log_dir / "creation.err"
        timestamp = datetime.now(timezone.utc).isoformat()
        lines = [
            f"[{timestamp}] event={event} organism_id={plan.organism_id} "
            f"generation={plan.generation} island={plan.island_id} route={plan.route}",
            f"  {message}",
        ]
        if include_traceback:
            tb_text = traceback.format_exc()
            if tb_text and tb_text.strip() and tb_text.strip() != "NoneType: None":
                lines.append("  traceback:")
                lines.extend("    " + raw_line for raw_line in tb_text.rstrip().splitlines())
        lines.append("")
        try:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
                handle.flush()
        except OSError:
            return

    def _write_planned_organism_stub(self, plan: PlannedOrganismCreation) -> None:
        placeholder = OrganismMeta(
            organism_id=plan.organism_id,
            island_id=plan.island_id,
            generation_created=plan.generation,
            current_generation_active=plan.generation,
            timestamp=plan.timestamp,
            mother_id=plan.mother_id,
            father_id=plan.father_id,
            operator=plan.operator,
            genetic_code_path=str(genetic_code_path(plan.organism_dir)),
            implementation_path=str(implementation_path(plan.organism_dir)),
            lineage_path=str(lineage_path(plan.organism_dir)),
            organism_dir=plan.organism_dir,
            ancestor_ids=[],
            experiment_report_index={},
            status="pending",
            pipeline_state=plan.pipeline_state,
            error_msg=plan.error_msg,
            planned_phase_evaluations={
                phase: phase_plan.to_dict() for phase, phase_plan in plan.planned_phase_evaluations.items()
            },
        )
        write_organism_meta(placeholder)

    def _serialize_planned_creation_state(
        self,
        planned_organisms: list[PlannedOrganismCreation],
        *,
        planned_key: str,
        include_parent_snapshot: bool,
    ) -> dict[str, Any]:
        payload = {
            "target_generation": self.generation,
            planned_key: [plan.to_dict() for plan in planned_organisms],
            "creation_queue": {
                "pending": [plan.organism_id for plan in planned_organisms if plan.pipeline_state == "planned_creation"],
                "active": [plan.organism_id for plan in planned_organisms if plan.pipeline_state == "creating"],
                "completed": [
                    plan.organism_id
                    for plan in planned_organisms
                    if plan.pipeline_state
                    in {"pending_simple_eval", "running_simple_eval", "simple_complete", "failed_simple_eval"}
                ],
                "failed": [plan.organism_id for plan in planned_organisms if plan.pipeline_state == "failed_creation"],
            },
            "simple_eval_queue": {
                "pending": [plan.organism_id for plan in planned_organisms if plan.pipeline_state == "pending_simple_eval"],
                "active": [plan.organism_id for plan in planned_organisms if plan.pipeline_state == "running_simple_eval"],
                "completed": [plan.organism_id for plan in planned_organisms if plan.pipeline_state == "simple_complete"],
                "failed": [
                    plan.organism_id for plan in planned_organisms if plan.pipeline_state == "failed_simple_eval"
                ],
            },
            "completed": all(
                plan.pipeline_state in {"simple_complete", "failed_creation", "failed_simple_eval"}
                for plan in planned_organisms
            ),
            "failed": any(
                plan.pipeline_state in {"failed_creation", "failed_simple_eval"} for plan in planned_organisms
            ),
            "finalized": False,
        }
        if include_parent_snapshot:
            payload["parent_snapshot"] = [organism.to_dict() for organism in self.population]
        return payload

    def _serialize_inflight_generation(
        self,
        planned_offspring: list[PlannedOrganismCreation],
    ) -> dict[str, Any]:
        return self._serialize_planned_creation_state(
            planned_offspring,
            planned_key="planned_offspring",
            include_parent_snapshot=True,
        )

    def _serialize_inflight_seed(
        self,
        planned_organisms: list[PlannedOrganismCreation],
    ) -> dict[str, Any]:
        return self._serialize_planned_creation_state(
            planned_organisms,
            planned_key="planned_organisms",
            include_parent_snapshot=False,
        )

    def _persist_inflight_generation(self, planned_offspring: list[PlannedOrganismCreation]) -> None:
        self._save_state(
            finalized_generation=max(0, self.generation - 1),
            inflight_generation=self._serialize_inflight_generation(planned_offspring),
        )

    def _persist_inflight_seed(self, planned_organisms: list[PlannedOrganismCreation]) -> None:
        self._save_state(
            finalized_generation=0,
            inflight_seed=self._serialize_inflight_seed(planned_organisms),
            inflight_generation=None,
        )

    def _resolve_parent_meta(
        self,
        *,
        parent_id: str | None,
        parent_dir: str | None,
        active_by_id: dict[str, OrganismMeta],
    ) -> OrganismMeta | None:
        if not parent_id:
            return None
        parent = active_by_id.get(parent_id)
        if parent is not None:
            return parent
        if parent_dir:
            return read_organism_meta(parent_dir)
        raise FileNotFoundError(f"Unable to resolve parent organism '{parent_id}' for inflight generation.")

    def _assign_batch_routes(
        self,
        planned_organisms: list[PlannedOrganismCreation],
    ) -> dict[str, str]:
        """Return a balanced organism_id -> route_id assignment for a batch.

        For small batches (e.g. the 2-organism seed on two islands) the
        per-organism hash inside `BaseLlmGenerator.sample_route_id` has ~50%
        chance of sending both organisms to the same route, which leaves one
        Ollama server idle and doubles wallclock. Here we pre-compute a
        deterministic, weight-proportional allocation and stride-interleave
        the slots so that even if `max_parallel_organisms` is smaller than the
        batch size, each wave still hits distinct routes when possible.

        The allocation uses Hamilton (largest-remainder) apportionment:
        `count_i = floor(n * w_i / sum(w))`, then the `n - sum(count_i)`
        leftover slots go to the routes with the largest fractional
        remainders, ties broken by route id (stable). The resulting slot
        sequence is produced by iteratively picking the route with the
        largest remaining quota, tie-broken by route id — this interleaves
        slots so the first k plans hit k distinct routes whenever possible.
        """

        if not planned_organisms:
            return {}

        route_weights = self.generator.route_weights
        positive_routes = sorted(
            (route_id for route_id, weight in route_weights.items() if weight > 0)
        )
        if len(positive_routes) <= 1:
            return {}

        total_weight = sum(route_weights[route_id] for route_id in positive_routes)
        if total_weight <= 0:
            return {}

        n = len(planned_organisms)
        exact = [
            (route_id, n * route_weights[route_id] / total_weight)
            for route_id in positive_routes
        ]
        counts = {route_id: int(value) for route_id, value in exact}
        leftover = n - sum(counts.values())
        if leftover > 0:
            remainders = sorted(
                exact,
                key=lambda pair: (-(pair[1] - int(pair[1])), pair[0]),
            )
            for idx in range(leftover):
                counts[remainders[idx % len(remainders)][0]] += 1

        slot_sequence: list[str] = []
        remaining = dict(counts)
        while sum(remaining.values()) > 0:
            pick = min(
                ((route_id, qty) for route_id, qty in remaining.items() if qty > 0),
                key=lambda pair: (-pair[1], pair[0]),
            )[0]
            slot_sequence.append(pick)
            remaining[pick] -= 1

        ordered_plans = sorted(planned_organisms, key=lambda plan: plan.organism_id)
        return {
            plan.organism_id: slot_sequence[idx]
            for idx, plan in enumerate(ordered_plans)
        }

    def _plan_seed_population(self) -> list[PlannedOrganismCreation]:
        if self.seed_organisms_per_island <= 0:
            return []

        gen_dir = generation_dir(self.population_root, 0)
        planned_organisms: list[PlannedOrganismCreation] = []
        for island in self.islands:
            for _ in range(self.seed_organisms_per_island):
                organism_id, org_dir = self._newborn_org_dir(gen_dir=gen_dir, island_id=island.island_id)
                operator_seed = self.rng.randint(0, 2**31 - 1)
                timestamp = utc_now_iso()
                phase_plan = self.orchestrator.plan_phase_evaluation(
                    self._simple_evaluation_request(
                        organism_id=organism_id,
                        organism_dir_path=org_dir,
                        created_at=timestamp,
                    )
                )
                plan = PlannedOrganismCreation(
                    organism_id=organism_id,
                    organism_dir=str(org_dir),
                    island_id=island.island_id,
                    generation=0,
                    route="seed",
                    operator="seed",
                    mother_id=None,
                    mother_organism_dir=None,
                    father_id=None,
                    father_organism_dir=None,
                    father_island_id=None,
                    operator_seed=operator_seed,
                    timestamp=timestamp,
                    planned_phase_evaluations={"simple": phase_plan},
                )
                self._write_planned_organism_stub(plan)
                planned_organisms.append(plan)
        return planned_organisms

    def _plan_offspring_generation(
        self,
        active_by_island: dict[str, list[OrganismMeta]],
    ) -> list[PlannedOrganismCreation]:
        if self.offspring_per_generation <= 0:
            return []

        gen_dir = generation_dir(self.population_root, self.generation)
        planned_offspring: list[PlannedOrganismCreation] = []
        parent_offspring_counts: dict[str, int] = {}
        for route in self._planned_reproduction_routes(active_by_island):
            if route == _ROUTE_MUTATION:
                candidate_islands = self._eligible_island_ids_for_route(_ROUTE_MUTATION, active_by_island)
                island_id = self._sample_island_ids(
                    route=_ROUTE_MUTATION,
                    candidate_island_ids=candidate_islands,
                    count=1,
                    distinct=False,
                )[0]
                parent = self._select_parent_organisms(
                    active_by_island[island_id],
                    k=1,
                    distinct=False,
                    softmax_temperature=self.mutation_softmax_temperature,
                    parent_offspring_counts=parent_offspring_counts,
                )[0]
                organism_id, org_dir = self._newborn_org_dir(gen_dir=gen_dir, island_id=island_id)
                operator_seed = self.rng.randint(0, 2**31 - 1)
                timestamp = utc_now_iso()
                phase_plan = self.orchestrator.plan_phase_evaluation(
                    self._simple_evaluation_request(
                        organism_id=organism_id,
                        organism_dir_path=org_dir,
                        created_at=timestamp,
                    )
                )
                plan = PlannedOrganismCreation(
                    organism_id=organism_id,
                    organism_dir=str(org_dir),
                    island_id=island_id,
                    generation=self.generation,
                    route=route,
                    operator="mutation",
                    mother_id=parent.organism_id,
                    mother_organism_dir=parent.organism_dir,
                    father_id=None,
                    father_organism_dir=None,
                    father_island_id=None,
                    operator_seed=operator_seed,
                    timestamp=timestamp,
                    planned_phase_evaluations={"simple": phase_plan},
                )
            elif route == _ROUTE_WITHIN_ISLAND_CROSSOVER:
                candidate_islands = self._eligible_island_ids_for_route(_ROUTE_WITHIN_ISLAND_CROSSOVER, active_by_island)
                island_id = self._sample_island_ids(
                    route=_ROUTE_WITHIN_ISLAND_CROSSOVER,
                    candidate_island_ids=candidate_islands,
                    count=1,
                    distinct=False,
                )[0]
                primary_parent, secondary_parent = self._select_parent_organisms(
                    active_by_island[island_id],
                    k=2,
                    distinct=True,
                    softmax_temperature=self.within_island_crossover_softmax_temperature,
                    parent_offspring_counts=parent_offspring_counts,
                )
                if primary_parent.organism_id == secondary_parent.organism_id:
                    raise ValueError("Within-island crossover sampled duplicate parents; expected distinct organisms.")
                organism_id, org_dir = self._newborn_org_dir(gen_dir=gen_dir, island_id=island_id)
                operator_seed = self.rng.randint(0, 2**31 - 1)
                timestamp = utc_now_iso()
                phase_plan = self.orchestrator.plan_phase_evaluation(
                    self._simple_evaluation_request(
                        organism_id=organism_id,
                        organism_dir_path=org_dir,
                        created_at=timestamp,
                    )
                )
                plan = PlannedOrganismCreation(
                    organism_id=organism_id,
                    organism_dir=str(org_dir),
                    island_id=island_id,
                    generation=self.generation,
                    route=route,
                    operator="crossover",
                    mother_id=primary_parent.organism_id,
                    mother_organism_dir=primary_parent.organism_dir,
                    father_id=secondary_parent.organism_id,
                    father_organism_dir=secondary_parent.organism_dir,
                    father_island_id=secondary_parent.island_id,
                    operator_seed=operator_seed,
                    timestamp=timestamp,
                    planned_phase_evaluations={"simple": phase_plan},
                )
            elif route == _ROUTE_INTER_ISLAND_CROSSOVER:
                candidate_islands = self._eligible_island_ids_for_route(_ROUTE_INTER_ISLAND_CROSSOVER, active_by_island)
                primary_island_id, secondary_island_id = self._sample_island_ids(
                    route=_ROUTE_INTER_ISLAND_CROSSOVER,
                    candidate_island_ids=candidate_islands,
                    count=2,
                    distinct=True,
                )
                primary_parent = self._select_parent_organisms(
                    active_by_island[primary_island_id],
                    k=1,
                    distinct=False,
                    softmax_temperature=self.inter_island_crossover_softmax_temperature,
                    parent_offspring_counts=parent_offspring_counts,
                )[0]
                secondary_parent = self._select_parent_organisms(
                    active_by_island[secondary_island_id],
                    k=1,
                    distinct=False,
                    softmax_temperature=self.inter_island_crossover_softmax_temperature,
                    parent_offspring_counts=parent_offspring_counts,
                )[0]
                organism_id, org_dir = self._newborn_org_dir(gen_dir=gen_dir, island_id=primary_island_id)
                operator_seed = self.rng.randint(0, 2**31 - 1)
                timestamp = utc_now_iso()
                phase_plan = self.orchestrator.plan_phase_evaluation(
                    self._simple_evaluation_request(
                        organism_id=organism_id,
                        organism_dir_path=org_dir,
                        created_at=timestamp,
                    )
                )
                plan = PlannedOrganismCreation(
                    organism_id=organism_id,
                    organism_dir=str(org_dir),
                    island_id=primary_island_id,
                    generation=self.generation,
                    route=route,
                    operator="crossover",
                    mother_id=primary_parent.organism_id,
                    mother_organism_dir=primary_parent.organism_dir,
                    father_id=secondary_parent.organism_id,
                    father_organism_dir=secondary_parent.organism_dir,
                    father_island_id=secondary_parent.island_id,
                    operator_seed=operator_seed,
                    timestamp=timestamp,
                    planned_phase_evaluations={"simple": phase_plan},
                )
            else:
                raise ValueError(f"Unsupported reproduction route '{route}'")

            self._write_planned_organism_stub(plan)
            if route == _ROUTE_MUTATION:
                self._increment_parent_offspring_counts(parent_offspring_counts, parent)
            else:
                self._increment_parent_offspring_counts(parent_offspring_counts, primary_parent, secondary_parent)
            planned_offspring.append(plan)
        return planned_offspring

    def _existing_eval_results(self, phase_plan: PlannedPhaseEvaluation) -> dict[str, dict[str, Any]]:
        existing: dict[str, dict[str, Any]] = {}
        for exp_name, task_state in phase_plan.task_states.items():
            task_status = str(task_state.get("status", "planned"))
            if task_status in {"planned", "queued", "running"}:
                continue
            result_path = str(task_state.get("result_path", "")).strip()
            error_history = task_state.get("errors", [])
            if result_path and Path(result_path).exists():
                payload = read_json(result_path)
                if isinstance(payload, dict):
                    if isinstance(error_history, list) and error_history:
                        payload = dict(payload)
                        payload["errors"] = [dict(entry) for entry in error_history if isinstance(entry, dict)]
                    existing[exp_name] = payload
                    continue
            existing[exp_name] = {
                "status": task_status,
                "score": None,
                "error_msg": task_state.get("error_msg"),
                "errors": [dict(entry) for entry in error_history if isinstance(entry, dict)]
                if isinstance(error_history, list)
                else [],
            }
        return existing

    def _repair_organism_after_eval_error(
        self,
        organism_dir: str,
        phase: str,
        experiment_name: str,
        errors: list[dict[str, Any]],
    ) -> None:
        organism = read_organism_meta(organism_dir)
        _announce(
            f"organism {organism.organism_id} repair requested "
            f"(phase={phase}, experiment={experiment_name}, attempt={len(errors)})"
        )
        self.generator.repair_organism_after_error(
            organism=organism,
            phase=phase,
            experiment_name=experiment_name,
            errors=errors,
        )

    def _materialize_planned_organism(
        self,
        plan: PlannedOrganismCreation,
        active_by_id: dict[str, OrganismMeta],
    ) -> OrganismMeta:
        mother = self._resolve_parent_meta(
            parent_id=plan.mother_id,
            parent_dir=plan.mother_organism_dir,
            active_by_id=active_by_id,
        )
        father = self._resolve_parent_meta(
            parent_id=plan.father_id,
            parent_dir=plan.father_organism_dir,
            active_by_id=active_by_id,
        )
        org_dir = Path(plan.organism_dir)

        if plan.route == "seed":
            island = self.islands_by_id.get(plan.island_id)
            if island is None:
                raise ValueError(f"Seed plan for organism {plan.organism_id} references unknown island '{plan.island_id}'.")
            organism = self.generator.generate_seed_organism(
                island=island,
                organism_id=plan.organism_id,
                generation=plan.generation,
                organism_dir=org_dir,
            )
        elif plan.route == _ROUTE_MUTATION:
            if mother is None:
                raise ValueError(f"Mutation plan for organism {plan.organism_id} is missing parent.")
            organism = self._create_mutation_offspring(
                parent=mother,
                organism_id=plan.organism_id,
                org_dir=org_dir,
                operator_seed=plan.operator_seed,
            )
        elif plan.route == _ROUTE_WITHIN_ISLAND_CROSSOVER:
            if mother is None or father is None:
                raise ValueError(f"Within-island crossover plan for organism {plan.organism_id} is missing parents.")
            organism = self._create_within_island_crossover_offspring(
                mother=mother,
                father=father,
                organism_id=plan.organism_id,
                org_dir=org_dir,
                operator_seed=plan.operator_seed,
            )
        elif plan.route == _ROUTE_INTER_ISLAND_CROSSOVER:
            if mother is None or father is None:
                raise ValueError(f"Inter-island crossover plan for organism {plan.organism_id} is missing parents.")
            organism = self._create_inter_island_crossover_offspring(
                mother=mother,
                father=father,
                organism_id=plan.organism_id,
                org_dir=org_dir,
                operator_seed=plan.operator_seed,
            )
        else:
            raise ValueError(f"Unsupported reproduction route '{plan.route}'")

        organism.pipeline_state = "pending_simple_eval"
        organism.planned_phase_evaluations = {
            phase: phase_plan.to_dict() for phase, phase_plan in plan.planned_phase_evaluations.items()
        }
        write_organism_meta(organism)
        return organism

    def _phase_request_from_plan(self, plan: PlannedOrganismCreation, *, phase: str) -> OrganismEvaluationRequest:
        if phase != "simple":
            raise ValueError(f"Unsupported planned phase '{phase}'")
        return self._simple_evaluation_request(
            organism_id=plan.organism_id,
            organism_dir_path=Path(plan.organism_dir),
            created_at=plan.timestamp,
        )

    def _simple_summary_status_to_pipeline_state(self, summary_status: str) -> str:
        return "simple_complete" if summary_status in {"ok", "partial"} else "failed_simple_eval"

    def _summary_error_excerpt(self, summary: Any) -> str | None:
        messages: list[str] = []
        per_experiment = getattr(summary, "per_experiment", {})
        if not isinstance(per_experiment, dict):
            return None
        for exp_name, payload in per_experiment.items():
            if not isinstance(payload, dict):
                continue
            status = str(payload.get("status", "")).strip()
            error_msg = payload.get("error_msg")
            raw_report = payload.get("raw_report")
            if not error_msg and isinstance(raw_report, dict):
                error_msg = raw_report.get("error_msg")
            if status == "ok" and not error_msg:
                continue
            detail = str(error_msg).strip() if error_msg else status or "unknown failure"
            messages.append(f"{exp_name}: {detail}")
        if not messages:
            return None
        return "; ".join(messages[:2])

    def _announce_phase_summary(
        self,
        organism_id: str,
        summary: Any,
    ) -> None:
        error_excerpt = self._summary_error_excerpt(summary)
        if getattr(summary, "status", "") in {"ok", "partial"}:
            _announce(
                f"organism {organism_id} {summary.phase} eval completed "
                f"(status={summary.status}, score={summary.aggregate_score})"
            )
            return
        message = (
            f"organism {organism_id} {summary.phase} eval FAILED "
            f"(status={summary.status}, score={summary.aggregate_score})"
        )
        if error_excerpt:
            message += f": {error_excerpt}"
        _announce(message)

    def _planned_phase_error_excerpt(self, phase_plan: PlannedPhaseEvaluation | None) -> str | None:
        if phase_plan is None:
            return None
        messages: list[str] = []
        for exp_name, task_state in phase_plan.task_states.items():
            if not isinstance(task_state, dict):
                continue
            status = str(task_state.get("status", "")).strip()
            error_msg = task_state.get("error_msg")
            result_path = str(task_state.get("result_path", "")).strip()
            if not error_msg and result_path and Path(result_path).exists():
                try:
                    payload = read_json(result_path)
                except Exception:  # noqa: BLE001
                    payload = None
                if isinstance(payload, dict):
                    error_msg = payload.get("error_msg")
                    if not status:
                        status = str(payload.get("status", "")).strip()
            if status == "ok" and not error_msg:
                continue
            if not status and not error_msg:
                continue
            detail = str(error_msg).strip() if error_msg else status or "unknown failure"
            messages.append(f"{exp_name}: {detail}")
        if not messages:
            return None
        return "; ".join(messages[:2])

    def _seed_failure_message(
        self,
        planned_organisms: list[PlannedOrganismCreation],
        newborns: list[OrganismMeta],
    ) -> str:
        pipeline_counts: dict[str, int] = {}
        for plan in planned_organisms:
            pipeline_state = str(plan.pipeline_state or "unknown")
            pipeline_counts[pipeline_state] = pipeline_counts.get(pipeline_state, 0) + 1

        counts_text = ", ".join(
            f"{state}={count}" for state, count in sorted(pipeline_counts.items())
        )
        message = (
            "Seed population produced 0 active organisms after simple evaluation. "
            f"planned={len(planned_organisms)}, materialized={len(newborns)}, "
            f"pipeline_states=[{counts_text}]"
        )

        examples: list[str] = []
        for plan in planned_organisms:
            if plan.pipeline_state == "failed_creation":
                detail = str(plan.error_msg).strip() if plan.error_msg else "unknown creation error"
                examples.append(f"{plan.organism_id} creation failed: {detail}")
            elif plan.pipeline_state == "failed_simple_eval":
                detail = self._planned_phase_error_excerpt(plan.planned_phase_evaluations.get("simple"))
                if not detail:
                    detail = "see organism logs/results"
                examples.append(f"{plan.organism_id} simple eval failed: {detail}")
            if len(examples) >= 3:
                break

        if examples:
            message += " Example failures: " + " | ".join(examples)
        return message

    def _submit_or_resume_simple_evaluation(
        self,
        plan: PlannedOrganismCreation,
        organism: OrganismMeta,
    ) -> Any:
        phase_plan = plan.planned_phase_evaluations["simple"]
        existing_results = self._existing_eval_results(phase_plan)
        for exp_name, task_state in phase_plan.task_states.items():
            if exp_name not in existing_results and task_state.get("status") in {"planned", "queued", "running"}:
                task_state["status"] = "queued"
        phase_plan.status = "queued"
        plan.pipeline_state = "running_simple_eval"
        organism.pipeline_state = "running_simple_eval"
        organism.planned_phase_evaluations = {
            phase: phase_payload.to_dict() for phase, phase_payload in plan.planned_phase_evaluations.items()
        }
        write_organism_meta(organism)
        _announce(
            f"organism {plan.organism_id} queued simple eval "
            f"(experiments={','.join(phase_plan.selected_experiments)})"
        )
        return self.orchestrator.submit_planned_request(
            self._phase_request_from_plan(plan, phase="simple"),
            phase_plan,
            existing_results=existing_results,
        )

    def _apply_completed_summary(
        self,
        plan: PlannedOrganismCreation,
        organism: OrganismMeta,
        summary: Any,
    ) -> None:
        if summary.phase == "simple":
            organism.simple_score = summary.aggregate_score
        elif summary.phase == "hard":
            organism.hard_score = summary.aggregate_score
        else:
            raise ValueError(f"Unsupported phase '{summary.phase}'")

        phase_plan = plan.planned_phase_evaluations.get(summary.phase)
        if phase_plan is not None:
            phase_plan.status = "completed" if summary.status in {"ok", "partial"} else "failed"
        pipeline_state = self._simple_summary_status_to_pipeline_state(summary.status)
        plan.pipeline_state = pipeline_state
        organism.pipeline_state = pipeline_state
        organism.status = "evaluated" if summary.status in {"ok", "partial"} else "pending"
        organism.planned_phase_evaluations = {
            phase: phase_payload.to_dict() for phase, phase_payload in plan.planned_phase_evaluations.items()
        }
        update_latest_lineage_entry(
            organism,
            phase=summary.phase,
            phase_score=summary.aggregate_score,
            selected_experiments=summary.selected_experiments,
        )
        write_organism_meta(organism)
        self._write_phase_summary(organism, summary)

    async def _execute_planned_creations(
        self,
        planned_organisms: list[PlannedOrganismCreation],
        *,
        state_kind: str,
    ) -> list[OrganismMeta]:
        if not planned_organisms:
            _announce(
                f"no organisms to create for state_kind={state_kind} (generation={self.generation})"
            )
            return []

        batch_route_assignments = self._assign_batch_routes(planned_organisms)
        if batch_route_assignments:
            self.generator.set_batch_route_assignments(batch_route_assignments)
            route_counts: dict[str, int] = {}
            for route_id in batch_route_assignments.values():
                route_counts[route_id] = route_counts.get(route_id, 0) + 1
            distribution = ", ".join(
                f"{route_id}={count}" for route_id, count in sorted(route_counts.items())
            )
            _announce(
                f"creating {len(planned_organisms)} organism(s) for state_kind={state_kind} "
                f"(generation={self.generation}, concurrency={self.max_parallel_organisms}, "
                f"route_distribution=[{distribution}])"
            )
        else:
            _announce(
                f"creating {len(planned_organisms)} organism(s) for state_kind={state_kind} "
                f"(generation={self.generation}, concurrency={self.max_parallel_organisms})"
            )

        persist_state = self._persist_inflight_generation if state_kind == "generation" else self._persist_inflight_seed
        persist_state(planned_organisms)
        active_by_id = {organism.organism_id: organism for organism in self.population}
        plans_by_id = {plan.organism_id: plan for plan in planned_organisms}
        created_offspring: dict[str, OrganismMeta] = {}
        semaphore = asyncio.Semaphore(self.max_parallel_organisms)

        async def _run_creation(plan: PlannedOrganismCreation) -> tuple[str, OrganismMeta | None]:
            async with semaphore:
                plan.pipeline_state = "creating"
                self._write_planned_organism_stub(plan)
                persist_state(planned_organisms)
                self._record_creation_event(
                    plan,
                    event="started",
                    message=(
                        f"creating organism (route={plan.route}, island={plan.island_id}, "
                        f"generation={plan.generation})"
                    ),
                )
                _announce(
                    f"creating organism {plan.organism_id} "
                    f"(route={plan.route}, island={plan.island_id}, generation={plan.generation})"
                )
                try:
                    organism = await asyncio.to_thread(self._materialize_planned_organism, plan, active_by_id)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception(
                        "Organism creation failed for %s (generation=%d, island=%s, route=%s)",
                        plan.organism_id,
                        plan.generation,
                        plan.island_id,
                        plan.route,
                    )
                    self._record_creation_event(
                        plan,
                        event="failed",
                        message=f"{type(exc).__name__}: {exc}",
                        include_traceback=True,
                    )
                    _announce(
                        f"organism {plan.organism_id} FAILED creation "
                        f"(island={plan.island_id}, route={plan.route}): "
                        f"{type(exc).__name__}: {exc}"
                    )
                    plan.pipeline_state = "failed_creation"
                    plan.error_msg = str(exc)
                    self._write_planned_organism_stub(plan)
                    persist_state(planned_organisms)
                    return plan.organism_id, None
                self._record_creation_event(
                    plan,
                    event="completed",
                    message="creation stages finished",
                )
                _announce(
                    f"organism {plan.organism_id} creation stages finished "
                    f"(island={plan.island_id}, route={plan.route})"
                )

                plan.pipeline_state = "pending_simple_eval"
                plan.error_msg = None
                persist_state(planned_organisms)
                return plan.organism_id, organism

        creation_tasks: set[asyncio.Task[tuple[str, OrganismMeta | None]]] = set()
        for plan in planned_organisms:
            if plan.pipeline_state in {"planned_creation", "creating"}:
                creation_tasks.add(asyncio.create_task(_run_creation(plan)))
                continue
            if plan.pipeline_state == "failed_creation":
                continue

            organism = read_organism_meta(plan.organism_dir)
            created_offspring[plan.organism_id] = organism
            if plan.pipeline_state in {"pending_simple_eval", "running_simple_eval"}:
                immediate_summary = self._submit_or_resume_simple_evaluation(plan, organism)
                persist_state(planned_organisms)
                if immediate_summary is not None:
                    self._announce_phase_summary(organism.organism_id, immediate_summary)
                    self._apply_completed_summary(plan, organism, immediate_summary)
                    persist_state(planned_organisms)

        try:
            while creation_tasks or self.orchestrator.has_pending_requests:
                if creation_tasks:
                    done, pending = await asyncio.wait(
                        creation_tasks,
                        timeout=0.05,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    creation_tasks = set(pending)
                    for task in done:
                        organism_id, organism = await task
                        if organism is None:
                            continue
                        created_offspring[organism_id] = organism
                        immediate_summary = self._submit_or_resume_simple_evaluation(
                            plans_by_id[organism_id],
                            organism,
                        )
                        persist_state(planned_organisms)
                        if immediate_summary is not None:
                            self._announce_phase_summary(organism_id, immediate_summary)
                            self._apply_completed_summary(plans_by_id[organism_id], organism, immediate_summary)
                            persist_state(planned_organisms)

                event = await self.orchestrator.poll_result(timeout=0.05)
                if event is not None:
                    result, payload, summary = event
                    plan = plans_by_id[result.organism_id]
                    if summary is not None:
                        organism = created_offspring[result.organism_id]
                        self._announce_phase_summary(result.organism_id, summary)
                        self._apply_completed_summary(plan, organism, summary)
                    persist_state(planned_organisms)

                if event is None:
                    await asyncio.sleep(0.01)
        finally:
            self.orchestrator.close()
            self.generator.clear_batch_route_assignments()

        return list(created_offspring.values())

    async def _produce_offspring(
        self,
        active_by_island: dict[str, list[OrganismMeta]],
    ) -> list[OrganismMeta]:
        """Compatibility shim for tests and narrow callers."""

        return await self._execute_planned_creations(
            self._plan_offspring_generation(active_by_island),
            state_kind="generation",
        )

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
        summaries = await self.orchestrator.evaluate_organisms(requests)
        summary_by_id = {summary.organism_id: summary for summary in summaries}

        for organism in organisms:
            summary = summary_by_id.get(organism.organism_id)
            if summary is None:
                continue

            if phase == "simple":
                organism.simple_score = summary.aggregate_score
                organism.pipeline_state = self._simple_summary_status_to_pipeline_state(summary.status)
            elif phase == "hard":
                organism.hard_score = summary.aggregate_score
            else:
                raise ValueError(f"Unsupported phase '{phase}'")

            organism.status = "evaluated" if summary.status in {"ok", "partial"} else "pending"
            update_latest_lineage_entry(
                organism,
                phase=phase,
                phase_score=summary.aggregate_score,
                selected_experiments=summary.selected_experiments,
            )
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

        experiment_index = dict(organism.experiment_report_index)
        phase_index = dict(experiment_index.get(summary.phase, {}))
        for exp_name, exp_payload in summary.per_experiment.items():
            raw_report = exp_payload.get("raw_report", exp_payload) if isinstance(exp_payload, dict) else exp_payload
            if not isinstance(raw_report, dict):
                raw_report = {}
            phase_index[exp_name] = {
                "path": str(phase_result_path(organism.organism_dir, summary.phase, exp_name)),
                "status": raw_report.get("status", exp_payload.get("status") if isinstance(exp_payload, dict) else None),
                "score": raw_report.get("score", exp_payload.get("score") if isinstance(exp_payload, dict) else None),
            }
        experiment_index[summary.phase] = phase_index
        organism.experiment_report_index = experiment_index

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
            "simple_score": organism.simple_score,
            "hard_score": organism.hard_score,
            "experiment_report_index": experiment_index,
            "phase_results": phase_results,
        }
        write_organism_summary(organism.organism_dir, payload)
        write_organism_meta(organism)

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

    def _restore_planned_organisms(
        self,
        inflight_payload: dict[str, Any],
        *,
        planned_key: str,
        state_name: str,
    ) -> list[PlannedOrganismCreation]:
        planned_payload = inflight_payload.get(planned_key, [])
        if not isinstance(planned_payload, list):
            raise ValueError(f"population_state.json {state_name}.{planned_key} must be a list")
        return [PlannedOrganismCreation.from_dict(dict(entry)) for entry in planned_payload]

    def _restore_inflight_generation(self, inflight_generation: dict[str, Any]) -> list[PlannedOrganismCreation]:
        return self._restore_planned_organisms(
            inflight_generation,
            planned_key="planned_offspring",
            state_name="inflight_generation",
        )

    def _restore_inflight_seed(self, inflight_seed: dict[str, Any]) -> list[PlannedOrganismCreation]:
        return self._restore_planned_organisms(
            inflight_seed,
            planned_key="planned_organisms",
            state_name="inflight_seed",
        )

    async def _run_generation(
        self,
        generation: int,
        *,
        planned_offspring: list[PlannedOrganismCreation] | None = None,
    ) -> None:
        self.generation = generation
        _announce(
            f"starting sampling for generation {generation} "
            f"(active_population={len(self.population)}, "
            f"offspring_per_generation={self.offspring_per_generation}, "
            f"max_parallel_organisms={self.max_parallel_organisms})"
        )
        if planned_offspring is None:
            active_by_island = self._group_by_island(self.population)
            planned_offspring = self._plan_offspring_generation(active_by_island)

        offspring = await self._execute_planned_creations(planned_offspring, state_kind="generation")
        candidate_pool = list(self.population) + offspring
        simple_survivors = select_top_k_per_island(
            candidate_pool,
            self.max_organisms_per_island,
            score_field="simple_score",
        )
        self._mark_eliminated(candidate_pool, simple_survivors)

        if self._should_run_great_filter(generation):
            await self._evaluate_phase(simple_survivors, phase="hard")
            self.population = select_top_h_per_island(
                simple_survivors,
                self._great_filter_top_h(),
                score_field="hard_score",
            )
            self._mark_eliminated(simple_survivors, self.population)
        else:
            for organism in simple_survivors:
                organism.current_generation_active = self.generation
                write_organism_meta(organism)
            self.population = simple_survivors

        self._save_state(finalized_generation=generation, inflight_seed=None, inflight_generation=None)
        self._render_progress_snapshot()

    async def seed_population(self) -> dict[str, Any]:
        if self._owns_llm_registry:
            self.llm_registry.start()

        try:
            self.generation = 0
            _announce(
                "starting seed sampling for generation 0 "
                f"(islands={len(self.islands)}, seed_per_island={self.seed_organisms_per_island}, "
                f"max_parallel_organisms={self.max_parallel_organisms})"
            )
            state = self._load_state()
            if state is not None:
                if isinstance(state.get("inflight_generation"), dict):
                    raise RuntimeError(
                        "population_state.json contains inflight_generation; continue it with run_evolution, not seed_population."
                    )
                inflight_seed = state.get("inflight_seed")
                if isinstance(inflight_seed, dict):
                    planned_organisms = self._restore_inflight_seed(inflight_seed)
                else:
                    raise RuntimeError(
                        "Population is already initialized. Refusing to reseed an existing population root."
                    )
            else:
                planned_organisms = self._plan_seed_population()

            newborns = await self._execute_planned_creations(planned_organisms, state_kind="seed")
            self.population = select_top_k_per_island(
                newborns,
                self.max_organisms_per_island,
                score_field="simple_score",
            )
            if planned_organisms and not self.population:
                failure_message = self._seed_failure_message(planned_organisms, newborns)
                _announce(failure_message)
                self._persist_inflight_seed(planned_organisms)
                raise RuntimeError(failure_message)
            self._mark_eliminated(newborns, self.population)
            for organism in self.population:
                organism.current_generation_active = 0
                write_organism_meta(organism)
            self._save_state(finalized_generation=0, inflight_seed=None, inflight_generation=None)
            self._render_progress_snapshot()

            best = self._best_active_organism()
            return {
                "total_generations": 0,
                "active_population_size": len(self.population),
                "best_organism_id": best.organism_id if best is not None else None,
                "best_simple_score": best.simple_score if best is not None else None,
                "best_implementation_path": best.implementation_path if best is not None else None,
            }
        finally:
            self.orchestrator.close()
            if self._owns_llm_registry:
                self.llm_registry.stop()

    async def run(self) -> dict[str, Any]:
        max_generations = int(self._require_cfg_value("evolver.max_generations"))
        if self._owns_llm_registry:
            self.llm_registry.start()

        try:
            state = self._load_state()
            if state is None:
                raise FileNotFoundError(
                    "population_state.json is required for seeded-only evolution. "
                    "Run scripts/seed_population.sh first."
                )

            inflight_seed = state.get("inflight_seed")
            if isinstance(inflight_seed, dict):
                raise RuntimeError(
                    "population_state.json contains inflight_seed. "
                    "Continue seeding with scripts/seed_population.sh before running evolution."
                )

            finalized_generation = int(state.get("current_generation", 0))
            self.population = self._restore_population_from_state(finalized_generation)
            if not self.population:
                raise RuntimeError(
                    "population_state.json contains no active organisms. "
                    "Run scripts/seed_population.sh to initialize the population first."
                )
            self.generation = finalized_generation

            inflight_generation = state.get("inflight_generation")
            if isinstance(inflight_generation, dict) and bool(self.evolver_cfg.get("resume", True)):
                target_generation = int(inflight_generation.get("target_generation", finalized_generation + 1))
                planned_offspring = self._restore_inflight_generation(inflight_generation)
                await self._run_generation(target_generation, planned_offspring=planned_offspring)

            for generation in range(self.generation + 1, max_generations + 1):
                await self._run_generation(generation)

            best = self._best_active_organism()
            return {
                "total_generations": self.generation,
                "active_population_size": len(self.population),
                "best_organism_id": best.organism_id if best is not None else None,
                "best_simple_score": best.simple_score if best is not None else None,
                "best_implementation_path": best.implementation_path if best is not None else None,
            }
        finally:
            self.orchestrator.close()
            if self._owns_llm_registry:
                self.llm_registry.stop()
