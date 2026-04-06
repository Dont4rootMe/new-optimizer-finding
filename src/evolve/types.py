"""Typed dataclasses used by the evolution pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict


@dataclass(slots=True)
class Island:
    """Canonical research-island definition."""

    island_id: str
    name: str
    description_path: str
    description_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LineageEntry:
    """One score-bearing lineage entry in the organism history."""

    generation: int
    operator: str
    mother_id: str | None
    father_id: str | None
    change_description: str
    selected_simple_experiments: list[str] = field(default_factory=list)
    selected_hard_experiments: list[str] = field(default_factory=list)
    simple_score: float | None = None
    hard_score: float | None = None
    cross_island: bool = False
    father_island_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OrganismMeta:
    """Thin organism.json projection for the island-aware population."""

    organism_id: str
    island_id: str
    generation_created: int
    current_generation_active: int
    timestamp: str
    mother_id: str | None
    father_id: str | None
    operator: str  # "seed" | "mutation" | "crossover"
    genetic_code_path: str
    implementation_path: str
    lineage_path: str
    organism_dir: str
    ancestor_ids: list[str] = field(default_factory=list)
    experiment_report_index: dict[str, Any] = field(default_factory=dict)
    simple_score: float | None = None
    hard_score: float | None = None
    status: str = "pending"  # "pending" | "evaluated" | "eliminated" | "archived"
    llm_route_id: str = ""
    llm_provider: str = ""
    provider_model_id: str = ""
    model_name: str = ""  # Deprecated compatibility alias; new writes use provider_model_id.
    prompt_hash: str = ""
    seed: int = 0
    pipeline_state: str = ""
    planned_phase_evaluations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "organism_id": self.organism_id,
            "island_id": self.island_id,
            "generation_created": self.generation_created,
            "current_generation_active": self.current_generation_active,
            "timestamp": self.timestamp,
            "mother_id": self.mother_id,
            "father_id": self.father_id,
            "operator": self.operator,
            "genetic_code_path": self.genetic_code_path,
            "implementation_path": self.implementation_path,
            "lineage_path": self.lineage_path,
            "organism_dir": self.organism_dir,
            "ancestor_ids": list(self.ancestor_ids),
            "experiment_report_index": dict(self.experiment_report_index),
            "simple_score": self.simple_score,
            "hard_score": self.hard_score,
            "status": self.status,
            "llm_route_id": self.llm_route_id,
            "llm_provider": self.llm_provider,
            "provider_model_id": self.provider_model_id or self.model_name,
            "prompt_hash": self.prompt_hash,
            "seed": self.seed,
            "pipeline_state": self.pipeline_state,
            "planned_phase_evaluations": dict(self.planned_phase_evaluations),
        }

    @property
    def generation(self) -> int:
        """Backward-compatible alias for the generation when the organism was born."""

        return self.generation_created

    @property
    def parent_ids(self) -> list[str]:
        parents = [parent_id for parent_id in (self.mother_id, self.father_id) if parent_id]
        return parents


class ManifestEntry(TypedDict):
    organism_id: str
    island_id: str
    organism_dir: str
    generation_created: int
    current_generation_active: int
    simple_score: float | None
    hard_score: float | None


@dataclass(slots=True)
class EvalTask:
    """Single evaluation task mapped to one experiment and one organism directory."""

    task_id: str
    organism_id: str
    generation: int
    phase: str
    experiment_name: str
    organism_dir: str
    output_json_path: str
    stdout_path: str
    stderr_path: str
    seed: int
    device: str
    precision: str
    mode: str
    config_path: str
    entrypoint_module: str
    timeout_sec: int
    max_retries: int
    workdir: str
    resource_class: str
    assigned_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CreationStageResult:
    """Output of one two-stage organism creation request."""

    parsed_design: dict[str, str]
    implementation_code: str
    prompt_hash: str
    llm_route_id: str
    llm_provider: str
    provider_model_id: str


@dataclass(slots=True)
class EvalTaskResult:
    """Outcome of one subprocess evaluation task."""

    task_id: str
    organism_id: str
    generation: int
    phase: str
    experiment_name: str
    status: str
    result_json_path: str
    duration_sec: float
    attempts: int
    resource_class: str
    assigned_device: str
    assigned_rank: int | None = None
    worker_slot_id: str = ""
    return_code: int | None = None
    error_msg: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OrganismEvaluationRequest:
    """Public evaluation request emitted by the organism evolution loop."""

    organism_id: str
    organism_dir: str
    phase: str  # "simple" | "hard"
    experiments: list[str]
    allocation_cfg: dict[str, Any]
    eval_mode: str
    timeout_sec: int
    created_at: str


@dataclass(slots=True)
class OrganismEvaluationSummary:
    """Public evaluation summary returned to the organism evolution loop."""

    organism_id: str
    phase: str
    aggregate_score: float | None
    per_experiment: dict[str, dict[str, Any]]
    selected_experiments: list[str]
    allocation_snapshot: dict[str, Any]
    status: str
    created_at: str
    eval_finished_at: str
    error_msg: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlannedPhaseEvaluation:
    """Persistent phase plan attached to one organism before execution starts."""

    phase: str
    allocation_snapshot: dict[str, Any]
    selected_experiments: list[str]
    task_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    status: str = "planned"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlannedPhaseEvaluation":
        return cls(
            phase=str(payload["phase"]),
            allocation_snapshot=dict(payload.get("allocation_snapshot", {})),
            selected_experiments=[str(name) for name in payload.get("selected_experiments", [])],
            task_states={
                str(name): dict(state)
                for name, state in dict(payload.get("task_states", {})).items()
            },
            status=str(payload.get("status", "planned")),
        )


@dataclass(slots=True)
class PlannedOrganismCreation:
    """Persistent creation-plan entry for one not-yet-finalized organism."""

    organism_id: str
    organism_dir: str
    island_id: str
    generation: int
    route: str
    operator: str
    mother_id: str | None
    mother_organism_dir: str | None
    father_id: str | None
    father_organism_dir: str | None
    father_island_id: str | None
    operator_seed: int
    timestamp: str
    pipeline_state: str = "planned_creation"
    error_msg: str | None = None
    planned_phase_evaluations: dict[str, PlannedPhaseEvaluation] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["planned_phase_evaluations"] = {
            phase: plan.to_dict() for phase, plan in self.planned_phase_evaluations.items()
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlannedOrganismCreation":
        phase_payload = dict(payload.get("planned_phase_evaluations", {}))
        return cls(
            organism_id=str(payload["organism_id"]),
            organism_dir=str(payload["organism_dir"]),
            island_id=str(payload["island_id"]),
            generation=int(payload["generation"]),
            route=str(payload["route"]),
            operator=str(payload["operator"]),
            mother_id=(str(payload["mother_id"]) if payload.get("mother_id") else None),
            mother_organism_dir=(
                str(payload["mother_organism_dir"]) if payload.get("mother_organism_dir") else None
            ),
            father_id=(str(payload["father_id"]) if payload.get("father_id") else None),
            father_organism_dir=(
                str(payload["father_organism_dir"]) if payload.get("father_organism_dir") else None
            ),
            father_island_id=(str(payload["father_island_id"]) if payload.get("father_island_id") else None),
            operator_seed=int(payload["operator_seed"]),
            timestamp=str(payload["timestamp"]),
            pipeline_state=str(payload.get("pipeline_state", "planned_creation")),
            error_msg=(str(payload["error_msg"]) if payload.get("error_msg") else None),
            planned_phase_evaluations={
                str(phase): PlannedPhaseEvaluation.from_dict(dict(plan))
                for phase, plan in phase_payload.items()
            },
        )


# Backward-compatible alias while the runtime migrates from offspring-only planning
# to a shared seed/offspring creation pipeline.
PlannedOffspring = PlannedOrganismCreation
