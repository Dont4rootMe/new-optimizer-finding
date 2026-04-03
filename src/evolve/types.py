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
    """One score-bearing lineage entry in the maternal organism history."""

    generation: int
    operator: str
    mother_id: str | None
    father_id: str | None
    change_description: str
    gene_diff_summary: str
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
    """Full metadata for an organism in the island-aware population."""

    organism_id: str
    island_id: str
    generation_created: int
    current_generation_active: int
    timestamp: str
    mother_id: str | None
    father_id: str | None
    operator: str  # "seed" | "mutation" | "crossover"
    genetic_code_path: str
    optimizer_path: str
    lineage_path: str
    organism_dir: str
    simple_reward: float | None = None
    hard_reward: float | None = None
    selection_reward: float | None = None
    status: str = "pending"  # "pending" | "evaluated" | "eliminated" | "archived"
    model_name: str = ""
    prompt_hash: str = ""
    seed: int = 0
    genetic_code: dict[str, Any] = field(default_factory=dict, repr=False)
    lineage: list[dict[str, Any]] = field(default_factory=list, repr=False)

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
            "optimizer_path": self.optimizer_path,
            "lineage_path": self.lineage_path,
            "organism_dir": self.organism_dir,
            "simple_reward": self.simple_reward,
            "hard_reward": self.hard_reward,
            "selection_reward": self.selection_reward,
            "status": self.status,
            "model_name": self.model_name,
            "prompt_hash": self.prompt_hash,
            "seed": self.seed,
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
    simple_reward: float | None
    hard_reward: float | None
    selection_reward: float | None


@dataclass(slots=True)
class EvalTask:
    """Single evaluation task mapped to one experiment and one candidate."""

    task_id: str
    candidate_id: str
    generation: int
    experiment_name: str
    optimizer_path: str
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalTaskResult:
    """Outcome of one subprocess evaluation task."""

    task_id: str
    candidate_id: str
    generation: int
    experiment_name: str
    status: str
    result_json_path: str
    duration_sec: float
    attempts: int
    worker_gpu: int
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
