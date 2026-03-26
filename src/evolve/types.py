"""Typed dataclasses used by the evolution pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CandidateMeta:
    """Metadata for a generated optimizer candidate."""

    candidate_id: str
    generation: int
    timestamp: str
    model_name: str
    prompt_hash: str
    seed: int
    candidate_dir: str
    optimizer_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvolutionEntry:
    """One entry in an organism's evolution log."""

    generation: int
    change_description: str
    score: float | None
    parent_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OrganismMeta:
    """Full metadata for an organism in the evolutionary population."""

    organism_id: str
    generation: int
    timestamp: str
    parent_ids: list[str]
    operator: str  # "seed" | "mutation" | "crossover"
    idea_dna: list[str]
    evolution_log: list[dict[str, Any]]
    model_name: str
    prompt_hash: str
    seed: int
    organism_dir: str
    optimizer_path: str
    score: float | None = None
    simple_score: float | None = None
    hard_score: float | None = None
    status: str = "pending"  # "pending" | "evaluated" | "eliminated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
class CandidateSummary:
    """Aggregated summary across all experiment evaluations for one candidate."""

    candidate_id: str
    generation: int
    aggregate_score: float | None
    experiments: dict[str, dict[str, Any]]
    selected_experiments: list[str]
    allocation: dict[str, Any]
    status: str
    created_at: str
    eval_finished_at: str
    seed: int
    error_msg: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenerationSummary:
    """High-level generation run summary."""

    generation: int
    requested_candidates: int
    generated_candidates: int
    completed_candidates: int
    ok_candidates: int
    partial_candidates: int
    failed_candidates: int
    output_dir: str
    candidate_summaries: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
