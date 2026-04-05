"""Canonical organism evaluation seam used by the organism-first evolution loop."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Sequence
import logging
import uuid
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.evolve.allocation import build_allocation_snapshot
from src.evolve.gpu_pool import GpuJobPool
from src.evolve.scoring import mean_score
from src.evolve.storage import (
    phase_result_path,
    phase_stderr_path,
    phase_stdout_path,
    read_json,
    utc_now_iso,
    write_json,
)
from src.evolve.types import (
    EvalTask,
    EvalTaskResult,
    OrganismEvaluationRequest,
    OrganismEvaluationSummary,
    PlannedPhaseEvaluation,
)

LOGGER = logging.getLogger(__name__)


class EvolverOrchestrator:
    """Evaluate canonical organisms on request-scoped experiments and phase settings."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.evolver_cfg = cfg.evolver
        self.population_root = Path(str(cfg.paths.population_root)).expanduser().resolve()
        self.eval_config_dir = self._persist_eval_config_snapshot()
        self.experiment_requirements = self._resolve_experiment_requirements()
        self.gpu_ranks, self.cpu_parallel_jobs = self._resolve_evaluation_resources()
        self._validate_gpu_disjointness()

        self.pool = GpuJobPool(gpu_ranks=self.gpu_ranks, cpu_parallel_jobs=self.cpu_parallel_jobs)
        self._request_states: dict[str, dict[str, Any]] = {}
        self._started = False

    def _persist_eval_config_snapshot(self) -> Path:
        """Persist the active runtime config for subprocess evaluators."""

        config_dir = self.population_root / ".eval_config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"
        config_path.write_text(OmegaConf.to_yaml(self.cfg, resolve=True), encoding="utf-8")
        return config_dir

    def _normalize_gpu_ranks(self, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [int(value)]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [int(item) for item in value]
        raise TypeError(f"gpu_ranks must be null, int, or list[int], got {type(value).__name__}")

    def _resolve_evaluation_resources(self) -> tuple[list[int], int]:
        resources_cfg = self.cfg.get("resources", {})
        evaluation_cfg = resources_cfg.get("evaluation")
        if evaluation_cfg is None:
            raise ValueError("Canonical runtime config must define resources.evaluation")

        gpu_ranks = self._normalize_gpu_ranks(evaluation_cfg.get("gpu_ranks"))
        if len(set(gpu_ranks)) != len(gpu_ranks):
            raise ValueError("resources.evaluation.gpu_ranks must not contain duplicates")

        cpu_parallel_jobs = int(evaluation_cfg.get("cpu_parallel_jobs", 0))
        if cpu_parallel_jobs < 0:
            raise ValueError("resources.evaluation.cpu_parallel_jobs must be >= 0")
        if not gpu_ranks and cpu_parallel_jobs <= 0:
            raise ValueError("resources.evaluation must enable at least one GPU rank or one CPU worker")

        return gpu_ranks, cpu_parallel_jobs

    def _collect_api_platform_gpu_ranks(self) -> list[int]:
        route_gpu_ranks: list[int] = []
        for route_cfg in dict(self.cfg.get("api_platforms", {})).values():
            if not hasattr(route_cfg, "get"):
                continue
            route_gpu_ranks.extend(self._normalize_gpu_ranks(route_cfg.get("gpu_ranks")))
        return route_gpu_ranks

    def _validate_gpu_disjointness(self) -> None:
        eval_gpu_ranks = set(self.gpu_ranks)
        route_gpu_ranks = self._collect_api_platform_gpu_ranks()
        if len(set(route_gpu_ranks)) != len(route_gpu_ranks):
            raise ValueError("api_platform route gpu_ranks must not contain duplicates across enabled routes")
        overlap = sorted(eval_gpu_ranks.intersection(route_gpu_ranks))
        if overlap:
            raise ValueError(
                f"Evaluation gpu_ranks must not overlap api_platform gpu_ranks; overlapping ranks: {overlap}"
            )

    def _resolve_experiment_requirements(self) -> dict[str, bool]:
        requirements: dict[str, bool] = {}
        for exp_name, exp_cfg in dict(self.cfg.get("experiments", {})).items():
            if not hasattr(exp_cfg, "get"):
                continue
            if "need_cuda" not in exp_cfg:
                raise ValueError(f"Experiment '{exp_name}' must define top-level need_cuda: bool")
            need_cuda = bool(exp_cfg.get("need_cuda"))
            intended_device = str(exp_cfg.get("compute", {}).get("device", "")).strip()
            if need_cuda and not intended_device.startswith("cuda"):
                raise ValueError(
                    f"Experiment '{exp_name}' sets need_cuda=true but compute.device is '{intended_device}'"
                )
            if not need_cuda and intended_device.startswith("cuda"):
                raise ValueError(
                    f"Experiment '{exp_name}' sets need_cuda=false but compute.device is '{intended_device}'"
                )
            requirements[str(exp_name)] = need_cuda
        return requirements

    def _validate_requested_experiments(self, experiments: list[str]) -> None:
        config_experiments = set(self.cfg.get("experiments", {}).keys())
        missing = [name for name in experiments if name not in config_experiments]
        if missing:
            raise ValueError(f"Unknown requested experiment entries: {missing}")

    def _load_or_build_payload(self, result_path_str: str, status: str, error_msg: str | None) -> dict[str, Any]:
        out_path = Path(result_path_str)
        if out_path.exists():
            payload = read_json(out_path)
        else:
            payload = {
                "status": "failed",
                "error_msg": error_msg or "missing output json",
            }
            write_json(out_path, payload)

        if status != "ok":
            payload["status"] = status
            if not payload.get("error_msg"):
                payload["error_msg"] = error_msg
            write_json(out_path, payload)

        return payload

    def _resource_class_for_experiment(self, exp_name: str) -> str:
        need_cuda = bool(self.experiment_requirements.get(exp_name))
        if need_cuda:
            if not self.gpu_ranks:
                raise ValueError(
                    f"Experiment '{exp_name}' requires CUDA but resources.evaluation.gpu_ranks is empty"
                )
            return "gpu"
        if self.cpu_parallel_jobs <= 0:
            raise ValueError(
                f"Experiment '{exp_name}' is CPU-only but resources.evaluation.cpu_parallel_jobs is 0"
            )
        return "cpu"

    def _build_organism_task(
        self,
        request: OrganismEvaluationRequest,
        exp_name: str,
    ) -> EvalTask:
        organism_dir = Path(request.organism_dir)
        resource_class = self._resource_class_for_experiment(exp_name)
        return EvalTask(
            task_id=uuid.uuid4().hex,
            organism_id=request.organism_id,
            generation=0,
            phase=request.phase,
            experiment_name=exp_name,
            organism_dir=str(organism_dir),
            output_json_path=str(phase_result_path(organism_dir, request.phase, exp_name)),
            stdout_path=str(phase_stdout_path(organism_dir, request.phase, exp_name)),
            stderr_path=str(phase_stderr_path(organism_dir, request.phase, exp_name)),
            seed=int(self.cfg.seed),
            device="cuda" if resource_class == "gpu" else "cpu",
            precision=str(self.cfg.precision),
            mode=request.eval_mode,
            config_path=str(self.eval_config_dir),
            entrypoint_module=str(self.evolver_cfg.eval_entrypoint_module),
            timeout_sec=int(request.timeout_sec),
            max_retries=int(self.evolver_cfg.get("max_retries_per_eval", 0)),
            workdir=str(Path.cwd()),
            resource_class=resource_class,
        )

    def _build_request_allocation(
        self,
        organism_id: str,
        *,
        experiments: list[str],
        allocation_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        return build_allocation_snapshot(
            population_root=str(self.population_root),
            experiments=list(experiments),
            allocation_cfg=allocation_cfg,
            seed=int(self.cfg.seed),
            entity_id=organism_id,
        )

    def start(self) -> None:
        if self._started:
            return
        self.pool.start()
        self._started = True

    def close(self) -> None:
        self.pool.close()
        self._started = False
        self._request_states.clear()

    def plan_phase_evaluation(self, request: OrganismEvaluationRequest) -> PlannedPhaseEvaluation:
        self._validate_requested_experiments(list(request.experiments))
        allocation_snapshot = self._build_request_allocation(
            request.organism_id,
            experiments=request.experiments,
            allocation_cfg=request.allocation_cfg,
        )
        selected_experiments = list(allocation_snapshot["selected_experiments"])
        organism_dir = Path(request.organism_dir)
        task_states = {
            exp_name: {
                "status": "planned",
                "resource_class": self._resource_class_for_experiment(exp_name),
                "result_path": str(phase_result_path(organism_dir, request.phase, exp_name)),
                "stdout_path": str(phase_stdout_path(organism_dir, request.phase, exp_name)),
                "stderr_path": str(phase_stderr_path(organism_dir, request.phase, exp_name)),
                "assigned_device": None,
                "assigned_rank": None,
                "attempts": None,
                "error_msg": None,
            }
            for exp_name in selected_experiments
        }
        return PlannedPhaseEvaluation(
            phase=request.phase,
            allocation_snapshot=allocation_snapshot,
            selected_experiments=selected_experiments,
            task_states=task_states,
            status="planned",
        )

    def _request_key(self, organism_id: str, phase: str) -> str:
        return f"{organism_id}:{phase}"

    def submit_planned_request(
        self,
        request: OrganismEvaluationRequest,
        planned_evaluation: PlannedPhaseEvaluation,
        *,
        existing_results: dict[str, dict[str, Any]] | None = None,
    ) -> OrganismEvaluationSummary | None:
        self.start()
        request_key = self._request_key(request.organism_id, request.phase)
        if request_key in self._request_states:
            raise ValueError(f"Duplicate active evaluation request for {request_key}")

        selected_experiments = list(planned_evaluation.selected_experiments)
        completed_results = dict(existing_results or {})
        self._request_states[request_key] = {
            "request": request,
            "planned": planned_evaluation,
            "results": completed_results,
            "pending": len([name for name in selected_experiments if name not in completed_results]),
        }
        planned_evaluation.status = (
            "queued" if self._request_states[request_key]["pending"] > 0 else "completed"
        )

        for exp_name in selected_experiments:
            if exp_name in completed_results:
                continue
            task = self._build_organism_task(request, exp_name)
            self.pool.submit(task)
        if self._request_states[request_key]["pending"] <= 0:
            return self._finalize_summary(request_key)
        return None

    @property
    def has_pending_requests(self) -> bool:
        return bool(self._request_states)

    def _finalize_summary(self, request_key: str) -> OrganismEvaluationSummary:
        state = self._request_states.pop(request_key)
        request = state["request"]
        planned = state["planned"]
        aggregate_score, status, per_experiment = mean_score(
            eval_results=state["results"],
            selected_experiments=planned.selected_experiments,
            inclusion_prob=planned.allocation_snapshot.get("inclusion_prob", {}),
            total_experiments=len(request.experiments),
        )
        return OrganismEvaluationSummary(
            organism_id=request.organism_id,
            phase=request.phase,
            aggregate_score=aggregate_score,
            per_experiment=per_experiment,
            selected_experiments=list(planned.selected_experiments),
            allocation_snapshot=dict(planned.allocation_snapshot),
            status=status,
            created_at=request.created_at,
            eval_finished_at=utc_now_iso(),
        )

    async def poll_result(
        self,
        timeout: float = 0.2,
    ) -> tuple[EvalTaskResult, dict[str, Any], OrganismEvaluationSummary | None] | None:
        result = await self.pool.get_result(timeout=timeout)
        if result is None:
            return None

        payload = self._load_or_build_payload(
            result.result_json_path,
            result.status,
            result.error_msg,
        )
        request_key = self._request_key(result.organism_id, result.phase)
        state = self._request_states.get(request_key)
        if state is None:
            raise KeyError(f"Received evaluation result for unknown request '{request_key}'")

        state["results"][result.experiment_name] = payload
        state["pending"] -= 1
        summary = None
        if state["pending"] <= 0:
            summary = self._finalize_summary(request_key)

        return result, payload, summary

    async def evaluate_organisms(
        self,
        requests: list[OrganismEvaluationRequest],
    ) -> list[OrganismEvaluationSummary]:
        """Evaluate canonical organisms and aggregate phase rewards honestly."""

        if not requests:
            return []

        self.start()
        try:
            summaries: list[OrganismEvaluationSummary] = []
            for request in requests:
                immediate_summary = self.submit_planned_request(request, self.plan_phase_evaluation(request))
                if immediate_summary is not None:
                    summaries.append(immediate_summary)

            while len(summaries) < len(requests):
                event = await self.poll_result(timeout=0.2)
                if event is None:
                    await asyncio.sleep(0.05)
                    continue
                _, _, summary = event
                if summary is not None:
                    summaries.append(summary)
            return summaries
        finally:
            self.close()
