"""Canonical organism evaluation seam used by the organism-first evolution loop."""

from __future__ import annotations

import asyncio
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
    OrganismEvaluationRequest,
    OrganismEvaluationSummary,
)

LOGGER = logging.getLogger(__name__)


class EvolverOrchestrator:
    """Evaluate canonical organisms on request-scoped experiments and phase settings."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.evolver_cfg = cfg.evolver
        self.population_root = Path(str(cfg.paths.population_root)).expanduser().resolve()
        self.eval_config_dir = self._persist_eval_config_snapshot()

        self.gpu_ids = self._resolve_gpu_ids()
        self.pool = GpuJobPool(gpu_ids=self.gpu_ids)

    def _persist_eval_config_snapshot(self) -> Path:
        """Persist the active runtime config for subprocess evaluators."""

        config_dir = self.population_root / ".eval_config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"
        config_path.write_text(OmegaConf.to_yaml(self.cfg, resolve=True), encoding="utf-8")
        return config_dir

    def _resolve_gpu_ids(self) -> list[int]:
        configured = self.cfg.resources.get("gpu_ids")
        if configured:
            gpu_ids = [int(item) for item in configured]
        else:
            num = int(self.cfg.resources.num_gpus)
            gpu_ids = list(range(num))

        max_eval = self.evolver_cfg.get("max_evaluation_jobs")
        if max_eval is not None:
            gpu_ids = gpu_ids[: max(1, int(max_eval))]

        if not gpu_ids:
            raise ValueError("No GPU ids resolved from resources.num_gpus/resources.gpu_ids")

        return gpu_ids

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

    def _build_organism_task(
        self,
        request: OrganismEvaluationRequest,
        exp_name: str,
    ) -> EvalTask:
        organism_dir = Path(request.organism_dir)
        return EvalTask(
            task_id=uuid.uuid4().hex,
            organism_id=request.organism_id,
            generation=0,
            experiment_name=exp_name,
            organism_dir=str(organism_dir),
            output_json_path=str(phase_result_path(organism_dir, request.phase, exp_name)),
            stdout_path=str(phase_stdout_path(organism_dir, request.phase, exp_name)),
            stderr_path=str(phase_stderr_path(organism_dir, request.phase, exp_name)),
            seed=int(self.cfg.seed),
            device="cuda",
            precision=str(self.cfg.precision),
            mode=request.eval_mode,
            config_path=str(self.eval_config_dir),
            entrypoint_module=str(self.evolver_cfg.eval_entrypoint_module),
            timeout_sec=int(request.timeout_sec),
            max_retries=int(self.evolver_cfg.get("max_retries_per_eval", 0)),
            workdir=str(Path.cwd()),
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

    async def evaluate_organisms(
        self,
        requests: list[OrganismEvaluationRequest],
    ) -> list[OrganismEvaluationSummary]:
        """Evaluate canonical organisms and aggregate phase rewards honestly."""

        if not requests:
            return []

        request_states: dict[str, dict[str, Any]] = {}
        requested_experiments = sorted({exp for request in requests for exp in request.experiments})
        self._validate_requested_experiments(requested_experiments)

        total_tasks = 0
        completed_tasks = 0
        self.pool.start()
        try:
            for request in requests:
                allocation_snapshot = self._build_request_allocation(
                    request.organism_id,
                    experiments=request.experiments,
                    allocation_cfg=request.allocation_cfg,
                )
                selected_experiments = list(allocation_snapshot["selected_experiments"])
                request_states[request.organism_id] = {
                    "request": request,
                    "allocation": allocation_snapshot,
                    "selected_experiments": selected_experiments,
                    "results": {},
                }
                for exp_name in selected_experiments:
                    self.pool.submit(self._build_organism_task(request, exp_name))
                    total_tasks += 1

            while completed_tasks < total_tasks:
                result = await self.pool.get_result(timeout=0.2)
                if result is None:
                    await asyncio.sleep(0.05)
                    continue

                payload = self._load_or_build_payload(
                    result.result_json_path,
                    result.status,
                    result.error_msg,
                )
                state = request_states.get(result.organism_id)
                if state is not None:
                    state["results"][result.experiment_name] = payload
                completed_tasks += 1

            summaries: list[OrganismEvaluationSummary] = []
            for organism_id, state in request_states.items():
                request = state["request"]
                allocation_snapshot = state["allocation"]
                selected_experiments = state["selected_experiments"]
                aggregate_score, status, per_experiment = mean_score(
                    eval_results=state["results"],
                    selected_experiments=selected_experiments,
                    inclusion_prob=allocation_snapshot.get("inclusion_prob", {}),
                    total_experiments=len(request.experiments),
                )
                summaries.append(
                    OrganismEvaluationSummary(
                        organism_id=organism_id,
                        phase=request.phase,
                        aggregate_score=aggregate_score,
                        per_experiment=per_experiment,
                        selected_experiments=selected_experiments,
                        allocation_snapshot=allocation_snapshot,
                        status=status,
                        created_at=request.created_at,
                        eval_finished_at=utc_now_iso(),
                    )
                )
            return summaries
        finally:
            self.pool.close()
