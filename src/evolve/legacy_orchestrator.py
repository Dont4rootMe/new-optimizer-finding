"""LEGACY candidate-first orchestration kept behind an explicit legacy entrypoint."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.evolve.allocation import build_allocation_snapshot
from src.evolve.gpu_pool import GpuJobPool
from src.evolve.legacy_generator import LegacyCandidateGenerator
from src.evolve.scoring import mean_score
from src.evolve.storage import (
    append_jsonl,
    candidate_dir,
    generation_dir,
    load_best_legacy_candidate_context,
    meta_path,
    read_json,
    result_path,
    selection_path,
    stderr_path,
    stdout_path,
    summary_path,
    utc_now_iso,
    write_json,
)
from src.evolve.types import (
    CandidateSummary,
    EvalTask,
    EvalTaskResult,
    GenerationSummary,
)
from optbench.utils.baselines import load_baseline_profile

LOGGER = logging.getLogger(__name__)


class LegacyCandidateOrchestrator:
    """LEGACY candidate-first generation and evaluation loop."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.evolver_cfg = cfg.evolver
        allocation_raw = cfg.evolver.get("allocation")
        if allocation_raw is None:
            simple_phase = cfg.evolver.get("phases", {}).get("simple")
            allocation_raw = simple_phase.get("allocation", {}) if simple_phase is not None else {}
        if isinstance(allocation_raw, DictConfig):
            allocation_obj = OmegaConf.to_container(allocation_raw, resolve=True)
            self.allocation_cfg = allocation_obj if isinstance(allocation_obj, dict) else {}
        elif isinstance(allocation_raw, dict):
            self.allocation_cfg = allocation_raw
        else:
            self.allocation_cfg = {}

        self.population_root = Path(str(cfg.paths.population_root)).expanduser().resolve()
        self.generation = int(self.evolver_cfg.generation)

        raw_expected_experiments = self.evolver_cfg.get("eval_experiments")
        if raw_expected_experiments is None:
            simple_phase = cfg.evolver.get("phases", {}).get("simple")
            raw_expected_experiments = simple_phase.get("experiments", []) if simple_phase is not None else []
        self.expected_experiments = [str(name) for name in raw_expected_experiments]

        self.experiment_cfgs: dict[str, dict[str, Any]] = {
            exp_name: OmegaConf.to_container(cfg.experiments[exp_name], resolve=True)
            for exp_name in self.expected_experiments
            if exp_name in cfg.experiments
        }
        self.baseline_profiles, self.baseline_errors = self._load_baseline_profiles()

        self.generator = LegacyCandidateGenerator(cfg)

        self.gen_dir = generation_dir(self.population_root, self.generation)
        self.index_file = self.gen_dir / "index.jsonl"

        self.candidate_states: dict[str, dict[str, Any]] = {}
        self.finished_summaries: list[dict[str, Any]] = []

        self.total_eval_submitted = 0
        self.total_eval_completed = 0
        self.dispatcher_done = False
        self.stop_requested = False

        self._validate_eval_experiments()

        self.gpu_ids = self._resolve_gpu_ids()
        self.pool = GpuJobPool(gpu_ids=self.gpu_ids)

    def _load_baseline_profiles_for(
        self,
        experiments: list[str],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        profiles: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        stats_root = self.cfg.paths.stats_root

        for exp_name in experiments:
            try:
                profiles[exp_name] = load_baseline_profile(stats_root, exp_name)
            except (FileNotFoundError, ValueError) as exc:
                errors[exp_name] = str(exc)

        return profiles, errors

    def _load_baseline_profiles(self) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        return self._load_baseline_profiles_for(self.expected_experiments)

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

    def _validate_eval_experiments(self) -> None:
        config_experiments = set(self.cfg.get("experiments", {}).keys())
        missing = [name for name in self.expected_experiments if name not in config_experiments]
        if missing:
            raise ValueError(f"Unknown legacy eval_experiments entries: {missing}")

        sample_size = int(self.allocation_cfg.get("sample_size", len(self.expected_experiments)))
        if sample_size < 1:
            raise ValueError("evolver.allocation.sample_size must be >= 1")
        if self.expected_experiments and sample_size > len(self.expected_experiments):
            raise ValueError(
                "evolver.allocation.sample_size must be <= len(legacy eval experiments)"
            )

    def _append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": utc_now_iso(),
            "generation": self.generation,
            "event": event_type,
            **payload,
        }
        append_jsonl(self.index_file, record)

    def _register_candidate(
        self,
        candidate_id: str,
        candidate_dir_path: Path,
        created_at: str,
        selected_experiments: list[str],
        allocation_snapshot: dict[str, Any],
    ) -> None:
        self.candidate_states[candidate_id] = {
            "candidate_dir": candidate_dir_path,
            "created_at": created_at,
            "selected_experiments": list(selected_experiments),
            "allocation": allocation_snapshot,
            "pending": set(selected_experiments),
            "results": {},
            "finalized": False,
        }

    def _build_task(self, candidate_id: str, exp_name: str) -> EvalTask:
        state = self.candidate_states[candidate_id]
        cand_dir_path = state["candidate_dir"]

        return EvalTask(
            task_id=uuid.uuid4().hex,
            candidate_id=candidate_id,
            generation=self.generation,
            experiment_name=exp_name,
            optimizer_path=str(cand_dir_path / "optimizer.py"),
            output_json_path=str(result_path(cand_dir_path, exp_name)),
            stdout_path=str(stdout_path(cand_dir_path, exp_name)),
            stderr_path=str(stderr_path(cand_dir_path, exp_name)),
            seed=int(self.cfg.seed),
            device="cuda",
            precision=str(self.cfg.precision),
            mode=str(self.evolver_cfg.get("eval_mode", "smoke")),
            config_path="conf",
            entrypoint_module=str(self.evolver_cfg.eval_entrypoint_module),
            timeout_sec=int(self.evolver_cfg.get("timeout_sec_per_eval", 7200)),
            max_retries=int(self.evolver_cfg.get("max_retries_per_eval", 0)),
            workdir=str(Path.cwd()),
        )

    def _build_candidate_allocation(self, candidate_id: str) -> dict[str, Any]:
        return build_allocation_snapshot(
            population_root=str(self.population_root),
            experiments=self.expected_experiments,
            allocation_cfg=self.allocation_cfg,
            seed=int(self.cfg.seed),
            entity_id=candidate_id,
            history_source="legacy_candidate",
        )

    def _experiments_to_resume(self, cand_dir: Path, selected_experiments: list[str]) -> list[str]:
        pending = []
        for exp_name in selected_experiments:
            exp_result_file = result_path(cand_dir, exp_name)
            if not exp_result_file.exists():
                pending.append(exp_name)
                continue

            try:
                payload = read_json(exp_result_file)
            except Exception:
                pending.append(exp_name)
                continue

            if str(payload.get("status", "failed")) != "ok":
                pending.append(exp_name)

        return pending

    async def _proposal_worker(
        self,
        worker_idx: int,
        proposal_queue: asyncio.Queue[Any],
        eval_queue: asyncio.Queue[Any],
    ) -> None:
        del worker_idx
        while True:
            item = await proposal_queue.get()
            if item is None:
                proposal_queue.task_done()
                return

            if self.stop_requested:
                proposal_queue.task_done()
                continue

            slot = int(item)
            candidate_id = uuid.uuid4().hex
            cand_dir = candidate_dir(self.gen_dir, candidate_id)
            created_at = utc_now_iso()

            self._append_event("candidate_slot_started", {"slot": slot, "candidate_id": candidate_id})

            try:
                context = load_best_legacy_candidate_context(self.population_root, limit=3)
                meta = self.generator.generate_candidate(
                    candidate_id=candidate_id,
                    generation=self.generation,
                    candidate_dir=cand_dir,
                    context=context,
                )

                allocation_snapshot = self._build_candidate_allocation(candidate_id)
                selected_experiments = list(allocation_snapshot["selected_experiments"])

                self._register_candidate(
                    candidate_id=candidate_id,
                    candidate_dir_path=cand_dir,
                    created_at=meta.timestamp,
                    selected_experiments=selected_experiments,
                    allocation_snapshot=allocation_snapshot,
                )
                write_json(selection_path(cand_dir), allocation_snapshot)

                self._append_event(
                    "generated",
                    {
                        **meta.to_dict(),
                        "selected_experiments": selected_experiments,
                    },
                )

                for exp_name in selected_experiments:
                    await eval_queue.put(self._build_task(candidate_id, exp_name))
            except Exception as exc:
                summary = CandidateSummary(
                    candidate_id=candidate_id,
                    generation=self.generation,
                    aggregate_score=None,
                    experiments={
                        exp_name: {
                            "raw_metric": None,
                            "metric_direction": str(
                                self.experiment_cfgs.get(exp_name, {}).get("primary_metric", {}).get("direction", "max")
                            ),
                            "quality_ratio": None,
                            "steps_ratio": None,
                            "exp_score": None,
                            "time_sec": None,
                            "steps": None,
                            "status": "failed",
                            "error_msg": f"generation_failed: {exc}",
                        }
                        for exp_name in self.expected_experiments
                    },
                    selected_experiments=list(self.expected_experiments),
                    allocation={
                        "method": "uniform",
                        "enabled": False,
                        "history_window": 0,
                        "sample_size": len(self.expected_experiments),
                        "weights": {exp_name: 1.0 / max(1.0, float(len(self.expected_experiments))) for exp_name in self.expected_experiments},
                        "inclusion_prob": {exp_name: 1.0 for exp_name in self.expected_experiments},
                        "stats": {},
                        "selected_experiments": list(self.expected_experiments),
                    },
                    status="failed",
                    created_at=created_at,
                    eval_finished_at=utc_now_iso(),
                    seed=int(self.cfg.seed),
                    error_msg=str(exc),
                )
                write_json(summary_path(cand_dir), summary.to_dict())
                self.finished_summaries.append(summary.to_dict())
                self._append_event("summary_written", summary.to_dict())

                if bool(self.evolver_cfg.fail_fast):
                    self.stop_requested = True

            proposal_queue.task_done()

    async def _eval_dispatcher(self, eval_queue: asyncio.Queue[Any]) -> None:
        while True:
            task = await eval_queue.get()
            if task is None:
                eval_queue.task_done()
                break

            assert isinstance(task, EvalTask)
            self.pool.submit(task)
            self.total_eval_submitted += 1
            self._append_event(
                "eval_submitted",
                {
                    "task_id": task.task_id,
                    "candidate_id": task.candidate_id,
                    "experiment": task.experiment_name,
                },
            )
            eval_queue.task_done()

        self.dispatcher_done = True

    def _load_or_build_payload(self, result: EvalTaskResult) -> dict[str, Any]:
        out_path = Path(result.result_json_path)
        if out_path.exists():
            payload = read_json(out_path)
        else:
            payload = {
                "status": "failed",
                "error_msg": result.error_msg or "missing output json",
            }
            write_json(out_path, payload)

        if result.status != "ok":
            payload["status"] = result.status
            if not payload.get("error_msg"):
                payload["error_msg"] = result.error_msg
            write_json(out_path, payload)

        return payload

    def _finalize_candidate_if_ready(self, candidate_id: str) -> None:
        state = self.candidate_states[candidate_id]
        if state["finalized"] or state["pending"]:
            return

        selected_experiments = list(state["selected_experiments"])
        allocation_snapshot = state["allocation"]
        aggregate_score, status, per_experiment = mean_score(
            eval_results=state["results"],
            selected_experiments=selected_experiments,
            experiment_cfgs=self.experiment_cfgs,
            baseline_profiles=self.baseline_profiles,
            baseline_errors=self.baseline_errors,
            inclusion_prob=allocation_snapshot.get("inclusion_prob", {}),
            total_experiments=len(self.expected_experiments),
            scoring_cfg=self.allocation_cfg,
        )

        summary = CandidateSummary(
            candidate_id=candidate_id,
            generation=self.generation,
            aggregate_score=aggregate_score,
            experiments=per_experiment,
            selected_experiments=selected_experiments,
            allocation=allocation_snapshot,
            status=status,
            created_at=state["created_at"],
            eval_finished_at=utc_now_iso(),
            seed=int(self.cfg.seed),
            error_msg=None,
        )

        write_json(summary_path(state["candidate_dir"]), summary.to_dict())
        self.finished_summaries.append(summary.to_dict())
        state["finalized"] = True
        self._append_event("summary_written", summary.to_dict())

        if bool(self.evolver_cfg.fail_fast) and status == "failed":
            self.stop_requested = True

    async def _result_collector(self) -> None:
        while True:
            result = await self.pool.get_result(timeout=0.2)
            if result is not None:
                self.total_eval_completed += 1
                payload = self._load_or_build_payload(result)

                state = self.candidate_states.get(result.candidate_id)
                if state is not None:
                    state["results"][result.experiment_name] = payload
                    state["pending"].discard(result.experiment_name)
                    self._finalize_candidate_if_ready(result.candidate_id)

                self._append_event("eval_done", result.to_dict())

            if self.dispatcher_done and self.total_eval_completed >= self.total_eval_submitted:
                break

            await asyncio.sleep(0.05)

    async def _enqueue_resume_candidates(self, eval_queue: asyncio.Queue[Any]) -> int:
        resumed = 0
        if not bool(self.evolver_cfg.resume) and not bool(self.evolver_cfg.force):
            return resumed

        for cand_dir in sorted(self.gen_dir.glob("cand_*")):
            meta_file = meta_path(cand_dir)
            if not meta_file.exists():
                continue

            meta_payload = read_json(meta_file)
            candidate_id = str(meta_payload["candidate_id"])

            selection_file = selection_path(cand_dir)
            if selection_file.exists():
                allocation_snapshot = read_json(selection_file)
            else:
                allocation_snapshot = self._build_candidate_allocation(candidate_id)

            selected_experiments = allocation_snapshot.get("selected_experiments")
            if not isinstance(selected_experiments, list) or not selected_experiments:
                selected_experiments = list(self.expected_experiments)
                allocation_snapshot["selected_experiments"] = selected_experiments

            summary_file = summary_path(cand_dir)
            if summary_file.exists() and not bool(self.evolver_cfg.force):
                pending_from_summary = self._experiments_to_resume(
                    cand_dir=cand_dir,
                    selected_experiments=[str(exp_name) for exp_name in selected_experiments],
                )
                if not pending_from_summary:
                    summary_payload = read_json(summary_file)
                    self.finished_summaries.append(summary_payload)
                    self._append_event("candidate_skipped_resume", {"candidate_id": candidate_id})
                    continue

            self._register_candidate(
                candidate_id=candidate_id,
                candidate_dir_path=cand_dir,
                created_at=str(meta_payload.get("timestamp", utc_now_iso())),
                selected_experiments=[str(exp_name) for exp_name in selected_experiments],
                allocation_snapshot=allocation_snapshot,
            )

            if bool(self.evolver_cfg.force):
                missing = list(self.candidate_states[candidate_id]["selected_experiments"])
            else:
                missing = self._experiments_to_resume(
                    cand_dir=cand_dir,
                    selected_experiments=list(self.candidate_states[candidate_id]["selected_experiments"]),
                )
            self.candidate_states[candidate_id]["pending"] = set(missing)

            if not missing:
                state = self.candidate_states[candidate_id]
                for exp_name in state["selected_experiments"]:
                    exp_result_file = result_path(cand_dir, exp_name)
                    if exp_result_file.exists():
                        state["results"][exp_name] = read_json(exp_result_file)
                self._finalize_candidate_if_ready(candidate_id)
                continue

            for exp_name in missing:
                await eval_queue.put(self._build_task(candidate_id, exp_name))

            resumed += 1

        return resumed

    async def run(self) -> dict[str, Any]:
        """LEGACY: execute one candidate-first generation run."""

        self._append_event(
            "generation_started",
            {
                "requested_candidates": int(self.evolver_cfg.num_candidates),
                "gpu_ids": self.gpu_ids,
                "expected_experiments": self.expected_experiments,
            },
        )

        self.pool.start()

        proposal_queue: asyncio.Queue[Any] = asyncio.Queue()
        eval_queue: asyncio.Queue[Any] = asyncio.Queue()

        resumed_candidates = await self._enqueue_resume_candidates(eval_queue)
        existing_candidate_dirs = list(self.gen_dir.glob("cand_*"))
        to_generate = max(0, int(self.evolver_cfg.num_candidates) - len(existing_candidate_dirs))

        for idx in range(to_generate):
            await proposal_queue.put(idx)

        dispatcher_task = asyncio.create_task(self._eval_dispatcher(eval_queue))
        collector_task = asyncio.create_task(self._result_collector())

        proposal_workers = [
            asyncio.create_task(self._proposal_worker(worker_idx=i, proposal_queue=proposal_queue, eval_queue=eval_queue))
            for i in range(max(1, int(self.evolver_cfg.max_proposal_jobs)))
        ]

        await proposal_queue.join()
        for _ in proposal_workers:
            await proposal_queue.put(None)
        await asyncio.gather(*proposal_workers)

        await eval_queue.join()
        await eval_queue.put(None)
        await dispatcher_task
        await collector_task

        summary = GenerationSummary(
            generation=self.generation,
            requested_candidates=int(self.evolver_cfg.num_candidates),
            generated_candidates=max(0, len(self.finished_summaries) - resumed_candidates),
            completed_candidates=len(self.finished_summaries),
            ok_candidates=sum(1 for item in self.finished_summaries if item.get("status") == "ok"),
            partial_candidates=sum(1 for item in self.finished_summaries if item.get("status") == "partial"),
            failed_candidates=sum(1 for item in self.finished_summaries if item.get("status") == "failed"),
            output_dir=str(self.gen_dir),
            candidate_summaries=list(self.finished_summaries),
        )
        return summary.to_dict()
