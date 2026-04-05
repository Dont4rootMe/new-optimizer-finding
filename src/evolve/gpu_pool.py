"""Resource-aware subprocess pool for evaluation tasks."""

from __future__ import annotations

import asyncio
from collections import deque
import multiprocessing as mp
import os
import queue
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evolve.storage import ensure_dir
from src.evolve.types import EvalTask, EvalTaskResult


def _kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    """Terminate subprocess and children when timeout occurs."""

    try:
        import psutil
    except ImportError:
        try:
            proc.kill()
        except Exception:
            pass
        return

    try:
        parent = psutil.Process(proc.pid)
    except psutil.Error:
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.Error:
            pass

    _, alive = psutil.wait_procs(children, timeout=3)
    for child in alive:
        try:
            child.kill()
        except psutil.Error:
            pass

    try:
        parent.terminate()
        parent.wait(3)
    except psutil.Error:
        try:
            parent.kill()
        except psutil.Error:
            pass


def _run_eval_subprocess(
    task: EvalTask,
    *,
    resource_class: str,
    assigned_rank: int | None,
    worker_slot_id: str,
) -> EvalTaskResult:
    """Execute one evaluation command with retry and timeout."""

    env = dict(os.environ)
    if resource_class == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif resource_class == "gpu":
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        raise ValueError(f"Unsupported resource class '{resource_class}'.")

    out_path = Path(task.stdout_path)
    err_path = Path(task.stderr_path)
    ensure_dir(out_path.parent)
    ensure_dir(err_path.parent)
    ensure_dir(Path(task.output_json_path).parent)

    command = [
        sys.executable,
        "-m",
        task.entrypoint_module,
        "--experiment",
        task.experiment_name,
        "--organism_dir",
        task.organism_dir,
        "--output_json",
        task.output_json_path,
        "--seed",
        str(task.seed),
        "--device",
        task.device,
        "--precision",
        task.precision,
        "--mode",
        task.mode,
        "--config_path",
        task.config_path,
    ]

    started_at = time.perf_counter()
    status = "failed"
    error_msg: str | None = None
    return_code: int | None = None

    max_attempts = max(1, int(task.max_retries) + 1)
    attempt_count = 0

    for attempt in range(1, max_attempts + 1):
        attempt_count = attempt
        with out_path.open("a", encoding="utf-8") as out_handle, err_path.open("a", encoding="utf-8") as err_handle:
            out_handle.write(
                f"\n=== attempt {attempt}/{max_attempts} task={task.task_id} "
                f"resource={resource_class} device={task.device} ===\n"
            )
            err_handle.write(
                f"\n=== attempt {attempt}/{max_attempts} task={task.task_id} "
                f"resource={resource_class} device={task.device} ===\n"
            )

            proc = subprocess.Popen(
                command,
                stdout=out_handle,
                stderr=err_handle,
                env=env,
                cwd=task.workdir,
                start_new_session=True,
            )

            try:
                return_code = proc.wait(timeout=float(task.timeout_sec))
            except subprocess.TimeoutExpired:
                status = "timeout"
                error_msg = f"timeout after {task.timeout_sec}s"
                _kill_process_tree(proc)
                return_code = None
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                error_msg = str(exc)
                _kill_process_tree(proc)
                return_code = None
            else:
                if return_code == 0:
                    status = "ok"
                    error_msg = None
                    break
                status = "failed"
                error_msg = f"subprocess exited with code {return_code}"

        if status == "ok":
            break

    elapsed = time.perf_counter() - started_at
    return EvalTaskResult(
        task_id=task.task_id,
        organism_id=task.organism_id,
        generation=task.generation,
        phase=task.phase,
        experiment_name=task.experiment_name,
        status=status,
        result_json_path=task.output_json_path,
        duration_sec=float(elapsed),
        attempts=attempt_count,
        resource_class=resource_class,
        assigned_device=task.device,
        assigned_rank=assigned_rank,
        worker_slot_id=worker_slot_id,
        return_code=return_code,
        error_msg=error_msg,
    )


def _worker_loop(
    resource_class: str,
    assigned_rank: int | None,
    worker_slot_id: str,
    task_queue: mp.Queue[Any],
    result_queue: mp.Queue[Any],
) -> None:
    """Worker process loop pinned to one abstract evaluation resource."""

    while True:
        task_payload = task_queue.get()
        if task_payload is None:
            return

        task = EvalTask(**task_payload)
        result = _run_eval_subprocess(
            task,
            resource_class=resource_class,
            assigned_rank=assigned_rank,
            worker_slot_id=worker_slot_id,
        )
        result_queue.put(result.to_dict())


@dataclass(slots=True)
class _WorkerHandle:
    worker_slot_id: str
    resource_class: str
    assigned_rank: int | None
    assigned_device: str
    task_queue: mp.Queue[Any]
    process: mp.Process


class GpuJobPool:
    """Process-based evaluation pool with explicit GPU and CPU worker classes."""

    def __init__(self, *, gpu_ranks: list[int], cpu_parallel_jobs: int) -> None:
        self.gpu_ranks = list(gpu_ranks)
        self.cpu_parallel_jobs = max(0, int(cpu_parallel_jobs))
        if not self.gpu_ranks and self.cpu_parallel_jobs <= 0:
            raise ValueError("GpuJobPool requires at least one GPU rank or one CPU worker.")

        self._ctx = mp.get_context("spawn")
        self._result_queue: mp.Queue[Any] = self._ctx.Queue()
        self._workers: dict[str, _WorkerHandle] = {}
        self._available: dict[str, deque[str]] = {"gpu": deque(), "cpu": deque()}
        self._pending: dict[str, deque[dict[str, Any]]] = {"gpu": deque(), "cpu": deque()}
        self._started = False

    def _spawn_worker(self, *, resource_class: str, assigned_rank: int | None, worker_slot_id: str) -> None:
        task_queue: mp.Queue[Any] = self._ctx.Queue()
        assigned_device = "cpu" if resource_class == "cpu" else f"cuda:{assigned_rank}"
        proc = self._ctx.Process(
            target=_worker_loop,
            args=(resource_class, assigned_rank, worker_slot_id, task_queue, self._result_queue),
            daemon=True,
        )
        proc.start()
        self._workers[worker_slot_id] = _WorkerHandle(
            worker_slot_id=worker_slot_id,
            resource_class=resource_class,
            assigned_rank=assigned_rank,
            assigned_device=assigned_device,
            task_queue=task_queue,
            process=proc,
        )
        self._available[resource_class].append(worker_slot_id)

    def start(self) -> None:
        """Start worker processes."""

        if self._started:
            return

        for gpu_rank in self.gpu_ranks:
            self._spawn_worker(
                resource_class="gpu",
                assigned_rank=int(gpu_rank),
                worker_slot_id=f"gpu:{int(gpu_rank)}",
            )
        for cpu_slot in range(self.cpu_parallel_jobs):
            self._spawn_worker(
                resource_class="cpu",
                assigned_rank=None,
                worker_slot_id=f"cpu:{cpu_slot}",
            )

        self._started = True

    def _dispatch_payload(self, payload: dict[str, Any]) -> None:
        resource_class = str(payload["resource_class"])
        available = self._available[resource_class]
        if not available:
            self._pending[resource_class].append(payload)
            return

        worker_slot_id = available.popleft()
        worker = self._workers[worker_slot_id]
        task_payload = dict(payload)
        task_payload["device"] = worker.assigned_device
        task_payload["assigned_rank"] = worker.assigned_rank
        worker.task_queue.put(task_payload)

    def submit(self, task: EvalTask) -> None:
        """Submit evaluation task to the matching resource queue."""

        if not self._started:
            raise RuntimeError("GpuJobPool.start() must be called before submit().")
        if task.resource_class not in {"gpu", "cpu"}:
            raise ValueError(f"Unsupported task resource class '{task.resource_class}'.")
        self._dispatch_payload(task.to_dict())

    def _assign_next_pending(self, resource_class: str, worker_slot_id: str) -> None:
        pending = self._pending[resource_class]
        if pending:
            worker = self._workers[worker_slot_id]
            payload = pending.popleft()
            payload["device"] = worker.assigned_device
            payload["assigned_rank"] = worker.assigned_rank
            worker.task_queue.put(payload)
            return
        self._available[resource_class].append(worker_slot_id)

    def _get_result_blocking(self, timeout: float) -> dict[str, Any] | None:
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    async def get_result(self, timeout: float = 0.2) -> EvalTaskResult | None:
        """Non-blocking async poll for one task result."""

        payload = await asyncio.to_thread(self._get_result_blocking, timeout)
        if payload is None:
            return None
        result = EvalTaskResult(**payload)
        self._assign_next_pending(result.resource_class, result.worker_slot_id)
        return result

    def close(self) -> None:
        """Gracefully terminate all worker processes."""

        if not self._started:
            return

        for worker in self._workers.values():
            worker.task_queue.put(None)

        for worker in self._workers.values():
            worker.process.join(timeout=5)
            if worker.process.is_alive():
                try:
                    os.kill(worker.process.pid, signal.SIGTERM)
                except Exception:
                    pass
                worker.process.join(timeout=2)
                if worker.process.is_alive():
                    worker.process.kill()
                    worker.process.join(timeout=1)

        self._workers.clear()
        self._available = {"gpu": deque(), "cpu": deque()}
        self._pending = {"gpu": deque(), "cpu": deque()}
        self._started = False
