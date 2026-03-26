"""GPU-bound subprocess pool for one-task-per-GPU evaluation."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import queue
import signal
import subprocess
import sys
import time
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


def _run_eval_subprocess(task: EvalTask, gpu_id: int) -> EvalTaskResult:
    """Execute one evaluation command with retry and timeout."""

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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
        "--optimizer_path",
        task.optimizer_path,
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
            out_handle.write(f"\n=== attempt {attempt}/{max_attempts} task={task.task_id} gpu={gpu_id} ===\n")
            err_handle.write(f"\n=== attempt {attempt}/{max_attempts} task={task.task_id} gpu={gpu_id} ===\n")

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
        candidate_id=task.candidate_id,
        generation=task.generation,
        experiment_name=task.experiment_name,
        status=status,
        result_json_path=task.output_json_path,
        duration_sec=float(elapsed),
        attempts=attempt_count,
        worker_gpu=gpu_id,
        return_code=return_code,
        error_msg=error_msg,
    )


def _worker_loop(
    gpu_id: int,
    task_queue: mp.Queue[Any],
    result_queue: mp.Queue[Any],
) -> None:
    """Worker process loop pinned to one GPU id."""

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    while True:
        task_payload = task_queue.get()
        if task_payload is None:
            return

        task = EvalTask(**task_payload)
        result = _run_eval_subprocess(task, gpu_id)
        result_queue.put(result.to_dict())


class GpuJobPool:
    """Process-based pool with one worker process per GPU."""

    def __init__(self, gpu_ids: list[int]) -> None:
        if not gpu_ids:
            raise ValueError("GpuJobPool requires at least one gpu_id.")

        self.gpu_ids = gpu_ids
        self._ctx = mp.get_context("spawn")
        self._task_queue: mp.Queue[Any] = self._ctx.Queue()
        self._result_queue: mp.Queue[Any] = self._ctx.Queue()
        self._workers: list[mp.Process] = []
        self._started = False

    def start(self) -> None:
        """Start worker processes."""

        if self._started:
            return

        for gpu_id in self.gpu_ids:
            proc = self._ctx.Process(
                target=_worker_loop,
                args=(gpu_id, self._task_queue, self._result_queue),
                daemon=True,
            )
            proc.start()
            self._workers.append(proc)

        self._started = True

    def submit(self, task: EvalTask) -> None:
        """Submit evaluation task to the worker queue."""

        if not self._started:
            raise RuntimeError("GpuJobPool.start() must be called before submit().")
        self._task_queue.put(task.to_dict())

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
        return EvalTaskResult(**payload)

    def close(self) -> None:
        """Gracefully terminate all worker processes."""

        if not self._started:
            return

        for _ in self._workers:
            self._task_queue.put(None)

        for proc in self._workers:
            proc.join(timeout=5)
            if proc.is_alive():
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                except Exception:
                    pass
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1)

        self._workers.clear()
        self._started = False
