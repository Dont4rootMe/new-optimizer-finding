"""Singleton route broker with Unix-socket IPC."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import socketserver
import threading
import uuid
from pathlib import Path
from typing import Any

from api_platforms._core.ipc import read_json_line, write_json_line
from api_platforms._core.local_worker import local_worker_main
from api_platforms._core.providers import generate_direct
from api_platforms._core.types import ApiPlatformBroker, ApiRouteConfig, LlmRequest, LlmResponse
from src.evolve.storage import ensure_dir, write_json


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


class _LocalWorkerPool:
    def __init__(self, route_cfg: ApiRouteConfig) -> None:
        self.route_cfg = route_cfg
        self.ctx = mp.get_context("spawn")
        self.input_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        ranks = route_cfg.gpu_ranks or [None]
        self.processes = [
            self.ctx.Process(
                target=local_worker_main,
                args=(route_cfg.to_dict(), rank, self.input_queue, self.result_queue),
                daemon=True,
            )
            for rank in ranks
        ]
        self.pending: dict[str, "queue.Queue[dict[str, Any]]"] = {}
        self.pending_lock = threading.Lock()
        self.listener = threading.Thread(target=self._listen_results, daemon=True)
        self.running = False

    def start(self) -> None:
        if self.running:
            return
        for process in self.processes:
            process.start()
        self.running = True
        self.listener.start()

    def _listen_results(self) -> None:
        while self.running:
            try:
                payload = self.result_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if not isinstance(payload, dict):
                continue
            task_id = str(payload.get("task_id", ""))
            if not task_id:
                continue
            with self.pending_lock:
                waiter = self.pending.pop(task_id, None)
            if waiter is not None:
                waiter.put(payload)

    def generate(self, request: LlmRequest) -> LlmResponse:
        if not self.running:
            self.start()
        task_id = uuid.uuid4().hex
        waiter: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=1)
        with self.pending_lock:
            self.pending[task_id] = waiter
        self.input_queue.put(
            {
                "task_id": task_id,
                "request": request.to_dict(),
                "started_at": _utc_now_iso(),
            }
        )
        try:
            payload = waiter.get(timeout=max(1.0, float(self.route_cfg.timeout_sec) + 5.0))
        except queue.Empty as exc:
            raise TimeoutError(
                f"Route '{self.route_cfg.route_id}' timed out waiting for local worker result."
            ) from exc
        if not bool(payload.get("ok")):
            raise RuntimeError(str(payload.get("error", "local worker failed")))
        return LlmResponse.from_dict(dict(payload["response"]))

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        for _ in self.processes:
            self.input_queue.put(None)
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()


class RouteBroker(ApiPlatformBroker):
    """Unix-socket broker serving one route config."""

    def __init__(self, route_cfg: ApiRouteConfig, runtime_dir: str | Path) -> None:
        self.route_cfg = route_cfg
        self.runtime_dir = ensure_dir(runtime_dir)
        self.socket_path = self.runtime_dir / "broker.sock"
        self.pid_path = self.runtime_dir / "broker.pid"
        self.health_path = self.runtime_dir / "health.json"
        self._server: socketserver.ThreadingUnixStreamServer | None = None
        self._semaphore = threading.BoundedSemaphore(max(1, int(route_cfg.max_concurrency)))
        self._local_pool = _LocalWorkerPool(route_cfg) if route_cfg.backend in {"transformers", "mock_local"} else None

    def start(self) -> None:
        if self.socket_path.exists():
            self.socket_path.unlink()

        broker = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:  # type: ignore[override]
                try:
                    payload = read_json_line(self.rfile)
                    response = broker._handle_message(payload)
                except Exception as exc:  # noqa: BLE001
                    response = {"ok": False, "error": str(exc)}
                write_json_line(self.wfile, response)

        self._server = socketserver.ThreadingUnixStreamServer(str(self.socket_path), Handler)
        self._server.daemon_threads = True
        if self._local_pool is not None:
            self._local_pool.start()

        write_json(self.pid_path, {"pid": os.getpid(), "route_id": self.route_cfg.route_id})
        write_json(
            self.health_path,
            {
                "pid": os.getpid(),
                "route_id": self.route_cfg.route_id,
                "provider": self.route_cfg.provider,
                "provider_model_id": self.route_cfg.provider_model_id,
                "backend": self.route_cfg.backend,
                "started_at": _utc_now_iso(),
            },
        )

    def _handle_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        message_type = str(payload.get("type", "")).strip()
        if message_type == "ping":
            return {"ok": True, "pong": True, "route_id": self.route_cfg.route_id}
        if message_type == "shutdown":
            threading.Thread(target=self.stop, daemon=True).start()
            return {"ok": True, "shutting_down": True}
        if message_type != "generate":
            raise ValueError(f"Unsupported broker message type '{message_type}'.")

        request = LlmRequest.from_dict(dict(payload["request"]))
        with self._semaphore:
            if self._local_pool is not None:
                response = self._local_pool.generate(request)
            else:
                response = generate_direct(self.route_cfg, request)
        return {"ok": True, "response": response.to_dict()}

    def serve_forever(self) -> None:
        if self._server is None:
            self.start()
        assert self._server is not None
        try:
            self._server.serve_forever()
        finally:
            self.stop()

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._server.server_close()
            except Exception:  # noqa: BLE001
                pass
            self._server = None
        if self._local_pool is not None:
            self._local_pool.stop()
        for path in (self.socket_path, self.pid_path, self.health_path):
            if path.exists():
                path.unlink()
