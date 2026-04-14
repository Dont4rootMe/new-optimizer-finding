"""Route registry that manages singleton broker processes and IPC clients."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from api_platforms._core.config import build_route_config
from api_platforms._core.config import stable_config_hash
from api_platforms._core.discovery import load_route_configs
from api_platforms._core.ipc import send_ipc_message
from api_platforms._core.types import ApiPlatformClient, ApiRouteConfig, LlmRequest, LlmResponse
from src.evolve.storage import ensure_dir, write_json


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@contextmanager
def _locked_file(path: Path):
    ensure_dir(path.parent)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class _BrokerClient(ApiPlatformClient):
    def __init__(self, socket_path: Path, timeout_sec: float) -> None:
        self.socket_path = socket_path
        self.timeout_sec = float(timeout_sec)

    def ping(self, timeout_sec: float | None = None) -> bool:
        try:
            payload = send_ipc_message(
                str(self.socket_path),
                {"type": "ping"},
                self.timeout_sec if timeout_sec is None else float(timeout_sec),
            )
        except Exception:  # noqa: BLE001
            return False
        return bool(payload.get("ok") and payload.get("pong"))

    def shutdown(self) -> None:
        send_ipc_message(str(self.socket_path), {"type": "shutdown"}, self.timeout_sec)

    def generate(self, request: LlmRequest) -> LlmResponse:
        payload = send_ipc_message(
            str(self.socket_path),
            {"type": "generate", "request": request.to_dict()},
            self.timeout_sec,
        )
        if not bool(payload.get("ok")):
            raise RuntimeError(str(payload.get("error", "broker request failed")))
        return LlmResponse.from_dict(dict(payload["response"]))


class ApiPlatformRegistry:
    """Process-safe route broker registry for one evolve runtime."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        runtime_root_value = None
        paths_cfg = getattr(cfg, "paths", None)
        if paths_cfg is not None:
            runtime_root_value = getattr(paths_cfg, "api_platform_runtime_root", None)
        configured_runtime_root = Path(str(runtime_root_value or (Path.cwd() / ".api_platform_runtime"))).expanduser()
        if len(str(configured_runtime_root.resolve())) > 48:
            root_hash = stable_config_hash(
                build_route_config(
                    route_id="runtime_root",
                    provider="internal",
                    provider_model_id=str(configured_runtime_root),
                    backend="mock",
                )
            )[:12]
            configured_runtime_root = Path("/tmp") / "api_platforms" / root_hash
        self.runtime_root = ensure_dir(str(configured_runtime_root))
        self.route_configs = load_route_configs(cfg)
        if not self.route_configs:
            self.route_configs = self._legacy_route_configs()
        self.registry_id = uuid.uuid4().hex
        self._leases: dict[str, Path] = {}
        self._clients: dict[str, _BrokerClient] = {}

    def _legacy_route_configs(self) -> dict[str, ApiRouteConfig]:
        llm_cfg = getattr(getattr(self.cfg, "evolver", None), "llm", None)
        provider = str(getattr(llm_cfg, "provider", "")).strip().lower() if llm_cfg is not None else ""
        model_name = str(getattr(llm_cfg, "model", "mock-model")).strip() if llm_cfg is not None else "mock-model"
        if provider == "mock":
            return {
                "mock": build_route_config(
                    route_id="mock",
                    provider="mock",
                    provider_model_id=model_name or "mock-model",
                    backend="mock",
                    max_concurrency=16,
                    max_output_tokens=int(getattr(llm_cfg, "max_output_tokens", 2048) or 2048),
                    temperature=float(getattr(llm_cfg, "temperature", 0.0) or 0.0),
                )
            }
        return {}

    def __enter__(self) -> "ApiPlatformRegistry":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.stop()

    @property
    def available_route_ids(self) -> list[str]:
        return sorted(self.route_configs.keys())

    def validate_route_weights(self, weights: dict[str, float]) -> None:
        unknown = sorted(set(weights) - set(self.route_configs))
        if unknown:
            raise ValueError(
                "evolver.llm.route_weights references unknown route ids: " + ", ".join(unknown)
            )

    def _route_runtime_dir(self, route_id: str) -> Path:
        route_cfg = self.route_configs[route_id]
        cfg_hash = stable_config_hash(route_cfg)[:12]
        return ensure_dir(self.runtime_root / f"{route_id}_{cfg_hash}")

    def _lock_path(self, route_id: str) -> Path:
        return self._route_runtime_dir(route_id) / "startup.lock"

    def _lease_dir(self, route_id: str) -> Path:
        return ensure_dir(self._route_runtime_dir(route_id) / "leases")

    def _lease_path(self, route_id: str) -> Path:
        return self._lease_dir(route_id) / f"lease_{os.getpid()}_{self.registry_id}.json"

    def _socket_path(self, route_id: str) -> Path:
        return self._route_runtime_dir(route_id) / "broker.sock"

    def _client_for(self, route_id: str) -> _BrokerClient:
        if route_id not in self._clients:
            route_cfg = self.route_configs[route_id]
            self._clients[route_id] = _BrokerClient(self._socket_path(route_id), route_cfg.timeout_sec)
        return self._clients[route_id]

    def _broker_probe_timeout_sec(self, route_id: str) -> float:
        route_timeout = float(self.route_configs[route_id].timeout_sec)
        return max(0.2, min(2.0, route_timeout))

    def _cleanup_stale_leases(self, route_id: str) -> None:
        for lease_path in self._lease_dir(route_id).glob("lease_*_*.json"):
            parts = lease_path.stem.split("_")
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[1])
            except ValueError:
                continue
            if not _pid_exists(pid):
                lease_path.unlink(missing_ok=True)

    def _write_lease(self, route_id: str) -> None:
        lease_path = self._lease_path(route_id)
        if lease_path.exists():
            self._leases[route_id] = lease_path
            return
        write_json(
            lease_path,
            {
                "pid": os.getpid(),
                "registry_id": self.registry_id,
                "route_id": route_id,
                "created_at": time.time(),
            },
        )
        self._leases[route_id] = lease_path

    def _broker_healthy(self, route_id: str) -> bool:
        socket_path = self._socket_path(route_id)
        if not socket_path.exists():
            return False
        return self._client_for(route_id).ping(timeout_sec=self._broker_probe_timeout_sec(route_id))

    def _spawn_broker(self, route_id: str) -> subprocess.Popen[Any]:
        route_cfg = self.route_configs[route_id]
        runtime_dir = self._route_runtime_dir(route_id)
        repo_root = Path(__file__).resolve().parents[2]
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        repo_root_text = str(repo_root)
        env["PYTHONPATH"] = (
            repo_root_text
            if not existing_pythonpath
            else repo_root_text + os.pathsep + existing_pythonpath
        )
        cmd = [
            sys.executable,
            "-m",
            "api_platforms._core.broker_runner",
            "--runtime-dir",
            str(runtime_dir),
            "--config-json",
            json.dumps(route_cfg.to_dict(), sort_keys=True),
        ]
        stdout_handle = (runtime_dir / "broker.stdout.log").open("a", encoding="utf-8")
        stderr_handle = (runtime_dir / "broker.stderr.log").open("a", encoding="utf-8")
        return subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )

    def ensure_route_started(self, route_id: str) -> None:
        if route_id not in self.route_configs:
            raise KeyError(f"Unknown API platform route '{route_id}'.")
        with _locked_file(self._lock_path(route_id)):
            self._cleanup_stale_leases(route_id)
            if not self._broker_healthy(route_id):
                process = self._spawn_broker(route_id)
                timeout_deadline = time.time() + max(5.0, float(self.route_configs[route_id].timeout_sec))
                while time.time() < timeout_deadline:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"Broker for route '{route_id}' exited early with code {process.returncode}."
                        )
                    if self._broker_healthy(route_id):
                        break
                    time.sleep(0.1)
                else:
                    process.terminate()
                    raise TimeoutError(f"Timed out while starting broker for route '{route_id}'.")
            self._write_lease(route_id)

    def start(self) -> None:
        for route_id in self.available_route_ids:
            self.ensure_route_started(route_id)

    def client_for(self, route_id: str) -> ApiPlatformClient:
        self.ensure_route_started(route_id)
        return self._client_for(route_id)

    def generate(self, request: LlmRequest) -> LlmResponse:
        client = self.client_for(request.route_id)
        return client.generate(request)

    def stop(self) -> None:
        for route_id in self.available_route_ids:
            with _locked_file(self._lock_path(route_id)):
                self._cleanup_stale_leases(route_id)
                lease_path = self._leases.pop(route_id, None)
                if lease_path is not None:
                    lease_path.unlink(missing_ok=True)
                self._cleanup_stale_leases(route_id)
                remaining = list(self._lease_dir(route_id).glob("lease_*_*.json"))
                if not remaining and self._broker_healthy(route_id):
                    try:
                        self._client_for(route_id).shutdown()
                    except Exception:  # noqa: BLE001
                        pass
