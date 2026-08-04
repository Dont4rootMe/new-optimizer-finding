"""Shared, side-effect-free cluster job primitives.

The scheduler-facing modules deliberately keep ``client_lib`` behind lazy
imports so command construction, manifests, and monitors remain locally
testable without an ML Space session.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
MODEL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
SGLANG_VERSION = "0.5.16"
SERVED_MODEL_NAME = MODEL_ID

REGION = "SR008"
INSTANCE_TYPE = "a100plus.8gpu.80vG.96C.1456G"
BASE_IMAGE = (
    "cr.ai.cloud.ru/2754eb6e-ae19-4123-87ce-06ec3cc96500/"
    "job-latentdiffusion:flash-clear"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_absolute_safe_path(value: str | os.PathLike[str], *, label: str) -> Path:
    """Resolve a user-controlled job path while rejecting broad targets."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    resolved = path.resolve()
    if resolved in {Path("/"), Path("/home"), Path("/home/jovyan")}:
        raise ValueError(f"refusing unsafe {label}: {resolved}")
    return resolved


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically replace a JSON manifest on a shared filesystem."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def read_json_if_present(path: str | Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def build_sglang_command(
    *,
    python: str,
    model_path: str,
    port: int = 30000,
    max_running_requests: int = 8,
    context_length: int = 65536,
) -> list[str]:
    """Return the conservative throughput-oriented 8xH100 launch command.

    DeepSeek's stock FP4 checkpoint is intentionally left to SGLang's Hopper
    auto-selection (W4A16/Marlin). Blackwell-only MXFP4 flags are forbidden.
    DSpark is bundled in the 0731 checkpoint and needs no separate draft model.
    """

    if max_running_requests < 1:
        raise ValueError("max_running_requests must be positive")
    if context_length < 8192:
        raise ValueError("context_length must be at least 8192")
    return [
        python,
        "-m",
        "sglang.launch_server",
        "--trust-remote-code",
        "--model-path",
        model_path,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--tp",
        "8",
        "--speculative-algorithm",
        "DSPARK",
        "--mem-fraction-static",
        "0.88",
        "--chunked-prefill-size",
        "8192",
        "--context-length",
        str(context_length),
        "--max-running-requests",
        str(max_running_requests),
        "--cuda-graph-max-bs",
        str(max_running_requests),
        "--swa-full-tokens-ratio",
        "0.1",
        "--reasoning-parser",
        "deepseek-v4",
        "--tool-call-parser",
        "deepseekv4",
        "--enable-metrics",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]


def build_evolution_command(
    *,
    project_root: Path,
    population_root: Path,
    hydra_run_dir: Path,
    config_name: str,
    backbone: str,
    max_generations: int,
    max_parallel_organisms: int,
    extra_overrides: list[str] | None = None,
) -> list[str]:
    if max_generations < 1:
        raise ValueError("max_generations must be positive")
    if max_parallel_organisms < 1:
        raise ValueError("max_parallel_organisms must be positive")
    return [
        "bash",
        str(project_root / "scripts" / "run_evolution.sh"),
        "--seed",
        "--config-name",
        config_name,
        f"backbone={backbone}",
        f"evolver.max_generations={max_generations}",
        f"evolver.creation.max_parallel_organisms={max_parallel_organisms}",
        f"paths.population_root={population_root}",
        f"paths.api_platform_runtime_root={population_root.parent / '.api-platform-runtime'}",
        f"hydra.run.dir={hydra_run_dir}",
        "comet.enabled=false",
        *(extra_overrides or []),
    ]
