"""Shared, side-effect-free cluster job primitives.

The scheduler-facing modules deliberately keep ``client_lib`` behind lazy
imports so command construction, manifests, and monitors remain locally
testable without an ML Space session.
"""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
MODEL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
SGLANG_VERSION = "0.5.16"
SGLANG_CUDA_VARIANT = "cu126"
SGLANG_RUNTIME_ID = f"sglang-{SGLANG_VERSION}-{SGLANG_CUDA_VARIANT}"
DEEPGEMM_NVCC_VERSION = "12.9.86"
DEEPGEMM_TOOLCHAIN_ID = f"cuda-nvcc-{DEEPGEMM_NVCC_VERSION}"
TVM_FFI_VERSION = "0.1.11"
TVM_FFI_CACHE_ID = f"tvm-ffi-sm90-{DEEPGEMM_TOOLCHAIN_ID}-tvmffi-{TVM_FFI_VERSION}"
SERVED_MODEL_NAME = MODEL_ID

REGION = "SR008"
INSTANCE_TYPE = "a100plus.8gpu.80vG.96C.1456G"
BASE_IMAGE = (
    "cr.ai.cloud.ru/2754eb6e-ae19-4123-87ce-06ec3cc96500/"
    "job-latentdiffusion:flash-clear"
)
REPOSITORY_URL = "https://github.com/Dont4rootMe/new-optimizer-finding.git"
REGIONAL_JOB_ROOT = Path("/home/jovyan/evolutionloop-deepseek-v4")


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


def require_regional_job_path(value: str | os.PathLike[str], *, label: str) -> Path:
    """Validate a logical SR008 NFS path without resolving host symlinks.

    macOS resolves ``/home`` through ``/System/Volumes/Data`` even when merely
    constructing a remote path.  Regional paths must retain their literal
    ``/home/jovyan/...`` spelling in scheduler requests.
    """

    path = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    if path in {Path("/"), Path("/home"), Path("/home/jovyan")}:
        raise ValueError(f"refusing unsafe {label}: {path}")
    if not str(path).startswith("/home/jovyan/"):
        raise ValueError(f"{label} must be below /home/jovyan: {path}")
    return path


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


def build_git_bootstrap_command(
    *,
    repository_url: str,
    commit: str,
    job_root: Path,
    entrypoint: str = "scripts/cluster/run_deepseek_v4_circle.sh",
) -> str:
    """Build a validation-safe one-line job command that checks out one commit.

    SR008 does not mount the submitting Jupyter server's NFS namespace.  The
    binary job therefore bootstraps its immutable source into regional NFS.
    A base64-encoded stdlib Python payload avoids shell interpolation of URLs,
    paths, or scheduler environment values.
    """

    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"source commit must be a full lowercase SHA-1: {commit!r}")
    root = require_regional_job_path(job_root, label="regional job root")
    if not repository_url.startswith("https://"):
        raise ValueError("cluster bootstrap repository URL must use HTTPS")
    entrypoint_path = Path(entrypoint)
    if entrypoint_path.is_absolute() or ".." in entrypoint_path.parts:
        raise ValueError(f"cluster entrypoint must be a safe repository-relative path: {entrypoint}")

    payload = f'''import os
import pathlib
import subprocess

rank_text = os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "0"))
try:
    rank = int(rank_text)
except ValueError:
    rank = 0
if rank != 0:
    print(f"[git-bootstrap] rank {{rank}}: coordinator owned by rank 0; exiting", flush=True)
    raise SystemExit(0)

job_root = pathlib.Path({str(root)!r})
commit = {commit!r}
entrypoint = {entrypoint!r}
project_root = job_root / "source" / commit
project_root.parent.mkdir(parents=True, exist_ok=True)
if not (project_root / ".git").is_dir():
    if project_root.exists():
        raise RuntimeError(f"incomplete source path preserved for diagnosis: {{project_root}}")
    subprocess.run([
        "git", "clone", "--filter=blob:none", "--no-checkout", "--no-tags",
        {repository_url!r}, str(project_root),
    ], check=True)
subprocess.run(["git", "-C", str(project_root), "fetch", "--depth=1", "origin", commit], check=True)
subprocess.run(["git", "-C", str(project_root), "checkout", "--detach", commit], check=True)
head = subprocess.run(
    ["git", "-C", str(project_root), "rev-parse", "HEAD"],
    check=True, capture_output=True, text=True,
).stdout.strip()
if head != commit:
    raise RuntimeError(f"source verification failed: expected {{commit}}, got {{head}}")
os.environ["PROJECT_ROOT"] = str(project_root)
print(f"[git-bootstrap] verified commit={{commit}} project_root={{project_root}}", flush=True)
runtime_entrypoint = project_root / entrypoint
if not runtime_entrypoint.is_file():
    raise RuntimeError(f"cluster entrypoint does not exist: {{runtime_entrypoint}}")
os.execv("/bin/bash", ["bash", str(runtime_entrypoint)])
'''
    encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    return f"python3 -c 'import base64;exec(base64.b64decode(\"{encoded}\"))'"


def build_sglang_command(
    *,
    python: str,
    model_path: str,
    port: int = 30000,
    max_running_requests: int = 8,
    context_length: int = 65536,
) -> list[str]:
    """Return the conservative throughput-oriented 8xH100 launch command.

    DeepSeek's stock FP4 checkpoint is pinned to SGLang's Hopper W4A16/Marlin
    runner. Blackwell-only MXFP4 flags are forbidden.
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
        "--moe-runner-backend",
        "marlin",
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
        "--cuda-graph-max-bs-decode",
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
