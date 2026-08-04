"""Run DeepSeek-V4-Flash-0731 and one canonical EvolutionLoop in a binary job.

This is the only process the scheduler starts. It owns the eight-GPU SGLang
server, waits for an OpenAI-compatible smoke test, launches the single-device
EvolutionLoop coordinator, emits durable manifests, and always stops SGLang.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import IO, Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from scripts.cluster.common import (
    DEEPGEMM_NVCC_VERSION,
    DEEPGEMM_TOOLCHAIN_ID,
    MODEL_ID,
    MODEL_REVISION,
    SGLANG_CUDA_VARIANT,
    SGLANG_RUNTIME_ID,
    SGLANG_VERSION,
    SERVED_MODEL_NAME,
    TVM_FFI_CACHE_ID,
    TVM_FFI_VERSION,
    atomic_write_json,
    build_evolution_command,
    build_sglang_command,
    read_json_if_present,
    require_absolute_safe_path,
    utc_now,
)

SERVER_PORT = 30000
SERVER_BASE_URL = f"http://127.0.0.1:{SERVER_PORT}"


def _log(message: str) -> None:
    print(f"[deepseek-job] {utc_now()} {message}", flush=True)


def _event(kind: str, payload: dict[str, Any]) -> None:
    """Emit a single parseable scheduler-log record for remote monitoring."""

    print(
        "EVOLUTIONLOOP_EVENT "
        + json.dumps({"event": kind, "observed_at": utc_now(), **payload}, sort_keys=True),
        flush=True,
    )


def _run_checked(argv: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    _log("exec: " + " ".join(argv))
    subprocess.run(argv, cwd=str(cwd), env=env, check=True)


def _git_provenance(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=str(project_root), check=False, capture_output=True, text=True
        )
        return completed.stdout.strip()

    status = run("status", "--porcelain=v1")
    discovered_commit = run("rev-parse", "HEAD")
    source_commit = os.environ.get("SOURCE_COMMIT", "")
    return {
        "commit": source_commit or discovered_commit,
        "runtime_git_commit": discovered_commit,
        "branch": run("branch", "--show-current"),
        "describe": run("describe", "--always", "--dirty", "--broken"),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def _gpu_inventory() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,uuid,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(query, check=True, capture_output=True, text=True)
    rows = []
    for raw_line in completed.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",", 5)]
        if len(parts) == 6:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_mib": int(parts[2]),
                    "uuid": parts[3],
                    "driver_version": parts[4],
                    "compute_capability": parts[5],
                }
            )
    if len(rows) != 8:
        raise RuntimeError(f"job requires exactly 8 visible GPUs; found {len(rows)}")
    non_h100 = [row["name"] for row in rows if "H100" not in row["name"].upper()]
    if non_h100:
        raise RuntimeError(f"job requires H100 GPUs; found {non_h100}")
    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"], check=False, capture_output=True, text=True
    ).stdout
    return {"query": query, "gpus": rows, "topology": topology}


def _tee_stream(source: IO[str], destination: IO[str]) -> None:
    try:
        for line in iter(source.readline, ""):
            destination.write(line)
            destination.flush()
            sys.stdout.write("[sglang] " + line)
            sys.stdout.flush()
    finally:
        source.close()


def _server_ready() -> bool:
    for suffix in ("/health", "/v1/models"):
        try:
            with urllib_request.urlopen(SERVER_BASE_URL + suffix, timeout=5) as response:
                if 200 <= response.status < 300:
                    return True
        except (OSError, urllib_error.URLError):
            continue
    return False


def _wait_for_server(process: subprocess.Popen[str], *, timeout_sec: int) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"SGLang exited before readiness with code {return_code}")
        if _server_ready():
            _log("SGLang health endpoint is ready")
            return
        time.sleep(10)
    raise TimeoutError(f"SGLang did not become ready within {timeout_sec} seconds")


def _terminate_process(process: subprocess.Popen[str] | None, *, grace_sec: int = 90) -> None:
    if process is None or process.poll() is not None:
        return
    _log(f"stopping SGLang process group pid={process.pid}")
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        _log("SGLang did not stop gracefully; sending SIGKILL")
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)


def _population_progress(population_root: Path) -> dict[str, Any]:
    state = read_json_if_present(population_root / "population_state.json") or {}
    return {
        "state_present": bool(state),
        "current_generation": state.get("current_generation"),
        "active_organisms": len(state.get("active_organisms", []))
        if isinstance(state.get("active_organisms"), list)
        else None,
        "inflight_seed": isinstance(state.get("inflight_seed"), dict),
        "inflight_generation": isinstance(state.get("inflight_generation"), dict),
        "usage_events": sum(
            1
            for line in (population_root / "llm_usage.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if (population_root / "llm_usage.jsonl").exists()
        else 0,
    }


def _report_progress_until_stopped(
    population_root: Path,
    stop: threading.Event,
    *,
    interval_sec: float,
) -> None:
    while not stop.wait(max(10.0, interval_sec)):
        _event("evolution_progress", _population_progress(population_root))


def main() -> int:
    project_root = require_absolute_safe_path(
        os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[2])),
        label="PROJECT_ROOT",
    )
    run_id = os.environ.get("RUN_ID", "deepseek-v4-circle-300")
    run_dir = require_absolute_safe_path(
        os.environ.get("RUN_DIR", str(project_root / "cluster_runs" / run_id)),
        label="RUN_DIR",
    )
    env_dir = require_absolute_safe_path(
        os.environ.get(
            "DEEPSEEK_ENV_DIR",
            str(project_root.parent / ".inference_runtime" / SGLANG_RUNTIME_ID),
        ),
        label="DEEPSEEK_ENV_DIR",
    )
    toolchain_dir = require_absolute_safe_path(
        os.environ.get(
            "DEEPGEMM_CUDA_TOOLCHAIN_DIR",
            str(project_root.parent / ".inference_toolchains" / DEEPGEMM_TOOLCHAIN_ID),
        ),
        label="DEEPGEMM_CUDA_TOOLCHAIN_DIR",
    )
    deep_gemm_cache_dir = require_absolute_safe_path(
        os.environ.get(
            "SGLANG_DG_CACHE_DIR",
            str(project_root.parent / ".inference_kernel_cache" / DEEPGEMM_TOOLCHAIN_ID),
        ),
        label="SGLANG_DG_CACHE_DIR",
    )
    tvm_ffi_cache_dir = require_absolute_safe_path(
        os.environ.get(
            "TVM_FFI_CACHE_DIR",
            str(project_root.parent / ".inference_kernel_cache" / TVM_FFI_CACHE_ID),
        ),
        label="TVM_FFI_CACHE_DIR",
    )
    hf_home = require_absolute_safe_path(
        os.environ.get("HF_HOME", str(project_root.parent / ".model_cache" / "huggingface")),
        label="HF_HOME",
    )
    config_name = os.environ.get("CONFIG_NAME", "config_circle_packing_shinka")
    backbone = os.environ.get("BACKBONE", "deepseek_v4_flash_0731")
    max_generations = int(os.environ.get("MAX_GENERATIONS", "300"))
    max_parallel = int(os.environ.get("MAX_PARALLEL_ORGANISMS", "8"))
    server_timeout = int(os.environ.get("SERVER_START_TIMEOUT_SEC", "10800"))
    extra_overrides_raw = os.environ.get("HYDRA_OVERRIDES_JSON", "[]")
    extra_overrides = json.loads(extra_overrides_raw)
    if not isinstance(extra_overrides, list) or not all(isinstance(value, str) for value in extra_overrides):
        raise ValueError("HYDRA_OVERRIDES_JSON must encode a list of strings")

    run_dir.mkdir(parents=True, exist_ok=True)
    population_root = run_dir / "population"
    hydra_dir = run_dir / "hydra"
    manifest_path = run_dir / "run_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "run_id": run_id,
        "started_at": utc_now(),
        "project_root": str(project_root),
        "run_dir": str(run_dir),
        "population_root": str(population_root),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "sglang_version": SGLANG_VERSION,
        "sglang_cuda_variant": SGLANG_CUDA_VARIANT,
        "deep_gemm_toolchain": {
            "id": DEEPGEMM_TOOLCHAIN_ID,
            "nvcc_version": DEEPGEMM_NVCC_VERSION,
            "path": str(toolchain_dir),
            "kernel_cache": str(deep_gemm_cache_dir),
            "tvm_ffi_version": TVM_FFI_VERSION,
            "tvm_ffi_cache": str(tvm_ffi_cache_dir),
        },
        "cuda_driver_environment": {
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
            "compat_path_present": any(
                "compat" in Path(component).parts
                for component in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
                if component
            ),
        },
        "config_name": config_name,
        "backbone": backbone,
        "max_generations": max_generations,
        "max_parallel_organisms": max_parallel,
        "extra_overrides": extra_overrides,
        "git": _git_provenance(project_root),
    }
    atomic_write_json(manifest_path, manifest)

    environment = os.environ.copy()
    toolchain_cuda_library_dir = toolchain_dir / "targets" / "x86_64-linux" / "lib"
    environment.update(
        {
            "DEEPSEEK_ENV_DIR": str(env_dir),
            "SGLANG_VERSION": SGLANG_VERSION,
            "SGLANG_CUDA_VARIANT": SGLANG_CUDA_VARIANT,
            "HF_HOME": str(hf_home),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "SGLANG_DSV4_COMPRESS_STATE_DTYPE": "bf16",
            "DEEPGEMM_CUDA_TOOLCHAIN_DIR": str(toolchain_dir),
            "DEEPGEMM_NVCC_VERSION": DEEPGEMM_NVCC_VERSION,
            "DG_JIT_NVCC_COMPILER": str(toolchain_dir / "bin" / "nvcc"),
            "DG_JIT_PRINT_COMPILER_COMMAND": "1",
            "SGLANG_DG_CACHE_DIR": str(deep_gemm_cache_dir),
            "CUDA_HOME": str(toolchain_dir),
            "LIBRARY_PATH": str(toolchain_cuda_library_dir)
            + (
                os.pathsep + environment["LIBRARY_PATH"]
                if environment.get("LIBRARY_PATH")
                else ""
            ),
            "TVM_FFI_CACHE_DIR": str(tvm_ffi_cache_dir),
            "TVM_FFI_CUDA_ARCH_LIST": "9.0a",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(project_root)
            + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""),
            "COMET_ENABLED": "false",
            "DEEPSEEK_V4_BASE_URL": f"{SERVER_BASE_URL}/v1",
        }
    )

    server: subprocess.Popen[str] | None = None
    tee_thread: threading.Thread | None = None
    progress_thread: threading.Thread | None = None
    progress_stop = threading.Event()
    server_log: IO[str] | None = None
    stop_requested = False

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        _log(f"received signal {signum}")
        _terminate_process(server)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, handle_signal)

    try:
        inventory = _gpu_inventory()
        atomic_write_json(run_dir / "gpu_inventory.json", inventory)
        manifest.update(status="bootstrapping", gpu_inventory=str(run_dir / "gpu_inventory.json"))
        atomic_write_json(manifest_path, manifest)

        _run_checked(
            ["bash", str(project_root / "scripts" / "cluster" / "bootstrap_cuda_toolchain.sh")],
            cwd=project_root,
            env=environment,
        )
        _run_checked(
            ["bash", str(project_root / "scripts" / "cluster" / "bootstrap_deepseek_env.sh")],
            cwd=project_root,
            env=environment,
        )
        env_python = str(env_dir / "bin" / "python")
        environment["PATH"] = str(env_dir / "bin") + os.pathsep + environment.get("PATH", "")
        deep_gemm_cache_dir.mkdir(parents=True, exist_ok=True)
        tvm_ffi_cache_dir.mkdir(parents=True, exist_ok=True)
        _run_checked(
            [
                env_python,
                "-m",
                "scripts.cluster.smoke_deepgemm_toolchain",
                "--compiler",
                environment["DG_JIT_NVCC_COMPILER"],
                "--cache-dir",
                str(deep_gemm_cache_dir),
                "--expected-version",
                DEEPGEMM_NVCC_VERSION,
                "--output",
                str(run_dir / "deepgemm_toolchain_smoke.json"),
            ],
            cwd=project_root,
            env=environment,
        )
        _run_checked(
            [
                env_python,
                "-m",
                "scripts.cluster.smoke_sglang_jit_toolchain",
                "--cuda-home",
                str(toolchain_dir),
                "--cache-dir",
                str(tvm_ffi_cache_dir),
                "--expected-nvcc-version",
                DEEPGEMM_NVCC_VERSION,
                "--expected-tvm-ffi-version",
                TVM_FFI_VERSION,
                "--world-size",
                "8",
                "--output",
                str(run_dir / "sglang_jit_toolchain_smoke.json"),
            ],
            cwd=project_root,
            env=environment,
        )

        model_manifest_path = run_dir / "model_snapshot.json"
        _run_checked(
            [
                env_python,
                "-m",
                "scripts.cluster.download_model",
                "--repo-id",
                MODEL_ID,
                "--revision",
                MODEL_REVISION,
                "--output",
                str(model_manifest_path),
            ],
            cwd=project_root,
            env=environment,
        )
        model_manifest = read_json_if_present(model_manifest_path)
        model_path = str((model_manifest or {}).get("snapshot_path", ""))
        if not model_path or not Path(model_path).is_dir():
            raise RuntimeError("model download did not produce a valid snapshot_path")

        server_command = build_sglang_command(
            python=env_python,
            model_path=model_path,
            port=SERVER_PORT,
            max_running_requests=max_parallel,
        )
        atomic_write_json(
            run_dir / "sglang_launch.json",
            {"created_at": utc_now(), "argv": server_command, "environment": {
                "SGLANG_DSV4_COMPRESS_STATE_DTYPE": environment["SGLANG_DSV4_COMPRESS_STATE_DTYPE"],
                "DG_JIT_NVCC_COMPILER": environment["DG_JIT_NVCC_COMPILER"],
                "DG_JIT_PRINT_COMPILER_COMMAND": environment["DG_JIT_PRINT_COMPILER_COMMAND"],
                "SGLANG_DG_CACHE_DIR": environment["SGLANG_DG_CACHE_DIR"],
                "CUDA_HOME": environment["CUDA_HOME"],
                "LIBRARY_PATH": environment["LIBRARY_PATH"],
                "TVM_FFI_CACHE_DIR": environment["TVM_FFI_CACHE_DIR"],
                "TVM_FFI_CUDA_ARCH_LIST": environment["TVM_FFI_CUDA_ARCH_LIST"],
                "HF_HOME": str(hf_home),
                "LD_LIBRARY_PATH": environment.get("LD_LIBRARY_PATH", ""),
            }},
        )
        server_log = (run_dir / "sglang.log").open("a", encoding="utf-8", buffering=1)
        _log("starting SGLang")
        server = subprocess.Popen(
            server_command,
            cwd=str(project_root),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        if server.stdout is None:
            raise RuntimeError("failed to capture SGLang stdout")
        tee_thread = threading.Thread(
            target=_tee_stream, args=(server.stdout, server_log), name="sglang-log-tee", daemon=True
        )
        tee_thread.start()
        manifest.update(status="starting_server", server_pid=server.pid)
        atomic_write_json(manifest_path, manifest)
        _wait_for_server(server, timeout_sec=server_timeout)

        _run_checked(
            [
                env_python,
                "-m",
                "scripts.cluster.smoke_endpoint",
                "--base-url",
                f"{SERVER_BASE_URL}/v1",
                "--model",
                SERVED_MODEL_NAME,
                "--concurrency",
                str(min(4, max_parallel)),
                "--output",
                str(run_dir / "smoke.json"),
            ],
            cwd=project_root,
            env=environment,
        )

        evolution_command = build_evolution_command(
            project_root=project_root,
            population_root=population_root,
            hydra_run_dir=hydra_dir,
            config_name=config_name,
            backbone=backbone,
            max_generations=max_generations,
            max_parallel_organisms=max_parallel,
            extra_overrides=extra_overrides,
        )
        atomic_write_json(run_dir / "evolution_launch.json", {"created_at": utc_now(), "argv": evolution_command})
        manifest.update(status="running_evolution", evolution_started_at=utc_now())
        atomic_write_json(manifest_path, manifest)
        _log("starting canonical EvolutionLoop")
        _event("evolution_started", _population_progress(population_root))
        progress_thread = threading.Thread(
            target=_report_progress_until_stopped,
            args=(population_root, progress_stop),
            kwargs={"interval_sec": float(os.environ.get("PROGRESS_LOG_INTERVAL_SEC", "60"))},
            name="evolution-progress-reporter",
            daemon=True,
        )
        progress_thread.start()
        evolution_return_code = subprocess.run(
            evolution_command, cwd=str(project_root), env=environment, check=False
        ).returncode
        progress_stop.set()
        progress_thread.join(timeout=10)
        manifest["evolution_return_code"] = evolution_return_code
        manifest["progress"] = _population_progress(population_root)
        if evolution_return_code != 0:
            raise RuntimeError(f"EvolutionLoop exited with code {evolution_return_code}")

        usage_path = population_root / "llm_usage.jsonl"
        if usage_path.exists():
            _run_checked(
                [
                    env_python,
                    "-m",
                    "src.evolve.token_usage_report",
                    str(usage_path),
                    "--output",
                    str(run_dir / "token_usage_summary.json"),
                ],
                cwd=project_root,
                env=environment,
            )
        manifest.update(status="completed", completed_at=utc_now(), progress=_population_progress(population_root))
        atomic_write_json(manifest_path, manifest)
        token_summary = read_json_if_present(run_dir / "token_usage_summary.json") or {}
        _event(
            "run_completed",
            {"run_id": run_id, "progress": manifest["progress"], "token_usage": token_summary},
        )
        _log("job completed")
        return 0
    except BaseException as exc:  # noqa: BLE001
        manifest.update(
            status="interrupted" if stop_requested else "failed",
            finished_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            progress=_population_progress(population_root),
        )
        atomic_write_json(manifest_path, manifest)
        _event(
            "run_failed",
            {"run_id": run_id, "error": manifest["error"], "progress": manifest["progress"]},
        )
        _log(f"job failed: {type(exc).__name__}: {exc}")
        return 130 if stop_requested else 1
    finally:
        progress_stop.set()
        if progress_thread is not None:
            progress_thread.join(timeout=10)
        _terminate_process(server)
        if tee_thread is not None:
            tee_thread.join(timeout=10)
        if server_log is not None:
            server_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
