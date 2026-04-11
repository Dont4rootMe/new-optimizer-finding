"""Runtime entrypoint tests for seed-only population initialization."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from omegaconf import OmegaConf

from src.evolve.seed_run import run_seed_population

ROOT = Path(__file__).resolve().parents[1]


def _write_fake_python(
    path: Path,
    *,
    ollama_routes: list[tuple[str, ...]] | None = None,
) -> None:
    ollama_lines = [
        "printf '%s\\n' \"API_PLATFORM_RUNTIME_ROOT=${API_PLATFORM_RUNTIME_ROOT:-/tmp/api_platform_runtime}\"",
        "printf '%s\\n' \"OLLAMA_MODELS_DIR=${OLLAMA_MODELS_DIR:-/tmp/ollama_cache}\"",
    ]
    for route in ollama_routes or []:
        route_id, base_url, model = route[:3]
        gpu_ranks_csv = route[3] if len(route) > 3 else ""
        ollama_lines.append(f"printf '%s\\n' 'OLLAMA_ROUTE={route_id}|{base_url}|{model}|{gpu_ranks_csv}'")

    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ "${{1:-}}" == "-c" ]]; then
          exec python3 "$@"
        fi
        if [[ "${{1:-}}" == "-" ]]; then
          cat >/dev/null
          for arg in "$@"; do
            if [[ "$arg" == "__codex_inspect_ollama__" ]]; then
        {textwrap.indent(chr(10).join(ollama_lines), '        ')}
              exit 0
            fi
          done
        fi
        printf '%s\\n' "$*" >> "${{PYTHON_CALLS_FILE:?}}"
        exit 0
        """
    )
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def _write_fake_local_ollama_commands(bin_dir: Path) -> None:
    fake_curl = bin_dir / "curl"
    fake_curl.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            url="${@: -1}"
            printf '%s\\n' "$*" >> "${CURL_CALLS_FILE:?}"
            origin="${url#*://}"
            origin="${origin%%/*}"
            state_dir="${OLLAMA_STATE_DIR:?}/${origin//[:\\/]/_}"
            mkdir -p "$state_dir"
            case "$url" in
              */api/tags)
                if [[ ! -f "${state_dir}/server_ready" ]]; then
                  exit 7
                fi
                if [[ ! -f "${state_dir}/models.txt" ]]; then
                  printf '{"models":[]}\\n'
                  exit 0
                fi
                python3 - "${state_dir}/models.txt" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

models_path = Path(sys.argv[1])
models = [line.strip() for line in models_path.read_text(encoding="utf-8").splitlines() if line.strip()]
print(json.dumps({"models": [{"name": name} for name in models]}))
PY
                ;;
              */api/pull)
                body=""
                prev=""
                for arg in "$@"; do
                  if [[ "$prev" == "-d" ]]; then
                    body="$arg"
                    break
                  fi
                  prev="$arg"
                done
                model="$(printf '%s' "$body" | sed -E 's/.*"(model|name)":"([^"]+)".*/\\2/')"
                touch "${state_dir}/server_ready"
                touch "${state_dir}/models.txt"
                if ! grep -Fxq "$model" "${state_dir}/models.txt"; then
                  printf '%s\\n' "$model" >> "${state_dir}/models.txt"
                fi
                printf '{"status":"pulling manifest"}\\n'
                printf '{"status":"downloading","completed":1048576,"total":2097152}\\n'
                printf '{"status":"verifying sha256 digest"}\\n'
                printf '{"status":"success"}\\n'
                ;;
              *)
                echo "unexpected curl url: $url" >&2
                exit 1
                ;;
            esac
            """
        ),
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)

    fake_ollama = bin_dir / "ollama"
    fake_ollama.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            host="${OLLAMA_HOST:-}"
            state_dir="${OLLAMA_STATE_DIR:?}/${host//[:\\/]/_}"
            mkdir -p "$state_dir"
            printf 'host=%s gpu=%s | %s\\n' "$host" "${CUDA_VISIBLE_DEVICES:-}" "$*" >> "${OLLAMA_CALLS_FILE:?}"
            if [[ "${1:-}" == "serve" ]]; then
              touch "${state_dir}/server_ready"
              trap 'printf "host=%s gpu=%s | stopped\\n" "$host" "${CUDA_VISIBLE_DEVICES:-}" >> "${OLLAMA_CALLS_FILE:?}"; rm -f "${state_dir}/server_ready"; exit 0' TERM INT
              while true; do
                sleep 1
              done
            fi
            echo "unexpected ollama invocation: $*" >&2
            exit 1
            """
        ),
        encoding="utf-8",
    )
    fake_ollama.chmod(0o755)


def _cfg() -> object:
    return OmegaConf.create(
        {
            "evolver": {
                "enabled": True,
                "llm": {"route_weights": {"mock": 1.0}, "seed": 1},
            },
            "paths": {
                "population_root": "/tmp/pop",
                "api_platform_runtime_root": "/tmp/api_platform_runtime",
            },
            "api_platforms": {
                "mock": {
                    "_target_": "api_platforms.mock.platform.build_platform",
                }
            },
            "resources": {"evaluation": {"gpu_ranks": [0], "cpu_parallel_jobs": 1}},
            "seed": 1,
            "precision": "fp32",
            "experiments": {},
        }
    )


def test_run_seed_population_always_uses_evolution_loop(monkeypatch) -> None:
    cfg = _cfg()
    called = {"seed": False}

    class DummyLoop:
        def __init__(self, received_cfg, llm_registry=None):
            assert received_cfg is cfg
            assert llm_registry is not None

        async def seed_population(self):
            called["seed"] = True
            return {"mode": "seed"}

    monkeypatch.setattr("src.evolve.evolution_loop.EvolutionLoop", DummyLoop)

    result = run_seed_population(cfg)

    assert called["seed"] is True
    assert result == {"mode": "seed"}


def test_run_seed_population_skips_when_disabled() -> None:
    cfg = _cfg()
    cfg.evolver.enabled = False

    assert run_seed_population(cfg) == {}


def test_seed_population_shell_wrapper_exists_and_is_executable() -> None:
    script = ROOT / "scripts" / "seed_population.sh"

    assert script.exists()
    assert script.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
    assert os.access(script, os.X_OK)


def test_kill_ollama_shell_wrapper_exists_and_is_executable() -> None:
    script = ROOT / "scripts" / "kill_ollama.sh"

    assert script.exists()
    assert script.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
    assert os.access(script, os.X_OK)


def test_seed_population_shell_wrapper_prints_help() -> None:
    script = ROOT / "scripts" / "seed_population.sh"

    completed = subprocess.run(
        [str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert completed.returncode == 0
    assert "Seed island populations" in completed.stdout
    assert "HYDRA_OVERRIDES" in completed.stdout
    assert completed.stderr == ""


def test_kill_ollama_shell_wrapper_prints_help() -> None:
    script = ROOT / "scripts" / "kill_ollama.sh"

    completed = subprocess.run(
        [str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert completed.returncode == 0
    assert "Stop local Ollama servers" in completed.stdout
    assert "HYDRA_OVERRIDES" in completed.stdout
    assert completed.stderr == ""


def test_seed_population_shell_wrapper_requires_config_name() -> None:
    script = ROOT / "scripts" / "seed_population.sh"

    completed = subprocess.run(
        [str(script)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert completed.returncode == 2
    assert "requires an explicit --config-name" in completed.stderr


def test_ollama_device_summary_treats_uppercase_cuda_discrete_device_as_gpu(tmp_path: Path) -> None:
    stderr_log = tmp_path / "serve.stderr.log"
    stderr_log.write_text(
        textwrap.dedent(
            """\
            time=2026-04-11T21:10:01.858+03:00 level=INFO source=types.go:42 msg="inference compute" id=GPU-ad350ace-2f24-5f11-bca1-0683d636b105 filter_id="" library=CUDA compute=9.0 name=CUDA0 description="NVIDIA H100 80GB HBM3" libdirs=ollama,cuda_v12 driver=12.6 pci_id=0000:19:00.0 type=discrete total="79.6 GiB" available="62.1 GiB"
            time=2026-04-11T21:10:01.858+03:00 level=INFO source=routes.go:1860 msg="vram-based default context" total_vram="79.6 GiB" default_num_ctx=262144
            """
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            f"source scripts/lib_runtime.sh && _summarize_ollama_compute_devices '{stderr_log}'",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert completed.returncode == 0
    assert "device summary: 1 GPU, 0 CPU" in completed.stdout
    assert "gpu[0]:" in completed.stdout
    assert "library=CUDA" in completed.stdout
    assert "type=discrete" in completed.stdout
    assert "VERDICT=gpu gpus=1" in completed.stdout


def test_kill_ollama_shell_wrapper_requires_config_name() -> None:
    script = ROOT / "scripts" / "kill_ollama.sh"

    completed = subprocess.run(
        [str(script)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert completed.returncode == 2
    assert "requires an explicit --config-name" in completed.stderr


def test_seed_population_shell_wrapper_auto_starts_local_ollama(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "seed_population.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    calls_path = tmp_path / "python_calls.log"
    curl_calls_path = tmp_path / "curl_calls.log"
    ollama_calls_path = tmp_path / "ollama_calls.log"
    runtime_root = tmp_path / "runtime"

    fake_python = fake_bin / "python"
    _write_fake_python(
        fake_python,
        ollama_routes=[
            ("ollama_qwen35_27b", "http://127.0.0.1:11434/api", "qwen3.5:27b", "0"),
        ],
    )
    _write_fake_local_ollama_commands(fake_bin)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["PYTHON_CALLS_FILE"] = str(calls_path)
    env["CURL_CALLS_FILE"] = str(curl_calls_path)
    env["OLLAMA_CALLS_FILE"] = str(ollama_calls_path)
    env["OLLAMA_STATE_DIR"] = str(tmp_path / "ollama_state")
    env["OLLAMA_MODELS_DIR"] = str(tmp_path / "ollama_models")
    env["API_PLATFORM_RUNTIME_ROOT"] = str(runtime_root)

    completed = subprocess.run(
        [str(script), "--config-name", "config_circle_packing_shinka"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )

    assert completed.returncode == 0
    assert "Starting local Ollama server" in completed.stdout
    assert "Pulling Ollama model qwen3.5:27b" in completed.stdout
    assert "local model store:" in completed.stdout
    assert "pulling manifest" in completed.stdout
    assert "downloading:" in completed.stdout
    module_calls = [
        line
        for line in calls_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("-m ")
    ]
    assert module_calls == [
        "-m src.evolve.seed_run --config-name config_circle_packing_shinka",
    ]
    ollama_calls = ollama_calls_path.read_text(encoding="utf-8")
    assert "host=127.0.0.1:11434 gpu=0 | serve" in ollama_calls
    assert "host=127.0.0.1:11434 gpu=0 | stopped" in ollama_calls
    curl_calls = curl_calls_path.read_text(encoding="utf-8").splitlines()
    assert any("/api/tags" in line for line in curl_calls)
    assert any("/api/pull" in line for line in curl_calls)
    assert (runtime_root / "ollama").is_dir()


def test_seed_population_shell_wrapper_auto_starts_multiple_local_ollama_servers(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "seed_population.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    calls_path = tmp_path / "python_calls.log"
    curl_calls_path = tmp_path / "curl_calls.log"
    ollama_calls_path = tmp_path / "ollama_calls.log"
    runtime_root = tmp_path / "runtime"

    fake_python = fake_bin / "python"
    _write_fake_python(
        fake_python,
        ollama_routes=[
            ("ollama_gemma4_26b", "http://127.0.0.1:11434/api", "gemma4:26b", "0"),
            ("ollama_qwen35_27b", "http://127.0.0.1:11435/api", "qwen3.5:27b", "1"),
        ],
    )
    _write_fake_local_ollama_commands(fake_bin)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["PYTHON_CALLS_FILE"] = str(calls_path)
    env["CURL_CALLS_FILE"] = str(curl_calls_path)
    env["OLLAMA_CALLS_FILE"] = str(ollama_calls_path)
    env["OLLAMA_STATE_DIR"] = str(tmp_path / "ollama_state")
    env["OLLAMA_MODELS_DIR"] = str(tmp_path / "ollama_models")
    env["API_PLATFORM_RUNTIME_ROOT"] = str(runtime_root)

    completed = subprocess.run(
        [str(script), "--config-name", "config_circle_packing_shinka"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )

    assert completed.returncode == 0
    assert "Starting local Ollama server for http://127.0.0.1:11434/api on gpu:0" in completed.stdout
    assert "Starting local Ollama server for http://127.0.0.1:11435/api on gpu:1" in completed.stdout
    assert "Pulling Ollama model gemma4:26b" in completed.stdout
    assert "Pulling Ollama model qwen3.5:27b" in completed.stdout
    assert "local model store:" in completed.stdout
    assert "verifying sha256 digest" in completed.stdout
    module_calls = [
        line
        for line in calls_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("-m ")
    ]
    assert module_calls == [
        "-m src.evolve.seed_run --config-name config_circle_packing_shinka",
    ]
    ollama_calls = ollama_calls_path.read_text(encoding="utf-8").splitlines()
    assert any("host=127.0.0.1:11434 gpu=0 | serve" in line for line in ollama_calls)
    assert any("host=127.0.0.1:11435 gpu=1 | serve" in line for line in ollama_calls)
    assert any("host=127.0.0.1:11434 gpu=0 | stopped" in line for line in ollama_calls)
    assert any("host=127.0.0.1:11435 gpu=1 | stopped" in line for line in ollama_calls)
    curl_calls = curl_calls_path.read_text(encoding="utf-8").splitlines()
    assert any("http://127.0.0.1:11434/api/pull" in line for line in curl_calls)
    assert any("http://127.0.0.1:11435/api/pull" in line for line in curl_calls)


def test_seed_population_shell_wrapper_rejects_conflicting_local_ollama_gpu_assignments(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "seed_population.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    calls_path = tmp_path / "python_calls.log"

    fake_python = fake_bin / "python"
    _write_fake_python(
        fake_python,
        ollama_routes=[
            ("ollama_gemma4_26b", "http://127.0.0.1:11434/api", "gemma4:26b", "0"),
            ("ollama_qwen35_27b", "http://127.0.0.1:11434/api", "qwen3.5:27b", "1"),
        ],
    )

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["PYTHON_CALLS_FILE"] = str(calls_path)
    env["OLLAMA_MODELS_DIR"] = str(tmp_path / "custom_models")
    env["API_PLATFORM_RUNTIME_ROOT"] = str(tmp_path / "runtime")

    completed = subprocess.run(
        [str(script), "--config-name", "config_circle_packing_shinka"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )

    assert completed.returncode == 1
    assert "conflicting gpu_ranks" in completed.stderr
    assert not calls_path.exists()


def test_kill_ollama_shell_wrapper_stops_running_local_server(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "kill_ollama.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    calls_path = tmp_path / "python_calls.log"
    ollama_calls_path = tmp_path / "ollama_calls.log"
    runtime_root = tmp_path / "runtime"

    fake_python = fake_bin / "python"
    _write_fake_python(
        fake_python,
        ollama_routes=[
            ("ollama_gemma4_26b", "http://127.0.0.1:12434/api", "gemma4:26b", "0"),
        ],
    )
    _write_fake_local_ollama_commands(fake_bin)

    state_dir = tmp_path / "ollama_state"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["PYTHON_CALLS_FILE"] = str(calls_path)
    env["OLLAMA_CALLS_FILE"] = str(ollama_calls_path)
    env["OLLAMA_STATE_DIR"] = str(state_dir)
    env["OLLAMA_MODELS_DIR"] = str(tmp_path / "ollama_models")
    env["API_PLATFORM_RUNTIME_ROOT"] = str(runtime_root)

    serve_proc = subprocess.Popen(
        [str(fake_bin / "ollama"), "serve"],
        cwd=str(ROOT),
        env={**env, "OLLAMA_HOST": "127.0.0.1:12434", "CUDA_VISIBLE_DEVICES": "0"},
    )
    pid_dir = runtime_root / "ollama"
    pid_dir.mkdir(parents=True, exist_ok=True)
    pid_file = pid_dir / "serve.127.0.0.1_12434.pid"
    pid_file.write_text(f"{serve_proc.pid}\n", encoding="utf-8")

    completed = subprocess.run(
        [str(script), "--config-name", "config_circle_packing_shinka"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )

    assert completed.returncode == 0
    serve_proc.wait(timeout=5)
    assert not pid_file.exists()
    ollama_calls = ollama_calls_path.read_text(encoding="utf-8")
    assert "host=127.0.0.1:12434 gpu=0 | serve" in ollama_calls
    assert "host=127.0.0.1:12434 gpu=0 | stopped" in ollama_calls


def test_kill_ollama_shell_wrapper_stops_stale_runtime_server_from_old_port(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "kill_ollama.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    calls_path = tmp_path / "python_calls.log"
    ollama_calls_path = tmp_path / "ollama_calls.log"
    runtime_root = tmp_path / "runtime"

    fake_python = fake_bin / "python"
    _write_fake_python(
        fake_python,
        ollama_routes=[
            ("ollama_gemma4_26b", "http://127.0.0.1:12434/api", "gemma4:26b", "0"),
        ],
    )
    _write_fake_local_ollama_commands(fake_bin)

    state_dir = tmp_path / "ollama_state"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["PYTHON_CALLS_FILE"] = str(calls_path)
    env["OLLAMA_CALLS_FILE"] = str(ollama_calls_path)
    env["OLLAMA_STATE_DIR"] = str(state_dir)
    env["OLLAMA_MODELS_DIR"] = str(tmp_path / "ollama_models")
    env["API_PLATFORM_RUNTIME_ROOT"] = str(runtime_root)

    serve_proc = subprocess.Popen(
        [str(fake_bin / "ollama"), "serve"],
        cwd=str(ROOT),
        env={**env, "OLLAMA_HOST": "127.0.0.1:11434", "CUDA_VISIBLE_DEVICES": "0"},
    )
    pid_dir = runtime_root / "ollama"
    pid_dir.mkdir(parents=True, exist_ok=True)
    stale_pid_file = pid_dir / "serve.127.0.0.1_11434.pid"
    stale_pid_file.write_text(f"{serve_proc.pid}\n", encoding="utf-8")

    completed = subprocess.run(
        [str(script), "--config-name", "config_circle_packing_shinka"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )

    assert completed.returncode == 0
    serve_proc.wait(timeout=5)
    assert not stale_pid_file.exists()
    ollama_calls = ollama_calls_path.read_text(encoding="utf-8")
    assert "host=127.0.0.1:11434 gpu=0 | serve" in ollama_calls
    assert "host=127.0.0.1:11434 gpu=0 | stopped" in ollama_calls


def test_seed_run_entrypoint_requires_config_name() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "src.evolve.seed_run"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert completed.returncode != 0
    assert "requires an explicit hydra preset" in completed.stderr.lower()
