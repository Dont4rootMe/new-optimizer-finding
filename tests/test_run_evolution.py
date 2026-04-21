"""Runtime entrypoint tests for canonical evolve mode."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from omegaconf import OmegaConf

from src.evolve.run import run_evolution

ROOT = Path(__file__).resolve().parents[1]


def _write_fake_python(
    path: Path,
    *,
    population_logic: str | None = None,
    ollama_routes: list[tuple[str, ...]] | None = None,
) -> None:
    population_logic = textwrap.dedent(
        population_logic
        or """
        printf '%s\\n' "${POP_ROOT:?}"
        printf 'ready\\n'
        printf '\\n'
        """
    ).strip()
    ollama_lines = [
        "printf '%s\\n' \"API_PLATFORM_RUNTIME_ROOT=${API_PLATFORM_RUNTIME_ROOT:-/tmp/api_platform_runtime}\"",
        "printf '%s\\n' \"OLLAMA_MODELS_DIR=${OLLAMA_MODELS_DIR:-/tmp/ollama_cache}\"",
    ]
    for route in ollama_routes or []:
        route_id, base_url, model = route[:3]
        gpu_ranks_csv = route[3] if len(route) > 3 else ""
        fields = [route_id, base_url, model, gpu_ranks_csv]
        if len(route) > 4:
            fields.append(route[4])
        if len(route) > 5:
            fields.append(route[5])
        ollama_lines.append(f"printf '%s\\n' 'OLLAMA_ROUTE={'|'.join(fields)}'")

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
            case "$arg" in
              __codex_inspect_ollama__)
        {textwrap.indent(chr(10).join(ollama_lines), '          ')}
                exit 0
                ;;
              __codex_inspect_population__)
        {textwrap.indent(population_logic, '          ')}
                exit 0
                ;;
            esac
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
              */api/chat)
                printf '%s' "${OLLAMA_CHAT_STATUS:-200}"
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
            printf 'host=%s gpu=%s | %s (parallel=%s ctx=%s)\\n' "$host" "${CUDA_VISIBLE_DEVICES:-}" "$*" "${OLLAMA_NUM_PARALLEL:-}" "${OLLAMA_CONTEXT_LENGTH:-}" >> "${OLLAMA_CALLS_FILE:?}"
            if [[ "${1:-}" == "serve" ]]; then
              touch "${state_dir}/server_ready"
              trap 'printf "host=%s gpu=%s | stopped (parallel=%s ctx=%s)\\n" "$host" "${CUDA_VISIBLE_DEVICES:-}" "${OLLAMA_NUM_PARALLEL:-}" "${OLLAMA_CONTEXT_LENGTH:-}" >> "${OLLAMA_CALLS_FILE:?}"; rm -f "${state_dir}/server_ready"; exit 0' TERM INT
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


def test_run_evolution_always_uses_evolution_loop(monkeypatch) -> None:
    cfg = _cfg()
    called = {"loop": False}

    class DummyLoop:
        def __init__(self, received_cfg, llm_registry=None):
            assert received_cfg is cfg
            assert llm_registry is not None

        async def run(self):
            called["loop"] = True
            return {"mode": "canonical"}

    monkeypatch.setattr("src.evolve.evolution_loop.EvolutionLoop", DummyLoop)

    result = run_evolution(cfg)

    assert called["loop"] is True
    assert result == {"mode": "canonical"}


def test_run_evolution_shell_wrapper_exists_and_is_executable() -> None:
    script = ROOT / "scripts" / "run_evolution.sh"

    assert script.exists()
    assert script.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
    assert os.access(script, os.X_OK)


def test_run_evolution_shell_wrapper_prints_help() -> None:
    script = ROOT / "scripts" / "run_evolution.sh"

    completed = subprocess.run(
        [str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert completed.returncode == 0
    assert "Run the main evolution loop" in completed.stdout
    assert "mode=evolve" in completed.stdout
    assert "--seed" in completed.stdout
    assert completed.stderr == ""


def test_run_evolution_shell_wrapper_requires_config_name() -> None:
    script = ROOT / "scripts" / "run_evolution.sh"

    completed = subprocess.run(
        [str(script)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert completed.returncode == 2
    assert "requires an explicit --config-name" in completed.stderr


def test_run_evolution_shell_wrapper_requires_config_name_even_with_seed_flag() -> None:
    script = ROOT / "scripts" / "run_evolution.sh"

    completed = subprocess.run(
        [str(script), "--seed"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert completed.returncode == 2
    assert "requires an explicit --config-name" in completed.stderr


def test_run_evolution_shell_wrapper_auto_seeds_when_generation_zero_is_missing(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "run_evolution.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    calls_path = tmp_path / "python_calls.log"
    pop_root = tmp_path / "pop_missing"
    fake_python = fake_bin / "python"
    _write_fake_python(
        fake_python,
        population_logic="""
        count_file="${INSPECT_COUNT_FILE:?}"
        count=0
        if [[ -f "$count_file" ]]; then
          count="$(cat "$count_file")"
        fi
        count="$((count + 1))"
        printf '%s' "$count" > "$count_file"
        printf '%s\\n' "${POP_ROOT:?}"
        if [[ "$count" -eq 1 ]]; then
          printf 'missing_state\\n'
          printf 'population_state.json is required for seeded-only evolution. Run scripts/seed_population.sh first.\\n'
        else
          printf 'ready\\n'
          printf '\\n'
        fi
        """,
    )

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["POP_ROOT"] = str(pop_root)
    env["PYTHON_CALLS_FILE"] = str(calls_path)
    env["INSPECT_COUNT_FILE"] = str(tmp_path / "inspect_count.txt")

    completed = subprocess.run(
        [str(script), "--seed", "--config-name", "config_circle_packing_shinka"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )

    assert completed.returncode == 0
    assert "running seed_population.sh first" in completed.stdout
    calls = calls_path.read_text(encoding="utf-8").splitlines()
    assert calls == [
        "-m src.evolve.seed_run --config-name config_circle_packing_shinka",
        "-m src.main --config-name config_circle_packing_shinka mode=evolve",
    ]


def test_run_evolution_shell_wrapper_skips_auto_seed_when_generation_zero_exists(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "run_evolution.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    calls_path = tmp_path / "python_calls.log"
    pop_root = tmp_path / "pop_existing"
    (pop_root / "gen_0000").mkdir(parents=True, exist_ok=True)
    fake_python = fake_bin / "python"
    _write_fake_python(fake_python)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["POP_ROOT"] = str(pop_root)
    env["PYTHON_CALLS_FILE"] = str(calls_path)

    completed = subprocess.run(
        [str(script), "--seed", "--config-name", "config_circle_packing_shinka"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )

    assert completed.returncode == 0
    assert "running seed_population.sh first" not in completed.stdout
    calls = calls_path.read_text(encoding="utf-8").splitlines()
    assert calls == [
        "-m src.main --config-name config_circle_packing_shinka mode=evolve",
    ]


def test_run_evolution_shell_wrapper_auto_starts_local_ollama_before_main(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "run_evolution.sh"
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
            ("ollama_qwen35_122b", "http://127.0.0.1:11434/api", "qwen3.5:122b", "0,1,2"),
            ("ollama_qwen35_122b", "http://127.0.0.1:11435/api", "qwen3.5:122b", "3,4,5"),
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
    assert completed.stdout.count("Pulling Ollama model qwen3.5:122b") == 2
    assert "local model store:" in completed.stdout
    assert "downloading:" in completed.stdout
    module_calls = [
        line
        for line in calls_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("-m ")
    ]
    assert module_calls == [
        "-m src.main --config-name config_circle_packing_shinka mode=evolve",
    ]
    ollama_calls = ollama_calls_path.read_text(encoding="utf-8")
    assert "host=127.0.0.1:11434 gpu=0,1,2 | serve" in ollama_calls
    assert "host=127.0.0.1:11435 gpu=3,4,5 | serve" in ollama_calls
    assert "host=127.0.0.1:11434 gpu=0,1,2 | stopped" in ollama_calls
    assert "host=127.0.0.1:11435 gpu=3,4,5 | stopped" in ollama_calls
    curl_calls = curl_calls_path.read_text(encoding="utf-8").splitlines()
    assert any("/api/tags" in line for line in curl_calls)
    assert any("/api/pull" in line for line in curl_calls)
    assert (runtime_root / "ollama").is_dir()


def test_run_evolution_shell_wrapper_prompts_and_reseeds_empty_population(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "run_evolution.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    calls_path = tmp_path / "python_calls.log"
    pop_root = tmp_path / "pop_empty"
    pop_root.mkdir(parents=True, exist_ok=True)
    (pop_root / "population_state.json").write_text("{}", encoding="utf-8")
    fake_python = fake_bin / "python"
    _write_fake_python(
        fake_python,
        population_logic="""
        count_file="${INSPECT_COUNT_FILE:?}"
        count=0
        if [[ -f "$count_file" ]]; then
          count="$(cat "$count_file")"
        fi
        count="$((count + 1))"
        printf '%s' "$count" > "$count_file"
        printf '%s\\n' "${POP_ROOT:?}"
        if [[ "$count" -eq 1 ]]; then
          printf 'empty_population\\n'
          printf 'population_state.json contains no active organisms. Run scripts/seed_population.sh to initialize the population first.\\n'
        else
          printf 'ready\\n'
          printf '\\n'
        fi
        """,
    )

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["POP_ROOT"] = str(pop_root)
    env["PYTHON_CALLS_FILE"] = str(calls_path)
    env["INSPECT_COUNT_FILE"] = str(tmp_path / "inspect_count.txt")

    completed = subprocess.run(
        [str(script), "--seed", "--config-name", "config_circle_packing_shinka"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        input="Y\n",
    )

    assert completed.returncode == 0
    assert "contains no active organisms" in completed.stdout
    assert "Backing up stale population root" in completed.stdout
    backups = list(tmp_path.glob("pop_empty.stale.*"))
    assert len(backups) == 1
    calls = calls_path.read_text(encoding="utf-8").splitlines()
    assert calls == [
        "-m src.evolve.seed_run --config-name config_circle_packing_shinka",
        "-m src.main --config-name config_circle_packing_shinka mode=evolve",
    ]


def test_run_evolution_shell_wrapper_stops_when_seed_still_produces_empty_population(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "run_evolution.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    calls_path = tmp_path / "python_calls.log"
    pop_root = tmp_path / "pop_seed_fail"
    fake_python = fake_bin / "python"
    _write_fake_python(
        fake_python,
        population_logic="""
        count_file="${INSPECT_COUNT_FILE:?}"
        count=0
        if [[ -f "$count_file" ]]; then
          count="$(cat "$count_file")"
        fi
        count="$((count + 1))"
        printf '%s' "$count" > "$count_file"
        printf '%s\\n' "${POP_ROOT:?}"
        if [[ "$count" -eq 1 ]]; then
          printf 'missing_state\\n'
          printf 'population_state.json is required for seeded-only evolution. Run scripts/seed_population.sh first.\\n'
        else
          printf 'empty_population\\n'
          printf 'population_state.json contains no active organisms. Run scripts/seed_population.sh to initialize the population first.\\n'
        fi
        """,
    )

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["POP_ROOT"] = str(pop_root)
    env["PYTHON_CALLS_FILE"] = str(calls_path)
    env["INSPECT_COUNT_FILE"] = str(tmp_path / "inspect_count.txt")

    completed = subprocess.run(
        [str(script), "--seed", "--config-name", "config_circle_packing_shinka"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )

    assert completed.returncode == 1
    assert "Seed run did not produce an active generation-0 population." in completed.stderr
    assert "Ollama server is reachable" in completed.stderr
    calls = calls_path.read_text(encoding="utf-8").splitlines()
    assert calls == [
        "-m src.evolve.seed_run --config-name config_circle_packing_shinka",
    ]


def test_main_entrypoint_requires_config_name() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "src.main"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert completed.returncode != 0
    assert "requires an explicit hydra preset" in completed.stderr.lower()
