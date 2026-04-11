"""Runtime entrypoint tests for canonical evolve mode."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.evolve.run import run_evolution

ROOT = Path(__file__).resolve().parents[1]


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


def test_run_evolution_skips_when_disabled() -> None:
    cfg = _cfg()
    cfg.evolver.enabled = False

    assert run_evolution(cfg) == {}


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
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  printf '%s\\n' "${POP_ROOT:?}"
  exit 0
fi
printf '%s\\n' "$*" >> "${PYTHON_CALLS_FILE:?}"
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

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
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  printf '%s\\n' "${POP_ROOT:?}"
  exit 0
fi
printf '%s\\n' "$*" >> "${PYTHON_CALLS_FILE:?}"
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

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
