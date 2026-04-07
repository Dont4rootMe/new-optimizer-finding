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
