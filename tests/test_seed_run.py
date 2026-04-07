"""Runtime entrypoint tests for seed-only population initialization."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.evolve.seed_run import run_seed_population

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
