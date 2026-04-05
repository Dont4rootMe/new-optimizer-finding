"""Runtime entrypoint tests for canonical evolve mode."""

from __future__ import annotations

from omegaconf import OmegaConf

from src.evolve.run import run_evolution


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
            "resources": {"num_gpus": 1, "gpu_ids": [0]},
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
