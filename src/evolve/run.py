"""Hydra entrypoints for canonical organism-first evolution."""

from __future__ import annotations

import asyncio
import logging

import hydra
from omegaconf import DictConfig

from api_platforms import ApiPlatformRegistry
from src.runtime_config import ensure_root_runtime_config

LOGGER = logging.getLogger(__name__)


def run_evolution(cfg: DictConfig) -> dict:
    """Run canonical organism-first evolution from an already seeded population."""

    ensure_root_runtime_config(cfg, context="src.evolve.run")

    from src.evolve.evolution_loop import EvolutionLoop

    registry = ApiPlatformRegistry(cfg)
    try:
        registry.start()
        loop = EvolutionLoop(cfg, llm_registry=registry)
        result = asyncio.run(loop.run())
        LOGGER.info("Evolution complete: %s", result)
        return result
    finally:
        registry.stop()


@hydra.main(config_path="../../conf", config_name=None, version_base=None)
def main(cfg: DictConfig) -> None:
    """Standalone module entrypoint for evolve mode."""

    from src.evolve.seed_run import _ensure_console_logging

    _ensure_console_logging()
    run_evolution(cfg)


if __name__ == "__main__":
    main()
