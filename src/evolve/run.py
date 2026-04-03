"""Hydra entrypoints for canonical organism-first evolution."""

from __future__ import annotations

import asyncio
import logging

import hydra
from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)

def run_evolution(cfg: DictConfig) -> dict:
    """Run canonical organism-first multi-generation evolution loop."""

    if not bool(cfg.evolver.enabled):
        LOGGER.info("evolver.enabled=false, skipping evolve run")
        return {}

    from src.evolve.evolution_loop import EvolutionLoop

    loop = EvolutionLoop(cfg)
    result = asyncio.run(loop.run())
    LOGGER.info("Evolution complete: %s", result)
    return result


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Standalone module entrypoint for evolve mode."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_evolution(cfg)


if __name__ == "__main__":
    main()
