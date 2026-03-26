"""Hydra entrypoint for evolve mode."""

from __future__ import annotations

import asyncio
import logging

import hydra
from omegaconf import DictConfig

from src.evolve.orchestrator import EvolverOrchestrator

LOGGER = logging.getLogger(__name__)


def run_single_generation(cfg: DictConfig) -> dict:
    """Run one evolution generation (legacy single-gen mode)."""

    if not bool(cfg.evolver.enabled):
        LOGGER.info("evolver.enabled=false, skipping evolve run")
        return {}

    orchestrator = EvolverOrchestrator(cfg)
    result = asyncio.run(orchestrator.run())
    LOGGER.info("Evolution generation finished: %s", result)
    return result


def run_evolution(cfg: DictConfig) -> dict:
    """Run multi-generation evolution loop."""

    if not bool(cfg.evolver.enabled):
        LOGGER.info("evolver.enabled=false, skipping evolve run")
        return {}

    evo_cfg = cfg.evolver.get("evolution")
    if evo_cfg is None:
        LOGGER.info("No evolution config found, falling back to single generation")
        return run_single_generation(cfg)

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
