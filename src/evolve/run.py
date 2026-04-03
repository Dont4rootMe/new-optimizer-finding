"""Hydra entrypoints for canonical evolve mode and explicit legacy mode."""

from __future__ import annotations

import asyncio
import logging

import hydra
from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


def run_legacy_single_generation(cfg: DictConfig) -> dict:
    """LEGACY: run one candidate-first generation via the explicit legacy entry."""

    if not bool(cfg.evolver.enabled):
        LOGGER.info("evolver.enabled=false, skipping evolve run")
        return {}

    from src.evolve.legacy_orchestrator import LegacyCandidateOrchestrator

    orchestrator = LegacyCandidateOrchestrator(cfg)
    result = asyncio.run(orchestrator.run())
    LOGGER.info("Legacy single-generation evolution finished: %s", result)
    return result


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
