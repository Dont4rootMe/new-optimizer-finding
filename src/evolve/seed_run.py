"""Hydra entrypoints for seed-only canonical population initialization."""

from __future__ import annotations

import asyncio
import logging

import hydra
from omegaconf import DictConfig

from api_platforms import ApiPlatformRegistry

LOGGER = logging.getLogger(__name__)


def run_seed_population(cfg: DictConfig) -> dict:
    """Create and score generation-0 organisms without running later generations."""

    if not bool(cfg.evolver.enabled):
        LOGGER.info("evolver.enabled=false, skipping seed run")
        return {}

    from src.evolve.evolution_loop import EvolutionLoop

    registry = ApiPlatformRegistry(cfg)
    try:
        registry.start()
        loop = EvolutionLoop(cfg, llm_registry=registry)
        result = asyncio.run(loop.seed_population())
        LOGGER.info("Seed population complete: %s", result)
        return result
    finally:
        registry.stop()


@hydra.main(config_path="../../conf", config_name="config_optimization_survey", version_base=None)
def main(cfg: DictConfig) -> None:
    """Standalone module entrypoint for seed-only population initialization."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_seed_population(cfg)


if __name__ == "__main__":
    main()
