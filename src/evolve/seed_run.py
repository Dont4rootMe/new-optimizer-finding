"""Hydra entrypoints for seed-only canonical population initialization."""

from __future__ import annotations

import asyncio
import logging

import hydra
from omegaconf import DictConfig

from api_platforms import ApiPlatformRegistry
from src.runtime_config import ensure_root_runtime_config

LOGGER = logging.getLogger(__name__)


def run_seed_population(cfg: DictConfig) -> dict:
    """Create and score generation-0 organisms without running later generations."""

    ensure_root_runtime_config(cfg, context="src.evolve.seed_run")

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


def _ensure_console_logging() -> None:
    """Force an INFO-level stderr handler onto the root logger.

    Hydra's `@hydra.main` installs its own logging config which may hide the
    evolve-loop logs from stderr (sending them only to the Hydra job log file
    under `outputs/.../*.log`). For long-running LLM creation stages that is
    unhelpful — the user expects to see live progress on the terminal they
    launched the script from. We attach a dedicated stream handler (idempotent
    via a marker attribute) so that `LOGGER.info`/`LOGGER.warning` output from
    the evolution loop, the generator, and the orchestrator always surfaces on
    stderr in addition to any Hydra-managed log files.
    """

    root = logging.getLogger()
    marker = "_evolve_console_handler_attached"
    if getattr(root, marker, False):
        root.setLevel(min(root.level or logging.INFO, logging.INFO))
        return
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(min(root.level or logging.INFO, logging.INFO))
    setattr(root, marker, True)


@hydra.main(config_path="../../conf", config_name=None, version_base=None)
def main(cfg: DictConfig) -> None:
    """Standalone module entrypoint for seed-only population initialization."""

    _ensure_console_logging()
    run_seed_population(cfg)


if __name__ == "__main__":
    main()
