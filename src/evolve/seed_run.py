"""Hydra entrypoints for seed-only canonical population initialization."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from api_platforms import ApiPlatformRegistry
from src.runtime_config import ensure_root_runtime_config

LOGGER = logging.getLogger(__name__)


def run_seed_population(cfg: DictConfig) -> dict:
    """Create and score generation-0 organisms without running later generations."""

    ensure_root_runtime_config(cfg, context="src.evolve.seed_run")

    population_root = Path(str(cfg.paths.population_root)).expanduser().resolve()
    _ensure_file_logging(population_root, "seed_run.log")

    from src.evolve.evolution_loop import EvolutionLoop

    registry = ApiPlatformRegistry(cfg)
    try:
        LOGGER.info("Starting API platform registry for routes: %s", ", ".join(registry.available_route_ids))
        registry.start()
        LOGGER.info("API platform registry ready")
        loop = EvolutionLoop(cfg, llm_registry=registry)
        result = asyncio.run(loop.seed_population())
        LOGGER.info("Seed population complete: %s", result)
        return result
    finally:
        registry.stop()


def _ensure_file_logging(population_root: Path, log_name: str) -> None:
    """Attach a FileHandler writing to `population_root/<log_name>`.

    Idempotent — a second call with the same path is a no-op.
    """

    population_root.mkdir(parents=True, exist_ok=True)
    log_path = population_root / log_name
    marker = f"_evolve_file_handler_{log_path}"
    root = logging.getLogger()
    if any(getattr(h, "_evolve_file_marker", None) == marker for h in root.handlers):
        return
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    handler._evolve_file_marker = marker  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(min(root.level or logging.DEBUG, logging.DEBUG))


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
