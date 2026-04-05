"""Unified entrypoint for validation and evolution modes."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Dispatch to validate or evolve mode based on config."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    mode = str(cfg.mode)
    if mode == "evolve":
        from src.evolve.run import run_evolution

        run_evolution(cfg)
        return

    from src.validate.runner import ExperimentRunner

    ExperimentRunner(cfg).run()


if __name__ == "__main__":
    main()
