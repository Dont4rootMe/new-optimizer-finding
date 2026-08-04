"""Unified entrypoint for validation and evolution modes."""

from __future__ import annotations

import sys

import hydra
from omegaconf import DictConfig

from src.evolve.seed_run import _ensure_console_logging
from src.runtime_config import ensure_root_runtime_config


@hydra.main(config_path="../conf", config_name=None, version_base=None)
def main(cfg: DictConfig) -> None:
    """Dispatch to validate or evolve mode based on config."""

    _ensure_console_logging()
    ensure_root_runtime_config(cfg, context="src.main")

    mode = str(cfg.get("mode", "evolve"))
    if mode == "evolve":
        from src.evolve.run import run_evolution

        run_evolution(cfg)
        return

    from src.validate.runner import ExperimentRunner

    ExperimentRunner(cfg).run()


if __name__ == "__main__":
    if not any(
        arg == "--config-name" or arg.startswith("--config-name=")
        for arg in sys.argv[1:]
    ):
        raise SystemExit(
            "Error: src.main requires an explicit Hydra preset via "
            "--config-name <preset>."
        )
    main()
